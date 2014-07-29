#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "db_loader.h"

/* NOTE: single-reader, single-writer model */

#define ROUND ((size_t)(-1))

#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* bug: 0xFFFFFFFF == 0x00000000 when mod == 0xFFFFFFFF */
#define ADD(a, b, mod) \
	(((mod) - (a)) >= (b) ? (a) + (b) : (b) - ((mod) - (a) + 1))
#define SUB(large, small, mod) \
	((large) >= (small) ? (large) - (small) : (mod) - ((small) - (large) - 1))

#if 0
#define pthread_cond_wait(args...) \
	printf("%lu:WAIT(%s): %s %d\n", pthread_self(), #args, __func__, __LINE__); \
pthread_cond_wait(args)

#define pthread_cond_signal(args...) \
	printf("%lu:SIGNAL(%s): %s %d\n", pthread_self(), #args, __func__, __LINE__); \
pthread_cond_signal(args)

#define pthread_mutex_unlock(args...) \
	printf("%lu:UNLOCK(%s): %s %d\n", pthread_self(), #args, __func__, __LINE__); \
pthread_mutex_unlock(args)

#define pthread_mutex_lock(args...) \
	printf("%lu:LOCK(%s): %s %d\n", pthread_self(), #args, __func__, __LINE__); \
pthread_mutex_lock(args)
#endif

/* Assume big endian for convenience
 *      [0]          [1]      ...
 * [LSB ... MSB][LSB ... MSB] ... */
#define BITMAP_SET(_bitmap, from, to) \
	do { int i; uint8_t *bitmap = (uint8_t *)(_bitmap); \
		assert((from) < (to)); \
		if (((from) / 8) == ((to) / 8)) { \
			for (i = (from); i < (to); i++) \
			bitmap[i / 8] \
			|= i % 8 == 0 ? 0x01 \
			: i % 8 == 1 ? 0x02 \
			: i % 8 == 2 ? 0x04 \
			: i % 8 == 3 ? 0x08 \
			: i % 8 == 4 ? 0x10 \
			: i % 8 == 5 ? 0x20 \
			: i % 8 == 6 ? 0x40 : 0x80; \
		} \
		else { \
			bitmap[(from) / 8] \
			|= (from) % 8 == 0 ? 0xFF \
			: (from) % 8 == 1 ? 0xFE \
			: (from) % 8 == 2 ? 0xFC \
			: (from) % 8 == 3 ? 0xF8 \
			: (from) % 8 == 4 ? 0xF0 \
			: (from) % 8 == 5 ? 0xE0 \
			: (from) % 8 == 6 ? 0xC0 : 0x80; \
			for (i = ((from) / 8) + 1; i < ((to) / 8); i++) \
			bitmap[i] |= 0xFF; \
			bitmap[(to) / 8] \
			|= (to) % 8 == 0 ? 0x00 \
			: (to) % 8 == 1 ? 0x01 \
			: (to) % 8 == 2 ? 0x03 \
			: (to) % 8 == 3 ? 0x07 \
			: (to) % 8 == 4 ? 0x0F \
			: (to) % 8 == 5 ? 0x1F \
			: (to) % 8 == 6 ? 0x3F : 0x7F; \
		} \
	} while (0);
#define BITMAP_CLR(_bitmap, from, to) \
	do { int i; uint8_t *bitmap = (uint8_t *)(_bitmap); \
		assert((from) < (to)); \
		if (((from) / 8) == ((to) / 8)) { \
			for (i = (from); i < (to); i++) \
			bitmap[i / 8] \
			&= i % 8 == 0 ? 0xFE \
			: i % 8 == 1 ? 0xFD \
			: i % 8 == 2 ? 0xFB \
			: i % 8 == 3 ? 0xF7 \
			: i % 8 == 4 ? 0xEF \
			: i % 8 == 5 ? 0xDF \
			: i % 8 == 6 ? 0xBF : 0x7F; \
		} \
		else { \
			bitmap[from / 8] \
			&= (from) % 8 == 0 ? 0x00 \
			: (from) % 8 == 1 ? 0x01 \
			: (from) % 8 == 2 ? 0x03 \
			: (from) % 8 == 3 ? 0x07 \
			: (from) % 8 == 4 ? 0x0F \
			: (from) % 8 == 5 ? 0x1F \
			: (from) % 8 == 6 ? 0x3F : 0x7F; \
			for (i = ((from) / 8) + 1; i < ((to) / 8); i++) \
			bitmap[i] &= 0x00; \
			bitmap[(to) / 8] \
			&= (to) % 8 == 0 ? 0xFF \
			: (to) % 8 == 1 ? 0xFE \
			: (to) % 8 == 2 ? 0xFC \
			: (to) % 8 == 3 ? 0xF8 \
			: (to) % 8 == 4 ? 0xF0 \
			: (to) % 8 == 5 ? 0xE0 \
			: (to) % 8 == 6 ? 0xC0 : 0x80; \
		} \
	} while (0);
#define BITMAP_LEN_ZEROS(_bitmap, from, until, len) \
	do { int i; uint8_t temp; uint8_t *bitmap = (uint8_t *)(_bitmap); \
		assert((until) >= (from)); \
		if ((temp = (0xFF << ((from) % 8)) & bitmap[(from) / 8])) { \
			len = (temp & 0x01) ? 0 \
			: (temp & 0x03) ? 1 \
			: (temp & 0x07) ? 2 \
			: (temp & 0x0F) ? 3 \
			: (temp & 0x1F) ? 4 \
			: (temp & 0x3F) ? 5 \
			: (temp & 0x7F) ? 6 : 7; \
			len -= (from) % 8; \
		} \
		else { \
			for (i = ((from) / 8) + 1; i < ((until) / 8); i++) \
			if (bitmap[i] != 0) break; \
			if (i == ((until) / 8)) \
				(len) = (until) - (from); \
			else { \
				(len) = (i * 8) - (from) \
				+ ((bitmap[i] & 0x01) ? 0 \
						: (bitmap[i] & 0x03) ? 1 \
						: (bitmap[i] & 0x07) ? 2 \
						: (bitmap[i] & 0x0F) ? 3 \
						: (bitmap[i] & 0x1F) ? 4 \
						: (bitmap[i] & 0x3F) ? 5 \
						: (bitmap[i] & 0x7F) ? 6 : 7); \
			} \
		} \
	} while (0)
#define BITMAP_LEN_ONES(_bitmap, from, until, len) \
	do { int i; uint8_t temp; uint8_t *bitmap = (uint8_t *)(_bitmap); \
		assert((until) >= (from)); \
		if ((temp = (0xFF << ((from) % 8)) & ~bitmap[(from) / 8])) { \
			len = (temp & 0x01) ? 0 \
			: (temp & 0x03) ? 1 \
			: (temp & 0x07) ? 2 \
			: (temp & 0x0F) ? 3 \
			: (temp & 0x1F) ? 4 \
			: (temp & 0x3F) ? 5 \
			: (temp & 0x7F) ? 6 : 7; \
			len -= (from) % 8; \
		} \
		else { \
			for (i = ((from) / 8) + 1; i < ((until) / 8); i++) \
			if (bitmap[i] != 0xFF) {break;} \
			if (i == ((until) / 8)) \
				(len) = (until) - (from); \
			else { \
				(len) = (i * 8) - (from) \
				+ ((~bitmap[i] & 0x01) ? 0 \
						: (~bitmap[i] & 0x03) ? 1 \
						: (~bitmap[i] & 0x07) ? 2 \
						: (~bitmap[i] & 0x0F) ? 3 \
						: (~bitmap[i] & 0x1F) ? 4 \
						: (~bitmap[i] & 0x3F) ? 5 \
						: (~bitmap[i] & 0x7F) ? 6 : 7); \
			} \
		} \
	} while (0)

int db_init(db_t *db, int fd, size_t db_len, int align)
{
	db->fd = fd;
	db->db_len = db_len;
	db->align = align;
	
	/* head is always smaller than tail */
	db->head = 0;
	db->middle = 0;
	db->tail = 0;
	db->buffer_len = (MEM_LIMIT / align) * align;
#if 0
	cuInit(0);
	if (!(db->buffer = (uint8_t *)malloc(db->buffer_len))) {
		fprintf(stderr, "%s:%d: Failed to allocate buffer\n",
				__func__, __LINE__);
		exit(0);
	}
#if 0
	if (cudaHostRegister(db->buffer, db->buffer_len, cudaHostRegisterPortable)
			!= cudaSuccess) {
		printf("WRONG!\n");
		fflush(stdout);
		exit(0);
	}
	int *a;
	cudaMallocHost(&a, sizeof(int));
	*a = 25;
	printf("%d@%p@%p\n", *a, a, &a);
#endif
#else
	cudaDeviceSynchronize();
	if (cudaMallocHost(&(db->buffer), db->buffer_len, cudaHostAllocDefault)
			!= cudaSuccess) {
		printf("WRONG!\n");
		fflush(stdout);
		exit(0);
	}
	else
		printf("%p\n", db->buffer);

#endif
	if (!(db->bitmap = (uint8_t *)malloc((db->buffer_len / align / 8) + 1))) {
		fprintf(stderr, "%s:%d: Failed to allocate bitmap\n",
				__func__, __LINE__);
		exit(0);
	}
	memset(db->bitmap, 0, (db->buffer_len / align / 8) + 2);

	db->expired = 0;

	pthread_mutex_init(&db->mx_db, NULL);
	pthread_cond_init(&db->cd_reader, NULL);
	pthread_cond_init(&db->cd_writer, NULL);

	return 0;
}

size_t db_readable(db_t *db)
{
	size_t len;
	size_t from = db->middle % db->buffer_len;
	size_t to = db->tail % db->buffer_len;
	to = ((from < to) ? to : db->buffer_len);
	BITMAP_LEN_ONES(db->bitmap, from / db->align, to / db->align, len);
	return len * db->align;
}

size_t db_writable(db_t *db)
{
	size_t len;
	size_t from = db->tail % db->buffer_len;
	size_t to = db->head % db->buffer_len;
	to = ((from < to) ? to : db->buffer_len);
	BITMAP_LEN_ZEROS(db->bitmap, from / db->align, to / db->align, len);
	return len * db->align;
}

size_t db_acquire(db_t *db, void **ptr, size_t len)
{
	size_t readable = MIN(db_readable(db), len);
	readable = (readable / db->align) * db->align;

	if (readable == 0)
		return 0;

	size_t from, to;
	from = db->middle % db->buffer_len;
	db->middle = ADD(db->middle, readable, ROUND);
	to = db->middle % db->buffer_len;

	*ptr = db->buffer + from;
	if (to < from)
		readable = db->buffer_len - from;
#if 0
	printf("head: %lu, middle: %lu, tail: %lu, \n"
			"acquired %lu - %lu, buffer_len: %lu\n"
			"-----------------------------------------------------\n",
			db->head, db->middle, db->tail,
			(size_t)*ptr - (size_t)db->buffer,
			readable + (size_t)*ptr - (size_t)db->buffer,
			db->buffer_len);
	fflush(stdout);
#endif

	return readable;
}

void db_release(db_t *db, void *ptr, size_t len)
{
#if 0
	printf("head: %lu, middle: %lu, tail: %lu, \n"
			"released %lu - %lu, buffer_len: %lu\n"
			"-----------------------------------------------------\n",
			db->head, db->middle, db->tail,
			(size_t)ptr - (size_t)db->buffer,
			len + (size_t)ptr - (size_t)db->buffer,
			db->buffer_len);
	fflush(stdout);
#endif
	size_t offset = (size_t)ptr - (size_t)db->buffer;

	BITMAP_CLR(db->bitmap, offset / db->align, (offset + len) / db->align);

	size_t from, temp = 0;
	from = db->head % db->buffer_len;

	BITMAP_LEN_ZEROS(db->bitmap, from / db->align, db->buffer_len / db->align,
			len);
	if (len * db->align + from == db->buffer_len) {
		BITMAP_LEN_ZEROS(db->bitmap,
				0, db->buffer_len / db->align, temp);
		len += temp;
	}
	db->head = ADD(db->head, len * db->align, ROUND);

	pthread_mutex_lock(&db->mx_db);
	pthread_cond_signal(&db->cd_writer);
	pthread_mutex_unlock(&db->mx_db);
}

size_t db_write(db_t *db, void *buffer, size_t len)
{
	size_t writable = MIN(db_writable(db), len);
	writable = (writable / db->align) * db->align;
	if (writable == 0)
		return 0;

	size_t from, to;
	from = db->tail % db->buffer_len;
	to = ADD(db->tail, writable, ROUND) % db->buffer_len;

	if (from < to) {
		memcpy((uint8_t *)db->buffer + from, buffer, to - from);
		BITMAP_SET(db->bitmap, from / db->align, to / db->align);
	}
	else {
		memcpy((uint8_t *)db->buffer + from, buffer, db->buffer_len - from);
		BITMAP_SET(db->bitmap, from / db->align, db->buffer_len / db->align);
		if (to != 0) {
			memcpy(db->buffer, (uint8_t *)buffer + db->buffer_len - from, to);
			BITMAP_SET(db->bitmap, 0, to / db->align);
		}
	}

	db->tail = ADD(db->tail, writable, ROUND);

	return writable;
}

void db_kill(db_t *db)
{
	pthread_mutex_lock(&db->mx_db);
	db->expired = 1;
	pthread_cond_signal(&db->cd_writer);
	pthread_mutex_unlock(&db->mx_db);
}

void db_destroy(db_t *db)
{
	if (db->expired) {
		pthread_mutex_destroy(&db->mx_db);
		pthread_cond_destroy(&db->cd_reader);
		pthread_cond_destroy(&db->cd_writer);
		free(db->buffer);
	}

	return;
}

void *db_loader_main(void *arg_void)
{
	db_t *db = (db_t *)arg_void;

	uint8_t *buffer;

	size_t db_left = db->db_len, db_try_read;
	size_t buffer_left = 0;
	int buffer_offset = -1, written;
	size_t disk_read_at_once = (DISK_READ_AT_ONCE / db->align) * db->align;

	if (!(buffer = (uint8_t *)malloc(disk_read_at_once))) {
		fprintf(stderr, "%s:%d: Failed to allocate buffer\n",
				__func__, __LINE__);
		exit(0);
	}

	while (1) {
		if (db->expired)
			break;

		pthread_mutex_lock(&db->mx_db);
		if (MIN(buffer_offset == -1 ? disk_read_at_once : buffer_left,
						db_writable(db)) <= 0) {
			pthread_cond_wait(&db->cd_writer, &db->mx_db);
			pthread_mutex_unlock(&db->mx_db);
			continue;
		}
		pthread_mutex_unlock(&db->mx_db);

		/* read db from disk */
		if (buffer_offset == -1) {
			db_try_read = MIN(db_left, disk_read_at_once);
			if (read(db->fd, buffer, db_try_read)
					!= (int)db_try_read) {
				fprintf(stderr, "Failed to read database file\n");
				close(db->fd);
				exit(0);
			}
			else {
				db_left -= db_try_read;
			}

			if (db_left <= 0) {
				db_left = db->db_len;
				lseek(db->fd, 0, SEEK_SET);
			}

			buffer_offset = 0;
			buffer_left = db_try_read;
		}

		pthread_mutex_lock(&db->mx_db);
		written = db_write(db, (void *)(buffer + (size_t)buffer_offset),
				buffer_left);
		if (written > 0) {
			buffer_left -= written;
			if (buffer_left == 0)
				buffer_offset = -1;
			else
				buffer_offset += written;
			pthread_cond_signal(&db->cd_reader);
		}
		pthread_mutex_unlock(&db->mx_db);
	}

//	db_destroy(db);

	return NULL;
}
