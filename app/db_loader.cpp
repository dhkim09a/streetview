#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

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

int db_init(db_t *db, int fd, size_t db_len)
{
	db->fd = fd;
	db->db_len = db_len;
	
	/* head is always smaller than tail */
	db->head = 0;
	db->tail = 0;
	db->buffer_len = MEM_LIMIT;
	if (!(db->buffer = (ipoint_t *)malloc(db->buffer_len))) {
		fprintf(stderr, "%s:%d: Failed to allocate shard pool\n",
				__func__, __LINE__);
		exit(0);
	}

	db->expired = 0;

	pthread_mutex_init(&db->mx_db, NULL);
	pthread_cond_init(&db->cd_reader, NULL);
	pthread_cond_init(&db->cd_writer, NULL);

	return 0;
}

size_t db_readable(db_t *db)
{
	return SUB(db->tail, db->head, ROUND);
}

size_t db_read(db_t *db, void *buffer, size_t len)
{
	size_t readable = MIN(db_readable(db), len);

	if (readable == 0)
		return 0;

	size_t from, to;
	from = db->head % db->buffer_len;
	to = ADD(db->head, readable, ROUND) % db->buffer_len;

	if (from < to) {
		memcpy(buffer, (uint8_t *)db->buffer + from, to - from);
	}
	else {
		memcpy(buffer, (uint8_t *)db->buffer + from, db->buffer_len - from);
		memcpy((uint8_t *)buffer + db->buffer_len - from, db->buffer, to);
	}

	db->head = ADD(db->head, readable, ROUND);

	return readable;
}

size_t db_writable(db_t *db)
{
	size_t readable = db_readable(db);
	if (readable >= db->buffer_len)
		return 0;
	return SUB(db->buffer_len, readable, ROUND);
}

size_t db_write(db_t *db, void *buffer, size_t len)
{
	size_t writable = MIN(db_writable(db), len);

	if (writable == 0)
		return 0;

	size_t from, to;
	from = db->tail % db->buffer_len;
	to = ADD(db->tail, writable, ROUND) % db->buffer_len;

	if (from < to) {
		memcpy((uint8_t *)db->buffer + from, buffer, to - from);
	}
	else {
		memcpy((uint8_t *)db->buffer + from, buffer, db->buffer_len - from);
		memcpy(db->buffer, (uint8_t *)buffer + db->buffer_len - from, to);
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

	if (!(buffer = (uint8_t *)malloc(DISK_READ_AT_ONCE))) {
		fprintf(stderr, "%s:%d: Failed to allocate buffer\n",
				__func__, __LINE__);
		exit(0);
	}

	while (1) {
		if (db->expired)
			break;

		pthread_mutex_lock(&db->mx_db);
		if (MIN(buffer_offset == -1 ? DISK_READ_AT_ONCE : buffer_left,
						db_writable(db)) <= 0) {
			pthread_cond_wait(&db->cd_writer, &db->mx_db);
			pthread_mutex_unlock(&db->mx_db);
			continue;
		}
		pthread_mutex_unlock(&db->mx_db);

		/* read db from disk */
		if (buffer_offset == -1) {
			db_try_read = MIN(db_left, DISK_READ_AT_ONCE);
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

	db_destroy(db);

	return NULL;
}
