#ifndef __DB_LOADER_H_
#define __DB_LOADER_H_

#include "db.h"

/* NOTE: single-reader, single-writer model */

#define DISK_READ_AT_ONCE 1000000

typedef struct _db_t {
	int fd;
	size_t db_len;

	size_t head;
	size_t tail;
	ipoint_t *buffer;
	size_t buffer_len;

	int expired;

	pthread_mutex_t mx_db;
	pthread_cond_t cd_reader;
	pthread_cond_t cd_writer;
} db_t;

int db_init(db_t *db, int fd, size_t db_len);
size_t db_readable(db_t *db);
size_t db_read(db_t *db, void *buffer, size_t len);
size_t db_writable(db_t *db);
size_t db_write(db_t *db, void *buffer, size_t len);
void *db_loader_main(void *arg_void);
void db_kill(db_t *db);
#endif /* __DB_LOADER_H_ */
