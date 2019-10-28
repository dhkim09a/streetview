#ifndef __SEARCH_H_
#define __SEARCH_H_

#include <pthread.h>

#include "surflib.h"
#include "db.h"
#include "db_loader.h"
#include "message.h"

#ifndef MATCH_THRESH
#error Define MATCH_THRESH!
#endif

#define MATCH_THRESH_SQUARE (MATCH_THRESH * MATCH_THRESH)

typedef struct _result_t {
	FPF latitude;
	FPF longitude;
	float score;
} result_t;

typedef std::vector<result_t> ResVec;

static bool comp_result (result_t i, result_t j)
{
	/* > for descending order, < for ascending order */
	return i.score > j.score;
}

struct _interim {
	FPF lat_first __attribute__((aligned (4)));
	FPF lng_first __attribute__((aligned (4)));
	float dist_first __attribute__((aligned (4)));
	float dist_second __attribute__((aligned (4)));
} __attribute__((packed));

typedef struct _req_t {
	IplImage *img;
	FPF latitude;
	FPF longitude;
	FPF score;
} req_t;

typedef struct _search_t {
	db_t *db;
	msgbox_t msgbox;
} search_t;

typedef struct _task_t {
	ipoint_t *haystack;
	int haystack_size;
	IpVec needle;
	struct _interim *result;
} task_t;

typedef struct _worker {
	bool dead;
	bool ready;

	pthread_t tid;
	msgbox_t msgbox;
	bool isbusy;
	size_t chunk_size;
	int did; // GPU device ID for GPU workers

	int stat_called;
	size_t stat_bytes;

	db_t *db;
	pthread_cond_t *cd_wait_worker;
} worker_t;

int sc_init(search_t *sc, db_t *db);
void *sc_main(void *arg);
#endif /* __SEARCH_H_ */
