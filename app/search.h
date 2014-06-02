#ifndef __SEARCH_H_
#define __SEARCH_H_

#include "surflib.h"
#include "db.h"
#include "db_loader.h"

#ifndef MATCH_THRESH
#error Define MATCH_THRESH!
#endif

#define MATCH_THRESH_SQUARE (MATCH_THRESH * MATCH_THRESH)

#define BACKLOG 20

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

typedef struct _req_msg_t req_msg_t;
typedef struct _req_msg_t {
	IplImage *img;
	void (*cb)(req_msg_t *msg, FPF latitude, FPF longitude, float score,
			void *arg);
	void *cb_arg;
} req_msg_t;

typedef struct _search_t {
	int expired;
	db_t *db;

	/* circular queue */
	req_msg_t req_queue[BACKLOG];
	int head;
	int tail;
	int empty;

	pthread_mutex_t mx_queue;
	pthread_cond_t cd_worker;
} search_t;

int sc_init(search_t *sc, db_t *db);
int sc_request(search_t *sc, IplImage *img,
		void (*cb)(req_msg_t *msg, FPF latitude, FPF longitude, float score,
			void *arg),
		void *cb_arg);
void *sc_main(void *arg);
#endif /* __SEARCH_H_ */
