#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>
#include <math.h>
#include <cv.h>
#include "surflib.h"

#include "db.h"
#include "search.h"
#include "db_loader.h"

#ifdef PROFILE
#define PROFILE_ON
#endif
#include "profile.h"

#define SURF_THRESHOLD 0.0001f

#ifndef RUN_UNIT
#error Define RUN_UNIT!
#endif

#ifndef MEM_LIMIT
#error Define MEM_LIMIT!
#endif

#ifndef NUMCPU
#error Define NUMCPU as a positive value!
#endif

#ifndef search
#error Define search()!
#endif

#define MAX_IPTS 2000

extern int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int numcpu);

int sc_init(search_t *sc, db_t *db)
{
	int i;
	sc->db = db;
	for (i = 0; i < BACKLOG; i++) {
		sc->req_queue[i].img = NULL;
		sc->req_queue[i].cb = NULL;
		sc->req_queue[i].cb_arg = NULL;
	}
	sc->head = 0; 
	sc->tail = 0;
	sc->empty = 1;

	pthread_mutex_init(&sc->mx_queue, NULL);
	pthread_cond_init(&sc->cd_worker, NULL);

	return 0;
}

int sc_request(search_t *sc, IplImage *img,
		void (*cb)(req_msg_t *msg, FPF latitude, FPF longitude, float score,
			void *arg),
		void *cb_arg)
{
	pthread_mutex_lock(&sc->mx_queue);
	req_msg_t *msg;
	if (!sc->empty && sc->head == sc->tail) {
		pthread_mutex_unlock(&sc->mx_queue);
		return -1;
	}
	else {
		msg = &sc->req_queue[sc->tail];
		sc->tail = (sc->tail + 1) % BACKLOG;
	}
	msg->img = img;
	msg->cb = cb;
	msg->cb_arg = cb_arg;

	sc->empty = 0;
	pthread_cond_signal(&sc->cd_worker);
	pthread_mutex_unlock(&sc->mx_queue);

	return 0;
}

inline void sc_destroy_req_msg(req_msg_t *msg)
{
	msg->img = NULL;
	msg->cb = NULL;
	msg->cb_arg = NULL;
}

void *sc_main(void *arg)
{
	req_msg_t *msg;
	search_t *sc = (search_t *)arg;
	db_t *db = sc->db;

	struct _interim *result;
	result_t answer;
	ResVec answer_vec;
	IpVec ipts_vec;

	ipoint_t *haystack = NULL;
	int haystack_size; /* number of entries in haystack */
	size_t haystack_mem_size;
	size_t db_left;

	int i, j, found;
	float dist_ratio;


	result = (struct _interim *)malloc(
			MAX_IPTS * sizeof(struct _interim));

	while (1) {
		pthread_mutex_lock(&sc->mx_queue);
		if (sc->empty)
			pthread_cond_wait(&sc->cd_worker, &sc->mx_queue);
		msg = &sc->req_queue[sc->head];
		if (!msg->img || !msg->cb_arg || !msg->cb) {
			fprintf(stderr, "%s:%d: Something goes wrong\n",
					__func__, __LINE__);
			exit(0);
		}
		sc->head = (sc->head + 1) % BACKLOG;
		if (sc->head == sc->tail)
			sc->empty = 1;
		pthread_mutex_unlock(&sc->mx_queue);

		PROFILE_START();
		PROFILE_VAR(surf_input_image);
		PROFILE_VAR(load_database);
		PROFILE_VAR(vector_matching);

		PROFILE_FROM(surf_input_image);

		surfDetDes(msg->img, ipts_vec, false, 3, 4, 3, SURF_THRESHOLD);

		printf("Extracted %lu interesting points from input image\n",
				ipts_vec.size());

		/* truncate ipts_vec */
		while (ipts_vec.size() > MAX_IPTS)
			ipts_vec.pop_back();

		PROFILE_TO(surf_input_image);

		/* init reault */
		for (i = 0; i < (int)ipts_vec.size(); i++) {
			result[i].dist_first = FLT_MAX;
			result[i].dist_second = FLT_MAX;
			result[i].lat_first = 0;
			result[i].lng_first = 0;
		}

		/* Read DB and do searching */
		db_left = db->db_len;

		while (1) {
			haystack_size = MIN(RUN_UNIT, db_left) / sizeof(ipoint_t);
			haystack_mem_size = haystack_size * sizeof(ipoint_t);

			if (haystack_size <= 0)
				break;

			if (haystack == NULL)
				haystack = (ipoint_t *)malloc(haystack_mem_size);

			PROFILE_FROM(load_database);
			pthread_mutex_lock(&db->mx_db);
			while (db_readable(db) < haystack_mem_size) {
				pthread_cond_wait(&db->cd_reader, &db->mx_db);
			}

			if (db_read(db, haystack, haystack_mem_size)
					!= (long long)haystack_mem_size) {
				fprintf(stderr, "Failed to read database file\n");
				exit(0);
			}
			else {
				db_left -= haystack_mem_size;
				printf("\rRead %lu / %lu bytes from DB",
						db->db_len - db_left, db->db_len);
				fflush(stdout);
				pthread_cond_signal(&db->cd_writer);
			}
			pthread_mutex_unlock(&db->mx_db);
			PROFILE_TO(load_database);

			PROFILE_FROM(vector_matching);
			search(ipts_vec, haystack, haystack_size, result, ipts_vec.size(),
					NUMCPU);
			PROFILE_TO(vector_matching);
		}
		printf("\n");

		for (i = 0; i < (int)ipts_vec.size(); i++) {
			dist_ratio = result[i].dist_first / result[i].dist_second;
			if (dist_ratio < MATCH_THRESH_SQUARE){
				found = -1;
				for (j = 0; j < (int)answer_vec.size(); j++) {
					if (answer_vec[j].latitude == result[i].lat_first
							&& answer_vec[j].longitude == result[i].lng_first) {
						answer_vec[j].score += 1 - dist_ratio;
						found = 1;
						break;
					}
				}

				if (found < 0) {
					answer.latitude = result[i].lat_first;
					answer.longitude = result[i].lng_first;
					answer.score = 1 - dist_ratio;

					answer_vec.push_back(answer);
				}
			}
		}
		float sum = 0;
		for (i = 0; i < (int)answer_vec.size(); i++) {
			sum += answer_vec[i].score;
		}
		for (i = 0; i < (int)answer_vec.size(); i++)
			answer_vec[i].score *= 100 / sum;
		std::sort(answer_vec.begin(), answer_vec.end(), comp_result);

		PROFILE_END();
		PROFILE_PRINT(stdout);

		msg->cb(msg, answer_vec[0].latitude, answer_vec[0].longitude,
				answer_vec[0].score, msg->cb_arg);

		/* XXX: Is this sufficient to free all the memory? */
		sc_destroy_req_msg(msg);
		ipts_vec.clear();
		answer_vec.clear();
	}

	return NULL;
}
