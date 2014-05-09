#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <algorithm>
#include <vector>

#include "db.h"
#include "search.h"
/*
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
*/
typedef struct _arg_t {
	IpVec *needle;
	ipoint_t *haystack;
	int haystack_size;
	struct _interim *interim;
	int interim_size;

	int *running;
	pthread_mutex_t *mx_running;
	pthread_cond_t *wakeup_master;
} arg_t;

int doSearch (IpVec *needle, ipoint_t *haystack, int haystack_size,
		struct _interim *interim, int interim_size)
{
	float dist, temp;
	int i, j, k;
	int iter;

	for (i = 0; i < interim_size; i++) {
		interim[i].dist_first = FLT_MAX;
		interim[i].dist_second = FLT_MAX;
	}

	iter = MIN(interim_size, (int)(*needle).size());
	for (i = 0; i < iter; i++) {
		for (j = 0; j < haystack_size; j++) {
			dist = 0;
			for (k = 0; k < VEC_DIM; k++) {
				temp = (*needle)[i].descriptor[k] - haystack[j].vec[k];
				dist += temp * temp;
			}

			if (dist < interim[i].dist_first) {
				interim[i].lat_first = haystack[j].latitude;
				interim[i].lng_first = haystack[j].longitude;
				interim[i].dist_second = interim[i].dist_first;
				interim[i].dist_first = dist;
			}
			else if (dist < interim[i].dist_second)
				interim[i].dist_second = dist;
		}
		printf("\33[2K\r%d/%d", i+1, iter);
		fflush(stdout);
	}

	return 0;
}

void *thread_main(void *arg)
{
	arg_t *arg1 = (arg_t *)arg;

	doSearch(arg1->needle, arg1->haystack, arg1->haystack_size,
			arg1->interim, arg1->interim_size);

	pthread_mutex_lock(arg1->mx_running);
	if (--(*arg1->running) == 0)
		pthread_cond_signal(arg1->wakeup_master);
	pthread_mutex_unlock(arg1->mx_running);

	return NULL;
}

int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec *result_vec, int numcpu)
{
	if (numcpu <= 0)
		return -1;

	pthread_t *threads = (pthread_t *)malloc(numcpu * sizeof(pthread_t));
	arg_t *args = (arg_t *)malloc(numcpu * sizeof(arg_t));
	int i, j, found, err;
	int status;
	struct _interim interim;
	result_t result;
	float dist;

	int running = numcpu;
	pthread_mutex_t mx_running;
	pthread_cond_t wakeup_master;

	pthread_mutex_init(&mx_running, NULL);
	pthread_cond_init(&wakeup_master, NULL);

	/* FIXME: Threads should divide needle, NOT haystack */
	pthread_mutex_lock(&mx_running);
	for (i = 0; i < numcpu; i++) {
		args[i].needle = &needle;
		args[i].haystack = &haystack[ haystack_size / numcpu * i ];
		args[i].haystack_size = MIN( haystack_size / numcpu,
				haystack_size - (haystack_size / numcpu * i) );
		args[i].interim = (struct _interim *)malloc(
				needle.size() * sizeof(struct _interim));
		args[i].interim_size = needle.size();
		args[i].running = &running;
		args[i].mx_running = &mx_running;
		args[i].wakeup_master = &wakeup_master;

		pthread_create(&(threads[i]), NULL, &thread_main, (void *)(&args[i]));
	}

	pthread_cond_wait(&wakeup_master, &mx_running);
	pthread_mutex_unlock(&mx_running);

	for (i = 0; i < numcpu; i++) {
		if (err = pthread_join(threads[i], (void **)&status)) {
			fprintf(stderr, "pthread_join(%d) failed (returned %d)\n",
					i, err);
			fflush(stderr);
		}
	}

	printf("\n");
	fflush(stdout);

	for (i = 0; i < (int)needle.size(); i++) {
		interim.dist_first = FLT_MAX;
		interim.dist_second = FLT_MAX;

		for (j = 0; j < numcpu; j++) {
			dist = args[j].interim[i].dist_first;
			if (dist < interim.dist_first) {
				interim.lat_first = args[j].interim[i].lat_first;
				interim.lng_first = args[j].interim[i].lng_first;
				interim.dist_second = interim.dist_first;
				interim.dist_first = dist;
			}
			else if (dist < interim.dist_second)
				interim.dist_second = dist;
		}

		if (interim.dist_first / interim.dist_second < MATCH_THRESH_SQUARE) {
			found = -1;
			for (j = 0; j < (int)(*result_vec).size(); j++) {
				if ((*result_vec)[j].latitude == interim.lat_first
						&& (*result_vec)[j].longitude == interim.lng_first) {
					(*result_vec)[j].occurence++;
					found = 1;
					break;
				}
			}

			if (found < 0) {
				result.latitude = interim.lat_first;
				result.longitude = interim.lng_first;
				result.occurence = 1;

				(*result_vec).push_back(result);
			}
		}
	}

	std::sort((*result_vec).begin(), (*result_vec).end(), comp_result);

	for (i = 0; i < numcpu; i++)
		free(args[i].interim);
	free(args);
	free(threads);

	return 0;
}

