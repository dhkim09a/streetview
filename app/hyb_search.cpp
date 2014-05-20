#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>
#include <cv.h>
#include "surflib.h"

#include "db.h"
#include "search.h"

#ifndef CHUNK_SIZE
#error Define CHUNK_SIZE
#endif

#define GPU 0
#define CPU 1

extern int searchCPU (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int numcpu);

extern int searchGPU (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int numcpu);

typedef struct _arg_t {
	IpVec needle;
	ipoint_t *haystack;
	int haystack_size;
	int chunk_size;
	int *chunk_id;
	pthread_mutex_t *mx_chunk_id;

	struct _interim *result;
	int result_size;
	int numcpu;

	int *running;
	pthread_mutex_t *mx_running;
	pthread_cond_t *wakeup_master;
} arg_t;

static void *GPUmain(void *arg)
{
	arg_t *arg1 = (arg_t *)arg;
	int chunk_id = 0;
	int chunk_size = arg1->chunk_size;

	while(1) {
		pthread_mutex_lock(arg1->mx_chunk_id);
		chunk_id = (*arg1->chunk_id)++;
		pthread_mutex_unlock(arg1->mx_chunk_id);

		if (chunk_id * arg1->chunk_size > arg1->haystack_size)
			break;
		else if ((chunk_id + 1) * arg1->chunk_size > arg1->haystack_size)
			chunk_size = arg1->haystack_size % arg1->chunk_size;

		searchGPU(arg1->needle,
				&(arg1->haystack[chunk_id * arg1->chunk_size]),
				chunk_size,
				arg1->result, arg1->result_size, 0);
	}

	pthread_mutex_lock(arg1->mx_running);
	if (--(*arg1->running) == 0)
		pthread_cond_signal(arg1->wakeup_master);
	pthread_mutex_unlock(arg1->mx_running);

	return NULL;
}

static void *CPUmain(void *arg)
{
	arg_t *arg1 = (arg_t *)arg;
	int chunk_id = 0;
	int chunk_size = arg1->chunk_size;

	while(1) {
		pthread_mutex_lock(arg1->mx_chunk_id);
		chunk_id = (*arg1->chunk_id)++;
		pthread_mutex_unlock(arg1->mx_chunk_id);

		if (chunk_id * arg1->chunk_size > arg1->haystack_size)
			break;
		else if ((chunk_id + 1) * arg1->chunk_size > arg1->haystack_size)
			chunk_size = arg1->haystack_size % arg1->chunk_size;

		searchCPU(arg1->needle,
				&(arg1->haystack[chunk_id * arg1->chunk_size]),
				chunk_size,
				arg1->result, arg1->result_size, arg1->numcpu);
	}

	pthread_mutex_lock(arg1->mx_running);
	if (--(*arg1->running) == 0)
		pthread_cond_signal(arg1->wakeup_master);
	pthread_mutex_unlock(arg1->mx_running);

	return NULL;
}

int searchHyb (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int numcpu)
{
	if (numcpu <= 0)
		return -1;

	pthread_t threads[2];
	arg_t args[2];
	int i, j, err, iter;
	int status;
	float dist;

	int chunk_size = CHUNK_SIZE / sizeof(ipoint_t);
	int chunk_id = 0;
	pthread_mutex_t mx_chunk_id;
	pthread_mutex_init(&mx_chunk_id, NULL);

	int running = 2;
	pthread_mutex_t mx_running;
	pthread_cond_t wakeup_master;

	pthread_mutex_init(&mx_running, NULL);
	pthread_cond_init(&wakeup_master, NULL);

	pthread_mutex_lock(&mx_running);

	args[GPU].needle = args[CPU].needle = needle;
	args[GPU].haystack = args[CPU].haystack = haystack;
	args[GPU].haystack_size = args[CPU].haystack_size = haystack_size;
	args[GPU].chunk_size = args[CPU].chunk_size = chunk_size;
	args[GPU].chunk_id = args[CPU].chunk_id = &chunk_id;
	args[GPU].mx_chunk_id = args[CPU].mx_chunk_id = &mx_chunk_id;
	args[GPU].result = (struct _interim *)malloc(
			needle.size() * sizeof(struct _interim));
	args[CPU].result = (struct _interim *)malloc(
			needle.size() * sizeof(struct _interim));
	args[GPU].result_size = args[CPU].result_size = needle.size();
	args[GPU].running = args[CPU].running = &running;
	args[GPU].mx_running = args[CPU].mx_running = &mx_running;
	args[GPU].wakeup_master = args[CPU].wakeup_master = &wakeup_master;
	args[GPU].numcpu = 0;
	args[CPU].numcpu = numcpu - 1;
	for (i = 0; i < (int)needle.size(); i++) {
		for (j = 0; j < 2; j++) {
		args[j].result[i].dist_first = FLT_MAX;
		args[j].result[i].dist_second = FLT_MAX;
		args[j].result[i].lat_first = 0;
		args[j].result[i].lng_first = 0;
		}
	}

	pthread_create(&(threads[GPU]), NULL, &GPUmain, (void *)(&args[GPU]));
	pthread_create(&(threads[CPU]), NULL, &CPUmain, (void *)(&args[CPU]));

	pthread_cond_wait(&wakeup_master, &mx_running);
	pthread_mutex_unlock(&mx_running);

	for (i = 0; i < 2; i++) {
		if ((err = pthread_join(threads[i], (void **)&status))) {
			fprintf(stderr, "pthread_join(%d) failed (returned %d)\n",
					i, err);
			fflush(stderr);
		}
	}

	iter = MIN((int)needle.size(), result_size);
	for (j = 0; j < 2; j++) {
		for (i = 0; i < iter; i++) {
			dist = args[j].result[i].dist_first;
			if (dist < result[i].dist_first) {
				result[i].lat_first = args[j].result[i].lat_first;
				result[i].lng_first = args[j].result[i].lng_first;
				result[i].dist_second = result[i].dist_first;
				result[i].dist_first = dist;
			}
			else if (dist < result[i].dist_second)
				result[i].dist_second = dist;
		}
	}

	free(args[GPU].result);
	free(args[CPU].result);

	return 0;
}
