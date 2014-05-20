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

#ifndef GPU_OFFLOAD
#error Define GPU_OFFLOAD btw 0 and 1!
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

	searchGPU(arg1->needle, arg1->haystack, arg1->haystack_size,
			arg1->result, arg1->result_size, 0);

	pthread_mutex_lock(arg1->mx_running);
	if (--(*arg1->running) == 0)
		pthread_cond_signal(arg1->wakeup_master);
	pthread_mutex_unlock(arg1->mx_running);

	return NULL;
}

static void *CPUmain(void *arg)
{
	arg_t *arg1 = (arg_t *)arg;

	searchCPU(arg1->needle, arg1->haystack, arg1->haystack_size,
			arg1->result, arg1->result_size, arg1->numcpu);

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

	int running = 2;
	pthread_mutex_t mx_running;
	pthread_cond_t wakeup_master;

	pthread_mutex_init(&mx_running, NULL);
	pthread_cond_init(&wakeup_master, NULL);

	pthread_mutex_lock(&mx_running);

	args[GPU].needle = args[CPU].needle = needle;
	args[GPU].haystack = haystack;
	args[GPU].haystack_size = haystack_size * GPU_OFFLOAD;
	args[CPU].haystack = &haystack[args[GPU].haystack_size];
	args[CPU].haystack_size = haystack_size - args[GPU].haystack_size;
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
