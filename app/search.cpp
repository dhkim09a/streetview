#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
#include "message.h"

#ifdef PROFILE
#define PROFILE_ON
#endif
#include "profile.h"

#define SURF_THRESHOLD 0.0001f

#ifndef CPU_CHUNK_SIZE
#error Define CPU_CHUNK_SIZE!
#endif

#ifndef GPU_CHUNK_SIZE
#error Define GPU_CHUNK_SIZE!
#endif

#ifndef MEM_LIMIT
#error Define MEM_LIMIT!
#endif

#ifndef NUMCPU
#error Define NUMCPU as a positive value!
#endif

#ifndef NUMGPU
#error Define NUMGPU as a positive value!
#endif

#define NUM_WORKER (NUMCPU + NUMGPU)
#define MAX_IPTS 20000

extern void *search_cpu_main(void *arg);
extern void *search_gpu_main(void *arg);

int sc_init(search_t *sc, db_t *db)
{
	sc->db = db;
	msg_init_box(&sc->msgbox);

	return 0;
}

static int init_worker_pool(worker_t *workers,
		db_t *db, pthread_cond_t *cd_wait_worker,
		void *(*cpu_thread_main)(void *arg),
		size_t cpu_chunk_size, int num_cpu_threads,
		void *(*gpu_thread_main)(void *arg),
		size_t gpu_chunk_size, int num_gpu_threads)
{
	int i;

	// Worker 0 is the most preferred because the worker pool is always searched from 0.
	for (i = 0; i < num_gpu_threads; i++) {
		msg_init_box(&workers[i].msgbox);
		workers[i].dead = false;
		workers[i].ready = false;
		workers[i].isbusy = false;
		workers[i].chunk_size = gpu_chunk_size;
		workers[i].did = i;
		workers[i].db = db;
		workers[i].cd_wait_worker = cd_wait_worker;
		workers[i].stat_called = 0;
		workers[i].stat_bytes = 0;
		pthread_create(&workers[i].tid, NULL, gpu_thread_main, &workers[i]);
	}
	for (i = num_gpu_threads; i < num_gpu_threads + num_cpu_threads; i++) {
		msg_init_box(&workers[i].msgbox);
		workers[i].dead = false;
		workers[i].ready = false;
		workers[i].isbusy = false;
		workers[i].chunk_size = cpu_chunk_size;
		workers[i].db = db;
		workers[i].cd_wait_worker = cd_wait_worker;
		workers[i].stat_called = 0;
		workers[i].stat_bytes = 0;
		pthread_create(&workers[i].tid, NULL, cpu_thread_main, &workers[i]);
	}

	// Wait for GPU threads to be bound to GPUs. The program
	// works even without this line, but profile result might
	// be wrong.
	for (i = 0; i < num_cpu_threads + num_gpu_threads; i++) {
		while (workers[i].ready == false)
			usleep(1000);
	}

	return 0;
}

void merge_result(struct _interim *resultA, struct _interim *resultB, int size)
{
	float dist;
	int i;

	for (i = 0; i < size; i++) {
		if (resultA[i].dist_first == FLT_MAX) {
			resultA[i].lat_first = resultB[i].lat_first;
			resultA[i].lng_first = resultB[i].lng_first;
			resultA[i].dist_first = resultB[i].dist_first;
			resultA[i].dist_second = resultB[i].dist_second;
			continue;
		}

		dist = resultB[i].dist_first;
		if (dist < resultA[i].dist_first) {
			resultA[i].lat_first = resultB[i].lat_first;
			resultA[i].lng_first = resultB[i].lng_first;
			resultA[i].dist_second = resultA[i].dist_first;
			resultA[i].dist_first = dist;
		}
		else if (dist < resultA[i].dist_second)
			resultA[i].dist_second = dist;

		dist = resultB[i].dist_second;
		if (dist < resultA[i].dist_first) {
			resultA[i].lat_first = resultB[i].lat_first;
			resultA[i].lng_first = resultB[i].lng_first;
			resultA[i].dist_second = resultA[i].dist_first;
			resultA[i].dist_first = dist;
		}
		else if (dist < resultA[i].dist_second)
			resultA[i].dist_second = dist;
	}

	return;
}

void *sc_main(void *arg)
{
	search_t *sc = (search_t *)arg;
	db_t *db = sc->db;
	msg_t msg;
	req_t *request;
	IpVec needle;

	result_t answer;
	ResVec answer_vec;
	int haystack_size; /* number of entries in haystack */
	size_t haystack_mem_size;
	size_t db_left;

	worker_t workers[NUM_WORKER];
	task_t tasks[NUM_WORKER];
	int i, j, worker, found;
	float dist_ratio;

	pthread_mutex_t mx_wait_worker;
	pthread_cond_t cd_wait_worker;

	pthread_mutex_init(&mx_wait_worker, NULL);
	pthread_cond_init(&cd_wait_worker, NULL);

	for (i = 0; i < NUM_WORKER; i++)
		tasks[i].result = (struct _interim *)malloc(
				MAX_IPTS * sizeof(struct _interim));

	init_worker_pool(workers, db, &cd_wait_worker,
			search_cpu_main, CPU_CHUNK_SIZE, NUMCPU,
			search_gpu_main, GPU_CHUNK_SIZE, NUMGPU);

	while (1) {
		PROFILE_INIT();
		PROFILE_VAR(search);
		PROFILE_VAR(tail);

		msg_read(&sc->msgbox, &msg);

		PROFILE_START();

		if (!msg.content)
			continue;

		request = (req_t *)msg.content;

		if (!request->img)
			continue;

		surfDetDes(request->img, needle, false, 3, 4, 3, SURF_THRESHOLD);

		printf("needle: %d ipts (max: %d)\n", needle.size(), MAX_IPTS);

		while (needle.size() > MAX_IPTS)
			needle.pop_back();

		for (i = 0; i < NUM_WORKER; i++)
			for (j = 0; j < (int)needle.size(); j++) {
				struct _interim *result = &(tasks[i].result[j]);
				result->dist_first = FLT_MAX;
				result->dist_second = FLT_MAX;
				result->lat_first = 0;
				result->lng_first = 0;
			}

		PROFILE_FROM(search);
		/* Read DB and do searching */
		db_left = db->db_len;
		while (1) {

			/* Choose an idle worker */
			for (worker = 0; worker < NUM_WORKER; worker++)
				if (workers[worker].isbusy == false)
					break;
			if (worker == NUM_WORKER) {
				/* No worker is available. Wait */
				pthread_mutex_lock(&mx_wait_worker);
				pthread_cond_wait(&cd_wait_worker, &mx_wait_worker);
				pthread_mutex_unlock(&mx_wait_worker);
				continue;
			}

			/* Calculate how much DB should process */
			haystack_size = MIN(workers[worker].chunk_size, db_left)
				/ sizeof(ipoint_t);
			haystack_mem_size = haystack_size * sizeof(ipoint_t);

			if (haystack_size <= 0) {
				/* Nothing left. Finish searching */
				for (i = 0; i < NUM_WORKER; i++)
					if (workers[i].isbusy == true) break;
				if (i == NUM_WORKER)
					break;
				else {
					PROFILE_FROM(tail);
					/* Wait until any worker finishes */
					pthread_mutex_lock(&mx_wait_worker);
					pthread_cond_wait(&cd_wait_worker, &mx_wait_worker);
					pthread_mutex_unlock(&mx_wait_worker);
					PROFILE_TO(tail);
					continue;
				}
			}

			pthread_mutex_lock(&db->mx_db);
			/*
			while (db_readable(db) < haystack_mem_size)
				pthread_cond_wait(&db->cd_reader, &db->mx_db);
				*/

			haystack_mem_size = db_acquire(db,
					(void **)&(tasks[worker].haystack), haystack_mem_size);
			if (haystack_mem_size <= 0) {
				pthread_mutex_unlock(&db->mx_db);
				continue;
			}
			tasks[worker].haystack_size = haystack_mem_size / sizeof(ipoint_t);
			tasks[worker].needle = needle;

			workers[worker].isbusy = true;
			msg_write(&(workers[worker].msgbox), &tasks[worker], NULL, NULL);

			db_left -= haystack_mem_size;

			pthread_mutex_unlock(&db->mx_db);
		}
		PROFILE_TO(search);

		/* merge results from workers */
		for (i = 1; i < NUM_WORKER; i++)
			merge_result(tasks[0].result,
					tasks[i].result, needle.size());

		struct _interim *merged_result = tasks[0].result;

		for (i = 0; i < (int)needle.size(); i++) {
			dist_ratio = merged_result[i].dist_first
				/ merged_result[i].dist_second;
			if (dist_ratio < MATCH_THRESH_SQUARE){
				found = -1;
				for (j = 0; j < (int)answer_vec.size(); j++) {
					if (answer_vec[j].latitude == merged_result[i].lat_first
							&& answer_vec[j].longitude == merged_result[i].lng_first) {
						answer_vec[j].score += 1 - dist_ratio;
						found = 1;
						break;
					}
				}

				if (found < 0) {
					answer.latitude = merged_result[i].lat_first;
					answer.longitude = merged_result[i].lng_first;
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

		if (answer_vec.size() > 0) {
			request->latitude = answer_vec[0].latitude;
			request->longitude = answer_vec[0].longitude;
			request->score = answer_vec[0].score;
		}

		PROFILE_END();
		PROFILE_PRINT(stdout);
		if (msg.cb != NULL)
			msg.cb(&msg);

		/* XXX: Is this sufficient to free all the memory? */
		needle.clear();
		answer_vec.clear();

		for (i = 0; i < NUMCPU + NUMGPU; i++) {
			printf("%s%d: %d times called, %lu bytes processed\n",
					(i < NUMGPU) ? "GPU" : "CPU",
					(i < NUMGPU) ? i : i - NUMGPU,
					workers[i].stat_called,
					workers[i].stat_bytes);
			workers[i].stat_called = 0;
			workers[i].stat_bytes = 0;
		}
	}

	return NULL;
}

