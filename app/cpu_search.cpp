#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <algorithm>
#include <vector>

#ifdef AVX
#include <immintrin.h>
#endif

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

static int doSearch (IpVec *needle, ipoint_t *haystack, int haystack_size,
		struct _interim *interim, int interim_size)
{
	float dist;
	int i, j, k, l;
	int iter;

	iter = MIN(interim_size, (int)(*needle).size());
	int batch = MIN(100, iter);
	for (l = 0; l <= (iter / batch); l++) {
		for (j = 0; j < haystack_size; j++) {
			for (i = l * batch; (i < iter) && (i < (l + 1) * batch); i++) {
#ifdef AVX
				float res[8] = {0};

				__m256 a_v0 = _mm256_loadu_ps(&((*needle)[i].descriptor[0])),
					   a_v1 = _mm256_loadu_ps(&((*needle)[i].descriptor[8])),
					   a_v2 = _mm256_loadu_ps(&((*needle)[i].descriptor[16])),
					   a_v3 = _mm256_loadu_ps(&((*needle)[i].descriptor[24])),
					   a_v4 = _mm256_loadu_ps(&((*needle)[i].descriptor[32])),
					   a_v5 = _mm256_loadu_ps(&((*needle)[i].descriptor[40])),
					   a_v6 = _mm256_loadu_ps(&((*needle)[i].descriptor[48])),
					   a_v7 = _mm256_loadu_ps(&((*needle)[i].descriptor[56]));

				__m256 b_v0 = _mm256_loadu_ps(&(haystack[j].vec[0])),
					   b_v1 = _mm256_loadu_ps(&(haystack[j].vec[8])),
					   b_v2 = _mm256_loadu_ps(&(haystack[j].vec[16])),
					   b_v3 = _mm256_loadu_ps(&(haystack[j].vec[24])),
					   b_v4 = _mm256_loadu_ps(&(haystack[j].vec[32])),
					   b_v5 = _mm256_loadu_ps(&(haystack[j].vec[40])),
					   b_v6 = _mm256_loadu_ps(&(haystack[j].vec[48])),
					   b_v7 = _mm256_loadu_ps(&(haystack[j].vec[56]));

				a_v0 = _mm256_sub_ps(a_v0, b_v0);
				a_v1 = _mm256_sub_ps(a_v1, b_v1);
				a_v2 = _mm256_sub_ps(a_v2, b_v2);
				a_v3 = _mm256_sub_ps(a_v3, b_v3);
				a_v4 = _mm256_sub_ps(a_v4, b_v4);
				a_v5 = _mm256_sub_ps(a_v5, b_v5);
				a_v6 = _mm256_sub_ps(a_v6, b_v6);
				a_v7 = _mm256_sub_ps(a_v7, b_v7);

				a_v0 = _mm256_mul_ps(a_v0, a_v0);
				a_v1 = _mm256_mul_ps(a_v1, a_v1);
				a_v2 = _mm256_mul_ps(a_v2, a_v2);
				a_v3 = _mm256_mul_ps(a_v3, a_v3);
				a_v4 = _mm256_mul_ps(a_v4, a_v4);
				a_v5 = _mm256_mul_ps(a_v5, a_v5);
				a_v6 = _mm256_mul_ps(a_v6, a_v6);
				a_v7 = _mm256_mul_ps(a_v7, a_v7);

				a_v0 = _mm256_add_ps(a_v0, a_v1);
				a_v0 = _mm256_add_ps(a_v0, a_v2);
				a_v0 = _mm256_add_ps(a_v0, a_v3);
				a_v0 = _mm256_add_ps(a_v0, a_v4);
				a_v0 = _mm256_add_ps(a_v0, a_v5);
				a_v0 = _mm256_add_ps(a_v0, a_v6);
				a_v0 = _mm256_add_ps(a_v0, a_v7);

				_mm256_storeu_ps(res, a_v0);
				
				dist = 0;
				for (k = 0; k < 8; k++)
					dist += res[k];
#else /* AVX */
				float temp;
				dist = 0;
				for (k = 0; k < VEC_DIM; k++) {
					temp = (*needle)[i].descriptor[k] - haystack[j].vec[k];
					dist += temp * temp;
				}
#endif /* AVX */

				if (dist < interim[i].dist_first) {
					interim[i].lat_first = haystack[j].latitude;
					interim[i].lng_first = haystack[j].longitude;
					interim[i].dist_second = interim[i].dist_first;
					interim[i].dist_first = dist;
				}
				else if (dist < interim[i].dist_second)
					interim[i].dist_second = dist;
			}
		}
	}

	return 0;
}

void *search_cpu_main(void *arg)
{
	worker_t *me = (worker_t *)arg;
	msg_t msg;

	while (1) {
		if (me->dead)
			break;

		msg_read(&me->msgbox, &msg);
		task_t *task = (task_t *)msg.content;

		doSearch(&task->needle, task->haystack, task->haystack_size,
				task->result, (task->needle).size());

		me->isbusy = false;
		db_release(me->db,
				task->haystack, task->haystack_size * sizeof(ipoint_t));
		pthread_cond_signal(me->cd_wait_worker);
	}

	return NULL;
}
