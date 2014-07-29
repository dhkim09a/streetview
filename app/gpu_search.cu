#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#include "db.h"
#include "search.h"

#ifdef PROFILE_CUDA
#define PROFILE_ON
#endif
#include "profile.h"

/* Use device 0 */
#define DEV_ID 0

#define GRID_DIM 1

#define CALL(result, func, handle_failure) \
	if ((result = func) != CUDA_SUCCESS) { \
		printf("%s failed: %d\n", #func, result); \
		handle_failure; \
	}

typedef struct _ipoint_essence_t {
	float vec[VEC_DIM] __attribute__((aligned (4)));
} __attribute__((packed)) ipoint_essence_t;

/* FIXME: The result is a bit different from CPU's */
__global__ void doSearchKernel (int shared_mem_size, int needle_idx,
		ipoint_essence_t *needle, int needle_size,
		ipoint_t *haystack, int haystack_size,
		struct _interim *interim, int interim_size_local)
{
	if (threadIdx.x + needle_idx >= needle_size)
		return;

	register float dist;//, temp;
	int i, j, k;
	int batch;

	struct _interim *interim_local =
		&(interim[(interim_size_local * blockIdx.x)
				+ threadIdx.x + needle_idx]);
	batch = haystack_size / gridDim.x;
	int haystack_size_local = ((blockIdx.x + 1) * batch) > haystack_size ?
		(haystack_size % batch) : batch;

	/* Copy needle into local memory */
	ipoint_essence_t needle_local;
	for (i = 0; i < VEC_DIM; i++)
		needle_local.vec[i] = needle[threadIdx.x + needle_idx].vec[i];

	struct _interim interim_temp;
	if (interim_local->lat_first == 0) {
		interim_temp.dist_first = FLT_MAX;
		interim_temp.dist_second = FLT_MAX;
	}
	else {
		interim_temp.dist_first = interim_local->dist_first;
		interim_temp.dist_second = interim_local->dist_second;
		interim_temp.lat_first = interim_local->lat_first;
		interim_temp.lng_first = interim_local->lng_first;
	}

	extern __shared__ ipoint_t haystack_shared[];
	batch = shared_mem_size / sizeof(ipoint_t);
	int iter;
	for (k = 0; k <= (haystack_size_local / batch); k++) {

		iter = ((k + 1) * batch) > haystack_size_local ?
			(haystack_size_local % batch) : batch;

		/* Copy haystack into shared memory */
		if (threadIdx.x == 0)
			for (i = 0; i < iter; i++)
				haystack_shared[i] =
					haystack[((haystack_size / gridDim.x) * blockIdx.x)
					+ (k * batch) + i];

		__syncthreads();

		for (i = 0; i < iter; i++) {
			dist = 0;
#if REG >= 128
			register float hss[0x10];
			for (j = 0; j < VEC_DIM; j += 0x10)
#else
			register float hss[0x8];
			for (j = 0; j < VEC_DIM; j += 0x8)
#endif
			{
				hss[0x0] = haystack_shared[i].vec[j];
				hss[0x1] = haystack_shared[i].vec[j + 0x1];
				hss[0x2] = haystack_shared[i].vec[j + 0x2];
				hss[0x3] = haystack_shared[i].vec[j + 0x3];
				hss[0x4] = haystack_shared[i].vec[j + 0x4];
				hss[0x5] = haystack_shared[i].vec[j + 0x5];
				hss[0x6] = haystack_shared[i].vec[j + 0x6];
				hss[0x7] = haystack_shared[i].vec[j + 0x7];
#if REG >= 128
				hss[0x8] = haystack_shared[i].vec[j + 0x8];
				hss[0x9] = haystack_shared[i].vec[j + 0x9];
				hss[0xA] = haystack_shared[i].vec[j + 0xA];
				hss[0xB] = haystack_shared[i].vec[j + 0xB];
				hss[0xC] = haystack_shared[i].vec[j + 0xC];
				hss[0xD] = haystack_shared[i].vec[j + 0xD];
				hss[0xE] = haystack_shared[i].vec[j + 0xE];
				hss[0xF] = haystack_shared[i].vec[j + 0xF];
#endif
				dist
					+= ((needle_local.vec[j] - hss[0x0])
							* (needle_local.vec[j] - hss[0x0]))
					+ ((needle_local.vec[j + 0x1] - hss[0x1])
							* (needle_local.vec[j + 0x1] - hss[0x1]))
					+ ((needle_local.vec[j + 0x2] - hss[0x2])
							* (needle_local.vec[j + 0x2] - hss[0x2]))
					+ ((needle_local.vec[j + 0x3] - hss[0x3])
							* (needle_local.vec[j + 0x3] - hss[0x3]))
					+ ((needle_local.vec[j + 0x4] - hss[0x4])
							* (needle_local.vec[j + 0x4] - hss[0x4]))
					+ ((needle_local.vec[j + 0x5] - hss[0x5])
							* (needle_local.vec[j + 0x5] - hss[0x5]))
					+ ((needle_local.vec[j + 0x6] - hss[0x6])
							* (needle_local.vec[j + 0x6] - hss[0x6]))
					+ ((needle_local.vec[j + 0x7] - hss[0x7])
							* (needle_local.vec[j + 0x7] - hss[0x7]));
#if REG >= 128
					+ ((needle_local.vec[j + 0x8] - hss[0x8])
							* (needle_local.vec[j + 0x8] - hss[0x8]))
					+ ((needle_local.vec[j + 0x9] - hss[0x9])
							* (needle_local.vec[j + 0x9] - hss[0x9]))
					+ ((needle_local.vec[j + 0xA] - hss[0xA])
							* (needle_local.vec[j + 0xA] - hss[0xA]))
					+ ((needle_local.vec[j + 0xB] - hss[0xB])
							* (needle_local.vec[j + 0xB] - hss[0xB]))
					+ ((needle_local.vec[j + 0xC] - hss[0xC])
							* (needle_local.vec[j + 0xC] - hss[0xC]))
					+ ((needle_local.vec[j + 0xD] - hss[0xD])
							* (needle_local.vec[j + 0xD] - hss[0xD]))
					+ ((needle_local.vec[j = 0xE] - hss[0xE])
							* (needle_local.vec[j + 0xE] - hss[0xE]))
					+ ((needle_local.vec[j + 0xF] - hss[0xF])
							* (needle_local.vec[j + 0xF] - hss[0xF]));
#endif
			}
			if (dist < interim_temp.dist_first) {
				interim_temp.lat_first =
					haystack_shared[i].latitude;
				interim_temp.lng_first =
					haystack_shared[i].longitude;
				interim_temp.dist_second =
					interim_temp.dist_first;
				interim_temp.dist_first = dist;
			}
			else if (dist < interim_temp.dist_second)
				interim_temp.dist_second = dist;
		}
	}

	interim_local->lat_first = interim_temp.lat_first;
	interim_local->lng_first = interim_temp.lng_first;
	interim_local->dist_first = interim_temp.dist_first;
	interim_local->dist_second = interim_temp.dist_second;

	return;
}

typedef struct _local_vars_t {
//	cudaDeviceProp device_prop;
	int device_maxThreadsPerBlock;
	int device_regsPerBlock;
	int device_sharedMemPerBlock;
	cudaStream_t stream;

	ipoint_essence_t *needle_essence_h, *needle_essence_d;
	ipoint_t *haystack_d;
	struct _interim *interim_h, *interim_d;
} local_vars_t;

static int doSearch (local_vars_t *lv,
		IpVec *needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size)
{
	PROFILE_INIT();
	PROFILE_START();
	PROFILE_VAR(copy_needle);
	PROFILE_VAR(run_kernel);
	PROFILE_VAR(copy_result);
	PROFILE_VAR(post_processing);

	int i, j, iter;
	float dist;

	int needle_size = (*needle).size();
	cudaError_t err;

//	unsigned int stream_dim = (unsigned int)device_prop.multiProcessorCount;
//	unsigned int grid_dim = 1;
	unsigned int block_dim =
		MIN(needle_size, (unsigned int)lv->device_maxThreadsPerBlock);
	block_dim = MIN(block_dim, (unsigned int)(lv->device_regsPerBlock / REG));

	for (i = 0; i < needle_size; i++)
		for (j = 0; j < VEC_DIM; j++)
			lv->needle_essence_h[i].vec[j] = (*needle)[i].descriptor[j];

	PROFILE_FROM(copy_needle);
	if (cudaMemcpyAsync(lv->needle_essence_d, lv->needle_essence_h,
				needle_size * sizeof(ipoint_essence_t),
				cudaMemcpyHostToDevice, lv->stream) != cudaSuccess) {
		fprintf(stderr,
				"cudaMemcpy(needle_essence_d, needle_essence_h) failed\n");
		return -1;
	}
	PROFILE_TO(copy_needle);
#if 0
	if (cudaMemsetAsync(lv->interim_d, 0,
				GRID_DIM * sizeof(struct _interim) * needle_size, lv->stream)
			!= cudaSuccess) {
		fprintf(stderr, "cudaMemset(interim_d) failed\n");
		return -1;
	}
#endif
	int stream_haystack_quota = haystack_size;
	int stream_haystack_size = stream_haystack_quota > haystack_size ?
		(haystack_size % stream_haystack_quota) : stream_haystack_quota;

	if (cudaMemcpyAsync(
				(ipoint_t *)(lv->haystack_d),
				(ipoint_t *)(haystack),
				stream_haystack_size * sizeof(ipoint_t),
				cudaMemcpyHostToDevice, lv->stream) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(haystack_d, haystack) failed\n");
		return -1;
	}

	for (i = 0; i <= needle_size / block_dim; i++) {

		PROFILE_FROM(run_kernel);
		/* Run CUDA kernel */
		stream_haystack_size
			= stream_haystack_quota > haystack_size ?
			(haystack_size % stream_haystack_quota) : stream_haystack_quota;

		doSearchKernel <<<
			GRID_DIM,
			(block_dim * (i + 1)) > needle_size ?
				(needle_size % block_dim) : block_dim,
			lv->device_sharedMemPerBlock,
			lv->stream >>>
				(lv->device_sharedMemPerBlock, i * block_dim,
				 lv->needle_essence_d, needle_size,
				 (ipoint_t *)(lv->haystack_d),
				 stream_haystack_size,
				 lv->interim_d, needle_size);
		PROFILE_TO(run_kernel);

	}

	PROFILE_FROM(copy_result);
	/* Copy result to host */
	err = cudaMemcpyAsync(lv->interim_h, lv->interim_d,
			GRID_DIM * sizeof(struct _interim) * needle_size,
			cudaMemcpyDeviceToHost, lv->stream);
	if (err != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy(interim_h, interim_d): %s\n",
	//			cudaGetErrorString(err));
		return -1;
	}
	PROFILE_TO(copy_result);

	PROFILE_FROM(post_processing);
	iter = MIN((int)(*needle).size(), result_size);
	for (i = 0; i < iter; i++) {
		for (j = 0; j < (int)GRID_DIM; j++) {
			if (result[i].dist_first == FLT_MAX) {
				result[i].lat_first =
					lv->interim_h[(j * needle_size) + i].lat_first;
				result[i].lng_first =
					lv->interim_h[(j * needle_size) + i].lng_first;
				result[i].dist_first =
					lv->interim_h[(j * needle_size) + i].dist_first;
				result[i].dist_second =
					lv->interim_h[(j * needle_size) + i].dist_second;
				continue;
			}

			dist = lv->interim_h[(j * needle_size) + i].dist_first;
			if (dist < result[i].dist_first) {
				result[i].lat_first =
					lv->interim_h[(j * needle_size) + i].lat_first;
				result[i].lng_first =
					lv->interim_h[(j * needle_size) + i].lng_first;
				result[i].dist_second = result[i].dist_first;
				result[i].dist_first = dist;
			}
			else if (dist < result[i].dist_second)
				result[i].dist_second = dist;

			dist = lv->interim_h[(j * needle_size) + i].dist_second;
			if (dist < result[i].dist_first) {
				result[i].lat_first =
					lv->interim_h[(j * needle_size) + i].lat_first;
				result[i].lng_first =
					lv->interim_h[(j * needle_size) + i].lng_first;
				result[i].dist_second = result[i].dist_first;
				result[i].dist_first = dist;
			}
			else if (dist < result[i].dist_second)
				result[i].dist_second = dist;
		}
	}
	PROFILE_TO(post_processing);

	PROFILE_END();
	PROFILE_PRINT(stdout);
	
	return 0;
}

int init_gpu_worker_pool(worker_t *workers,
		db_t *db, pthread_cond_t *cd_wait_worker,
		void *(*thread_main)(void *arg),
		size_t chunk_size, int num_threads)
{
	int i;

	for (i = 0; i < num_threads; i++) {
		local_vars_t *local_vars = (local_vars_t *)malloc(sizeof(local_vars_t));

		msg_init_box(&workers[i].msgbox);
		workers[i].dead = false;
		workers[i].isbusy = false;
		workers[i].chunk_size = chunk_size;
		workers[i].ptr = (void *)local_vars;
		workers[i].db = db;
		workers[i].cd_wait_worker = cd_wait_worker;
		pthread_create(&workers[i].tid, NULL, thread_main, &workers[i]);
	}

	return 0;
}


void *search_gpu_main(void *arg)
{
	worker_t *me = (worker_t *)arg;
	local_vars_t *local_vars = (local_vars_t *)me->ptr;
	msg_t msg;

	CUdevice device;
	CUcontext ctx;
	CUresult res;

	CALL(res, cuDeviceGet(&device, 0), return NULL);

	CALL(res, cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, device),
			return NULL);

//	cudaGetDeviceProperties(&(local_vars->device_prop), DEV_ID);
	CALL(res, cuDeviceGetAttribute(&local_vars->device_maxThreadsPerBlock,
				CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device),
			return NULL);
	CALL(res, cuDeviceGetAttribute(&local_vars->device_regsPerBlock,
				CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device),
			return NULL);
	CALL(res, cuDeviceGetAttribute(&local_vars->device_sharedMemPerBlock,
				CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device),
			return NULL);

//	cudaStreamCreate(&local_vars->stream);
	CALL(res, cuStreamCreate(&local_vars->stream, CU_STREAM_NON_BLOCKING),
			return NULL);

#if 0
	local_vars->needle_essence_h = (ipoint_essence_t *)malloc(
			MAX_IPTS * sizeof(ipoint_essence_t));
	local_vars->interim_h = (struct _interim *)malloc(
			GRID_DIM * sizeof(struct _interim) * MAX_IPTS);
#if 0
	cudaHostRegister(local_vars->needle_essence_h,
			MAX_IPTS * sizeof(ipoint_essence_t),
			cudaHostRegisterPortable);
	cudaHostRegister(local_vars->interim_h,
			GRID_DIM * sizeof(struct _interim) * MAX_IPTS,
			cudaHostRegisterPortable);
#endif
#else
	CALL(res, cuMemAllocHost((void **)&(local_vars->needle_essence_h),
			MAX_IPTS * sizeof(ipoint_essence_t)), return NULL);
	CALL(res, cuMemAllocHost((void **)&(local_vars->interim_h),
			GRID_DIM * sizeof(struct _interim) * MAX_IPTS), return NULL);

	printf("%p, %p\n", local_vars->needle_essence_h, local_vars->interim_h);
#endif
	if (cudaMalloc((void **)&(local_vars->needle_essence_d),
				MAX_IPTS * sizeof(ipoint_essence_t)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(needle_essence_d) failed\n");
		return NULL;
	}

	if (cudaMalloc((void **)&(local_vars->haystack_d), me->chunk_size)
			!= cudaSuccess) {
		fprintf(stderr, "cudaMalloc(haystack_d) failed\n");
		return NULL;
	}

	if (cudaMalloc((void **)&(local_vars->interim_d),
				GRID_DIM * sizeof(struct _interim) * MAX_IPTS)
			!= cudaSuccess) {
		fprintf(stderr, "cudaMalloc(interim_d) failed\n");
		return NULL;
	}

	while (1) {
		if (me->dead)
			break;

		msg_read(&me->msgbox, &msg);
		task_t *task = (task_t *)msg.content;

		doSearch((local_vars_t *)me->ptr,
				&task->needle, task->haystack, task->haystack_size,
				task->result, (task->needle).size());

		me->isbusy = false;
		db_release(me->db,
				task->haystack, task->haystack_size * sizeof(ipoint_t));
		pthread_cond_signal(me->cd_wait_worker);
	}
#if 0
	free(needle_essence_h);
	free(interim_h);
#if 0
	for (i = 0; i < stream_dim; i++) {
		cudaStreamDestroy(stream[i]);
		free(&stream[i]);
	}
#endif

	cudaFree(needle_essence_d);
	cudaFree(haystack_d);
	cudaFree(interim_d);
#endif
	return NULL;
}
