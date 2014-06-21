#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "db.h"
#include "search.h"

#ifdef PROFILE_CUDA
#define PROFILE_ON
#endif
#include "profile.h"

/* Use device 0 */
#define DEV_ID 0

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

int searchGPU (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int dummy)
{
	cudaSetDevice(DEV_ID);
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

	PROFILE_START();
	PROFILE_VAR(init_device);
	PROFILE_VAR(copy_needle);
	PROFILE_VAR(copy_haystack);
	PROFILE_VAR(run_kernel);
	PROFILE_VAR(copy_result);
	PROFILE_VAR(post_processing);

	int i, j, iter;
	float dist;

	ipoint_essence_t *needle_essence_h, *needle_essence_d;
	ipoint_t *haystack_d;
	struct _interim *interim_h, *interim_d;
	int needle_size = needle.size();
	cudaError_t err;

	PROFILE_FROM(init_device);
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(init_device);

	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, DEV_ID);

	cudaStream_t *stream;

	unsigned int stream_dim = (unsigned int)device_prop.multiProcessorCount;
	unsigned int grid_dim = 1;
	unsigned int block_dim =
		MIN(needle_size, (unsigned int)device_prop.maxThreadsPerBlock);
	block_dim = MIN(block_dim, (unsigned int)(device_prop.regsPerBlock / REG));

	stream = (cudaStream_t *)malloc(stream_dim * sizeof(cudaStream_t));
	for (i = 0; i < (int)stream_dim; i++)
		cudaStreamCreate(&stream[i]);

	needle_essence_h = (ipoint_essence_t *)malloc(
			needle_size * sizeof(ipoint_essence_t));
	for (i = 0; i < needle_size; i++)
		for (j = 0; j < VEC_DIM; j++)
			needle_essence_h[i].vec[j] = needle[i].descriptor[j];

	PROFILE_FROM(copy_needle);
	/* Copy needle to device */
	if (cudaMalloc((void **)&needle_essence_d,
				needle_size * sizeof(ipoint_essence_t)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(needle_essence_d) failed\n");
		return -1;
	}
	if (cudaMemcpy(needle_essence_d, needle_essence_h,
				needle_size * sizeof(ipoint_essence_t),
				cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr,
				"cudaMemcpy(needle_essence_d, needle_essence_h) failed\n");
		return -1;
	}
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(copy_needle);

	PROFILE_FROM(copy_haystack);
	/* Copy haystack to device */
	if (cudaMalloc((void **)&haystack_d,
				haystack_size * sizeof(ipoint_t)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(haystack_d) failed\n");
		return -1;
	}
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(copy_haystack);

	/* Allocate memory for result
	 * TODO: Still the result must be copied from device is about
	 * hundreads of MB. Need to reduce them. */
	if (cudaMalloc((void **)&interim_d,
			grid_dim * stream_dim * sizeof(struct _interim) * needle_size) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(interim_d) failed\n");
		return -1;
	}
	if (cudaMemset(interim_d, 0,
				grid_dim * stream_dim * sizeof(struct _interim) * needle_size) != cudaSuccess) {
		fprintf(stderr, "cudaMemset(interim_d) failed\n");
		return -1;
	}
	interim_h = (struct _interim *)malloc(
			grid_dim * stream_dim * sizeof(struct _interim) * needle_size);

	int stream_haystack_quota = haystack_size / stream_dim;
	int stream_haystack_size;
	for (j = 0; j < (int)stream_dim; j++) {
		stream_haystack_size
			= (j + 1) * stream_haystack_quota > haystack_size ?
			(haystack_size % stream_haystack_quota) : stream_haystack_quota;

		if (cudaMemcpyAsync(
					(ipoint_t *)(&haystack_d[stream_haystack_quota * j]),
					(ipoint_t *)(&haystack[stream_haystack_quota * j]),
					stream_haystack_size * sizeof(ipoint_t),
					cudaMemcpyHostToDevice, stream[j]) != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy(haystack_d, haystack) failed\n");
			return -1;
		}
	}
	for (i = 0; i <= needle_size / block_dim; i++) {

		PROFILE_FROM(run_kernel);
		/* Run CUDA kernel */
		for (j = 0; j < (int)stream_dim; j++) {
		stream_haystack_size
			= (j + 1) * stream_haystack_quota > haystack_size ?
			(haystack_size % stream_haystack_quota) : stream_haystack_quota;

			doSearchKernel <<<
				grid_dim,
				(block_dim * (i + 1)) > needle_size ?
					(needle_size % block_dim) : block_dim,
				device_prop.sharedMemPerBlock,
				stream[j] >>>
					(device_prop.sharedMemPerBlock, i * block_dim,
					 needle_essence_d, needle_size,
					 (ipoint_t *)(&haystack_d[stream_haystack_quota * j]),
					 stream_haystack_size,
					 &interim_d[needle_size * j], needle_size);
		}
#ifdef PROFILE_CUDA
		cudaDeviceSynchronize();
#endif
		PROFILE_TO(run_kernel);

	}

	PROFILE_FROM(copy_result);
	/* Copy result to host */
	err = cudaMemcpy(interim_h, interim_d,
			grid_dim * stream_dim * sizeof(struct _interim) * needle_size,
			cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(interim_h, interim_d): %s\n",
				cudaGetErrorString(err));
		return -1;
	}
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(copy_result);

	PROFILE_FROM(post_processing);
	iter = MIN((int)needle.size(), result_size);
	for (i = 0; i < iter; i++) {
		for (j = 0; j < (int)(grid_dim * stream_dim); j++) {
			if (result[i].dist_first == FLT_MAX) {
				result[i].lat_first =
					interim_h[(j * needle_size) + i].lat_first;
				result[i].lng_first =
					interim_h[(j * needle_size) + i].lng_first;
				result[i].dist_first =
					interim_h[(j * needle_size) + i].dist_first;
				result[i].dist_second =
					interim_h[(j * needle_size) + i].dist_second;
				continue;
			}

			dist = interim_h[(j * needle_size) + i].dist_first;
			if (dist < result[i].dist_first) {
				result[i].lat_first =
					interim_h[(j * needle_size) + i].lat_first;
				result[i].lng_first =
					interim_h[(j * needle_size) + i].lng_first;
				result[i].dist_second = result[i].dist_first;
				result[i].dist_first = dist;
			}
			else if (dist < result[i].dist_second)
				result[i].dist_second = dist;

			dist = interim_h[(j * needle_size) + i].dist_second;
			if (dist < result[i].dist_first) {
				result[i].lat_first =
					interim_h[(j * needle_size) + i].lat_first;
				result[i].lng_first =
					interim_h[(j * needle_size) + i].lng_first;
				result[i].dist_second = result[i].dist_first;
				result[i].dist_first = dist;
			}
			else if (dist < result[i].dist_second)
				result[i].dist_second = dist;
		}
	}
	PROFILE_TO(post_processing);

	free(needle_essence_h);
	free(interim_h);

	cudaFree(needle_essence_d);
	cudaFree(haystack_d);
	cudaFree(interim_d);

	PROFILE_END();
	PROFILE_PRINT(stdout);
	
	return 0;
}
