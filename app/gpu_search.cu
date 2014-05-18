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
#define PROFILE_FROM(tv_from) \
do { \
	__sync_synchronize(); \
	gettimeofday((tv_from), NULL); \
} while (0)

#define PROFILE_TO(tv_from, tv_to, time_ms) \
do { \
	__sync_synchronize(); \
	gettimeofday(tv_to, NULL); \
	time_ms = ((tv_to)->tv_sec - (tv_from)->tv_sec) * 1000 \
		+ ((tv_to)->tv_usec - (tv_from)->tv_usec) / 1000; \
} while (0)
#else
#define PROFILE_FROM(args...)
#define PROFILE_TO(args...)
#endif

/* Use device 0 */
#define DEV_ID 0

typedef struct _ipoint_essence_t {
	float vec[VEC_DIM] __attribute__((aligned (4)));
} __attribute__((packed)) ipoint_essence_t;

/* FIXME: The result is a bit different from CPU's */
__global__ void doSearchKernel (int shared_mem_size,
		ipoint_essence_t *needle, int needle_size,
		ipoint_t *haystack, int haystack_size,
		struct _interim *interim, int interim_size_local)
{
	if (threadIdx.x >= needle_size)
		return;

	float dist, temp;
	int i, j, k;

	struct _interim *interim_local =
		&(interim[(interim_size_local * blockIdx.x) + threadIdx.x]);
	int haystack_size_local = MIN( haystack_size / gridDim.x,
			haystack_size - ((haystack_size / gridDim.x) * blockIdx.x));

	/* Copy needle into local memory */
	ipoint_essence_t needle_local;
	for (i = 0; i < VEC_DIM; i++)
		needle_local.vec[i] = needle[threadIdx.x].vec[i];

	struct _interim interim_temp;
	interim_temp.dist_first = FLT_MAX;
	interim_temp.dist_second = FLT_MAX;

	extern __shared__ ipoint_t haystack_shared[];
	int batch = shared_mem_size / sizeof(ipoint_t);
	int iter;
	for (k = 0; k < (haystack_size_local / batch + 1); k++) {

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
			for (j = 0; j < VEC_DIM; j++) {
				temp = needle_local.vec[j] - haystack_shared[i].vec[j];
				dist += temp * temp;
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

int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int dummy)
{
#ifdef PROFILE_CUDA
	struct timeval tv_from, tv_to;
	struct timeval tv_total_from, tv_total_to;
	unsigned long init_device_ms = 0;
	unsigned long copy_needle_ms = 0, copy_haystack_ms = 0;
	unsigned long run_kernel_ms = 0, copy_result_ms = 0;
	unsigned long postprocessing_ms = 0;
	unsigned long etc_ms = 0, total_ms = 0;

	PROFILE_FROM(&tv_total_from);
#endif

	int i, j, iter;
	float dist;

	ipoint_essence_t *needle_essence_h, *needle_essence_d;
	ipoint_t *haystack_d;
	struct _interim *interim_h, *interim_d;
	int needle_size = needle.size();
	cudaError_t err;

	int numcore = 128;
	unsigned int block_dim = needle_size;
	unsigned int grid_dim = numcore;

	cudaSetDevice(DEV_ID);
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, DEV_ID);

	PROFILE_FROM(&tv_from);
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(&tv_from, &tv_to, init_device_ms);

	needle_essence_h = (ipoint_essence_t *)malloc(
			needle_size * sizeof(ipoint_essence_t));
	for (i = 0; i < needle_size; i++)
		for (j = 0; j < VEC_DIM; j++)
			needle_essence_h[i].vec[j] = needle[i].descriptor[j];

	PROFILE_FROM(&tv_from);
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
	PROFILE_TO(&tv_from, &tv_to, copy_needle_ms);

	PROFILE_FROM(&tv_from);
	/* Copy haystack to device */
	if (cudaMalloc((void **)&haystack_d,
				haystack_size * sizeof(ipoint_t)) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(haystack_d) failed\n");
		return -1;
	}
	if (cudaMemcpy(haystack_d, haystack,
				haystack_size * sizeof(ipoint_t),
				cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(haystack_d, haystack) failed\n");
		return -1;
	}
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(&tv_from, &tv_to, copy_haystack_ms);

	/* Allocate memory for result
	 * TODO: Still the result must be copied from device is about
	 * hundreads of MB. Need to reduce them. */
	if (cudaMalloc((void **)&interim_d,
			numcore * sizeof(struct _interim) * needle_size) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(interim_d) failed\n");
		return -1;
	}
	interim_h = (struct _interim *)malloc(
			numcore * sizeof(struct _interim) * needle_size);

	PROFILE_FROM(&tv_from);
	/* Run CUDA kernel */
	doSearchKernel <<< grid_dim, block_dim, device_prop.sharedMemPerBlock >>>
		(device_prop.sharedMemPerBlock,
		 needle_essence_d, needle_size,
		 haystack_d, haystack_size,
		 interim_d, needle_size);
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(&tv_from, &tv_to, run_kernel_ms);

	PROFILE_FROM(&tv_from);
	/* Copy result to host */
	err = cudaMemcpy(interim_h, interim_d,
			numcore * sizeof(struct _interim) * needle_size,
			cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(interim_h, interim_d): %s\n",
				cudaGetErrorString(err));
		return -1;
	}
#ifdef PROFILE_CUDA
	cudaDeviceSynchronize();
#endif
	PROFILE_TO(&tv_from, &tv_to, copy_result_ms);

	PROFILE_FROM(&tv_from);
	iter = MIN((int)needle.size(), result_size);
	for (i = 0; i < iter; i++) {
		for (j = 0; j < numcore; j++) {
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
		}
	}
	PROFILE_TO(&tv_from, &tv_to, postprocessing_ms);

	free(needle_essence_h);
	free(interim_h);

	cudaFree(needle_essence_d);
	cudaFree(haystack_d);
	cudaFree(interim_d);

#ifdef PROFILE_CUDA
	PROFILE_TO(&tv_total_from, &tv_total_to, total_ms);
	etc_ms = total_ms
		- init_device_ms
		- copy_needle_ms - copy_haystack_ms
		- run_kernel_ms - copy_result_ms
		- postprocessing_ms;

	printf("[CUDA Profile]\n"
		   "Initialize device      : %7lu ms (%5.2f %%)\n"
		   "Copy needle to device  : %7lu ms (%5.2f %%)\n"
		   "Copy haystack to device: %7lu ms (%5.2f %%)\n"
		   "Run CUDA kernel        : %7lu ms (%5.2f %%)\n"
		   "Copy result from device: %7lu ms (%5.2f %%)\n"
		   "Post processing        : %7lu ms (%5.2f %%)\n"
		   "etc.                   : %7lu ms (%5.2f %%)\n"
		   "Total                  : %7lu ms\n",
		   init_device_ms, 100 * (float)init_device_ms / (float)total_ms,
		   copy_needle_ms, 100 * (float)copy_needle_ms / (float)total_ms,
		   copy_haystack_ms, 100 * (float)copy_haystack_ms / (float)total_ms,
		   run_kernel_ms, 100 * (float)run_kernel_ms / (float)total_ms,
		   copy_result_ms, 100 * (float)copy_result_ms / (float)total_ms,
		   postprocessing_ms, 100 * (float)postprocessing_ms / (float)total_ms,
		   etc_ms, 100 * (float)etc_ms / (float)total_ms,
		   total_ms);
#endif
	return 0;
}
