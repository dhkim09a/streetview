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
	gettimeofday((tv_from), NULL)
#define PROFILE_TO(tv_from, tv_to, time_ms) \
do { \
	gettimeofday(tv_to, NULL); \
	time_ms = ((tv_to)->tv_sec - (tv_from)->tv_sec) * 1000 \
		+ ((tv_to)->tv_usec - (tv_from)->tv_usec) / 1000; \
} while (0)
#else
#define PROFILE_FROM(args...)
#define PROFILE_TO(args...)
#endif



typedef struct _ipoint_essence_t {
	float vec[VEC_DIM] __attribute__((aligned (4)));
} __attribute__((packed)) ipoint_essence_t;

__global__ void doSearchKernel (ipoint_essence_t *needle, int needle_size,
		ipoint_t *haystack, int haystack_size,
		struct _interim *interim, int interim_size_local)
{
	if (threadIdx.x >= needle_size)
		return;

	ipoint_t *haystack_local =
		&(haystack[haystack_size / gridDim.x * blockIdx.x]);
	int haystack_size_local = MIN( haystack_size / gridDim.x,
			haystack_size - ((haystack_size / gridDim.x) * blockIdx.x));
	struct _interim *interim_local =
		&(interim[interim_size_local * blockIdx.x]);

	float dist, temp;
	int i, j;

	interim_local[threadIdx.x].dist_first = FLT_MAX;
	interim_local[threadIdx.x].dist_second = FLT_MAX;

	for (i = 0; i < haystack_size_local; i++) {
		dist = 0;
		for (j = 0; j < VEC_DIM; j++) {
			temp = needle[threadIdx.x].vec[j] - haystack_local[i].vec[j];
			dist += temp * temp;
		}
		if (dist < interim_local[threadIdx.x].dist_first) {
			interim_local[threadIdx.x].lat_first =
				haystack_local[i].latitude;
			interim_local[threadIdx.x].lng_first =
				haystack_local[i].longitude;
			interim_local[threadIdx.x].dist_second =
				interim_local[threadIdx.x].dist_first;
			interim_local[threadIdx.x].dist_first = dist;
		}
		else if (dist < interim_local[threadIdx.x].dist_second)
			interim_local[threadIdx.x].dist_second = dist;
	}
	return;
}

int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec *result_vec, int dummy)
{
#ifdef PROFILE_CUDA
	struct timeval tv_from, tv_to;
	struct timeval tv_total_from, tv_total_to;
	unsigned long copy_needle_ms = 0, copy_haystack_ms = 0;
	unsigned long run_kernel_ms = 0, copy_result_ms = 0;
	unsigned long postprocessing_ms = 0;
	unsigned long etc_ms = 0, total_ms = 0;

	PROFILE_FROM(&tv_total_from);
#endif

	int i, j, found;
	struct _interim interim;
	result_t result;
	float dist;

	ipoint_essence_t *needle_essence_h, *needle_essence_d;
	ipoint_t *haystack_d;
	struct _interim *interim_h, *interim_d;
	int needle_size = needle.size();
	cudaError_t err;

	int numcore = 512;
	unsigned int block_dim = needle_size;
	unsigned int grid_dim = numcore;

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
	doSearchKernel <<< grid_dim, block_dim >>> (needle_essence_d, needle_size,
			haystack_d, haystack_size, interim_d, needle_size);
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
	for (i = 0; i < (int)needle_size; i++) {
		interim.dist_first = FLT_MAX;
		interim.dist_second = FLT_MAX;

		for (j = 0; j < numcore; j++) {
			dist = interim_h[(j * needle_size) + i].dist_first;
			if (dist < interim.dist_first) {
				interim.lat_first =
					interim_h[(j * needle_size) + i].lat_first;
				interim.lng_first =
					interim_h[(j * needle_size) + i].lng_first;
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

	PROFILE_TO(&tv_from, &tv_to, postprocessing_ms);

	free(needle_essence_h);
	free(interim_h);

	cudaFree(needle_essence_d);
	cudaFree(haystack_d);
	cudaFree(interim_d);

#ifdef PROFILE_CUDA
	PROFILE_TO(&tv_total_from, &tv_total_to, total_ms);
	etc_ms = total_ms
		- copy_needle_ms - copy_haystack_ms
		- run_kernel_ms - copy_result_ms
		- postprocessing_ms;

	printf("[CUDA Profile]\n"
		   "Copy needle to device  : %7lu ms (%5.2f %%)\n"
		   "Copy haystack to device: %7lu ms (%5.2f %%)\n"
		   "Run CUDA kernel        : %7lu ms (%5.2f %%)\n"
		   "Copy result from device: %7lu ms (%5.2f %%)\n"
		   "Post processing        : %7lu ms (%5.2f %%)\n"
		   "etc.                   : %7lu ms (%5.2f %%)\n"
		   "Total                  : %7lu ms\n",
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

