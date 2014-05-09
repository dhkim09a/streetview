#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "db.h"
#include "search.h"

typedef struct _ipoint_essence_t {
	float vec[VEC_DIM] __attribute__((aligned (4)));
} __attribute__((packed)) ipoint_essence_t;

__global__ void doSearchKernel (ipoint_essence_t *needle, int needle_size,
		ipoint_t *haystack, int haystack_size,
		struct _interim *interim, int interim_size_local)
{
	int id = (blockDim.x * blockIdx.x) + threadIdx.x;
	int numcore = blockDim.x * gridDim.x;

	if (id >= numcore)
		return;

	ipoint_t *haystack_local = &(haystack[haystack_size / numcore * id]);
	int haystack_size_local = MIN( haystack_size / numcore,
			haystack_size - ((haystack_size / numcore) * id));
	struct _interim *interim_local = &(interim[interim_size_local * id]);

	float dist, temp;
	int i, j, k;
	int iter;

	for (i = 0; i < interim_size_local; i++) {
		interim_local[i].dist_first = FLT_MAX;
		interim_local[i].dist_second = FLT_MAX;
	}
	iter = MIN(interim_size_local, needle_size);
	for (i = 0; i < iter; i++) {
		for (j = 0; j < haystack_size_local; j++) {
			dist = 0;
			for (k = 0; k < VEC_DIM; k++) {
				temp = needle[i].vec[k] - haystack_local[j].vec[k];
				dist += temp * temp;
			}
			if (dist < interim_local[i].dist_first) {
				interim_local[i].lat_first = haystack_local[j].latitude;
				interim_local[i].lng_first = haystack_local[j].longitude;
				interim_local[i].dist_second = interim_local[i].dist_first;
				interim_local[i].dist_first = dist;
			}
			else if (dist < interim_local[i].dist_second)
				interim_local[i].dist_second = dist;
		}
	}
	return;
}

int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec *result_vec, int dummy)
{
	/* FIXME: We have to handle malloc() and cudaMalloc() failure */
	unsigned int block_dim = 256;
	unsigned int grid_dim = 128;
	int numcore = block_dim * grid_dim;

	int i, j, found;
	struct _interim interim;
	result_t result;
	float dist;

	ipoint_essence_t *needle_essence_h, *needle_essence_d;
	ipoint_t *haystack_d;
	struct _interim *interim_h, *interim_d;
	int needle_size = needle.size();
	cudaError_t err;

	/* Copy needle to device */
	needle_essence_h = (ipoint_essence_t *)malloc(
			needle_size * sizeof(ipoint_essence_t));
	for (i = 0; i < needle_size; i++)
		for (j = 0; j < VEC_DIM; j++)
			needle_essence_h[i].vec[j] = needle[i].descriptor[j];
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

	/* Run CUDA kernel */
	doSearchKernel <<< grid_dim, block_dim >>> (needle_essence_d, needle_size,
			haystack_d, haystack_size, interim_d, needle_size);

	/* Copy result to host */
	err = cudaMemcpy(interim_h, interim_d,
			numcore * sizeof(struct _interim) * needle_size,
			cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(interim_h, interim_d): %s\n",
				cudaGetErrorString(err));
		return -1;
	}

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

	free(needle_essence_h);
	free(interim_h);

	cudaFree(needle_essence_d);
	cudaFree(haystack_d);
	cudaFree(interim_d);

	std::sort((*result_vec).begin(), (*result_vec).end(), comp_result);

	return 0;
}

