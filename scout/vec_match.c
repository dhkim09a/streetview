#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <float.h>


#define DIMENSION 128
#define PRECISION 64


#define BOUND 100

#if PRECISION == 64
#define FLOAT double
#define sqrt(arg...) sqrt(arg)
#elif PRECISION == 32
#define FLOAT float
#define sqrt(arg...) sqrtf(arg)
#else
#error Define precision as 32 or 64
#endif

typedef struct ipoint {
	FLOAT vec[DIMENSION];
}__attribute__((packed)) ipoint_t;

ipoint_t zerovec;

inline FLOAT dist(ipoint_t *vecA, ipoint_t *vecB)
{
	int i;
	FLOAT ret = 0;
	FLOAT temp;

	for (i = 0; i < DIMENSION; i++) {
		temp = (vecA->vec)[i] - (vecB->vec)[i];
		ret += temp * temp;
	}

//	return sqrt(ret);
	return ret;
}

#define ampl(vec) dist((vec), &zerovec)

void norm(ipoint_t *vec)
{
	FLOAT amp = ampl(vec);
	int i;

	for (i = 0; i < DIMENSION; i++)
		vec->vec[i] /= amp;
}

void init (void)
{
	int i;
	for (i = 0; i < DIMENSION; i++)
		zerovec.vec[i] = 0;
}

void init_vec (ipoint_t *vec, int num)
{
	int i, j;
	int random;

	for (i = 0; i < num; i++) {
		for (j = 0; j < DIMENSION; j++) {
			random = rand();
			memcpy(&(vec[i].vec[j]), &random, sizeof(int));
		}
	}
}

int main (int argc, char **argv)
{
	init();

	volatile ipoint_t *dummyA;
	volatile ipoint_t *dummyB;

	struct timeval tv_start, tv_end;

	if (argc < 3)
		return 0;

	int numA = atoi(argv[1]);
	int numB = atoi(argv[2]);

	long time_ms;

	int i, j, bestA, bestB;
	FLOAT minDist = FLT_MAX, newDist;

	printf("vector matching %d x %d\n"
		   "(%lu bytes per a vector)\n",
		   numA, numB, sizeof(ipoint_t));

	ipoint_t *vecA = malloc(sizeof(ipoint_t) * numA);
	ipoint_t *vecB = malloc(sizeof(ipoint_t) * numB);
	
	init_vec(vecA, numA);
	init_vec(vecB, numB);

	printf("start matching...\n");

	gettimeofday(&tv_start, NULL);
	for (i = 0; i < numA; i++) {
		for (j = 0; j < numB; j++) {
			newDist = dist(&vecA[i], &vecB[j]);
			if (newDist < minDist) {
				bestA = i;
				bestB = j;
				minDist = newDist;
			}
		}
		if (minDist < 0.65) {
			dummyA = &vecA[bestA];
			dummyB = &vecB[bestB];
		}
	}
	gettimeofday(&tv_end, NULL);

	time_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1000
		+ (tv_end.tv_usec - tv_start.tv_usec) / 1000;

	printf("result: %ld min %ld sec %ld msec\n",
		   time_ms / (60 * 1000),
		   (time_ms / (1000)) % 60,
		   time_ms % 1000);

	return 0;
}
