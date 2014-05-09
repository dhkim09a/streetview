#ifndef __SEARCH_H_
#define __SEARCH_H_

#include "surflib.h"

#define MATCH_THRESH 0.80 /* 0.0 ~ 1.0 */
#define MATCH_THRESH_SQUARE (MATCH_THRESH * MATCH_THRESH)

#ifdef CUDA
#define search(arg...) GPUsearch(arg)
#else
#ifndef NUMCPU
#error Define NUMCPU as a positive value!
#else
#define search(arg...) CPUsearch(arg, NUMCPU)
#endif /* NUMCPU */
#endif /* CUDA */

typedef struct _result_t {
	double latitude;
	double longitude;
	int occurence;
} result_t;

typedef std::vector<result_t> ResVec;

static bool comp_result (result_t i, result_t j)
{
	return i.occurence > j.occurence;
}

#ifdef CUDA
int GPUsearch (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec *result);
#else
int CPUsearch (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec *result, int numcpu);
#endif

#endif /* __SEARCH_H_ */
