#ifndef __SEARCH_H_
#define __SEARCH_H_

#include "surflib.h"
#include "db.h"

#define MATCH_THRESH 1.0 /* 0.0 ~ 1.0 */
#define MATCH_THRESH_SQUARE (MATCH_THRESH * MATCH_THRESH)

typedef struct _result_t {
	FPF latitude;
	FPF longitude;
	int occurence;
} result_t;

typedef std::vector<result_t> ResVec;

static bool comp_result (result_t i, result_t j)
{
	return i.occurence > j.occurence;
}

struct _interim {
	FPF lat_first __attribute__((aligned (4)));
	FPF lng_first __attribute__((aligned (4)));
	float dist_first __attribute__((aligned (4)));
	float dist_second __attribute__((aligned (4)));
} __attribute__((packed));

int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int numcpu);

#endif /* __SEARCH_H_ */
