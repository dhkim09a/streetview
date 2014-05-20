#ifndef __SEARCH_H_
#define __SEARCH_H_

#include "surflib.h"
#include "db.h"

#ifndef MATCH_THRESH
#error Define MATCH_THRESH!
#endif

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

#endif /* __SEARCH_H_ */
