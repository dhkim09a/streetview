#ifndef __SEARCH_H_
#define __SEARCH_H_

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
	float latitude;
	float longitude;
	int occurence;
} result_t;

typedef std::vector<result_t> ResVec;

int GPUsearch (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec result);

int CPUsearch (IpVec needle, ipoint_t *haystack, int haystack_size,
		ResVec result, int numcpu);

#endif /* __SEARCH_H_ */
