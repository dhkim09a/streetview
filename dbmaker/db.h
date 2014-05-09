#ifndef __DB_H_
#define __DB_H_

#define VEC_DIM 64

typedef struct _ipoint_t {
	double latitude;
	double longitude;
	float vec[VEC_DIM];
} __attribute__((packed)) ipoint_t;

#endif /* __DB_H_ */
