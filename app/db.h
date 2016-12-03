#ifndef __DB_H_
#define __DB_H_

#define VEC_DIM 2048

#ifndef TAGLEN
#error Define TAGLEN!
#endif

union _tag {
	char str[TAGLEN];
	struct {
		float longitude;
		float latitude;
	} geo;
} __attribute__((packed));

typedef struct _ipoint_t {
	union _tag tag __attribute__((aligned (4)));
	float vec[VEC_DIM] __attribute__((aligned (4)));
} __attribute__((packed)) ipoint_t;

//typedef struct 

#endif /* __DB_H_ */
