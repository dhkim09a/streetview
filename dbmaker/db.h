#ifndef __DB_H_
#define __DB_H_

#define VEC_DIM 64
#define PRECISION 32 /* Floating Point Format (FPF) precision */

#if PRECISION == 64
#define FPF double
#define FPF_T "%lf"
#elif PRECISION == 32
#define FPF float
#define FPF_T "%f"
#else
#error Define precision as 32 or 64
#endif

typedef struct _ipoint_t {
	FPF latitude __attribute__((aligned (4)));
	FPF longitude __attribute__((aligned (4)));
	float vec[VEC_DIM] __attribute__((aligned (4)));
} __attribute__((packed)) ipoint_t;

#endif /* __DB_H_ */
