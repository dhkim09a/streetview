
#define VEC_DIM 64

typedef struct _ipoint_t {
	float latitude;
	float longitude;
	float vec[VEC_DIM];
} __attribute__((packed)) ipoint_t;

