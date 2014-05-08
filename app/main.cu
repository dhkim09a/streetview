#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cv.h>
#include "surflib.h"

#include "db.h"
#include "search.h"

#define TOP 10
#define SURF_THRESHOLD 0.0001f

#ifndef DB_LIMIT
#error Define DB_LIMIT!
#endif

#ifndef MEM_LIMIT
#error Define MEM_LIMIT!
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int main (int argc, char **argv)
{
	if (argc != 3) {
		printf("usage: %s [input image] [database file]\n", argv[0]);
		exit(0);
	}

	int *db;
	struct stat status;
	ipoint_t *haystack = NULL;
	int haystack_size; /* number of entries in haystack */
	size_t haystack_mem_size;
	size_t db_left;
	ResVec result;

	IplImage *input_img;
	IpVec input_ipts;

	int i;

	if ((db = open(argv[2], "rb")) < 0) {
		fprintf(stderr, "Cannot open file, %s\n", argv[2]);
		exit(0);
	}

	if (fstat(db, &status)) {
		fprintf(stderr, "Cannot read file stat, %s\n", argv[2]);
		close(db);
		exit(0);
	}

	if (status.st_size % sizeof(ipoint_t)) {
		fprintf(stderr, "Database file might be corrupted\n");
		close(db);
		exit(0);
	}

	/* SURF input image */
	if (!(input_img = cvLoadImage(argv[1]))) {
		fprintf(stderr, "Failed to load image, %s\n", file_path);
		close(db);
		exit(0);
	}

	surfDetDes(input_img, input_ipts, false, 3, 4, 3, SURF_THRESHOLD);

	cvReleaseImage(&input_img);

	/* Read DB and do searching */
	db_left = status.st_size;

	while (1) {
		haystack_size = MIN(MEM_LIMIT, db_left) / sizeof(ipoint_t);
		haystack_mem_size = haystack_size * sizeof(ipoint_t);

		if (haystack_size <= 0)
			break;

		if (haystack == NULL)
			haystack = (ipoint_t *)malloc(haystack_mem_size);

		if (read(fd, haystack, haystack_mem_size)
				!= haystack_mem_size) {
			fprintf(stderr, "Failed to read database file\n");
			close(db);
			exit(0);
		}
		else
			db_left -= haystack_mem_size;

		search(input_ipts, haystack, haystack_size, result);
	}

	close(db);

	printf("Result: [latitude] [longitude] [score]\n"
	for (i = 0; i < MIN(TOP, result.size()); i++)
		printf("%f %f %d\n",
			result[i].latitude, result[i].longitude, result[i].occurence);

	return 0;
}
