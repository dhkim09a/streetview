#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>
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
/*
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
*/
int main (int argc, char **argv)
{
#ifdef PROFILE
	struct timeval tv_from, tv_to;
	struct timeval tv_total_from, tv_total_to;
	unsigned long surf_input_ms = 0, load_db_ms = 0, vec_match_ms = 0;
	unsigned long etc_ms = 0, total_ms = 0;

	gettimeofday(&tv_total_from, NULL);
#endif

	if (argc != 3) {
		printf("usage: %s [input image] [database file]\n", argv[0]);
		exit(0);
	}

	int db;
	struct stat status;
	ipoint_t *haystack = NULL;
	int haystack_size; /* number of entries in haystack */
	size_t haystack_mem_size;
	size_t db_left;
	ResVec result;

	IplImage *input_img;
	IpVec input_ipts;

	int i;

	if ((db = open(argv[2], O_RDONLY)) < 0) {
		fprintf(stderr, "Cannot open file, %s\n", argv[2]);
		exit(0);
	}

	if (fstat(db, &status)) {
		fprintf(stderr, "Cannot read file stat, %s\n", argv[2]);
		close(db);
		exit(0);
	}

	if (status.st_size % sizeof(ipoint_t)) {
		fprintf(stderr, 
				"Database file might be corrupted (file_size %% %lu != 0)\n",
				sizeof(ipoint_t));
		close(db);
		exit(0);
	}

	/* SURF input image */
	if (!(input_img = cvLoadImage(argv[1]))) {
		fprintf(stderr, "Failed to load image, %s\n", argv[1]);
		close(db);
		exit(0);
	}
#ifdef PROFILE
	gettimeofday(&tv_from, NULL);
#endif
	surfDetDes(input_img, input_ipts, false, 3, 4, 3, SURF_THRESHOLD);
#ifdef PROFILE
	gettimeofday(&tv_to, NULL);
	surf_input_ms = (tv_to.tv_sec - tv_from.tv_sec) * 1000
		+ (tv_to.tv_usec - tv_from.tv_usec) / 1000;
#endif

	cvReleaseImage(&input_img);

	printf("Extracted %lu interesting points from input image\n",
			input_ipts.size());

	/* Read DB and do searching */
	db_left = status.st_size;

	while (1) {
		haystack_size = MIN(MEM_LIMIT, db_left) / sizeof(ipoint_t);
		haystack_mem_size = haystack_size * sizeof(ipoint_t);

		if (haystack_size <= 0)
			break;

#ifdef PROFILE
		gettimeofday(&tv_from, NULL);
#endif
		if (haystack == NULL)
			haystack = (ipoint_t *)malloc(haystack_mem_size);

		if (read(db, haystack, haystack_mem_size)
				!= haystack_mem_size) {
			fprintf(stderr, "Failed to read database file\n");
			close(db);
			exit(0);
		}
		else {
			db_left -= haystack_mem_size;
			printf("Read %lu / %lu bytes from DB\n",
					status.st_size - db_left, status.st_size);
		}
#ifdef PROFILE
	gettimeofday(&tv_to, NULL);
	load_db_ms += (tv_to.tv_sec - tv_from.tv_sec) * 1000
		+ (tv_to.tv_usec - tv_from.tv_usec) / 1000;
#endif

		printf("Finding %lu needles from haystack of %d\n",
				input_ipts.size(), haystack_size);
#ifdef PROFILE
		gettimeofday(&tv_from, NULL);
#endif
		search(input_ipts, haystack, haystack_size, &result);
#ifdef PROFILE
	gettimeofday(&tv_to, NULL);
	vec_match_ms += (tv_to.tv_sec - tv_from.tv_sec) * 1000
		+ (tv_to.tv_usec - tv_from.tv_usec) / 1000;
#endif
	}

	close(db);

#ifdef PROFILE
	gettimeofday(&tv_total_to, NULL);
	total_ms += (tv_total_to.tv_sec - tv_total_from.tv_sec) * 1000
		+ (tv_total_to.tv_usec - tv_total_from.tv_usec) / 1000;
	etc_ms = total_ms - surf_input_ms - load_db_ms - vec_match_ms;

	printf("[Profile]\n"
		   "SURF input image: %7lu ms (%5.2f %%)\n"
		   "Load database   : %7lu ms (%5.2f %%)\n"
		   "Vector searching: %7lu ms (%5.2f %%)\n"
		   "etc.            : %7lu ms (%5.2f %%)\n"
		   "Total           : %7lu ms\n",
		   surf_input_ms, 100 * (float)surf_input_ms / (float)total_ms,
		   load_db_ms, 100 * (float)load_db_ms / (float)total_ms,
		   vec_match_ms, 100 * (float)vec_match_ms / (float)total_ms,
		   etc_ms, 100 * (float)etc_ms / (float)total_ms,
		   total_ms);
#endif
	printf("[Result] latitude longitude score\n");
	for (i = 0; i < MIN(TOP, result.size()); i++)
		printf(FPF_T" "FPF_T" %d\n",
			result[i].latitude, result[i].longitude, result[i].occurence);

	return 0;
}
