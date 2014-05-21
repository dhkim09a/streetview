#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>
#include <math.h>
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

#ifndef NUMCPU
#error Define NUMCPU as a positive value!
#endif

#ifndef search
#error Define search()!
#endif

#ifndef RESIZE
#error Define RESIZE!
#endif

#ifdef GEO_CORRECTION
#ifndef LAT_MPD
#error Define LAT_MPD!
#endif
#ifndef LNG_MPD
#error Define LNG_MPD!
#endif
#endif

#define NEIGHBOR_DIST 15

extern int search (IpVec needle, ipoint_t *haystack, int haystack_size,
		struct _interim *result, int result_size, int numcpu);

#ifdef GEO_CORRECTION
int isClose(FPF lat1, FPF lng1, FPF lat2, FPF lng2)
{
	if (((LAT_MPD * (lat1 - lat2)) * (LAT_MPD * (lat1 - lat2)))
			+ ((LNG_MPD * (lng1 - lng2)) * (LNG_MPD * (lng1 - lng2)))
			< NEIGHBOR_DIST * NEIGHBOR_DIST)
		return 1;
	else
		return 0;
}
#endif

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
	struct _interim *result;
	result_t answer;
	ResVec answer_vec;

	IplImage *input_img;
	IpVec input_ipts;

	int i, j, found;
	float dist_ratio;

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
	if (input_img->width > RESIZE || input_img->height > RESIZE) {
		IplImage *resized_img;
		float ratio = 1;
		ratio = MIN(RESIZE / (float)input_img->width,
				RESIZE / (float)input_img->height);
		resized_img = cvCreateImage(cvSize(ratio * input_img->width,
					ratio * input_img->height),
				input_img->depth, input_img->nChannels);
		cvResize(input_img, resized_img, CV_INTER_LINEAR);

		cvReleaseImage(&input_img);
		input_img = resized_img;
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

	result = (struct _interim *)malloc(
			input_ipts.size() * sizeof(struct _interim));
	for (i = 0; i < (int)input_ipts.size(); i++) {
		result[i].dist_first = FLT_MAX;
		result[i].dist_second = FLT_MAX;
		result[i].lat_first = 0;
		result[i].lng_first = 0;
	}

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
				!= (long long)haystack_mem_size) {
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
		search(input_ipts, haystack, haystack_size, result, input_ipts.size(),
				NUMCPU);
#ifdef PROFILE
		gettimeofday(&tv_to, NULL);
		vec_match_ms += (tv_to.tv_sec - tv_from.tv_sec) * 1000
			+ (tv_to.tv_usec - tv_from.tv_usec) / 1000;
#endif
	}

	close(db);

	for (i = 0; i < (int)input_ipts.size(); i++) {
		dist_ratio = result[i].dist_first / result[i].dist_second;
		if (dist_ratio < MATCH_THRESH_SQUARE){
			found = -1;
			for (j = 0; j < (int)answer_vec.size(); j++) {
				if (answer_vec[j].latitude == result[i].lat_first
						&& answer_vec[j].longitude == result[i].lng_first) {
					answer_vec[j].score += 1 - dist_ratio;
					found = 1;
					break;
				}
			}

			if (found < 0) {
				answer.latitude = result[i].lat_first;
				answer.longitude = result[i].lng_first;
				answer.score = 1 - dist_ratio;

				answer_vec.push_back(answer);
			}
		}
	}
#ifdef GEO_CORRECTION
	float *correction = (float *)malloc(answer_vec.size() * sizeof(float));
	memset(correction, 0, answer_vec.size() * sizeof(float));
	for (i = 0; i < (int)answer_vec.size(); i++) {
		for (j = 0; j < (int)answer_vec.size(); j++) {
			if (i == j)
				continue;
			if (isClose(answer_vec[i].latitude, answer_vec[i].longitude,
						answer_vec[j].latitude, answer_vec[j].longitude))
				correction[i] += answer_vec[j].score;
		}
	}
	for (i = 0; i < (int)answer_vec.size(); i++)
		answer_vec[i].score += correction[i];
#endif
#if 0
	/* Normalize the score */
	float mean = 0, square_mean = 0, sigma = 0;
	for (i = 0; i < (int)answer_vec.size(); i++) {
		mean += answer_vec[i].score;
		square_mean += answer_vec[i].score * answer_vec[i].score;
	}
	mean /= answer_vec.size();
	square_mean /= answer_vec.size();
	sigma = sqrtf(square_mean - (mean * mean));
	for (i = 0; i < (int)answer_vec.size(); i++)
		answer_vec[i].score = (answer_vec[i].score - mean) / sigma;
#else
	float sum = 0;
	for (i = 0; i < (int)answer_vec.size(); i++) {
		sum += answer_vec[i].score;
	}
	for (i = 0; i < (int)answer_vec.size(); i++)
		answer_vec[i].score *= 100 / sum;
#endif
	std::sort(answer_vec.begin(), answer_vec.end(), comp_result);

	float max_gap = 0, gap;
	int cutline = 0;
	for (i = 1; i < (int)answer_vec.size(); i++) {
		gap = answer_vec[i-1].score - answer_vec[i].score;
		if (gap > max_gap) {
			max_gap = gap;
			cutline = i;
		}
	}

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
	printf("[Result]\n"
		   "latitude   longitude  score\n");
	for (i = 0; i < MIN(cutline, answer_vec.size()); i++)
		printf(FPF_T" "FPF_T" %.3f\n",
			answer_vec[i].latitude, answer_vec[i].longitude,
			answer_vec[i].score);

	return 0;
}
