#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
#include "db_loader.h"
#include "net.h"
#include "message.h"

#ifdef PROFILE
#define PROFILE_ON
#endif
#include "profile.h"

#define TOP 10
#define MAX_IMG_PATH_LEN 200

#ifndef RESIZE
#error Define RESIZE!
#endif

typedef struct _cb_arg_t {
	pthread_mutex_t mx_block;
	pthread_cond_t cd_block;
} cb_arg_t;

void callback (msg_t *msg)
{
	req_t *request = (req_t *)msg->content;
	cb_arg_t *arg = (cb_arg_t *)msg->arg;

	printf("latitude   longitude  score\n");
	printf(FPF_T" "FPF_T" %.3f\n",
			request->latitude, request->longitude, request->score);
	fflush(stdout);

	cvReleaseImage(&request->img);

	pthread_cond_signal(&arg->cd_block);
}

int main (int argc, char **argv)
{
	if (argc != 2 && argc != 3) {
		printf("usage: %s [database file]\n", argv[0]);
		printf("usage: %s [database file] [input image file]\n", argv[0]);
		exit(0);
	}

	int db_fd;
	struct stat status;
	
	int dummy;

	IplImage *input_img;
	char img_path[MAX_IMG_PATH_LEN] = {0};

	pthread_t db_loader_thread, search_thread, net_thread;
	db_t db;
	search_t sc;

	req_t request;
	cb_arg_t cb_arg;
	pthread_mutex_init(&cb_arg.mx_block, NULL);
	pthread_cond_init(&cb_arg.cd_block, NULL);

	if ((db_fd = open(argv[1], O_RDONLY)) < 0) {
		fprintf(stderr, "Cannot open file, %s\n", argv[1]);
		exit(0);
	}

	if (fstat(db_fd, &status)) {
		fprintf(stderr, "Cannot read file stat, %s\n", argv[1]);
		close(db_fd);
		exit(0);
	}

	if (status.st_size % sizeof(ipoint_t)) {
		fprintf(stderr, 
				"Database file might be corrupted (file_size %% %lu != 0)\n",
				sizeof(ipoint_t));
		close(db_fd);
		exit(0);
	}

	db_init(&db, db_fd, status.st_size, sizeof(ipoint_t));
	pthread_create(&db_loader_thread, NULL, &db_loader_main, (void*)(&db));

	sc_init(&sc, &db);
	pthread_create(&search_thread, NULL, &sc_main, (void*)(&sc));

	pthread_create(&net_thread, NULL, &net_main, (void *)(&sc));

	while (1) {
		if (argc == 3) {
			strncpy(img_path, argv[2], MAX_IMG_PATH_LEN);
		}
		else {
			printf("> ");
			fflush(stdout);

			if (!fgets(img_path, MAX_IMG_PATH_LEN, stdin))
				continue;
			if (img_path[0] == '\0')
				continue;

			// remove the trailing '\n'
			int pathlen = strlen(img_path);
			img_path[pathlen - 1] = '\0';
		}

		int i;
		for (i = 0; i < 10; i++) {

		/* load input image */
		if (!(input_img = cvLoadImage(img_path))) {
			fprintf(stderr, "Failed to load image, %s\n", img_path);
			continue;
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

		request.img = input_img;
		if (msg_write(&sc.msgbox,
					(void *)&request, callback, (void *)&cb_arg)) {
			printf("Request failed: %s\n", img_path);
			cvReleaseImage(&input_img);
		}

		pthread_mutex_lock(&cb_arg.mx_block);
		pthread_cond_wait(&cb_arg.cd_block, &cb_arg.mx_block);
		pthread_mutex_unlock(&cb_arg.mx_block);

		}

		if (argc == 3)
			break;
	}

	db_kill(&db);
	pthread_join(db_loader_thread, (void **)&dummy);
	close(db_fd);

	return 0;
}
