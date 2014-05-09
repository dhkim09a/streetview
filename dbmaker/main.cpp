#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cv.h>
#include "db.h"
#include "surflib.h"

#define NUMCPU 8
#define SURF_THRESHOLD 0.0001f
#define NAME_CONV "sv_"FPF_T"_"FPF_T"_%d.jpg"
#define PATH_LEN 200

#define ADD_STR(dst, src)											\
	strcpy(&dst[strlen(dst)], src)

typedef struct _arg_t {
	char *input_dir_path;
	DIR *input_dir;
	FILE *output_file;
	int *filecnt;
	float *avgvec;
	pthread_mutex_t *mx_input_dir;
	pthread_mutex_t *mx_output_file;

	int *running;
	pthread_mutex_t *mx_running;
	pthread_cond_t *wakeup_master;
} arg_t;

int doWork(char *input_dir_path, DIR *input_dir, FILE *output_file,
		int *filecnt, float *avgvec,
		pthread_mutex_t *mx_input_dir, pthread_mutex_t *mx_output_file)
{
	struct dirent *ent;
	struct stat status;
	char file_path[PATH_LEN];
	FPF latitude, longitude;
	int headling;
	int i, j;

	IplImage *img;
	IpVec ipts;
	ipoint_t fipt;

	while (1) {
		pthread_mutex_lock(mx_input_dir);
		ent = readdir(input_dir);
		pthread_mutex_unlock(mx_input_dir);

		if (ent == NULL)
			break;

		ADD_STR(file_path, input_dir_path);
		ADD_STR(file_path, "/");
		ADD_STR(file_path, ent->d_name);

		if (stat(file_path, &status)) {
			fprintf(stderr, "\33[2K\rCannot read file stat, %s\n", file_path);
			goto skip_file;
		}

		if (!S_ISREG(status.st_mode)) {
			fprintf(stderr,
					"\33[2K\rSkipping non-regular file, %s\n", file_path);
			goto skip_file;
		}

		if (sscanf(ent->d_name, NAME_CONV,
					&latitude, &longitude, &headling) != 3) {
			fprintf(stderr,
					"\33[2K\rFailed to get location from file name, %s\n",
					ent->d_name);
			goto skip_file;
		}

		/* SURF files */

		if (!(img = cvLoadImage(file_path))) {
			fprintf(stderr, "\33[2K\rFailed to load image, %s\n", file_path);
			goto skip_file;
		}

		surfDetDes(img, ipts, false, 3, 4, 3, SURF_THRESHOLD);

		cvReleaseImage(&img);

		pthread_mutex_lock(mx_output_file);
		fipt.latitude = latitude;
		fipt.longitude = longitude;
		for (i = 0; i < (int)ipts.size(); i++) {
			for (j = 0; j < VEC_DIM; j++)
				fipt.vec[j] = ipts[i].descriptor[j];

			fwrite(&fipt, sizeof(fipt), 1, output_file);
		}
		(*filecnt)++;
		(*avgvec) += (float)ipts.size();
		printf("\33[2K\r%d files processed, %.2f vectors per a file",
				(*filecnt), (*avgvec) / (float)(*filecnt));
		fflush(stdout);
		pthread_mutex_unlock(mx_output_file);

skip_file:
		memset(file_path, 0, PATH_LEN);
	}

	return 0;
}

void *thread_main(void *arg)
{
	arg_t *arg1 = (arg_t *)arg;

	doWork(arg1->input_dir_path, arg1->input_dir, arg1->output_file,
			arg1->filecnt, arg1->avgvec,
			arg1->mx_input_dir, arg1->mx_output_file);

	pthread_mutex_lock(arg1->mx_running);
	if (--(*arg1->running) == 0)
		pthread_cond_signal(arg1->wakeup_master);
	pthread_mutex_unlock(arg1->mx_running);

	return NULL;
}

int main (int argc, char **argv)
{
	if (argc != 3) {
		printf("usage: %s [input dir] [output file]\n", argv[0]);
		exit(0);
	}

	DIR *input_dir;
	FILE *output_file;
	int filecnt = 0;
	float avgvec = 0;
	pthread_mutex_t mx_input_dir;
	pthread_mutex_t mx_output_file;

	int i;
	int err, status;

	if (!(input_dir = opendir(argv[1]))) {
		fprintf(stderr, "No such directory, %s\n", argv[1]);
		exit(0);
	}

	if (!(output_file = fopen(argv[2], "wb"))) {
		fprintf(stderr, "Cannot open file, %s\n", argv[2]);
		exit(0);
	}

	pthread_t *threads = (pthread_t *)malloc(NUMCPU * sizeof(pthread_t));
	arg_t *args = (arg_t *)malloc(NUMCPU * sizeof(arg_t));

	pthread_mutex_init(&mx_input_dir, NULL);
	pthread_mutex_init(&mx_output_file, NULL);

	int running = NUMCPU;
	pthread_mutex_t mx_running;
	pthread_cond_t wakeup_master;

	pthread_mutex_init(&mx_running, NULL);
	pthread_cond_init(&wakeup_master, NULL);

	pthread_mutex_lock(&mx_running);
	for (i = 0; i < NUMCPU; i++) {
		args[i].input_dir_path = argv[1];
		args[i].input_dir = input_dir;
		args[i].output_file = output_file;
		args[i].filecnt = &filecnt;
		args[i].avgvec = &avgvec;
		args[i].mx_input_dir = &mx_input_dir;
		args[i].mx_output_file = &mx_output_file;

		args[i].running = &running;
		args[i].mx_running = &mx_running;
		args[i].wakeup_master = &wakeup_master;

		pthread_create(&(threads[i]), NULL, &thread_main, (void *)(&args[i]));
	}

	pthread_cond_wait(&wakeup_master, &mx_running);
	pthread_mutex_unlock(&mx_running);

	for (i = 0; i < NUMCPU; i++) {
		if (err = pthread_join(threads[i], (void **)&status)) {
			fprintf(stderr, "pthread_join(%d) failed (returned %d)\n",
					i, err);
			fflush(stderr);
		}
	}

	printf("\n");
	fflush(stdout);

	free(args);
	free(threads);
	fclose(output_file);
	closedir(input_dir);

	return 0;
}
