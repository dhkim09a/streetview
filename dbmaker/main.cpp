#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cv.h>
#include "db.h"
#include "surflib.h"

#define SURF_THRESHOLD 0.0001f
#define NAME_CONV "sv_%lf_%lf_%d.jpg"

#define ADD_STR(dst, src)											\
do{ char *temp;														\
	if (dst == NULL)												\
		while((dst = (char*)calloc(									\
						strlen(src) + 1, sizeof(char)))==NULL);		\
	else {															\
		while((temp = (char*)realloc(dst,							\
						(strlen(dst) + strlen(src) + 1)))==NULL);	\
		dst = temp;													\
	}																\
	strcpy(&dst[strlen(dst)], src);									\
} while(0)

int main (int argc, char **argv)
{
	if (argc != 3) {
		printf("usage: %s [input dir] [output file]\n", argv[0]);
		exit(0);
	}

	DIR *input_dir;
	FILE *output_file;
	struct dirent *ent;
	struct stat status;
	char *file_path = NULL;
	double latitude, longitude;
	int headling;
	int i, j, cnt = 0;
	float avg_vec = 0;

	IplImage *img;
	IpVec ipts;
	ipoint_t fipt;

	if (!(input_dir = opendir(argv[1]))) {
		fprintf(stderr, "No such directory, %s\n", argv[1]);
		exit(0);
	}

	if (!(output_file = fopen(argv[2], "wb"))) {
		fprintf(stderr, "Cannot open file, %s\n", argv[2]);
		exit(0);
	}

	while ((ent = readdir(input_dir)) != NULL) {
		ADD_STR(file_path, argv[1]);
		ADD_STR(file_path, "/");
		ADD_STR(file_path, ent->d_name);

		if (stat(file_path, &status)) {
			fprintf(stderr, "Cannot read file stat, %s\n", file_path);
			goto skip_file;
		}

		if (!S_ISREG(status.st_mode)) {
			fprintf(stderr, "Skipping non-regular file, %s\n", file_path);
			goto skip_file;
		}

		if (sscanf(ent->d_name, NAME_CONV,
					&latitude, &longitude, &headling) != 3) {
			fprintf(stderr, "Failed to get location from file name, %s\n",
					ent->d_name);
			goto skip_file;
		}

		/* SURF files */

		if (!(img = cvLoadImage(file_path))) {
			fprintf(stderr, "Failed to load image, %s\n", file_path);
			goto skip_file;
		}

		surfDetDes(img, ipts, false, 3, 4, 3, SURF_THRESHOLD);

		cvReleaseImage(&img);

		fipt.latitude = latitude;
		fipt.longitude = longitude;
		for (i = 0; i < (int)ipts.size(); i++) {
			for (j = 0; j < VEC_DIM; j++)
				fipt.vec[j] = ipts[i].descriptor[j];

			fwrite(&fipt, sizeof(fipt), 1, output_file);
		}

		avg_vec += (float)ipts.size();
		cnt++;
		printf("\r%10d files processed, %.2f vectors per a file",
				cnt, avg_vec / (float)cnt);
		fflush(stdout);

skip_file:
		free(file_path);
		file_path = NULL;
	}

	printf("\n");
	fflush(stdout);

	fclose(output_file);
	closedir(input_dir);

	return 0;
}
