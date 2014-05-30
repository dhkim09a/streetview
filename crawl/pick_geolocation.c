#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cv.h>
#include <highgui.h>

#include "geolo_config.h"

#define CHANNEL_B    0
#define CHANNEL_G    1
#define CHANNEL_R    2
#define CHANNEL_GREY 0

#define PIXEL(img, widthStep, nChannel, depth, row, col, channel) \
	((uint##depth##_t *)((size_t)((size_t)img + widthStep * row \
			+ ((nChannel) * (col) + (CHANNEL_##channel)))))

#define LATITUDE(height, row) \
	((double)(row) / (double)(height) * (LAT_D - LAT_U) + LAT_U)
#define LONGITUDE(width, col) \
	((double)(col) / (double)(width) * (LNG_R - LNG_L) + LNG_L)

int main (int argc, char **argv)
{
	IplImage *img;
	FILE *output;
	int i, j, cnt = 0;

	if (argc != 3) {
		printf("usage: %s [input image] [output txt]\n", argv[0]);
		return 0;
	}

	if (!(img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR))) {
		fprintf(stderr, "Failed to load image, %s\n", argv[1]);
		return 0;
	}

	if ((output = fopen(argv[2], "w+")) == NULL) {
		fprintf(stderr, "Failed to open %s\n", argv[2]);
		return 0;
	}

	for (i = 0; i < img->height; i++)
		for (j = 0; j < img->width; j++) {
			if (*PIXEL(img->imageData, img->widthStep, img->nChannels, 8,
						i, j, R) != 255 && i % 3 == 0 && j % 3 == 0) {
				cnt++;
				fprintf(output, "%lf %lf\n",
						LATITUDE(img->height, i), LONGITUDE(img->width, j));
			}
			else {
				*PIXEL(img->imageData, img->widthStep, img->nChannels, 8,
						i, j, G) = 255;
				*PIXEL(img->imageData, img->widthStep, img->nChannels, 8,
						i, j, B) = 255;
				*PIXEL(img->imageData, img->widthStep, img->nChannels, 8,
						i, j, R) = 255;
			}

		}

	fclose(output);

	printf("%d / %d\n", cnt, img->width * img->height);

	cvNamedWindow("Test", CV_WINDOW_AUTOSIZE);
	cvShowImage("Test", img);
	cvWaitKey(0);

	return 0;
}
