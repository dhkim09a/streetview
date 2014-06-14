#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <time.h>
#include <errno.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <errno.h> 
#include <string.h> 
#include <netdb.h> 
#include <sys/types.h> 
#include <netinet/in.h> 
#include <sys/stat.h>
#include <cv.h>
#include <highgui.h>
#include <sys/time.h>

#include "db.h"
#include "search.h"
#include "db_loader.h"

#define HIST_DIR "hist"
#define RESP_FILE "resp.html"

#define BUFSIZE            32000
#define ACCEPT_BACKLOG     10

#define MAX_SOCK           1000

#define HTTP_HDR_A "HTTP/1.0 200 OK\r\nConnection: close\r\nContent-Type: text/html; charset=UTF-8\r\nContent-Length: "
#define HTTP_HDR_B "\r\n\r\n"

#define CLOSE(fd) \
{int dummy; while(read(fd, &dummy, sizeof(dummy)) > 0); close(fd);}

typedef struct _cb_arg_t {
	int sock;
	fd_set *wfds;
	int pipe_wkup;

	float latitude;
	float longitude;
	float score;
} cb_arg_t;

int SprintHTTPHdr(char *buffer, int content_len)
{
	return sprintf(buffer, "%s%d%s", HTTP_HDR_A, content_len, HTTP_HDR_B);
}

int ParseHTTPPOST(char *buffer, int buf_len,
		char **img_buffer, int *img_buf_len,
		char **img_name, int *img_name_len)
{
	char *ptr, *ptr_img, *ptr_img_end;
	int r_cnt = 2;
	int content_length = 0;

	if (strncmp(buffer, "POST", sizeof("POST") - 1))
		return -2;
	if (!(ptr = strstr(buffer, "Content-Length")))
		return -1;
	sscanf(ptr, "Content-Length: %d", &content_length);
	if (content_length == 0)
		return -1;
	if (!(ptr = strstr(buffer, "\r\n\r\n")))
		return -1;
	ptr += 4;

	if ((int)((size_t)(ptr - buffer) + content_length) > buf_len)
		return -1;

	if (!(*img_name = strstr(ptr, "filename=\"")))
		return -1;
	*img_name += sizeof("filename=\"") - 1;
	for (*img_name_len = 0;
			(*img_name)[*img_name_len] != '\"';
			(*img_name_len)++) {
	}

	if (!(ptr_img = strstr(ptr, "\r\n\r\n")))
		return -1;
	ptr_img += 4;

	*img_buffer = ptr_img;
	*img_buf_len = content_length - (size_t)(ptr_img - ptr);

	ptr_img_end = ptr_img + (*img_buf_len - 1);
	
	for (; ptr_img_end > ptr_img; ptr_img_end--) {
		if (*ptr_img_end == '\r') {
			if (--r_cnt == 0)
				break;
		}
	}

	*img_buf_len = (size_t)(ptr_img_end - ptr_img);

	return 0;
}

int NewListenSocket(void)
{
	int listen_fd;
	struct sockaddr_in my_addr;    /* my address information */
	int sock_opt = 1;

	listen_fd = socket(AF_INET, SOCK_STREAM, 0);

	my_addr.sin_family = AF_INET;         /* host byte order */
	my_addr.sin_port = htons(PORT);     /* short, network byte order */
	my_addr.sin_addr.s_addr = INADDR_ANY; /* auto-fill with my IP */
	bzero(&(my_addr.sin_zero), 8);        /* zero the rest of the struct */

	setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(int));

	if (bind(listen_fd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr))
			== -1) {
		perror("bind");
		exit(1);
	}
	if (listen(listen_fd, ACCEPT_BACKLOG) == -1) {
		perror("listen");
		exit(1);
	}
	if (fcntl(listen_fd, F_SETFL,
				fcntl(listen_fd, F_GETFL, 0) | O_NONBLOCK)) {
		printf("Setting NON-BLOCK failed\n");
		exit(0);
	}

	return listen_fd;
}

int GetImgBufIdx(int sock, int *img_buf_tag)
{
	int i;
	for (i = 0; i < BACKLOG; i++) {
		if (img_buf_tag[i] == sock)
			return i;
		else if (img_buf_tag[i] == 0) {
			img_buf_tag[i] = sock;
			return i;
		}
	}

	return -1;
}

void ReleaseImgBuf(int idx, int *img_buf_tag, int *img_buf_len)
{
	img_buf_tag[idx] = 0;
	img_buf_len[idx] = 0;
}

IplImage *LogAndReadImage(void *buffer, int len, char *name, int name_len)
{
	FILE *fp;
	IplImage *input_img;
	char path[100] = {0};

	strcpy(path, HIST_DIR);
	path[sizeof(HIST_DIR) - 1] = '/';
	strncpy(&path[sizeof(HIST_DIR)], name, name_len);

	fp = fopen(path, "w+");
	fwrite(buffer, 1, len, fp);
	fclose(fp);

	if (!(input_img = cvLoadImage(path))) {
		fprintf(stderr, "Failed to load image, %s\n", path);
		return NULL;
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
#if 0
	cvNamedWindow("Test", CV_WINDOW_AUTOSIZE);
	cvShowImage("Test", input_img);
	cvWaitKey(0);
#endif
	return input_img;
}

static void
callback (req_msg_t *msg, FPF latitude, FPF longitude, float score,
		void *arg_void)
{
	cb_arg_t *arg = (cb_arg_t *)arg_void;

	arg->latitude = latitude;
	arg->longitude = longitude;
	arg->score = score;
#if 0
	printf("latitude   longitude  score\n");
	printf(FPF_T" "FPF_T" %.3f\n", latitude, longitude, score);
	fflush(stdout);
#endif
	cvReleaseImage(&msg->img);

	FD_SET(arg->sock, arg->wfds);
	while(write(arg->pipe_wkup, "dummy", 1) < 0);
}

void *net_main (void *arg)
{
	search_t *sc = (search_t *)arg;

	int listen_fd, new_fd, max_fd;
	struct sockaddr_in their_addr; /* connector's address information */
	int sin_size = sizeof(struct sockaddr_in);
	int do_accept = 0;

	int i, idx, dummy, err;

	char buffer[BUFSIZE] = {0};

	int len;

	char *img_buf[BACKLOG] = {NULL};
	int img_buf_tag[BACKLOG] = {0};
	int img_buf_len[BACKLOG] = {0};
	char *img_ptr;
	int img_len;
	char *img_name_ptr;
	int img_name_len;
	IplImage *img;

	struct stat status;
	int resp_file_fd;
	char *resp_file_buf;
	int resp_file_size;
	char *resp_lat_ptr;
	char *resp_lng_ptr;
	char *resp_score_ptr;

	cb_arg_t cb_arg[MAX_SOCK];

	fd_set rfds, rfds_tmp;
	fd_set wfds, wfds_tmp;
	FD_ZERO(&rfds);
	FD_ZERO(&wfds);

	int pipe_select[2];
	if (pipe(pipe_select) == -1) {
		fprintf(stderr, "Failed to create pipe\n");
		exit(0);
	}
	if (fcntl(pipe_select[0], F_SETFL,
				fcntl(pipe_select[0], F_GETFL, 0) | O_NONBLOCK)
			|| fcntl(pipe_select[1], F_SETFL,
				fcntl(pipe_select[1], F_GETFL, 0) | O_NONBLOCK)) {
		printf("Setting pipe NON-BLOCK failed\n");
		exit(0);
	}
	FD_SET(pipe_select[0], &rfds);

	/* read response file */
	if ((resp_file_fd = open(RESP_FILE, O_RDONLY)) < 0) {
		fprintf(stderr, "Cannot open file, %s\n", RESP_FILE);
		exit(0);
	}
	if (fstat(resp_file_fd, &status)) {
		fprintf(stderr, "Cannot read file stat of %s\n", RESP_FILE);
		exit(0);
	}
	resp_file_size = status.st_size;
	resp_file_buf = (char *)malloc(resp_file_size);
	if (read(resp_file_fd, resp_file_buf, resp_file_size) != resp_file_size) {
		fprintf(stderr, "Cannot read file, %s\n", RESP_FILE);
		exit(0);
	}
	if (!(resp_lat_ptr = strstr(resp_file_buf, "$LAT"))
			|| !(resp_lng_ptr = strstr(resp_file_buf, "$LNG"))
			|| !(resp_score_ptr = strstr(resp_file_buf, "$SCORE"))) {
		fprintf(stderr, "Cannot find $LAT or $LNG from %s\n", RESP_FILE);
		exit(0);
	}
	close(resp_file_fd);

	/* allocate image buffer */
	for (i = 0; i < BACKLOG; i++)
		if (!(img_buf[i] = (char *)malloc(NET_IMG_SIZE_LIMIT))) {
			fprintf(stderr, "%s: Failed to allocate image buffer\n", __func__);
			exit(0);
		}

	listen_fd = NewListenSocket();

	FD_SET(listen_fd, &rfds);
	max_fd = listen_fd;
	while(1) {
		rfds_tmp = rfds;
		wfds_tmp = wfds;

		select(max_fd + 1, &rfds_tmp, &wfds_tmp, NULL, NULL);
#if 0
		struct timeval tv;
		gettimeofday(&tv, NULL);
		printf("[%lu]listen: %d, pipe_read: %d, pipe_write: %d\n",
				tv.tv_sec,
				listen_fd, pipe_select[0], pipe_select[1]);
		printf("rfds: ");
		for (i = 0; i < max_fd + 1 ; i++)
			if (FD_ISSET(i, &rfds))
				printf("%d ", i);
		printf("\n");
		printf("rfds_tmp: ");
		for (i = 0; i < max_fd + 1 ; i++)
			if (FD_ISSET(i, &rfds_tmp))
				printf("%d ", i);
		printf("\n");
		printf("wfds: ");
		for (i = 0; i < max_fd + 1 ; i++)
			if (FD_ISSET(i, &wfds))
				printf("%d ", i);
		printf("\n");
		printf("wfds_tmp: ");
		for (i = 0; i < max_fd + 1 ; i++)
			if (FD_ISSET(i, &wfds_tmp))
				printf("%d ", i);
		printf("\n");
		printf("\n");
#endif
		if (FD_ISSET(listen_fd, &rfds_tmp)) {
			do_accept = 1;
			FD_CLR(listen_fd, &rfds_tmp);
		}

		if (FD_ISSET(pipe_select[0], &rfds_tmp)) {
			while(read(pipe_select[0], &dummy, sizeof(dummy)) > 0);
			FD_CLR(pipe_select[0], &rfds_tmp);
		}

		for (i = 0; i < max_fd + 1; i++) {
			/* read */
			if (FD_ISSET(i, &rfds_tmp)) {
				if ((idx = GetImgBufIdx(i, img_buf_tag)) == -1) {
					FD_CLR(i, &rfds);
					CLOSE(i);
					continue;
				}

				if (NET_IMG_SIZE_LIMIT - img_buf_len[idx] <= 0) {
					ReleaseImgBuf(idx, img_buf_tag, img_buf_len);
					FD_CLR(i, &rfds);
					CLOSE(i);
					continue;
				}

				if ((len = recv(i, &(img_buf[idx][img_buf_len[idx]]),
								NET_IMG_SIZE_LIMIT - img_buf_len[idx], 0)) <= 0)
					continue;
				img_buf_len[idx] += len;

				err = ParseHTTPPOST(img_buf[idx], img_buf_len[idx],
							&img_ptr, &img_len, &img_name_ptr, &img_name_len);
				if (err == -1)
					continue;
				else if (err == -2) {
					ReleaseImgBuf(idx, img_buf_tag, img_buf_len);
					FD_CLR(i, &rfds);
					CLOSE(i);
					continue;
				}
#if 0
				printf("sock %d: image(%d)-------------\n", i, img_len);
				for (j = 0; j < img_len; j++) {
					if (img_ptr[j] == '\r')
						printf("\\r");
					else if (img_ptr[j] == '\n')
						printf("\\n\n");
					else
						printf("%c", img_ptr[j]);
				}
				printf("\n");
#endif
				if ((img = LogAndReadImage(img_ptr, img_len,
						img_name_ptr, img_name_len)) != NULL) {
					cb_arg[i].sock = i;
					cb_arg[i].wfds = &wfds;
					cb_arg[i].pipe_wkup = pipe_select[1];

					if (sc_request(sc, img, &callback, (void*)&cb_arg[i])) {
						cvReleaseImage(&img);
						CLOSE(i);
					}
				}

				ReleaseImgBuf(idx, img_buf_tag, img_buf_len);
				FD_CLR(i, &rfds);
			}
			/* write */
			else if (FD_ISSET(i, &wfds_tmp)) {
				len = SprintHTTPHdr(buffer, resp_file_size);
				len = send(i, buffer, len, 0);
				len = sprintf(resp_lat_ptr, "%10.6f", cb_arg[i].latitude);
				resp_lat_ptr[len] = ' ';
				len = sprintf(resp_lng_ptr, "%10.6f", cb_arg[i].longitude);
				resp_lng_ptr[len] = ' ';
				len = sprintf(resp_score_ptr, "%6.2f", cb_arg[i].score);
				resp_score_ptr[len] = ' ';
				len = send(i, resp_file_buf, resp_file_size, 0);
				FD_CLR(i, &wfds);
				CLOSE(i);
			}
		}

		if (do_accept) {
			if ((new_fd = accept(listen_fd, (struct sockaddr *)&their_addr,
							(socklen_t*)&sin_size)) == -1) {
				perror("accept");
			}
			else {
				if (new_fd >= MAX_SOCK) {
					CLOSE(new_fd);
					continue;
				}
				max_fd = max_fd > new_fd ? max_fd : new_fd;
				FD_SET(new_fd, &rfds);
				if (fcntl(new_fd, F_SETFL,
							fcntl(new_fd, F_GETFL, 0) | O_NONBLOCK)) {
					printf("Setting NON-BLOCK failed\n");
					FD_CLR(new_fd, &rfds);
					CLOSE(new_fd);
					continue;
				}
			}
			do_accept = 0;
		}
	}
}
