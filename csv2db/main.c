#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <libgen.h>

#include "xxhash.h"

#define XXHSEED 18446744073709551557ULL

#define METALEN_DEFAULT (32)
#define STRLEN_MAX      (1024 * 1024)
#define VECDIM_MAX      (4096)
#define FILENAMELEN_MAX (128)

struct conf {
	int metalen;
	int vecdim;
};

void
print_help(void)
{
	printf("Usage: csv2db [-m filename length] -f CSV_FILE\n");
}

int
read_line(int fd, char *buf, int len)
{
	int i = 0;

	assert(len > 0);

	while (i < len - 1) {
		if (read(fd, &buf[i], 1) < 1)
			return -1;
		if (buf[i] == '\n' || buf[i] == '\r') {
			buf[i] = '\0';
			return i + 1;
		}
		i++;
	}

	return i;
}

int
csve2dbe(struct conf *conf, void *buf, int len, char *str, int strlen)
{
	char *saveptr = NULL;
	char *w;
	int cnt = 0;
	float *vec;

	if (!(w = strtok_r(str, ",", &saveptr)))
		return -1;

	snprintf((char *)buf, conf->metalen, "%s", w);

	vec = (float *)&((char *)buf)[conf->metalen];
	while ((w = strtok_r(NULL, ",", &saveptr)))
		vec[cnt++] = atof(w);

	if (conf->vecdim == 0)
		conf->vecdim = cnt;
	else if (conf->vecdim != cnt)
		return -1;

	return conf->metalen + conf->vecdim * sizeof(float);
}

int
main (int argc, char **argv)
{
	unsigned char opt;
	char *infile = NULL, outfile[FILENAMELEN_MAX + 1] = {0};
	int infd = -1, outfd = -1;
	char *inbuf = NULL, *outbuf = NULL;
	int inbuflen = STRLEN_MAX, outbuflen = 0;
	int r;
	struct conf conf = {
		.metalen = METALEN_DEFAULT,
		.vecdim = 0,
	};

	while ((opt = getopt(argc, argv, "m:f:")) != 0xFF) {
		switch (opt) {
			case 'm':
				conf.metalen = atoi(optarg);
				break;

			case 'f':
				infile = optarg;
				break;

			default:
				print_help();
				return 0;
		}
	}

	if (!infile) {
		print_help();
		return 0;
	}

	if ((infd = open(infile, O_RDONLY)) < 0) {
		fprintf(stderr, "%s not found", infile);
		return 0;
	}

	if (!(inbuf = (char *)malloc(inbuflen))) {
		perror("malloc");
		return 0;
	}

	outbuflen = conf.metalen + sizeof(float) * VECDIM_MAX;
	if (!(outbuf = (char *)malloc(outbuflen))) {
		perror("malloc");
		return 0;
	}

	while ((r = read_line(infd, inbuf, inbuflen)) > 0) {
		if (r == 1) /* skip empty lines */
			continue;

		r = csve2dbe(&conf, outbuf, outbuflen, inbuf, r);
		assert(r > 0);

		if (!*outfile) {
			snprintf(outfile, FILENAMELEN_MAX,
					"%d_%d_%s.db", conf.vecdim, conf.metalen, basename(infile));
			printf("outfile: %s\n", outfile);
		}

		if (outfd < 0 && (outfd = open(outfile, O_CREAT | O_WRONLY, 0664)) < 0) {
			fprintf(stderr, "Cannot create %s", outfile);
			return 0;
		}

		if (r != write(outfd, outbuf, r)) {
			fprintf(stderr, "Writing failure\n");
			return 0;
		}
	}

	close(infd);
	close(outfd);

	return 0;
}
