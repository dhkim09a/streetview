#ifndef __PROFILE_H_
#define __PROFILE_H_

#include <sys/time.h>

#ifndef PROFILE_MAX_VAR
#define PROFILE_MAX_VAR 20
#endif

#ifdef PROFILE_ON
#define PROFILE_START() \
	struct timeval __tv_total_from, __tv_total_to; \
	long __c_total_ms = 0, __c_etc_ms = 0; \
	long *__var_list[PROFILE_MAX_VAR] = {NULL}; \
	char *__name_list[PROFILE_MAX_VAR] = {NULL}; \
do { \
	gettimeofday(&__tv_total_from, NULL); \
} while (0)

#define PROFILE_END() \
do {int i; \
	__sync_synchronize(); \
	gettimeofday(&__tv_total_to, NULL); \
	__c_total_ms \
		+= (__tv_total_to.tv_sec - __tv_total_from.tv_sec) * 1000 \
		+ (__tv_total_to.tv_usec - __tv_total_from.tv_usec) / 1000; \
	__c_etc_ms = __c_total_ms; \
	for (i = 0; i < PROFILE_MAX_VAR; i++) \
		if (__var_list[i]) __c_etc_ms -= *__var_list[i]; \
} while (0)

#define PROFILE_VAR(var) \
	struct timeval __##var##_tv_from, __##var##_tv_to; \
	long __##var##_ms = 0; \
	{int i; for (i = 0; i < PROFILE_MAX_VAR; i++) \
		if (__var_list[i] == NULL) { \
			__var_list[i] = &__##var##_ms; \
			__name_list[i] = (char *)#var; break;}}

#define PROFILE_FROM(var) \
do { \
	__sync_synchronize(); \
	gettimeofday(&__##var##_tv_from, NULL); \
} while (0)

#define PROFILE_TO(var) \
do { \
	__sync_synchronize(); \
	gettimeofday(&__##var##_tv_to, NULL); \
	__##var##_ms \
		+= (__##var##_tv_to.tv_sec - __##var##_tv_from.tv_sec) * 1000 \
		+ (__##var##_tv_to.tv_usec - __##var##_tv_from.tv_usec) / 1000; \
} while (0)

#define PROFILE_PRINT(fp) \
do { int i; \
	fprintf(fp, "[PROFILE %s]\n", __func__); \
	for (i = 0; i < PROFILE_MAX_VAR; i++) \
		if (__var_list[i] != NULL) { \
			printf("                              : %7ld ms (%5.2f %%)", \
					*__var_list[i], \
					100 * (float)*__var_list[i] / (float)__c_total_ms); \
			printf("\r%s\n", __name_list[i]); \
		} \
	fprintf(fp, "etc.                          : %7ld ms (%5.2f %%)\n", \
			__c_etc_ms, 100 * (float)__c_etc_ms / (float)__c_total_ms); \
	fprintf(fp, "total                         : %7ld ms\n", __c_total_ms); \
	fflush(fp); \
} while (0)

#else /* PROFILE_ON */
#define PROFILE_START()
#define PROFILE_END()
#define PROFILE_VAR(args...)
#define PROFILE_FROM(args...)
#define PROFILE_TO(args...)
#define PROFILE_PRINT(args...)
#endif /* PROFILE_ON */

#endif /* __PROFILE_H_ */


