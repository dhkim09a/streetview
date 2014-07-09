#ifndef __MESSAGE_H_
#define __MESSAGE_H_

#include <pthread.h>

typedef struct _msg_t msg_t;
typedef struct _msg_t {
	void *content;
	void *arg;
	void (*cb)(msg_t *msg);
} msg_t;

typedef struct _msgbox_t {
	/* circular queue */
	msg_t queue[BACKLOG];
	int head;
	int tail;
	int empty;

	pthread_mutex_t mx_queue;
	pthread_cond_t cd_owner;
} msgbox_t;

int msg_init_box(msgbox_t *box);
int msg_write(msgbox_t *box, void *content, void (*cb)(msg_t *msg), void *arg);
int msg_read(msgbox_t *box, msg_t *msg);

#endif /* __MESSAGE_H_ */
