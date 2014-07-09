
#include "message.h"

int msg_init_box(msgbox_t *box)
{
	int i;

	if (!box)
		return -1;

	for (i = 0; i < BACKLOG; i++) {
		box->queue[i].content = NULL;
		box->queue[i].cb = NULL;
	}
	box->head = 0; 
	box->tail = 0;
	box->empty = 1;

	pthread_mutex_init(&box->mx_queue, NULL);
	pthread_cond_init(&box->cd_owner, NULL);

	return 0;
}

int msg_write(msgbox_t *box,
		void *content, void (*cb)(msg_t *msg), void *arg)
{
	msg_t *msg_slot;

	if (!box)
		return -1;

	pthread_mutex_lock(&box->mx_queue);
	if (!box->empty && box->head == box->tail) {
		pthread_mutex_unlock(&box->mx_queue);
		return -1;
	}
	else {
		msg_slot = &box->queue[box->tail];
		box->tail = (box->tail + 1) % BACKLOG;
	}

	msg_slot->content = content;
	msg_slot->cb = cb;
	msg_slot->arg = arg;

	box->empty = 0;
	pthread_cond_signal(&box->cd_owner);
	pthread_mutex_unlock(&box->mx_queue);

	return 0;
}

int msg_read(msgbox_t *box, msg_t *msg)
{
	msg_t *msg_slot;

	if (!msg || !box)
		return -1;

	pthread_mutex_lock(&box->mx_queue);
	if (box->empty)
		pthread_cond_wait(&box->cd_owner, &box->mx_queue);
	msg_slot = &box->queue[box->head];

	msg->content = msg_slot->content;
	msg->cb = msg_slot->cb;
	msg->arg = msg_slot->arg;

	box->head = (box->head + 1) % BACKLOG;
	if (box->head == box->tail)
		box->empty = 1;
	pthread_mutex_unlock(&box->mx_queue);

	return 0;
}
