/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_WORK_QUEUE_H
#define PIPEWIRE_WORK_QUEUE_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_work_queue Work Queue
 * Queued processing of work items.
 */

/**
 * \addtogroup pw_work_queue
 * \{
 */
struct pw_work_queue;

#include <pipewire/loop.h>

typedef void (*pw_work_func_t) (void *obj, void *data, int res, uint32_t id);

struct pw_work_queue *
pw_work_queue_new(struct pw_loop *loop);

void
pw_work_queue_destroy(struct pw_work_queue *queue);

uint32_t
pw_work_queue_add(struct pw_work_queue *queue,
		  void *obj, int res,
		  pw_work_func_t func, void *data);

int
pw_work_queue_cancel(struct pw_work_queue *queue, void *obj, uint32_t id);

int
pw_work_queue_complete(struct pw_work_queue *queue, void *obj, uint32_t seq, int res);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_WORK_QUEUE_H */
