/*
 * Wslay - The WebSocket Library
 *
 * Copyright (c) 2011, 2012 Tatsuhiro Tsujikawa
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef WSLAY_QUEUE_H
#define WSLAY_QUEUE_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif /* HAVE_CONFIG_H */

#include <wslay/wslay.h>

struct wslay_queue_entry {
  struct wslay_queue_entry *next;
};

struct wslay_queue {
  struct wslay_queue_entry *top;
  struct wslay_queue_entry **tail;
};

void wslay_queue_init(struct wslay_queue *queue);
void wslay_queue_deinit(struct wslay_queue *queue);
void wslay_queue_push(struct wslay_queue *queue, struct wslay_queue_entry *ent);
void wslay_queue_push_front(struct wslay_queue *queue,
                            struct wslay_queue_entry *ent);
void wslay_queue_pop(struct wslay_queue *queue);
struct wslay_queue_entry *wslay_queue_top(struct wslay_queue *queue);
struct wslay_queue_entry *wslay_queue_tail(struct wslay_queue *queue);
int wslay_queue_empty(struct wslay_queue *queue);

#endif /* WSLAY_QUEUE_H */
