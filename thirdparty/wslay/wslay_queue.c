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
#include "wslay_queue.h"

#include <string.h>
#include <assert.h>

#include "wslay_macro.h"

void wslay_queue_init(struct wslay_queue *queue) {
  queue->top = NULL;
  queue->tail = &queue->top;
}

void wslay_queue_deinit(struct wslay_queue *queue) { (void)queue; }

void wslay_queue_push(struct wslay_queue *queue,
                      struct wslay_queue_entry *ent) {
  ent->next = NULL;
  *queue->tail = ent;
  queue->tail = &ent->next;
}

void wslay_queue_push_front(struct wslay_queue *queue,
                            struct wslay_queue_entry *ent) {
  ent->next = queue->top;
  queue->top = ent;

  if (ent->next == NULL) {
    queue->tail = &ent->next;
  }
}

void wslay_queue_pop(struct wslay_queue *queue) {
  assert(queue->top);
  queue->top = queue->top->next;
  if (queue->top == NULL) {
    queue->tail = &queue->top;
  }
}

struct wslay_queue_entry *wslay_queue_top(struct wslay_queue *queue) {
  assert(queue->top);
  return queue->top;
}

struct wslay_queue_entry *wslay_queue_tail(struct wslay_queue *queue) {
  assert(queue->top);
  return wslay_struct_of(queue->tail, struct wslay_queue_entry, next);
}

int wslay_queue_empty(struct wslay_queue *queue) {
  assert(queue->top || queue->tail == &queue->top);
  return queue->top == NULL;
}
