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

struct wslay_queue* wslay_queue_new(void)
{
  struct wslay_queue *queue = (struct wslay_queue*)malloc
    (sizeof(struct wslay_queue));
  if(!queue) {
    return NULL;
  }
  queue->top = queue->tail = NULL;
  return queue;
}

void wslay_queue_free(struct wslay_queue *queue)
{
  if(!queue) {
    return;
  } else {
    struct wslay_queue_cell *p = queue->top;
    while(p) {
      struct wslay_queue_cell *next = p->next;
      free(p);
      p = next;
    }
    free(queue);
  }
}

int wslay_queue_push(struct wslay_queue *queue, void *data)
{
  struct wslay_queue_cell *new_cell = (struct wslay_queue_cell*)malloc
    (sizeof(struct wslay_queue_cell));
  if(!new_cell) {
    return WSLAY_ERR_NOMEM;
  }
  new_cell->data = data;
  new_cell->next = NULL;
  if(queue->tail) {
    queue->tail->next = new_cell;
    queue->tail = new_cell;

  } else {
    queue->top = queue->tail = new_cell;
  }
  return 0;
}

int wslay_queue_push_front(struct wslay_queue *queue, void *data)
{
  struct wslay_queue_cell *new_cell = (struct wslay_queue_cell*)malloc
    (sizeof(struct wslay_queue_cell));
  if(!new_cell) {
    return WSLAY_ERR_NOMEM;
  }
  new_cell->data = data;
  new_cell->next = queue->top;
  queue->top = new_cell;
  if(!queue->tail) {
    queue->tail = queue->top;
  }
  return 0;
}

void wslay_queue_pop(struct wslay_queue *queue)
{
  struct wslay_queue_cell *top = queue->top;
  assert(top);
  queue->top = top->next;
  if(top == queue->tail) {
    queue->tail = NULL;
  }
  free(top);
}

void* wslay_queue_top(struct wslay_queue *queue)
{
  assert(queue->top);
  return queue->top->data;
}

void* wslay_queue_tail(struct wslay_queue *queue)
{
  assert(queue->tail);
  return queue->tail->data;
}

int wslay_queue_empty(struct wslay_queue *queue)
{
  return queue->top == NULL;
}
