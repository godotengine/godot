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
#include "wslay_stack.h"

#include <string.h>
#include <assert.h>

struct wslay_stack* wslay_stack_new()
{
  struct wslay_stack *stack = (struct wslay_stack*)malloc
    (sizeof(struct wslay_stack));
  if(!stack) {
    return NULL;
  }
  stack->top = NULL;
  return stack;
}

void wslay_stack_free(struct wslay_stack *stack)
{
  struct wslay_stack_cell *p;
  if(!stack) {
    return;
  }
  p = stack->top;
  while(p) {
    struct wslay_stack_cell *next = p->next;
    free(p);
    p = next;
  }
  free(stack);
}

int wslay_stack_push(struct wslay_stack *stack, void *data)
{
  struct wslay_stack_cell *new_cell = (struct wslay_stack_cell*)malloc
    (sizeof(struct wslay_stack_cell));
  if(!new_cell) {
    return WSLAY_ERR_NOMEM;
  }
  new_cell->data = data;
  new_cell->next = stack->top;
  stack->top = new_cell;
  return 0;
}

void wslay_stack_pop(struct wslay_stack *stack)
{
  struct wslay_stack_cell *top = stack->top;
  assert(top);
  stack->top = top->next;
  free(top);
}

void* wslay_stack_top(struct wslay_stack *stack)
{
  assert(stack->top);
  return stack->top->data;
}

int wslay_stack_empty(struct wslay_stack *stack)
{
  return stack->top == NULL;
}
