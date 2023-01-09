/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 * u_vector is a vector based queue for storing arbitrary
 * sized arrays of objects without using a linked list.
 */

#ifndef U_VECTOR_H
#define U_VECTOR_H

#include <stdint.h>
#include <stdlib.h>
#include "util/macros.h"
#include "util/u_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TODO - move to u_math.h - name it better etc */
static inline uint32_t
u_align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -((int32_t) a)));
   return (v + a - 1) & ~(a - 1);
}

struct u_vector {
   uint32_t head;
   uint32_t tail;
   uint32_t element_size;
   uint32_t size;
   void *data;
};

int u_vector_init_pow2(struct u_vector *queue,
                       uint32_t initial_element_count,
                       uint32_t element_size);

void *u_vector_add(struct u_vector *queue);
void *u_vector_remove(struct u_vector *queue);

static inline int
u_vector_init(struct u_vector *queue,
              uint32_t initial_element_count,
              uint32_t element_size)
{
   initial_element_count = util_next_power_of_two(initial_element_count);
   element_size = util_next_power_of_two(element_size);
   return u_vector_init_pow2(queue, initial_element_count, element_size);
}

static inline int
u_vector_length(struct u_vector *queue)
{
   return (queue->head - queue->tail) / queue->element_size;
}

static inline void *
u_vector_head(struct u_vector *vector)
{
   assert(vector->tail < vector->head);
   return (void *)((char *)vector->data +
                   ((vector->head - vector->element_size) &
                    (vector->size - 1)));
}

static inline void *
u_vector_tail(struct u_vector *vector)
{
   return (void *)((char *)vector->data + (vector->tail & (vector->size - 1)));
}

static inline void
u_vector_finish(struct u_vector *queue)
{
   free(queue->data);
}

#ifdef __cplusplus
#define u_vector_element_cast(elem) (decltype(elem))
#else
#define u_vector_element_cast(elem) (void *)
#endif

#define u_vector_foreach(elem, queue)                                  \
   STATIC_ASSERT(__builtin_types_compatible_p(__typeof__(queue), struct u_vector *)); \
   for (uint32_t __u_vector_offset = (queue)->tail;                                \
        elem = u_vector_element_cast(elem)((char *)(queue)->data + \
                                           (__u_vector_offset & ((queue)->size - 1))), \
           __u_vector_offset != (queue)->head;                          \
        __u_vector_offset += (queue)->element_size)

#ifdef __cplusplus
}
#endif

#endif

