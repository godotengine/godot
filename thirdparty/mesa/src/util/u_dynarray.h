/**************************************************************************
 *
 * Copyright 2010 Luca Barbieri
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef U_DYNARRAY_H
#define U_DYNARRAY_H

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "ralloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A zero-initialized version of this is guaranteed to represent an
 * empty array.
 *
 * Also, size <= capacity and data != 0 if and only if capacity != 0
 * capacity will always be the allocation size of data
 */
struct util_dynarray
{
   void *mem_ctx;
   void *data;
   unsigned size;
   unsigned capacity;
};

static inline void
util_dynarray_init(struct util_dynarray *buf, void *mem_ctx)
{
   memset(buf, 0, sizeof(*buf));
   buf->mem_ctx = mem_ctx;
}

static inline void
util_dynarray_fini(struct util_dynarray *buf)
{
   if (buf->data) {
      if (buf->mem_ctx) {
         ralloc_free(buf->data);
      } else {
         free(buf->data);
      }
      util_dynarray_init(buf, buf->mem_ctx);
   }
}

static inline void
util_dynarray_clear(struct util_dynarray *buf)
{
	buf->size = 0;
}

#define DYN_ARRAY_INITIAL_SIZE 64

MUST_CHECK static inline void *
util_dynarray_ensure_cap(struct util_dynarray *buf, unsigned newcap)
{
   if (newcap > buf->capacity) {
      unsigned capacity = MAX3(DYN_ARRAY_INITIAL_SIZE, buf->capacity * 2, newcap);
      void *data;

      if (buf->mem_ctx) {
         data = reralloc_size(buf->mem_ctx, buf->data, capacity);
      } else {
         data = realloc(buf->data, capacity);
      }
      if (!data)
         return NULL;

      buf->data = data;
      buf->capacity = capacity;
   }

   return (void *)((char *)buf->data + buf->size);
}

/* use util_dynarray_trim to reduce the allocated storage */
MUST_CHECK static inline void *
util_dynarray_resize_bytes(struct util_dynarray *buf, unsigned nelts, size_t eltsize)
{
   if (unlikely(nelts > UINT_MAX / eltsize))
      return NULL;

   unsigned newsize = nelts * eltsize;
   void *p = util_dynarray_ensure_cap(buf, newsize);
   if (!p)
      return NULL;

   buf->size = newsize;

   return p;
}

static inline void
util_dynarray_clone(struct util_dynarray *buf, void *mem_ctx,
                    struct util_dynarray *from_buf)
{
   util_dynarray_init(buf, mem_ctx);
   if (util_dynarray_resize_bytes(buf, from_buf->size, 1))
      memcpy(buf->data, from_buf->data, from_buf->size);
}

MUST_CHECK static inline void *
util_dynarray_grow_bytes(struct util_dynarray *buf, unsigned ngrow, size_t eltsize)
{
   unsigned growbytes = ngrow * eltsize;

   if (unlikely(ngrow > (UINT_MAX / eltsize) ||
                growbytes > UINT_MAX - buf->size))
      return NULL;

   unsigned newsize = buf->size + growbytes;
   void *p = util_dynarray_ensure_cap(buf, newsize);
   if (!p)
      return NULL;

   buf->size = newsize;

   return p;
}

static inline void
util_dynarray_trim(struct util_dynarray *buf)
{
   if (buf->size != buf->capacity) {
      if (buf->size) {
         if (buf->mem_ctx) {
            buf->data = reralloc_size(buf->mem_ctx, buf->data, buf->size);
         } else {
            buf->data = realloc(buf->data, buf->size);
         }
         buf->capacity = buf->size;
      } else {
         if (buf->mem_ctx) {
            ralloc_free(buf->data);
         } else {
            free(buf->data);
         }
         buf->data = NULL;
         buf->capacity = 0;
      }
   }
}

static inline void
util_dynarray_append_dynarray(struct util_dynarray *buf,
                              const struct util_dynarray *other)
{
   if (other->size > 0) {
      void *p = util_dynarray_grow_bytes(buf, 1, other->size);
      memcpy(p, other->data, other->size);
   }
}

#define util_dynarray_append(buf, type, v) do {type __v = (v); memcpy(util_dynarray_grow_bytes((buf), 1, sizeof(type)), &__v, sizeof(type));} while(0)
/* Returns a pointer to the space of the first new element (in case of growth) or NULL on failure. */
#define util_dynarray_resize(buf, type, nelts) util_dynarray_resize_bytes(buf, (nelts), sizeof(type))
#define util_dynarray_grow(buf, type, ngrow) util_dynarray_grow_bytes(buf, (ngrow), sizeof(type))
#define util_dynarray_top_ptr(buf, type) (type*)((char*)(buf)->data + (buf)->size - sizeof(type))
#define util_dynarray_top(buf, type) *util_dynarray_top_ptr(buf, type)
#define util_dynarray_pop_ptr(buf, type) (type*)((char*)(buf)->data + ((buf)->size -= sizeof(type)))
#define util_dynarray_pop(buf, type) *util_dynarray_pop_ptr(buf, type)
#define util_dynarray_contains(buf, type) ((buf)->size >= sizeof(type))
#define util_dynarray_element(buf, type, idx) ((type*)(buf)->data + (idx))
#define util_dynarray_begin(buf) ((buf)->data)
#define util_dynarray_end(buf) ((void*)util_dynarray_element((buf), char, (buf)->size))
#define util_dynarray_num_elements(buf, type) ((buf)->size / sizeof(type))

#define util_dynarray_foreach(buf, type, elem) \
   for (type *elem = (type *)(buf)->data; \
        elem < (type *)((char *)(buf)->data + (buf)->size); elem++)

#define util_dynarray_foreach_reverse(buf, type, elem)          \
   if ((buf)->size > 0)                                         \
      for (type *elem = util_dynarray_top_ptr(buf, type);       \
           elem;                                                \
           elem = elem > (type *)(buf)->data ? elem - 1 : NULL)

#define util_dynarray_delete_unordered(buf, type, v)                    \
   do {                                                                 \
      unsigned num_elements = (buf)->size / sizeof(type);               \
      unsigned i;                                                       \
      for (i = 0; i < num_elements; i++) {                              \
         type __v = *util_dynarray_element((buf), type, (i));           \
         if (v == __v) {                                                \
            memcpy(util_dynarray_element((buf), type, (i)),             \
                   util_dynarray_pop_ptr((buf), type), sizeof(type));   \
            break;                                                      \
         }                                                              \
      }                                                                 \
   } while (0)

#ifdef __cplusplus
}
#endif

#endif /* U_DYNARRAY_H */

