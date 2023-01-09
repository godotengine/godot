/**************************************************************************
 *
 * Copyright 2017 Valve Corporation
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/* Allocator of IDs (e.g. OpenGL object IDs), or simply an allocator of
 * numbers.
 *
 * The allocator uses a bit array to track allocated IDs.
 */

#ifndef U_IDALLOC_H
#define U_IDALLOC_H

#include <inttypes.h>
#include <stdbool.h>
#include "simple_mtx.h"

#ifdef __cplusplus
extern "C" {
#endif

struct util_idalloc
{
   uint32_t *data;
   unsigned num_elements;    /* number of allocated elements of "data" */
   unsigned lowest_free_idx;
};

void
util_idalloc_init(struct util_idalloc *buf, unsigned initial_num_ids);

void
util_idalloc_fini(struct util_idalloc *buf);

unsigned
util_idalloc_alloc(struct util_idalloc *buf);

unsigned
util_idalloc_alloc_range(struct util_idalloc *buf, unsigned num);

void
util_idalloc_free(struct util_idalloc *buf, unsigned id);

void
util_idalloc_reserve(struct util_idalloc *buf, unsigned id);

static inline bool
util_idalloc_exists(struct util_idalloc *buf, unsigned id)
{
   return id / 32 < buf->num_elements &&
          buf->data[id / 32] & BITFIELD_BIT(id % 32);
}

#define util_idalloc_foreach(buf, id) \
   for (uint32_t i = 0, mask = (buf)->num_elements ? (buf)->data[0] : 0, id, \
                 count = (buf)->num_elements; \
        i < count; mask = ++i < count ? (buf)->data[i] : 0) \
      while (mask) \
         if ((id = i * 32 + u_bit_scan(&mask)), true)


/* Thread-safe variant. */
struct util_idalloc_mt {
   struct util_idalloc buf;
   simple_mtx_t mutex;
   bool skip_zero;
};

void
util_idalloc_mt_init(struct util_idalloc_mt *buf,
                     unsigned initial_num_ids, bool skip_zero);

void
util_idalloc_mt_init_tc(struct util_idalloc_mt *buf);

void
util_idalloc_mt_fini(struct util_idalloc_mt *buf);

unsigned
util_idalloc_mt_alloc(struct util_idalloc_mt *buf);

void
util_idalloc_mt_free(struct util_idalloc_mt *buf, unsigned id);


#ifdef __cplusplus
}
#endif

#endif /* U_IDALLOC_H */
