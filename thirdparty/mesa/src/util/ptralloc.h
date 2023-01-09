/*
 * Copyright © 2021 Valve Corporation
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
 *
 * Authors:
 *    Mike Blumenkrantz <michael.blumenkrantz@gmail.com>
 */

#pragma once

#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
#include "macros.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void *
ptralloc(size_t base_size, unsigned member_count, size_t *member_sizes, void ***members)
{
   size_t size = base_size;
   for (unsigned i = 0; i < member_count; i++)
      size += member_sizes[i];

   uint8_t *ptr = (uint8_t*)malloc(size);
   if (!ptr)
      return NULL;

   size_t accum = base_size;
   for (unsigned i = 0; i < member_count; i++) {
      *members[i] = (void*)(ptr + accum);
      accum += member_sizes[i];
   }
   return (void*)ptr;
}

static inline void *
ptrzalloc(size_t base_size, unsigned member_count, size_t *member_sizes, void ***members)
{
   size_t size = base_size;
   for (unsigned i = 0; i < member_count; i++)
      size += member_sizes[i];

   uint8_t *ptr = (uint8_t*)calloc(1, size);
   if (!ptr)
      return NULL;

   size_t accum = base_size;
   for (unsigned i = 0; i < member_count; i++) {
      *members[i] = (void*)(ptr + accum);
      accum += member_sizes[i];
   }
   return (void*)ptr;
}

#ifdef __cplusplus
}
#endif
