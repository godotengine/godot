/*
 * Copyright © 2018 Intel Corporation
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

#ifndef _UTIL_VMA_H
#define _UTIL_VMA_H

#include <stdint.h>
#include <stdio.h>

#include "list.h"

#ifdef __cplusplus
extern "C" {
#endif

struct util_vma_heap {
   struct list_head holes;

   /** Total size of free memory. */
   uint64_t free_size;

   /** If true, util_vma_heap_alloc will prefer high addresses
    *
    * Default is true.
    */
   bool alloc_high;

   /**
    * If non-zero, util_vma_heap_alloc will avoid allocating regions which
    * span (1 << nospan_shift) ranges.  For example, to avoid allocations
    * which straddle 4GB boundaries, use nospan_shift=log2(4GB)
    */
   unsigned nospan_shift;
};

void util_vma_heap_init(struct util_vma_heap *heap,
                        uint64_t start, uint64_t size);
void util_vma_heap_finish(struct util_vma_heap *heap);

uint64_t util_vma_heap_alloc(struct util_vma_heap *heap,
                             uint64_t size, uint64_t alignment);

bool util_vma_heap_alloc_addr(struct util_vma_heap *heap,
                              uint64_t addr, uint64_t size);

void util_vma_heap_free(struct util_vma_heap *heap,
                        uint64_t offset, uint64_t size);

void util_vma_heap_print(struct util_vma_heap *heap, FILE *fp,
                         const char *tab, uint64_t total_size);

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* _UTIL_DEBUG_H */
