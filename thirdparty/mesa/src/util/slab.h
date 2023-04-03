/*
 * Copyright 2010 Marek Olšák <maraeo@gmail.com>
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE. */

/**
 * Slab allocator for equally sized memory allocations.
 *
 * Objects are allocated from "child" pools that are connected to a "parent"
 * pool.
 *
 * Calls to slab_alloc/slab_free for the same child pool must not occur from
 * multiple threads simultaneously.
 *
 * Allocations obtained from one child pool should usually be freed in the
 * same child pool. Freeing an allocation in a different child pool associated
 * to the same parent is allowed (and requires no locking by the caller), but
 * it is discouraged because it implies a performance penalty.
 *
 * For convenience and to ease the transition, there is also a set of wrapper
 * functions around a single parent-child pair.
 */

#ifndef SLAB_H
#define SLAB_H

#include "simple_mtx.h"

#ifdef __cplusplus
extern "C" {
#endif

struct slab_element_header;
struct slab_page_header;

struct slab_parent_pool {
   simple_mtx_t mutex;
   unsigned element_size;
   unsigned num_elements;
   unsigned item_size;
};

struct slab_child_pool {
   struct slab_parent_pool *parent;

   struct slab_page_header *pages;

   /* Free elements. */
   struct slab_element_header *free;

   /* Elements that are owned by this pool but were freed with a different
    * pool as the argument to slab_free.
    *
    * This list is protected by the parent mutex.
    */
   struct slab_element_header *migrated;
};

void slab_create_parent(struct slab_parent_pool *parent,
                        unsigned item_size,
                        unsigned num_items);
void slab_destroy_parent(struct slab_parent_pool *parent);
void slab_create_child(struct slab_child_pool *pool,
                       struct slab_parent_pool *parent);
void slab_destroy_child(struct slab_child_pool *pool);
void *slab_alloc(struct slab_child_pool *pool);
void *slab_zalloc(struct slab_child_pool *pool);
void slab_free(struct slab_child_pool *pool, void *ptr);

struct slab_mempool {
   struct slab_parent_pool parent;
   struct slab_child_pool child;
};

void slab_create(struct slab_mempool *mempool,
                 unsigned item_size,
                 unsigned num_items);
void slab_destroy(struct slab_mempool *mempool);
void *slab_alloc_st(struct slab_mempool *mempool);
void slab_free_st(struct slab_mempool *mempool, void *ptr);

#ifdef __cplusplus
}
#endif

#endif
