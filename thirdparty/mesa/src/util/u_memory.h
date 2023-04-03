/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
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
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


/*
 * Memory functions
 */


#ifndef U_MEMORY_H
#define U_MEMORY_H

#include "util/u_debug.h"
#include "util/os_memory.h"


#ifdef __cplusplus
extern "C" {
#endif


#define MALLOC(_size)  os_malloc(_size)

#define CALLOC(_count, _size) os_calloc(_count, _size)

#define FREE(_ptr ) os_free(_ptr)

#define REALLOC(_ptr, _old_size, _size) os_realloc(_ptr, _old_size, _size)

#define MALLOC_STRUCT(T)   (struct T *) MALLOC(sizeof(struct T))

#define CALLOC_STRUCT(T)   (struct T *) CALLOC(1, sizeof(struct T))

#define CALLOC_VARIANT_LENGTH_STRUCT(T,more_size)   ((struct T *) CALLOC(1, sizeof(struct T) + more_size))


#define align_malloc(_size, _alignment) os_malloc_aligned(_size, _alignment)
#define align_free(_ptr) os_free_aligned(_ptr)
#define align_realloc(_ptr, _oldsize, _newsize, _alignment) os_realloc_aligned(_ptr, _oldsize, _newsize, _alignment)

static inline void *
align_calloc(size_t size, unsigned long alignment)
{
   void *ptr = align_malloc(size, alignment);
   if (ptr)
      memset(ptr, 0, size);
   return ptr;
}

/**
 * Duplicate a block of memory.
 */
static inline void *
mem_dup(const void *src, size_t size)
{
   void *dup = MALLOC(size);
   if (dup)
      memcpy(dup, src, size);
   return dup;
}


/**
 * Offset of a field in a struct, in bytes.
 */
#define Offset(TYPE, MEMBER) ((uintptr_t)&(((TYPE *)NULL)->MEMBER))

/* TODO: this could be different on non-x86 architectures. */
#define CACHE_LINE_SIZE 64

/**
 * Declare a variable on its own cache line.
 *
 * This helps eliminate "False sharing" to make atomic operations
 * on pipe_reference::count faster and/or access to adjacent fields faster.
 *
 * https://en.wikipedia.org/wiki/False_sharing
 *
 * CALLOC_STRUCT_CL or MALLOC_STRUCT_CL and FREE_CL should be used to allocate
 * structures that contain this.
 *
 * NOTE: Don't use c11 alignas because it causes the whole structure to be
 *       aligned, but we only want to align the field.
 */
#define EXCLUSIVE_CACHELINE(decl) \
   union { char __cl_space[CACHE_LINE_SIZE]; \
           decl; }

/* Allocate a structure aligned to a cache line. (used to make atomic ops faster) */
#define MALLOC_STRUCT_CL(T) (struct T *)align_malloc(sizeof(struct T), CACHE_LINE_SIZE)
#define CALLOC_STRUCT_CL(T) (struct T *)align_calloc(sizeof(struct T), CACHE_LINE_SIZE)
#define FREE_CL(ptr) align_free(ptr)

#ifdef __cplusplus
}
#endif


#endif /* U_MEMORY_H */
