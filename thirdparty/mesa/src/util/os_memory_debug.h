/**************************************************************************
 *
 * Copyright 2008-2010 VMware, Inc.
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
 * Debugging wrappers for OS memory management abstractions.
 */


#ifndef _OS_MEMORY_H_
#error "Must not be included directly. Include os_memory.h instead"
#endif


#ifdef __cplusplus
extern "C" {
#endif


void *
debug_malloc(const char *file, unsigned line, const char *function,
             size_t size);

void *
debug_calloc(const char *file, unsigned line, const char *function,
             size_t count, size_t size );

void
debug_free(const char *file, unsigned line, const char *function,
           void *ptr);

void *
debug_realloc(const char *file, unsigned line, const char *function,
              void *old_ptr, size_t old_size, size_t new_size );

unsigned long
debug_memory_begin(void);

void
debug_memory_end(unsigned long start_no);

void
debug_memory_tag(void *ptr, unsigned tag);

void
debug_memory_check_block(void *ptr);

void
debug_memory_check(void);


#ifdef __cplusplus
}
#endif


#ifndef DEBUG_MEMORY_IMPLEMENTATION

#define os_malloc( _size ) \
   debug_malloc( __FILE__, __LINE__, __func__, _size )
#define os_calloc( _count, _size ) \
   debug_calloc(__FILE__, __LINE__, __func__, _count, _size )
#define os_free( _ptr ) \
   debug_free( __FILE__, __LINE__, __func__,  _ptr )
#define os_realloc( _ptr, _old_size, _new_size ) \
   debug_realloc( __FILE__, __LINE__, __func__,  _ptr, _old_size, _new_size )

/* TODO: wrap os_malloc_aligned() and os_free_aligned() too */
#include "os_memory_aligned.h"

#endif /* !DEBUG_MEMORY_IMPLEMENTATION */
