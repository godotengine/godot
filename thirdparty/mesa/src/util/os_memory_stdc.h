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
 * OS memory management abstractions for the standard C library.
 */


#ifndef _OS_MEMORY_H_
#error "Must not be included directly. Include os_memory.h instead"
#endif

#include <stdlib.h>


#define os_malloc(_size)  malloc(_size)
#define os_calloc(_count, _size )  calloc(_count, _size )
#define os_free(_ptr)  free(_ptr)

#define os_realloc( _old_ptr, _old_size, _new_size) \
   realloc(_old_ptr, _new_size + 0*(_old_size))

#if DETECT_OS_WINDOWS

#include <malloc.h>

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr) _aligned_free(_ptr)
#define os_realloc_aligned(_ptr, _oldsize, _newsize, _alignment) _aligned_realloc(_ptr, _newsize, _alignment)

#else

#include "os_memory_aligned.h"

#if DETECT_OS_UNIX

#include "os_memory_fd.h"

#endif

#endif
