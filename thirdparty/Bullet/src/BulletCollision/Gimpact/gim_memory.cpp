/*
-----------------------------------------------------------------------------
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2006 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of EITHER:
   (1) The GNU Lesser General Public License as published by the Free
       Software Foundation; either version 2.1 of the License, or (at
       your option) any later version. The text of the GNU Lesser
       General Public License is included with this library in the
       file GIMPACT-LICENSE-LGPL.TXT.
   (2) The BSD-style license that is included with this library in
       the file GIMPACT-LICENSE-BSD.TXT.
   (3) The zlib/libpng license that is included with this library in
       the file GIMPACT-LICENSE-ZLIB.TXT.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 GIMPACT-LICENSE-LGPL.TXT, GIMPACT-LICENSE-ZLIB.TXT and GIMPACT-LICENSE-BSD.TXT for more details.

-----------------------------------------------------------------------------
*/


#include "gim_memory.h"
#include "stdlib.h"

#ifdef GIM_SIMD_MEMORY
#include "LinearMath/btAlignedAllocator.h"
#endif

static gim_alloc_function *g_allocfn = 0;
static gim_alloca_function *g_allocafn = 0;
static gim_realloc_function *g_reallocfn = 0;
static gim_free_function *g_freefn = 0;

void gim_set_alloc_handler (gim_alloc_function *fn)
{
  g_allocfn = fn;
}

void gim_set_alloca_handler (gim_alloca_function *fn)
{
  g_allocafn = fn;
}

void gim_set_realloc_handler (gim_realloc_function *fn)
{
  g_reallocfn = fn;
}

void gim_set_free_handler (gim_free_function *fn)
{
  g_freefn = fn;
}

gim_alloc_function *gim_get_alloc_handler()
{
  return g_allocfn;
}

gim_alloca_function *gim_get_alloca_handler()
{
  return g_allocafn;
}


gim_realloc_function *gim_get_realloc_handler ()
{
  return g_reallocfn;
}


gim_free_function  *gim_get_free_handler ()
{
  return g_freefn;
}


void * gim_alloc(size_t size)
{
	void * ptr;
	if (g_allocfn)
	{
		ptr = g_allocfn(size);
	}
	else
	{
#ifdef GIM_SIMD_MEMORY
		ptr = btAlignedAlloc(size,16);
#else
		ptr = malloc(size);
#endif
	}
  	return ptr;
}

void * gim_alloca(size_t size)
{
  if (g_allocafn) return g_allocafn(size); else return gim_alloc(size);
}


void * gim_realloc(void *ptr, size_t oldsize, size_t newsize)
{
 	void * newptr = gim_alloc(newsize);
    size_t copysize = oldsize<newsize?oldsize:newsize;
    gim_simd_memcpy(newptr,ptr,copysize);
    gim_free(ptr);
    return newptr;
}

void gim_free(void *ptr)
{
	if (!ptr) return;
	if (g_freefn)
	{
	   g_freefn(ptr);
	}
	else
	{
	#ifdef GIM_SIMD_MEMORY
		btAlignedFree(ptr);
	#else
		free(ptr);
	#endif
	}
}

