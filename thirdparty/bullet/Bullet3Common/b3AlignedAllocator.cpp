/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "b3AlignedAllocator.h"

int b3g_numAlignedAllocs = 0;
int b3g_numAlignedFree = 0;
int b3g_totalBytesAlignedAllocs = 0;//detect memory leaks

static void *b3AllocDefault(size_t size)
{
	return malloc(size);
}

static void b3FreeDefault(void *ptr)
{
	free(ptr);
}

static b3AllocFunc* b3s_allocFunc = b3AllocDefault;
static b3FreeFunc* b3s_freeFunc = b3FreeDefault;



#if defined (B3_HAS_ALIGNED_ALLOCATOR)
#include <malloc.h>
static void *b3AlignedAllocDefault(size_t size, int alignment)
{
	return _aligned_malloc(size, (size_t)alignment);
}

static void b3AlignedFreeDefault(void *ptr)
{
	_aligned_free(ptr);
}
#elif defined(__CELLOS_LV2__)
#include <stdlib.h>

static inline void *b3AlignedAllocDefault(size_t size, int alignment)
{
	return memalign(alignment, size);
}

static inline void b3AlignedFreeDefault(void *ptr)
{
	free(ptr);
}
#else





static inline void *b3AlignedAllocDefault(size_t size, int alignment)
{
  void *ret;
  char *real;
  real = (char *)b3s_allocFunc(size + sizeof(void *) + (alignment-1));
  if (real) {
	ret = b3AlignPointer(real + sizeof(void *),alignment);
    *((void **)(ret)-1) = (void *)(real);
  } else {
    ret = (void *)(real);
  }
  return (ret);
}

static inline void b3AlignedFreeDefault(void *ptr)
{
  void* real;

  if (ptr) {
    real = *((void **)(ptr)-1);
    b3s_freeFunc(real);
  }
}
#endif


static b3AlignedAllocFunc* b3s_alignedAllocFunc = b3AlignedAllocDefault;
static b3AlignedFreeFunc* b3s_alignedFreeFunc = b3AlignedFreeDefault;

void b3AlignedAllocSetCustomAligned(b3AlignedAllocFunc *allocFunc, b3AlignedFreeFunc *freeFunc)
{
  b3s_alignedAllocFunc = allocFunc ? allocFunc : b3AlignedAllocDefault;
  b3s_alignedFreeFunc = freeFunc ? freeFunc : b3AlignedFreeDefault;
}

void b3AlignedAllocSetCustom(b3AllocFunc *allocFunc, b3FreeFunc *freeFunc)
{
  b3s_allocFunc = allocFunc ? allocFunc : b3AllocDefault;
  b3s_freeFunc = freeFunc ? freeFunc : b3FreeDefault;
}

#ifdef B3_DEBUG_MEMORY_ALLOCATIONS
//this generic allocator provides the total allocated number of bytes
#include <stdio.h>

void*   b3AlignedAllocInternal  (size_t size, int alignment,int line,char* filename)
{
 void *ret;
 char *real;

 b3g_totalBytesAlignedAllocs += size;
 b3g_numAlignedAllocs++;

 
 real = (char *)b3s_allocFunc(size + 2*sizeof(void *) + (alignment-1));
 if (real) {
   ret = (void*) b3AlignPointer(real + 2*sizeof(void *), alignment);
   *((void **)(ret)-1) = (void *)(real);
       *((int*)(ret)-2) = size;

 } else {
   ret = (void *)(real);//??
 }

 b3Printf("allocation#%d at address %x, from %s,line %d, size %d\n",b3g_numAlignedAllocs,real, filename,line,size);

 int* ptr = (int*)ret;
 *ptr = 12;
 return (ret);
}

void    b3AlignedFreeInternal   (void* ptr,int line,char* filename)
{

 void* real;
 b3g_numAlignedFree++;

 if (ptr) {
   real = *((void **)(ptr)-1);
       int size = *((int*)(ptr)-2);
       b3g_totalBytesAlignedAllocs -= size;

	   b3Printf("free #%d at address %x, from %s,line %d, size %d\n",b3g_numAlignedFree,real, filename,line,size);

   b3s_freeFunc(real);
 } else
 {
	 b3Printf("NULL ptr\n");
 }
}

#else //B3_DEBUG_MEMORY_ALLOCATIONS

void*	b3AlignedAllocInternal	(size_t size, int alignment)
{
	b3g_numAlignedAllocs++;
	void* ptr;
	ptr = b3s_alignedAllocFunc(size, alignment);
//	b3Printf("b3AlignedAllocInternal %d, %x\n",size,ptr);
	return ptr;
}

void	b3AlignedFreeInternal	(void* ptr)
{
	if (!ptr)
	{
		return;
	}

	b3g_numAlignedFree++;
//	b3Printf("b3AlignedFreeInternal %x\n",ptr);
	b3s_alignedFreeFunc(ptr);
}

#endif //B3_DEBUG_MEMORY_ALLOCATIONS

