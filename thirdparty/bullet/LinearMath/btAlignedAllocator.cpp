/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btAlignedAllocator.h"
#include <string.h>

#ifdef BT_DEBUG_MEMORY_ALLOCATIONS
int gNumAlignedAllocs = 0;
int gNumAlignedFree = 0;
int gTotalBytesAlignedAllocs = 0;  //detect memory leaks
#endif                             //BT_DEBUG_MEMORY_ALLOCATIONST_DEBUG_ALLOCATIONS

static void *btAllocDefault(size_t size)
{
  char* data = (char*) malloc(size);
  memset(data,0,size);//keep msan happy
  return data;
}

static void btFreeDefault(void *ptr)
{
	free(ptr);
}

static btAllocFunc *sAllocFunc = btAllocDefault;
static btFreeFunc *sFreeFunc = btFreeDefault;

#if defined(BT_HAS_ALIGNED_ALLOCATOR)
#include <malloc.h>
static void *btAlignedAllocDefault(size_t size, int alignment)
{
	return _aligned_malloc(size, (size_t)alignment);
}

static void btAlignedFreeDefault(void *ptr)
{
	_aligned_free(ptr);
}
#elif defined(__CELLOS_LV2__)
#include <stdlib.h>

static inline void *btAlignedAllocDefault(size_t size, int alignment)
{
	return memalign(alignment, size);
}

static inline void btAlignedFreeDefault(void *ptr)
{
	free(ptr);
}
#else

static inline void *btAlignedAllocDefault(size_t size, int alignment)
{
	void *ret;
	char *real;
	real = (char *)sAllocFunc(size + sizeof(void *) + (alignment - 1));
	if (real)
	{
		ret = btAlignPointer(real + sizeof(void *), alignment);
		*((void **)(ret)-1) = (void *)(real);
	}
	else
	{
		ret = (void *)(real);
	}
  //keep msan happy
  memset((char*) ret, 0, size);
	return (ret);
}

static inline void btAlignedFreeDefault(void *ptr)
{
	void *real;

	if (ptr)
	{
		real = *((void **)(ptr)-1);
		sFreeFunc(real);
	}
}
#endif

static btAlignedAllocFunc *sAlignedAllocFunc = btAlignedAllocDefault;
static btAlignedFreeFunc *sAlignedFreeFunc = btAlignedFreeDefault;

void btAlignedAllocSetCustomAligned(btAlignedAllocFunc *allocFunc, btAlignedFreeFunc *freeFunc)
{
	sAlignedAllocFunc = allocFunc ? allocFunc : btAlignedAllocDefault;
	sAlignedFreeFunc = freeFunc ? freeFunc : btAlignedFreeDefault;
}

void btAlignedAllocSetCustom(btAllocFunc *allocFunc, btFreeFunc *freeFunc)
{
	sAllocFunc = allocFunc ? allocFunc : btAllocDefault;
	sFreeFunc = freeFunc ? freeFunc : btFreeDefault;
}

#ifdef BT_DEBUG_MEMORY_ALLOCATIONS

static int allocations_id[10241024];
static int allocations_bytes[10241024];
static int mynumallocs = 0;
#include <stdio.h>

int btDumpMemoryLeaks()
{
	int totalLeak = 0;

	for (int i = 0; i < mynumallocs; i++)
	{
		printf("Error: leaked memory of allocation #%d (%d bytes)\n", allocations_id[i], allocations_bytes[i]);
		totalLeak += allocations_bytes[i];
	}
	if (totalLeak)
	{
		printf("Error: memory leaks: %d allocations were not freed and leaked together %d bytes\n", mynumallocs, totalLeak);
	}
	return totalLeak;
}
//this generic allocator provides the total allocated number of bytes
#include <stdio.h>

struct btDebugPtrMagic
{
	union {
		void **vptrptr;
		void *vptr;
		int *iptr;
		char *cptr;
	};
};

void *btAlignedAllocInternal(size_t size, int alignment, int line, const char *filename)
{
	if (size == 0)
	{
		printf("Whaat? size==0");
		return 0;
	}
	static int allocId = 0;

	void *ret;
	char *real;

	// to find some particular memory leak, you could do something like this:
	//	if (allocId==172)
	//	{
	//		printf("catch me!\n");
	//	}
	//	if (size>1024*1024)
	//	{
	//		printf("big alloc!%d\n", size);
	//	}

	gTotalBytesAlignedAllocs += size;
	gNumAlignedAllocs++;

	int sz4prt = 4 * sizeof(void *);

	real = (char *)sAllocFunc(size + sz4prt + (alignment - 1));
	if (real)
	{
		ret = (void *)btAlignPointer(real + sz4prt, alignment);
		btDebugPtrMagic p;
		p.vptr = ret;
		p.cptr -= sizeof(void *);
		*p.vptrptr = (void *)real;
		p.cptr -= sizeof(void *);
		*p.iptr = size;
		p.cptr -= sizeof(void *);
		*p.iptr = allocId;

		allocations_id[mynumallocs] = allocId;
		allocations_bytes[mynumallocs] = size;
		mynumallocs++;
	}
	else
	{
		ret = (void *)(real);  //??
	}

	printf("allocation %d at address %x, from %s,line %d, size %d (total allocated = %d)\n", allocId, real, filename, line, size, gTotalBytesAlignedAllocs);
	allocId++;

	int *ptr = (int *)ret;
	*ptr = 12;
	return (ret);
}

void btAlignedFreeInternal(void *ptr, int line, const char *filename)
{
	void *real;

	if (ptr)
	{
		gNumAlignedFree++;

		btDebugPtrMagic p;
		p.vptr = ptr;
		p.cptr -= sizeof(void *);
		real = *p.vptrptr;
		p.cptr -= sizeof(void *);
		int size = *p.iptr;
		p.cptr -= sizeof(void *);
		int allocId = *p.iptr;

		bool found = false;

		for (int i = 0; i < mynumallocs; i++)
		{
			if (allocations_id[i] == allocId)
			{
				allocations_id[i] = allocations_id[mynumallocs - 1];
				allocations_bytes[i] = allocations_bytes[mynumallocs - 1];
				mynumallocs--;
				found = true;
				break;
			}
		}

		gTotalBytesAlignedAllocs -= size;

		int diff = gNumAlignedAllocs - gNumAlignedFree;
		printf("free %d at address %x, from %s,line %d, size %d (total remain = %d in %d non-freed allocations)\n", allocId, real, filename, line, size, gTotalBytesAlignedAllocs, diff);

		sFreeFunc(real);
	}
	else
	{
		//printf("deleting a NULL ptr, no effect\n");
	}
}

#else  //BT_DEBUG_MEMORY_ALLOCATIONS

void *btAlignedAllocInternal(size_t size, int alignment)
{
	void *ptr;
	ptr = sAlignedAllocFunc(size, alignment);
	//	printf("btAlignedAllocInternal %d, %x\n",size,ptr);
	return ptr;
}

void btAlignedFreeInternal(void *ptr)
{
	if (!ptr)
	{
		return;
	}

	//	printf("btAlignedFreeInternal %x\n",ptr);
	sAlignedFreeFunc(ptr);
}

#endif  //BT_DEBUG_MEMORY_ALLOCATIONS
