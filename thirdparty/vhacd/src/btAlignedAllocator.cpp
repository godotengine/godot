/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

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

namespace VHACD {

#ifdef _MSC_VER
#pragma warning(disable:4311 4302)
#endif

int32_t gNumAlignedAllocs = 0;
int32_t gNumAlignedFree = 0;
int32_t gTotalBytesAlignedAllocs = 0; //detect memory leaks

static void* btAllocDefault(size_t size)
{
    return malloc(size);
}

static void btFreeDefault(void* ptr)
{
    free(ptr);
}

static btAllocFunc* sAllocFunc = btAllocDefault;
static btFreeFunc* sFreeFunc = btFreeDefault;

#if defined(BT_HAS_ALIGNED_ALLOCATOR)
#include <malloc.h>
static void* btAlignedAllocDefault(size_t size, int32_t alignment)
{
    return _aligned_malloc(size, (size_t)alignment);
}

static void btAlignedFreeDefault(void* ptr)
{
    _aligned_free(ptr);
}
#elif defined(__CELLOS_LV2__)
#include <stdlib.h>

static inline void* btAlignedAllocDefault(size_t size, int32_t alignment)
{
    return memalign(alignment, size);
}

static inline void btAlignedFreeDefault(void* ptr)
{
    free(ptr);
}
#else
static inline void* btAlignedAllocDefault(size_t size, int32_t alignment)
{
    void* ret;
    char* real;
    unsigned long offset;

    real = (char*)sAllocFunc(size + sizeof(void*) + (alignment - 1));
    if (real) {
        // Synced with Bullet 2.88 to fix GH-27926
        //offset = (alignment - (unsigned long)(real + sizeof(void*))) & (alignment - 1);
        //ret = (void*)((real + sizeof(void*)) + offset);
        ret = btAlignPointer(real + sizeof(void *), alignment);
        *((void**)(ret)-1) = (void*)(real);
    }
    else {
        ret = (void*)(real);
    }
    return (ret);
}

static inline void btAlignedFreeDefault(void* ptr)
{
    void* real;

    if (ptr) {
        real = *((void**)(ptr)-1);
        sFreeFunc(real);
    }
}
#endif

static btAlignedAllocFunc* sAlignedAllocFunc = btAlignedAllocDefault;
static btAlignedFreeFunc* sAlignedFreeFunc = btAlignedFreeDefault;

void btAlignedAllocSetCustomAligned(btAlignedAllocFunc* allocFunc, btAlignedFreeFunc* freeFunc)
{
    sAlignedAllocFunc = allocFunc ? allocFunc : btAlignedAllocDefault;
    sAlignedFreeFunc = freeFunc ? freeFunc : btAlignedFreeDefault;
}

void btAlignedAllocSetCustom(btAllocFunc* allocFunc, btFreeFunc* freeFunc)
{
    sAllocFunc = allocFunc ? allocFunc : btAllocDefault;
    sFreeFunc = freeFunc ? freeFunc : btFreeDefault;
}

#ifdef BT_DEBUG_MEMORY_ALLOCATIONS
//this generic allocator provides the total allocated number of bytes
#include <stdio.h>

void* btAlignedAllocInternal(size_t size, int32_t alignment, int32_t line, char* filename)
{
    void* ret;
    char* real;
    unsigned long offset;

    gTotalBytesAlignedAllocs += size;
    gNumAlignedAllocs++;

    real = (char*)sAllocFunc(size + 2 * sizeof(void*) + (alignment - 1));
    if (real) {
        offset = (alignment - (unsigned long)(real + 2 * sizeof(void*))) & (alignment - 1);
        ret = (void*)((real + 2 * sizeof(void*)) + offset);
        *((void**)(ret)-1) = (void*)(real);
        *((int32_t*)(ret)-2) = size;
    }
    else {
        ret = (void*)(real); //??
    }

    printf("allocation#%d at address %x, from %s,line %d, size %d\n", gNumAlignedAllocs, real, filename, line, size);

    int32_t* ptr = (int32_t*)ret;
    *ptr = 12;
    return (ret);
}

void btAlignedFreeInternal(void* ptr, int32_t line, char* filename)
{

    void* real;
    gNumAlignedFree++;

    if (ptr) {
        real = *((void**)(ptr)-1);
        int32_t size = *((int32_t*)(ptr)-2);
        gTotalBytesAlignedAllocs -= size;

        printf("free #%d at address %x, from %s,line %d, size %d\n", gNumAlignedFree, real, filename, line, size);

        sFreeFunc(real);
    }
    else {
        printf("NULL ptr\n");
    }
}

#else //BT_DEBUG_MEMORY_ALLOCATIONS

void* btAlignedAllocInternal(size_t size, int32_t alignment)
{
    gNumAlignedAllocs++;
    void* ptr;
    ptr = sAlignedAllocFunc(size, alignment);
    //	printf("btAlignedAllocInternal %d, %x\n",size,ptr);
    return ptr;
}

void btAlignedFreeInternal(void* ptr)
{
    if (!ptr) {
        return;
    }

    gNumAlignedFree++;
    //	printf("btAlignedFreeInternal %x\n",ptr);
    sAlignedFreeFunc(ptr);
}

}; // namespace VHACD

#endif //BT_DEBUG_MEMORY_ALLOCATIONS
