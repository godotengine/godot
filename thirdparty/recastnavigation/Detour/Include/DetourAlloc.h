//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef DETOURALLOCATOR_H
#define DETOURALLOCATOR_H

#include <stddef.h>

/// Provides hint values to the memory allocator on how long the
/// memory is expected to be used.
enum dtAllocHint
{
	DT_ALLOC_PERM,		///< Memory persist after a function call.
	DT_ALLOC_TEMP		///< Memory used temporarily within a function.
};

/// A memory allocation function.
//  @param[in]		size			The size, in bytes of memory, to allocate.
//  @param[in]		rcAllocHint	A hint to the allocator on how long the memory is expected to be in use.
//  @return A pointer to the beginning of the allocated memory block, or null if the allocation failed.
///  @see dtAllocSetCustom
typedef void* (dtAllocFunc)(size_t size, dtAllocHint hint);

/// A memory deallocation function.
///  @param[in]		ptr		A pointer to a memory block previously allocated using #dtAllocFunc.
/// @see dtAllocSetCustom
typedef void (dtFreeFunc)(void* ptr);

/// Sets the base custom allocation functions to be used by Detour.
///  @param[in]		allocFunc	The memory allocation function to be used by #dtAlloc
///  @param[in]		freeFunc	The memory de-allocation function to be used by #dtFree
void dtAllocSetCustom(dtAllocFunc *allocFunc, dtFreeFunc *freeFunc);

/// Allocates a memory block.
///  @param[in]		size	The size, in bytes of memory, to allocate.
///  @param[in]		hint	A hint to the allocator on how long the memory is expected to be in use.
///  @return A pointer to the beginning of the allocated memory block, or null if the allocation failed.
/// @see dtFree
void* dtAlloc(size_t size, dtAllocHint hint);

/// Deallocates a memory block.
///  @param[in]		ptr		A pointer to a memory block previously allocated using #dtAlloc.
/// @see dtAlloc
void dtFree(void* ptr);

#endif
