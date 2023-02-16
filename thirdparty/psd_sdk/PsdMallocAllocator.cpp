// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdMallocAllocator.h"

#if defined(__APPLE__)
#include <stdlib.h>
#include <errno.h>
#else
#include <malloc.h>
#endif


PSD_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void* MallocAllocator::DoAllocate(size_t size, size_t alignment)
{
#if defined(__APPLE__)
    void *m = 0;
    size_t minAlignment = sizeof(void *);
    while (alignment > minAlignment) {
        minAlignment *= 2;
    }
    errno = posix_memalign(&m, minAlignment, size);
    return errno ? NULL : m;
#elif defined(__GNUG__)
	return memalign(alignment, size);
#else
	return _aligned_malloc(size, alignment);
#endif
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void MallocAllocator::DoFree(void* ptr)
{
#if defined(__APPLE__) || defined(__GNUG__)
	free(ptr);
#else
	_aligned_free(ptr);
#endif
}

PSD_NAMESPACE_END
