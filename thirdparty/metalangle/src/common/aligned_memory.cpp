//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// aligned_memory: An aligned memory allocator. Based on Chrome's base/memory/aligned_memory.
//

#include "common/aligned_memory.h"

#include "common/debug.h"
#include "common/platform.h"

#if defined(COMPILER_MSVC)
#    include <malloc.h>
#else
#    include <stdlib.h>
#endif

namespace angle
{

void *AlignedAlloc(size_t size, size_t alignment)
{
    ASSERT(size > 0);
    ASSERT((alignment & (alignment - 1)) == 0);
    ASSERT((alignment % sizeof(void *)) == 0);
    void *ptr = nullptr;
#if defined(ANGLE_PLATFORM_WINDOWS)
    ptr = _aligned_malloc(size, alignment);
// Android technically supports posix_memalign(), but does not expose it in
// the current version of the library headers used by Chrome.  Luckily,
// memalign() on Android returns pointers which can safely be used with
// free(), so we can use it instead.  Issue filed to document this:
// http://code.google.com/p/android/issues/detail?id=35391
#elif defined(ANGLE_PLATFORM_ANDROID)
    ptr = memalign(alignment, size);
#else
    if (posix_memalign(&ptr, alignment, size))
        ptr = nullptr;
#endif
    // Since aligned allocations may fail for non-memory related reasons, force a
    // crash if we encounter a failed allocation.
    if (!ptr)
    {
        ERR() << "If you crashed here, your aligned allocation is incorrect: "
              << "size=" << size << ", alignment=" << alignment;
        ASSERT(false);
    }
    // Sanity check alignment just to be safe.
    ASSERT((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0);
    return ptr;
}

void AlignedFree(void *ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

}  // namespace angle
