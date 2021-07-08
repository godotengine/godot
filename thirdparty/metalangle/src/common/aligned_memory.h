//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// aligned_memory: An aligned memory allocator. Based on Chrome's base/memory/aligned_memory.
//

#ifndef COMMON_ALIGNED_MEMORY_H_
#define COMMON_ALIGNED_MEMORY_H_

#include <cstddef>

namespace angle
{

// This can be replaced with std::aligned_malloc when we have C++17.
void *AlignedAlloc(size_t size, size_t alignment);
void AlignedFree(void *ptr);

}  // namespace angle

#endif  // COMMON_ALIGNED_MEMORY_H_
