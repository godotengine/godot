//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// hash_utils.h: Hashing based helper functions.

#ifndef COMMON_HASHUTILS_H_
#define COMMON_HASHUTILS_H_

#include "common/debug.h"
#include "common/third_party/xxhash/xxhash.h"

namespace angle
{
// Computes a hash of "key". Any data passed to this function must be multiples of
// 4 bytes, since the PMurHash32 method can only operate increments of 4-byte words.
inline std::size_t ComputeGenericHash(const void *key, size_t keySize)
{
    static constexpr unsigned int kSeed = 0xABCDEF98;

    // We can't support "odd" alignments.  ComputeGenericHash requires aligned types
    ASSERT(keySize % 4 == 0);
#if defined(ANGLE_IS_64_BIT_CPU)
    return XXH64(key, keySize, kSeed);
#else
    return XXH32(key, keySize, kSeed);
#endif  // defined(ANGLE_IS_64_BIT_CPU)
}

template <typename T>
std::size_t ComputeGenericHash(const T &key)
{
    static_assert(sizeof(key) % 4 == 0, "ComputeGenericHash requires aligned types");
    return ComputeGenericHash(&key, sizeof(key));
}
}  // namespace angle

#endif  // COMMON_HASHUTILS_H_
