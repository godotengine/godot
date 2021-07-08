//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// xxHash Fuzzer test:
//      Integration with Chromium's libfuzzer for xxHash.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "xxhash.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
#if !defined(XXH_NO_LONG_LONG)
    // Test 64-bit hash.
    size_t seedSize64 = sizeof(unsigned long long);
    if (size < seedSize64)
    {
        XXH64(data, size, 0ull);
    }
    else
    {
        unsigned long long seed64;
        memcpy(&seed64, data, seedSize64);
        XXH64(&data[seedSize64], size - seedSize64, seed64);
    }
#endif  // !defined(XXH_NO_LONG_LONG)

    // Test 32-bit hash.
    size_t seedSize32 = sizeof(unsigned int);
    if (size < seedSize32)
    {
        XXH64(data, size, 0ull);
    }
    else
    {
        unsigned long long seed32;
        memcpy(&seed32, data, seedSize32);
        XXH32(&data[seedSize32], size - seedSize32, seed32);
    }

    return 0;
}
