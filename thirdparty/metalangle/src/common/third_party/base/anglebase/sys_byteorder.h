//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// sys_byteorder.h: Compatiblity hacks for importing Chromium's base/SHA1.

#ifndef ANGLEBASE_SYS_BYTEORDER_H_
#define ANGLEBASE_SYS_BYTEORDER_H_

namespace angle
{

namespace base
{

// Returns a value with all bytes in |x| swapped, i.e. reverses the endianness.
inline uint16_t ByteSwap(uint16_t x)
{
#if defined(_MSC_VER)
    return _byteswap_ushort(x);
#else
    return __builtin_bswap16(x);
#endif
}

inline uint32_t ByteSwap(uint32_t x)
{
#if defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return __builtin_bswap32(x);
#endif
}

inline uint64_t ByteSwap(uint64_t x)
{
#if defined(_MSC_VER)
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}

}  // namespace base

}  // namespace angle

#endif  // ANGLEBASE_SYS_BYTEORDER_H_