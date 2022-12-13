// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2015, SIL International, All rights reserved.


#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>

namespace
{

#if defined(_MSC_VER)
typedef unsigned __int8 u8;
typedef unsigned __int16 u16;
typedef unsigned __int32 u32;
typedef unsigned __int64 u64;
#else
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#endif

ptrdiff_t const     MINMATCH = 4,
                    LASTLITERALS = 5,
                    MINCODA  = LASTLITERALS+1,
                    MINSRCSIZE = 13;

template<int S>
inline
void unaligned_copy(void * d, void const * s) {
  ::memcpy(d, s, S);
}

inline
size_t align(size_t p) {
    return (p + sizeof(unsigned long)-1) & ~(sizeof(unsigned long)-1);
}

inline
u8 * safe_copy(u8 * d, u8 const * s, size_t n) {
    while (n--) *d++ = *s++;
    return d;
}

inline
u8 * overrun_copy(u8 * d, u8 const * s, size_t n) {
    size_t const WS = sizeof(unsigned long);
    u8 const * e = s + n;
    do
    {
        unaligned_copy<WS>(d, s);
        d += WS;
        s += WS;
    }
    while (s < e);
    d-=(s-e);

    return d;
}


inline
u8 * fast_copy(u8 * d, u8 const * s, size_t n) {
    size_t const WS = sizeof(unsigned long);
    size_t wn = n/WS;
    while (wn--)
    {
        unaligned_copy<WS>(d, s);
        d += WS;
        s += WS;
    }
    n &= WS-1;
    return safe_copy(d, s, n);
}


} // end of anonymous namespace
