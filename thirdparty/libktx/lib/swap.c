/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* $Id$ */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <KHR/khrplatform.h>
#include "ktx.h"

/*
 * SwapEndian16: Swaps endianness in an array of 16-bit values
 */
void
_ktxSwapEndian16(khronos_uint16_t* pData16, ktx_size_t count)
{
    for (ktx_size_t i = 0; i < count; ++i)
    {
        khronos_uint16_t x = *pData16;
        *pData16++ = (x << 8) | (x >> 8);
    }
}

/*
 * SwapEndian32: Swaps endianness in an array of 32-bit values
 */
void
_ktxSwapEndian32(khronos_uint32_t* pData32, ktx_size_t count)
{
    for (ktx_size_t i = 0; i < count; ++i)
    {
        khronos_uint32_t x = *pData32;
        *pData32++ = (x << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | (x >> 24);
    }
}

/*
 * SwapEndian364: Swaps endianness in an array of 32-bit values
 */
void
_ktxSwapEndian64(khronos_uint64_t* pData64, ktx_size_t count)
{
    for (ktx_size_t i = 0; i < count; ++i)
    {
        khronos_uint64_t x = *pData64;
        *pData64++ = (x << 56) | ((x & 0xFF00) << 40) | ((x & 0xFF0000) << 24)
                     | ((x & 0xFF000000) << 8 ) | ((x & 0xFF00000000) >> 8)
                     | ((x & 0xFF0000000000) >> 24)
                     | ((x & 0xFF000000000000) << 40) | (x >> 56);
    }
}



