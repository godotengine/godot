// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2015, SIL International, All rights reserved.

#include <cassert>

#include "inc/Decompressor.h"
#include "inc/Compression.h"

using namespace lz4;

namespace {

inline
u32 read_literal(u8 const * &s, u8 const * const e, u32 l) {
    if (l == 15 && s != e)
    {
        u8 b = 0;
        do { l += b = *s++; } while(b==0xff && s != e);
    }
    return l;
}

bool read_sequence(u8 const * &src, u8 const * const end, u8 const * &literal,
                    u32 & literal_len, u32 & match_len, u32 & match_dist)
{
    u8 const token = *src++;

    literal_len = read_literal(src, end, token >> 4);
    literal = src;
    src += literal_len;

    // Normal exit for end of stream, wrap arround check and parital match check.
    if (src > end - sizeof(u16) || src < literal)
        return false;

    match_dist  = *src++;
    match_dist |= *src++ << 8;
    match_len = read_literal(src, end, token & 0xf) + MINMATCH;

    // Malformed stream check.
    return src <= end-MINCODA;
}

}

int lz4::decompress(void const *in, size_t in_size, void *out, size_t out_size)
{
    if (out_size <= in_size || in_size < MINSRCSIZE)
        return -1;

    u8 const *       src     = static_cast<u8 const *>(in),
             *       literal = 0,
             * const src_end = src + in_size;

    u8 *       dst     = static_cast<u8*>(out),
       * const dst_end = dst + out_size;

    // Check the in and out size hasn't wrapped around.
    if (src >= src_end || dst >= dst_end)
        return -1;

    u32 literal_len = 0,
        match_len = 0,
        match_dist = 0;

    while (read_sequence(src, src_end, literal, literal_len, match_len,
                         match_dist))
    {
        if (literal_len != 0)
        {
            // Copy in literal. At this point the a minimal literal + minminal
            // match plus the coda (1 + 2 + 5) must be 8 bytes or more allowing
            // us to remain within the src buffer for an overrun_copy on
            // machines upto 64 bits.
            if (align(literal_len) > out_size)
                return -1;
            dst = overrun_copy(dst, literal, literal_len);
            out_size -= literal_len;
        }

        // Copy, possibly repeating, match from earlier in the
        //  decoded output.
        u8 const * const pcpy = dst - match_dist;
        if (pcpy < static_cast<u8*>(out)
              || match_len > unsigned(out_size - LASTLITERALS)
              // Wrap around checks:
              || out_size < LASTLITERALS || pcpy >= dst)
            return -1;
        if (dst > pcpy+sizeof(unsigned long)
            && align(match_len) <= out_size)
            dst = overrun_copy(dst, pcpy, match_len);
        else
            dst = safe_copy(dst, pcpy, match_len);
        out_size -= match_len;
    }

    if (literal > src_end - literal_len || literal_len > out_size)
        return -1;
    dst = fast_copy(dst, literal, literal_len);

    return int(dst - (u8*)out);
}
