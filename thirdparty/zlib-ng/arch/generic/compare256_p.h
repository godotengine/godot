/* compare256_p.h -- 256 byte memory comparison with match length return
 * Copyright (C) 2020 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zmemory.h"
#include "deflate.h"
#include "fallback_builtins.h"

/* 8-bit integer comparison */
static inline uint32_t compare256_8(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
        if (*src0 != *src1)
            return len;
        src0 += 1, src1 += 1, len += 1;
    } while (len < 256);

    return 256;
}

/* 16-bit integer comparison */
static inline uint32_t compare256_16(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;

        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;

        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;

        if (zng_memcmp_2(src0, src1) != 0)
            return len + (*src0 == *src1);
        src0 += 2, src1 += 2, len += 2;
    } while (len < 256);

    return 256;
}

#ifdef HAVE_BUILTIN_CTZ
/* 32-bit integer comparison */
static inline uint32_t compare256_32(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        uint32_t sv, mv, diff;

        sv = zng_memread_4(src0);
        mv = zng_memread_4(src1);

        diff = sv ^ mv;
        if (diff) {
#  if BYTE_ORDER == LITTLE_ENDIAN
            uint32_t match_byte = __builtin_ctz(diff) / 8;
#  else
            uint32_t match_byte = __builtin_clz(diff) / 8;
#  endif
            return len + match_byte;
        }

        src0 += 4, src1 += 4, len += 4;
    } while (len < 256);

    return 256;
}
#endif

#ifdef HAVE_BUILTIN_CTZLL
/* 64-bit integer comparison */
static inline uint32_t compare256_64(const uint8_t *src0, const uint8_t *src1) {
    uint32_t len = 0;

    do {
        uint64_t sv, mv, diff;

        sv = zng_memread_8(src0);
        mv = zng_memread_8(src1);

        diff = sv ^ mv;
        if (diff) {
#  if BYTE_ORDER == LITTLE_ENDIAN
            uint64_t match_byte = __builtin_ctzll(diff) / 8;
#  else
            uint64_t match_byte = __builtin_clzll(diff) / 8;
#  endif
            return len + (uint32_t)match_byte;
        }

        src0 += 8, src1 += 8, len += 8;
    } while (len < 256);

    return 256;
}
#endif
