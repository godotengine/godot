/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "functable.h"
#include "adler32_p.h"

/* ========================================================================= */
Z_INTERNAL uint32_t adler32_c(uint32_t adler, const uint8_t *buf, size_t len) {
    uint32_t sum2;
    unsigned n;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (UNLIKELY(len == 1))
        return adler32_len_1(adler, buf, sum2);

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (UNLIKELY(buf == NULL))
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (UNLIKELY(len < 16))
        return adler32_len_16(adler, buf, len, sum2);

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
#ifdef UNROLL_MORE
        n = NMAX / 16;          /* NMAX is divisible by 16 */
#else
        n = NMAX / 8;           /* NMAX is divisible by 8 */
#endif
        do {
#ifdef UNROLL_MORE
            ADLER_DO16(adler, sum2, buf);          /* 16 sums unrolled */
            buf += 16;
#else
            ADLER_DO8(adler, sum2, buf, 0);         /* 8 sums unrolled */
            buf += 8;
#endif
        } while (--n);
        adler %= BASE;
        sum2 %= BASE;
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    return adler32_len_64(adler, buf, len, sum2);
}
