/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "functable.h"
#include "adler32_p.h"

#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(adler32_z)(unsigned long adler, const unsigned char *buf, size_t len) {
    return (unsigned long)FUNCTABLE_CALL(adler32)((uint32_t)adler, buf, len);
}
#else
uint32_t Z_EXPORT PREFIX(adler32_z)(uint32_t adler, const unsigned char *buf, size_t len) {
    return FUNCTABLE_CALL(adler32)(adler, buf, len);
}
#endif

/* ========================================================================= */
#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(adler32)(unsigned long adler, const unsigned char *buf, unsigned int len) {
    return (unsigned long)FUNCTABLE_CALL(adler32)((uint32_t)adler, buf, len);
}
#else
uint32_t Z_EXPORT PREFIX(adler32)(uint32_t adler, const unsigned char *buf, uint32_t len) {
    return FUNCTABLE_CALL(adler32)(adler, buf, len);
}
#endif

/* ========================================================================= */
static uint32_t adler32_combine_(uint32_t adler1, uint32_t adler2, z_off64_t len2) {
    uint32_t sum1;
    uint32_t sum2;
    unsigned rem;

    /* for negative len, return invalid adler32 as a clue for debugging */
    if (len2 < 0)
        return 0xffffffff;

    /* the derivation of this formula is left as an exercise for the reader */
    len2 %= BASE;                 /* assumes len2 >= 0 */
    rem = (unsigned)len2;
    sum1 = adler1 & 0xffff;
    sum2 = rem * sum1;
    sum2 %= BASE;
    sum1 += (adler2 & 0xffff) + BASE - 1;
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + BASE - rem;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum2 >= ((unsigned long)BASE << 1)) sum2 -= ((unsigned long)BASE << 1);
    if (sum2 >= BASE) sum2 -= BASE;
    return sum1 | (sum2 << 16);
}

/* ========================================================================= */
#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(adler32_combine)(unsigned long adler1, unsigned long adler2, z_off_t len2) {
    return (unsigned long)adler32_combine_((uint32_t)adler1, (uint32_t)adler2, len2);
}

unsigned long Z_EXPORT PREFIX4(adler32_combine)(unsigned long adler1, unsigned long adler2, z_off64_t len2) {
    return (unsigned long)adler32_combine_((uint32_t)adler1, (uint32_t)adler2, len2);
}
#else
uint32_t Z_EXPORT PREFIX4(adler32_combine)(uint32_t adler1, uint32_t adler2, z_off64_t len2) {
    return adler32_combine_(adler1, adler2, len2);
}
#endif
