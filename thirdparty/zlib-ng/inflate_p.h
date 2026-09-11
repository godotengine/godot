/* inflate_p.h -- Private inline functions and macros shared with more than one deflate method
 *
 */

#ifndef INFLATE_P_H
#define INFLATE_P_H

#include <stdlib.h>
#include "zmemory.h"

/* Architecture-specific hooks. */
#ifdef S390_DFLTCC_INFLATE
#  include "arch/s390/dfltcc_inflate.h"
/* DFLTCC instructions require window to be page-aligned */
#  define PAD_WINDOW            PAD_4096
#  define WINDOW_PAD_SIZE       4096
#  define HINT_ALIGNED_WINDOW   HINT_ALIGNED_4096
#else
#  define PAD_WINDOW            PAD_64
#  define WINDOW_PAD_SIZE       64
#  define HINT_ALIGNED_WINDOW   HINT_ALIGNED_64
/* Adjust the window size for the arch-specific inflate code. */
#  define INFLATE_ADJUST_WINDOW_SIZE(n) (n)
/* Invoked at the end of inflateResetKeep(). Useful for initializing arch-specific extension blocks. */
#  define INFLATE_RESET_KEEP_HOOK(strm) do {} while (0)
/* Invoked at the beginning of inflatePrime(). Useful for updating arch-specific buffers. */
#  define INFLATE_PRIME_HOOK(strm, bits, value) do {} while (0)
/* Invoked at the beginning of each block. Useful for plugging arch-specific inflation code. */
#  define INFLATE_TYPEDO_HOOK(strm, flush) do {} while (0)
/* Returns whether zlib-ng should compute a checksum. Set to 0 if arch-specific inflation code already does that. */
#  define INFLATE_NEED_CHECKSUM(strm) 1
/* Returns whether zlib-ng should update a window. Set to 0 if arch-specific inflation code already does that. */
#  define INFLATE_NEED_UPDATEWINDOW(strm) 1
/* Invoked at the beginning of inflateMark(). Useful for updating arch-specific pointers and offsets. */
#  define INFLATE_MARK_HOOK(strm) do {} while (0)
/* Invoked at the beginning of inflateSyncPoint(). Useful for performing arch-specific state checks. */
#  define INFLATE_SYNC_POINT_HOOK(strm) do {} while (0)
/* Invoked at the beginning of inflateSetDictionary(). Useful for checking arch-specific window data. */
#  define INFLATE_SET_DICTIONARY_HOOK(strm, dict, dict_len) do {} while (0)
/* Invoked at the beginning of inflateGetDictionary(). Useful for adjusting arch-specific window data. */
#  define INFLATE_GET_DICTIONARY_HOOK(strm, dict, dict_len) do {} while (0)
#endif

/*
 *   Macros shared by inflate() and inflateBack()
 */

/* check function to use adler32() for zlib or crc32() for gzip */
#ifdef GUNZIP
#  define UPDATE(check, buf, len) \
    (state->flags ? PREFIX(crc32)(check, buf, len) : FUNCTABLE_CALL(adler32)(check, buf, len))
#else
#  define UPDATE(check, buf, len) FUNCTABLE_CALL(adler32)(check, buf, len)
#endif

/* check macros for header crc */
#ifdef GUNZIP
#  define CRC2(check, word) \
    do { \
        hbuf[0] = (unsigned char)(word); \
        hbuf[1] = (unsigned char)((word) >> 8); \
        check = PREFIX(crc32)(check, hbuf, 2); \
    } while (0)

#  define CRC4(check, word) \
    do { \
        hbuf[0] = (unsigned char)(word); \
        hbuf[1] = (unsigned char)((word) >> 8); \
        hbuf[2] = (unsigned char)((word) >> 16); \
        hbuf[3] = (unsigned char)((word) >> 24); \
        check = PREFIX(crc32)(check, hbuf, 4); \
    } while (0)
#endif

/* Load registers with state in inflate() for speed */
#define LOAD() \
    do { \
        put = strm->next_out; \
        left = strm->avail_out; \
        next = strm->next_in; \
        have = strm->avail_in; \
        hold = state->hold; \
        bits = state->bits; \
    } while (0)

/* Restore state from registers in inflate() */
#define RESTORE() \
    do { \
        strm->next_out = put; \
        strm->avail_out = left; \
        strm->next_in = (z_const unsigned char *)next; \
        strm->avail_in = have; \
        state->hold = hold; \
        state->bits = bits; \
    } while (0)

/* Clear the input bit accumulator */
#define INITBITS() \
    do { \
        hold = 0; \
        bits = 0; \
    } while (0)

/* Ensure that there is at least n bits in the bit accumulator.  If there is
   not enough available input to do that, then return from inflate()/inflateBack(). */
#define NEEDBITS(n) \
    do { \
        while (bits < (unsigned)(n)) \
            PULLBYTE(); \
    } while (0)

/* Return the low n bits of the bit accumulator (n < 16) */
#define BITS(n) \
    (hold & ((1U << (unsigned)(n)) - 1))

/* Remove n bits from the bit accumulator */
#define DROPBITS(n) \
    do { \
        hold >>= (n); \
        bits -= (unsigned)(n); \
    } while (0)

/* Remove zero to seven bits as needed to go to a byte boundary */
#define BYTEBITS() \
    do { \
        hold >>= bits & 7; \
        bits -= bits & 7; \
    } while (0)

/* Set mode=BAD and prepare error message */
#define SET_BAD(errmsg) \
    do { \
        state->mode = BAD; \
        strm->msg = (char *)errmsg; \
    } while (0)

#define INFLATE_FAST_MIN_HAVE 15
#define INFLATE_FAST_MIN_LEFT 260

/* Load 64 bits from IN and place the bytes at offset BITS in the result. */
static inline uint64_t load_64_bits(const unsigned char *in, unsigned bits) {
    uint64_t chunk = zng_memread_8(in);

#if BYTE_ORDER == LITTLE_ENDIAN
    return chunk << bits;
#else
    return ZSWAP64(chunk) << bits;
#endif
}

/* Behave like chunkcopy, but avoid writing beyond of legal output. */
static inline uint8_t* chunkcopy_safe(uint8_t *out, uint8_t *from, uint64_t len, uint8_t *safe) {
    uint64_t safelen = safe - out;
    len = MIN(len, safelen);
    int32_t olap_src = from >= out && from < out + len;
    int32_t olap_dst = out >= from && out < from + len;
    uint64_t tocopy;

    /* For all cases without overlap, memcpy is ideal */
    if (!(olap_src || olap_dst)) {
        memcpy(out, from, (size_t)len);
        return out + len;
    }

    /* Complete overlap: Source == destination */
    if (out == from) {
        return out + len;
    }

    /* We are emulating a self-modifying copy loop here. To do this in a way that doesn't produce undefined behavior,
     * we have to get a bit clever. First if the overlap is such that src falls between dst and dst+len, we can do the
     * initial bulk memcpy of the nonoverlapping region. Then, we can leverage the size of this to determine the safest
     * atomic memcpy size we can pick such that we have non-overlapping regions. This effectively becomes a safe look
     * behind or lookahead distance. */
    uint64_t non_olap_size = llabs(from - out); // llabs vs labs for compatibility with windows

    /* So this doesn't give use a worst case scenario of function calls in a loop,
     * we want to instead break this down into copy blocks of fixed lengths
     *
     * TODO: The memcpy calls aren't inlined on architectures with strict memory alignment
     */
    while (len) {
        tocopy = MIN(non_olap_size, len);
        len -= tocopy;

        while (tocopy >= 16) {
            memcpy(out, from, 16);
            out += 16;
            from += 16;
            tocopy -= 16;
        }

        if (tocopy >= 8) {
            memcpy(out, from, 8);
            out += 8;
            from += 8;
            tocopy -= 8;
        }

        if (tocopy >= 4) {
            memcpy(out, from, 4);
            out += 4;
            from += 4;
            tocopy -= 4;
        }

        while (tocopy--) {
            *out++ = *from++;
        }
    }

    return out;
}

#endif
