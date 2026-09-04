/* chunkset_tpl.h -- inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include <stdlib.h>

/* Returns the chunk size */
static inline size_t CHUNKSIZE(void) {
    return sizeof(chunk_t);
}

/* Behave like memcpy, but assume that it's OK to overwrite at least
   chunk_t bytes of output even if the length is shorter than this,
   that the length is non-zero, and that `from` lags `out` by at least
   sizeof chunk_t bytes (or that they don't overlap at all or simply that
   the distance is less than the length of the copy).

   Aside from better memory bus utilisation, this means that short copies
   (chunk_t bytes or fewer) will fall straight through the loop
   without iteration, which will hopefully make the branch prediction more
   reliable. */
#ifndef HAVE_CHUNKCOPY
static inline uint8_t* CHUNKCOPY(uint8_t *out, uint8_t const *from, unsigned len) {
    Assert(len > 0, "chunkcopy should never have a length 0");
    chunk_t chunk;
    int32_t align = ((len - 1) % sizeof(chunk_t)) + 1;
    loadchunk(from, &chunk);
    storechunk(out, &chunk);
    out += align;
    from += align;
    len -= align;
    while (len > 0) {
        loadchunk(from, &chunk);
        storechunk(out, &chunk);
        out += sizeof(chunk_t);
        from += sizeof(chunk_t);
        len -= sizeof(chunk_t);
    }
    return out;
}
#endif

/* Perform short copies until distance can be rewritten as being at least
   sizeof chunk_t.

   This assumes that it's OK to overwrite at least the first
   2*sizeof(chunk_t) bytes of output even if the copy is shorter than this.
   This assumption holds because inflate_fast() starts every iteration with at
   least 258 bytes of output space available (258 being the maximum length
   output from a single token; see inflate_fast()'s assumptions below). */
#ifndef HAVE_CHUNKUNROLL
static inline uint8_t* CHUNKUNROLL(uint8_t *out, unsigned *dist, unsigned *len) {
    unsigned char const *from = out - *dist;
    chunk_t chunk;
    while (*dist < *len && *dist < sizeof(chunk_t)) {
        loadchunk(from, &chunk);
        storechunk(out, &chunk);
        out += *dist;
        *len -= *dist;
        *dist += *dist;
    }
    return out;
}
#endif

#ifndef HAVE_CHUNK_MAG
/* Loads a magazine to feed into memory of the pattern */
static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
        /* This code takes string of length dist from "from" and repeats
         * it for as many times as can fit in a chunk_t (vector register) */
        uint64_t cpy_dist;
        uint64_t bytes_remaining = sizeof(chunk_t);
        chunk_t chunk_load;
        uint8_t *cur_chunk = (uint8_t *)&chunk_load;
        while (bytes_remaining) {
            cpy_dist = MIN(dist, bytes_remaining);
            memcpy(cur_chunk, buf, (size_t)cpy_dist);
            bytes_remaining -= cpy_dist;
            cur_chunk += cpy_dist;
            /* This allows us to bypass an expensive integer division since we're effectively
             * counting in this loop, anyway */
            *chunk_rem = (uint32_t)cpy_dist;
        }

        return chunk_load;
}
#endif

#if defined(HAVE_HALF_CHUNK) && !defined(HAVE_HALFCHUNKCOPY)
static inline uint8_t* HALFCHUNKCOPY(uint8_t *out, uint8_t const *from, unsigned len) {
    halfchunk_t chunk;
    int32_t align = ((len - 1) % sizeof(halfchunk_t)) + 1;
    loadhalfchunk(from, &chunk);
    storehalfchunk(out, &chunk);
    out += align;
    from += align;
    len -= align;
    while (len > 0) {
        loadhalfchunk(from, &chunk);
        storehalfchunk(out, &chunk);
        out += sizeof(halfchunk_t);
        from += sizeof(halfchunk_t);
        len -= sizeof(halfchunk_t);
    }
    return out;
}
#endif

/* Copy DIST bytes from OUT - DIST into OUT + DIST * k, for 0 <= k < LEN/DIST.
   Return OUT + LEN. */
static inline uint8_t* CHUNKMEMSET(uint8_t *out, uint8_t *from, unsigned len) {
    /* Debug performance related issues when len < sizeof(uint64_t):
       Assert(len >= sizeof(uint64_t), "chunkmemset should be called on larger chunks"); */
    Assert(from != out, "chunkmemset cannot have a distance 0");

    chunk_t chunk_load;
    uint32_t chunk_mod = 0;
    uint32_t adv_amount;
    int64_t sdist = out - from;
    uint64_t dist = llabs(sdist);

    /* We are supporting the case for when we are reading bytes from ahead in the buffer.
     * We now have to handle this, though it wasn't _quite_ clear if this rare circumstance
     * always needed to be handled here or if we're just now seeing it because we are
     * dispatching to this function, more */
    if (sdist < 0 && dist < len) {
#ifdef HAVE_MASKED_READWRITE
        /* We can still handle this case if we can mitigate over writing _and_ we
         * fit the entirety of the copy length with one load */
        if (len <= sizeof(chunk_t)) {
            /* Tempting to add a goto to the block below but hopefully most compilers
             * collapse these identical code segments as one label to jump to */
            return CHUNKCOPY(out, from, len);
        }
#endif
        /* Here the memmove semantics match perfectly, as when this happens we are
         * effectively sliding down the contents of memory by dist bytes */
        memmove(out, from, len);
        return out + len;
    }

    if (dist == 1) {
        memset(out, *from, len);
        return out + len;
    } else if (dist >= sizeof(chunk_t)) {
        return CHUNKCOPY(out, from, len);
    }

    /* Only AVX2+ as there's 128 bit vectors and 256 bit. We allow for shorter vector
     * lengths because they serve to allow more cases to fall into chunkcopy, as the
     * distance of the shorter length is still deemed a safe distance. We rewrite this
     * here rather than calling the ssse3 variant directly now because doing so required
     * dispatching to another function and broke inlining for this function entirely. We
     * also can merge an assert and some remainder peeling behavior into the same code blocks,
     * making the code a little smaller.  */
#ifdef HAVE_HALF_CHUNK
    if (len <= sizeof(halfchunk_t)) {
        if (dist >= sizeof(halfchunk_t))
            return HALFCHUNKCOPY(out, from, len);

        if ((dist % 2) != 0 || dist == 6) {
            halfchunk_t halfchunk_load = GET_HALFCHUNK_MAG(from, &chunk_mod, (unsigned)dist);

            if (len == sizeof(halfchunk_t)) {
                storehalfchunk(out, &halfchunk_load);
                len -= sizeof(halfchunk_t);
                out += sizeof(halfchunk_t);
            }

            chunk_load = halfchunk2whole(&halfchunk_load);
            goto rem_bytes;
        }
    }
#endif

#ifdef HAVE_CHUNKMEMSET_2
    if (dist == 2) {
        chunkmemset_2(from, &chunk_load);
    } else
#endif
#ifdef HAVE_CHUNKMEMSET_4
    if (dist == 4) {
        chunkmemset_4(from, &chunk_load);
    } else
#endif
#ifdef HAVE_CHUNKMEMSET_8
    if (dist == 8) {
        chunkmemset_8(from, &chunk_load);
    } else
#endif
#ifdef HAVE_CHUNKMEMSET_16
    if (dist == 16) {
        chunkmemset_16(from, &chunk_load);
    } else
#endif
    chunk_load = GET_CHUNK_MAG(from, &chunk_mod, (unsigned)dist);

    adv_amount = sizeof(chunk_t) - chunk_mod;

    while (len >= (2 * sizeof(chunk_t))) {
        storechunk(out, &chunk_load);
        storechunk(out + adv_amount, &chunk_load);
        out += 2 * adv_amount;
        len -= 2 * adv_amount;
    }

    /* If we don't have a "dist" length that divides evenly into a vector
     * register, we can write the whole vector register but we need only
     * advance by the amount of the whole string that fits in our chunk_t.
     * If we do divide evenly into the vector length, adv_amount = chunk_t size*/
    while (len >= sizeof(chunk_t)) {
        storechunk(out, &chunk_load);
        len -= adv_amount;
        out += adv_amount;
    }

#ifdef HAVE_HALF_CHUNK
rem_bytes:
#endif
    if (len) {
        memcpy(out, &chunk_load, len);
        out += len;
    }

    return out;
}

Z_INTERNAL uint8_t* CHUNKMEMSET_SAFE(uint8_t *out, uint8_t *from, unsigned len, unsigned left) {
#if OPTIMAL_CMP < 32
    static const uint32_t align_mask = 7;
#elif OPTIMAL_CMP == 32
    static const uint32_t align_mask = 3;
#endif

    len = MIN(len, left);

#if OPTIMAL_CMP < 64
    while (((uintptr_t)out & align_mask) && (len > 0)) {
        *out++ = *from++;
        --len;
        --left;
    }
#endif

#ifndef HAVE_MASKED_READWRITE
    if (UNLIKELY(left < sizeof(chunk_t))) {
        while (len > 0) {
            *out++ = *from++;
            --len;
        }

        return out;
    }
#endif

    if (len)
        out = CHUNKMEMSET(out, from, len);

    return out;
}

static inline uint8_t *CHUNKCOPY_SAFE(uint8_t *out, uint8_t *from, uint64_t len, uint8_t *safe)
{
    if (out == from)
        return out + len;

    uint64_t safelen = (safe - out);
    len = MIN(len, safelen);

#ifndef HAVE_MASKED_READWRITE
    uint64_t from_dist = (uint64_t)llabs(safe - from);
    if (UNLIKELY(from_dist < sizeof(chunk_t) || safelen < sizeof(chunk_t))) {
        while (len--) {
            *out++ = *from++;
        }

        return out;
    }
#endif

    return CHUNKMEMSET(out, from, (unsigned)len);
}
