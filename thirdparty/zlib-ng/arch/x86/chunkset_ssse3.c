/* chunkset_ssse3.c -- SSSE3 inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zmemory.h"

#if defined(X86_SSSE3)
#include <immintrin.h>
#include "arch/generic/chunk_128bit_perm_idx_lut.h"

typedef __m128i chunk_t;

#define HAVE_CHUNKMEMSET_2
#define HAVE_CHUNKMEMSET_4
#define HAVE_CHUNKMEMSET_8
#define HAVE_CHUNK_MAG


static inline void chunkmemset_2(uint8_t *from, chunk_t *chunk) {
    *chunk = _mm_set1_epi16(zng_memread_2(from));
}

static inline void chunkmemset_4(uint8_t *from, chunk_t *chunk) {
    *chunk = _mm_set1_epi32(zng_memread_4(from));
}

static inline void chunkmemset_8(uint8_t *from, chunk_t *chunk) {
    *chunk = _mm_set1_epi64x(zng_memread_8(from));
}

static inline void loadchunk(uint8_t const *s, chunk_t *chunk) {
    *chunk = _mm_loadu_si128((__m128i *)s);
}

static inline void storechunk(uint8_t *out, chunk_t *chunk) {
    _mm_storeu_si128((__m128i *)out, *chunk);
}

static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
    lut_rem_pair lut_rem = perm_idx_lut[dist - 3];
    __m128i perm_vec, ret_vec;
    /* Important to note:
     * This is _not_ to subvert the memory sanitizer but to instead unpoison some
     * bytes we willingly and purposefully load uninitialized that we swizzle over
     * in a vector register, anyway.  If what we assume is wrong about what is used,
     * the memory sanitizer will still usefully flag it */
    __msan_unpoison(buf + dist, 16 - dist);
    ret_vec = _mm_loadu_si128((__m128i*)buf);
    *chunk_rem = lut_rem.remval;

    perm_vec = _mm_load_si128((__m128i*)(permute_table + lut_rem.idx));
    ret_vec = _mm_shuffle_epi8(ret_vec, perm_vec);

    return ret_vec;
}

#define CHUNKSIZE        chunksize_ssse3
#define CHUNKMEMSET      chunkmemset_ssse3
#define CHUNKMEMSET_SAFE chunkmemset_safe_ssse3
#define CHUNKCOPY        chunkcopy_ssse3
#define CHUNKUNROLL      chunkunroll_ssse3

#include "chunkset_tpl.h"

#define INFLATE_FAST     inflate_fast_ssse3

#include "inffast_tpl.h"

#endif
