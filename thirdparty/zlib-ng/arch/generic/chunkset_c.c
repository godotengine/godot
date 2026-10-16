/* chunkset.c -- inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zmemory.h"

typedef uint64_t chunk_t;

#define CHUNK_SIZE 8

#define HAVE_CHUNKMEMSET_4
#define HAVE_CHUNKMEMSET_8

static inline void chunkmemset_4(uint8_t *from, chunk_t *chunk) {
    uint32_t tmp = zng_memread_4(from);
    *chunk = tmp | ((chunk_t)tmp << 32);
}

static inline void chunkmemset_8(uint8_t *from, chunk_t *chunk) {
    *chunk = zng_memread_8(from);
}

static inline void loadchunk(uint8_t const *s, chunk_t *chunk) {
    *chunk = zng_memread_8(s);
}

static inline void storechunk(uint8_t *out, chunk_t *chunk) {
    zng_memwrite_8(out, *chunk);
}

#define CHUNKSIZE        chunksize_c
#define CHUNKCOPY        chunkcopy_c
#define CHUNKUNROLL      chunkunroll_c
#define CHUNKMEMSET      chunkmemset_c
#define CHUNKMEMSET_SAFE chunkmemset_safe_c

#include "chunkset_tpl.h"

#define INFLATE_FAST     inflate_fast_c

#include "inffast_tpl.h"
