/* functable.h -- Struct containing function pointers to optimized functions
 * Copyright (C) 2017 Hans Kristian Rosbach
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef FUNCTABLE_H_
#define FUNCTABLE_H_

#include "deflate.h"
#include "crc32.h"

#ifdef DISABLE_RUNTIME_CPU_DETECTION

#  include "arch_functions.h"

/* When compiling with native instructions it is not necessary to use functable.
 * Instead we use native_ macro indicating the best available variant of arch-specific
 * functions for the current platform.
 */
#  define FUNCTABLE_INIT ((void)0)
#  define FUNCTABLE_CALL(name) native_ ## name
#  define FUNCTABLE_FPTR(name) &native_ ## name

#else

struct functable_s {
    int      (* force_init)         (void);
    uint32_t (* adler32)            (uint32_t adler, const uint8_t *buf, size_t len);
    uint32_t (* adler32_fold_copy)  (uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
    uint8_t* (* chunkmemset_safe)   (uint8_t *out, uint8_t *from, unsigned len, unsigned left);
    uint32_t (* compare256)         (const uint8_t *src0, const uint8_t *src1);
    uint32_t (* crc32)              (uint32_t crc, const uint8_t *buf, size_t len);
    void     (* crc32_fold)         (struct crc32_fold_s *crc, const uint8_t *src, size_t len, uint32_t init_crc);
    void     (* crc32_fold_copy)    (struct crc32_fold_s *crc, uint8_t *dst, const uint8_t *src, size_t len);
    uint32_t (* crc32_fold_final)   (struct crc32_fold_s *crc);
    uint32_t (* crc32_fold_reset)   (struct crc32_fold_s *crc);
    void     (* inflate_fast)       (PREFIX3(stream) *strm, uint32_t start);
    uint32_t (* longest_match)      (deflate_state *const s, Pos cur_match);
    uint32_t (* longest_match_slow) (deflate_state *const s, Pos cur_match);
    void     (* slide_hash)         (deflate_state *s);
};

Z_INTERNAL extern struct functable_s functable;


/* Explicitly indicate functions are conditionally dispatched.
 */
#  define FUNCTABLE_INIT if (functable.force_init()) {return Z_VERSION_ERROR;}
#  define FUNCTABLE_CALL(name) functable.name
#  define FUNCTABLE_FPTR(name) functable.name

#endif

#endif
