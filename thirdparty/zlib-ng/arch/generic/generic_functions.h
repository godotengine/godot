/* generic_functions.h -- generic C implementations for arch-specific functions.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef GENERIC_FUNCTIONS_H_
#define GENERIC_FUNCTIONS_H_

#include "zendian.h"
#include "deflate.h"
#include "crc32_braid_p.h"

typedef uint32_t (*adler32_func)(uint32_t adler, const uint8_t *buf, size_t len);
typedef uint32_t (*compare256_func)(const uint8_t *src0, const uint8_t *src1);
typedef uint32_t (*crc32_func)(uint32_t crc32, const uint8_t *buf, size_t len);
typedef uint32_t (*crc32_fold_reset_func)(crc32_fold *crc);
typedef void     (*crc32_fold_copy_func)(crc32_fold *crc, uint8_t *dst, const uint8_t *src, size_t len);
typedef uint32_t (*crc32_fold_final_func)(crc32_fold *crc);
typedef void     (*slide_hash_func)(deflate_state *s);


uint32_t adler32_c(uint32_t adler, const uint8_t *buf, size_t len);
uint32_t adler32_fold_copy_c(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);

uint8_t* chunkmemset_safe_c(uint8_t *out, uint8_t *from, unsigned len, unsigned left);

uint32_t compare256_c(const uint8_t *src0, const uint8_t *src1);

uint32_t crc32_braid(uint32_t c, const uint8_t *buf, size_t len);
uint32_t crc32_braid_internal(uint32_t c, const uint8_t *buf, size_t len);

#ifndef WITHOUT_CHORBA
  uint32_t crc32_chorba(uint32_t crc, const uint8_t *buf, size_t len);
  uint32_t crc32_chorba_118960_nondestructive (uint32_t crc, const z_word_t* input, size_t len);
  uint32_t crc32_chorba_32768_nondestructive (uint32_t crc, const uint64_t* input, size_t len);
  uint32_t crc32_chorba_small_nondestructive (uint32_t crc, const uint64_t* input, size_t len);
  uint32_t crc32_chorba_small_nondestructive_32bit (uint32_t crc, const uint32_t* input, size_t len);
#endif

uint32_t crc32_fold_reset_c(crc32_fold *crc);
void     crc32_fold_copy_c(crc32_fold *crc, uint8_t *dst, const uint8_t *src, size_t len);
void     crc32_fold_c(crc32_fold *crc, const uint8_t *src, size_t len, uint32_t init_crc);
uint32_t crc32_fold_final_c(crc32_fold *crc);

void     inflate_fast_c(PREFIX3(stream) *strm, uint32_t start);

uint32_t longest_match_c(deflate_state *const s, Pos cur_match);
uint32_t longest_match_slow_c(deflate_state *const s, Pos cur_match);

void     slide_hash_c(deflate_state *s);

#ifdef DISABLE_RUNTIME_CPU_DETECTION
// Generic code
#  define native_adler32 adler32_c
#  define native_adler32_fold_copy adler32_fold_copy_c
#  define native_chunkmemset_safe chunkmemset_safe_c
#ifndef WITHOUT_CHORBA
#  define native_crc32 crc32_chorba
#else
#  define native_crc32 crc32_braid
#endif
#  define native_crc32_fold crc32_fold_c
#  define native_crc32_fold_copy crc32_fold_copy_c
#  define native_crc32_fold_final crc32_fold_final_c
#  define native_crc32_fold_reset crc32_fold_reset_c
#  define native_inflate_fast inflate_fast_c
#  define native_slide_hash slide_hash_c
#  define native_longest_match longest_match_c
#  define native_longest_match_slow longest_match_slow_c
#  define native_compare256 compare256_c
#endif

#endif
