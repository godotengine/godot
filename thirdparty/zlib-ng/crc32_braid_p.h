#ifndef CRC32_BRAID_P_H_
#define CRC32_BRAID_P_H_

#include "zendian.h"

/* Define BRAID_N, valid range is 1..6 */
#define BRAID_N 5

/* Define BRAID_W and the associated z_word_t type. If BRAID_W is not defined, then a braided
   calculation is not used, and the associated tables and code are not compiled.
 */
#if defined(__x86_64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64) || defined(__powerpc64__)
#  define BRAID_W 8
    typedef uint64_t z_word_t;
#else
#  define BRAID_W 4
    typedef uint32_t z_word_t;
#endif

#if BYTE_ORDER == LITTLE_ENDIAN
#  define ZSWAPWORD(word) (word)
#  define BRAID_TABLE crc_braid_table
#elif BYTE_ORDER == BIG_ENDIAN
#  if BRAID_W == 8
#    define ZSWAPWORD(word) ZSWAP64(word)
#  elif BRAID_W == 4
#    define ZSWAPWORD(word) ZSWAP32(word)
#  endif
#  define BRAID_TABLE crc_braid_big_table
#else
#  error "No endian defined"
#endif

#define CRC_DO1 c = crc_table[(c ^ *buf++) & 0xff] ^ (c >> 8)
#define CRC_DO8 CRC_DO1; CRC_DO1; CRC_DO1; CRC_DO1; CRC_DO1; CRC_DO1; CRC_DO1; CRC_DO1

/* CRC polynomial. */
#define POLY 0xedb88320         /* p(x) reflected, with x^32 implied */

#endif /* CRC32_BRAID_P_H_ */
