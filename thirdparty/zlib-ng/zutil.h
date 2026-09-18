#ifndef ZUTIL_H_
#define ZUTIL_H_
/* zutil.h -- internal interface and configuration of the compression library
 * Copyright (C) 1995-2024 Jean-loup Gailly, Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

#include "zbuild.h"
#ifdef ZLIB_COMPAT
#  include "zlib.h"
#else
#  include "zlib-ng.h"
#endif

typedef unsigned char uch; /* Included for compatibility with external code only */
typedef uint16_t ush;      /* Included for compatibility with external code only */
typedef unsigned long ulg;

extern z_const char * const PREFIX(z_errmsg)[10]; /* indexed by 2-zlib_error */
/* (size given to avoid silly warnings with Visual C++) */

#define ERR_MSG(err) PREFIX(z_errmsg)[(err) < -6 || (err) > 2 ? 9 : 2 - (err)]

#define ERR_RETURN(strm, err) return (strm->msg = ERR_MSG(err), (err))
/* To be used only when the state is known to be valid */

        /* common constants */

#ifndef DEF_WBITS
#  define DEF_WBITS MAX_WBITS
#endif
/* default windowBits for decompression. MAX_WBITS is for compression only */

#define MAX_BITS 15
/* all codes must not exceed MAX_BITS bits */
#define MAX_DIST_EXTRA_BITS 13
/* maximum number of extra distance bits */

#if MAX_MEM_LEVEL >= 8
#  define DEF_MEM_LEVEL 8
#else
#  define DEF_MEM_LEVEL  MAX_MEM_LEVEL
#endif
/* default memLevel */

#define STORED_BLOCK 0
#define STATIC_TREES 1
#define DYN_TREES    2
/* The three kinds of block type */

#define STD_MIN_MATCH  3
#define STD_MAX_MATCH  258
/* The minimum and maximum match lengths mandated by the deflate standard */

#define WANT_MIN_MATCH  4
/* The minimum wanted match length, affects deflate_quick, deflate_fast, deflate_medium and deflate_slow  */

#define PRESET_DICT 0x20 /* preset dictionary flag in zlib header */

#define ADLER32_INITIAL_VALUE 1 /* initial adler-32 hash value */
#define CRC32_INITIAL_VALUE   0 /* initial crc-32 hash value */

#define ZLIB_WRAPLEN 6      /* zlib format overhead */
#define GZIP_WRAPLEN 18     /* gzip format overhead */

#define DEFLATE_HEADER_BITS 3
#define DEFLATE_EOBS_BITS   15
#define DEFLATE_PAD_BITS    6
#define DEFLATE_BLOCK_OVERHEAD ((DEFLATE_HEADER_BITS + DEFLATE_EOBS_BITS + DEFLATE_PAD_BITS) >> 3)
/* deflate block overhead: 3 bits for block start + 15 bits for block end + padding to nearest byte */

#define DEFLATE_QUICK_LIT_MAX_BITS 9
#define DEFLATE_QUICK_OVERHEAD(x) ((x * (DEFLATE_QUICK_LIT_MAX_BITS - 8) + 7) >> 3)
/* deflate_quick worst-case overhead: 9 bits per literal, round up to next byte (+7) */


        /* target dependencies */

#ifdef AMIGA
#  define OS_CODE  1
#endif

#ifdef __370__
#  if __TARGET_LIB__ < 0x20000000
#    define OS_CODE 4
#  elif __TARGET_LIB__ < 0x40000000
#    define OS_CODE 11
#  else
#    define OS_CODE 8
#  endif
#endif

#if defined(ATARI) || defined(atarist)
#  define OS_CODE  5
#endif

#ifdef OS2
#  define OS_CODE  6
#endif

#if defined(MACOS)
#  define OS_CODE  7
#endif

#ifdef __acorn
#  define OS_CODE 13
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#  define OS_CODE  10
#endif

#ifdef __APPLE__
#  define OS_CODE 19
#endif

        /* common defaults */

#ifndef OS_CODE
#  define OS_CODE  3  /* assume Unix */
#endif

         /* macros */

#define CHECK_VER_STSIZE(_ver,_stsize) ((_ver) == NULL || (_ver)[0] != PREFIX2(VERSION)[0] || (_stsize) != (int32_t)sizeof(PREFIX3(stream)))

         /* memory allocation functions */

void Z_INTERNAL *PREFIX(zcalloc)(void *opaque, unsigned items, unsigned size);
void Z_INTERNAL  PREFIX(zcfree)(void *opaque, void *ptr);
void Z_INTERNAL *zng_alloc_aligned(unsigned size, unsigned align);
void Z_INTERNAL zng_free_aligned(void *ptr);

typedef void *zng_calloc_func(void *opaque, unsigned items, unsigned size);
typedef void  zng_cfree_func(void *opaque, void *ptr);

#endif /* ZUTIL_H_ */
