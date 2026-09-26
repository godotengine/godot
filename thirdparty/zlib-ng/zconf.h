/* zconf.h -- configuration of the zlib compression library
 * Copyright (C) 1995-2024 Jean-loup Gailly, Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ZCONF_H
#define ZCONF_H

#include "zlib_name_mangling.h"

#if !defined(_WIN32) && defined(__WIN32__)
#  define _WIN32
#endif

/* Clang macro for detecting declspec support
 * https://clang.llvm.org/docs/LanguageExtensions.html#has-declspec-attribute
 */
#ifndef __has_declspec_attribute
#  define __has_declspec_attribute(x) 0
#endif

#if defined(ZLIB_CONST) && !defined(z_const)
#  define z_const const
#else
#  define z_const
#endif

/* Maximum value for memLevel in deflateInit2 */
#ifndef MAX_MEM_LEVEL
#  define MAX_MEM_LEVEL 9
#endif

/* Maximum value for windowBits in deflateInit2 and inflateInit2.
 * WARNING: reducing MAX_WBITS makes minigzip unable to extract .gz files
 * created by gzip. (Files created by minigzip can still be extracted by
 * gzip.)
 */
#ifndef MIN_WBITS
#  define MIN_WBITS   8  /* 256 LZ77 window */
#endif
#ifndef MAX_WBITS
#  define MAX_WBITS   15 /* 32K LZ77 window */
#endif

/* The memory requirements for deflate are (in bytes):
            (1 << (windowBits+2)) +  (1 << (memLevel+9))
 that is: 128K for windowBits=15  +  128K for memLevel = 8  (default values)
 plus a few kilobytes for small objects. For example, if you want to reduce
 the default memory requirements from 256K to 128K, compile with
     make CFLAGS="-O -DMAX_WBITS=14 -DMAX_MEM_LEVEL=7"
 Of course this will generally degrade compression (there's no free lunch).

   The memory requirements for inflate are (in bytes) 1 << windowBits
 that is, 32K for windowBits=15 (default value) plus about 7 kilobytes
 for small objects.
*/

/* Type declarations */


#ifndef OF /* function prototypes */
#  define OF(args)  args
#endif

#ifdef ZLIB_INTERNAL
#  define Z_INTERNAL ZLIB_INTERNAL
#endif

/* If building or using zlib as a DLL, define ZLIB_DLL.
 * This is not mandatory, but it offers a little performance increase.
 */
#if defined(ZLIB_DLL) && (defined(_WIN32) || (__has_declspec_attribute(dllexport) && __has_declspec_attribute(dllimport)))
#  ifdef Z_INTERNAL
#    define Z_EXTERN extern __declspec(dllexport)
#  else
#    define Z_EXTERN extern __declspec(dllimport)
#  endif
#endif

/* If building or using zlib with the WINAPI/WINAPIV calling convention,
 * define ZLIB_WINAPI.
 * Caution: the standard ZLIB1.DLL is NOT compiled using ZLIB_WINAPI.
 */
#if defined(ZLIB_WINAPI) && defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
   /* No need for _export, use ZLIB.DEF instead. */
   /* For complete Windows compatibility, use WINAPI, not __stdcall. */
#  define Z_EXPORT WINAPI
#  define Z_EXPORTVA WINAPIV
#endif

#ifndef Z_EXTERN
#  define Z_EXTERN extern
#endif
#ifndef Z_EXPORT
#  define Z_EXPORT
#endif
#ifndef Z_EXPORTVA
#  define Z_EXPORTVA
#endif

/* Conditional exports */
#define ZNG_CONDEXPORT Z_INTERNAL

/* For backwards compatibility */

#ifndef ZEXTERN
#  define ZEXTERN Z_EXTERN
#endif
#ifndef ZEXPORT
#  define ZEXPORT Z_EXPORT
#endif
#ifndef ZEXPORTVA
#  define ZEXPORTVA Z_EXPORTVA
#endif
#ifndef FAR
#  define FAR
#endif

/* Legacy zlib typedefs for backwards compatibility. Don't assume stdint.h is defined. */
typedef unsigned char Byte;
typedef Byte Bytef;

typedef unsigned int   uInt;  /* 16 bits or more */
typedef unsigned long  uLong; /* 32 bits or more */

typedef char  charf;
typedef int   intf;
typedef uInt  uIntf;
typedef uLong uLongf;

typedef void const *voidpc;
typedef void       *voidpf;
typedef void       *voidp;

typedef unsigned int z_crc_t;

#ifdef HAVE_UNISTD_H    /* may be set to #if 1 by configure/cmake/etc */
#  define Z_HAVE_UNISTD_H
#endif

#ifdef NEED_PTRDIFF_T    /* may be set to #if 1 by configure/cmake/etc */
typedef PTRDIFF_TYPE ptrdiff_t;
#endif

#include <sys/types.h>      /* for off_t */

#include <stddef.h>         /* for wchar_t and NULL */

/* a little trick to accommodate both "#define _LARGEFILE64_SOURCE" and
 * "#define _LARGEFILE64_SOURCE 1" as requesting 64-bit operations, (even
 * though the former does not conform to the LFS document), but considering
 * both "#undef _LARGEFILE64_SOURCE" and "#define _LARGEFILE64_SOURCE 0" as
 * equivalently requesting no 64-bit operations
 */
#if defined(_LARGEFILE64_SOURCE) && -_LARGEFILE64_SOURCE - -1 == 1
#  undef _LARGEFILE64_SOURCE
#endif

#if defined(Z_HAVE_UNISTD_H) || defined(_LARGEFILE64_SOURCE)
#  include <unistd.h>         /* for SEEK_*, off_t, and _LFS64_LARGEFILE */
#  ifndef z_off_t
#    define z_off_t off_t
#  endif
#endif

#if defined(_LFS64_LARGEFILE) && _LFS64_LARGEFILE-0
#  define Z_LFS64
#endif

#if defined(_LARGEFILE64_SOURCE) && defined(Z_LFS64)
#  define Z_LARGE64
#endif

#if defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS-0 == 64 && defined(Z_LFS64)
#  define Z_WANT64
#endif

#if !defined(SEEK_SET)
#  define SEEK_SET        0       /* Seek from beginning of file.  */
#  define SEEK_CUR        1       /* Seek from current position.  */
#  define SEEK_END        2       /* Set file pointer to EOF plus "offset" */
#endif

#ifndef z_off_t
#  define z_off_t long
#endif

#if !defined(_WIN32) && defined(Z_LARGE64)
#  define z_off64_t off64_t
#else
#  if defined(__MSYS__)
#    define z_off64_t _off64_t
#  elif defined(_WIN32) && !defined(__GNUC__)
#    define z_off64_t __int64
#  else
#    define z_off64_t z_off_t
#  endif
#endif

typedef size_t z_size_t;

#endif /* ZCONF_H */
