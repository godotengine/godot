/*==============================================================================
 Copyright (c) 2020 YaoYuan <ibireme@gmail.com>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *============================================================================*/

/**
 @file yyjson.h
 @date 2019-03-09
 @author YaoYuan
 */

#ifndef YYJSON_H
#define YYJSON_H



/*==============================================================================
 * Header Files
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <float.h>



/*==============================================================================
 * Compile-time Options
 *============================================================================*/

/*
 Define as 1 to disable JSON reader at compile-time.
 This disables functions with "read" in their name.
 Reduces binary size by about 60%.
 */
#ifndef YYJSON_DISABLE_READER
#endif

/*
 Define as 1 to disable JSON writer at compile-time.
 This disables functions with "write" in their name.
 Reduces binary size by about 30%.
 */
#ifndef YYJSON_DISABLE_WRITER
#endif

/*
 Define as 1 to disable JSON incremental reader at compile-time.
 This disables functions with "incr" in their name.
 */
#ifndef YYJSON_DISABLE_INCR_READER
#endif

/*
 Define as 1 to disable JSON Pointer, JSON Patch and JSON Merge Patch supports.
 This disables functions with "ptr" or "patch" in their name.
 */
#ifndef YYJSON_DISABLE_UTILS
#endif

/*
 Define as 1 to disable the fast floating-point number conversion in yyjson.
 Libc's `strtod/snprintf` will be used instead.

 This reduces binary size by about 30%, but significantly slows down the
 floating-point read/write speed.
 */
#ifndef YYJSON_DISABLE_FAST_FP_CONV
#endif

/*
 Define as 1 to disable non-standard JSON features support at compile-time:
 - YYJSON_READ_ALLOW_INF_AND_NAN
 - YYJSON_READ_ALLOW_COMMENTS
 - YYJSON_READ_ALLOW_TRAILING_COMMAS
 - YYJSON_READ_ALLOW_INVALID_UNICODE
 - YYJSON_READ_ALLOW_BOM
 - YYJSON_WRITE_ALLOW_INF_AND_NAN
 - YYJSON_WRITE_ALLOW_INVALID_UNICODE

 This reduces binary size by about 10%, and slightly improves performance.
 */
#ifndef YYJSON_DISABLE_NON_STANDARD
#endif

/*
 Define as 1 to disable UTF-8 validation at compile-time.

 Use this if all input strings are guaranteed to be valid UTF-8
 (e.g. language-level String types are already validated).

 Disabling UTF-8 validation improves performance for non-ASCII strings by about
 3% to 7%.

 Note: If this flag is enabled while passing illegal UTF-8 strings,
 the following errors may occur:
 - Escaped characters may be ignored when parsing JSON strings.
 - Ending quotes may be ignored when parsing JSON strings, causing the
   string to merge with the next value.
 - When serializing with `yyjson_mut_val`, the string's end may be accessed
   out of bounds, potentially causing a segmentation fault.
 */
#ifndef YYJSON_DISABLE_UTF8_VALIDATION
#endif

/*
 Define as 1 to improve performance on architectures that do not support
 unaligned memory access.

 Normally, this does not need to be set manually. See the C file for details.
 */
#ifndef YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
#endif

/* Define as 1 to export symbols when building this library as a Windows DLL. */
#ifndef YYJSON_EXPORTS
#endif

/* Define as 1 to import symbols when using this library as a Windows DLL. */
#ifndef YYJSON_IMPORTS
#endif

/* Define as 1 to include <stdint.h> for compilers without C99 support. */
#ifndef YYJSON_HAS_STDINT_H
#endif

/* Define as 1 to include <stdbool.h> for compilers without C99 support. */
#ifndef YYJSON_HAS_STDBOOL_H
#endif



/*==============================================================================
 * Compiler Macros
 *============================================================================*/

/** compiler version (MSVC) */
#ifdef _MSC_VER
#   define YYJSON_MSC_VER _MSC_VER
#else
#   define YYJSON_MSC_VER 0
#endif

/** compiler version (GCC) */
#ifdef __GNUC__
#   define YYJSON_GCC_VER __GNUC__
#   if defined(__GNUC_PATCHLEVEL__)
#       define yyjson_gcc_available(major, minor, patch) \
            ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) \
            >= (major * 10000 + minor * 100 + patch))
#   else
#       define yyjson_gcc_available(major, minor, patch) \
            ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100) \
            >= (major * 10000 + minor * 100 + patch))
#   endif
#else
#   define YYJSON_GCC_VER 0
#   define yyjson_gcc_available(major, minor, patch) 0
#endif

/** real gcc check */
#if !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__ICC) && \
    defined(__GNUC__)
#   define YYJSON_IS_REAL_GCC 1
#else
#   define YYJSON_IS_REAL_GCC 0
#endif

/** C version (STDC) */
#if defined(__STDC__) && (__STDC__ >= 1) && defined(__STDC_VERSION__)
#   define YYJSON_STDC_VER __STDC_VERSION__
#else
#   define YYJSON_STDC_VER 0
#endif

/** C++ version */
#if defined(__cplusplus)
#   define YYJSON_CPP_VER __cplusplus
#else
#   define YYJSON_CPP_VER 0
#endif

/** compiler builtin check (since gcc 10.0, clang 2.6, icc 2021) */
#ifndef yyjson_has_builtin
#   ifdef __has_builtin
#       define yyjson_has_builtin(x) __has_builtin(x)
#   else
#       define yyjson_has_builtin(x) 0
#   endif
#endif

/** compiler attribute check (since gcc 5.0, clang 2.9, icc 17) */
#ifndef yyjson_has_attribute
#   ifdef __has_attribute
#       define yyjson_has_attribute(x) __has_attribute(x)
#   else
#       define yyjson_has_attribute(x) 0
#   endif
#endif

/** compiler feature check (since clang 2.6, icc 17) */
#ifndef yyjson_has_feature
#   ifdef __has_feature
#       define yyjson_has_feature(x) __has_feature(x)
#   else
#       define yyjson_has_feature(x) 0
#   endif
#endif

/** include check (since gcc 5.0, clang 2.7, icc 16, msvc 2017 15.3) */
#ifndef yyjson_has_include
#   ifdef __has_include
#       define yyjson_has_include(x) __has_include(x)
#   else
#       define yyjson_has_include(x) 0
#   endif
#endif

/** inline for compiler */
#ifndef yyjson_inline
#   if YYJSON_MSC_VER >= 1200
#       define yyjson_inline __forceinline
#   elif defined(_MSC_VER)
#       define yyjson_inline __inline
#   elif yyjson_has_attribute(always_inline) || YYJSON_GCC_VER >= 4
#       define yyjson_inline __inline__ __attribute__((always_inline))
#   elif defined(__clang__) || defined(__GNUC__)
#       define yyjson_inline __inline__
#   elif defined(__cplusplus) || YYJSON_STDC_VER >= 199901L
#       define yyjson_inline inline
#   else
#       define yyjson_inline
#   endif
#endif

/** noinline for compiler */
#ifndef yyjson_noinline
#   if YYJSON_MSC_VER >= 1400
#       define yyjson_noinline __declspec(noinline)
#   elif yyjson_has_attribute(noinline) || YYJSON_GCC_VER >= 4
#       define yyjson_noinline __attribute__((noinline))
#   else
#       define yyjson_noinline
#   endif
#endif

/** align for compiler */
#ifndef yyjson_align
#   if YYJSON_MSC_VER >= 1300
#       define yyjson_align(x) __declspec(align(x))
#   elif yyjson_has_attribute(aligned) || defined(__GNUC__)
#       define yyjson_align(x) __attribute__((aligned(x)))
#   elif YYJSON_CPP_VER >= 201103L
#       define yyjson_align(x) alignas(x)
#   else
#       define yyjson_align(x)
#   endif
#endif

/** likely for compiler */
#ifndef yyjson_likely
#   if yyjson_has_builtin(__builtin_expect) || \
    (YYJSON_GCC_VER >= 4 && YYJSON_GCC_VER != 5)
#       define yyjson_likely(expr) __builtin_expect(!!(expr), 1)
#   else
#       define yyjson_likely(expr) (expr)
#   endif
#endif

/** unlikely for compiler */
#ifndef yyjson_unlikely
#   if yyjson_has_builtin(__builtin_expect) || \
    (YYJSON_GCC_VER >= 4 && YYJSON_GCC_VER != 5)
#       define yyjson_unlikely(expr) __builtin_expect(!!(expr), 0)
#   else
#       define yyjson_unlikely(expr) (expr)
#   endif
#endif

/** compile-time constant check for compiler */
#ifndef yyjson_constant_p
#   if yyjson_has_builtin(__builtin_constant_p) || (YYJSON_GCC_VER >= 3)
#       define YYJSON_HAS_CONSTANT_P 1
#       define yyjson_constant_p(value) __builtin_constant_p(value)
#   else
#       define YYJSON_HAS_CONSTANT_P 0
#       define yyjson_constant_p(value) 0
#   endif
#endif

/** deprecate warning */
#ifndef yyjson_deprecated
#   if YYJSON_MSC_VER >= 1400
#       define yyjson_deprecated(msg) __declspec(deprecated(msg))
#   elif yyjson_has_feature(attribute_deprecated_with_message) || \
        (YYJSON_GCC_VER > 4 || (YYJSON_GCC_VER == 4 && __GNUC_MINOR__ >= 5))
#       define yyjson_deprecated(msg) __attribute__((deprecated(msg)))
#   elif YYJSON_GCC_VER >= 3
#       define yyjson_deprecated(msg) __attribute__((deprecated))
#   else
#       define yyjson_deprecated(msg)
#   endif
#endif

/** function export */
#ifndef yyjson_api
#   if defined(_WIN32)
#       if defined(YYJSON_EXPORTS) && YYJSON_EXPORTS
#           define yyjson_api __declspec(dllexport)
#       elif defined(YYJSON_IMPORTS) && YYJSON_IMPORTS
#           define yyjson_api __declspec(dllimport)
#       else
#           define yyjson_api
#       endif
#   elif yyjson_has_attribute(visibility) || YYJSON_GCC_VER >= 4
#       define yyjson_api __attribute__((visibility("default")))
#   else
#       define yyjson_api
#   endif
#endif

/** inline function export */
#ifndef yyjson_api_inline
#   define yyjson_api_inline static yyjson_inline
#endif

/** stdint (C89 compatible) */
#if (defined(YYJSON_HAS_STDINT_H) && YYJSON_HAS_STDINT_H) || \
    YYJSON_MSC_VER >= 1600 || YYJSON_STDC_VER >= 199901L || \
    defined(_STDINT_H) || defined(_STDINT_H_) || \
    defined(__CLANG_STDINT_H) || defined(_STDINT_H_INCLUDED) || \
    yyjson_has_include(<stdint.h>)
#   include <stdint.h>
#elif defined(_MSC_VER)
#   if _MSC_VER < 1300
        typedef signed char         int8_t;
        typedef signed short        int16_t;
        typedef signed int          int32_t;
        typedef unsigned char       uint8_t;
        typedef unsigned short      uint16_t;
        typedef unsigned int        uint32_t;
        typedef signed __int64      int64_t;
        typedef unsigned __int64    uint64_t;
#   else
        typedef signed __int8       int8_t;
        typedef signed __int16      int16_t;
        typedef signed __int32      int32_t;
        typedef unsigned __int8     uint8_t;
        typedef unsigned __int16    uint16_t;
        typedef unsigned __int32    uint32_t;
        typedef signed __int64      int64_t;
        typedef unsigned __int64    uint64_t;
#   endif
#else
#   if UCHAR_MAX == 0xFFU
        typedef signed char     int8_t;
        typedef unsigned char   uint8_t;
#   else
#       error cannot find 8-bit integer type
#   endif
#   if USHRT_MAX == 0xFFFFU
        typedef unsigned short  uint16_t;
        typedef signed short    int16_t;
#   elif UINT_MAX == 0xFFFFU
        typedef unsigned int    uint16_t;
        typedef signed int      int16_t;
#   else
#       error cannot find 16-bit integer type
#   endif
#   if UINT_MAX == 0xFFFFFFFFUL
        typedef unsigned int    uint32_t;
        typedef signed int      int32_t;
#   elif ULONG_MAX == 0xFFFFFFFFUL
        typedef unsigned long   uint32_t;
        typedef signed long     int32_t;
#   elif USHRT_MAX == 0xFFFFFFFFUL
        typedef unsigned short  uint32_t;
        typedef signed short    int32_t;
#   else
#       error cannot find 32-bit integer type
#   endif
#   if defined(__INT64_TYPE__) && defined(__UINT64_TYPE__)
        typedef __INT64_TYPE__  int64_t;
        typedef __UINT64_TYPE__ uint64_t;
#   elif defined(__GNUC__) || defined(__clang__)
#       if !defined(_SYS_TYPES_H) && !defined(__int8_t_defined)
        __extension__ typedef long long             int64_t;
#       endif
        __extension__ typedef unsigned long long    uint64_t;
#   elif defined(_LONG_LONG) || defined(__MWERKS__) || defined(_CRAYC) || \
        defined(__SUNPRO_C) || defined(__SUNPRO_CC)
        typedef long long           int64_t;
        typedef unsigned long long  uint64_t;
#   elif (defined(__BORLANDC__) && __BORLANDC__ > 0x460) || \
        defined(__WATCOM_INT64__) || defined (__alpha) || defined (__DECC)
        typedef __int64             int64_t;
        typedef unsigned __int64    uint64_t;
#   else
#       error cannot find 64-bit integer type
#   endif
#endif

/** stdbool (C89 compatible) */
#if (defined(YYJSON_HAS_STDBOOL_H) && YYJSON_HAS_STDBOOL_H) || \
    (yyjson_has_include(<stdbool.h>) && !defined(__STRICT_ANSI__)) || \
    YYJSON_MSC_VER >= 1800 || YYJSON_STDC_VER >= 199901L
#   include <stdbool.h>
#elif !defined(__bool_true_false_are_defined)
#   define __bool_true_false_are_defined 1
#   if defined(__cplusplus)
#       if defined(__GNUC__) && !defined(__STRICT_ANSI__)
#           define _Bool bool
#           if __cplusplus < 201103L
#               define bool bool
#               define false false
#               define true true
#           endif
#       endif
#   else
#       define bool unsigned char
#       define true 1
#       define false 0
#   endif
#endif

/** char bit check */
#if defined(CHAR_BIT)
#   if CHAR_BIT != 8
#       error non 8-bit char is not supported
#   endif
#endif

/**
 Microsoft Visual C++ 6.0 doesn't support converting number from u64 to f64:
 error C2520: conversion from unsigned __int64 to double not implemented.
 */
#ifndef YYJSON_U64_TO_F64_NO_IMPL
#   if (0 < YYJSON_MSC_VER) && (YYJSON_MSC_VER <= 1200)
#       define YYJSON_U64_TO_F64_NO_IMPL 1
#   else
#       define YYJSON_U64_TO_F64_NO_IMPL 0
#   endif
#endif



/*==============================================================================
 * Compile Hint Begin
 *============================================================================*/

/* extern "C" begin */
#ifdef __cplusplus
extern "C" {
#endif

/* warning suppress begin */
#if defined(__clang__)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wunused-function"
#   pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#   if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#       pragma GCC diagnostic push
#   endif
#   pragma GCC diagnostic ignored "-Wunused-function"
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable:4800) /* 'int': forcing value to 'true' or 'false' */
#endif



/*==============================================================================
 * Version
 *============================================================================*/

/** The major version of yyjson. */
#define YYJSON_VERSION_MAJOR  0

/** The minor version of yyjson. */
#define YYJSON_VERSION_MINOR  11

/** The patch version of yyjson. */
#define YYJSON_VERSION_PATCH  1

/** The version of yyjson in hex: `(major << 16) | (minor << 8) | (patch)`. */
#define YYJSON_VERSION_HEX    0x000B01

/** The version string of yyjson. */
#define YYJSON_VERSION_STRING "0.11.1"

/** The version of yyjson in hex, same as `YYJSON_VERSION_HEX`. */
yyjson_api uint32_t yyjson_version(void);



/*==============================================================================
 * JSON Types
 *============================================================================*/

/** Type of a JSON value (3 bit). */
typedef uint8_t yyjson_type;
/** No type, invalid. */
#define YYJSON_TYPE_NONE        ((uint8_t)0)        /* _____000 */
/** Raw string type, no subtype. */
#define YYJSON_TYPE_RAW         ((uint8_t)1)        /* _____001 */
/** Null type: `null` literal, no subtype. */
#define YYJSON_TYPE_NULL        ((uint8_t)2)        /* _____010 */
/** Boolean type, subtype: TRUE, FALSE. */
#define YYJSON_TYPE_BOOL        ((uint8_t)3)        /* _____011 */
/** Number type, subtype: UINT, SINT, REAL. */
#define YYJSON_TYPE_NUM         ((uint8_t)4)        /* _____100 */
/** String type, subtype: NONE, NOESC. */
#define YYJSON_TYPE_STR         ((uint8_t)5)        /* _____101 */
/** Array type, no subtype. */
#define YYJSON_TYPE_ARR         ((uint8_t)6)        /* _____110 */
/** Object type, no subtype. */
#define YYJSON_TYPE_OBJ         ((uint8_t)7)        /* _____111 */

/** Subtype of a JSON value (2 bit). */
typedef uint8_t yyjson_subtype;
/** No subtype. */
#define YYJSON_SUBTYPE_NONE     ((uint8_t)(0 << 3)) /* ___00___ */
/** False subtype: `false` literal. */
#define YYJSON_SUBTYPE_FALSE    ((uint8_t)(0 << 3)) /* ___00___ */
/** True subtype: `true` literal. */
#define YYJSON_SUBTYPE_TRUE     ((uint8_t)(1 << 3)) /* ___01___ */
/** Unsigned integer subtype: `uint64_t`. */
#define YYJSON_SUBTYPE_UINT     ((uint8_t)(0 << 3)) /* ___00___ */
/** Signed integer subtype: `int64_t`. */
#define YYJSON_SUBTYPE_SINT     ((uint8_t)(1 << 3)) /* ___01___ */
/** Real number subtype: `double`. */
#define YYJSON_SUBTYPE_REAL     ((uint8_t)(2 << 3)) /* ___10___ */
/** String that do not need to be escaped for writing (internal use). */
#define YYJSON_SUBTYPE_NOESC    ((uint8_t)(1 << 3)) /* ___01___ */

/** The mask used to extract the type of a JSON value. */
#define YYJSON_TYPE_MASK        ((uint8_t)0x07)     /* _____111 */
/** The number of bits used by the type. */
#define YYJSON_TYPE_BIT         ((uint8_t)3)
/** The mask used to extract the subtype of a JSON value. */
#define YYJSON_SUBTYPE_MASK     ((uint8_t)0x18)     /* ___11___ */
/** The number of bits used by the subtype. */
#define YYJSON_SUBTYPE_BIT      ((uint8_t)2)
/** The mask used to extract the reserved bits of a JSON value. */
#define YYJSON_RESERVED_MASK    ((uint8_t)0xE0)     /* 111_____ */
/** The number of reserved bits. */
#define YYJSON_RESERVED_BIT     ((uint8_t)3)
/** The mask used to extract the tag of a JSON value. */
#define YYJSON_TAG_MASK         ((uint8_t)0xFF)     /* 11111111 */
/** The number of bits used by the tag. */
#define YYJSON_TAG_BIT          ((uint8_t)8)

/** Padding size for JSON reader. */
#define YYJSON_PADDING_SIZE     4



/*==============================================================================
 * Allocator
 *============================================================================*/

/**
 A memory allocator.

 Typically you don't need to use it, unless you want to customize your own
 memory allocator.
 */
typedef struct yyjson_alc {
    /** Same as libc's malloc(size), should not be NULL. */
    void *(*malloc)(void *ctx, size_t size);
    /** Same as libc's realloc(ptr, size), should not be NULL. */
    void *(*realloc)(void *ctx, void *ptr, size_t old_size, size_t size);
    /** Same as libc's free(ptr), should not be NULL. */
    void (*free)(void *ctx, void *ptr);
    /** A context for malloc/realloc/free, can be NULL. */
    void *ctx;
} yyjson_alc;

/**
 A pool allocator uses fixed length pre-allocated memory.

 This allocator may be used to avoid malloc/realloc calls. The pre-allocated
 memory should be held by the caller. The maximum amount of memory required to
 read a JSON can be calculated using the `yyjson_read_max_memory_usage()`
 function, but the amount of memory required to write a JSON cannot be directly
 calculated.

 This is not a general-purpose allocator. It is designed to handle a single JSON
 data at a time. If it is used for overly complex memory tasks, such as parsing
 multiple JSON documents using the same allocator but releasing only a few of
 them, it may cause memory fragmentation, resulting in performance degradation
 and memory waste.

 @param alc The allocator to be initialized.
    If this parameter is NULL, the function will fail and return false.
    If `buf` or `size` is invalid, this will be set to an empty allocator.
 @param buf The buffer memory for this allocator.
    If this parameter is NULL, the function will fail and return false.
 @param size The size of `buf`, in bytes.
    If this parameter is less than 8 words (32/64 bytes on 32/64-bit OS), the
    function will fail and return false.
 @return true if the `alc` has been successfully initialized.

 @b Example
 @code
    // parse JSON with stack memory
    char buf[1024];
    yyjson_alc alc;
    yyjson_alc_pool_init(&alc, buf, 1024);

    const char *json = "{\"name\":\"Helvetica\",\"size\":16}"
    yyjson_doc *doc = yyjson_read_opts(json, strlen(json), 0, &alc, NULL);
    // the memory of `doc` is on the stack
 @endcode

 @warning This Allocator is not thread-safe.
 */
yyjson_api bool yyjson_alc_pool_init(yyjson_alc *alc, void *buf, size_t size);

/**
 A dynamic allocator.

 This allocator has a similar usage to the pool allocator above. However, when
 there is not enough memory, this allocator will dynamically request more memory
 using libc's `malloc` function, and frees it all at once when it is destroyed.

 @return A new dynamic allocator, or NULL if memory allocation failed.
 @note The returned value should be freed with `yyjson_alc_dyn_free()`.

 @warning This Allocator is not thread-safe.
 */
yyjson_api yyjson_alc *yyjson_alc_dyn_new(void);

/**
 Free a dynamic allocator which is created by `yyjson_alc_dyn_new()`.
 @param alc The dynamic allocator to be destroyed.
 */
yyjson_api void yyjson_alc_dyn_free(yyjson_alc *alc);



/*==============================================================================
 * Text Locating
 *============================================================================*/

/**
 Locate the line and column number for a byte position in a string.
 This can be used to get better description for error position.

 @param str The input string.
 @param len The byte length of the input string.
 @param pos The byte position within the input string.
 @param line A pointer to receive the line number, starting from 1.
 @param col  A pointer to receive the column number, starting from 1.
 @param chr  A pointer to receive the character index, starting from 0.
 @return true on success, false if `str` is NULL or `pos` is out of bounds.
 @note Line/column/character are calculated based on Unicode characters for
    compatibility with text editors. For multi-byte UTF-8 characters,
    the returned value may not directly correspond to the byte position.
 */
yyjson_api bool yyjson_locate_pos(const char *str, size_t len, size_t pos,
                                  size_t *line, size_t *col, size_t *chr);



/*==============================================================================
 * JSON Structure
 *============================================================================*/

/**
 An immutable document for reading JSON.
 This document holds memory for all its JSON values and strings. When it is no
 longer used, the user should call `yyjson_doc_free()` to free its memory.
 */
typedef struct yyjson_doc yyjson_doc;

/**
 An immutable value for reading JSON.
 A JSON Value has the same lifetime as its document. The memory is held by its
 document and and cannot be freed alone.
 */
typedef struct yyjson_val yyjson_val;

/**
 A mutable document for building JSON.
 This document holds memory for all its JSON values and strings. When it is no
 longer used, the user should call `yyjson_mut_doc_free()` to free its memory.
 */
typedef struct yyjson_mut_doc yyjson_mut_doc;

/**
 A mutable value for building JSON.
 A JSON Value has the same lifetime as its document. The memory is held by its
 document and and cannot be freed alone.
 */
typedef struct yyjson_mut_val yyjson_mut_val;



/*==============================================================================
 * JSON Reader API
 *============================================================================*/

/** Run-time options for JSON reader. */
typedef uint32_t yyjson_read_flag;

/** Default option (RFC 8259 compliant):
    - Read positive integer as uint64_t.
    - Read negative integer as int64_t.
    - Read floating-point number as double with round-to-nearest mode.
    - Read integer which cannot fit in uint64_t or int64_t as double.
    - Report error if double number is infinity.
    - Report error if string contains invalid UTF-8 character or BOM.
    - Report error on trailing commas, comments, inf and nan literals. */
static const yyjson_read_flag YYJSON_READ_NOFLAG                = 0;

/** Read the input data in-situ.
    This option allows the reader to modify and use input data to store string
    values, which can increase reading speed slightly.
    The caller should hold the input data before free the document.
    The input data must be padded by at least `YYJSON_PADDING_SIZE` bytes.
    For example: `[1,2]` should be `[1,2]\0\0\0\0`, input length should be 5. */
static const yyjson_read_flag YYJSON_READ_INSITU                = 1 << 0;

/** Stop when done instead of issuing an error if there's additional content
    after a JSON document. This option may be used to parse small pieces of JSON
    in larger data, such as `NDJSON`. */
static const yyjson_read_flag YYJSON_READ_STOP_WHEN_DONE        = 1 << 1;

/** Allow single trailing comma at the end of an object or array,
    such as `[1,2,3,]`, `{"a":1,"b":2,}` (non-standard). */
static const yyjson_read_flag YYJSON_READ_ALLOW_TRAILING_COMMAS = 1 << 2;

/** Allow C-style single line and multiple line comments (non-standard). */
static const yyjson_read_flag YYJSON_READ_ALLOW_COMMENTS        = 1 << 3;

/** Allow inf/nan number and literal, case-insensitive,
    such as 1e999, NaN, inf, -Infinity (non-standard). */
static const yyjson_read_flag YYJSON_READ_ALLOW_INF_AND_NAN     = 1 << 4;

/** Read all numbers as raw strings (value with `YYJSON_TYPE_RAW` type),
    inf/nan literal is also read as raw with `ALLOW_INF_AND_NAN` flag. */
static const yyjson_read_flag YYJSON_READ_NUMBER_AS_RAW         = 1 << 5;

/** Allow reading invalid unicode when parsing string values (non-standard).
    Invalid characters will be allowed to appear in the string values, but
    invalid escape sequences will still be reported as errors.
    This flag does not affect the performance of correctly encoded strings.

    @warning Strings in JSON values may contain incorrect encoding when this
    option is used, you need to handle these strings carefully to avoid security
    risks. */
static const yyjson_read_flag YYJSON_READ_ALLOW_INVALID_UNICODE = 1 << 6;

/** Read big numbers as raw strings. These big numbers include integers that
    cannot be represented by `int64_t` and `uint64_t`, and floating-point
    numbers that cannot be represented by finite `double`.
    The flag will be overridden by `YYJSON_READ_NUMBER_AS_RAW` flag. */
static const yyjson_read_flag YYJSON_READ_BIGNUM_AS_RAW         = 1 << 7;

/** Allow UTF-8 BOM and skip it before parsing if any (non-standard). */
static const yyjson_read_flag YYJSON_READ_ALLOW_BOM             = 1 << 8;



/** Result code for JSON reader. */
typedef uint32_t yyjson_read_code;

/** Success, no error. */
static const yyjson_read_code YYJSON_READ_SUCCESS                       = 0;

/** Invalid parameter, such as NULL input string or 0 input length. */
static const yyjson_read_code YYJSON_READ_ERROR_INVALID_PARAMETER       = 1;

/** Memory allocation failure occurs. */
static const yyjson_read_code YYJSON_READ_ERROR_MEMORY_ALLOCATION       = 2;

/** Input JSON string is empty. */
static const yyjson_read_code YYJSON_READ_ERROR_EMPTY_CONTENT           = 3;

/** Unexpected content after document, such as `[123]abc`. */
static const yyjson_read_code YYJSON_READ_ERROR_UNEXPECTED_CONTENT      = 4;

/** Unexpected ending, such as `[123`. */
static const yyjson_read_code YYJSON_READ_ERROR_UNEXPECTED_END          = 5;

/** Unexpected character inside the document, such as `[abc]`. */
static const yyjson_read_code YYJSON_READ_ERROR_UNEXPECTED_CHARACTER    = 6;

/** Invalid JSON structure, such as `[1,]`. */
static const yyjson_read_code YYJSON_READ_ERROR_JSON_STRUCTURE          = 7;

/** Invalid comment, such as unclosed multi-line comment. */
static const yyjson_read_code YYJSON_READ_ERROR_INVALID_COMMENT         = 8;

/** Invalid number, such as `123.e12`, `000`. */
static const yyjson_read_code YYJSON_READ_ERROR_INVALID_NUMBER          = 9;

/** Invalid string, such as invalid escaped character inside a string. */
static const yyjson_read_code YYJSON_READ_ERROR_INVALID_STRING          = 10;

/** Invalid JSON literal, such as `truu`. */
static const yyjson_read_code YYJSON_READ_ERROR_LITERAL                 = 11;

/** Failed to open a file. */
static const yyjson_read_code YYJSON_READ_ERROR_FILE_OPEN               = 12;

/** Failed to read a file. */
static const yyjson_read_code YYJSON_READ_ERROR_FILE_READ               = 13;

/** Unexpected ending during incremental parsing. Parsing state is saved. */
static const yyjson_read_code YYJSON_READ_ERROR_MORE                    = 14;

/** Error information for JSON reader. */
typedef struct yyjson_read_err {
    /** Error code, see `yyjson_read_code` for all possible values. */
    yyjson_read_code code;
    /** Error message, constant, no need to free (NULL if success). */
    const char *msg;
    /** Error byte position for input data (0 if success). */
    size_t pos;
} yyjson_read_err;



#if !defined(YYJSON_DISABLE_READER) || !YYJSON_DISABLE_READER

/**
 Read JSON with options.

 This function is thread-safe when:
 1. The `dat` is not modified by other threads.
 2. The `alc` is thread-safe or NULL.

 @param dat The JSON data (UTF-8 without BOM), null-terminator is not required.
    If this parameter is NULL, the function will fail and return NULL.
    The `dat` will not be modified without the flag `YYJSON_READ_INSITU`, so you
    can pass a `const char *` string and case it to `char *` if you don't use
    the `YYJSON_READ_INSITU` flag.
 @param len The length of JSON data in bytes.
    If this parameter is 0, the function will fail and return NULL.
 @param flg The JSON read options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON reader.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return A new JSON document, or NULL if an error occurs.
    When it's no longer needed, it should be freed with `yyjson_doc_free()`.
 */
yyjson_api yyjson_doc *yyjson_read_opts(char *dat,
                                        size_t len,
                                        yyjson_read_flag flg,
                                        const yyjson_alc *alc,
                                        yyjson_read_err *err);

/**
 Read a JSON file.

 This function is thread-safe when:
 1. The file is not modified by other threads.
 2. The `alc` is thread-safe or NULL.

 @param path The JSON file's path.
    This should be a null-terminated string using the system's native encoding.
    If this path is NULL or invalid, the function will fail and return NULL.
 @param flg The JSON read options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON reader.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return A new JSON document, or NULL if an error occurs.
    When it's no longer needed, it should be freed with `yyjson_doc_free()`.

 @warning On 32-bit operating system, files larger than 2GB may fail to read.
 */
yyjson_api yyjson_doc *yyjson_read_file(const char *path,
                                        yyjson_read_flag flg,
                                        const yyjson_alc *alc,
                                        yyjson_read_err *err);

/**
 Read JSON from a file pointer.

 @param fp The file pointer.
    The data will be read from the current position of the FILE to the end.
    If this fp is NULL or invalid, the function will fail and return NULL.
 @param flg The JSON read options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON reader.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return A new JSON document, or NULL if an error occurs.
    When it's no longer needed, it should be freed with `yyjson_doc_free()`.

 @warning On 32-bit operating system, files larger than 2GB may fail to read.
 */
yyjson_api yyjson_doc *yyjson_read_fp(FILE *fp,
                                      yyjson_read_flag flg,
                                      const yyjson_alc *alc,
                                      yyjson_read_err *err);

/**
 Read a JSON string.

 This function is thread-safe.

 @param dat The JSON data (UTF-8 without BOM), null-terminator is not required.
    If this parameter is NULL, the function will fail and return NULL.
 @param len The length of JSON data in bytes.
    If this parameter is 0, the function will fail and return NULL.
 @param flg The JSON read options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @return A new JSON document, or NULL if an error occurs.
    When it's no longer needed, it should be freed with `yyjson_doc_free()`.
 */
yyjson_api_inline yyjson_doc *yyjson_read(const char *dat,
                                          size_t len,
                                          yyjson_read_flag flg) {
    flg &= ~YYJSON_READ_INSITU; /* const string cannot be modified */
    return yyjson_read_opts((char *)(void *)(size_t)(const void *)dat,
                            len, flg, NULL, NULL);
}



#if !defined(YYJSON_DISABLE_INCR_READER) || !YYJSON_DISABLE_INCR_READER

/** Opaque state for incremental JSON reader. */
typedef struct yyjson_incr_state yyjson_incr_state;

/**
 Initialize state for incremental read.

 To read a large JSON document incrementally:
 1. Call `yyjson_incr_new()` to create the state for incremental reading.
 2. Call `yyjson_incr_read()` repeatedly.
 3. Call `yyjson_incr_free()` to free the state.

 @param buf The JSON data, null-terminator is not required.
    If this parameter is NULL, the function will fail and return NULL.
 @param buf_len The length of the JSON data in `buf`.
    If use `YYJSON_READ_INSITU`, `buf_len` should not include the padding size.
 @param flg The JSON read options.
    Multiple options can be combined with `|` operator.
 @param alc The memory allocator used by JSON reader.
    Pass NULL to use the libc's default allocator.
 @return A state for incremental reading.
    It should be freed with `yyjson_incr_free()`.
    NULL is returned if memory allocation fails.
*/
yyjson_api yyjson_incr_state *yyjson_incr_new(char *buf, size_t buf_len,
                                              yyjson_read_flag flg,
                                              const yyjson_alc *alc);

/**
 Performs incremental read of up to `len` bytes.

 If NULL is returned and `err->code` is set to `YYJSON_READ_ERROR_MORE`, it
 indicates that more data is required to continue parsing. Then, call this
 function again with incremented `len`. Continue until a document is returned or
 an error other than `YYJSON_READ_ERROR_MORE` is returned.

 Note: Parsing in very small increments is not efficient. An increment of
 several kilobytes or megabytes is recommended.

 @param state The state for incremental reading, created using
    `yyjson_incr_new()`.
 @param len The number of bytes of JSON data available to parse.
    If this parameter is 0, the function will fail and return NULL.
 @param err A pointer to receive error information.
 @return A new JSON document, or NULL if an error occurs.
    When the document is no longer needed, it should be freed with
    `yyjson_doc_free()`.
*/
yyjson_api yyjson_doc *yyjson_incr_read(yyjson_incr_state *state, size_t len,
                                        yyjson_read_err *err);

/** Release the incremental read state and free the memory. */
yyjson_api void yyjson_incr_free(yyjson_incr_state *state);

#endif /* YYJSON_DISABLE_INCR_READER */

/**
 Returns the size of maximum memory usage to read a JSON data.

 You may use this value to avoid malloc() or calloc() call inside the reader
 to get better performance, or read multiple JSON with one piece of memory.

 @param len The length of JSON data in bytes.
 @param flg The JSON read options.
 @return The maximum memory size to read this JSON, or 0 if overflow.

 @b Example
 @code
    // read multiple JSON with same pre-allocated memory

    char *dat1, *dat2, *dat3; // JSON data
    size_t len1, len2, len3; // JSON length
    size_t max_len = MAX(len1, MAX(len2, len3));
    yyjson_doc *doc;

    // use one allocator for multiple JSON
    size_t size = yyjson_read_max_memory_usage(max_len, 0);
    void *buf = malloc(size);
    yyjson_alc alc;
    yyjson_alc_pool_init(&alc, buf, size);

    // no more alloc() or realloc() call during reading
    doc = yyjson_read_opts(dat1, len1, 0, &alc, NULL);
    yyjson_doc_free(doc);
    doc = yyjson_read_opts(dat2, len2, 0, &alc, NULL);
    yyjson_doc_free(doc);
    doc = yyjson_read_opts(dat3, len3, 0, &alc, NULL);
    yyjson_doc_free(doc);

    free(buf);
 @endcode
 @see yyjson_alc_pool_init()
 */
yyjson_api_inline size_t yyjson_read_max_memory_usage(size_t len,
                                                      yyjson_read_flag flg) {
    /*
     1. The max value count is (json_size / 2 + 1),
        for example: "[1,2,3,4]" size is 9, value count is 5.
     2. Some broken JSON may cost more memory during reading, but fail at end,
        for example: "[[[[[[[[".
     3. yyjson use 16 bytes per value, see struct yyjson_val.
     4. yyjson use dynamic memory with a growth factor of 1.5.

     The max memory size is (json_size / 2 * 16 * 1.5 + padding).
     */
    size_t mul = (size_t)12 + !(flg & YYJSON_READ_INSITU);
    size_t pad = 256;
    size_t max = (size_t)(~(size_t)0);
    if (flg & YYJSON_READ_STOP_WHEN_DONE) len = len < 256 ? 256 : len;
    if (len >= (max - pad - mul) / mul) return 0;
    return len * mul + pad;
}

/**
 Read a JSON number.

 This function is thread-safe when data is not modified by other threads.

 @param dat The JSON data (UTF-8 without BOM), null-terminator is required.
    If this parameter is NULL, the function will fail and return NULL.
 @param val The output value where result is stored.
    If this parameter is NULL, the function will fail and return NULL.
    The value will hold either UINT or SINT or REAL number;
 @param flg The JSON read options.
    Multiple options can be combined with `|` operator. 0 means no options.
    Supports `YYJSON_READ_NUMBER_AS_RAW` and `YYJSON_READ_ALLOW_INF_AND_NAN`.
 @param alc The memory allocator used for long number.
    It is only used when the built-in floating point reader is disabled.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return If successful, a pointer to the character after the last character
    used in the conversion, NULL if an error occurs.
 */
yyjson_api const char *yyjson_read_number(const char *dat,
                                          yyjson_val *val,
                                          yyjson_read_flag flg,
                                          const yyjson_alc *alc,
                                          yyjson_read_err *err);

/** Same as `yyjson_read_number()`. */
yyjson_api_inline const char *yyjson_mut_read_number(const char *dat,
                                                     yyjson_mut_val *val,
                                                     yyjson_read_flag flg,
                                                     const yyjson_alc *alc,
                                                     yyjson_read_err *err) {
    return yyjson_read_number(dat, (yyjson_val *)val, flg, alc, err);
}

#endif /* YYJSON_DISABLE_READER) */



/*==============================================================================
 * JSON Writer API
 *============================================================================*/

/** Run-time options for JSON writer. */
typedef uint32_t yyjson_write_flag;

/** Default option:
    - Write JSON minify.
    - Report error on inf or nan number.
    - Report error on invalid UTF-8 string.
    - Do not escape unicode or slash. */
static const yyjson_write_flag YYJSON_WRITE_NOFLAG                  = 0;

/** Write JSON pretty with 4 space indent. */
static const yyjson_write_flag YYJSON_WRITE_PRETTY                  = 1 << 0;

/** Escape unicode as `uXXXX`, make the output ASCII only. */
static const yyjson_write_flag YYJSON_WRITE_ESCAPE_UNICODE          = 1 << 1;

/** Escape '/' as '\/'. */
static const yyjson_write_flag YYJSON_WRITE_ESCAPE_SLASHES          = 1 << 2;

/** Write inf and nan number as 'Infinity' and 'NaN' literal (non-standard). */
static const yyjson_write_flag YYJSON_WRITE_ALLOW_INF_AND_NAN       = 1 << 3;

/** Write inf and nan number as null literal.
    This flag will override `YYJSON_WRITE_ALLOW_INF_AND_NAN` flag. */
static const yyjson_write_flag YYJSON_WRITE_INF_AND_NAN_AS_NULL     = 1 << 4;

/** Allow invalid unicode when encoding string values (non-standard).
    Invalid characters in string value will be copied byte by byte.
    If `YYJSON_WRITE_ESCAPE_UNICODE` flag is also set, invalid character will be
    escaped as `U+FFFD` (replacement character).
    This flag does not affect the performance of correctly encoded strings. */
static const yyjson_write_flag YYJSON_WRITE_ALLOW_INVALID_UNICODE   = 1 << 5;

/** Write JSON pretty with 2 space indent.
    This flag will override `YYJSON_WRITE_PRETTY` flag. */
static const yyjson_write_flag YYJSON_WRITE_PRETTY_TWO_SPACES       = 1 << 6;

/** Adds a newline character `\n` at the end of the JSON.
    This can be helpful for text editors or NDJSON. */
static const yyjson_write_flag YYJSON_WRITE_NEWLINE_AT_END          = 1 << 7;



/** The highest 8 bits of `yyjson_write_flag` and real number value's `tag`
    are reserved for controlling the output format of floating-point numbers. */
#define YYJSON_WRITE_FP_FLAG_BITS 8

/** The highest 4 bits of flag are reserved for precision value. */
#define YYJSON_WRITE_FP_PREC_BITS 4

/** Write floating-point number using fixed-point notation.
    - This is similar to ECMAScript `Number.prototype.toFixed(prec)`,
      but with trailing zeros removed. The `prec` ranges from 1 to 15.
    - This will produce shorter output but may lose some precision. */
#define YYJSON_WRITE_FP_TO_FIXED(prec) ((yyjson_write_flag)( \
    (uint32_t)((uint32_t)(prec)) << (32 - 4) ))

/** Write floating-point numbers using single-precision (float).
    - This casts `double` to `float` before serialization.
    - This will produce shorter output, but may lose some precision.
    - This flag is ignored if `YYJSON_WRITE_FP_TO_FIXED(prec)` is also used. */
#define YYJSON_WRITE_FP_TO_FLOAT ((yyjson_write_flag)(1 << (32 - 5)))



/** Result code for JSON writer */
typedef uint32_t yyjson_write_code;

/** Success, no error. */
static const yyjson_write_code YYJSON_WRITE_SUCCESS                     = 0;

/** Invalid parameter, such as NULL document. */
static const yyjson_write_code YYJSON_WRITE_ERROR_INVALID_PARAMETER     = 1;

/** Memory allocation failure occurs. */
static const yyjson_write_code YYJSON_WRITE_ERROR_MEMORY_ALLOCATION     = 2;

/** Invalid value type in JSON document. */
static const yyjson_write_code YYJSON_WRITE_ERROR_INVALID_VALUE_TYPE    = 3;

/** NaN or Infinity number occurs. */
static const yyjson_write_code YYJSON_WRITE_ERROR_NAN_OR_INF            = 4;

/** Failed to open a file. */
static const yyjson_write_code YYJSON_WRITE_ERROR_FILE_OPEN             = 5;

/** Failed to write a file. */
static const yyjson_write_code YYJSON_WRITE_ERROR_FILE_WRITE            = 6;

/** Invalid unicode in string. */
static const yyjson_write_code YYJSON_WRITE_ERROR_INVALID_STRING        = 7;

/** Error information for JSON writer. */
typedef struct yyjson_write_err {
    /** Error code, see `yyjson_write_code` for all possible values. */
    yyjson_write_code code;
    /** Error message, constant, no need to free (NULL if success). */
    const char *msg;
} yyjson_write_err;



#if !defined(YYJSON_DISABLE_WRITER) || !YYJSON_DISABLE_WRITER

/*==============================================================================
 * JSON Document Writer API
 *============================================================================*/

/**
 Write a document to JSON string with options.

 This function is thread-safe when:
 The `alc` is thread-safe or NULL.

 @param doc The JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free() or alc->free().
 */
yyjson_api char *yyjson_write_opts(const yyjson_doc *doc,
                                   yyjson_write_flag flg,
                                   const yyjson_alc *alc,
                                   size_t *len,
                                   yyjson_write_err *err);

/**
 Write a document to JSON file with options.

 This function is thread-safe when:
 1. The file is not accessed by other threads.
 2. The `alc` is thread-safe or NULL.

 @param path The JSON file's path.
    This should be a null-terminated string using the system's native encoding.
    If this path is NULL or invalid, the function will fail and return false.
    If this file is not empty, the content will be discarded.
 @param doc The JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_write_file(const char *path,
                                  const yyjson_doc *doc,
                                  yyjson_write_flag flg,
                                  const yyjson_alc *alc,
                                  yyjson_write_err *err);

/**
 Write a document to file pointer with options.

 @param fp The file pointer.
    The data will be written to the current position of the file.
    If this fp is NULL or invalid, the function will fail and return false.
 @param doc The JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_write_fp(FILE *fp,
                                const yyjson_doc *doc,
                                yyjson_write_flag flg,
                                const yyjson_alc *alc,
                                yyjson_write_err *err);

/**
 Write a document to JSON string.

 This function is thread-safe.

 @param doc The JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free().
 */
yyjson_api_inline char *yyjson_write(const yyjson_doc *doc,
                                     yyjson_write_flag flg,
                                     size_t *len) {
    return yyjson_write_opts(doc, flg, NULL, len, NULL);
}



/**
 Write a document to JSON string with options.

 This function is thread-safe when:
 1. The `doc` is not modified by other threads.
 2. The `alc` is thread-safe or NULL.

 @param doc The mutable JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free() or alc->free().
 */
yyjson_api char *yyjson_mut_write_opts(const yyjson_mut_doc *doc,
                                       yyjson_write_flag flg,
                                       const yyjson_alc *alc,
                                       size_t *len,
                                       yyjson_write_err *err);

/**
 Write a document to JSON file with options.

 This function is thread-safe when:
 1. The file is not accessed by other threads.
 2. The `doc` is not modified by other threads.
 3. The `alc` is thread-safe or NULL.

 @param path The JSON file's path.
    This should be a null-terminated string using the system's native encoding.
    If this path is NULL or invalid, the function will fail and return false.
    If this file is not empty, the content will be discarded.
 @param doc The mutable JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_mut_write_file(const char *path,
                                      const yyjson_mut_doc *doc,
                                      yyjson_write_flag flg,
                                      const yyjson_alc *alc,
                                      yyjson_write_err *err);

/**
 Write a document to file pointer with options.

 @param fp The file pointer.
    The data will be written to the current position of the file.
    If this fp is NULL or invalid, the function will fail and return false.
 @param doc The mutable JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_mut_write_fp(FILE *fp,
                                    const yyjson_mut_doc *doc,
                                    yyjson_write_flag flg,
                                    const yyjson_alc *alc,
                                    yyjson_write_err *err);

/**
 Write a document to JSON string.

 This function is thread-safe when:
 The `doc` is not modified by other threads.

 @param doc The JSON document.
    If this doc is NULL or has no root, the function will fail and return false.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free().
 */
yyjson_api_inline char *yyjson_mut_write(const yyjson_mut_doc *doc,
                                         yyjson_write_flag flg,
                                         size_t *len) {
    return yyjson_mut_write_opts(doc, flg, NULL, len, NULL);
}



/*==============================================================================
 * JSON Value Writer API
 *============================================================================*/

/**
 Write a value to JSON string with options.

 This function is thread-safe when:
 The `alc` is thread-safe or NULL.

 @param val The JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free() or alc->free().
 */
yyjson_api char *yyjson_val_write_opts(const yyjson_val *val,
                                       yyjson_write_flag flg,
                                       const yyjson_alc *alc,
                                       size_t *len,
                                       yyjson_write_err *err);

/**
 Write a value to JSON file with options.

 This function is thread-safe when:
 1. The file is not accessed by other threads.
 2. The `alc` is thread-safe or NULL.

 @param path The JSON file's path.
    This should be a null-terminated string using the system's native encoding.
    If this path is NULL or invalid, the function will fail and return false.
    If this file is not empty, the content will be discarded.
 @param val The JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_val_write_file(const char *path,
                                      const yyjson_val *val,
                                      yyjson_write_flag flg,
                                      const yyjson_alc *alc,
                                      yyjson_write_err *err);

/**
 Write a value to file pointer with options.

 @param fp The file pointer.
    The data will be written to the current position of the file.
    If this path is NULL or invalid, the function will fail and return false.
 @param val The JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_val_write_fp(FILE *fp,
                                    const yyjson_val *val,
                                    yyjson_write_flag flg,
                                    const yyjson_alc *alc,
                                    yyjson_write_err *err);

/**
 Write a value to JSON string.

 This function is thread-safe.

 @param val The JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free().
 */
yyjson_api_inline char *yyjson_val_write(const yyjson_val *val,
                                         yyjson_write_flag flg,
                                         size_t *len) {
    return yyjson_val_write_opts(val, flg, NULL, len, NULL);
}

/**
 Write a value to JSON string with options.

 This function is thread-safe when:
 1. The `val` is not modified by other threads.
 2. The `alc` is thread-safe or NULL.

 @param val The mutable JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return  A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free() or alc->free().
 */
yyjson_api char *yyjson_mut_val_write_opts(const yyjson_mut_val *val,
                                           yyjson_write_flag flg,
                                           const yyjson_alc *alc,
                                           size_t *len,
                                           yyjson_write_err *err);

/**
 Write a value to JSON file with options.

 This function is thread-safe when:
 1. The file is not accessed by other threads.
 2. The `val` is not modified by other threads.
 3. The `alc` is thread-safe or NULL.

 @param path The JSON file's path.
    This should be a null-terminated string using the system's native encoding.
    If this path is NULL or invalid, the function will fail and return false.
    If this file is not empty, the content will be discarded.
 @param val The mutable JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_mut_val_write_file(const char *path,
                                          const yyjson_mut_val *val,
                                          yyjson_write_flag flg,
                                          const yyjson_alc *alc,
                                          yyjson_write_err *err);

/**
 Write a value to JSON file with options.

 @param fp The file pointer.
    The data will be written to the current position of the file.
    If this path is NULL or invalid, the function will fail and return false.
 @param val The mutable JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param alc The memory allocator used by JSON writer.
    Pass NULL to use the libc's default allocator.
 @param err A pointer to receive error information.
    Pass NULL if you don't need error information.
 @return true if successful, false if an error occurs.

 @warning On 32-bit operating system, files larger than 2GB may fail to write.
 */
yyjson_api bool yyjson_mut_val_write_fp(FILE *fp,
                                        const yyjson_mut_val *val,
                                        yyjson_write_flag flg,
                                        const yyjson_alc *alc,
                                        yyjson_write_err *err);

/**
 Write a value to JSON string.

 This function is thread-safe when:
 The `val` is not modified by other threads.

 @param val The JSON root value.
    If this parameter is NULL, the function will fail and return NULL.
 @param flg The JSON write options.
    Multiple options can be combined with `|` operator. 0 means no options.
 @param len A pointer to receive output length in bytes (not including the
    null-terminator). Pass NULL if you don't need length information.
 @return A new JSON string, or NULL if an error occurs.
    This string is encoded as UTF-8 with a null-terminator.
    When it's no longer needed, it should be freed with free().
 */
yyjson_api_inline char *yyjson_mut_val_write(const yyjson_mut_val *val,
                                             yyjson_write_flag flg,
                                             size_t *len) {
    return yyjson_mut_val_write_opts(val, flg, NULL, len, NULL);
}

/**
 Write a JSON number.

 @param val A JSON number value to be converted to a string.
    If this parameter is invalid, the function will fail and return NULL.
 @param buf A buffer to store the resulting null-terminated string.
    If this parameter is NULL, the function will fail and return NULL.
    For integer values, the buffer must be at least 21 bytes.
    For floating-point values, the buffer must be at least 40 bytes.
 @return On success, returns a pointer to the character after the last
    written character. On failure, returns NULL.
 @note
    - This function is thread-safe and does not allocate memory
        (when `YYJSON_DISABLE_FAST_FP_CONV` is not defined).
    - This function will fail and return NULL only in the following cases:
        1) `val` or `buf` is NULL;
        2) `val` is not a number type;
        3) `val` is `inf` or `nan`, and non-standard JSON is explicitly disabled
            via the `YYJSON_DISABLE_NON_STANDARD` flag.
 */
yyjson_api char *yyjson_write_number(const yyjson_val *val, char *buf);

/** Same as `yyjson_write_number()`. */
yyjson_api_inline char *yyjson_mut_write_number(const yyjson_mut_val *val,
                                                char *buf) {
    return yyjson_write_number((const yyjson_val *)val, buf);
}

#endif /* YYJSON_DISABLE_WRITER */



/*==============================================================================
 * JSON Document API
 *============================================================================*/

/** Returns the root value of this JSON document.
    Returns NULL if `doc` is NULL. */
yyjson_api_inline yyjson_val *yyjson_doc_get_root(yyjson_doc *doc);

/** Returns read size of input JSON data.
    Returns 0 if `doc` is NULL.
    For example: the read size of `[1,2,3]` is 7 bytes.  */
yyjson_api_inline size_t yyjson_doc_get_read_size(yyjson_doc *doc);

/** Returns total value count in this JSON document.
    Returns 0 if `doc` is NULL.
    For example: the value count of `[1,2,3]` is 4. */
yyjson_api_inline size_t yyjson_doc_get_val_count(yyjson_doc *doc);

/** Release the JSON document and free the memory.
    After calling this function, the `doc` and all values from the `doc` are no
    longer available. This function will do nothing if the `doc` is NULL. */
yyjson_api_inline void yyjson_doc_free(yyjson_doc *doc);



/*==============================================================================
 * JSON Value Type API
 *============================================================================*/

/** Returns whether the JSON value is raw.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_raw(yyjson_val *val);

/** Returns whether the JSON value is `null`.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_null(yyjson_val *val);

/** Returns whether the JSON value is `true`.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_true(yyjson_val *val);

/** Returns whether the JSON value is `false`.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_false(yyjson_val *val);

/** Returns whether the JSON value is bool (true/false).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_bool(yyjson_val *val);

/** Returns whether the JSON value is unsigned integer (uint64_t).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_uint(yyjson_val *val);

/** Returns whether the JSON value is signed integer (int64_t).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_sint(yyjson_val *val);

/** Returns whether the JSON value is integer (uint64_t/int64_t).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_int(yyjson_val *val);

/** Returns whether the JSON value is real number (double).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_real(yyjson_val *val);

/** Returns whether the JSON value is number (uint64_t/int64_t/double).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_num(yyjson_val *val);

/** Returns whether the JSON value is string.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_str(yyjson_val *val);

/** Returns whether the JSON value is array.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_arr(yyjson_val *val);

/** Returns whether the JSON value is object.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_obj(yyjson_val *val);

/** Returns whether the JSON value is container (array/object).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_is_ctn(yyjson_val *val);



/*==============================================================================
 * JSON Value Content API
 *============================================================================*/

/** Returns the JSON value's type.
    Returns YYJSON_TYPE_NONE if `val` is NULL. */
yyjson_api_inline yyjson_type yyjson_get_type(yyjson_val *val);

/** Returns the JSON value's subtype.
    Returns YYJSON_SUBTYPE_NONE if `val` is NULL. */
yyjson_api_inline yyjson_subtype yyjson_get_subtype(yyjson_val *val);

/** Returns the JSON value's tag.
    Returns 0 if `val` is NULL. */
yyjson_api_inline uint8_t yyjson_get_tag(yyjson_val *val);

/** Returns the JSON value's type description.
    The return value should be one of these strings: "raw", "null", "string",
    "array", "object", "true", "false", "uint", "sint", "real", "unknown". */
yyjson_api_inline const char *yyjson_get_type_desc(yyjson_val *val);

/** Returns the content if the value is raw.
    Returns NULL if `val` is NULL or type is not raw. */
yyjson_api_inline const char *yyjson_get_raw(yyjson_val *val);

/** Returns the content if the value is bool.
    Returns NULL if `val` is NULL or type is not bool. */
yyjson_api_inline bool yyjson_get_bool(yyjson_val *val);

/** Returns the content and cast to uint64_t.
    Returns 0 if `val` is NULL or type is not integer(sint/uint). */
yyjson_api_inline uint64_t yyjson_get_uint(yyjson_val *val);

/** Returns the content and cast to int64_t.
    Returns 0 if `val` is NULL or type is not integer(sint/uint). */
yyjson_api_inline int64_t yyjson_get_sint(yyjson_val *val);

/** Returns the content and cast to int.
    Returns 0 if `val` is NULL or type is not integer(sint/uint). */
yyjson_api_inline int yyjson_get_int(yyjson_val *val);

/** Returns the content if the value is real number, or 0.0 on error.
    Returns 0.0 if `val` is NULL or type is not real(double). */
yyjson_api_inline double yyjson_get_real(yyjson_val *val);

/** Returns the content and typecast to `double` if the value is number.
    Returns 0.0 if `val` is NULL or type is not number(uint/sint/real). */
yyjson_api_inline double yyjson_get_num(yyjson_val *val);

/** Returns the content if the value is string.
    Returns NULL if `val` is NULL or type is not string. */
yyjson_api_inline const char *yyjson_get_str(yyjson_val *val);

/** Returns the content length (string length, array size, object size.
    Returns 0 if `val` is NULL or type is not string/array/object. */
yyjson_api_inline size_t yyjson_get_len(yyjson_val *val);

/** Returns whether the JSON value is equals to a string.
    Returns false if input is NULL or type is not string. */
yyjson_api_inline bool yyjson_equals_str(yyjson_val *val, const char *str);

/** Returns whether the JSON value is equals to a string.
    The `str` should be a UTF-8 string, null-terminator is not required.
    Returns false if input is NULL or type is not string. */
yyjson_api_inline bool yyjson_equals_strn(yyjson_val *val, const char *str,
                                          size_t len);

/** Returns whether two JSON values are equal (deep compare).
    Returns false if input is NULL.
    @note the result may be inaccurate if object has duplicate keys.
    @warning This function is recursive and may cause a stack overflow
        if the object level is too deep. */
yyjson_api_inline bool yyjson_equals(yyjson_val *lhs, yyjson_val *rhs);

/** Set the value to raw.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_raw(yyjson_val *val,
                                      const char *raw, size_t len);

/** Set the value to null.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_null(yyjson_val *val);

/** Set the value to bool.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_bool(yyjson_val *val, bool num);

/** Set the value to uint.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_uint(yyjson_val *val, uint64_t num);

/** Set the value to sint.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_sint(yyjson_val *val, int64_t num);

/** Set the value to int.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_int(yyjson_val *val, int num);

/** Set the value to float.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_float(yyjson_val *val, float num);

/** Set the value to double.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_double(yyjson_val *val, double num);

/** Set the value to real.
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_real(yyjson_val *val, double num);

/** Set the floating-point number's output format to fixed-point notation.
    Returns false if input is NULL or `val` is not real type.
    @see YYJSON_WRITE_FP_TO_FIXED flag.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_fp_to_fixed(yyjson_val *val, int prec);

/** Set the floating-point number's output format to single-precision.
    Returns false if input is NULL or `val` is not real type.
    @see YYJSON_WRITE_FP_TO_FLOAT flag.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_fp_to_float(yyjson_val *val, bool flt);

/** Set the value to string (null-terminated).
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_str(yyjson_val *val, const char *str);

/** Set the value to string (with length).
    Returns false if input is NULL or `val` is object or array.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_strn(yyjson_val *val,
                                       const char *str, size_t len);

/** Marks this string as not needing to be escaped during JSON writing.
    This can be used to avoid the overhead of escaping if the string contains
    only characters that do not require escaping.
    Returns false if input is NULL or `val` is not string.
    @see YYJSON_SUBTYPE_NOESC subtype.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_set_str_noesc(yyjson_val *val, bool noesc);



/*==============================================================================
 * JSON Array API
 *============================================================================*/

/** Returns the number of elements in this array.
    Returns 0 if `arr` is NULL or type is not array. */
yyjson_api_inline size_t yyjson_arr_size(yyjson_val *arr);

/** Returns the element at the specified position in this array.
    Returns NULL if array is NULL/empty or the index is out of bounds.
    @warning This function takes a linear search time if array is not flat.
        For example: `[1,{},3]` is flat, `[1,[2],3]` is not flat. */
yyjson_api_inline yyjson_val *yyjson_arr_get(yyjson_val *arr, size_t idx);

/** Returns the first element of this array.
    Returns NULL if `arr` is NULL/empty or type is not array. */
yyjson_api_inline yyjson_val *yyjson_arr_get_first(yyjson_val *arr);

/** Returns the last element of this array.
    Returns NULL if `arr` is NULL/empty or type is not array.
    @warning This function takes a linear search time if array is not flat.
        For example: `[1,{},3]` is flat, `[1,[2],3]` is not flat.*/
yyjson_api_inline yyjson_val *yyjson_arr_get_last(yyjson_val *arr);



/*==============================================================================
 * JSON Array Iterator API
 *============================================================================*/

/**
 A JSON array iterator.

 @b Example
 @code
    yyjson_val *val;
    yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
    while ((val = yyjson_arr_iter_next(&iter))) {
        your_func(val);
    }
 @endcode
 */
typedef struct yyjson_arr_iter {
    size_t idx; /**< next value's index */
    size_t max; /**< maximum index (arr.size) */
    yyjson_val *cur; /**< next value */
} yyjson_arr_iter;

/**
 Initialize an iterator for this array.

 @param arr The array to be iterated over.
    If this parameter is NULL or not an array, `iter` will be set to empty.
 @param iter The iterator to be initialized.
    If this parameter is NULL, the function will fail and return false.
 @return true if the `iter` has been successfully initialized.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline bool yyjson_arr_iter_init(yyjson_val *arr,
                                            yyjson_arr_iter *iter);

/**
 Create an iterator with an array , same as `yyjson_arr_iter_init()`.

 @param arr The array to be iterated over.
    If this parameter is NULL or not an array, an empty iterator will returned.
 @return A new iterator for the array.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline yyjson_arr_iter yyjson_arr_iter_with(yyjson_val *arr);

/**
 Returns whether the iteration has more elements.
 If `iter` is NULL, this function will return false.
 */
yyjson_api_inline bool yyjson_arr_iter_has_next(yyjson_arr_iter *iter);

/**
 Returns the next element in the iteration, or NULL on end.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_val *yyjson_arr_iter_next(yyjson_arr_iter *iter);

/**
 Macro for iterating over an array.
 It works like iterator, but with a more intuitive API.

 @b Example
 @code
    size_t idx, max;
    yyjson_val *val;
    yyjson_arr_foreach(arr, idx, max, val) {
        your_func(idx, val);
    }
 @endcode
 */
#define yyjson_arr_foreach(arr, idx, max, val) \
    for ((idx) = 0, \
        (max) = yyjson_arr_size(arr), \
        (val) = yyjson_arr_get_first(arr); \
        (idx) < (max); \
        (idx)++, \
        (val) = unsafe_yyjson_get_next(val))



/*==============================================================================
 * JSON Object API
 *============================================================================*/

/** Returns the number of key-value pairs in this object.
    Returns 0 if `obj` is NULL or type is not object. */
yyjson_api_inline size_t yyjson_obj_size(yyjson_val *obj);

/** Returns the value to which the specified key is mapped.
    Returns NULL if this object contains no mapping for the key.
    Returns NULL if `obj/key` is NULL, or type is not object.

    The `key` should be a null-terminated UTF-8 string.

    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_val *yyjson_obj_get(yyjson_val *obj, const char *key);

/** Returns the value to which the specified key is mapped.
    Returns NULL if this object contains no mapping for the key.
    Returns NULL if `obj/key` is NULL, or type is not object.

    The `key` should be a UTF-8 string, null-terminator is not required.
    The `key_len` should be the length of the key, in bytes.

    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_val *yyjson_obj_getn(yyjson_val *obj, const char *key,
                                              size_t key_len);



/*==============================================================================
 * JSON Object Iterator API
 *============================================================================*/

/**
 A JSON object iterator.

 @b Example
 @code
    yyjson_val *key, *val;
    yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
    while ((key = yyjson_obj_iter_next(&iter))) {
        val = yyjson_obj_iter_get_val(key);
        your_func(key, val);
    }
 @endcode

 If the ordering of the keys is known at compile-time, you can use this method
 to speed up value lookups:
 @code
    // {"k1":1, "k2": 3, "k3": 3}
    yyjson_val *key, *val;
    yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
    yyjson_val *v1 = yyjson_obj_iter_get(&iter, "k1");
    yyjson_val *v3 = yyjson_obj_iter_get(&iter, "k3");
 @endcode
 @see yyjson_obj_iter_get() and yyjson_obj_iter_getn()
 */
typedef struct yyjson_obj_iter {
    size_t idx; /**< next key's index */
    size_t max; /**< maximum key index (obj.size) */
    yyjson_val *cur; /**< next key */
    yyjson_val *obj; /**< the object being iterated */
} yyjson_obj_iter;

/**
 Initialize an iterator for this object.

 @param obj The object to be iterated over.
    If this parameter is NULL or not an object, `iter` will be set to empty.
 @param iter The iterator to be initialized.
    If this parameter is NULL, the function will fail and return false.
 @return true if the `iter` has been successfully initialized.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline bool yyjson_obj_iter_init(yyjson_val *obj,
                                            yyjson_obj_iter *iter);

/**
 Create an iterator with an object, same as `yyjson_obj_iter_init()`.

 @param obj The object to be iterated over.
    If this parameter is NULL or not an object, an empty iterator will returned.
 @return A new iterator for the object.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline yyjson_obj_iter yyjson_obj_iter_with(yyjson_val *obj);

/**
 Returns whether the iteration has more elements.
 If `iter` is NULL, this function will return false.
 */
yyjson_api_inline bool yyjson_obj_iter_has_next(yyjson_obj_iter *iter);

/**
 Returns the next key in the iteration, or NULL on end.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_val *yyjson_obj_iter_next(yyjson_obj_iter *iter);

/**
 Returns the value for key inside the iteration.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_val *yyjson_obj_iter_get_val(yyjson_val *key);

/**
 Iterates to a specified key and returns the value.

 This function does the same thing as `yyjson_obj_get()`, but is much faster
 if the ordering of the keys is known at compile-time and you are using the same
 order to look up the values. If the key exists in this object, then the
 iterator will stop at the next key, otherwise the iterator will not change and
 NULL is returned.

 @param iter The object iterator, should not be NULL.
 @param key The key, should be a UTF-8 string with null-terminator.
 @return The value to which the specified key is mapped.
    NULL if this object contains no mapping for the key or input is invalid.

 @warning This function takes a linear search time if the key is not nearby.
 */
yyjson_api_inline yyjson_val *yyjson_obj_iter_get(yyjson_obj_iter *iter,
                                                  const char *key);

/**
 Iterates to a specified key and returns the value.

 This function does the same thing as `yyjson_obj_getn()`, but is much faster
 if the ordering of the keys is known at compile-time and you are using the same
 order to look up the values. If the key exists in this object, then the
 iterator will stop at the next key, otherwise the iterator will not change and
 NULL is returned.

 @param iter The object iterator, should not be NULL.
 @param key The key, should be a UTF-8 string, null-terminator is not required.
 @param key_len The the length of `key`, in bytes.
 @return The value to which the specified key is mapped.
    NULL if this object contains no mapping for the key or input is invalid.

 @warning This function takes a linear search time if the key is not nearby.
 */
yyjson_api_inline yyjson_val *yyjson_obj_iter_getn(yyjson_obj_iter *iter,
                                                   const char *key,
                                                   size_t key_len);

/**
 Macro for iterating over an object.
 It works like iterator, but with a more intuitive API.

 @b Example
 @code
    size_t idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(obj, idx, max, key, val) {
        your_func(key, val);
    }
 @endcode
 */
#define yyjson_obj_foreach(obj, idx, max, key, val) \
    for ((idx) = 0, \
        (max) = yyjson_obj_size(obj), \
        (key) = (obj) ? unsafe_yyjson_get_first(obj) : NULL, \
        (val) = (key) + 1; \
        (idx) < (max); \
        (idx)++, \
        (key) = unsafe_yyjson_get_next(val), \
        (val) = (key) + 1)



/*==============================================================================
 * Mutable JSON Document API
 *============================================================================*/

/** Returns the root value of this JSON document.
    Returns NULL if `doc` is NULL. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_get_root(yyjson_mut_doc *doc);

/** Sets the root value of this JSON document.
    Pass NULL to clear root value of the document. */
yyjson_api_inline void yyjson_mut_doc_set_root(yyjson_mut_doc *doc,
                                               yyjson_mut_val *root);

/**
 Set the string pool size for a mutable document.
 This function does not allocate memory immediately, but uses the size when
 the next memory allocation is needed.

 If the caller knows the approximate bytes of strings that the document needs to
 store (e.g. copy string with `yyjson_mut_strcpy` function), setting a larger
 size can avoid multiple memory allocations and improve performance.

 @param doc The mutable document.
 @param len The desired string pool size in bytes (total string length).
 @return true if successful, false if size is 0 or overflow.
 */
yyjson_api bool yyjson_mut_doc_set_str_pool_size(yyjson_mut_doc *doc,
                                                 size_t len);

/**
 Set the value pool size for a mutable document.
 This function does not allocate memory immediately, but uses the size when
 the next memory allocation is needed.

 If the caller knows the approximate number of values that the document needs to
 store (e.g. create new value with `yyjson_mut_xxx` functions), setting a larger
 size can avoid multiple memory allocations and improve performance.

 @param doc The mutable document.
 @param count The desired value pool size (number of `yyjson_mut_val`).
 @return true if successful, false if size is 0 or overflow.
 */
yyjson_api bool yyjson_mut_doc_set_val_pool_size(yyjson_mut_doc *doc,
                                                 size_t count);

/** Release the JSON document and free the memory.
    After calling this function, the `doc` and all values from the `doc` are no
    longer available. This function will do nothing if the `doc` is NULL.  */
yyjson_api void yyjson_mut_doc_free(yyjson_mut_doc *doc);

/** Creates and returns a new mutable JSON document, returns NULL on error.
    If allocator is NULL, the default allocator will be used. */
yyjson_api yyjson_mut_doc *yyjson_mut_doc_new(const yyjson_alc *alc);

/** Copies and returns a new mutable document from input, returns NULL on error.
    This makes a `deep-copy` on the immutable document.
    If allocator is NULL, the default allocator will be used.
    @note `imut_doc` -> `mut_doc`. */
yyjson_api yyjson_mut_doc *yyjson_doc_mut_copy(yyjson_doc *doc,
                                               const yyjson_alc *alc);

/** Copies and returns a new mutable document from input, returns NULL on error.
    This makes a `deep-copy` on the mutable document.
    If allocator is NULL, the default allocator will be used.
    @note `mut_doc` -> `mut_doc`. */
yyjson_api yyjson_mut_doc *yyjson_mut_doc_mut_copy(yyjson_mut_doc *doc,
                                                   const yyjson_alc *alc);

/** Copies and returns a new mutable value from input, returns NULL on error.
    This makes a `deep-copy` on the immutable value.
    The memory was managed by mutable document.
    @note `imut_val` -> `mut_val`. */
yyjson_api yyjson_mut_val *yyjson_val_mut_copy(yyjson_mut_doc *doc,
                                               yyjson_val *val);

/** Copies and returns a new mutable value from input, returns NULL on error.
    This makes a `deep-copy` on the mutable value.
    The memory was managed by mutable document.
    @note `mut_val` -> `mut_val`.
    @warning This function is recursive and may cause a stack overflow
        if the object level is too deep. */
yyjson_api yyjson_mut_val *yyjson_mut_val_mut_copy(yyjson_mut_doc *doc,
                                                   yyjson_mut_val *val);

/** Copies and returns a new immutable document from input,
    returns NULL on error. This makes a `deep-copy` on the mutable document.
    The returned document should be freed with `yyjson_doc_free()`.
    @note `mut_doc` -> `imut_doc`.
    @warning This function is recursive and may cause a stack overflow
        if the object level is too deep. */
yyjson_api yyjson_doc *yyjson_mut_doc_imut_copy(yyjson_mut_doc *doc,
                                                const yyjson_alc *alc);

/** Copies and returns a new immutable document from input,
    returns NULL on error. This makes a `deep-copy` on the mutable value.
    The returned document should be freed with `yyjson_doc_free()`.
    @note `mut_val` -> `imut_doc`.
    @warning This function is recursive and may cause a stack overflow
        if the object level is too deep. */
yyjson_api yyjson_doc *yyjson_mut_val_imut_copy(yyjson_mut_val *val,
                                                const yyjson_alc *alc);



/*==============================================================================
 * Mutable JSON Value Type API
 *============================================================================*/

/** Returns whether the JSON value is raw.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_raw(yyjson_mut_val *val);

/** Returns whether the JSON value is `null`.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_null(yyjson_mut_val *val);

/** Returns whether the JSON value is `true`.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_true(yyjson_mut_val *val);

/** Returns whether the JSON value is `false`.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_false(yyjson_mut_val *val);

/** Returns whether the JSON value is bool (true/false).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_bool(yyjson_mut_val *val);

/** Returns whether the JSON value is unsigned integer (uint64_t).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_uint(yyjson_mut_val *val);

/** Returns whether the JSON value is signed integer (int64_t).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_sint(yyjson_mut_val *val);

/** Returns whether the JSON value is integer (uint64_t/int64_t).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_int(yyjson_mut_val *val);

/** Returns whether the JSON value is real number (double).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_real(yyjson_mut_val *val);

/** Returns whether the JSON value is number (uint/sint/real).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_num(yyjson_mut_val *val);

/** Returns whether the JSON value is string.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_str(yyjson_mut_val *val);

/** Returns whether the JSON value is array.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_arr(yyjson_mut_val *val);

/** Returns whether the JSON value is object.
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_obj(yyjson_mut_val *val);

/** Returns whether the JSON value is container (array/object).
    Returns false if `val` is NULL. */
yyjson_api_inline bool yyjson_mut_is_ctn(yyjson_mut_val *val);



/*==============================================================================
 * Mutable JSON Value Content API
 *============================================================================*/

/** Returns the JSON value's type.
    Returns `YYJSON_TYPE_NONE` if `val` is NULL. */
yyjson_api_inline yyjson_type yyjson_mut_get_type(yyjson_mut_val *val);

/** Returns the JSON value's subtype.
    Returns `YYJSON_SUBTYPE_NONE` if `val` is NULL. */
yyjson_api_inline yyjson_subtype yyjson_mut_get_subtype(yyjson_mut_val *val);

/** Returns the JSON value's tag.
    Returns 0 if `val` is NULL. */
yyjson_api_inline uint8_t yyjson_mut_get_tag(yyjson_mut_val *val);

/** Returns the JSON value's type description.
    The return value should be one of these strings: "raw", "null", "string",
    "array", "object", "true", "false", "uint", "sint", "real", "unknown". */
yyjson_api_inline const char *yyjson_mut_get_type_desc(yyjson_mut_val *val);

/** Returns the content if the value is raw.
    Returns NULL if `val` is NULL or type is not raw. */
yyjson_api_inline const char *yyjson_mut_get_raw(yyjson_mut_val *val);

/** Returns the content if the value is bool.
    Returns NULL if `val` is NULL or type is not bool. */
yyjson_api_inline bool yyjson_mut_get_bool(yyjson_mut_val *val);

/** Returns the content and cast to uint64_t.
    Returns 0 if `val` is NULL or type is not integer(sint/uint). */
yyjson_api_inline uint64_t yyjson_mut_get_uint(yyjson_mut_val *val);

/** Returns the content and cast to int64_t.
    Returns 0 if `val` is NULL or type is not integer(sint/uint). */
yyjson_api_inline int64_t yyjson_mut_get_sint(yyjson_mut_val *val);

/** Returns the content and cast to int.
    Returns 0 if `val` is NULL or type is not integer(sint/uint). */
yyjson_api_inline int yyjson_mut_get_int(yyjson_mut_val *val);

/** Returns the content if the value is real number.
    Returns 0.0 if `val` is NULL or type is not real(double). */
yyjson_api_inline double yyjson_mut_get_real(yyjson_mut_val *val);

/** Returns the content and typecast to `double` if the value is number.
    Returns 0.0 if `val` is NULL or type is not number(uint/sint/real). */
yyjson_api_inline double yyjson_mut_get_num(yyjson_mut_val *val);

/** Returns the content if the value is string.
    Returns NULL if `val` is NULL or type is not string. */
yyjson_api_inline const char *yyjson_mut_get_str(yyjson_mut_val *val);

/** Returns the content length (string length, array size, object size.
    Returns 0 if `val` is NULL or type is not string/array/object. */
yyjson_api_inline size_t yyjson_mut_get_len(yyjson_mut_val *val);

/** Returns whether the JSON value is equals to a string.
    The `str` should be a null-terminated UTF-8 string.
    Returns false if input is NULL or type is not string. */
yyjson_api_inline bool yyjson_mut_equals_str(yyjson_mut_val *val,
                                             const char *str);

/** Returns whether the JSON value is equals to a string.
    The `str` should be a UTF-8 string, null-terminator is not required.
    Returns false if input is NULL or type is not string. */
yyjson_api_inline bool yyjson_mut_equals_strn(yyjson_mut_val *val,
                                              const char *str, size_t len);

/** Returns whether two JSON values are equal (deep compare).
    Returns false if input is NULL.
    @note the result may be inaccurate if object has duplicate keys.
    @warning This function is recursive and may cause a stack overflow
        if the object level is too deep. */
yyjson_api_inline bool yyjson_mut_equals(yyjson_mut_val *lhs,
                                         yyjson_mut_val *rhs);

/** Set the value to raw.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_raw(yyjson_mut_val *val,
                                          const char *raw, size_t len);

/** Set the value to null.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_null(yyjson_mut_val *val);

/** Set the value to bool.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_bool(yyjson_mut_val *val, bool num);

/** Set the value to uint.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_uint(yyjson_mut_val *val, uint64_t num);

/** Set the value to sint.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_sint(yyjson_mut_val *val, int64_t num);

/** Set the value to int.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_int(yyjson_mut_val *val, int num);

/** Set the value to float.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_float(yyjson_mut_val *val, float num);

/** Set the value to double.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_double(yyjson_mut_val *val, double num);

/** Set the value to real.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_real(yyjson_mut_val *val, double num);

/** Set the floating-point number's output format to fixed-point notation.
    Returns false if input is NULL or `val` is not real type.
    @see YYJSON_WRITE_FP_TO_FIXED flag.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_mut_set_fp_to_fixed(yyjson_mut_val *val,
                                                  int prec);

/** Set the floating-point number's output format to single-precision.
    Returns false if input is NULL or `val` is not real type.
    @see YYJSON_WRITE_FP_TO_FLOAT flag.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_mut_set_fp_to_float(yyjson_mut_val *val,
                                                  bool flt);

/** Set the value to string (null-terminated).
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_str(yyjson_mut_val *val, const char *str);

/** Set the value to string (with length).
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_strn(yyjson_mut_val *val,
                                           const char *str, size_t len);

/** Marks this string as not needing to be escaped during JSON writing.
    This can be used to avoid the overhead of escaping if the string contains
    only characters that do not require escaping.
    Returns false if input is NULL or `val` is not string.
    @see YYJSON_SUBTYPE_NOESC subtype.
    @warning This will modify the `immutable` value, use with caution. */
yyjson_api_inline bool yyjson_mut_set_str_noesc(yyjson_mut_val *val,
                                                bool noesc);

/** Set the value to array.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_arr(yyjson_mut_val *val);

/** Set the value to array.
    Returns false if input is NULL.
    @warning This function should not be used on an existing object or array. */
yyjson_api_inline bool yyjson_mut_set_obj(yyjson_mut_val *val);



/*==============================================================================
 * Mutable JSON Value Creation API
 *============================================================================*/

/** Creates and returns a raw value, returns NULL on error.
    The `str` should be a null-terminated UTF-8 string.

    @warning The input string is not copied, you should keep this string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_raw(yyjson_mut_doc *doc,
                                                 const char *str);

/** Creates and returns a raw value, returns NULL on error.
    The `str` should be a UTF-8 string, null-terminator is not required.

    @warning The input string is not copied, you should keep this string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_rawn(yyjson_mut_doc *doc,
                                                  const char *str,
                                                  size_t len);

/** Creates and returns a raw value, returns NULL on error.
    The `str` should be a null-terminated UTF-8 string.
    The input string is copied and held by the document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_rawcpy(yyjson_mut_doc *doc,
                                                    const char *str);

/** Creates and returns a raw value, returns NULL on error.
    The `str` should be a UTF-8 string, null-terminator is not required.
    The input string is copied and held by the document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_rawncpy(yyjson_mut_doc *doc,
                                                     const char *str,
                                                     size_t len);

/** Creates and returns a null value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_null(yyjson_mut_doc *doc);

/** Creates and returns a true value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_true(yyjson_mut_doc *doc);

/** Creates and returns a false value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_false(yyjson_mut_doc *doc);

/** Creates and returns a bool value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_bool(yyjson_mut_doc *doc,
                                                  bool val);

/** Creates and returns an unsigned integer value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_uint(yyjson_mut_doc *doc,
                                                  uint64_t num);

/** Creates and returns a signed integer value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_sint(yyjson_mut_doc *doc,
                                                  int64_t num);

/** Creates and returns a signed integer value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_int(yyjson_mut_doc *doc,
                                                 int64_t num);

/** Creates and returns a float number value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_float(yyjson_mut_doc *doc,
                                                   float num);

/** Creates and returns a double number value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_double(yyjson_mut_doc *doc,
                                                    double num);

/** Creates and returns a real number value, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_real(yyjson_mut_doc *doc,
                                                  double num);

/** Creates and returns a string value, returns NULL on error.
    The `str` should be a null-terminated UTF-8 string.
    @warning The input string is not copied, you should keep this string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_str(yyjson_mut_doc *doc,
                                                 const char *str);

/** Creates and returns a string value, returns NULL on error.
    The `str` should be a UTF-8 string, null-terminator is not required.
    @warning The input string is not copied, you should keep this string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_strn(yyjson_mut_doc *doc,
                                                  const char *str,
                                                  size_t len);

/** Creates and returns a string value, returns NULL on error.
    The `str` should be a null-terminated UTF-8 string.
    The input string is copied and held by the document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_strcpy(yyjson_mut_doc *doc,
                                                    const char *str);

/** Creates and returns a string value, returns NULL on error.
    The `str` should be a UTF-8 string, null-terminator is not required.
    The input string is copied and held by the document. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_strncpy(yyjson_mut_doc *doc,
                                                     const char *str,
                                                     size_t len);



/*==============================================================================
 * Mutable JSON Array API
 *============================================================================*/

/** Returns the number of elements in this array.
    Returns 0 if `arr` is NULL or type is not array. */
yyjson_api_inline size_t yyjson_mut_arr_size(yyjson_mut_val *arr);

/** Returns the element at the specified position in this array.
    Returns NULL if array is NULL/empty or the index is out of bounds.
    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_get(yyjson_mut_val *arr,
                                                     size_t idx);

/** Returns the first element of this array.
    Returns NULL if `arr` is NULL/empty or type is not array. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_get_first(yyjson_mut_val *arr);

/** Returns the last element of this array.
    Returns NULL if `arr` is NULL/empty or type is not array. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_get_last(yyjson_mut_val *arr);



/*==============================================================================
 * Mutable JSON Array Iterator API
 *============================================================================*/

/**
 A mutable JSON array iterator.

 @warning You should not modify the array while iterating over it, but you can
    use `yyjson_mut_arr_iter_remove()` to remove current value.

 @b Example
 @code
    yyjson_mut_val *val;
    yyjson_mut_arr_iter iter = yyjson_mut_arr_iter_with(arr);
    while ((val = yyjson_mut_arr_iter_next(&iter))) {
        your_func(val);
        if (your_val_is_unused(val)) {
            yyjson_mut_arr_iter_remove(&iter);
        }
    }
 @endcode
 */
typedef struct yyjson_mut_arr_iter {
    size_t idx; /**< next value's index */
    size_t max; /**< maximum index (arr.size) */
    yyjson_mut_val *cur; /**< current value */
    yyjson_mut_val *pre; /**< previous value */
    yyjson_mut_val *arr; /**< the array being iterated */
} yyjson_mut_arr_iter;

/**
 Initialize an iterator for this array.

 @param arr The array to be iterated over.
    If this parameter is NULL or not an array, `iter` will be set to empty.
 @param iter The iterator to be initialized.
    If this parameter is NULL, the function will fail and return false.
 @return true if the `iter` has been successfully initialized.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline bool yyjson_mut_arr_iter_init(yyjson_mut_val *arr,
    yyjson_mut_arr_iter *iter);

/**
 Create an iterator with an array , same as `yyjson_mut_arr_iter_init()`.

 @param arr The array to be iterated over.
    If this parameter is NULL or not an array, an empty iterator will returned.
 @return A new iterator for the array.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline yyjson_mut_arr_iter yyjson_mut_arr_iter_with(
    yyjson_mut_val *arr);

/**
 Returns whether the iteration has more elements.
 If `iter` is NULL, this function will return false.
 */
yyjson_api_inline bool yyjson_mut_arr_iter_has_next(
    yyjson_mut_arr_iter *iter);

/**
 Returns the next element in the iteration, or NULL on end.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_iter_next(
    yyjson_mut_arr_iter *iter);

/**
 Removes and returns current element in the iteration.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_iter_remove(
    yyjson_mut_arr_iter *iter);

/**
 Macro for iterating over an array.
 It works like iterator, but with a more intuitive API.

 @warning You should not modify the array while iterating over it.

 @b Example
 @code
    size_t idx, max;
    yyjson_mut_val *val;
    yyjson_mut_arr_foreach(arr, idx, max, val) {
        your_func(idx, val);
    }
 @endcode
 */
#define yyjson_mut_arr_foreach(arr, idx, max, val) \
    for ((idx) = 0, \
        (max) = yyjson_mut_arr_size(arr), \
        (val) = yyjson_mut_arr_get_first(arr); \
        (idx) < (max); \
        (idx)++, \
        (val) = (val)->next)



/*==============================================================================
 * Mutable JSON Array Creation API
 *============================================================================*/

/**
 Creates and returns an empty mutable array.
 @param doc A mutable document, used for memory allocation only.
 @return The new array. NULL if input is NULL or memory allocation failed.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr(yyjson_mut_doc *doc);

/**
 Creates and returns a new mutable array with the given boolean values.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of boolean values.
 @param count The value count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const bool vals[3] = { true, false, true };
    yyjson_mut_val *arr = yyjson_mut_arr_with_bool(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_bool(
    yyjson_mut_doc *doc, const bool *vals, size_t count);

/**
 Creates and returns a new mutable array with the given sint numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of sint numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const int64_t vals[3] = { -1, 0, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_sint64(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint(
    yyjson_mut_doc *doc, const int64_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given uint numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of uint numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const uint64_t vals[3] = { 0, 1, 0 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_uint(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint(
    yyjson_mut_doc *doc, const uint64_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given real numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of real numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const double vals[3] = { 0.1, 0.2, 0.3 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_real(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_real(
    yyjson_mut_doc *doc, const double *vals, size_t count);

/**
 Creates and returns a new mutable array with the given int8 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of int8 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const int8_t vals[3] = { -1, 0, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_sint8(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint8(
    yyjson_mut_doc *doc, const int8_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given int16 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of int16 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const int16_t vals[3] = { -1, 0, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_sint16(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint16(
    yyjson_mut_doc *doc, const int16_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given int32 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of int32 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const int32_t vals[3] = { -1, 0, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_sint32(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint32(
    yyjson_mut_doc *doc, const int32_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given int64 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of int64 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const int64_t vals[3] = { -1, 0, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_sint64(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint64(
    yyjson_mut_doc *doc, const int64_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given uint8 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of uint8 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const uint8_t vals[3] = { 0, 1, 0 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_uint8(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint8(
    yyjson_mut_doc *doc, const uint8_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given uint16 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of uint16 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const uint16_t vals[3] = { 0, 1, 0 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_uint16(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint16(
    yyjson_mut_doc *doc, const uint16_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given uint32 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of uint32 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const uint32_t vals[3] = { 0, 1, 0 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_uint32(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint32(
    yyjson_mut_doc *doc, const uint32_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given uint64 numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of uint64 numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
     const uint64_t vals[3] = { 0, 1, 0 };
     yyjson_mut_val *arr = yyjson_mut_arr_with_uint64(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint64(
    yyjson_mut_doc *doc, const uint64_t *vals, size_t count);

/**
 Creates and returns a new mutable array with the given float numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of float numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const float vals[3] = { -1.0f, 0.0f, 1.0f };
    yyjson_mut_val *arr = yyjson_mut_arr_with_float(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_float(
    yyjson_mut_doc *doc, const float *vals, size_t count);

/**
 Creates and returns a new mutable array with the given double numbers.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of double numbers.
 @param count The number count. If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const double vals[3] = { -1.0, 0.0, 1.0 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_double(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_double(
    yyjson_mut_doc *doc, const double *vals, size_t count);

/**
 Creates and returns a new mutable array with the given strings, these strings
 will not be copied.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of UTF-8 null-terminator strings.
    If this array contains NULL, the function will fail and return NULL.
 @param count The number of values in `vals`.
    If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @warning The input strings are not copied, you should keep these strings
    unmodified for the lifetime of this JSON document. If these strings will be
    modified, you should use `yyjson_mut_arr_with_strcpy()` instead.

 @b Example
 @code
    const char *vals[3] = { "a", "b", "c" };
    yyjson_mut_val *arr = yyjson_mut_arr_with_str(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_str(
    yyjson_mut_doc *doc, const char **vals, size_t count);

/**
 Creates and returns a new mutable array with the given strings and string
 lengths, these strings will not be copied.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of UTF-8 strings, null-terminator is not required.
    If this array contains NULL, the function will fail and return NULL.
 @param lens A C array of string lengths, in bytes.
 @param count The number of strings in `vals`.
    If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @warning The input strings are not copied, you should keep these strings
    unmodified for the lifetime of this JSON document. If these strings will be
    modified, you should use `yyjson_mut_arr_with_strncpy()` instead.

 @b Example
 @code
    const char *vals[3] = { "a", "bb", "c" };
    const size_t lens[3] = { 1, 2, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_strn(doc, vals, lens, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_strn(
    yyjson_mut_doc *doc, const char **vals, const size_t *lens, size_t count);

/**
 Creates and returns a new mutable array with the given strings, these strings
 will be copied.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of UTF-8 null-terminator strings.
    If this array contains NULL, the function will fail and return NULL.
 @param count The number of values in `vals`.
    If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const char *vals[3] = { "a", "b", "c" };
    yyjson_mut_val *arr = yyjson_mut_arr_with_strcpy(doc, vals, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_strcpy(
    yyjson_mut_doc *doc, const char **vals, size_t count);

/**
 Creates and returns a new mutable array with the given strings and string
 lengths, these strings will be copied.

 @param doc A mutable document, used for memory allocation only.
    If this parameter is NULL, the function will fail and return NULL.
 @param vals A C array of UTF-8 strings, null-terminator is not required.
    If this array contains NULL, the function will fail and return NULL.
 @param lens A C array of string lengths, in bytes.
 @param count The number of strings in `vals`.
    If this value is 0, an empty array will return.
 @return The new array. NULL if input is invalid or memory allocation failed.

 @b Example
 @code
    const char *vals[3] = { "a", "bb", "c" };
    const size_t lens[3] = { 1, 2, 1 };
    yyjson_mut_val *arr = yyjson_mut_arr_with_strn(doc, vals, lens, 3);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_strncpy(
    yyjson_mut_doc *doc, const char **vals, const size_t *lens, size_t count);



/*==============================================================================
 * Mutable JSON Array Modification API
 *============================================================================*/

/**
 Inserts a value into an array at a given index.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param val The value to be inserted. Returns false if it is NULL.
 @param idx The index to which to insert the new value.
    Returns false if the index is out of range.
 @return Whether successful.
 @warning This function takes a linear search time.
 */
yyjson_api_inline bool yyjson_mut_arr_insert(yyjson_mut_val *arr,
                                             yyjson_mut_val *val, size_t idx);

/**
 Inserts a value at the end of the array.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param val The value to be inserted. Returns false if it is NULL.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_append(yyjson_mut_val *arr,
                                             yyjson_mut_val *val);

/**
 Inserts a value at the head of the array.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param val The value to be inserted. Returns false if it is NULL.
 @return    Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_prepend(yyjson_mut_val *arr,
                                              yyjson_mut_val *val);

/**
 Replaces a value at index and returns old value.
 @param arr The array to which the value is to be replaced.
    Returns false if it is NULL or not an array.
 @param idx The index to which to replace the value.
    Returns false if the index is out of range.
 @param val The new value to replace. Returns false if it is NULL.
 @return Old value, or NULL on error.
 @warning This function takes a linear search time.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_replace(yyjson_mut_val *arr,
                                                         size_t idx,
                                                         yyjson_mut_val *val);

/**
 Removes and returns a value at index.
 @param arr The array from which the value is to be removed.
    Returns false if it is NULL or not an array.
 @param idx The index from which to remove the value.
    Returns false if the index is out of range.
 @return Old value, or NULL on error.
 @warning This function takes a linear search time.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_remove(yyjson_mut_val *arr,
                                                        size_t idx);

/**
 Removes and returns the first value in this array.
 @param arr The array from which the value is to be removed.
    Returns false if it is NULL or not an array.
 @return The first value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_remove_first(
    yyjson_mut_val *arr);

/**
 Removes and returns the last value in this array.
 @param arr The array from which the value is to be removed.
    Returns false if it is NULL or not an array.
 @return The last value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_remove_last(
    yyjson_mut_val *arr);

/**
 Removes all values within a specified range in the array.
 @param arr The array from which the value is to be removed.
    Returns false if it is NULL or not an array.
 @param idx The start index of the range (0 is the first).
 @param len The number of items in the range (can be 0).
 @return Whether successful.
 @warning This function takes a linear search time.
 */
yyjson_api_inline bool yyjson_mut_arr_remove_range(yyjson_mut_val *arr,
                                                   size_t idx, size_t len);

/**
 Removes all values in this array.
 @param arr The array from which all of the values are to be removed.
    Returns false if it is NULL or not an array.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_clear(yyjson_mut_val *arr);

/**
 Rotates values in this array for the given number of times.
 For example: `[1,2,3,4,5]` rotate 2 is `[3,4,5,1,2]`.
 @param arr The array to be rotated.
 @param idx Index (or times) to rotate.
 @warning This function takes a linear search time.
 */
yyjson_api_inline bool yyjson_mut_arr_rotate(yyjson_mut_val *arr,
                                             size_t idx);



/*==============================================================================
 * Mutable JSON Array Modification Convenience API
 *============================================================================*/

/**
 Adds a value at the end of the array.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param val The value to be inserted. Returns false if it is NULL.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_val(yyjson_mut_val *arr,
                                              yyjson_mut_val *val);

/**
 Adds a `null` value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_null(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr);

/**
 Adds a `true` value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_true(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr);

/**
 Adds a `false` value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_false(yyjson_mut_doc *doc,
                                                yyjson_mut_val *arr);

/**
 Adds a bool value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param val The bool value to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_bool(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               bool val);

/**
 Adds an unsigned integer value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param num The number to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_uint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               uint64_t num);

/**
 Adds a signed integer value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param num The number to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_sint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               int64_t num);

/**
 Adds an integer value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param num The number to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_int(yyjson_mut_doc *doc,
                                              yyjson_mut_val *arr,
                                              int64_t num);

/**
 Adds a float value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param num The number to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_float(yyjson_mut_doc *doc,
                                                yyjson_mut_val *arr,
                                                float num);

/**
 Adds a double value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param num The number to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_double(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *arr,
                                                 double num);

/**
 Adds a double value at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param num The number to be added.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_real(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               double num);

/**
 Adds a string value at the end of the array (no copy).
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param str A null-terminated UTF-8 string.
 @return Whether successful.
 @warning The input string is not copied, you should keep this string unmodified
    for the lifetime of this JSON document.
 */
yyjson_api_inline bool yyjson_mut_arr_add_str(yyjson_mut_doc *doc,
                                              yyjson_mut_val *arr,
                                              const char *str);

/**
 Adds a string value at the end of the array (no copy).
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param str A UTF-8 string, null-terminator is not required.
 @param len The length of the string, in bytes.
 @return Whether successful.
 @warning The input string is not copied, you should keep this string unmodified
    for the lifetime of this JSON document.
 */
yyjson_api_inline bool yyjson_mut_arr_add_strn(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               const char *str,
                                               size_t len);

/**
 Adds a string value at the end of the array (copied).
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param str A null-terminated UTF-8 string.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_strcpy(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *arr,
                                                 const char *str);

/**
 Adds a string value at the end of the array (copied).
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @param str A UTF-8 string, null-terminator is not required.
 @param len The length of the string, in bytes.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_arr_add_strncpy(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *arr,
                                                  const char *str,
                                                  size_t len);

/**
 Creates and adds a new array at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @return The new array, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_add_arr(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *arr);

/**
 Creates and adds a new object at the end of the array.
 @param doc The `doc` is only used for memory allocation.
 @param arr The array to which the value is to be inserted.
    Returns false if it is NULL or not an array.
 @return The new object, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_add_obj(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *arr);



/*==============================================================================
 * Mutable JSON Object API
 *============================================================================*/

/** Returns the number of key-value pairs in this object.
    Returns 0 if `obj` is NULL or type is not object. */
yyjson_api_inline size_t yyjson_mut_obj_size(yyjson_mut_val *obj);

/** Returns the value to which the specified key is mapped.
    Returns NULL if this object contains no mapping for the key.
    Returns NULL if `obj/key` is NULL, or type is not object.

    The `key` should be a null-terminated UTF-8 string.

    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_get(yyjson_mut_val *obj,
                                                     const char *key);

/** Returns the value to which the specified key is mapped.
    Returns NULL if this object contains no mapping for the key.
    Returns NULL if `obj/key` is NULL, or type is not object.

    The `key` should be a UTF-8 string, null-terminator is not required.
    The `key_len` should be the length of the key, in bytes.

    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_getn(yyjson_mut_val *obj,
                                                      const char *key,
                                                      size_t key_len);



/*==============================================================================
 * Mutable JSON Object Iterator API
 *============================================================================*/

/**
 A mutable JSON object iterator.

 @warning You should not modify the object while iterating over it, but you can
    use `yyjson_mut_obj_iter_remove()` to remove current value.

 @b Example
 @code
    yyjson_mut_val *key, *val;
    yyjson_mut_obj_iter iter = yyjson_mut_obj_iter_with(obj);
    while ((key = yyjson_mut_obj_iter_next(&iter))) {
        val = yyjson_mut_obj_iter_get_val(key);
        your_func(key, val);
        if (your_val_is_unused(key, val)) {
            yyjson_mut_obj_iter_remove(&iter);
        }
    }
 @endcode

 If the ordering of the keys is known at compile-time, you can use this method
 to speed up value lookups:
 @code
    // {"k1":1, "k2": 3, "k3": 3}
    yyjson_mut_val *key, *val;
    yyjson_mut_obj_iter iter = yyjson_mut_obj_iter_with(obj);
    yyjson_mut_val *v1 = yyjson_mut_obj_iter_get(&iter, "k1");
    yyjson_mut_val *v3 = yyjson_mut_obj_iter_get(&iter, "k3");
 @endcode
 @see `yyjson_mut_obj_iter_get()` and `yyjson_mut_obj_iter_getn()`
 */
typedef struct yyjson_mut_obj_iter {
    size_t idx; /**< next key's index */
    size_t max; /**< maximum key index (obj.size) */
    yyjson_mut_val *cur; /**< current key */
    yyjson_mut_val *pre; /**< previous key */
    yyjson_mut_val *obj; /**< the object being iterated */
} yyjson_mut_obj_iter;

/**
 Initialize an iterator for this object.

 @param obj The object to be iterated over.
    If this parameter is NULL or not an array, `iter` will be set to empty.
 @param iter The iterator to be initialized.
    If this parameter is NULL, the function will fail and return false.
 @return true if the `iter` has been successfully initialized.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline bool yyjson_mut_obj_iter_init(yyjson_mut_val *obj,
    yyjson_mut_obj_iter *iter);

/**
 Create an iterator with an object, same as `yyjson_obj_iter_init()`.

 @param obj The object to be iterated over.
    If this parameter is NULL or not an object, an empty iterator will returned.
 @return A new iterator for the object.

 @note The iterator does not need to be destroyed.
 */
yyjson_api_inline yyjson_mut_obj_iter yyjson_mut_obj_iter_with(
    yyjson_mut_val *obj);

/**
 Returns whether the iteration has more elements.
 If `iter` is NULL, this function will return false.
 */
yyjson_api_inline bool yyjson_mut_obj_iter_has_next(
    yyjson_mut_obj_iter *iter);

/**
 Returns the next key in the iteration, or NULL on end.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_next(
    yyjson_mut_obj_iter *iter);

/**
 Returns the value for key inside the iteration.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_get_val(
    yyjson_mut_val *key);

/**
 Removes current key-value pair in the iteration, returns the removed value.
 If `iter` is NULL, this function will return NULL.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_remove(
    yyjson_mut_obj_iter *iter);

/**
 Iterates to a specified key and returns the value.

 This function does the same thing as `yyjson_mut_obj_get()`, but is much faster
 if the ordering of the keys is known at compile-time and you are using the same
 order to look up the values. If the key exists in this object, then the
 iterator will stop at the next key, otherwise the iterator will not change and
 NULL is returned.

 @param iter The object iterator, should not be NULL.
 @param key The key, should be a UTF-8 string with null-terminator.
 @return The value to which the specified key is mapped.
    NULL if this object contains no mapping for the key or input is invalid.

 @warning This function takes a linear search time if the key is not nearby.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_get(
    yyjson_mut_obj_iter *iter, const char *key);

/**
 Iterates to a specified key and returns the value.

 This function does the same thing as `yyjson_mut_obj_getn()` but is much faster
 if the ordering of the keys is known at compile-time and you are using the same
 order to look up the values. If the key exists in this object, then the
 iterator will stop at the next key, otherwise the iterator will not change and
 NULL is returned.

 @param iter The object iterator, should not be NULL.
 @param key The key, should be a UTF-8 string, null-terminator is not required.
 @param key_len The the length of `key`, in bytes.
 @return The value to which the specified key is mapped.
    NULL if this object contains no mapping for the key or input is invalid.

 @warning This function takes a linear search time if the key is not nearby.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_getn(
    yyjson_mut_obj_iter *iter, const char *key, size_t key_len);

/**
 Macro for iterating over an object.
 It works like iterator, but with a more intuitive API.

 @warning You should not modify the object while iterating over it.

 @b Example
 @code
    size_t idx, max;
    yyjson_mut_val *key, *val;
    yyjson_mut_obj_foreach(obj, idx, max, key, val) {
        your_func(key, val);
    }
 @endcode
 */
#define yyjson_mut_obj_foreach(obj, idx, max, key, val) \
    for ((idx) = 0, \
        (max) = yyjson_mut_obj_size(obj), \
        (key) = (max) ? ((yyjson_mut_val *)(obj)->uni.ptr)->next->next : NULL, \
        (val) = (key) ? (key)->next : NULL; \
        (idx) < (max); \
        (idx)++, \
        (key) = (val)->next, \
        (val) = (key)->next)



/*==============================================================================
 * Mutable JSON Object Creation API
 *============================================================================*/

/** Creates and returns a mutable object, returns NULL on error. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj(yyjson_mut_doc *doc);

/**
 Creates and returns a mutable object with keys and values, returns NULL on
 error. The keys and values are not copied. The strings should be a
 null-terminated UTF-8 string.

 @warning The input string is not copied, you should keep this string
    unmodified for the lifetime of this JSON document.

 @b Example
 @code
    const char *keys[2] = { "id", "name" };
    const char *vals[2] = { "01", "Harry" };
    yyjson_mut_val *obj = yyjson_mut_obj_with_str(doc, keys, vals, 2);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_with_str(yyjson_mut_doc *doc,
                                                          const char **keys,
                                                          const char **vals,
                                                          size_t count);

/**
 Creates and returns a mutable object with key-value pairs and pair count,
 returns NULL on error. The keys and values are not copied. The strings should
 be a null-terminated UTF-8 string.

 @warning The input string is not copied, you should keep this string
    unmodified for the lifetime of this JSON document.

 @b Example
 @code
    const char *kv_pairs[4] = { "id", "01", "name", "Harry" };
    yyjson_mut_val *obj = yyjson_mut_obj_with_kv(doc, kv_pairs, 2);
 @endcode
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_with_kv(yyjson_mut_doc *doc,
                                                         const char **kv_pairs,
                                                         size_t pair_count);



/*==============================================================================
 * Mutable JSON Object Modification API
 *============================================================================*/

/**
 Adds a key-value pair at the end of the object.
 This function allows duplicated key in one object.
 @param obj The object to which the new key-value pair is to be added.
 @param key The key, should be a string which is created by `yyjson_mut_str()`,
    `yyjson_mut_strn()`, `yyjson_mut_strcpy()` or `yyjson_mut_strncpy()`.
 @param val The value to add to the object.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_obj_add(yyjson_mut_val *obj,
                                          yyjson_mut_val *key,
                                          yyjson_mut_val *val);
/**
 Sets a key-value pair at the end of the object.
 This function may remove all key-value pairs for the given key before add.
 @param obj The object to which the new key-value pair is to be added.
 @param key The key, should be a string which is created by `yyjson_mut_str()`,
    `yyjson_mut_strn()`, `yyjson_mut_strcpy()` or `yyjson_mut_strncpy()`.
 @param val The value to add to the object. If this value is null, the behavior
    is same as `yyjson_mut_obj_remove()`.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_obj_put(yyjson_mut_val *obj,
                                          yyjson_mut_val *key,
                                          yyjson_mut_val *val);

/**
 Inserts a key-value pair to the object at the given position.
 This function allows duplicated key in one object.
 @param obj The object to which the new key-value pair is to be added.
 @param key The key, should be a string which is created by `yyjson_mut_str()`,
    `yyjson_mut_strn()`, `yyjson_mut_strcpy()` or `yyjson_mut_strncpy()`.
 @param val The value to add to the object.
 @param idx The index to which to insert the new pair.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_obj_insert(yyjson_mut_val *obj,
                                             yyjson_mut_val *key,
                                             yyjson_mut_val *val,
                                             size_t idx);

/**
 Removes all key-value pair from the object with given key.
 @param obj The object from which the key-value pair is to be removed.
 @param key The key, should be a string value.
 @return The first matched value, or NULL if no matched value.
 @warning This function takes a linear search time.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove(yyjson_mut_val *obj,
                                                        yyjson_mut_val *key);

/**
 Removes all key-value pair from the object with given key.
 @param obj The object from which the key-value pair is to be removed.
 @param key The key, should be a UTF-8 string with null-terminator.
 @return The first matched value, or NULL if no matched value.
 @warning This function takes a linear search time.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_key(
    yyjson_mut_val *obj, const char *key);

/**
 Removes all key-value pair from the object with given key.
 @param obj The object from which the key-value pair is to be removed.
 @param key The key, should be a UTF-8 string, null-terminator is not required.
 @param key_len The length of the key.
 @return The first matched value, or NULL if no matched value.
 @warning This function takes a linear search time.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_keyn(
    yyjson_mut_val *obj, const char *key, size_t key_len);

/**
 Removes all key-value pairs in this object.
 @param obj The object from which all of the values are to be removed.
 @return Whether successful.
 */
yyjson_api_inline bool yyjson_mut_obj_clear(yyjson_mut_val *obj);

/**
 Replaces value from the object with given key.
 If the key is not exist, or the value is NULL, it will fail.
 @param obj The object to which the value is to be replaced.
 @param key The key, should be a string value.
 @param val The value to replace into the object.
 @return Whether successful.
 @warning This function takes a linear search time.
 */
yyjson_api_inline bool yyjson_mut_obj_replace(yyjson_mut_val *obj,
                                              yyjson_mut_val *key,
                                              yyjson_mut_val *val);

/**
 Rotates key-value pairs in the object for the given number of times.
 For example: `{"a":1,"b":2,"c":3,"d":4}` rotate 1 is
 `{"b":2,"c":3,"d":4,"a":1}`.
 @param obj The object to be rotated.
 @param idx Index (or times) to rotate.
 @return Whether successful.
 @warning This function takes a linear search time.
 */
yyjson_api_inline bool yyjson_mut_obj_rotate(yyjson_mut_val *obj,
                                             size_t idx);



/*==============================================================================
 * Mutable JSON Object Modification Convenience API
 *============================================================================*/

/** Adds a `null` value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_null(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key);

/** Adds a `true` value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_true(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key);

/** Adds a `false` value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_false(yyjson_mut_doc *doc,
                                                yyjson_mut_val *obj,
                                                const char *key);

/** Adds a bool value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_bool(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key, bool val);

/** Adds an unsigned integer value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_uint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key, uint64_t val);

/** Adds a signed integer value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_sint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key, int64_t val);

/** Adds an int value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_int(yyjson_mut_doc *doc,
                                              yyjson_mut_val *obj,
                                              const char *key, int64_t val);

/** Adds a float value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_float(yyjson_mut_doc *doc,
                                                yyjson_mut_val *obj,
                                                const char *key, float val);

/** Adds a double value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_double(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *obj,
                                                 const char *key, double val);

/** Adds a real value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_real(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key, double val);

/** Adds a string value at the end of the object.
    The `key` and `val` should be null-terminated UTF-8 strings.
    This function allows duplicated key in one object.

    @warning The key/value strings are not copied, you should keep these strings
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_str(yyjson_mut_doc *doc,
                                              yyjson_mut_val *obj,
                                              const char *key, const char *val);

/** Adds a string value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    The `val` should be a UTF-8 string, null-terminator is not required.
    The `len` should be the length of the `val`, in bytes.
    This function allows duplicated key in one object.

    @warning The key/value strings are not copied, you should keep these strings
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_strn(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *key,
                                               const char *val, size_t len);

/** Adds a string value at the end of the object.
    The `key` and `val` should be null-terminated UTF-8 strings.
    The value string is copied.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_strcpy(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *obj,
                                                 const char *key,
                                                 const char *val);

/** Adds a string value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    The `val` should be a UTF-8 string, null-terminator is not required.
    The `len` should be the length of the `val`, in bytes.
    This function allows duplicated key in one object.

    @warning The key strings are not copied, you should keep these strings
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_strncpy(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *obj,
                                                  const char *key,
                                                  const char *val, size_t len);

/**
 Creates and adds a new array to the target object.
 The `key` should be a null-terminated UTF-8 string.
 This function allows duplicated key in one object.

 @warning The key string is not copied, you should keep these strings
          unmodified for the lifetime of this JSON document.
 @return The new array, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_add_arr(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *obj,
                                                         const char *key);

/**
 Creates and adds a new object to the target object.
 The `key` should be a null-terminated UTF-8 string.
 This function allows duplicated key in one object.

 @warning The key string is not copied, you should keep these strings
          unmodified for the lifetime of this JSON document.
 @return The new object, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_add_obj(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *obj,
                                                         const char *key);

/** Adds a JSON value at the end of the object.
    The `key` should be a null-terminated UTF-8 string.
    This function allows duplicated key in one object.

    @warning The key string is not copied, you should keep the string
        unmodified for the lifetime of this JSON document. */
yyjson_api_inline bool yyjson_mut_obj_add_val(yyjson_mut_doc *doc,
                                              yyjson_mut_val *obj,
                                              const char *key,
                                              yyjson_mut_val *val);

/** Removes all key-value pairs for the given key.
    Returns the first value to which the specified key is mapped or NULL if this
    object contains no mapping for the key.
    The `key` should be a null-terminated UTF-8 string.

    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_str(
    yyjson_mut_val *obj, const char *key);

/** Removes all key-value pairs for the given key.
    Returns the first value to which the specified key is mapped or NULL if this
    object contains no mapping for the key.
    The `key` should be a UTF-8 string, null-terminator is not required.
    The `len` should be the length of the key, in bytes.

    @warning This function takes a linear search time. */
yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_strn(
    yyjson_mut_val *obj, const char *key, size_t len);

/** Replaces all matching keys with the new key.
    Returns true if at least one key was renamed.
    The `key` and `new_key` should be a null-terminated UTF-8 string.
    The `new_key` is copied and held by doc.

    @warning This function takes a linear search time.
    If `new_key` already exists, it will cause duplicate keys.
 */
yyjson_api_inline bool yyjson_mut_obj_rename_key(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *obj,
                                                 const char *key,
                                                 const char *new_key);

/** Replaces all matching keys with the new key.
    Returns true if at least one key was renamed.
    The `key` and `new_key` should be a UTF-8 string,
    null-terminator is not required. The `new_key` is copied and held by doc.

    @warning This function takes a linear search time.
    If `new_key` already exists, it will cause duplicate keys.
 */
yyjson_api_inline bool yyjson_mut_obj_rename_keyn(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *obj,
                                                  const char *key,
                                                  size_t len,
                                                  const char *new_key,
                                                  size_t new_len);



#if !defined(YYJSON_DISABLE_UTILS) || !YYJSON_DISABLE_UTILS

/*==============================================================================
 * JSON Pointer API (RFC 6901)
 * https://tools.ietf.org/html/rfc6901
 *============================================================================*/

/** JSON Pointer error code. */
typedef uint32_t yyjson_ptr_code;

/** No JSON pointer error. */
static const yyjson_ptr_code YYJSON_PTR_ERR_NONE = 0;

/** Invalid input parameter, such as NULL input. */
static const yyjson_ptr_code YYJSON_PTR_ERR_PARAMETER = 1;

/** JSON pointer syntax error, such as invalid escape, token no prefix. */
static const yyjson_ptr_code YYJSON_PTR_ERR_SYNTAX = 2;

/** JSON pointer resolve failed, such as index out of range, key not found. */
static const yyjson_ptr_code YYJSON_PTR_ERR_RESOLVE = 3;

/** Document's root is NULL, but it is required for the function call. */
static const yyjson_ptr_code YYJSON_PTR_ERR_NULL_ROOT = 4;

/** Cannot set root as the target is not a document. */
static const yyjson_ptr_code YYJSON_PTR_ERR_SET_ROOT = 5;

/** The memory allocation failed and a new value could not be created. */
static const yyjson_ptr_code YYJSON_PTR_ERR_MEMORY_ALLOCATION = 6;

/** Error information for JSON pointer. */
typedef struct yyjson_ptr_err {
    /** Error code, see `yyjson_ptr_code` for all possible values. */
    yyjson_ptr_code code;
    /** Error message, constant, no need to free (NULL if no error). */
    const char *msg;
    /** Error byte position for input JSON pointer (0 if no error). */
    size_t pos;
} yyjson_ptr_err;

/**
 A context for JSON pointer operation.

 This struct stores the context of JSON Pointer operation result. The struct
 can be used with three helper functions: `ctx_append()`, `ctx_replace()`, and
 `ctx_remove()`, which perform the corresponding operations on the container
 without re-parsing the JSON Pointer.

 For example:
 @code
    // doc before: {"a":[0,1,null]}
    // ptr: "/a/2"
    val = yyjson_mut_doc_ptr_getx(doc, ptr, strlen(ptr), &ctx, &err);
    if (yyjson_is_null(val)) {
        yyjson_ptr_ctx_remove(&ctx);
    }
    // doc after: {"a":[0,1]}
 @endcode
 */
typedef struct yyjson_ptr_ctx {
    /**
     The container (parent) of the target value. It can be either an array or
     an object. If the target location has no value, but all its parent
     containers exist, and the target location can be used to insert a new
     value, then `ctn` is the parent container of the target location.
     Otherwise, `ctn` is NULL.
     */
    yyjson_mut_val *ctn;
    /**
     The previous sibling of the target value. It can be either a value in an
     array or a key in an object. As the container is a `circular linked list`
     of elements, `pre` is the previous node of the target value. If the
     operation is `add` or `set`, then `pre` is the previous node of the new
     value, not the original target value. If the target value does not exist,
     `pre` is NULL.
     */
    yyjson_mut_val *pre;
    /**
     The removed value if the operation is `set`, `replace` or `remove`. It can
     be used to restore the original state of the document if needed.
     */
    yyjson_mut_val *old;
} yyjson_ptr_ctx;

/**
 Get value by a JSON Pointer.
 @param doc The JSON document to be queried.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @return The value referenced by the JSON pointer.
    NULL if `doc` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_val *yyjson_doc_ptr_get(yyjson_doc *doc,
                                                 const char *ptr);

/**
 Get value by a JSON Pointer.
 @param doc The JSON document to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @return The value referenced by the JSON pointer.
    NULL if `doc` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_val *yyjson_doc_ptr_getn(yyjson_doc *doc,
                                                  const char *ptr, size_t len);

/**
 Get value by a JSON Pointer.
 @param doc The JSON document to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The value referenced by the JSON pointer.
    NULL if `doc` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_val *yyjson_doc_ptr_getx(yyjson_doc *doc,
                                                  const char *ptr, size_t len,
                                                  yyjson_ptr_err *err);

/**
 Get value by a JSON Pointer.
 @param val The JSON value to be queried.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @return The value referenced by the JSON pointer.
    NULL if `val` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_val *yyjson_ptr_get(yyjson_val *val,
                                             const char *ptr);

/**
 Get value by a JSON Pointer.
 @param val The JSON value to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @return The value referenced by the JSON pointer.
    NULL if `val` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_val *yyjson_ptr_getn(yyjson_val *val,
                                              const char *ptr, size_t len);

/**
 Get value by a JSON Pointer.
 @param val The JSON value to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The value referenced by the JSON pointer.
    NULL if `val` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_val *yyjson_ptr_getx(yyjson_val *val,
                                              const char *ptr, size_t len,
                                              yyjson_ptr_err *err);

/**
 Get value by a JSON Pointer.
 @param doc The JSON document to be queried.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @return The value referenced by the JSON pointer.
    NULL if `doc` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_get(yyjson_mut_doc *doc,
                                                         const char *ptr);

/**
 Get value by a JSON Pointer.
 @param doc The JSON document to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @return The value referenced by the JSON pointer.
    NULL if `doc` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_getn(yyjson_mut_doc *doc,
                                                          const char *ptr,
                                                          size_t len);

/**
 Get value by a JSON Pointer.
 @param doc The JSON document to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The value referenced by the JSON pointer.
    NULL if `doc` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_getx(yyjson_mut_doc *doc,
                                                          const char *ptr,
                                                          size_t len,
                                                          yyjson_ptr_ctx *ctx,
                                                          yyjson_ptr_err *err);

/**
 Get value by a JSON Pointer.
 @param val The JSON value to be queried.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @return The value referenced by the JSON pointer.
    NULL if `val` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_get(yyjson_mut_val *val,
                                                     const char *ptr);

/**
 Get value by a JSON Pointer.
 @param val The JSON value to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @return The value referenced by the JSON pointer.
    NULL if `val` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_getn(yyjson_mut_val *val,
                                                      const char *ptr,
                                                      size_t len);

/**
 Get value by a JSON Pointer.
 @param val The JSON value to be queried.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The value referenced by the JSON pointer.
    NULL if `val` or `ptr` is NULL, or the JSON pointer cannot be resolved.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_getx(yyjson_mut_val *val,
                                                      const char *ptr,
                                                      size_t len,
                                                      yyjson_ptr_ctx *ctx,
                                                      yyjson_ptr_err *err);

/**
 Add (insert) value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @param new_val The value to be added.
 @return true if JSON pointer is valid and new value is added, false otherwise.
 @note The parent nodes will be created if they do not exist.
 */
yyjson_api_inline bool yyjson_mut_doc_ptr_add(yyjson_mut_doc *doc,
                                              const char *ptr,
                                              yyjson_mut_val *new_val);

/**
 Add (insert) value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The value to be added.
 @return true if JSON pointer is valid and new value is added, false otherwise.
 @note The parent nodes will be created if they do not exist.
 */
yyjson_api_inline bool yyjson_mut_doc_ptr_addn(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val);

/**
 Add (insert) value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The value to be added.
 @param create_parent Whether to create parent nodes if not exist.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return true if JSON pointer is valid and new value is added, false otherwise.
 */
yyjson_api_inline bool yyjson_mut_doc_ptr_addx(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val,
                                               bool create_parent,
                                               yyjson_ptr_ctx *ctx,
                                               yyjson_ptr_err *err);

/**
 Add (insert) value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @param doc Only used to create new values when needed.
 @param new_val The value to be added.
 @return true if JSON pointer is valid and new value is added, false otherwise.
 @note The parent nodes will be created if they do not exist.
 */
yyjson_api_inline bool yyjson_mut_ptr_add(yyjson_mut_val *val,
                                          const char *ptr,
                                          yyjson_mut_val *new_val,
                                          yyjson_mut_doc *doc);

/**
 Add (insert) value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param doc Only used to create new values when needed.
 @param new_val The value to be added.
 @return true if JSON pointer is valid and new value is added, false otherwise.
 @note The parent nodes will be created if they do not exist.
 */
yyjson_api_inline bool yyjson_mut_ptr_addn(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc);

/**
 Add (insert) value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param doc Only used to create new values when needed.
 @param new_val The value to be added.
 @param create_parent Whether to create parent nodes if not exist.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return true if JSON pointer is valid and new value is added, false otherwise.
 */
yyjson_api_inline bool yyjson_mut_ptr_addx(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc,
                                           bool create_parent,
                                           yyjson_ptr_ctx *ctx,
                                           yyjson_ptr_err *err);

/**
 Set value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @param new_val The value to be set, pass NULL to remove.
 @return true if JSON pointer is valid and new value is set, false otherwise.
 @note The parent nodes will be created if they do not exist.
    If the target value already exists, it will be replaced by the new value.
 */
yyjson_api_inline bool yyjson_mut_doc_ptr_set(yyjson_mut_doc *doc,
                                              const char *ptr,
                                              yyjson_mut_val *new_val);

/**
 Set value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The value to be set, pass NULL to remove.
 @return true if JSON pointer is valid and new value is set, false otherwise.
 @note The parent nodes will be created if they do not exist.
    If the target value already exists, it will be replaced by the new value.
 */
yyjson_api_inline bool yyjson_mut_doc_ptr_setn(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val);

/**
 Set value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The value to be set, pass NULL to remove.
 @param create_parent Whether to create parent nodes if not exist.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return true if JSON pointer is valid and new value is set, false otherwise.
 @note If the target value already exists, it will be replaced by the new value.
 */
yyjson_api_inline bool yyjson_mut_doc_ptr_setx(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val,
                                               bool create_parent,
                                               yyjson_ptr_ctx *ctx,
                                               yyjson_ptr_err *err);

/**
 Set value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @param new_val The value to be set, pass NULL to remove.
 @param doc Only used to create new values when needed.
 @return true if JSON pointer is valid and new value is set, false otherwise.
 @note The parent nodes will be created if they do not exist.
    If the target value already exists, it will be replaced by the new value.
 */
yyjson_api_inline bool yyjson_mut_ptr_set(yyjson_mut_val *val,
                                          const char *ptr,
                                          yyjson_mut_val *new_val,
                                          yyjson_mut_doc *doc);

/**
 Set value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The value to be set, pass NULL to remove.
 @param doc Only used to create new values when needed.
 @return true if JSON pointer is valid and new value is set, false otherwise.
 @note The parent nodes will be created if they do not exist.
    If the target value already exists, it will be replaced by the new value.
 */
yyjson_api_inline bool yyjson_mut_ptr_setn(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc);

/**
 Set value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The value to be set, pass NULL to remove.
 @param doc Only used to create new values when needed.
 @param create_parent Whether to create parent nodes if not exist.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return true if JSON pointer is valid and new value is set, false otherwise.
 @note If the target value already exists, it will be replaced by the new value.
 */
yyjson_api_inline bool yyjson_mut_ptr_setx(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc,
                                           bool create_parent,
                                           yyjson_ptr_ctx *ctx,
                                           yyjson_ptr_err *err);

/**
 Replace value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @param new_val The new value to replace the old one.
 @return The old value that was replaced, or NULL if not found.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_replace(
    yyjson_mut_doc *doc, const char *ptr, yyjson_mut_val *new_val);

/**
 Replace value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The new value to replace the old one.
 @return The old value that was replaced, or NULL if not found.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_replacen(
    yyjson_mut_doc *doc, const char *ptr, size_t len, yyjson_mut_val *new_val);

/**
 Replace value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The new value to replace the old one.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The old value that was replaced, or NULL if not found.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_replacex(
    yyjson_mut_doc *doc, const char *ptr, size_t len, yyjson_mut_val *new_val,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err);

/**
 Replace value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @param new_val The new value to replace the old one.
 @return The old value that was replaced, or NULL if not found.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_replace(
    yyjson_mut_val *val, const char *ptr, yyjson_mut_val *new_val);

/**
 Replace value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The new value to replace the old one.
 @return The old value that was replaced, or NULL if not found.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_replacen(
    yyjson_mut_val *val, const char *ptr, size_t len, yyjson_mut_val *new_val);

/**
 Replace value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param new_val The new value to replace the old one.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The old value that was replaced, or NULL if not found.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_replacex(
    yyjson_mut_val *val, const char *ptr, size_t len, yyjson_mut_val *new_val,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err);

/**
 Remove value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @return The removed value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_remove(
    yyjson_mut_doc *doc, const char *ptr);

/**
 Remove value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @return The removed value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_removen(
    yyjson_mut_doc *doc, const char *ptr, size_t len);

/**
 Remove value by a JSON pointer.
 @param doc The target JSON document.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The removed value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_removex(
    yyjson_mut_doc *doc, const char *ptr, size_t len,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err);

/**
 Remove value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8 with null-terminator).
 @return The removed value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_remove(yyjson_mut_val *val,
                                                        const char *ptr);

/**
 Remove value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @return The removed value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_removen(yyjson_mut_val *val,
                                                         const char *ptr,
                                                         size_t len);

/**
 Remove value by a JSON pointer.
 @param val The target JSON value.
 @param ptr The JSON pointer string (UTF-8, null-terminator is not required).
 @param len The length of `ptr` in bytes.
 @param ctx A pointer to store the result context, or NULL if not needed.
 @param err A pointer to store the error information, or NULL if not needed.
 @return The removed value, or NULL on error.
 */
yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_removex(yyjson_mut_val *val,
                                                         const char *ptr,
                                                         size_t len,
                                                         yyjson_ptr_ctx *ctx,
                                                         yyjson_ptr_err *err);

/**
 Append value by JSON pointer context.
 @param ctx The context from the `yyjson_mut_ptr_xxx()` calls.
 @param key New key if `ctx->ctn` is object, or NULL if `ctx->ctn` is array.
 @param val New value to be added.
 @return true on success or false on fail.
 */
yyjson_api_inline bool yyjson_ptr_ctx_append(yyjson_ptr_ctx *ctx,
                                             yyjson_mut_val *key,
                                             yyjson_mut_val *val);

/**
 Replace value by JSON pointer context.
 @param ctx The context from the `yyjson_mut_ptr_xxx()` calls.
 @param val New value to be replaced.
 @return true on success or false on fail.
 @note If success, the old value will be returned via `ctx->old`.
 */
yyjson_api_inline bool yyjson_ptr_ctx_replace(yyjson_ptr_ctx *ctx,
                                              yyjson_mut_val *val);

/**
 Remove value by JSON pointer context.
 @param ctx The context from the `yyjson_mut_ptr_xxx()` calls.
 @return true on success or false on fail.
 @note If success, the old value will be returned via `ctx->old`.
 */
yyjson_api_inline bool yyjson_ptr_ctx_remove(yyjson_ptr_ctx *ctx);



/*==============================================================================
 * JSON Patch API (RFC 6902)
 * https://tools.ietf.org/html/rfc6902
 *============================================================================*/

/** Result code for JSON patch. */
typedef uint32_t yyjson_patch_code;

/** Success, no error. */
static const yyjson_patch_code YYJSON_PATCH_SUCCESS = 0;

/** Invalid parameter, such as NULL input or non-array patch. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_INVALID_PARAMETER = 1;

/** Memory allocation failure occurs. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_MEMORY_ALLOCATION = 2;

/** JSON patch operation is not object type. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_INVALID_OPERATION = 3;

/** JSON patch operation is missing a required key. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_MISSING_KEY = 4;

/** JSON patch operation member is invalid. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_INVALID_MEMBER = 5;

/** JSON patch operation `test` not equal. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_EQUAL = 6;

/** JSON patch operation failed on JSON pointer. */
static const yyjson_patch_code YYJSON_PATCH_ERROR_POINTER = 7;

/** Error information for JSON patch. */
typedef struct yyjson_patch_err {
    /** Error code, see `yyjson_patch_code` for all possible values. */
    yyjson_patch_code code;
    /** Index of the error operation (0 if no error). */
    size_t idx;
    /** Error message, constant, no need to free (NULL if no error). */
    const char *msg;
    /** JSON pointer error if `code == YYJSON_PATCH_ERROR_POINTER`. */
    yyjson_ptr_err ptr;
} yyjson_patch_err;

/**
 Creates and returns a patched JSON value (RFC 6902).
 The memory of the returned value is allocated by the `doc`.
 The `err` is used to receive error information, pass NULL if not needed.
 Returns NULL if the patch could not be applied.
 */
yyjson_api yyjson_mut_val *yyjson_patch(yyjson_mut_doc *doc,
                                        yyjson_val *orig,
                                        yyjson_val *patch,
                                        yyjson_patch_err *err);

/**
 Creates and returns a patched JSON value (RFC 6902).
 The memory of the returned value is allocated by the `doc`.
 The `err` is used to receive error information, pass NULL if not needed.
 Returns NULL if the patch could not be applied.
 */
yyjson_api yyjson_mut_val *yyjson_mut_patch(yyjson_mut_doc *doc,
                                            yyjson_mut_val *orig,
                                            yyjson_mut_val *patch,
                                            yyjson_patch_err *err);



/*==============================================================================
 * JSON Merge-Patch API (RFC 7386)
 * https://tools.ietf.org/html/rfc7386
 *============================================================================*/

/**
 Creates and returns a merge-patched JSON value (RFC 7386).
 The memory of the returned value is allocated by the `doc`.
 Returns NULL if the patch could not be applied.

 @warning This function is recursive and may cause a stack overflow if the
    object level is too deep.
 */
yyjson_api yyjson_mut_val *yyjson_merge_patch(yyjson_mut_doc *doc,
                                              yyjson_val *orig,
                                              yyjson_val *patch);

/**
 Creates and returns a merge-patched JSON value (RFC 7386).
 The memory of the returned value is allocated by the `doc`.
 Returns NULL if the patch could not be applied.

 @warning This function is recursive and may cause a stack overflow if the
    object level is too deep.
 */
yyjson_api yyjson_mut_val *yyjson_mut_merge_patch(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *orig,
                                                  yyjson_mut_val *patch);

#endif /* YYJSON_DISABLE_UTILS */



/*==============================================================================
 * JSON Structure (Implementation)
 *============================================================================*/

/** Payload of a JSON value (8 bytes). */
typedef union yyjson_val_uni {
    uint64_t    u64;
    int64_t     i64;
    double      f64;
    const char *str;
    void       *ptr;
    size_t      ofs;
} yyjson_val_uni;

/**
 Immutable JSON value, 16 bytes.
 */
struct yyjson_val {
    uint64_t tag; /**< type, subtype and length */
    yyjson_val_uni uni; /**< payload */
};

struct yyjson_doc {
    /** Root value of the document (nonnull). */
    yyjson_val *root;
    /** Allocator used by document (nonnull). */
    yyjson_alc alc;
    /** The total number of bytes read when parsing JSON (nonzero). */
    size_t dat_read;
    /** The total number of value read when parsing JSON (nonzero). */
    size_t val_read;
    /** The string pool used by JSON values (nullable). */
    char *str_pool;
};



/*==============================================================================
 * Unsafe JSON Value API (Implementation)
 *============================================================================*/

/*
 Whether the string does not need to be escaped for serialization.
 This function is used to optimize the writing speed of small constant strings.
 This function works only if the compiler can evaluate it at compile time.

 Clang supports it since v8.0,
    earlier versions do not support constant_p(strlen) and return false.
 GCC supports it since at least v4.4,
    earlier versions may compile it as run-time instructions.
 ICC supports it since at least v16,
    earlier versions are uncertain.

 @param str The C string.
 @param len The returnd value from strlen(str).
 */
yyjson_api_inline bool unsafe_yyjson_is_str_noesc(const char *str, size_t len) {
#if YYJSON_HAS_CONSTANT_P && \
    (!YYJSON_IS_REAL_GCC || yyjson_gcc_available(4, 4, 0))
    if (yyjson_constant_p(len) && len <= 32) {
        /*
         Same as the following loop:

         for (size_t i = 0; i < len; i++) {
             char c = str[i];
             if (c < ' ' || c > '~' || c == '"' || c == '\\') return false;
         }

         GCC evaluates it at compile time only if the string length is within 17
         and -O3 (which turns on the -fpeel-loops flag) is used.
         So the loop is unrolled for GCC.
         */
#       define yyjson_repeat32_incr(x) \
            x(0)  x(1)  x(2)  x(3)  x(4)  x(5)  x(6)  x(7)  \
            x(8)  x(9)  x(10) x(11) x(12) x(13) x(14) x(15) \
            x(16) x(17) x(18) x(19) x(20) x(21) x(22) x(23) \
            x(24) x(25) x(26) x(27) x(28) x(29) x(30) x(31)
#       define yyjson_check_char_noesc(i) \
            if (i < len) { \
                char c = str[i]; \
                if (c < ' ' || c > '~' || c == '"' || c == '\\') return false; }
        yyjson_repeat32_incr(yyjson_check_char_noesc)
#       undef yyjson_repeat32_incr
#       undef yyjson_check_char_noesc
        return true;
    }
#else
    (void)str;
    (void)len;
#endif
    return false;
}

yyjson_api_inline double unsafe_yyjson_u64_to_f64(uint64_t num) {
#if YYJSON_U64_TO_F64_NO_IMPL
        uint64_t msb = ((uint64_t)1) << 63;
        if ((num & msb) == 0) {
            return (double)(int64_t)num;
        } else {
            return ((double)(int64_t)((num >> 1) | (num & 1))) * (double)2.0;
        }
#else
        return (double)num;
#endif
}

yyjson_api_inline yyjson_type unsafe_yyjson_get_type(void *val) {
    uint8_t tag = (uint8_t)((yyjson_val *)val)->tag;
    return (yyjson_type)(tag & YYJSON_TYPE_MASK);
}

yyjson_api_inline yyjson_subtype unsafe_yyjson_get_subtype(void *val) {
    uint8_t tag = (uint8_t)((yyjson_val *)val)->tag;
    return (yyjson_subtype)(tag & YYJSON_SUBTYPE_MASK);
}

yyjson_api_inline uint8_t unsafe_yyjson_get_tag(void *val) {
    uint8_t tag = (uint8_t)((yyjson_val *)val)->tag;
    return (uint8_t)(tag & YYJSON_TAG_MASK);
}

yyjson_api_inline bool unsafe_yyjson_is_raw(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_RAW;
}

yyjson_api_inline bool unsafe_yyjson_is_null(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_NULL;
}

yyjson_api_inline bool unsafe_yyjson_is_bool(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_BOOL;
}

yyjson_api_inline bool unsafe_yyjson_is_num(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_NUM;
}

yyjson_api_inline bool unsafe_yyjson_is_str(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_STR;
}

yyjson_api_inline bool unsafe_yyjson_is_arr(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_ARR;
}

yyjson_api_inline bool unsafe_yyjson_is_obj(void *val) {
    return unsafe_yyjson_get_type(val) == YYJSON_TYPE_OBJ;
}

yyjson_api_inline bool unsafe_yyjson_is_ctn(void *val) {
    uint8_t mask = YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ;
    return (unsafe_yyjson_get_tag(val) & mask) == mask;
}

yyjson_api_inline bool unsafe_yyjson_is_uint(void *val) {
    const uint8_t patt = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_UINT;
    return unsafe_yyjson_get_tag(val) == patt;
}

yyjson_api_inline bool unsafe_yyjson_is_sint(void *val) {
    const uint8_t patt = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_SINT;
    return unsafe_yyjson_get_tag(val) == patt;
}

yyjson_api_inline bool unsafe_yyjson_is_int(void *val) {
    const uint8_t mask = YYJSON_TAG_MASK & (~YYJSON_SUBTYPE_SINT);
    const uint8_t patt = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_UINT;
    return (unsafe_yyjson_get_tag(val) & mask) == patt;
}

yyjson_api_inline bool unsafe_yyjson_is_real(void *val) {
    const uint8_t patt = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL;
    return unsafe_yyjson_get_tag(val) == patt;
}

yyjson_api_inline bool unsafe_yyjson_is_true(void *val) {
    const uint8_t patt = YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_TRUE;
    return unsafe_yyjson_get_tag(val) == patt;
}

yyjson_api_inline bool unsafe_yyjson_is_false(void *val) {
    const uint8_t patt = YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_FALSE;
    return unsafe_yyjson_get_tag(val) == patt;
}

yyjson_api_inline bool unsafe_yyjson_arr_is_flat(yyjson_val *val) {
    size_t ofs = val->uni.ofs;
    size_t len = (size_t)(val->tag >> YYJSON_TAG_BIT);
    return len * sizeof(yyjson_val) + sizeof(yyjson_val) == ofs;
}

yyjson_api_inline const char *unsafe_yyjson_get_raw(void *val) {
    return ((yyjson_val *)val)->uni.str;
}

yyjson_api_inline bool unsafe_yyjson_get_bool(void *val) {
    uint8_t tag = unsafe_yyjson_get_tag(val);
    return (bool)((tag & YYJSON_SUBTYPE_MASK) >> YYJSON_TYPE_BIT);
}

yyjson_api_inline uint64_t unsafe_yyjson_get_uint(void *val) {
    return ((yyjson_val *)val)->uni.u64;
}

yyjson_api_inline int64_t unsafe_yyjson_get_sint(void *val) {
    return ((yyjson_val *)val)->uni.i64;
}

yyjson_api_inline int unsafe_yyjson_get_int(void *val) {
    return (int)((yyjson_val *)val)->uni.i64;
}

yyjson_api_inline double unsafe_yyjson_get_real(void *val) {
    return ((yyjson_val *)val)->uni.f64;
}

yyjson_api_inline double unsafe_yyjson_get_num(void *val) {
    uint8_t tag = unsafe_yyjson_get_tag(val);
    if (tag == (YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL)) {
        return ((yyjson_val *)val)->uni.f64;
    } else if (tag == (YYJSON_TYPE_NUM | YYJSON_SUBTYPE_SINT)) {
        return (double)((yyjson_val *)val)->uni.i64;
    } else if (tag == (YYJSON_TYPE_NUM | YYJSON_SUBTYPE_UINT)) {
        return unsafe_yyjson_u64_to_f64(((yyjson_val *)val)->uni.u64);
    }
    return 0.0;
}

yyjson_api_inline const char *unsafe_yyjson_get_str(void *val) {
    return ((yyjson_val *)val)->uni.str;
}

yyjson_api_inline size_t unsafe_yyjson_get_len(void *val) {
    return (size_t)(((yyjson_val *)val)->tag >> YYJSON_TAG_BIT);
}

yyjson_api_inline yyjson_val *unsafe_yyjson_get_first(yyjson_val *ctn) {
    return ctn + 1;
}

yyjson_api_inline yyjson_val *unsafe_yyjson_get_next(yyjson_val *val) {
    bool is_ctn = unsafe_yyjson_is_ctn(val);
    size_t ctn_ofs = val->uni.ofs;
    size_t ofs = (is_ctn ? ctn_ofs : sizeof(yyjson_val));
    return (yyjson_val *)(void *)((uint8_t *)val + ofs);
}

yyjson_api_inline bool unsafe_yyjson_equals_strn(void *val, const char *str,
                                                 size_t len) {
    return unsafe_yyjson_get_len(val) == len &&
           memcmp(((yyjson_val *)val)->uni.str, str, len) == 0;
}

yyjson_api_inline bool unsafe_yyjson_equals_str(void *val, const char *str) {
    return unsafe_yyjson_equals_strn(val, str, strlen(str));
}

yyjson_api_inline void unsafe_yyjson_set_type(void *val, yyjson_type type,
                                              yyjson_subtype subtype) {
    uint8_t tag = (type | subtype);
    uint64_t new_tag = ((yyjson_val *)val)->tag;
    new_tag = (new_tag & (~(uint64_t)YYJSON_TAG_MASK)) | (uint64_t)tag;
    ((yyjson_val *)val)->tag = new_tag;
}

yyjson_api_inline void unsafe_yyjson_set_len(void *val, size_t len) {
    uint64_t tag = ((yyjson_val *)val)->tag & YYJSON_TAG_MASK;
    tag |= (uint64_t)len << YYJSON_TAG_BIT;
    ((yyjson_val *)val)->tag = tag;
}

yyjson_api_inline void unsafe_yyjson_set_tag(void *val, yyjson_type type,
                                             yyjson_subtype subtype,
                                             size_t len) {
    uint64_t tag = (uint64_t)len << YYJSON_TAG_BIT;
    tag |= (type | subtype);
    ((yyjson_val *)val)->tag = tag;
}

yyjson_api_inline void unsafe_yyjson_inc_len(void *val) {
    uint64_t tag = ((yyjson_val *)val)->tag;
    tag += (uint64_t)(1 << YYJSON_TAG_BIT);
    ((yyjson_val *)val)->tag = tag;
}

yyjson_api_inline void unsafe_yyjson_set_raw(void *val, const char *raw,
                                             size_t len) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_RAW, YYJSON_SUBTYPE_NONE, len);
    ((yyjson_val *)val)->uni.str = raw;
}

yyjson_api_inline void unsafe_yyjson_set_null(void *val) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_NULL, YYJSON_SUBTYPE_NONE, 0);
}

yyjson_api_inline void unsafe_yyjson_set_bool(void *val, bool num) {
    yyjson_subtype subtype = num ? YYJSON_SUBTYPE_TRUE : YYJSON_SUBTYPE_FALSE;
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_BOOL, subtype, 0);
}

yyjson_api_inline void unsafe_yyjson_set_uint(void *val, uint64_t num) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_NUM, YYJSON_SUBTYPE_UINT, 0);
    ((yyjson_val *)val)->uni.u64 = num;
}

yyjson_api_inline void unsafe_yyjson_set_sint(void *val, int64_t num) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_NUM, YYJSON_SUBTYPE_SINT, 0);
    ((yyjson_val *)val)->uni.i64 = num;
}

yyjson_api_inline void unsafe_yyjson_set_fp_to_fixed(void *val, int prec) {
    ((yyjson_val *)val)->tag &= ~((uint64_t)YYJSON_WRITE_FP_TO_FIXED(15) << 32);
    ((yyjson_val *)val)->tag |= (uint64_t)YYJSON_WRITE_FP_TO_FIXED(prec) << 32;
}

yyjson_api_inline void unsafe_yyjson_set_fp_to_float(void *val, bool flt) {
    uint64_t flag = (uint64_t)YYJSON_WRITE_FP_TO_FLOAT << 32;
    if (flt) ((yyjson_val *)val)->tag |= flag;
    else ((yyjson_val *)val)->tag &= ~flag;
}

yyjson_api_inline void unsafe_yyjson_set_float(void *val, float num) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_NUM, YYJSON_SUBTYPE_REAL, 0);
    ((yyjson_val *)val)->tag |= (uint64_t)YYJSON_WRITE_FP_TO_FLOAT << 32;
    ((yyjson_val *)val)->uni.f64 = (double)num;
}

yyjson_api_inline void unsafe_yyjson_set_double(void *val, double num) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_NUM, YYJSON_SUBTYPE_REAL, 0);
    ((yyjson_val *)val)->uni.f64 = num;
}

yyjson_api_inline void unsafe_yyjson_set_real(void *val, double num) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_NUM, YYJSON_SUBTYPE_REAL, 0);
    ((yyjson_val *)val)->uni.f64 = num;
}

yyjson_api_inline void unsafe_yyjson_set_str_noesc(void *val, bool noesc) {
    ((yyjson_val *)val)->tag &= ~(uint64_t)YYJSON_SUBTYPE_MASK;
    if (noesc) ((yyjson_val *)val)->tag |= (uint64_t)YYJSON_SUBTYPE_NOESC;
}

yyjson_api_inline void unsafe_yyjson_set_strn(void *val, const char *str,
                                              size_t len) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_STR, YYJSON_SUBTYPE_NONE, len);
    ((yyjson_val *)val)->uni.str = str;
}

yyjson_api_inline void unsafe_yyjson_set_str(void *val, const char *str) {
    size_t len = strlen(str);
    bool noesc = unsafe_yyjson_is_str_noesc(str, len);
    yyjson_subtype subtype = noesc ? YYJSON_SUBTYPE_NOESC : YYJSON_SUBTYPE_NONE;
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_STR, subtype, len);
    ((yyjson_val *)val)->uni.str = str;
}

yyjson_api_inline void unsafe_yyjson_set_arr(void *val, size_t size) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_ARR, YYJSON_SUBTYPE_NONE, size);
}

yyjson_api_inline void unsafe_yyjson_set_obj(void *val, size_t size) {
    unsafe_yyjson_set_tag(val, YYJSON_TYPE_OBJ, YYJSON_SUBTYPE_NONE, size);
}



/*==============================================================================
 * JSON Document API (Implementation)
 *============================================================================*/

yyjson_api_inline yyjson_val *yyjson_doc_get_root(yyjson_doc *doc) {
    return doc ? doc->root : NULL;
}

yyjson_api_inline size_t yyjson_doc_get_read_size(yyjson_doc *doc) {
    return doc ? doc->dat_read : 0;
}

yyjson_api_inline size_t yyjson_doc_get_val_count(yyjson_doc *doc) {
    return doc ? doc->val_read : 0;
}

yyjson_api_inline void yyjson_doc_free(yyjson_doc *doc) {
    if (doc) {
        yyjson_alc alc = doc->alc;
        memset(&doc->alc, 0, sizeof(alc));
        if (doc->str_pool) alc.free(alc.ctx, doc->str_pool);
        alc.free(alc.ctx, doc);
    }
}



/*==============================================================================
 * JSON Value Type API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_is_raw(yyjson_val *val) {
    return val ? unsafe_yyjson_is_raw(val) : false;
}

yyjson_api_inline bool yyjson_is_null(yyjson_val *val) {
    return val ? unsafe_yyjson_is_null(val) : false;
}

yyjson_api_inline bool yyjson_is_true(yyjson_val *val) {
    return val ? unsafe_yyjson_is_true(val) : false;
}

yyjson_api_inline bool yyjson_is_false(yyjson_val *val) {
    return val ? unsafe_yyjson_is_false(val) : false;
}

yyjson_api_inline bool yyjson_is_bool(yyjson_val *val) {
    return val ? unsafe_yyjson_is_bool(val) : false;
}

yyjson_api_inline bool yyjson_is_uint(yyjson_val *val) {
    return val ? unsafe_yyjson_is_uint(val) : false;
}

yyjson_api_inline bool yyjson_is_sint(yyjson_val *val) {
    return val ? unsafe_yyjson_is_sint(val) : false;
}

yyjson_api_inline bool yyjson_is_int(yyjson_val *val) {
    return val ? unsafe_yyjson_is_int(val) : false;
}

yyjson_api_inline bool yyjson_is_real(yyjson_val *val) {
    return val ? unsafe_yyjson_is_real(val) : false;
}

yyjson_api_inline bool yyjson_is_num(yyjson_val *val) {
    return val ? unsafe_yyjson_is_num(val) : false;
}

yyjson_api_inline bool yyjson_is_str(yyjson_val *val) {
    return val ? unsafe_yyjson_is_str(val) : false;
}

yyjson_api_inline bool yyjson_is_arr(yyjson_val *val) {
    return val ? unsafe_yyjson_is_arr(val) : false;
}

yyjson_api_inline bool yyjson_is_obj(yyjson_val *val) {
    return val ? unsafe_yyjson_is_obj(val) : false;
}

yyjson_api_inline bool yyjson_is_ctn(yyjson_val *val) {
    return val ? unsafe_yyjson_is_ctn(val) : false;
}



/*==============================================================================
 * JSON Value Content API (Implementation)
 *============================================================================*/

yyjson_api_inline yyjson_type yyjson_get_type(yyjson_val *val) {
    return val ? unsafe_yyjson_get_type(val) : YYJSON_TYPE_NONE;
}

yyjson_api_inline yyjson_subtype yyjson_get_subtype(yyjson_val *val) {
    return val ? unsafe_yyjson_get_subtype(val) : YYJSON_SUBTYPE_NONE;
}

yyjson_api_inline uint8_t yyjson_get_tag(yyjson_val *val) {
    return val ? unsafe_yyjson_get_tag(val) : 0;
}

yyjson_api_inline const char *yyjson_get_type_desc(yyjson_val *val) {
    switch (yyjson_get_tag(val)) {
        case YYJSON_TYPE_RAW  | YYJSON_SUBTYPE_NONE:  return "raw";
        case YYJSON_TYPE_NULL | YYJSON_SUBTYPE_NONE:  return "null";
        case YYJSON_TYPE_STR  | YYJSON_SUBTYPE_NONE:  return "string";
        case YYJSON_TYPE_STR  | YYJSON_SUBTYPE_NOESC: return "string";
        case YYJSON_TYPE_ARR  | YYJSON_SUBTYPE_NONE:  return "array";
        case YYJSON_TYPE_OBJ  | YYJSON_SUBTYPE_NONE:  return "object";
        case YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_TRUE:  return "true";
        case YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_FALSE: return "false";
        case YYJSON_TYPE_NUM  | YYJSON_SUBTYPE_UINT:  return "uint";
        case YYJSON_TYPE_NUM  | YYJSON_SUBTYPE_SINT:  return "sint";
        case YYJSON_TYPE_NUM  | YYJSON_SUBTYPE_REAL:  return "real";
        default:                                      return "unknown";
    }
}

yyjson_api_inline const char *yyjson_get_raw(yyjson_val *val) {
    return yyjson_is_raw(val) ? unsafe_yyjson_get_raw(val) : NULL;
}

yyjson_api_inline bool yyjson_get_bool(yyjson_val *val) {
    return yyjson_is_bool(val) ? unsafe_yyjson_get_bool(val) : false;
}

yyjson_api_inline uint64_t yyjson_get_uint(yyjson_val *val) {
    return yyjson_is_int(val) ? unsafe_yyjson_get_uint(val) : 0;
}

yyjson_api_inline int64_t yyjson_get_sint(yyjson_val *val) {
    return yyjson_is_int(val) ? unsafe_yyjson_get_sint(val) : 0;
}

yyjson_api_inline int yyjson_get_int(yyjson_val *val) {
    return yyjson_is_int(val) ? unsafe_yyjson_get_int(val) : 0;
}

yyjson_api_inline double yyjson_get_real(yyjson_val *val) {
    return yyjson_is_real(val) ? unsafe_yyjson_get_real(val) : 0.0;
}

yyjson_api_inline double yyjson_get_num(yyjson_val *val) {
    return val ? unsafe_yyjson_get_num(val) : 0.0;
}

yyjson_api_inline const char *yyjson_get_str(yyjson_val *val) {
    return yyjson_is_str(val) ? unsafe_yyjson_get_str(val) : NULL;
}

yyjson_api_inline size_t yyjson_get_len(yyjson_val *val) {
    return val ? unsafe_yyjson_get_len(val) : 0;
}

yyjson_api_inline bool yyjson_equals_str(yyjson_val *val, const char *str) {
    if (yyjson_likely(val && str)) {
        return unsafe_yyjson_is_str(val) &&
               unsafe_yyjson_equals_str(val, str);
    }
    return false;
}

yyjson_api_inline bool yyjson_equals_strn(yyjson_val *val, const char *str,
                                          size_t len) {
    if (yyjson_likely(val && str)) {
        return unsafe_yyjson_is_str(val) &&
               unsafe_yyjson_equals_strn(val, str, len);
    }
    return false;
}

yyjson_api bool unsafe_yyjson_equals(yyjson_val *lhs, yyjson_val *rhs);

yyjson_api_inline bool yyjson_equals(yyjson_val *lhs, yyjson_val *rhs) {
    if (yyjson_unlikely(!lhs || !rhs)) return false;
    return unsafe_yyjson_equals(lhs, rhs);
}

yyjson_api_inline bool yyjson_set_raw(yyjson_val *val,
                                      const char *raw, size_t len) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_raw(val, raw, len);
    return true;
}

yyjson_api_inline bool yyjson_set_null(yyjson_val *val) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_null(val);
    return true;
}

yyjson_api_inline bool yyjson_set_bool(yyjson_val *val, bool num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_bool(val, num);
    return true;
}

yyjson_api_inline bool yyjson_set_uint(yyjson_val *val, uint64_t num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_uint(val, num);
    return true;
}

yyjson_api_inline bool yyjson_set_sint(yyjson_val *val, int64_t num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_sint(val, num);
    return true;
}

yyjson_api_inline bool yyjson_set_int(yyjson_val *val, int num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_sint(val, (int64_t)num);
    return true;
}

yyjson_api_inline bool yyjson_set_float(yyjson_val *val, float num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_float(val, num);
    return true;
}

yyjson_api_inline bool yyjson_set_double(yyjson_val *val, double num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_double(val, num);
    return true;
}

yyjson_api_inline bool yyjson_set_real(yyjson_val *val, double num) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    unsafe_yyjson_set_real(val, num);
    return true;
}

yyjson_api_inline bool yyjson_set_fp_to_fixed(yyjson_val *val, int prec) {
    if (yyjson_unlikely(!yyjson_is_real(val))) return false;
    unsafe_yyjson_set_fp_to_fixed(val, prec);
    return true;
}

yyjson_api_inline bool yyjson_set_fp_to_float(yyjson_val *val, bool flt) {
    if (yyjson_unlikely(!yyjson_is_real(val))) return false;
    unsafe_yyjson_set_fp_to_float(val, flt);
    return true;
}

yyjson_api_inline bool yyjson_set_str(yyjson_val *val, const char *str) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    if (yyjson_unlikely(!str)) return false;
    unsafe_yyjson_set_str(val, str);
    return true;
}

yyjson_api_inline bool yyjson_set_strn(yyjson_val *val,
                                       const char *str, size_t len) {
    if (yyjson_unlikely(!val || unsafe_yyjson_is_ctn(val))) return false;
    if (yyjson_unlikely(!str)) return false;
    unsafe_yyjson_set_strn(val, str, len);
    return true;
}

yyjson_api_inline bool yyjson_set_str_noesc(yyjson_val *val, bool noesc) {
    if (yyjson_unlikely(!yyjson_is_str(val))) return false;
    unsafe_yyjson_set_str_noesc(val, noesc);
    return true;
}



/*==============================================================================
 * JSON Array API (Implementation)
 *============================================================================*/

yyjson_api_inline size_t yyjson_arr_size(yyjson_val *arr) {
    return yyjson_is_arr(arr) ? unsafe_yyjson_get_len(arr) : 0;
}

yyjson_api_inline yyjson_val *yyjson_arr_get(yyjson_val *arr, size_t idx) {
    if (yyjson_likely(yyjson_is_arr(arr))) {
        if (yyjson_likely(unsafe_yyjson_get_len(arr) > idx)) {
            yyjson_val *val = unsafe_yyjson_get_first(arr);
            if (unsafe_yyjson_arr_is_flat(arr)) {
                return val + idx;
            } else {
                while (idx-- > 0) val = unsafe_yyjson_get_next(val);
                return val;
            }
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_val *yyjson_arr_get_first(yyjson_val *arr) {
    if (yyjson_likely(yyjson_is_arr(arr))) {
        if (yyjson_likely(unsafe_yyjson_get_len(arr) > 0)) {
            return unsafe_yyjson_get_first(arr);
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_val *yyjson_arr_get_last(yyjson_val *arr) {
    if (yyjson_likely(yyjson_is_arr(arr))) {
        size_t len = unsafe_yyjson_get_len(arr);
        if (yyjson_likely(len > 0)) {
            yyjson_val *val = unsafe_yyjson_get_first(arr);
            if (unsafe_yyjson_arr_is_flat(arr)) {
                return val + (len - 1);
            } else {
                while (len-- > 1) val = unsafe_yyjson_get_next(val);
                return val;
            }
        }
    }
    return NULL;
}



/*==============================================================================
 * JSON Array Iterator API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_arr_iter_init(yyjson_val *arr,
                                            yyjson_arr_iter *iter) {
    if (yyjson_likely(yyjson_is_arr(arr) && iter)) {
        iter->idx = 0;
        iter->max = unsafe_yyjson_get_len(arr);
        iter->cur = unsafe_yyjson_get_first(arr);
        return true;
    }
    if (iter) memset(iter, 0, sizeof(yyjson_arr_iter));
    return false;
}

yyjson_api_inline yyjson_arr_iter yyjson_arr_iter_with(yyjson_val *arr) {
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(arr, &iter);
    return iter;
}

yyjson_api_inline bool yyjson_arr_iter_has_next(yyjson_arr_iter *iter) {
    return iter ? iter->idx < iter->max : false;
}

yyjson_api_inline yyjson_val *yyjson_arr_iter_next(yyjson_arr_iter *iter) {
    yyjson_val *val;
    if (iter && iter->idx < iter->max) {
        val = iter->cur;
        iter->cur = unsafe_yyjson_get_next(val);
        iter->idx++;
        return val;
    }
    return NULL;
}



/*==============================================================================
 * JSON Object API (Implementation)
 *============================================================================*/

yyjson_api_inline size_t yyjson_obj_size(yyjson_val *obj) {
    return yyjson_is_obj(obj) ? unsafe_yyjson_get_len(obj) : 0;
}

yyjson_api_inline yyjson_val *yyjson_obj_get(yyjson_val *obj,
                                             const char *key) {
    return yyjson_obj_getn(obj, key, key ? strlen(key) : 0);
}

yyjson_api_inline yyjson_val *yyjson_obj_getn(yyjson_val *obj,
                                              const char *_key,
                                              size_t key_len) {
    if (yyjson_likely(yyjson_is_obj(obj) && _key)) {
        size_t len = unsafe_yyjson_get_len(obj);
        yyjson_val *key = unsafe_yyjson_get_first(obj);
        while (len-- > 0) {
            if (unsafe_yyjson_equals_strn(key, _key, key_len)) return key + 1;
            key = unsafe_yyjson_get_next(key + 1);
        }
    }
    return NULL;
}



/*==============================================================================
 * JSON Object Iterator API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_obj_iter_init(yyjson_val *obj,
                                            yyjson_obj_iter *iter) {
    if (yyjson_likely(yyjson_is_obj(obj) && iter)) {
        iter->idx = 0;
        iter->max = unsafe_yyjson_get_len(obj);
        iter->cur = unsafe_yyjson_get_first(obj);
        iter->obj = obj;
        return true;
    }
    if (iter) memset(iter, 0, sizeof(yyjson_obj_iter));
    return false;
}

yyjson_api_inline yyjson_obj_iter yyjson_obj_iter_with(yyjson_val *obj) {
    yyjson_obj_iter iter;
    yyjson_obj_iter_init(obj, &iter);
    return iter;
}

yyjson_api_inline bool yyjson_obj_iter_has_next(yyjson_obj_iter *iter) {
    return iter ? iter->idx < iter->max : false;
}

yyjson_api_inline yyjson_val *yyjson_obj_iter_next(yyjson_obj_iter *iter) {
    if (iter && iter->idx < iter->max) {
        yyjson_val *key = iter->cur;
        iter->idx++;
        iter->cur = unsafe_yyjson_get_next(key + 1);
        return key;
    }
    return NULL;
}

yyjson_api_inline yyjson_val *yyjson_obj_iter_get_val(yyjson_val *key) {
    return key ? key + 1 : NULL;
}

yyjson_api_inline yyjson_val *yyjson_obj_iter_get(yyjson_obj_iter *iter,
                                                  const char *key) {
    return yyjson_obj_iter_getn(iter, key, key ? strlen(key) : 0);
}

yyjson_api_inline yyjson_val *yyjson_obj_iter_getn(yyjson_obj_iter *iter,
                                                   const char *key,
                                                   size_t key_len) {
    if (iter && key) {
        size_t idx = iter->idx;
        size_t max = iter->max;
        yyjson_val *cur = iter->cur;
        if (yyjson_unlikely(idx == max)) {
            idx = 0;
            cur = unsafe_yyjson_get_first(iter->obj);
        }
        while (idx++ < max) {
            yyjson_val *next = unsafe_yyjson_get_next(cur + 1);
            if (unsafe_yyjson_equals_strn(cur, key, key_len)) {
                iter->idx = idx;
                iter->cur = next;
                return cur + 1;
            }
            cur = next;
            if (idx == iter->max && iter->idx < iter->max) {
                idx = 0;
                max = iter->idx;
                cur = unsafe_yyjson_get_first(iter->obj);
            }
        }
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Structure (Implementation)
 *============================================================================*/

/**
 Mutable JSON value, 24 bytes.
 The 'tag' and 'uni' field is same as immutable value.
 The 'next' field links all elements inside the container to be a cycle.
 */
struct yyjson_mut_val {
    uint64_t tag; /**< type, subtype and length */
    yyjson_val_uni uni; /**< payload */
    yyjson_mut_val *next; /**< the next value in circular linked list */
};

/**
 A memory chunk in string memory pool.
 */
typedef struct yyjson_str_chunk {
    struct yyjson_str_chunk *next; /* next chunk linked list */
    size_t chunk_size; /* chunk size in bytes */
    /* char str[]; flexible array member */
} yyjson_str_chunk;

/**
 A memory pool to hold all strings in a mutable document.
 */
typedef struct yyjson_str_pool {
    char *cur; /* cursor inside current chunk */
    char *end; /* the end of current chunk */
    size_t chunk_size; /* chunk size in bytes while creating new chunk */
    size_t chunk_size_max; /* maximum chunk size in bytes */
    yyjson_str_chunk *chunks; /* a linked list of chunks, nullable */
} yyjson_str_pool;

/**
 A memory chunk in value memory pool.
 `sizeof(yyjson_val_chunk)` should not larger than `sizeof(yyjson_mut_val)`.
 */
typedef struct yyjson_val_chunk {
    struct yyjson_val_chunk *next; /* next chunk linked list */
    size_t chunk_size; /* chunk size in bytes */
    /* char pad[sizeof(yyjson_mut_val) - sizeof(yyjson_val_chunk)]; padding */
    /* yyjson_mut_val vals[]; flexible array member */
} yyjson_val_chunk;

/**
 A memory pool to hold all values in a mutable document.
 */
typedef struct yyjson_val_pool {
    yyjson_mut_val *cur; /* cursor inside current chunk */
    yyjson_mut_val *end; /* the end of current chunk */
    size_t chunk_size; /* chunk size in bytes while creating new chunk */
    size_t chunk_size_max; /* maximum chunk size in bytes */
    yyjson_val_chunk *chunks; /* a linked list of chunks, nullable */
} yyjson_val_pool;

struct yyjson_mut_doc {
    yyjson_mut_val *root; /**< root value of the JSON document, nullable */
    yyjson_alc alc; /**< a valid allocator, nonnull */
    yyjson_str_pool str_pool; /**< string memory pool */
    yyjson_val_pool val_pool; /**< value memory pool */
};

/* Ensures the capacity to at least equal to the specified byte length. */
yyjson_api bool unsafe_yyjson_str_pool_grow(yyjson_str_pool *pool,
                                            const yyjson_alc *alc,
                                            size_t len);

/* Ensures the capacity to at least equal to the specified value count. */
yyjson_api bool unsafe_yyjson_val_pool_grow(yyjson_val_pool *pool,
                                            const yyjson_alc *alc,
                                            size_t count);

/* Allocate memory for string. */
yyjson_api_inline char *unsafe_yyjson_mut_str_alc(yyjson_mut_doc *doc,
                                                  size_t len) {
    char *mem;
    const yyjson_alc *alc = &doc->alc;
    yyjson_str_pool *pool = &doc->str_pool;
    if (yyjson_unlikely((size_t)(pool->end - pool->cur) <= len)) {
        if (yyjson_unlikely(!unsafe_yyjson_str_pool_grow(pool, alc, len + 1))) {
            return NULL;
        }
    }
    mem = pool->cur;
    pool->cur = mem + len + 1;
    return mem;
}

yyjson_api_inline char *unsafe_yyjson_mut_strncpy(yyjson_mut_doc *doc,
                                                  const char *str, size_t len) {
    char *mem = unsafe_yyjson_mut_str_alc(doc, len);
    if (yyjson_unlikely(!mem)) return NULL;
    memcpy((void *)mem, (const void *)str, len);
    mem[len] = '\0';
    return mem;
}

yyjson_api_inline yyjson_mut_val *unsafe_yyjson_mut_val(yyjson_mut_doc *doc,
                                                        size_t count) {
    yyjson_mut_val *val;
    yyjson_alc *alc = &doc->alc;
    yyjson_val_pool *pool = &doc->val_pool;
    if (yyjson_unlikely((size_t)(pool->end - pool->cur) < count)) {
        if (yyjson_unlikely(!unsafe_yyjson_val_pool_grow(pool, alc, count))) {
            return NULL;
        }
    }
    val = pool->cur;
    pool->cur += count;
    return val;
}



/*==============================================================================
 * Mutable JSON Document API (Implementation)
 *============================================================================*/

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_get_root(yyjson_mut_doc *doc) {
    return doc ? doc->root : NULL;
}

yyjson_api_inline void yyjson_mut_doc_set_root(yyjson_mut_doc *doc,
                                               yyjson_mut_val *root) {
    if (doc) doc->root = root;
}



/*==============================================================================
 * Mutable JSON Value Type API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_mut_is_raw(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_raw(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_null(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_null(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_true(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_true(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_false(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_false(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_bool(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_bool(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_uint(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_uint(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_sint(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_sint(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_int(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_int(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_real(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_real(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_num(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_num(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_str(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_str(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_arr(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_arr(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_obj(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_obj(val) : false;
}

yyjson_api_inline bool yyjson_mut_is_ctn(yyjson_mut_val *val) {
    return val ? unsafe_yyjson_is_ctn(val) : false;
}



/*==============================================================================
 * Mutable JSON Value Content API (Implementation)
 *============================================================================*/

yyjson_api_inline yyjson_type yyjson_mut_get_type(yyjson_mut_val *val) {
    return yyjson_get_type((yyjson_val *)val);
}

yyjson_api_inline yyjson_subtype yyjson_mut_get_subtype(yyjson_mut_val *val) {
    return yyjson_get_subtype((yyjson_val *)val);
}

yyjson_api_inline uint8_t yyjson_mut_get_tag(yyjson_mut_val *val) {
    return yyjson_get_tag((yyjson_val *)val);
}

yyjson_api_inline const char *yyjson_mut_get_type_desc(yyjson_mut_val *val) {
    return yyjson_get_type_desc((yyjson_val *)val);
}

yyjson_api_inline const char *yyjson_mut_get_raw(yyjson_mut_val *val) {
    return yyjson_get_raw((yyjson_val *)val);
}

yyjson_api_inline bool yyjson_mut_get_bool(yyjson_mut_val *val) {
    return yyjson_get_bool((yyjson_val *)val);
}

yyjson_api_inline uint64_t yyjson_mut_get_uint(yyjson_mut_val *val) {
    return yyjson_get_uint((yyjson_val *)val);
}

yyjson_api_inline int64_t yyjson_mut_get_sint(yyjson_mut_val *val) {
    return yyjson_get_sint((yyjson_val *)val);
}

yyjson_api_inline int yyjson_mut_get_int(yyjson_mut_val *val) {
    return yyjson_get_int((yyjson_val *)val);
}

yyjson_api_inline double yyjson_mut_get_real(yyjson_mut_val *val) {
    return yyjson_get_real((yyjson_val *)val);
}

yyjson_api_inline double yyjson_mut_get_num(yyjson_mut_val *val) {
    return yyjson_get_num((yyjson_val *)val);
}

yyjson_api_inline const char *yyjson_mut_get_str(yyjson_mut_val *val) {
    return yyjson_get_str((yyjson_val *)val);
}

yyjson_api_inline size_t yyjson_mut_get_len(yyjson_mut_val *val) {
    return yyjson_get_len((yyjson_val *)val);
}

yyjson_api_inline bool yyjson_mut_equals_str(yyjson_mut_val *val,
                                             const char *str) {
    return yyjson_equals_str((yyjson_val *)val, str);
}

yyjson_api_inline bool yyjson_mut_equals_strn(yyjson_mut_val *val,
                                              const char *str, size_t len) {
    return yyjson_equals_strn((yyjson_val *)val, str, len);
}

yyjson_api bool unsafe_yyjson_mut_equals(yyjson_mut_val *lhs,
                                         yyjson_mut_val *rhs);

yyjson_api_inline bool yyjson_mut_equals(yyjson_mut_val *lhs,
                                         yyjson_mut_val *rhs) {
    if (yyjson_unlikely(!lhs || !rhs)) return false;
    return unsafe_yyjson_mut_equals(lhs, rhs);
}

yyjson_api_inline bool yyjson_mut_set_raw(yyjson_mut_val *val,
                                          const char *raw, size_t len) {
    if (yyjson_unlikely(!val || !raw)) return false;
    unsafe_yyjson_set_raw(val, raw, len);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_null(yyjson_mut_val *val) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_null(val);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_bool(yyjson_mut_val *val, bool num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_bool(val, num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_uint(yyjson_mut_val *val, uint64_t num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_uint(val, num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_sint(yyjson_mut_val *val, int64_t num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_sint(val, num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_int(yyjson_mut_val *val, int num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_sint(val, (int64_t)num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_float(yyjson_mut_val *val, float num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_float(val, num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_double(yyjson_mut_val *val, double num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_double(val, num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_real(yyjson_mut_val *val, double num) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_real(val, num);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_fp_to_fixed(yyjson_mut_val *val,
                                                  int prec) {
    if (yyjson_unlikely(!yyjson_mut_is_real(val))) return false;
    unsafe_yyjson_set_fp_to_fixed(val, prec);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_fp_to_float(yyjson_mut_val *val,
                                                  bool flt) {
    if (yyjson_unlikely(!yyjson_mut_is_real(val))) return false;
    unsafe_yyjson_set_fp_to_float(val, flt);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_str(yyjson_mut_val *val,
                                          const char *str) {
    if (yyjson_unlikely(!val || !str)) return false;
    unsafe_yyjson_set_str(val, str);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_strn(yyjson_mut_val *val,
                                           const char *str, size_t len) {
    if (yyjson_unlikely(!val || !str)) return false;
    unsafe_yyjson_set_strn(val, str, len);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_str_noesc(yyjson_mut_val *val,
                                                bool noesc) {
    if (yyjson_unlikely(!yyjson_mut_is_str(val))) return false;
    unsafe_yyjson_set_str_noesc(val, noesc);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_arr(yyjson_mut_val *val) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_arr(val, 0);
    return true;
}

yyjson_api_inline bool yyjson_mut_set_obj(yyjson_mut_val *val) {
    if (yyjson_unlikely(!val)) return false;
    unsafe_yyjson_set_obj(val, 0);
    return true;
}



/*==============================================================================
 * Mutable JSON Value Creation API (Implementation)
 *============================================================================*/

#define yyjson_mut_val_one(func) \
    if (yyjson_likely(doc)) { \
        yyjson_mut_val *val = unsafe_yyjson_mut_val(doc, 1); \
        if (yyjson_likely(val)) { \
            func \
            return val; \
        } \
    } \
    return NULL

#define yyjson_mut_val_one_str(func) \
    if (yyjson_likely(doc && str)) { \
        yyjson_mut_val *val = unsafe_yyjson_mut_val(doc, 1); \
        if (yyjson_likely(val)) { \
            func \
            return val; \
        } \
    } \
    return NULL

yyjson_api_inline yyjson_mut_val *yyjson_mut_raw(yyjson_mut_doc *doc,
                                                 const char *str) {
    yyjson_mut_val_one_str({ unsafe_yyjson_set_raw(val, str, strlen(str)); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_rawn(yyjson_mut_doc *doc,
                                                  const char *str,
                                                  size_t len) {
    yyjson_mut_val_one_str({ unsafe_yyjson_set_raw(val, str, len); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_rawcpy(yyjson_mut_doc *doc,
                                                    const char *str) {
    yyjson_mut_val_one_str({
        size_t len = strlen(str);
        char *new_str = unsafe_yyjson_mut_strncpy(doc, str, len);
        if (yyjson_unlikely(!new_str)) return NULL;
        unsafe_yyjson_set_raw(val, new_str, len);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_rawncpy(yyjson_mut_doc *doc,
                                                     const char *str,
                                                     size_t len) {
    yyjson_mut_val_one_str({
        char *new_str = unsafe_yyjson_mut_strncpy(doc, str, len);
        if (yyjson_unlikely(!new_str)) return NULL;
        unsafe_yyjson_set_raw(val, new_str, len);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_null(yyjson_mut_doc *doc) {
    yyjson_mut_val_one({ unsafe_yyjson_set_null(val); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_true(yyjson_mut_doc *doc) {
    yyjson_mut_val_one({ unsafe_yyjson_set_bool(val, true); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_false(yyjson_mut_doc *doc) {
    yyjson_mut_val_one({ unsafe_yyjson_set_bool(val, false); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_bool(yyjson_mut_doc *doc,
                                                  bool _val) {
    yyjson_mut_val_one({ unsafe_yyjson_set_bool(val, _val); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_uint(yyjson_mut_doc *doc,
                                                  uint64_t num) {
    yyjson_mut_val_one({ unsafe_yyjson_set_uint(val, num); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_sint(yyjson_mut_doc *doc,
                                                  int64_t num) {
    yyjson_mut_val_one({ unsafe_yyjson_set_sint(val, num); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_int(yyjson_mut_doc *doc,
                                                 int64_t num) {
    yyjson_mut_val_one({ unsafe_yyjson_set_sint(val, num); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_float(yyjson_mut_doc *doc,
                                                   float num) {
    yyjson_mut_val_one({ unsafe_yyjson_set_float(val, num); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_double(yyjson_mut_doc *doc,
                                                    double num) {
    yyjson_mut_val_one({ unsafe_yyjson_set_double(val, num); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_real(yyjson_mut_doc *doc,
                                                  double num) {
    yyjson_mut_val_one({ unsafe_yyjson_set_real(val, num); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_str(yyjson_mut_doc *doc,
                                                 const char *str) {
    yyjson_mut_val_one_str({ unsafe_yyjson_set_str(val, str); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_strn(yyjson_mut_doc *doc,
                                                  const char *str,
                                                  size_t len) {
    yyjson_mut_val_one_str({ unsafe_yyjson_set_strn(val, str, len); });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_strcpy(yyjson_mut_doc *doc,
                                                    const char *str) {
    yyjson_mut_val_one_str({
        size_t len = strlen(str);
        bool noesc = unsafe_yyjson_is_str_noesc(str, len);
        yyjson_subtype sub = noesc ? YYJSON_SUBTYPE_NOESC : YYJSON_SUBTYPE_NONE;
        char *new_str = unsafe_yyjson_mut_strncpy(doc, str, len);
        if (yyjson_unlikely(!new_str)) return NULL;
        unsafe_yyjson_set_tag(val, YYJSON_TYPE_STR, sub, len);
        val->uni.str = new_str;
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_strncpy(yyjson_mut_doc *doc,
                                                     const char *str,
                                                     size_t len) {
    yyjson_mut_val_one_str({
        char *new_str = unsafe_yyjson_mut_strncpy(doc, str, len);
        if (yyjson_unlikely(!new_str)) return NULL;
        unsafe_yyjson_set_strn(val, new_str, len);
    });
}

#undef yyjson_mut_val_one
#undef yyjson_mut_val_one_str



/*==============================================================================
 * Mutable JSON Array API (Implementation)
 *============================================================================*/

yyjson_api_inline size_t yyjson_mut_arr_size(yyjson_mut_val *arr) {
    return yyjson_mut_is_arr(arr) ? unsafe_yyjson_get_len(arr) : 0;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_get(yyjson_mut_val *arr,
                                                     size_t idx) {
    if (yyjson_likely(idx < yyjson_mut_arr_size(arr))) {
        yyjson_mut_val *val = (yyjson_mut_val *)arr->uni.ptr;
        while (idx-- > 0) val = val->next;
        return val->next;
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_get_first(
    yyjson_mut_val *arr) {
    if (yyjson_likely(yyjson_mut_arr_size(arr) > 0)) {
        return ((yyjson_mut_val *)arr->uni.ptr)->next;
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_get_last(
    yyjson_mut_val *arr) {
    if (yyjson_likely(yyjson_mut_arr_size(arr) > 0)) {
        return ((yyjson_mut_val *)arr->uni.ptr);
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Array Iterator API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_mut_arr_iter_init(yyjson_mut_val *arr,
                                                yyjson_mut_arr_iter *iter) {
    if (yyjson_likely(yyjson_mut_is_arr(arr) && iter)) {
        iter->idx = 0;
        iter->max = unsafe_yyjson_get_len(arr);
        iter->cur = iter->max ? (yyjson_mut_val *)arr->uni.ptr : NULL;
        iter->pre = NULL;
        iter->arr = arr;
        return true;
    }
    if (iter) memset(iter, 0, sizeof(yyjson_mut_arr_iter));
    return false;
}

yyjson_api_inline yyjson_mut_arr_iter yyjson_mut_arr_iter_with(
    yyjson_mut_val *arr) {
    yyjson_mut_arr_iter iter;
    yyjson_mut_arr_iter_init(arr, &iter);
    return iter;
}

yyjson_api_inline bool yyjson_mut_arr_iter_has_next(yyjson_mut_arr_iter *iter) {
    return iter ? iter->idx < iter->max : false;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_iter_next(
    yyjson_mut_arr_iter *iter) {
    if (iter && iter->idx < iter->max) {
        yyjson_mut_val *val = iter->cur;
        iter->pre = val;
        iter->cur = val->next;
        iter->idx++;
        return iter->cur;
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_iter_remove(
    yyjson_mut_arr_iter *iter) {
    if (yyjson_likely(iter && 0 < iter->idx && iter->idx <= iter->max)) {
        yyjson_mut_val *prev = iter->pre;
        yyjson_mut_val *cur = iter->cur;
        yyjson_mut_val *next = cur->next;
        if (yyjson_unlikely(iter->idx == iter->max)) iter->arr->uni.ptr = prev;
        iter->idx--;
        iter->max--;
        unsafe_yyjson_set_len(iter->arr, iter->max);
        prev->next = next;
        iter->cur = prev;
        return cur;
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Array Creation API (Implementation)
 *============================================================================*/

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr(yyjson_mut_doc *doc) {
    if (yyjson_likely(doc)) {
        yyjson_mut_val *val = unsafe_yyjson_mut_val(doc, 1);
        if (yyjson_likely(val)) {
            val->tag = YYJSON_TYPE_ARR | YYJSON_SUBTYPE_NONE;
            return val;
        }
    }
    return NULL;
}

#define yyjson_mut_arr_with_func(func) \
    if (yyjson_likely(doc && ((0 < count && count < \
        (~(size_t)0) / sizeof(yyjson_mut_val) && vals) || count == 0))) { \
        yyjson_mut_val *arr = unsafe_yyjson_mut_val(doc, 1 + count); \
        if (yyjson_likely(arr)) { \
            arr->tag = ((uint64_t)count << YYJSON_TAG_BIT) | YYJSON_TYPE_ARR; \
            if (count > 0) { \
                size_t i; \
                for (i = 0; i < count; i++) { \
                    yyjson_mut_val *val = arr + i + 1; \
                    func \
                    val->next = val + 1; \
                } \
                arr[count].next = arr + 1; \
                arr->uni.ptr = arr + count; \
            } \
            return arr; \
        } \
    } \
    return NULL

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_bool(
    yyjson_mut_doc *doc, const bool *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_bool(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint(
    yyjson_mut_doc *doc, const int64_t *vals, size_t count) {
    return yyjson_mut_arr_with_sint64(doc, vals, count);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint(
    yyjson_mut_doc *doc, const uint64_t *vals, size_t count) {
    return yyjson_mut_arr_with_uint64(doc, vals, count);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_real(
    yyjson_mut_doc *doc, const double *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_real(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint8(
    yyjson_mut_doc *doc, const int8_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_sint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint16(
    yyjson_mut_doc *doc, const int16_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_sint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint32(
    yyjson_mut_doc *doc, const int32_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_sint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_sint64(
    yyjson_mut_doc *doc, const int64_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_sint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint8(
    yyjson_mut_doc *doc, const uint8_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_uint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint16(
    yyjson_mut_doc *doc, const uint16_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_uint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint32(
    yyjson_mut_doc *doc, const uint32_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_uint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_uint64(
    yyjson_mut_doc *doc, const uint64_t *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_uint(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_float(
    yyjson_mut_doc *doc, const float *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_float(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_double(
    yyjson_mut_doc *doc, const double *vals, size_t count) {
    yyjson_mut_arr_with_func({
        unsafe_yyjson_set_double(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_str(
    yyjson_mut_doc *doc, const char **vals, size_t count) {
    yyjson_mut_arr_with_func({
        if (yyjson_unlikely(!vals[i])) return NULL;
        unsafe_yyjson_set_str(val, vals[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_strn(
    yyjson_mut_doc *doc, const char **vals, const size_t *lens, size_t count) {
    if (yyjson_unlikely(count > 0 && !lens)) return NULL;
    yyjson_mut_arr_with_func({
        if (yyjson_unlikely(!vals[i])) return NULL;
        unsafe_yyjson_set_strn(val, vals[i], lens[i]);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_strcpy(
    yyjson_mut_doc *doc, const char **vals, size_t count) {
    size_t len;
    const char *str, *new_str;
    yyjson_mut_arr_with_func({
        str = vals[i];
        if (yyjson_unlikely(!str)) return NULL;
        len = strlen(str);
        new_str = unsafe_yyjson_mut_strncpy(doc, str, len);
        if (yyjson_unlikely(!new_str)) return NULL;
        unsafe_yyjson_set_strn(val, new_str, len);
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_with_strncpy(
    yyjson_mut_doc *doc, const char **vals, const size_t *lens, size_t count) {
    size_t len;
    const char *str, *new_str;
    if (yyjson_unlikely(count > 0 && !lens)) return NULL;
    yyjson_mut_arr_with_func({
        str = vals[i];
        if (yyjson_unlikely(!str)) return NULL;
        len = lens[i];
        new_str = unsafe_yyjson_mut_strncpy(doc, str, len);
        if (yyjson_unlikely(!new_str)) return NULL;
        unsafe_yyjson_set_strn(val, new_str, len);
    });
}

#undef yyjson_mut_arr_with_func



/*==============================================================================
 * Mutable JSON Array Modification API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_mut_arr_insert(yyjson_mut_val *arr,
                                             yyjson_mut_val *val, size_t idx) {
    if (yyjson_likely(yyjson_mut_is_arr(arr) && val)) {
        size_t len = unsafe_yyjson_get_len(arr);
        if (yyjson_likely(idx <= len)) {
            unsafe_yyjson_set_len(arr, len + 1);
            if (len == 0) {
                val->next = val;
                arr->uni.ptr = val;
            } else {
                yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
                yyjson_mut_val *next = prev->next;
                if (idx == len) {
                    prev->next = val;
                    val->next = next;
                    arr->uni.ptr = val;
                } else {
                    while (idx-- > 0) {
                        prev = next;
                        next = next->next;
                    }
                    prev->next = val;
                    val->next = next;
                }
            }
            return true;
        }
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_append(yyjson_mut_val *arr,
                                             yyjson_mut_val *val) {
    if (yyjson_likely(yyjson_mut_is_arr(arr) && val)) {
        size_t len = unsafe_yyjson_get_len(arr);
        unsafe_yyjson_set_len(arr, len + 1);
        if (len == 0) {
            val->next = val;
        } else {
            yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
            yyjson_mut_val *next = prev->next;
            prev->next = val;
            val->next = next;
        }
        arr->uni.ptr = val;
        return true;
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_prepend(yyjson_mut_val *arr,
                                              yyjson_mut_val *val) {
    if (yyjson_likely(yyjson_mut_is_arr(arr) && val)) {
        size_t len = unsafe_yyjson_get_len(arr);
        unsafe_yyjson_set_len(arr, len + 1);
        if (len == 0) {
            val->next = val;
            arr->uni.ptr = val;
        } else {
            yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
            yyjson_mut_val *next = prev->next;
            prev->next = val;
            val->next = next;
        }
        return true;
    }
    return false;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_replace(yyjson_mut_val *arr,
                                                         size_t idx,
                                                         yyjson_mut_val *val) {
    if (yyjson_likely(yyjson_mut_is_arr(arr) && val)) {
        size_t len = unsafe_yyjson_get_len(arr);
        if (yyjson_likely(idx < len)) {
            if (yyjson_likely(len > 1)) {
                yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
                yyjson_mut_val *next = prev->next;
                while (idx-- > 0) {
                    prev = next;
                    next = next->next;
                }
                prev->next = val;
                val->next = next->next;
                if ((void *)next == arr->uni.ptr) arr->uni.ptr = val;
                return next;
            } else {
                yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
                val->next = val;
                arr->uni.ptr = val;
                return prev;
            }
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_remove(yyjson_mut_val *arr,
                                                        size_t idx) {
    if (yyjson_likely(yyjson_mut_is_arr(arr))) {
        size_t len = unsafe_yyjson_get_len(arr);
        if (yyjson_likely(idx < len)) {
            unsafe_yyjson_set_len(arr, len - 1);
            if (yyjson_likely(len > 1)) {
                yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
                yyjson_mut_val *next = prev->next;
                while (idx-- > 0) {
                    prev = next;
                    next = next->next;
                }
                prev->next = next->next;
                if ((void *)next == arr->uni.ptr) arr->uni.ptr = prev;
                return next;
            } else {
                return ((yyjson_mut_val *)arr->uni.ptr);
            }
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_remove_first(
    yyjson_mut_val *arr) {
    if (yyjson_likely(yyjson_mut_is_arr(arr))) {
        size_t len = unsafe_yyjson_get_len(arr);
        if (len > 1) {
            yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
            yyjson_mut_val *next = prev->next;
            prev->next = next->next;
            unsafe_yyjson_set_len(arr, len - 1);
            return next;
        } else if (len == 1) {
            yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
            unsafe_yyjson_set_len(arr, 0);
            return prev;
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_remove_last(
    yyjson_mut_val *arr) {
    if (yyjson_likely(yyjson_mut_is_arr(arr))) {
        size_t len = unsafe_yyjson_get_len(arr);
        if (yyjson_likely(len > 1)) {
            yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
            yyjson_mut_val *next = prev->next;
            unsafe_yyjson_set_len(arr, len - 1);
            while (--len > 0) prev = prev->next;
            prev->next = next;
            next = (yyjson_mut_val *)arr->uni.ptr;
            arr->uni.ptr = prev;
            return next;
        } else if (len == 1) {
            yyjson_mut_val *prev = ((yyjson_mut_val *)arr->uni.ptr);
            unsafe_yyjson_set_len(arr, 0);
            return prev;
        }
    }
    return NULL;
}

yyjson_api_inline bool yyjson_mut_arr_remove_range(yyjson_mut_val *arr,
                                                   size_t _idx, size_t _len) {
    if (yyjson_likely(yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *prev, *next;
        bool tail_removed;
        size_t len = unsafe_yyjson_get_len(arr);
        if (yyjson_unlikely(_idx + _len > len)) return false;
        if (yyjson_unlikely(_len == 0)) return true;
        unsafe_yyjson_set_len(arr, len - _len);
        if (yyjson_unlikely(len == _len)) return true;
        tail_removed = (_idx + _len == len);
        prev = ((yyjson_mut_val *)arr->uni.ptr);
        while (_idx-- > 0) prev = prev->next;
        next = prev->next;
        while (_len-- > 0) next = next->next;
        prev->next = next;
        if (yyjson_unlikely(tail_removed)) arr->uni.ptr = prev;
        return true;
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_clear(yyjson_mut_val *arr) {
    if (yyjson_likely(yyjson_mut_is_arr(arr))) {
        unsafe_yyjson_set_len(arr, 0);
        return true;
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_rotate(yyjson_mut_val *arr,
                                             size_t idx) {
    if (yyjson_likely(yyjson_mut_is_arr(arr) &&
                      unsafe_yyjson_get_len(arr) > idx)) {
        yyjson_mut_val *val = (yyjson_mut_val *)arr->uni.ptr;
        while (idx-- > 0) val = val->next;
        arr->uni.ptr = (void *)val;
        return true;
    }
    return false;
}



/*==============================================================================
 * Mutable JSON Array Modification Convenience API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_mut_arr_add_val(yyjson_mut_val *arr,
                                              yyjson_mut_val *val) {
    return yyjson_mut_arr_append(arr, val);
}

yyjson_api_inline bool yyjson_mut_arr_add_null(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_null(doc);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_true(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_true(doc);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_false(yyjson_mut_doc *doc,
                                                yyjson_mut_val *arr) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_false(doc);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_bool(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               bool _val) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_bool(doc, _val);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_uint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               uint64_t num) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_uint(doc, num);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_sint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               int64_t num) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_sint(doc, num);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_int(yyjson_mut_doc *doc,
                                              yyjson_mut_val *arr,
                                              int64_t num) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_sint(doc, num);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_float(yyjson_mut_doc *doc,
                                                yyjson_mut_val *arr,
                                                float num) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_float(doc, num);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_double(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *arr,
                                                 double num) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_double(doc, num);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_real(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               double num) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_real(doc, num);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_str(yyjson_mut_doc *doc,
                                              yyjson_mut_val *arr,
                                              const char *str) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_str(doc, str);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_strn(yyjson_mut_doc *doc,
                                               yyjson_mut_val *arr,
                                               const char *str, size_t len) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_strn(doc, str, len);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_strcpy(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *arr,
                                                 const char *str) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_strcpy(doc, str);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_arr_add_strncpy(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *arr,
                                                  const char *str, size_t len) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_strncpy(doc, str, len);
        return yyjson_mut_arr_append(arr, val);
    }
    return false;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_add_arr(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *arr) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_arr(doc);
        return yyjson_mut_arr_append(arr, val) ? val : NULL;
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_arr_add_obj(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *arr) {
    if (yyjson_likely(doc && yyjson_mut_is_arr(arr))) {
        yyjson_mut_val *val = yyjson_mut_obj(doc);
        return yyjson_mut_arr_append(arr, val) ? val : NULL;
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Object API (Implementation)
 *============================================================================*/

yyjson_api_inline size_t yyjson_mut_obj_size(yyjson_mut_val *obj) {
    return yyjson_mut_is_obj(obj) ? unsafe_yyjson_get_len(obj) : 0;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_get(yyjson_mut_val *obj,
                                                     const char *key) {
    return yyjson_mut_obj_getn(obj, key, key ? strlen(key) : 0);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_getn(yyjson_mut_val *obj,
                                                      const char *_key,
                                                      size_t key_len) {
    size_t len = yyjson_mut_obj_size(obj);
    if (yyjson_likely(len && _key)) {
        yyjson_mut_val *key = ((yyjson_mut_val *)obj->uni.ptr)->next->next;
        while (len-- > 0) {
            if (unsafe_yyjson_equals_strn(key, _key, key_len)) return key->next;
            key = key->next->next;
        }
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Object Iterator API (Implementation)
 *============================================================================*/

yyjson_api_inline bool yyjson_mut_obj_iter_init(yyjson_mut_val *obj,
                                                yyjson_mut_obj_iter *iter) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) && iter)) {
        iter->idx = 0;
        iter->max = unsafe_yyjson_get_len(obj);
        iter->cur = iter->max ? (yyjson_mut_val *)obj->uni.ptr : NULL;
        iter->pre = NULL;
        iter->obj = obj;
        return true;
    }
    if (iter) memset(iter, 0, sizeof(yyjson_mut_obj_iter));
    return false;
}

yyjson_api_inline yyjson_mut_obj_iter yyjson_mut_obj_iter_with(
    yyjson_mut_val *obj) {
    yyjson_mut_obj_iter iter;
    yyjson_mut_obj_iter_init(obj, &iter);
    return iter;
}

yyjson_api_inline bool yyjson_mut_obj_iter_has_next(yyjson_mut_obj_iter *iter) {
    return iter ? iter->idx < iter->max : false;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_next(
    yyjson_mut_obj_iter *iter) {
    if (iter && iter->idx < iter->max) {
        yyjson_mut_val *key = iter->cur;
        iter->pre = key;
        iter->cur = key->next->next;
        iter->idx++;
        return iter->cur;
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_get_val(
    yyjson_mut_val *key) {
    return key ? key->next : NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_remove(
    yyjson_mut_obj_iter *iter) {
    if (yyjson_likely(iter && 0 < iter->idx && iter->idx <= iter->max)) {
        yyjson_mut_val *prev = iter->pre;
        yyjson_mut_val *cur = iter->cur;
        yyjson_mut_val *next = cur->next->next;
        if (yyjson_unlikely(iter->idx == iter->max)) iter->obj->uni.ptr = prev;
        iter->idx--;
        iter->max--;
        unsafe_yyjson_set_len(iter->obj, iter->max);
        prev->next->next = next;
        iter->cur = prev;
        return cur->next;
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_get(
    yyjson_mut_obj_iter *iter, const char *key) {
    return yyjson_mut_obj_iter_getn(iter, key, key ? strlen(key) : 0);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_iter_getn(
    yyjson_mut_obj_iter *iter, const char *key, size_t key_len) {
    if (iter && key) {
        size_t idx = 0;
        size_t max = iter->max;
        yyjson_mut_val *pre, *cur = iter->cur;
        while (idx++ < max) {
            pre = cur;
            cur = cur->next->next;
            if (unsafe_yyjson_equals_strn(cur, key, key_len)) {
                iter->idx += idx;
                if (iter->idx > max) iter->idx -= max + 1;
                iter->pre = pre;
                iter->cur = cur;
                return cur->next;
            }
        }
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Object Creation API (Implementation)
 *============================================================================*/

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj(yyjson_mut_doc *doc) {
    if (yyjson_likely(doc)) {
        yyjson_mut_val *val = unsafe_yyjson_mut_val(doc, 1);
        if (yyjson_likely(val)) {
            val->tag = YYJSON_TYPE_OBJ | YYJSON_SUBTYPE_NONE;
            return val;
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_with_str(yyjson_mut_doc *doc,
                                                          const char **keys,
                                                          const char **vals,
                                                          size_t count) {
    if (yyjson_likely(doc && ((count > 0 && keys && vals) || (count == 0)))) {
        yyjson_mut_val *obj = unsafe_yyjson_mut_val(doc, 1 + count * 2);
        if (yyjson_likely(obj)) {
            obj->tag = ((uint64_t)count << YYJSON_TAG_BIT) | YYJSON_TYPE_OBJ;
            if (count > 0) {
                size_t i;
                for (i = 0; i < count; i++) {
                    yyjson_mut_val *key = obj + (i * 2 + 1);
                    yyjson_mut_val *val = obj + (i * 2 + 2);
                    uint64_t key_len = (uint64_t)strlen(keys[i]);
                    uint64_t val_len = (uint64_t)strlen(vals[i]);
                    key->tag = (key_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
                    val->tag = (val_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
                    key->uni.str = keys[i];
                    val->uni.str = vals[i];
                    key->next = val;
                    val->next = val + 1;
                }
                obj[count * 2].next = obj + 1;
                obj->uni.ptr = obj + (count * 2 - 1);
            }
            return obj;
        }
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_with_kv(yyjson_mut_doc *doc,
                                                         const char **pairs,
                                                         size_t count) {
    if (yyjson_likely(doc && ((count > 0 && pairs) || (count == 0)))) {
        yyjson_mut_val *obj = unsafe_yyjson_mut_val(doc, 1 + count * 2);
        if (yyjson_likely(obj)) {
            obj->tag = ((uint64_t)count << YYJSON_TAG_BIT) | YYJSON_TYPE_OBJ;
            if (count > 0) {
                size_t i;
                for (i = 0; i < count; i++) {
                    yyjson_mut_val *key = obj + (i * 2 + 1);
                    yyjson_mut_val *val = obj + (i * 2 + 2);
                    const char *key_str = pairs[i * 2 + 0];
                    const char *val_str = pairs[i * 2 + 1];
                    uint64_t key_len = (uint64_t)strlen(key_str);
                    uint64_t val_len = (uint64_t)strlen(val_str);
                    key->tag = (key_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
                    val->tag = (val_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
                    key->uni.str = key_str;
                    val->uni.str = val_str;
                    key->next = val;
                    val->next = val + 1;
                }
                obj[count * 2].next = obj + 1;
                obj->uni.ptr = obj + (count * 2 - 1);
            }
            return obj;
        }
    }
    return NULL;
}



/*==============================================================================
 * Mutable JSON Object Modification API (Implementation)
 *============================================================================*/

yyjson_api_inline void unsafe_yyjson_mut_obj_add(yyjson_mut_val *obj,
                                                 yyjson_mut_val *key,
                                                 yyjson_mut_val *val,
                                                 size_t len) {
    if (yyjson_likely(len)) {
        yyjson_mut_val *prev_val = ((yyjson_mut_val *)obj->uni.ptr)->next;
        yyjson_mut_val *next_key = prev_val->next;
        prev_val->next = key;
        val->next = next_key;
    } else {
        val->next = key;
    }
    key->next = val;
    obj->uni.ptr = (void *)key;
    unsafe_yyjson_set_len(obj, len + 1);
}

yyjson_api_inline yyjson_mut_val *unsafe_yyjson_mut_obj_remove(
    yyjson_mut_val *obj, const char *key, size_t key_len) {
    size_t obj_len = unsafe_yyjson_get_len(obj);
    if (obj_len) {
        yyjson_mut_val *pre_key = (yyjson_mut_val *)obj->uni.ptr;
        yyjson_mut_val *cur_key = pre_key->next->next;
        yyjson_mut_val *removed_item = NULL;
        size_t i;
        for (i = 0; i < obj_len; i++) {
            if (unsafe_yyjson_equals_strn(cur_key, key, key_len)) {
                if (!removed_item) removed_item = cur_key->next;
                cur_key = cur_key->next->next;
                pre_key->next->next = cur_key;
                if (i + 1 == obj_len) obj->uni.ptr = pre_key;
                i--;
                obj_len--;
            } else {
                pre_key = cur_key;
                cur_key = cur_key->next->next;
            }
        }
        unsafe_yyjson_set_len(obj, obj_len);
        return removed_item;
    } else {
        return NULL;
    }
}

yyjson_api_inline bool unsafe_yyjson_mut_obj_replace(yyjson_mut_val *obj,
                                                     yyjson_mut_val *key,
                                                     yyjson_mut_val *val) {
    size_t key_len = unsafe_yyjson_get_len(key);
    size_t obj_len = unsafe_yyjson_get_len(obj);
    if (obj_len) {
        yyjson_mut_val *pre_key = (yyjson_mut_val *)obj->uni.ptr;
        yyjson_mut_val *cur_key = pre_key->next->next;
        size_t i;
        for (i = 0; i < obj_len; i++) {
            if (unsafe_yyjson_equals_strn(cur_key, key->uni.str, key_len)) {
                cur_key->next->tag = val->tag;
                cur_key->next->uni.u64 = val->uni.u64;
                return true;
            } else {
                cur_key = cur_key->next->next;
            }
        }
    }
    return false;
}

yyjson_api_inline void unsafe_yyjson_mut_obj_rotate(yyjson_mut_val *obj,
                                                    size_t idx) {
    yyjson_mut_val *key = (yyjson_mut_val *)obj->uni.ptr;
    while (idx-- > 0) key = key->next->next;
    obj->uni.ptr = (void *)key;
}

yyjson_api_inline bool yyjson_mut_obj_add(yyjson_mut_val *obj,
                                          yyjson_mut_val *key,
                                          yyjson_mut_val *val) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) &&
                      yyjson_mut_is_str(key) && val)) {
        unsafe_yyjson_mut_obj_add(obj, key, val, unsafe_yyjson_get_len(obj));
        return true;
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_obj_put(yyjson_mut_val *obj,
                                          yyjson_mut_val *key,
                                          yyjson_mut_val *val) {
    bool replaced = false;
    size_t key_len;
    yyjson_mut_obj_iter iter;
    yyjson_mut_val *cur_key;
    if (yyjson_unlikely(!yyjson_mut_is_obj(obj) ||
                        !yyjson_mut_is_str(key))) return false;
    key_len = unsafe_yyjson_get_len(key);
    yyjson_mut_obj_iter_init(obj, &iter);
    while ((cur_key = yyjson_mut_obj_iter_next(&iter)) != 0) {
        if (unsafe_yyjson_equals_strn(cur_key, key->uni.str, key_len)) {
            if (!replaced && val) {
                replaced = true;
                val->next = cur_key->next->next;
                cur_key->next = val;
            } else {
                yyjson_mut_obj_iter_remove(&iter);
            }
        }
    }
    if (!replaced && val) unsafe_yyjson_mut_obj_add(obj, key, val, iter.max);
    return true;
}

yyjson_api_inline bool yyjson_mut_obj_insert(yyjson_mut_val *obj,
                                             yyjson_mut_val *key,
                                             yyjson_mut_val *val,
                                             size_t idx) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) &&
                      yyjson_mut_is_str(key) && val)) {
        size_t len = unsafe_yyjson_get_len(obj);
        if (yyjson_likely(len >= idx)) {
            if (len > idx) {
                void *ptr = obj->uni.ptr;
                unsafe_yyjson_mut_obj_rotate(obj, idx);
                unsafe_yyjson_mut_obj_add(obj, key, val, len);
                obj->uni.ptr = ptr;
            } else {
                unsafe_yyjson_mut_obj_add(obj, key, val, len);
            }
            return true;
        }
    }
    return false;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove(yyjson_mut_val *obj,
    yyjson_mut_val *key) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) && yyjson_mut_is_str(key))) {
        return unsafe_yyjson_mut_obj_remove(obj, key->uni.str,
                                            unsafe_yyjson_get_len(key));
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_key(
    yyjson_mut_val *obj, const char *key) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) && key)) {
        size_t key_len = strlen(key);
        return unsafe_yyjson_mut_obj_remove(obj, key, key_len);
    }
    return NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_keyn(
    yyjson_mut_val *obj, const char *key, size_t key_len) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) && key)) {
        return unsafe_yyjson_mut_obj_remove(obj, key, key_len);
    }
    return NULL;
}

yyjson_api_inline bool yyjson_mut_obj_clear(yyjson_mut_val *obj) {
    if (yyjson_likely(yyjson_mut_is_obj(obj))) {
        unsafe_yyjson_set_len(obj, 0);
        return true;
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_obj_replace(yyjson_mut_val *obj,
                                              yyjson_mut_val *key,
                                              yyjson_mut_val *val) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) &&
                      yyjson_mut_is_str(key) && val)) {
        return unsafe_yyjson_mut_obj_replace(obj, key, val);
    }
    return false;
}

yyjson_api_inline bool yyjson_mut_obj_rotate(yyjson_mut_val *obj,
                                             size_t idx) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) &&
                      unsafe_yyjson_get_len(obj) > idx)) {
        unsafe_yyjson_mut_obj_rotate(obj, idx);
        return true;
    }
    return false;
}



/*==============================================================================
 * Mutable JSON Object Modification Convenience API (Implementation)
 *============================================================================*/

#define yyjson_mut_obj_add_func(func) \
    if (yyjson_likely(doc && yyjson_mut_is_obj(obj) && _key)) { \
        yyjson_mut_val *key = unsafe_yyjson_mut_val(doc, 2); \
        if (yyjson_likely(key)) { \
            size_t len = unsafe_yyjson_get_len(obj); \
            yyjson_mut_val *val = key + 1; \
            size_t key_len = strlen(_key); \
            bool noesc = unsafe_yyjson_is_str_noesc(_key, key_len); \
            key->tag = YYJSON_TYPE_STR; \
            key->tag |= noesc ? YYJSON_SUBTYPE_NOESC : YYJSON_SUBTYPE_NONE; \
            key->tag |= (uint64_t)strlen(_key) << YYJSON_TAG_BIT; \
            key->uni.str = _key; \
            func \
            unsafe_yyjson_mut_obj_add(obj, key, val, len); \
            return true; \
        } \
    } \
    return false

yyjson_api_inline bool yyjson_mut_obj_add_null(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_null(val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_true(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_bool(val, true); });
}

yyjson_api_inline bool yyjson_mut_obj_add_false(yyjson_mut_doc *doc,
                                                yyjson_mut_val *obj,
                                                const char *_key) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_bool(val, false); });
}

yyjson_api_inline bool yyjson_mut_obj_add_bool(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key,
                                               bool _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_bool(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_uint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key,
                                               uint64_t _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_uint(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_sint(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key,
                                               int64_t _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_sint(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_int(yyjson_mut_doc *doc,
                                              yyjson_mut_val *obj,
                                              const char *_key,
                                              int64_t _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_sint(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_float(yyjson_mut_doc *doc,
                                                yyjson_mut_val *obj,
                                                const char *_key,
                                                float _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_float(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_double(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *obj,
                                                 const char *_key,
                                                 double _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_double(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_real(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key,
                                               double _val) {
    yyjson_mut_obj_add_func({ unsafe_yyjson_set_real(val, _val); });
}

yyjson_api_inline bool yyjson_mut_obj_add_str(yyjson_mut_doc *doc,
                                              yyjson_mut_val *obj,
                                              const char *_key,
                                              const char *_val) {
    if (yyjson_unlikely(!_val)) return false;
    yyjson_mut_obj_add_func({
        size_t val_len = strlen(_val);
        bool val_noesc = unsafe_yyjson_is_str_noesc(_val, val_len);
        val->tag = ((uint64_t)strlen(_val) << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
        val->tag |= val_noesc ? YYJSON_SUBTYPE_NOESC : YYJSON_SUBTYPE_NONE;
        val->uni.str = _val;
    });
}

yyjson_api_inline bool yyjson_mut_obj_add_strn(yyjson_mut_doc *doc,
                                               yyjson_mut_val *obj,
                                               const char *_key,
                                               const char *_val,
                                               size_t _len) {
    if (yyjson_unlikely(!_val)) return false;
    yyjson_mut_obj_add_func({
        val->tag = ((uint64_t)_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
        val->uni.str = _val;
    });
}

yyjson_api_inline bool yyjson_mut_obj_add_strcpy(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *obj,
                                                 const char *_key,
                                                 const char *_val) {
    if (yyjson_unlikely(!_val)) return false;
    yyjson_mut_obj_add_func({
        size_t _len = strlen(_val);
        val->uni.str = unsafe_yyjson_mut_strncpy(doc, _val, _len);
        if (yyjson_unlikely(!val->uni.str)) return false;
        val->tag = ((uint64_t)_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
    });
}

yyjson_api_inline bool yyjson_mut_obj_add_strncpy(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *obj,
                                                  const char *_key,
                                                  const char *_val,
                                                  size_t _len) {
    if (yyjson_unlikely(!_val)) return false;
    yyjson_mut_obj_add_func({
        val->uni.str = unsafe_yyjson_mut_strncpy(doc, _val, _len);
        if (yyjson_unlikely(!val->uni.str)) return false;
        val->tag = ((uint64_t)_len << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_add_arr(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *obj,
                                                         const char *_key) {
    yyjson_mut_val *key = yyjson_mut_str(doc, _key);
    yyjson_mut_val *val = yyjson_mut_arr(doc);
    return yyjson_mut_obj_add(obj, key, val) ? val : NULL;
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_add_obj(yyjson_mut_doc *doc,
                                                         yyjson_mut_val *obj,
                                                         const char *_key) {
    yyjson_mut_val *key = yyjson_mut_str(doc, _key);
    yyjson_mut_val *val = yyjson_mut_obj(doc);
    return yyjson_mut_obj_add(obj, key, val) ? val : NULL;
}

yyjson_api_inline bool yyjson_mut_obj_add_val(yyjson_mut_doc *doc,
                                              yyjson_mut_val *obj,
                                              const char *_key,
                                              yyjson_mut_val *_val) {
    if (yyjson_unlikely(!_val)) return false;
    yyjson_mut_obj_add_func({
        val = _val;
    });
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_str(yyjson_mut_val *obj,
                                                            const char *key) {
    return yyjson_mut_obj_remove_strn(obj, key, key ? strlen(key) : 0);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_obj_remove_strn(
    yyjson_mut_val *obj, const char *_key, size_t _len) {
    if (yyjson_likely(yyjson_mut_is_obj(obj) && _key)) {
        yyjson_mut_val *key;
        yyjson_mut_obj_iter iter;
        yyjson_mut_val *val_removed = NULL;
        yyjson_mut_obj_iter_init(obj, &iter);
        while ((key = yyjson_mut_obj_iter_next(&iter)) != NULL) {
            if (unsafe_yyjson_equals_strn(key, _key, _len)) {
                if (!val_removed) val_removed = key->next;
                yyjson_mut_obj_iter_remove(&iter);
            }
        }
        return val_removed;
    }
    return NULL;
}

yyjson_api_inline bool yyjson_mut_obj_rename_key(yyjson_mut_doc *doc,
                                                 yyjson_mut_val *obj,
                                                 const char *key,
                                                 const char *new_key) {
    if (!key || !new_key) return false;
    return yyjson_mut_obj_rename_keyn(doc, obj, key, strlen(key),
                                      new_key, strlen(new_key));
}

yyjson_api_inline bool yyjson_mut_obj_rename_keyn(yyjson_mut_doc *doc,
                                                  yyjson_mut_val *obj,
                                                  const char *key,
                                                  size_t len,
                                                  const char *new_key,
                                                  size_t new_len) {
    char *cpy_key = NULL;
    yyjson_mut_val *old_key;
    yyjson_mut_obj_iter iter;
    if (!doc || !obj || !key || !new_key) return false;
    yyjson_mut_obj_iter_init(obj, &iter);
    while ((old_key = yyjson_mut_obj_iter_next(&iter))) {
        if (unsafe_yyjson_equals_strn((void *)old_key, key, len)) {
            if (!cpy_key) {
                cpy_key = unsafe_yyjson_mut_strncpy(doc, new_key, new_len);
                if (!cpy_key) return false;
            }
            yyjson_mut_set_strn(old_key, cpy_key, new_len);
        }
    }
    return cpy_key != NULL;
}



#if !defined(YYJSON_DISABLE_UTILS) || !YYJSON_DISABLE_UTILS

/*==============================================================================
 * JSON Pointer API (Implementation)
 *============================================================================*/

#define yyjson_ptr_set_err(_code, _msg) do { \
    if (err) { \
        err->code = YYJSON_PTR_ERR_##_code; \
        err->msg = _msg; \
        err->pos = 0; \
    } \
} while(false)

/* require: val != NULL, *ptr == '/', len > 0 */
yyjson_api yyjson_val *unsafe_yyjson_ptr_getx(yyjson_val *val,
                                              const char *ptr, size_t len,
                                              yyjson_ptr_err *err);

/* require: val != NULL, *ptr == '/', len > 0 */
yyjson_api yyjson_mut_val *unsafe_yyjson_mut_ptr_getx(yyjson_mut_val *val,
                                                      const char *ptr,
                                                      size_t len,
                                                      yyjson_ptr_ctx *ctx,
                                                      yyjson_ptr_err *err);

/* require: val/new_val/doc != NULL, *ptr == '/', len > 0 */
yyjson_api bool unsafe_yyjson_mut_ptr_putx(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc,
                                           bool create_parent, bool insert_new,
                                           yyjson_ptr_ctx *ctx,
                                           yyjson_ptr_err *err);

/* require: val/err != NULL, *ptr == '/', len > 0 */
yyjson_api yyjson_mut_val *unsafe_yyjson_mut_ptr_replacex(
    yyjson_mut_val *val, const char *ptr, size_t len, yyjson_mut_val *new_val,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err);

/* require: val/err != NULL, *ptr == '/', len > 0 */
yyjson_api yyjson_mut_val *unsafe_yyjson_mut_ptr_removex(yyjson_mut_val *val,
                                                         const char *ptr,
                                                         size_t len,
                                                         yyjson_ptr_ctx *ctx,
                                                         yyjson_ptr_err *err);

yyjson_api_inline yyjson_val *yyjson_doc_ptr_get(yyjson_doc *doc,
                                                 const char *ptr) {
    if (yyjson_unlikely(!ptr)) return NULL;
    return yyjson_doc_ptr_getn(doc, ptr, strlen(ptr));
}

yyjson_api_inline yyjson_val *yyjson_doc_ptr_getn(yyjson_doc *doc,
                                                  const char *ptr, size_t len) {
    return yyjson_doc_ptr_getx(doc, ptr, len, NULL);
}

yyjson_api_inline yyjson_val *yyjson_doc_ptr_getx(yyjson_doc *doc,
                                                  const char *ptr, size_t len,
                                                  yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (yyjson_unlikely(!doc || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(!doc->root)) {
        yyjson_ptr_set_err(NULL_ROOT, "document's root is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        return doc->root;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_ptr_getx(doc->root, ptr, len, err);
}

yyjson_api_inline yyjson_val *yyjson_ptr_get(yyjson_val *val,
                                             const char *ptr) {
    if (yyjson_unlikely(!ptr)) return NULL;
    return yyjson_ptr_getn(val, ptr, strlen(ptr));
}

yyjson_api_inline yyjson_val *yyjson_ptr_getn(yyjson_val *val,
                                              const char *ptr, size_t len) {
    return yyjson_ptr_getx(val, ptr, len, NULL);
}

yyjson_api_inline yyjson_val *yyjson_ptr_getx(yyjson_val *val,
                                              const char *ptr, size_t len,
                                              yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (yyjson_unlikely(!val || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        return val;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_ptr_getx(val, ptr, len, err);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_get(yyjson_mut_doc *doc,
                                                         const char *ptr) {
    if (!ptr) return NULL;
    return yyjson_mut_doc_ptr_getn(doc, ptr, strlen(ptr));
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_getn(yyjson_mut_doc *doc,
                                                          const char *ptr,
                                                          size_t len) {
    return yyjson_mut_doc_ptr_getx(doc, ptr, len, NULL, NULL);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_getx(yyjson_mut_doc *doc,
                                                          const char *ptr,
                                                          size_t len,
                                                          yyjson_ptr_ctx *ctx,
                                                          yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!doc || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(!doc->root)) {
        yyjson_ptr_set_err(NULL_ROOT, "document's root is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        return doc->root;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_mut_ptr_getx(doc->root, ptr, len, ctx, err);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_get(yyjson_mut_val *val,
                                                     const char *ptr) {
    if (!ptr) return NULL;
    return yyjson_mut_ptr_getn(val, ptr, strlen(ptr));
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_getn(yyjson_mut_val *val,
                                                      const char *ptr,
                                                      size_t len) {
    return yyjson_mut_ptr_getx(val, ptr, len, NULL, NULL);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_getx(yyjson_mut_val *val,
                                                      const char *ptr,
                                                      size_t len,
                                                      yyjson_ptr_ctx *ctx,
                                                      yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!val || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        return val;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_mut_ptr_getx(val, ptr, len, ctx, err);
}

yyjson_api_inline bool yyjson_mut_doc_ptr_add(yyjson_mut_doc *doc,
                                              const char *ptr,
                                              yyjson_mut_val *new_val) {
    if (yyjson_unlikely(!ptr)) return false;
    return yyjson_mut_doc_ptr_addn(doc, ptr, strlen(ptr), new_val);
}

yyjson_api_inline bool yyjson_mut_doc_ptr_addn(yyjson_mut_doc *doc,
                                               const char *ptr,
                                               size_t len,
                                               yyjson_mut_val *new_val) {
    return yyjson_mut_doc_ptr_addx(doc, ptr, len, new_val, true, NULL, NULL);
}

yyjson_api_inline bool yyjson_mut_doc_ptr_addx(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val,
                                               bool create_parent,
                                               yyjson_ptr_ctx *ctx,
                                               yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!doc || !ptr || !new_val)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return false;
    }
    if (yyjson_unlikely(len == 0)) {
        if (doc->root) {
            yyjson_ptr_set_err(SET_ROOT, "cannot set document's root");
            return false;
        } else {
            doc->root = new_val;
            return true;
        }
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return false;
    }
    if (yyjson_unlikely(!doc->root && !create_parent)) {
        yyjson_ptr_set_err(NULL_ROOT, "document's root is NULL");
        return false;
    }
    if (yyjson_unlikely(!doc->root)) {
        yyjson_mut_val *root = yyjson_mut_obj(doc);
        if (yyjson_unlikely(!root)) {
            yyjson_ptr_set_err(MEMORY_ALLOCATION, "failed to create value");
            return false;
        }
        if (unsafe_yyjson_mut_ptr_putx(root, ptr, len, new_val, doc,
                                       create_parent, true, ctx, err)) {
            doc->root = root;
            return true;
        }
        return false;
    }
    return unsafe_yyjson_mut_ptr_putx(doc->root, ptr, len, new_val, doc,
                                      create_parent, true, ctx, err);
}

yyjson_api_inline bool yyjson_mut_ptr_add(yyjson_mut_val *val,
                                          const char *ptr,
                                          yyjson_mut_val *new_val,
                                          yyjson_mut_doc *doc) {
    if (yyjson_unlikely(!ptr)) return false;
    return yyjson_mut_ptr_addn(val, ptr, strlen(ptr), new_val, doc);
}

yyjson_api_inline bool yyjson_mut_ptr_addn(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc) {
    return yyjson_mut_ptr_addx(val, ptr, len, new_val, doc, true, NULL, NULL);
}

yyjson_api_inline bool yyjson_mut_ptr_addx(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc,
                                           bool create_parent,
                                           yyjson_ptr_ctx *ctx,
                                           yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!val || !ptr || !new_val || !doc)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return false;
    }
    if (yyjson_unlikely(len == 0)) {
        yyjson_ptr_set_err(SET_ROOT, "cannot set root");
        return false;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return false;
    }
    return unsafe_yyjson_mut_ptr_putx(val, ptr, len, new_val,
                                       doc, create_parent, true, ctx, err);
}

yyjson_api_inline bool yyjson_mut_doc_ptr_set(yyjson_mut_doc *doc,
                                              const char *ptr,
                                              yyjson_mut_val *new_val) {
    if (yyjson_unlikely(!ptr)) return false;
    return yyjson_mut_doc_ptr_setn(doc, ptr, strlen(ptr), new_val);
}

yyjson_api_inline bool yyjson_mut_doc_ptr_setn(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val) {
    return yyjson_mut_doc_ptr_setx(doc, ptr, len, new_val, true, NULL, NULL);
}

yyjson_api_inline bool yyjson_mut_doc_ptr_setx(yyjson_mut_doc *doc,
                                               const char *ptr, size_t len,
                                               yyjson_mut_val *new_val,
                                               bool create_parent,
                                               yyjson_ptr_ctx *ctx,
                                               yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!doc || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return false;
    }
    if (yyjson_unlikely(len == 0)) {
        if (ctx) ctx->old = doc->root;
        doc->root = new_val;
        return true;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return false;
    }
    if (!new_val) {
        if (!doc->root) {
            yyjson_ptr_set_err(RESOLVE, "JSON pointer cannot be resolved");
            return false;
        }
        return !!unsafe_yyjson_mut_ptr_removex(doc->root, ptr, len, ctx, err);
    }
    if (yyjson_unlikely(!doc->root && !create_parent)) {
        yyjson_ptr_set_err(NULL_ROOT, "document's root is NULL");
        return false;
    }
    if (yyjson_unlikely(!doc->root)) {
        yyjson_mut_val *root = yyjson_mut_obj(doc);
        if (yyjson_unlikely(!root)) {
            yyjson_ptr_set_err(MEMORY_ALLOCATION, "failed to create value");
            return false;
        }
        if (unsafe_yyjson_mut_ptr_putx(root, ptr, len, new_val, doc,
                                       create_parent, false, ctx, err)) {
            doc->root = root;
            return true;
        }
        return false;
    }
    return unsafe_yyjson_mut_ptr_putx(doc->root, ptr, len, new_val, doc,
                                      create_parent, false, ctx, err);
}

yyjson_api_inline bool yyjson_mut_ptr_set(yyjson_mut_val *val,
                                          const char *ptr,
                                          yyjson_mut_val *new_val,
                                          yyjson_mut_doc *doc) {
    if (yyjson_unlikely(!ptr)) return false;
    return yyjson_mut_ptr_setn(val, ptr, strlen(ptr), new_val, doc);
}

yyjson_api_inline bool yyjson_mut_ptr_setn(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc) {
    return yyjson_mut_ptr_setx(val, ptr, len, new_val, doc, true, NULL, NULL);
}

yyjson_api_inline bool yyjson_mut_ptr_setx(yyjson_mut_val *val,
                                           const char *ptr, size_t len,
                                           yyjson_mut_val *new_val,
                                           yyjson_mut_doc *doc,
                                           bool create_parent,
                                           yyjson_ptr_ctx *ctx,
                                           yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!val || !ptr || !doc)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return false;
    }
    if (yyjson_unlikely(len == 0)) {
        yyjson_ptr_set_err(SET_ROOT, "cannot set root");
        return false;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return false;
    }
    if (!new_val) {
        return !!unsafe_yyjson_mut_ptr_removex(val, ptr, len, ctx, err);
    }
    return unsafe_yyjson_mut_ptr_putx(val, ptr, len, new_val, doc,
                                      create_parent, false, ctx, err);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_replace(
    yyjson_mut_doc *doc, const char *ptr, yyjson_mut_val *new_val) {
    if (!ptr) return NULL;
    return yyjson_mut_doc_ptr_replacen(doc, ptr, strlen(ptr), new_val);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_replacen(
    yyjson_mut_doc *doc, const char *ptr, size_t len, yyjson_mut_val *new_val) {
    return yyjson_mut_doc_ptr_replacex(doc, ptr, len, new_val, NULL, NULL);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_replacex(
    yyjson_mut_doc *doc, const char *ptr, size_t len, yyjson_mut_val *new_val,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!doc || !ptr || !new_val)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        yyjson_mut_val *root = doc->root;
        if (yyjson_unlikely(!root)) {
            yyjson_ptr_set_err(RESOLVE, "JSON pointer cannot be resolved");
            return NULL;
        }
        if (ctx) ctx->old = root;
        doc->root = new_val;
        return root;
    }
    if (yyjson_unlikely(!doc->root)) {
        yyjson_ptr_set_err(NULL_ROOT, "document's root is NULL");
        return NULL;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_mut_ptr_replacex(doc->root, ptr, len, new_val,
                                          ctx, err);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_replace(
    yyjson_mut_val *val, const char *ptr, yyjson_mut_val *new_val) {
    if (!ptr) return NULL;
    return yyjson_mut_ptr_replacen(val, ptr, strlen(ptr), new_val);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_replacen(
    yyjson_mut_val *val, const char *ptr, size_t len, yyjson_mut_val *new_val) {
    return yyjson_mut_ptr_replacex(val, ptr, len, new_val, NULL, NULL);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_replacex(
    yyjson_mut_val *val, const char *ptr, size_t len, yyjson_mut_val *new_val,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!val || !ptr || !new_val)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        yyjson_ptr_set_err(SET_ROOT, "cannot set root");
        return NULL;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_mut_ptr_replacex(val, ptr, len, new_val, ctx, err);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_remove(
    yyjson_mut_doc *doc, const char *ptr) {
    if (!ptr) return NULL;
    return yyjson_mut_doc_ptr_removen(doc, ptr, strlen(ptr));
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_removen(
    yyjson_mut_doc *doc, const char *ptr, size_t len) {
    return yyjson_mut_doc_ptr_removex(doc, ptr, len, NULL, NULL);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_ptr_removex(
    yyjson_mut_doc *doc, const char *ptr, size_t len,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!doc || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(!doc->root)) {
        yyjson_ptr_set_err(NULL_ROOT, "document's root is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        yyjson_mut_val *root = doc->root;
        if (ctx) ctx->old = root;
        doc->root = NULL;
        return root;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_mut_ptr_removex(doc->root, ptr, len, ctx, err);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_remove(yyjson_mut_val *val,
                                                        const char *ptr) {
    if (!ptr) return NULL;
    return yyjson_mut_ptr_removen(val, ptr, strlen(ptr));
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_removen(yyjson_mut_val *val,
                                                         const char *ptr,
                                                         size_t len) {
    return yyjson_mut_ptr_removex(val, ptr, len, NULL, NULL);
}

yyjson_api_inline yyjson_mut_val *yyjson_mut_ptr_removex(yyjson_mut_val *val,
                                                         const char *ptr,
                                                         size_t len,
                                                         yyjson_ptr_ctx *ctx,
                                                         yyjson_ptr_err *err) {
    yyjson_ptr_set_err(NONE, NULL);
    if (ctx) memset(ctx, 0, sizeof(*ctx));

    if (yyjson_unlikely(!val || !ptr)) {
        yyjson_ptr_set_err(PARAMETER, "input parameter is NULL");
        return NULL;
    }
    if (yyjson_unlikely(len == 0)) {
        yyjson_ptr_set_err(SET_ROOT, "cannot set root");
        return NULL;
    }
    if (yyjson_unlikely(*ptr != '/')) {
        yyjson_ptr_set_err(SYNTAX, "no prefix '/'");
        return NULL;
    }
    return unsafe_yyjson_mut_ptr_removex(val, ptr, len, ctx, err);
}

yyjson_api_inline bool yyjson_ptr_ctx_append(yyjson_ptr_ctx *ctx,
                                             yyjson_mut_val *key,
                                             yyjson_mut_val *val) {
    yyjson_mut_val *ctn, *pre_key, *pre_val, *cur_key, *cur_val;
    if (!ctx || !ctx->ctn || !val) return false;
    ctn = ctx->ctn;

    if (yyjson_mut_is_obj(ctn)) {
        if (!key) return false;
        key->next = val;
        pre_key = ctx->pre;
        if (unsafe_yyjson_get_len(ctn) == 0) {
            val->next = key;
            ctn->uni.ptr = key;
            ctx->pre = key;
        } else if (!pre_key) {
            pre_key = (yyjson_mut_val *)ctn->uni.ptr;
            pre_val = pre_key->next;
            val->next = pre_val->next;
            pre_val->next = key;
            ctn->uni.ptr = key;
            ctx->pre = pre_key;
        } else {
            cur_key = pre_key->next->next;
            cur_val = cur_key->next;
            val->next = cur_val->next;
            cur_val->next = key;
            if (ctn->uni.ptr == cur_key) ctn->uni.ptr = key;
            ctx->pre = cur_key;
        }
    } else {
        pre_val = ctx->pre;
        if (unsafe_yyjson_get_len(ctn) == 0) {
            val->next = val;
            ctn->uni.ptr = val;
            ctx->pre = val;
        } else if (!pre_val) {
            pre_val = (yyjson_mut_val *)ctn->uni.ptr;
            val->next = pre_val->next;
            pre_val->next = val;
            ctn->uni.ptr = val;
            ctx->pre = pre_val;
        } else {
            cur_val = pre_val->next;
            val->next = cur_val->next;
            cur_val->next = val;
            if (ctn->uni.ptr == cur_val) ctn->uni.ptr = val;
            ctx->pre = cur_val;
        }
    }
    unsafe_yyjson_inc_len(ctn);
    return true;
}

yyjson_api_inline bool yyjson_ptr_ctx_replace(yyjson_ptr_ctx *ctx,
                                              yyjson_mut_val *val) {
    yyjson_mut_val *ctn, *pre_key, *cur_key, *pre_val, *cur_val;
    if (!ctx || !ctx->ctn || !ctx->pre || !val) return false;
    ctn = ctx->ctn;
    if (yyjson_mut_is_obj(ctn)) {
        pre_key = ctx->pre;
        pre_val = pre_key->next;
        cur_key = pre_val->next;
        cur_val = cur_key->next;
        /* replace current value */
        cur_key->next = val;
        val->next = cur_val->next;
        ctx->old = cur_val;
    } else {
        pre_val = ctx->pre;
        cur_val = pre_val->next;
        /* replace current value */
        if (pre_val != cur_val) {
            val->next = cur_val->next;
            pre_val->next = val;
            if (ctn->uni.ptr == cur_val) ctn->uni.ptr = val;
        } else {
            val->next = val;
            ctn->uni.ptr = val;
            ctx->pre = val;
        }
        ctx->old = cur_val;
    }
    return true;
}

yyjson_api_inline bool yyjson_ptr_ctx_remove(yyjson_ptr_ctx *ctx) {
    yyjson_mut_val *ctn, *pre_key, *pre_val, *cur_key, *cur_val;
    size_t len;
    if (!ctx || !ctx->ctn || !ctx->pre) return false;
    ctn = ctx->ctn;
    if (yyjson_mut_is_obj(ctn)) {
        pre_key = ctx->pre;
        pre_val = pre_key->next;
        cur_key = pre_val->next;
        cur_val = cur_key->next;
        /* remove current key-value */
        pre_val->next = cur_val->next;
        if (ctn->uni.ptr == cur_key) ctn->uni.ptr = pre_key;
        ctx->pre = NULL;
        ctx->old = cur_val;
    } else {
        pre_val = ctx->pre;
        cur_val = pre_val->next;
        /* remove current key-value */
        pre_val->next = cur_val->next;
        if (ctn->uni.ptr == cur_val) ctn->uni.ptr = pre_val;
        ctx->pre = NULL;
        ctx->old = cur_val;
    }
    len = unsafe_yyjson_get_len(ctn) - 1;
    if (len == 0) ctn->uni.ptr = NULL;
    unsafe_yyjson_set_len(ctn, len);
    return true;
}

#undef yyjson_ptr_set_err



/*==============================================================================
 * JSON Value at Pointer API (Implementation)
 *============================================================================*/

/**
 Set provided `value` if the JSON Pointer (RFC 6901) exists and is type bool.
 Returns true if value at `ptr` exists and is the correct type, otherwise false.
 */
yyjson_api_inline bool yyjson_ptr_get_bool(
    yyjson_val *root, const char *ptr, bool *value) {
    yyjson_val *val = yyjson_ptr_get(root, ptr);
    if (value && yyjson_is_bool(val)) {
        *value = unsafe_yyjson_get_bool(val);
        return true;
    } else {
        return false;
    }
}

/**
 Set provided `value` if the JSON Pointer (RFC 6901) exists and is an integer
 that fits in `uint64_t`. Returns true if successful, otherwise false.
 */
yyjson_api_inline bool yyjson_ptr_get_uint(
    yyjson_val *root, const char *ptr, uint64_t *value) {
    yyjson_val *val = yyjson_ptr_get(root, ptr);
    if (value && val) {
        uint64_t ret = val->uni.u64;
        if (unsafe_yyjson_is_uint(val) ||
            (unsafe_yyjson_is_sint(val) && !(ret >> 63))) {
            *value = ret;
            return true;
        }
    }
    return false;
}

/**
 Set provided `value` if the JSON Pointer (RFC 6901) exists and is an integer
 that fits in `int64_t`. Returns true if successful, otherwise false.
 */
yyjson_api_inline bool yyjson_ptr_get_sint(
    yyjson_val *root, const char *ptr, int64_t *value) {
    yyjson_val *val = yyjson_ptr_get(root, ptr);
    if (value && val) {
        int64_t ret = val->uni.i64;
        if (unsafe_yyjson_is_sint(val) ||
            (unsafe_yyjson_is_uint(val) && ret >= 0)) {
            *value = ret;
            return true;
        }
    }
    return false;
}

/**
 Set provided `value` if the JSON Pointer (RFC 6901) exists and is type real.
 Returns true if value at `ptr` exists and is the correct type, otherwise false.
 */
yyjson_api_inline bool yyjson_ptr_get_real(
    yyjson_val *root, const char *ptr, double *value) {
    yyjson_val *val = yyjson_ptr_get(root, ptr);
    if (value && yyjson_is_real(val)) {
        *value = unsafe_yyjson_get_real(val);
        return true;
    } else {
        return false;
    }
}

/**
 Set provided `value` if the JSON Pointer (RFC 6901) exists and is type sint,
 uint or real.
 Returns true if value at `ptr` exists and is the correct type, otherwise false.
 */
yyjson_api_inline bool yyjson_ptr_get_num(
    yyjson_val *root, const char *ptr, double *value) {
    yyjson_val *val = yyjson_ptr_get(root, ptr);
    if (value && yyjson_is_num(val)) {
        *value = unsafe_yyjson_get_num(val);
        return true;
    } else {
        return false;
    }
}

/**
 Set provided `value` if the JSON Pointer (RFC 6901) exists and is type string.
 Returns true if value at `ptr` exists and is the correct type, otherwise false.
 */
yyjson_api_inline bool yyjson_ptr_get_str(
    yyjson_val *root, const char *ptr, const char **value) {
    yyjson_val *val = yyjson_ptr_get(root, ptr);
    if (value && yyjson_is_str(val)) {
        *value = unsafe_yyjson_get_str(val);
        return true;
    } else {
        return false;
    }
}



/*==============================================================================
 * Deprecated
 *============================================================================*/

/** @deprecated renamed to `yyjson_doc_ptr_get` */
yyjson_deprecated("renamed to yyjson_doc_ptr_get")
yyjson_api_inline yyjson_val *yyjson_doc_get_pointer(yyjson_doc *doc,
                                                     const char *ptr) {
    return yyjson_doc_ptr_get(doc, ptr);
}

/** @deprecated renamed to `yyjson_doc_ptr_getn` */
yyjson_deprecated("renamed to yyjson_doc_ptr_getn")
yyjson_api_inline yyjson_val *yyjson_doc_get_pointern(yyjson_doc *doc,
                                                      const char *ptr,
                                                      size_t len) {
    return yyjson_doc_ptr_getn(doc, ptr, len);
}

/** @deprecated renamed to `yyjson_mut_doc_ptr_get` */
yyjson_deprecated("renamed to yyjson_mut_doc_ptr_get")
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_get_pointer(
    yyjson_mut_doc *doc, const char *ptr) {
    return yyjson_mut_doc_ptr_get(doc, ptr);
}

/** @deprecated renamed to `yyjson_mut_doc_ptr_getn` */
yyjson_deprecated("renamed to yyjson_mut_doc_ptr_getn")
yyjson_api_inline yyjson_mut_val *yyjson_mut_doc_get_pointern(
    yyjson_mut_doc *doc, const char *ptr, size_t len) {
    return yyjson_mut_doc_ptr_getn(doc, ptr, len);
}

/** @deprecated renamed to `yyjson_ptr_get` */
yyjson_deprecated("renamed to yyjson_ptr_get")
yyjson_api_inline yyjson_val *yyjson_get_pointer(yyjson_val *val,
                                                 const char *ptr) {
    return yyjson_ptr_get(val, ptr);
}

/** @deprecated renamed to `yyjson_ptr_getn` */
yyjson_deprecated("renamed to yyjson_ptr_getn")
yyjson_api_inline yyjson_val *yyjson_get_pointern(yyjson_val *val,
                                                  const char *ptr,
                                                  size_t len) {
    return yyjson_ptr_getn(val, ptr, len);
}

/** @deprecated renamed to `yyjson_mut_ptr_get` */
yyjson_deprecated("renamed to yyjson_mut_ptr_get")
yyjson_api_inline yyjson_mut_val *yyjson_mut_get_pointer(yyjson_mut_val *val,
                                                         const char *ptr) {
    return yyjson_mut_ptr_get(val, ptr);
}

/** @deprecated renamed to `yyjson_mut_ptr_getn` */
yyjson_deprecated("renamed to yyjson_mut_ptr_getn")
yyjson_api_inline yyjson_mut_val *yyjson_mut_get_pointern(yyjson_mut_val *val,
                                                          const char *ptr,
                                                          size_t len) {
    return yyjson_mut_ptr_getn(val, ptr, len);
}

/** @deprecated renamed to `yyjson_mut_ptr_getn` */
yyjson_deprecated("renamed to unsafe_yyjson_ptr_getn")
yyjson_api_inline yyjson_val *unsafe_yyjson_get_pointer(yyjson_val *val,
                                                        const char *ptr,
                                                        size_t len) {
    yyjson_ptr_err err;
    return unsafe_yyjson_ptr_getx(val, ptr, len, &err);
}

/** @deprecated renamed to `unsafe_yyjson_mut_ptr_getx` */
yyjson_deprecated("renamed to unsafe_yyjson_mut_ptr_getx")
yyjson_api_inline yyjson_mut_val *unsafe_yyjson_mut_get_pointer(
    yyjson_mut_val *val, const char *ptr, size_t len) {
    yyjson_ptr_err err;
    return unsafe_yyjson_mut_ptr_getx(val, ptr, len, NULL, &err);
}

#endif /* YYJSON_DISABLE_UTILS */



/*==============================================================================
 * Compiler Hint End
 *============================================================================*/

#if defined(__clang__)
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#       pragma GCC diagnostic pop
#   endif
#elif defined(_MSC_VER)
#   pragma warning(pop)
#endif /* warning suppress end */

#ifdef __cplusplus
}
#endif /* extern "C" end */

#endif /* YYJSON_H */
