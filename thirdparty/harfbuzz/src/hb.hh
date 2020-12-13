/*
 * Copyright © 2007,2008,2009  Red Hat, Inc.
 * Copyright © 2011,2012  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_HH
#define HB_HH


#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC
#ifdef _MSC_VER
#pragma warning( disable: 4068 ) /* Unknown pragma */
#endif
#if defined(__GNUC__) || defined(__clang__)
/* Rules:
 *
 * - All pragmas are declared GCC even if they are clang ones.  Otherwise GCC
 *   nags, even though we instruct it to ignore -Wunknown-pragmas. ¯\_(ツ)_/¯
 *
 * - Within each category, keep sorted.
 *
 * - Warnings whose scope can be expanded in future compiler versions shall
 *   be declared as "warning".  Otherwise, either ignored or error.
 */

/* Setup.  Don't sort order within this category. */
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_WARNING
#pragma GCC diagnostic warning "-Wall"
#pragma GCC diagnostic warning "-Wextra"
#endif
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_IGNORED
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#endif
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_WARNING
//#pragma GCC diagnostic warning "-Weverything"
#endif

/* Error.  Should never happen. */
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_ERROR
#pragma GCC diagnostic error   "-Wc++11-narrowing"
#pragma GCC diagnostic error   "-Wcast-align"
#pragma GCC diagnostic error   "-Wcast-function-type"
#pragma GCC diagnostic error   "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic error   "-Wembedded-directive"
#pragma GCC diagnostic error   "-Wextra-semi-stmt"
#pragma GCC diagnostic error   "-Wformat-security"
#pragma GCC diagnostic error   "-Wimplicit-function-declaration"
#pragma GCC diagnostic error   "-Winit-self"
#pragma GCC diagnostic error   "-Winjected-class-name"
#pragma GCC diagnostic error   "-Wmissing-braces"
#pragma GCC diagnostic error   "-Wmissing-declarations"
#pragma GCC diagnostic error   "-Wmissing-prototypes"
#pragma GCC diagnostic error   "-Wnested-externs"
#pragma GCC diagnostic error   "-Wold-style-definition"
#pragma GCC diagnostic error   "-Wpointer-arith"
#pragma GCC diagnostic error   "-Wredundant-decls"
#pragma GCC diagnostic error   "-Wreorder"
#pragma GCC diagnostic error   "-Wsign-compare"
#pragma GCC diagnostic error   "-Wstrict-prototypes"
#pragma GCC diagnostic error   "-Wstring-conversion"
#pragma GCC diagnostic error   "-Wswitch-enum"
#pragma GCC diagnostic error   "-Wtautological-overlap-compare"
#pragma GCC diagnostic error   "-Wunneeded-internal-declaration"
#pragma GCC diagnostic error   "-Wunused"
#pragma GCC diagnostic error   "-Wunused-local-typedefs"
#pragma GCC diagnostic error   "-Wunused-value"
#pragma GCC diagnostic error   "-Wunused-variable"
#pragma GCC diagnostic error   "-Wvla"
#pragma GCC diagnostic error   "-Wwrite-strings"
#endif

/* Warning.  To be investigated if happens. */
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_WARNING
#pragma GCC diagnostic warning "-Wbuiltin-macro-redefined"
#pragma GCC diagnostic warning "-Wdeprecated"
#pragma GCC diagnostic warning "-Wdeprecated-declarations"
#pragma GCC diagnostic warning "-Wdisabled-optimization"
#pragma GCC diagnostic warning "-Wdouble-promotion"
#pragma GCC diagnostic warning "-Wformat=2"
#pragma GCC diagnostic warning "-Wignored-pragma-optimize"
#pragma GCC diagnostic warning "-Wlogical-op"
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#pragma GCC diagnostic warning "-Wmissing-format-attribute"
#pragma GCC diagnostic warning "-Wundef"
#pragma GCC diagnostic warning "-Wunused-but-set-variable"
#endif

/* Ignored currently, but should be fixed at some point. */
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_IGNORED
#pragma GCC diagnostic ignored "-Wconversion"			// TODO fix
#pragma GCC diagnostic ignored "-Wformat-signedness"		// TODO fix
#pragma GCC diagnostic ignored "-Wshadow"			// TODO fix
#pragma GCC diagnostic ignored "-Wunsafe-loop-optimizations"	// TODO fix
#pragma GCC diagnostic ignored "-Wunused-parameter"		// TODO fix
#endif

/* Ignored intentionally. */
#ifndef HB_NO_PRAGMA_GCC_DIAGNOSTIC_IGNORED
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-zero-length"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpacked" // Erratic impl in clang
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wc++11-compat" // only gcc raises it
#endif

#endif
#endif


#include "hb-config.hh"


/*
 * Following added based on what AC_USE_SYSTEM_EXTENSIONS adds to
 * config.h.in.  Copied here for the convenience of those embedding
 * HarfBuzz and not using our build system.
 */
/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif

#if defined (_MSC_VER) && defined (HB_DLL_EXPORT)
#define HB_EXTERN __declspec (dllexport) extern
#endif

#include "hb.h"
#define HB_H_IN
#include "hb-ot.h"
#define HB_OT_H_IN
#include "hb-aat.h"
#define HB_AAT_H_IN

#include <limits.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>

#if (defined(_MSC_VER) && _MSC_VER >= 1500) || defined(__MINGW32__)
#ifdef __MINGW32_VERSION
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#else
#include <intrin.h>
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#include <winapifamily.h>
#endif

#define HB_PASTE1(a,b) a##b
#define HB_PASTE(a,b) HB_PASTE1(a,b)


/* Compile-time custom allocator support. */

#if !defined(HB_CUSTOM_MALLOC) \
  && defined(hb_malloc_impl) \
  && defined(hb_calloc_impl) \
  && defined(hb_realloc_impl) \
  && defined(hb_free_impl)
#define HB_CUSTOM_MALLOC
#endif

#ifdef HB_CUSTOM_MALLOC
extern "C" void* hb_malloc_impl(size_t size);
extern "C" void* hb_calloc_impl(size_t nmemb, size_t size);
extern "C" void* hb_realloc_impl(void *ptr, size_t size);
extern "C" void  hb_free_impl(void *ptr);
#define malloc hb_malloc_impl
#define calloc hb_calloc_impl
#define realloc hb_realloc_impl
#define free hb_free_impl
#endif


/*
 * Compiler attributes
 */

#if (defined(__GNUC__) || defined(__clang__)) && defined(__OPTIMIZE__)
#define likely(expr) (__builtin_expect (!!(expr), 1))
#define unlikely(expr) (__builtin_expect (!!(expr), 0))
#else
#define likely(expr) (expr)
#define unlikely(expr) (expr)
#endif

#if !defined(__GNUC__) && !defined(__clang__)
#undef __attribute__
#define __attribute__(x)
#endif

#if defined(__GNUC__) && (__GNUC__ >= 3)
#define HB_PURE_FUNC	__attribute__((pure))
#define HB_CONST_FUNC	__attribute__((const))
#define HB_PRINTF_FUNC(format_idx, arg_idx) __attribute__((__format__ (__printf__, format_idx, arg_idx)))
#else
#define HB_PURE_FUNC
#define HB_CONST_FUNC
#define HB_PRINTF_FUNC(format_idx, arg_idx)
#endif
#if defined(__GNUC__) && (__GNUC__ >= 4) || (__clang__)
#define HB_UNUSED	__attribute__((unused))
#elif defined(_MSC_VER) /* https://github.com/harfbuzz/harfbuzz/issues/635 */
#define HB_UNUSED __pragma(warning(suppress: 4100 4101))
#else
#define HB_UNUSED
#endif

#ifndef HB_INTERNAL
# if !defined(HB_NO_VISIBILITY) && !defined(__MINGW32__) && !defined(__CYGWIN__) && !defined(_MSC_VER) && !defined(__SUNPRO_CC)
#  define HB_INTERNAL __attribute__((__visibility__("hidden")))
# elif defined(__MINGW32__)
   /* We use -export-symbols on mingw32, since it does not support visibility attributes. */
#  define HB_INTERNAL
# elif defined (_MSC_VER) && defined (HB_DLL_EXPORT)
   /* We do not try to export internal symbols on Visual Studio */
#  define HB_INTERNAL
#else
#  define HB_INTERNAL
#  define HB_NO_VISIBILITY 1
# endif
#endif

/* https://github.com/harfbuzz/harfbuzz/issues/1651 */
#if defined(__clang__) && __clang_major__ < 10
#define static_const static
#else
#define static_const static const
#endif

#if defined(__GNUC__) && (__GNUC__ >= 3)
#define HB_FUNC __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define HB_FUNC __FUNCSIG__
#else
#define HB_FUNC __func__
#endif

#if defined(__SUNPRO_CC) && (__SUNPRO_CC < 0x5140)
/* https://github.com/harfbuzz/harfbuzz/issues/630 */
#define __restrict
#endif

/*
 * Borrowed from https://bugzilla.mozilla.org/show_bug.cgi?id=1215411
 * HB_FALLTHROUGH is an annotation to suppress compiler warnings about switch
 * cases that fall through without a break or return statement. HB_FALLTHROUGH
 * is only needed on cases that have code:
 *
 * switch (foo) {
 *   case 1: // These cases have no code. No fallthrough annotations are needed.
 *   case 2:
 *   case 3:
 *     foo = 4; // This case has code, so a fallthrough annotation is needed:
 *     HB_FALLTHROUGH;
 *   default:
 *     return foo;
 * }
 */
#if defined(__clang__) && __cplusplus >= 201103L
   /* clang's fallthrough annotations are only available starting in C++11. */
#  define HB_FALLTHROUGH [[clang::fallthrough]]
#elif defined(__GNUC__) && (__GNUC__ >= 7)
   /* GNU fallthrough attribute is available from GCC7 */
#  define HB_FALLTHROUGH __attribute__((fallthrough))
#elif defined(_MSC_VER)
   /*
    * MSVC's __fallthrough annotations are checked by /analyze (Code Analysis):
    * https://msdn.microsoft.com/en-us/library/ms235402%28VS.80%29.aspx
    */
#  include <sal.h>
#  define HB_FALLTHROUGH __fallthrough
#else
#  define HB_FALLTHROUGH /* FALLTHROUGH */
#endif

/* A tag to enforce use of return value for a function */
#if __cplusplus >= 201703L
#  define HB_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#  define HB_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
#  define HB_NODISCARD _Check_return_
#else
#  define HB_NODISCARD
#endif
#define hb_success_t HB_NODISCARD bool

/* https://github.com/harfbuzz/harfbuzz/issues/1852 */
#if defined(__clang__) && !(defined(_AIX) && (defined(__IBMCPP__) || defined(__ibmxl__)))
/* Disable certain sanitizer errors. */
/* https://github.com/harfbuzz/harfbuzz/issues/1247 */
#define HB_NO_SANITIZE_SIGNED_INTEGER_OVERFLOW __attribute__((no_sanitize("signed-integer-overflow")))
#else
#define HB_NO_SANITIZE_SIGNED_INTEGER_OVERFLOW
#endif


#ifdef _WIN32
   /* We need Windows Vista for both Uniscribe backend and for
    * MemoryBarrier.  We don't support compiling on Windows XP,
    * though we run on it fine. */
#  if defined(_WIN32_WINNT) && _WIN32_WINNT < 0x0600
#    undef _WIN32_WINNT
#  endif
#  ifndef _WIN32_WINNT
#    if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#      define _WIN32_WINNT 0x0600
#    endif
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN 1
#  endif
#  ifndef STRICT
#    define STRICT 1
#  endif

#  if defined(_WIN32_WCE)
     /* Some things not defined on Windows CE. */
#    define vsnprintf _vsnprintf
#    ifndef HB_NO_GETENV
#      define HB_NO_GETENV
#    endif
#    if _WIN32_WCE < 0x800
#      define HB_NO_SETLOCALE
#      define HB_NO_ERRNO
#    endif
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#    ifndef HB_NO_GETENV
#      define HB_NO_GETENV
#    endif
#  endif
#  if defined(_MSC_VER) && _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#endif

#ifdef HB_NO_GETENV
#define getenv(Name) nullptr
#endif

#ifndef HB_NO_ERRNO
#  include <errno.h>
#else
static int HB_UNUSED _hb_errno = 0;
#  undef errno
#  define errno _hb_errno
#endif

#if defined(HAVE_ATEXIT) && !defined(HB_USE_ATEXIT)
/* atexit() is only safe to be called from shared libraries on certain
 * platforms.  Whitelist.
 * https://bugs.freedesktop.org/show_bug.cgi?id=82246 */
#  if defined(__linux) && defined(__GLIBC_PREREQ)
#    if __GLIBC_PREREQ(2,3)
/* From atexit() manpage, it's safe with glibc 2.2.3 on Linux. */
#      define HB_USE_ATEXIT 1
#    endif
#  elif defined(_MSC_VER) || defined(__MINGW32__)
/* For MSVC:
 * https://msdn.microsoft.com/en-us/library/tze57ck3.aspx
 * https://msdn.microsoft.com/en-us/library/zk17ww08.aspx
 * mingw32 headers say atexit is safe to use in shared libraries.
 */
#    define HB_USE_ATEXIT 1
#  elif defined(__ANDROID__)
/* This is available since Android NKD r8 or r8b:
 * https://issuetracker.google.com/code/p/android/issues/detail?id=6455
 */
#    define HB_USE_ATEXIT 1
#  elif defined(__APPLE__)
/* For macOS and related platforms, the atexit man page indicates
 * that it will be invoked when the library is unloaded, not only
 * at application exit.
 */
#    define HB_USE_ATEXIT 1
#  endif
#endif
#ifdef HB_NO_ATEXIT
#  undef HB_USE_ATEXIT
#endif
#ifndef HB_USE_ATEXIT
#  define HB_USE_ATEXIT 0
#endif

#define HB_STMT_START do
#define HB_STMT_END   while (0)

/* Static-assert as expression. */
template <unsigned int cond> class hb_assert_constant_t;
template <> class hb_assert_constant_t<1> {};
#define ASSERT_STATIC_EXPR_ZERO(_cond) (0 * (unsigned int) sizeof (hb_assert_constant_t<_cond>))

/* Lets assert int types.  Saves trouble down the road. */
static_assert ((sizeof (int8_t) == 1), "");
static_assert ((sizeof (uint8_t) == 1), "");
static_assert ((sizeof (int16_t) == 2), "");
static_assert ((sizeof (uint16_t) == 2), "");
static_assert ((sizeof (int32_t) == 4), "");
static_assert ((sizeof (uint32_t) == 4), "");
static_assert ((sizeof (int64_t) == 8), "");
static_assert ((sizeof (uint64_t) == 8), "");
static_assert ((sizeof (hb_codepoint_t) == 4), "");
static_assert ((sizeof (hb_position_t) == 4), "");
static_assert ((sizeof (hb_mask_t) == 4), "");
static_assert ((sizeof (hb_var_int_t) == 4), "");

#define HB_DELETE_COPY_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete
#define HB_DELETE_CREATE_COPY_ASSIGN(TypeName) \
  TypeName() = delete; \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete


/* Flags */

/* Enable bitwise ops on enums marked as flags_t */
/* To my surprise, looks like the function resolver is happy to silently cast
 * one enum to another...  So this doesn't provide the type-checking that I
 * originally had in mind... :(.
 *
 * For MSVC warnings, see: https://github.com/harfbuzz/harfbuzz/pull/163
 */
#ifdef _MSC_VER
# pragma warning(disable:4200)
# pragma warning(disable:4800)
#endif
#define HB_MARK_AS_FLAG_T(T) \
	extern "C++" { \
	  static inline T operator | (T l, T r) { return T ((unsigned) l | (unsigned) r); } \
	  static inline T operator & (T l, T r) { return T ((unsigned) l & (unsigned) r); } \
	  static inline T operator ^ (T l, T r) { return T ((unsigned) l ^ (unsigned) r); } \
	  static inline T operator ~ (T r) { return T (~(unsigned int) r); } \
	  static inline T& operator |= (T &l, T r) { l = l | r; return l; } \
	  static inline T& operator &= (T& l, T r) { l = l & r; return l; } \
	  static inline T& operator ^= (T& l, T r) { l = l ^ r; return l; } \
	} \
	static_assert (true, "")

/* Useful for set-operations on small enums.
 * For example, for testing "x ∈ {x1, x2, x3}" use:
 * (FLAG_UNSAFE(x) & (FLAG(x1) | FLAG(x2) | FLAG(x3)))
 */
#define FLAG(x) (ASSERT_STATIC_EXPR_ZERO ((unsigned)(x) < 32) + (((uint32_t) 1U) << (unsigned)(x)))
#define FLAG_UNSAFE(x) ((unsigned)(x) < 32 ? (((uint32_t) 1U) << (unsigned)(x)) : 0)
#define FLAG_RANGE(x,y) (ASSERT_STATIC_EXPR_ZERO ((x) < (y)) + FLAG(y+1) - FLAG(x))
#define FLAG64(x) (ASSERT_STATIC_EXPR_ZERO ((unsigned)(x) < 64) + (((uint64_t) 1ULL) << (unsigned)(x)))
#define FLAG64_UNSAFE(x) ((unsigned)(x) < 64 ? (((uint64_t) 1ULL) << (unsigned)(x)) : 0)


/* Size signifying variable-sized array */
#ifndef HB_VAR_ARRAY
#define HB_VAR_ARRAY 1
#endif

static inline float
_hb_roundf (float x) { return floorf (x + .5f); }
#define roundf(x) _hb_roundf(x)

/* Endian swap, used in Windows related backends */
static inline uint16_t hb_uint16_swap (const uint16_t v)
{ return (v >> 8) | (v << 8); }
static inline uint32_t hb_uint32_swap (const uint32_t v)
{ return (hb_uint16_swap (v) << 16) | hb_uint16_swap (v >> 16); }

/*
 * Big-endian integers.  Here because fundamental.
 */

template <typename Type, int Bytes> struct BEInt;

template <typename Type>
struct BEInt<Type, 1>
{
  public:
  BEInt<Type, 1>& operator = (Type V)
  {
    v = V;
    return *this;
  }
  operator Type () const { return v; }
  private: uint8_t v;
};
template <typename Type>
struct BEInt<Type, 2>
{
  public:
  BEInt<Type, 2>& operator = (Type V)
  {
    v[0] = (V >>  8) & 0xFF;
    v[1] = (V      ) & 0xFF;
    return *this;
  }
  operator Type () const
  {
#if ((defined(__GNUC__) && __GNUC__ >= 5) || defined(__clang__)) && \
    defined(__BYTE_ORDER) && \
    (__BYTE_ORDER == __LITTLE_ENDIAN || __BYTE_ORDER == __BIG_ENDIAN)
    /* Spoon-feed the compiler a big-endian integer with alignment 1.
     * https://github.com/harfbuzz/harfbuzz/pull/1398 */
    struct __attribute__((packed)) packed_uint16_t { uint16_t v; };
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return __builtin_bswap16 (((packed_uint16_t *) this)->v);
#else /* __BYTE_ORDER == __BIG_ENDIAN */
    return ((packed_uint16_t *) this)->v;
#endif
#endif
    return (v[0] <<  8)
	 + (v[1]      );
  }
  private: uint8_t v[2];
};
template <typename Type>
struct BEInt<Type, 3>
{
  public:
  BEInt<Type, 3>& operator = (Type V)
  {
    v[0] = (V >> 16) & 0xFF;
    v[1] = (V >>  8) & 0xFF;
    v[2] = (V      ) & 0xFF;
    return *this;
  }
  operator Type () const
  {
    return (v[0] << 16)
	 + (v[1] <<  8)
	 + (v[2]      );
  }
  private: uint8_t v[3];
};
template <typename Type>
struct BEInt<Type, 4>
{
  public:
  BEInt<Type, 4>& operator = (Type V)
  {
    v[0] = (V >> 24) & 0xFF;
    v[1] = (V >> 16) & 0xFF;
    v[2] = (V >>  8) & 0xFF;
    v[3] = (V      ) & 0xFF;
    return *this;
  }
  operator Type () const
  {
    return (v[0] << 24)
	 + (v[1] << 16)
	 + (v[2] <<  8)
	 + (v[3]      );
  }
  private: uint8_t v[4];
};


/*
 * For lack of a better place, put Zawgyi script hack here.
 * https://github.com/harfbuzz/harfbuzz/issues/1162
 */

#define HB_SCRIPT_MYANMAR_ZAWGYI	((hb_script_t) HB_TAG ('Q','a','a','g'))


/* Headers we include for everyone.  Keep topologically sorted by dependency.
 * They express dependency amongst themselves, but no other file should include
 * them directly.*/
#include "hb-meta.hh"
#include "hb-mutex.hh"
#include "hb-number.hh"
#include "hb-atomic.hh"	// Requires: hb-meta
#include "hb-null.hh"	// Requires: hb-meta
#include "hb-algs.hh"	// Requires: hb-meta hb-null hb-number
#include "hb-iter.hh"	// Requires: hb-algs hb-meta
#include "hb-debug.hh"	// Requires: hb-algs hb-atomic
#include "hb-array.hh"	// Requires: hb-algs hb-iter hb-null
#include "hb-vector.hh"	// Requires: hb-array hb-null
#include "hb-object.hh"	// Requires: hb-atomic hb-mutex hb-vector

#endif /* HB_HH */
