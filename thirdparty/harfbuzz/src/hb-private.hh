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

#ifndef HB_PRIVATE_HH
#define HB_PRIVATE_HH

#define _GNU_SOURCE 1

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "hb.h"
#define HB_H_IN
#ifdef HAVE_OT
#include "hb-ot.h"
#define HB_OT_H_IN
#endif

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>

#if (defined(_MSC_VER) && _MSC_VER >= 1500) || defined(__MINGW32__)
#include <intrin.h>
#endif

#define HB_PASTE1(a,b) a##b
#define HB_PASTE(a,b) HB_PASTE1(a,b)


/* Compile-time custom allocator support. */

#if defined(hb_malloc_impl) \
 && defined(hb_calloc_impl) \
 && defined(hb_realloc_impl) \
 && defined(hb_free_impl)
extern "C" void* hb_malloc_impl(size_t size);
extern "C" void* hb_calloc_impl(size_t nmemb, size_t size);
extern "C" void* hb_realloc_impl(void *ptr, size_t size);
extern "C" void  hb_free_impl(void *ptr);
#define malloc hb_malloc_impl
#define calloc hb_calloc_impl
#define realloc hb_realloc_impl
#define free hb_free_impl

#if defined(hb_memalign_impl)
extern "C" int hb_memalign_impl(void **memptr, size_t alignment, size_t size);
#define posix_memalign hb_memalign_impl
#else
#undef HAVE_POSIX_MEMALIGN
#endif

#endif


/*
 * Compiler attributes
 */

#if __cplusplus < 201103L

#ifndef nullptr
#define nullptr NULL
#endif

#ifndef constexpr
#define constexpr const
#endif

#ifndef static_assert
#define static_assert(e, msg) \
	HB_UNUSED typedef int HB_PASTE(static_assertion_failed_at_line_, __LINE__) [(e) ? 1 : -1]
#endif // static_assert

#ifdef __GNUC__
#if (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
#define thread_local __thread
#endif
#else
#define thread_local
#endif

template <typename T>
struct _hb_alignof
{
  struct s
  {
    char c;
    T t;
  };
  static constexpr size_t value = offsetof (s, t);
};
#ifndef alignof
#define alignof(x) (_hb_alignof<x>::value)
#endif

/* https://github.com/harfbuzz/harfbuzz/issues/1127 */
#ifndef explicit_operator
#define explicit_operator
#endif

#else /* __cplusplus >= 201103L */

/* https://github.com/harfbuzz/harfbuzz/issues/1127 */
#ifndef explicit_operator
#define explicit_operator explicit
#endif

#endif /* __cplusplus < 201103L */


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

#if __GNUC__ >= 3
#define HB_PURE_FUNC	__attribute__((pure))
#define HB_CONST_FUNC	__attribute__((const))
#define HB_PRINTF_FUNC(format_idx, arg_idx) __attribute__((__format__ (__printf__, format_idx, arg_idx)))
#else
#define HB_PURE_FUNC
#define HB_CONST_FUNC
#define HB_PRINTF_FUNC(format_idx, arg_idx)
#endif
#if __GNUC__ >= 4
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
   /* We use -export-symbols on mingw32, since it does not support visibility
    * attribute. */
#  define HB_INTERNAL
#else
#  define HB_INTERNAL
#  define HB_NO_VISIBILITY 1
# endif
#endif

#if __GNUC__ >= 3
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
#elif __GNUC__ >= 7
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

#if defined(_WIN32) || defined(__CYGWIN__)
   /* We need Windows Vista for both Uniscribe backend and for
    * MemoryBarrier.  We don't support compiling on Windows XP,
    * though we run on it fine. */
#  if defined(_WIN32_WINNT) && _WIN32_WINNT < 0x0600
#    undef _WIN32_WINNT
#  endif
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0600
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
#    define getenv(Name) nullptr
#    if _WIN32_WCE < 0x800
#      define setlocale(Category, Locale) "C"
static int errno = 0; /* Use something better? */
#    endif
#  elif defined(WINAPI_FAMILY) && (WINAPI_FAMILY==WINAPI_FAMILY_PC_APP || WINAPI_FAMILY==WINAPI_FAMILY_PHONE_APP)
#    define getenv(Name) nullptr
#  endif
#  if defined(_MSC_VER) && _MSC_VER < 1900
#    define snprintf _snprintf
#  endif
#endif

#if HAVE_ATEXIT
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


/* We like our types POD */

#define _ASSERT_TYPE_POD1(_line, _type)	union _type_##_type##_on_line_##_line##_is_not_POD { _type instance; }
#define _ASSERT_TYPE_POD0(_line, _type)	_ASSERT_TYPE_POD1 (_line, _type)
#define ASSERT_TYPE_POD(_type)		_ASSERT_TYPE_POD0 (__LINE__, _type)

#ifdef __GNUC__
# define _ASSERT_INSTANCE_POD1(_line, _instance) \
	HB_STMT_START { \
		typedef __typeof__(_instance) _type_##_line; \
		_ASSERT_TYPE_POD1 (_line, _type_##_line); \
	} HB_STMT_END
#else
# define _ASSERT_INSTANCE_POD1(_line, _instance)	typedef int _assertion_on_line_##_line##_not_tested
#endif
# define _ASSERT_INSTANCE_POD0(_line, _instance)	_ASSERT_INSTANCE_POD1 (_line, _instance)
# define ASSERT_INSTANCE_POD(_instance)			_ASSERT_INSTANCE_POD0 (__LINE__, _instance)

/* Check _assertion in a method environment */
#define _ASSERT_POD1(_line) \
	HB_UNUSED inline void _static_assertion_on_line_##_line (void) const \
	{ _ASSERT_INSTANCE_POD1 (_line, *this); /* Make sure it's POD. */ }
# define _ASSERT_POD0(_line)	_ASSERT_POD1 (_line)
# define ASSERT_POD()		_ASSERT_POD0 (__LINE__)


#define HB_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&); \
  void operator=(const TypeName&)


/*
 * Compiler-assisted vectorization parameters.
 */

/*
 * Disable vectorization for now.  To correctly use them, we should
 * use posix_memalign() to allocate in hb_vector_t.  Otherwise, can
 * cause misaligned access.
 *
 * https://bugs.chromium.org/p/chromium/issues/detail?id=860184
 */
#if !defined(HB_VECTOR_SIZE)
#  define HB_VECTOR_SIZE 0
#endif

/* The `vector_size' attribute was introduced in gcc 3.1. */
#if !defined(HB_VECTOR_SIZE)
#  if defined( __GNUC__ ) && ( __GNUC__ >= 4 )
#    define HB_VECTOR_SIZE 128
#  else
#    define HB_VECTOR_SIZE 0
#  endif
#endif
static_assert (0 == (HB_VECTOR_SIZE & (HB_VECTOR_SIZE - 1)), "HB_VECTOR_SIZE is not power of 2.");
static_assert (0 == (HB_VECTOR_SIZE % 64), "HB_VECTOR_SIZE is not multiple of 64.");
#if HB_VECTOR_SIZE
typedef uint64_t hb_vector_size_impl_t __attribute__((vector_size (HB_VECTOR_SIZE / 8)));
#else
typedef uint64_t hb_vector_size_impl_t;
#endif


/* HB_NDEBUG disables some sanity checks that are very safe to disable and
 * should be disabled in production systems.  If NDEBUG is defined, enable
 * HB_NDEBUG; but if it's desirable that normal assert()s (which are very
 * light-weight) to be enabled, then HB_DEBUG can be defined to disable
 * the costlier checks. */
#ifdef NDEBUG
#define HB_NDEBUG 1
#endif


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
	}

/* Useful for set-operations on small enums.
 * For example, for testing "x ∈ {x1, x2, x3}" use:
 * (FLAG_UNSAFE(x) & (FLAG(x1) | FLAG(x2) | FLAG(x3)))
 */
#define FLAG(x) (ASSERT_STATIC_EXPR_ZERO ((unsigned int)(x) < 32) + (1U << (unsigned int)(x)))
#define FLAG_UNSAFE(x) ((unsigned int)(x) < 32 ? (1U << (unsigned int)(x)) : 0)
#define FLAG_RANGE(x,y) (ASSERT_STATIC_EXPR_ZERO ((x) < (y)) + FLAG(y+1) - FLAG(x))


/* Size signifying variable-sized array */
#define VAR 1


/* fallback for round() */
static inline double
_hb_round (double x)
{
  if (x >= 0)
    return floor (x + 0.5);
  else
    return ceil (x - 0.5);
}
#if !defined (HAVE_ROUND) && !defined (HAVE_DECL_ROUND)
#define round(x) _hb_round(x)
#endif


/* fallback for posix_memalign() */
static inline int
_hb_memalign(void **memptr, size_t alignment, size_t size)
{
  if (unlikely (0 != (alignment & (alignment - 1)) ||
		!alignment ||
		0 != (alignment & (sizeof (void *) - 1))))
    return EINVAL;

  char *p = (char *) malloc (size + alignment - 1);
  if (unlikely (!p))
    return ENOMEM;

  size_t off = (size_t) p & (alignment - 1);
  if (off)
    p += alignment - off;

  *memptr = (void *) p;

  return 0;
}
#if !defined(posix_memalign) && !defined(HAVE_POSIX_MEMALIGN)
#define posix_memalign _hb_memalign
#endif


/* Headers we include for everyone.  Keep sorted.  They express dependency amongst
 * themselves, but no other file should include them.*/
#include "hb-atomic-private.hh"
#include "hb-debug.hh"
#include "hb-dsalgs.hh"
#include "hb-mutex-private.hh"
#include "hb-null.hh"
#include "hb-object-private.hh"

#endif /* HB_PRIVATE_HH */
