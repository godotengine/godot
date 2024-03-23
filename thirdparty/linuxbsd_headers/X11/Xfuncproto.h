/*
 *
Copyright 1989, 1991, 1998  The Open Group

Permission to use, copy, modify, distribute, and sell this software and its
documentation for any purpose is hereby granted without fee, provided that
the above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
OPEN GROUP BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of The Open Group shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from The Open Group.
 *
 */

/* Definitions to make function prototypes manageable */

#ifndef _XFUNCPROTO_H_
#define _XFUNCPROTO_H_

#ifndef NeedFunctionPrototypes
#define NeedFunctionPrototypes 1
#endif /* NeedFunctionPrototypes */

#ifndef NeedVarargsPrototypes
#define NeedVarargsPrototypes 1
#endif /* NeedVarargsPrototypes */

#if NeedFunctionPrototypes

#ifndef NeedNestedPrototypes
#define NeedNestedPrototypes 1
#endif /* NeedNestedPrototypes */

#ifndef _Xconst
#define _Xconst const
#endif /* _Xconst */

/* Function prototype configuration (see configure for more info) */
#if !defined(NARROWPROTO) && \
    (defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__))
#define NARROWPROTO
#endif
#ifndef FUNCPROTO
#define FUNCPROTO 15
#endif

#ifndef NeedWidePrototypes
#ifdef NARROWPROTO
#define NeedWidePrototypes 0
#else
#define NeedWidePrototypes 1		/* default to make interropt. easier */
#endif
#endif /* NeedWidePrototypes */

#endif /* NeedFunctionPrototypes */

#ifndef _XFUNCPROTOBEGIN
#if defined(__cplusplus) || defined(c_plusplus) /* for C++ V2.0 */
#define _XFUNCPROTOBEGIN extern "C" {	/* do not leave open across includes */
#define _XFUNCPROTOEND }
#else
#define _XFUNCPROTOBEGIN
#define _XFUNCPROTOEND
#endif
#endif /* _XFUNCPROTOBEGIN */

/* http://clang.llvm.org/docs/LanguageExtensions.html#has-attribute */
#ifndef __has_attribute
# define __has_attribute(x) 0  /* Compatibility with non-clang compilers. */
#endif
#ifndef __has_feature
# define __has_feature(x) 0    /* Compatibility with non-clang compilers. */
#endif
#ifndef __has_extension
# define __has_extension(x) 0  /* Compatibility with non-clang compilers. */
#endif

/* Added in X11R6.9, so available in any version of modular xproto */
#if __has_attribute(__sentinel__) || (defined(__GNUC__) && (__GNUC__ >= 4))
# define _X_SENTINEL(x) __attribute__ ((__sentinel__(x)))
#else
# define _X_SENTINEL(x)
#endif /* GNUC >= 4 */

/* Added in X11R6.9, so available in any version of modular xproto */
#if (__has_attribute(visibility) || (defined(__GNUC__) && (__GNUC__ >= 4))) \
    && !defined(__CYGWIN__) && !defined(__MINGW32__)
# define _X_EXPORT      __attribute__((visibility("default")))
# define _X_HIDDEN      __attribute__((visibility("hidden")))
# define _X_INTERNAL    __attribute__((visibility("internal")))
#elif defined(__SUNPRO_C) && (__SUNPRO_C >= 0x550)
# define _X_EXPORT      __global
# define _X_HIDDEN      __hidden
# define _X_INTERNAL    __hidden
#else /* not gcc >= 4 and not Sun Studio >= 8 */
# define _X_EXPORT
# define _X_HIDDEN
# define _X_INTERNAL
#endif /* GNUC >= 4 */

/* Branch prediction hints for individual conditionals */
/* requires xproto >= 7.0.9 */
#if defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 303)
# define _X_LIKELY(x)   __builtin_expect(!!(x), 1)
# define _X_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else /* not gcc >= 3.3 */
# define _X_LIKELY(x)   (x)
# define _X_UNLIKELY(x) (x)
#endif

/* Bulk branch prediction hints via marking error path functions as "cold" */
/* requires xproto >= 7.0.25 */
#if __has_attribute(__cold__) || \
    (defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 403)) /* 4.3+ */
# define _X_COLD __attribute__((__cold__))
#else
# define _X_COLD /* nothing */
#endif

/* Added in X11R6.9, so available in any version of modular xproto */
#if __has_attribute(deprecated) \
    || (defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 301)) \
    || (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x5130))
# define _X_DEPRECATED  __attribute__((deprecated))
#else /* not gcc >= 3.1 */
# define _X_DEPRECATED
#endif

/* requires xproto >= 7.0.30 */
#if __has_extension(attribute_deprecated_with_message) || \
                (defined(__GNUC__) && ((__GNUC__ >= 5) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5))))
# define _X_DEPRECATED_MSG(_msg) __attribute__((deprecated(_msg)))
#else
# define _X_DEPRECATED_MSG(_msg) _X_DEPRECATED
#endif

/* requires xproto >= 7.0.17 */
#if __has_attribute(noreturn) \
    || (defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 205)) \
    || (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590))
# define _X_NORETURN __attribute((noreturn))
#else
# define _X_NORETURN
#endif /* GNUC  */

/* Added in X11R6.9, so available in any version of modular xproto */
#if __has_attribute(__format__) \
    || defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 203)
# define _X_ATTRIBUTE_PRINTF(x,y) __attribute__((__format__(__printf__,x,y)))
#else /* not gcc >= 2.3 */
# define _X_ATTRIBUTE_PRINTF(x,y)
#endif

/* requires xproto >= 7.0.22 - since this uses either gcc or C99 variable
   argument macros, must be only used inside #ifdef _X_NONNULL guards, as
   many legacy X clients are compiled in C89 mode still. */
#if __has_attribute(nonnull) \
    && defined(__STDC_VERSION__) && (__STDC_VERSION__ - 0 >= 199901L) /* C99 */
#define _X_NONNULL(...)  __attribute__((nonnull(__VA_ARGS__)))
#elif __has_attribute(nonnull) \
    || defined(__GNUC__) &&  ((__GNUC__ * 100 + __GNUC_MINOR__) >= 303)
#define _X_NONNULL(args...)  __attribute__((nonnull(args)))
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ - 0 >= 199901L) /* C99 */
#define _X_NONNULL(...)  /* */
#endif

/* requires xproto >= 7.0.22 */
#if __has_attribute(__unused__) \
    || defined(__GNUC__) &&  ((__GNUC__ * 100 + __GNUC_MINOR__) >= 205)
#define _X_UNUSED  __attribute__((__unused__))
#else
#define _X_UNUSED  /* */
#endif

/* C99 keyword "inline" or equivalent extensions in pre-C99 compilers */
/* requires xproto >= 7.0.9
   (introduced in 7.0.8 but didn't support all compilers until 7.0.9) */
#if defined(inline) /* assume autoconf set it correctly */ || \
   (defined(__STDC_VERSION__) && (__STDC_VERSION__ - 0 >= 199901L)) /* C99 */ || \
   (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x550))
# define _X_INLINE inline
#elif defined(__GNUC__) && !defined(__STRICT_ANSI__) /* gcc w/C89+extensions */
# define _X_INLINE __inline__
#else
# define _X_INLINE
#endif

/* C99 keyword "restrict" or equivalent extensions in pre-C99 compilers */
/* requires xproto >= 7.0.21 */
#ifndef _X_RESTRICT_KYWD
# if defined(restrict) /* assume autoconf set it correctly */ || \
    (defined(__STDC_VERSION__) && (__STDC_VERSION__ - 0 >= 199901L) /* C99 */ \
     && !defined(__cplusplus)) /* Workaround g++ issue on Solaris */
#  define _X_RESTRICT_KYWD  restrict
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__) /* gcc w/C89+extensions */
#  define _X_RESTRICT_KYWD __restrict__
# else
#  define _X_RESTRICT_KYWD
# endif
#endif

/* requires xproto >= 7.0.30 */
#if __has_attribute(no_sanitize_thread)
# define _X_NOTSAN __attribute__((no_sanitize_thread))
#else
# define _X_NOTSAN
#endif

/* Mark a char array/pointer as not containing a NUL-terminated string */
/* requires xproto >= 7.0.33 */
#if __has_attribute(nonstring)
# define _X_NONSTRING __attribute__((nonstring))
#else
# define _X_NONSTRING
#endif

#endif /* _XFUNCPROTO_H_ */
