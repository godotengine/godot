/*  A portable stdint.h
 ****************************************************************************
 *  BSD License:
 ****************************************************************************
 *
 *  Copyright (c) 2005-2016 Paul Hsieh
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *  3. The name of the author may not be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 *  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************
 *
 *  Version 0.1.15.4
 *
 *  The ANSI C standard committee, for the C99 standard, specified the
 *  inclusion of a new standard include file called stdint.h.  This is
 *  a very useful and long desired include file which contains several
 *  very precise definitions for integer scalar types that is
 *  critically important for making portable several classes of
 *  applications including cryptography, hashing, variable length
 *  integer libraries and so on.  But for most developers its likely
 *  useful just for programming sanity.
 *
 *  The problem is that some compiler vendors chose to ignore the C99
 *  standard and some older compilers have no opportunity to be updated.
 *  Because of this situation, simply including stdint.h in your code
 *  makes it unportable.
 *
 *  So that's what this file is all about.  Its an attempt to build a
 *  single universal include file that works on as many platforms as
 *  possible to deliver what stdint.h is supposed to.  Even compilers
 *  that already come with stdint.h can use this file instead without
 *  any loss of functionality.  A few things that should be noted about
 *  this file:
 *
 *    1) It is not guaranteed to be portable and/or present an identical
 *       interface on all platforms.  The extreme variability of the
 *       ANSI C standard makes this an impossibility right from the
 *       very get go. Its really only meant to be useful for the vast
 *       majority of platforms that possess the capability of
 *       implementing usefully and precisely defined, standard sized
 *       integer scalars.  Systems which are not intrinsically 2s
 *       complement may produce invalid constants.
 *
 *    2) There is an unavoidable use of non-reserved symbols.
 *
 *    3) Other standard include files are invoked.
 *
 *    4) This file may come in conflict with future platforms that do
 *       include stdint.h.  The hope is that one or the other can be
 *       used with no real difference.
 *
 *    5) In the current version, if your platform can't represent
 *       int32_t, int16_t and int8_t, it just dumps out with a compiler
 *       error.
 *
 *    6) 64 bit integers may or may not be defined.  Test for their
 *       presence with the test: #ifdef INT64_MAX or #ifdef UINT64_MAX.
 *       Note that this is different from the C99 specification which
 *       requires the existence of 64 bit support in the compiler.  If
 *       this is not defined for your platform, yet it is capable of
 *       dealing with 64 bits then it is because this file has not yet
 *       been extended to cover all of your system's capabilities.
 *
 *    7) (u)intptr_t may or may not be defined.  Test for its presence
 *       with the test: #ifdef PTRDIFF_MAX.  If this is not defined
 *       for your platform, then it is because this file has not yet
 *       been extended to cover all of your system's capabilities, not
 *       because its optional.
 *
 *    8) The following might not been defined even if your platform is
 *       capable of defining it:
 *
 *       WCHAR_MIN
 *       WCHAR_MAX
 *       (u)int64_t
 *       PTRDIFF_MIN
 *       PTRDIFF_MAX
 *       (u)intptr_t
 *
 *    9) The following have not been defined:
 *
 *       WINT_MIN
 *       WINT_MAX
 *
 *   10) The criteria for defining (u)int_least(*)_t isn't clear,
 *       except for systems which don't have a type that precisely
 *       defined 8, 16, or 32 bit types (which this include file does
 *       not support anyways). Default definitions have been given.
 *
 *   11) The criteria for defining (u)int_fast(*)_t isn't something I
 *       would trust to any particular compiler vendor or the ANSI C
 *       committee.  It is well known that "compatible systems" are
 *       commonly created that have very different performance
 *       characteristics from the systems they are compatible with,
 *       especially those whose vendors make both the compiler and the
 *       system.  Default definitions have been given, but its strongly
 *       recommended that users never use these definitions for any
 *       reason (they do *NOT* deliver any serious guarantee of
 *       improved performance -- not in this file, nor any vendor's
 *       stdint.h).
 *
 *   12) The following macros:
 *
 *       PRINTF_INTMAX_MODIFIER
 *       PRINTF_INT64_MODIFIER
 *       PRINTF_INT32_MODIFIER
 *       PRINTF_INT16_MODIFIER
 *       PRINTF_LEAST64_MODIFIER
 *       PRINTF_LEAST32_MODIFIER
 *       PRINTF_LEAST16_MODIFIER
 *       PRINTF_INTPTR_MODIFIER
 *
 *       are strings which have been defined as the modifiers required
 *       for the "d", "u" and "x" printf formats to correctly output
 *       (u)intmax_t, (u)int64_t, (u)int32_t, (u)int16_t, (u)least64_t,
 *       (u)least32_t, (u)least16_t and (u)intptr_t types respectively.
 *       PRINTF_INTPTR_MODIFIER is not defined for some systems which
 *       provide their own stdint.h.  PRINTF_INT64_MODIFIER is not
 *       defined if INT64_MAX is not defined.  These are an extension
 *       beyond what C99 specifies must be in stdint.h.
 *
 *       In addition, the following macros are defined:
 *
 *       PRINTF_INTMAX_HEX_WIDTH
 *       PRINTF_INT64_HEX_WIDTH
 *       PRINTF_INT32_HEX_WIDTH
 *       PRINTF_INT16_HEX_WIDTH
 *       PRINTF_INT8_HEX_WIDTH
 *       PRINTF_INTMAX_DEC_WIDTH
 *       PRINTF_INT64_DEC_WIDTH
 *       PRINTF_INT32_DEC_WIDTH
 *       PRINTF_INT16_DEC_WIDTH
 *       PRINTF_UINT8_DEC_WIDTH
 *       PRINTF_UINTMAX_DEC_WIDTH
 *       PRINTF_UINT64_DEC_WIDTH
 *       PRINTF_UINT32_DEC_WIDTH
 *       PRINTF_UINT16_DEC_WIDTH
 *       PRINTF_UINT8_DEC_WIDTH
 *
 *       Which specifies the maximum number of characters required to
 *       print the number of that type in either hexadecimal or decimal.
 *       These are an extension beyond what C99 specifies must be in
 *       stdint.h.
 *
 *  Compilers tested (all with 0 warnings at their highest respective
 *  settings): Borland Turbo C 2.0, WATCOM C/C++ 11.0 (16 bits and 32
 *  bits), Microsoft Visual C++ 6.0 (32 bit), Microsoft Visual Studio
 *  .net (VC7), Intel C++ 4.0, GNU gcc v3.3.3
 *
 *  This file should be considered a work in progress.  Suggestions for
 *  improvements, especially those which increase coverage are strongly
 *  encouraged.
 *
 *  Acknowledgements
 *
 *  The following people have made significant contributions to the
 *  development and testing of this file:
 *
 *  Chris Howie
 *  John Steele Scott
 *  Dave Thorup
 *  John Dill
 *  Florian Wobbe
 *  Christopher Sean Morrison
 *  Mikkel Fahnoe Jorgensen
 *
 */

#include <stddef.h>
#include <limits.h>
#include <signal.h>

/*
 *  For gcc with _STDINT_H, fill in the PRINTF_INT*_MODIFIER macros, and
 *  do nothing else.  On the Mac OS X version of gcc this is _STDINT_H_.
 */

#if ((defined(__SUNPRO_C) && __SUNPRO_C >= 0x570) || (defined(_MSC_VER) && _MSC_VER >= 1600) || (defined(__STDC__) && __STDC__ && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) || (defined (__WATCOMC__) && (defined (_STDINT_H_INCLUDED) || __WATCOMC__ >= 1250)) || (defined(__GNUC__) && (__GNUC__ > 3 || defined(_STDINT_H) || defined(_STDINT_H_) || defined (__UINT_FAST64_TYPE__)) )) && !defined (_PSTDINT_H_INCLUDED)
#include <stdint.h>
#define _PSTDINT_H_INCLUDED
# if defined(__GNUC__) && (defined(__x86_64__) || defined(__ppc64__)) && !(defined(__APPLE__) && defined(__MACH__))
#  ifndef PRINTF_INT64_MODIFIER
#   define PRINTF_INT64_MODIFIER "l"
#  endif
#  ifndef PRINTF_INT32_MODIFIER
#   define PRINTF_INT32_MODIFIER ""
#  endif
# else
#  ifndef PRINTF_INT64_MODIFIER
#   define PRINTF_INT64_MODIFIER "ll"
#  endif
#  ifndef PRINTF_INT32_MODIFIER
#   if (UINT_MAX == UINT32_MAX)
#    define PRINTF_INT32_MODIFIER ""
#   else
#    define PRINTF_INT32_MODIFIER "l"
#   endif
#  endif
# endif
# ifndef PRINTF_INT16_MODIFIER
#  define PRINTF_INT16_MODIFIER "h"
# endif
# ifndef PRINTF_INTMAX_MODIFIER
#  define PRINTF_INTMAX_MODIFIER PRINTF_INT64_MODIFIER
# endif
# ifndef PRINTF_INT64_HEX_WIDTH
#  define PRINTF_INT64_HEX_WIDTH "16"
# endif
# ifndef PRINTF_UINT64_HEX_WIDTH
#  define PRINTF_UINT64_HEX_WIDTH "16"
# endif
# ifndef PRINTF_INT32_HEX_WIDTH
#  define PRINTF_INT32_HEX_WIDTH "8"
# endif
# ifndef PRINTF_UINT32_HEX_WIDTH
#  define PRINTF_UINT32_HEX_WIDTH "8"
# endif
# ifndef PRINTF_INT16_HEX_WIDTH
#  define PRINTF_INT16_HEX_WIDTH "4"
# endif
# ifndef PRINTF_UINT16_HEX_WIDTH
#  define PRINTF_UINT16_HEX_WIDTH "4"
# endif
# ifndef PRINTF_INT8_HEX_WIDTH
#  define PRINTF_INT8_HEX_WIDTH "2"
# endif
# ifndef PRINTF_UINT8_HEX_WIDTH
#  define PRINTF_UINT8_HEX_WIDTH "2"
# endif
# ifndef PRINTF_INT64_DEC_WIDTH
#  define PRINTF_INT64_DEC_WIDTH "19"
# endif
# ifndef PRINTF_UINT64_DEC_WIDTH
#  define PRINTF_UINT64_DEC_WIDTH "20"
# endif
# ifndef PRINTF_INT32_DEC_WIDTH
#  define PRINTF_INT32_DEC_WIDTH "10"
# endif
# ifndef PRINTF_UINT32_DEC_WIDTH
#  define PRINTF_UINT32_DEC_WIDTH "10"
# endif
# ifndef PRINTF_INT16_DEC_WIDTH
#  define PRINTF_INT16_DEC_WIDTH "5"
# endif
# ifndef PRINTF_UINT16_DEC_WIDTH
#  define PRINTF_UINT16_DEC_WIDTH "5"
# endif
# ifndef PRINTF_INT8_DEC_WIDTH
#  define PRINTF_INT8_DEC_WIDTH "3"
# endif
# ifndef PRINTF_UINT8_DEC_WIDTH
#  define PRINTF_UINT8_DEC_WIDTH "3"
# endif
# ifndef PRINTF_INTMAX_HEX_WIDTH
#  define PRINTF_INTMAX_HEX_WIDTH PRINTF_UINT64_HEX_WIDTH
# endif
# ifndef PRINTF_UINTMAX_HEX_WIDTH
#  define PRINTF_UINTMAX_HEX_WIDTH PRINTF_UINT64_HEX_WIDTH
# endif
# ifndef PRINTF_INTMAX_DEC_WIDTH
#  define PRINTF_INTMAX_DEC_WIDTH PRINTF_UINT64_DEC_WIDTH
# endif
# ifndef PRINTF_UINTMAX_DEC_WIDTH
#  define PRINTF_UINTMAX_DEC_WIDTH PRINTF_UINT64_DEC_WIDTH
# endif

/*
 *  Something really weird is going on with Open Watcom.  Just pull some of
 *  these duplicated definitions from Open Watcom's stdint.h file for now.
 */

# if defined (__WATCOMC__) && __WATCOMC__ >= 1250
#  if !defined (INT64_C)
#   define INT64_C(x)   (x + (INT64_MAX - INT64_MAX))
#  endif
#  if !defined (UINT64_C)
#   define UINT64_C(x)  (x + (UINT64_MAX - UINT64_MAX))
#  endif
#  if !defined (INT32_C)
#   define INT32_C(x)   (x + (INT32_MAX - INT32_MAX))
#  endif
#  if !defined (UINT32_C)
#   define UINT32_C(x)  (x + (UINT32_MAX - UINT32_MAX))
#  endif
#  if !defined (INT16_C)
#   define INT16_C(x)   (x)
#  endif
#  if !defined (UINT16_C)
#   define UINT16_C(x)  (x)
#  endif
#  if !defined (INT8_C)
#   define INT8_C(x)   (x)
#  endif
#  if !defined (UINT8_C)
#   define UINT8_C(x)  (x)
#  endif
#  if !defined (UINT64_MAX)
#   define UINT64_MAX  18446744073709551615ULL
#  endif
#  if !defined (INT64_MAX)
#   define INT64_MAX  9223372036854775807LL
#  endif
#  if !defined (UINT32_MAX)
#   define UINT32_MAX  4294967295UL
#  endif
#  if !defined (INT32_MAX)
#   define INT32_MAX  2147483647L
#  endif
#  if !defined (INTMAX_MAX)
#   define INTMAX_MAX INT64_MAX
#  endif
#  if !defined (INTMAX_MIN)
#   define INTMAX_MIN INT64_MIN
#  endif
# endif
#endif

/*
 *  I have no idea what is the truly correct thing to do on older Solaris.
 *  From some online discussions, this seems to be what is being
 *  recommended.  For people who actually are developing on older Solaris,
 *  what I would like to know is, does this define all of the relevant
 *  macros of a complete stdint.h?  Remember, in pstdint.h 64 bit is
 *  considered optional.
 */

#if (defined(__SUNPRO_C) && __SUNPRO_C >= 0x420) && !defined(_PSTDINT_H_INCLUDED)
#include <sys/inttypes.h>
#define _PSTDINT_H_INCLUDED
#endif

#ifndef _PSTDINT_H_INCLUDED
#define _PSTDINT_H_INCLUDED

#ifndef SIZE_MAX
# define SIZE_MAX (~(size_t)0)
#endif

/*
 *  Deduce the type assignments from limits.h under the assumption that
 *  integer sizes in bits are powers of 2, and follow the ANSI
 *  definitions.
 */

#ifndef UINT8_MAX
# define UINT8_MAX 0xff
#endif
#if !defined(uint8_t) && !defined(_UINT8_T) && !defined(vxWorks)
# if (UCHAR_MAX == UINT8_MAX) || defined (S_SPLINT_S)
    typedef unsigned char uint8_t;
#   define UINT8_C(v) ((uint8_t) v)
# else
#   error "Platform not supported"
# endif
#endif

#ifndef INT8_MAX
# define INT8_MAX 0x7f
#endif
#ifndef INT8_MIN
# define INT8_MIN INT8_C(0x80)
#endif
#if !defined(int8_t) && !defined(_INT8_T) && !defined(vxWorks)
# if (SCHAR_MAX == INT8_MAX) || defined (S_SPLINT_S)
    typedef signed char int8_t;
#   define INT8_C(v) ((int8_t) v)
# else
#   error "Platform not supported"
# endif
#endif

#ifndef UINT16_MAX
# define UINT16_MAX 0xffff
#endif
#if !defined(uint16_t) && !defined(_UINT16_T) && !defined(vxWorks)
#if (UINT_MAX == UINT16_MAX) || defined (S_SPLINT_S)
  typedef unsigned int uint16_t;
# ifndef PRINTF_INT16_MODIFIER
#  define PRINTF_INT16_MODIFIER ""
# endif
# define UINT16_C(v) ((uint16_t) (v))
#elif (USHRT_MAX == UINT16_MAX)
  typedef unsigned short uint16_t;
# define UINT16_C(v) ((uint16_t) (v))
# ifndef PRINTF_INT16_MODIFIER
#  define PRINTF_INT16_MODIFIER "h"
# endif
#else
#error "Platform not supported"
#endif
#endif

#ifndef INT16_MAX
# define INT16_MAX 0x7fff
#endif
#ifndef INT16_MIN
# define INT16_MIN INT16_C(0x8000)
#endif
#if !defined(int16_t) && !defined(_INT16_T) && !defined(vxWorks)
#if (INT_MAX == INT16_MAX) || defined (S_SPLINT_S)
  typedef signed int int16_t;
# define INT16_C(v) ((int16_t) (v))
# ifndef PRINTF_INT16_MODIFIER
#  define PRINTF_INT16_MODIFIER ""
# endif
#elif (SHRT_MAX == INT16_MAX)
  typedef signed short int16_t;
# define INT16_C(v) ((int16_t) (v))
# ifndef PRINTF_INT16_MODIFIER
#  define PRINTF_INT16_MODIFIER "h"
# endif
#else
#error "Platform not supported"
#endif
#endif

#ifndef UINT32_MAX
# define UINT32_MAX (0xffffffffUL)
#endif
#if !defined(uint32_t) && !defined(_UINT32_T) && !defined(vxWorks)
#if (ULONG_MAX == UINT32_MAX) || defined (S_SPLINT_S)
  typedef unsigned long uint32_t;
# define UINT32_C(v) v ## UL
# ifndef PRINTF_INT32_MODIFIER
#  define PRINTF_INT32_MODIFIER "l"
# endif
#elif (UINT_MAX == UINT32_MAX)
  typedef unsigned int uint32_t;
# ifndef PRINTF_INT32_MODIFIER
#  define PRINTF_INT32_MODIFIER ""
# endif
# define UINT32_C(v) v ## U
#elif (USHRT_MAX == UINT32_MAX)
  typedef unsigned short uint32_t;
# define UINT32_C(v) ((unsigned short) (v))
# ifndef PRINTF_INT32_MODIFIER
#  define PRINTF_INT32_MODIFIER ""
# endif
#else
#error "Platform not supported"
#endif
#endif

#ifndef INT32_MAX
# define INT32_MAX (0x7fffffffL)
#endif
#ifndef INT32_MIN
# define INT32_MIN INT32_C(0x80000000)
#endif
#if !defined(int32_t) && !defined(_INT32_T) && !defined(vxWorks)
#if (LONG_MAX == INT32_MAX) || defined (S_SPLINT_S)
  typedef signed long int32_t;
# define INT32_C(v) v ## L
# ifndef PRINTF_INT32_MODIFIER
#  define PRINTF_INT32_MODIFIER "l"
# endif
#elif (INT_MAX == INT32_MAX)
  typedef signed int int32_t;
# define INT32_C(v) v
# ifndef PRINTF_INT32_MODIFIER
#  define PRINTF_INT32_MODIFIER ""
# endif
#elif (SHRT_MAX == INT32_MAX)
  typedef signed short int32_t;
# define INT32_C(v) ((short) (v))
# ifndef PRINTF_INT32_MODIFIER
#  define PRINTF_INT32_MODIFIER ""
# endif
#else
#error "Platform not supported"
#endif
#endif

/*
 *  The macro stdint_int64_defined is temporarily used to record
 *  whether or not 64 integer support is available.  It must be
 *  defined for any 64 integer extensions for new platforms that are
 *  added.
 */

#undef stdint_int64_defined
#if (defined(__STDC__) && defined(__STDC_VERSION__)) || defined (S_SPLINT_S)
# if (__STDC__ && __STDC_VERSION__ >= 199901L) || defined (S_SPLINT_S)
#  define stdint_int64_defined
   typedef long long int64_t;
   typedef unsigned long long uint64_t;
#  define UINT64_C(v) v ## ULL
#  define  INT64_C(v) v ## LL
#  ifndef PRINTF_INT64_MODIFIER
#   define PRINTF_INT64_MODIFIER "ll"
#  endif
# endif
#endif

#if !defined (stdint_int64_defined)
# if defined(__GNUC__) && !defined(vxWorks)
#  define stdint_int64_defined
   __extension__ typedef long long int64_t;
   __extension__ typedef unsigned long long uint64_t;
#  define UINT64_C(v) v ## ULL
#  define  INT64_C(v) v ## LL
#  ifndef PRINTF_INT64_MODIFIER
#   define PRINTF_INT64_MODIFIER "ll"
#  endif
# elif defined(__MWERKS__) || defined (__SUNPRO_C) || defined (__SUNPRO_CC) || defined (__APPLE_CC__) || defined (_LONG_LONG) || defined (_CRAYC) || defined (S_SPLINT_S)
#  define stdint_int64_defined
   typedef long long int64_t;
   typedef unsigned long long uint64_t;
#  define UINT64_C(v) v ## ULL
#  define  INT64_C(v) v ## LL
#  ifndef PRINTF_INT64_MODIFIER
#   define PRINTF_INT64_MODIFIER "ll"
#  endif
# elif (defined(__WATCOMC__) && defined(__WATCOM_INT64__)) || (defined(_MSC_VER) && _INTEGRAL_MAX_BITS >= 64) || (defined (__BORLANDC__) && __BORLANDC__ > 0x460) || defined (__alpha) || defined (__DECC)
#  define stdint_int64_defined
   typedef __int64 int64_t;
   typedef unsigned __int64 uint64_t;
#  define UINT64_C(v) v ## UI64
#  define  INT64_C(v) v ## I64
#  ifndef PRINTF_INT64_MODIFIER
#   define PRINTF_INT64_MODIFIER "I64"
#  endif
# endif
#endif

#if !defined (LONG_LONG_MAX) && defined (INT64_C)
# define LONG_LONG_MAX INT64_C (9223372036854775807)
#endif
#ifndef ULONG_LONG_MAX
# define ULONG_LONG_MAX UINT64_C (18446744073709551615)
#endif

#if !defined (INT64_MAX) && defined (INT64_C)
# define INT64_MAX INT64_C (9223372036854775807)
#endif
#if !defined (INT64_MIN) && defined (INT64_C)
# define INT64_MIN INT64_C (-9223372036854775808)
#endif
#if !defined (UINT64_MAX) && defined (INT64_C)
# define UINT64_MAX UINT64_C (18446744073709551615)
#endif

/*
 *  Width of hexadecimal for number field.
 */

#ifndef PRINTF_INT64_HEX_WIDTH
# define PRINTF_INT64_HEX_WIDTH "16"
#endif
#ifndef PRINTF_INT32_HEX_WIDTH
# define PRINTF_INT32_HEX_WIDTH "8"
#endif
#ifndef PRINTF_INT16_HEX_WIDTH
# define PRINTF_INT16_HEX_WIDTH "4"
#endif
#ifndef PRINTF_INT8_HEX_WIDTH
# define PRINTF_INT8_HEX_WIDTH "2"
#endif
#ifndef PRINTF_INT64_DEC_WIDTH
# define PRINTF_INT64_DEC_WIDTH "19"
#endif
#ifndef PRINTF_INT32_DEC_WIDTH
# define PRINTF_INT32_DEC_WIDTH "10"
#endif
#ifndef PRINTF_INT16_DEC_WIDTH
# define PRINTF_INT16_DEC_WIDTH "5"
#endif
#ifndef PRINTF_INT8_DEC_WIDTH
# define PRINTF_INT8_DEC_WIDTH "3"
#endif
#ifndef PRINTF_UINT64_DEC_WIDTH
# define PRINTF_UINT64_DEC_WIDTH "20"
#endif
#ifndef PRINTF_UINT32_DEC_WIDTH
# define PRINTF_UINT32_DEC_WIDTH "10"
#endif
#ifndef PRINTF_UINT16_DEC_WIDTH
# define PRINTF_UINT16_DEC_WIDTH "5"
#endif
#ifndef PRINTF_UINT8_DEC_WIDTH
# define PRINTF_UINT8_DEC_WIDTH "3"
#endif

/*
 *  Ok, lets not worry about 128 bit integers for now.  Moore's law says
 *  we don't need to worry about that until about 2040 at which point
 *  we'll have bigger things to worry about.
 */

#ifdef stdint_int64_defined
  typedef int64_t intmax_t;
  typedef uint64_t uintmax_t;
# define  INTMAX_MAX   INT64_MAX
# define  INTMAX_MIN   INT64_MIN
# define UINTMAX_MAX  UINT64_MAX
# define UINTMAX_C(v) UINT64_C(v)
# define  INTMAX_C(v)  INT64_C(v)
# ifndef PRINTF_INTMAX_MODIFIER
#   define PRINTF_INTMAX_MODIFIER PRINTF_INT64_MODIFIER
# endif
# ifndef PRINTF_INTMAX_HEX_WIDTH
#  define PRINTF_INTMAX_HEX_WIDTH PRINTF_INT64_HEX_WIDTH
# endif
# ifndef PRINTF_INTMAX_DEC_WIDTH
#  define PRINTF_INTMAX_DEC_WIDTH PRINTF_INT64_DEC_WIDTH
# endif
#else
  typedef int32_t intmax_t;
  typedef uint32_t uintmax_t;
# define  INTMAX_MAX   INT32_MAX
# define UINTMAX_MAX  UINT32_MAX
# define UINTMAX_C(v) UINT32_C(v)
# define  INTMAX_C(v)  INT32_C(v)
# ifndef PRINTF_INTMAX_MODIFIER
#   define PRINTF_INTMAX_MODIFIER PRINTF_INT32_MODIFIER
# endif
# ifndef PRINTF_INTMAX_HEX_WIDTH
#  define PRINTF_INTMAX_HEX_WIDTH PRINTF_INT32_HEX_WIDTH
# endif
# ifndef PRINTF_INTMAX_DEC_WIDTH
#  define PRINTF_INTMAX_DEC_WIDTH PRINTF_INT32_DEC_WIDTH
# endif
#endif

/*
 *  Because this file currently only supports platforms which have
 *  precise powers of 2 as bit sizes for the default integers, the
 *  least definitions are all trivial.  Its possible that a future
 *  version of this file could have different definitions.
 */

#ifndef stdint_least_defined
  typedef   int8_t   int_least8_t;
  typedef  uint8_t  uint_least8_t;
  typedef  int16_t  int_least16_t;
  typedef uint16_t uint_least16_t;
  typedef  int32_t  int_least32_t;
  typedef uint32_t uint_least32_t;
# define PRINTF_LEAST32_MODIFIER PRINTF_INT32_MODIFIER
# define PRINTF_LEAST16_MODIFIER PRINTF_INT16_MODIFIER
# define  UINT_LEAST8_MAX  UINT8_MAX
# define   INT_LEAST8_MAX   INT8_MAX
# define UINT_LEAST16_MAX UINT16_MAX
# define  INT_LEAST16_MAX  INT16_MAX
# define UINT_LEAST32_MAX UINT32_MAX
# define  INT_LEAST32_MAX  INT32_MAX
# define   INT_LEAST8_MIN   INT8_MIN
# define  INT_LEAST16_MIN  INT16_MIN
# define  INT_LEAST32_MIN  INT32_MIN
# ifdef stdint_int64_defined
    typedef  int64_t  int_least64_t;
    typedef uint64_t uint_least64_t;
#   define PRINTF_LEAST64_MODIFIER PRINTF_INT64_MODIFIER
#   define UINT_LEAST64_MAX UINT64_MAX
#   define  INT_LEAST64_MAX  INT64_MAX
#   define  INT_LEAST64_MIN  INT64_MIN
# endif
#endif
#undef stdint_least_defined

/*
 *  The ANSI C committee pretending to know or specify anything about
 *  performance is the epitome of misguided arrogance.  The mandate of
 *  this file is to *ONLY* ever support that absolute minimum
 *  definition of the fast integer types, for compatibility purposes.
 *  No extensions, and no attempt to suggest what may or may not be a
 *  faster integer type will ever be made in this file.  Developers are
 *  warned to stay away from these types when using this or any other
 *  stdint.h.
 */

typedef   int_least8_t   int_fast8_t;
typedef  uint_least8_t  uint_fast8_t;
typedef  int_least16_t  int_fast16_t;
typedef uint_least16_t uint_fast16_t;
typedef  int_least32_t  int_fast32_t;
typedef uint_least32_t uint_fast32_t;
#define  UINT_FAST8_MAX  UINT_LEAST8_MAX
#define   INT_FAST8_MAX   INT_LEAST8_MAX
#define UINT_FAST16_MAX UINT_LEAST16_MAX
#define  INT_FAST16_MAX  INT_LEAST16_MAX
#define UINT_FAST32_MAX UINT_LEAST32_MAX
#define  INT_FAST32_MAX  INT_LEAST32_MAX
#define   INT_FAST8_MIN   INT_LEAST8_MIN
#define  INT_FAST16_MIN  INT_LEAST16_MIN
#define  INT_FAST32_MIN  INT_LEAST32_MIN
#ifdef stdint_int64_defined
  typedef  int_least64_t  int_fast64_t;
  typedef uint_least64_t uint_fast64_t;
# define UINT_FAST64_MAX UINT_LEAST64_MAX
# define  INT_FAST64_MAX  INT_LEAST64_MAX
# define  INT_FAST64_MIN  INT_LEAST64_MIN
#endif

#undef stdint_int64_defined

/*
 *  Whatever piecemeal, per compiler thing we can do about the wchar_t
 *  type limits.
 */

#if defined(__WATCOMC__) || defined(_MSC_VER) || defined (__GNUC__) && !defined(vxWorks)
# include <wchar.h>
# ifndef WCHAR_MIN
#  define WCHAR_MIN 0
# endif
# ifndef WCHAR_MAX
#  define WCHAR_MAX ((wchar_t)-1)
# endif
#endif

/*
 *  Whatever piecemeal, per compiler/platform thing we can do about the
 *  (u)intptr_t types and limits.
 */

#if (defined (_MSC_VER) && defined (_UINTPTR_T_DEFINED)) || defined (_UINTPTR_T)
# define STDINT_H_UINTPTR_T_DEFINED
#endif

#ifndef STDINT_H_UINTPTR_T_DEFINED
# if defined (__alpha__) || defined (__ia64__) || defined (__x86_64__) || defined (_WIN64) || defined (__ppc64__)
#  define stdint_intptr_bits 64
# elif defined (__WATCOMC__) || defined (__TURBOC__)
#  if defined(__TINY__) || defined(__SMALL__) || defined(__MEDIUM__)
#    define stdint_intptr_bits 16
#  else
#    define stdint_intptr_bits 32
#  endif
# elif defined (__i386__) || defined (_WIN32) || defined (WIN32) || defined (__ppc64__)
#  define stdint_intptr_bits 32
# elif defined (__INTEL_COMPILER)
/* TODO -- what did Intel do about x86-64? */
# else
/* #error "This platform might not be supported yet" */
# endif

# ifdef stdint_intptr_bits
#  define stdint_intptr_glue3_i(a,b,c)  a##b##c
#  define stdint_intptr_glue3(a,b,c)    stdint_intptr_glue3_i(a,b,c)
#  ifndef PRINTF_INTPTR_MODIFIER
#    define PRINTF_INTPTR_MODIFIER      stdint_intptr_glue3(PRINTF_INT,stdint_intptr_bits,_MODIFIER)
#  endif
#  ifndef PTRDIFF_MAX
#    define PTRDIFF_MAX                 stdint_intptr_glue3(INT,stdint_intptr_bits,_MAX)
#  endif
#  ifndef PTRDIFF_MIN
#    define PTRDIFF_MIN                 stdint_intptr_glue3(INT,stdint_intptr_bits,_MIN)
#  endif
#  ifndef UINTPTR_MAX
#    define UINTPTR_MAX                 stdint_intptr_glue3(UINT,stdint_intptr_bits,_MAX)
#  endif
#  ifndef INTPTR_MAX
#    define INTPTR_MAX                  stdint_intptr_glue3(INT,stdint_intptr_bits,_MAX)
#  endif
#  ifndef INTPTR_MIN
#    define INTPTR_MIN                  stdint_intptr_glue3(INT,stdint_intptr_bits,_MIN)
#  endif
#  ifndef INTPTR_C
#    define INTPTR_C(x)                 stdint_intptr_glue3(INT,stdint_intptr_bits,_C)(x)
#  endif
#  ifndef UINTPTR_C
#    define UINTPTR_C(x)                stdint_intptr_glue3(UINT,stdint_intptr_bits,_C)(x)
#  endif
  typedef stdint_intptr_glue3(uint,stdint_intptr_bits,_t) uintptr_t;
  typedef stdint_intptr_glue3( int,stdint_intptr_bits,_t)  intptr_t;
# else
/* TODO -- This following is likely wrong for some platforms, and does
   nothing for the definition of uintptr_t. */
  typedef ptrdiff_t intptr_t;
# endif
# define STDINT_H_UINTPTR_T_DEFINED
#endif

/*
 *  Assumes sig_atomic_t is signed and we have a 2s complement machine.
 */

#ifndef SIG_ATOMIC_MAX
# define SIG_ATOMIC_MAX ((((sig_atomic_t) 1) << (sizeof (sig_atomic_t)*CHAR_BIT-1)) - 1)
#endif

#endif

#if defined (__TEST_PSTDINT_FOR_CORRECTNESS)

/*
 *  Please compile with the maximum warning settings to make sure macros are
 *  not defined more than once.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define glue3_aux(x,y,z) x ## y ## z
#define glue3(x,y,z) glue3_aux(x,y,z)

#define DECLU(bits) glue3(uint,bits,_t) glue3(u,bits,) = glue3(UINT,bits,_C) (0);
#define DECLI(bits) glue3(int,bits,_t) glue3(i,bits,) = glue3(INT,bits,_C) (0);

#define DECL(us,bits) glue3(DECL,us,) (bits)

#define TESTUMAX(bits) glue3(u,bits,) = ~glue3(u,bits,); if (glue3(UINT,bits,_MAX) != glue3(u,bits,)) printf ("Something wrong with UINT%d_MAX\n", bits)

#define REPORTERROR(msg) { err_n++; if (err_first <= 0) err_first = __LINE__; printf msg; }

int main () {
	int err_n = 0;
	int err_first = 0;
	DECL(I,8)
	DECL(U,8)
	DECL(I,16)
	DECL(U,16)
	DECL(I,32)
	DECL(U,32)
#ifdef INT64_MAX
	DECL(I,64)
	DECL(U,64)
#endif
	intmax_t imax = INTMAX_C(0);
	uintmax_t umax = UINTMAX_C(0);
	char str0[256], str1[256];

	sprintf (str0, "%" PRINTF_INT32_MODIFIER "d", INT32_C(2147483647));
	if (0 != strcmp (str0, "2147483647")) REPORTERROR (("Something wrong with PRINTF_INT32_MODIFIER : %s\n", str0));
	if (atoi(PRINTF_INT32_DEC_WIDTH) != (int) strlen(str0)) REPORTERROR (("Something wrong with PRINTF_INT32_DEC_WIDTH : %s\n", PRINTF_INT32_DEC_WIDTH));
	sprintf (str0, "%" PRINTF_INT32_MODIFIER "u", UINT32_C(4294967295));
	if (0 != strcmp (str0, "4294967295")) REPORTERROR (("Something wrong with PRINTF_INT32_MODIFIER : %s\n", str0));
	if (atoi(PRINTF_UINT32_DEC_WIDTH) != (int) strlen(str0)) REPORTERROR (("Something wrong with PRINTF_UINT32_DEC_WIDTH : %s\n", PRINTF_UINT32_DEC_WIDTH));
#ifdef INT64_MAX
	sprintf (str1, "%" PRINTF_INT64_MODIFIER "d", INT64_C(9223372036854775807));
	if (0 != strcmp (str1, "9223372036854775807")) REPORTERROR (("Something wrong with PRINTF_INT32_MODIFIER : %s\n", str1));
	if (atoi(PRINTF_INT64_DEC_WIDTH) != (int) strlen(str1)) REPORTERROR (("Something wrong with PRINTF_INT64_DEC_WIDTH : %s, %d\n", PRINTF_INT64_DEC_WIDTH, (int) strlen(str1)));
	sprintf (str1, "%" PRINTF_INT64_MODIFIER "u", UINT64_C(18446744073709550591));
	if (0 != strcmp (str1, "18446744073709550591")) REPORTERROR (("Something wrong with PRINTF_INT32_MODIFIER : %s\n", str1));
	if (atoi(PRINTF_UINT64_DEC_WIDTH) != (int) strlen(str1)) REPORTERROR (("Something wrong with PRINTF_UINT64_DEC_WIDTH : %s, %d\n", PRINTF_UINT64_DEC_WIDTH, (int) strlen(str1)));
#endif

	sprintf (str0, "%d %x\n", 0, ~0);

	sprintf (str1, "%d %x\n",  i8, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with i8 : %s\n", str1));
	sprintf (str1, "%u %x\n",  u8, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with u8 : %s\n", str1));
	sprintf (str1, "%d %x\n",  i16, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with i16 : %s\n", str1));
	sprintf (str1, "%u %x\n",  u16, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with u16 : %s\n", str1));
	sprintf (str1, "%" PRINTF_INT32_MODIFIER "d %x\n",  i32, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with i32 : %s\n", str1));
	sprintf (str1, "%" PRINTF_INT32_MODIFIER "u %x\n",  u32, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with u32 : %s\n", str1));
#ifdef INT64_MAX
	sprintf (str1, "%" PRINTF_INT64_MODIFIER "d %x\n",  i64, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with i64 : %s\n", str1));
#endif
	sprintf (str1, "%" PRINTF_INTMAX_MODIFIER "d %x\n",  imax, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with imax : %s\n", str1));
	sprintf (str1, "%" PRINTF_INTMAX_MODIFIER "u %x\n",  umax, ~0);
	if (0 != strcmp (str0, str1)) REPORTERROR (("Something wrong with umax : %s\n", str1));

	TESTUMAX(8);
	TESTUMAX(16);
	TESTUMAX(32);
#ifdef INT64_MAX
	TESTUMAX(64);
#endif

#define STR(v) #v
#define Q(v) printf ("sizeof " STR(v) " = %u\n", (unsigned) sizeof (v));
	if (err_n) {
		printf ("pstdint.h is not correct.  Please use sizes below to correct it:\n");
	}

	Q(int)
	Q(unsigned)
	Q(long int)
	Q(short int)
	Q(int8_t)
	Q(int16_t)
	Q(int32_t)
#ifdef INT64_MAX
	Q(int64_t)
#endif

	return EXIT_SUCCESS;
}

#endif
