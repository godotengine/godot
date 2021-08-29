#ifndef CURLINC_SYSTEM_H
#define CURLINC_SYSTEM_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

/*
 * Try to keep one section per platform, compiler and architecture, otherwise,
 * if an existing section is reused for a different one and later on the
 * original is adjusted, probably the piggybacking one can be adversely
 * changed.
 *
 * In order to differentiate between platforms/compilers/architectures use
 * only compiler built in predefined preprocessor symbols.
 *
 * curl_off_t
 * ----------
 *
 * For any given platform/compiler curl_off_t must be typedef'ed to a 64-bit
 * wide signed integral data type. The width of this data type must remain
 * constant and independent of any possible large file support settings.
 *
 * As an exception to the above, curl_off_t shall be typedef'ed to a 32-bit
 * wide signed integral data type if there is no 64-bit type.
 *
 * As a general rule, curl_off_t shall not be mapped to off_t. This rule shall
 * only be violated if off_t is the only 64-bit data type available and the
 * size of off_t is independent of large file support settings. Keep your
 * build on the safe side avoiding an off_t gating.  If you have a 64-bit
 * off_t then take for sure that another 64-bit data type exists, dig deeper
 * and you will find it.
 *
 */

#if defined(__DJGPP__) || defined(__GO32__)
#  if defined(__DJGPP__) && (__DJGPP__ > 1)
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  else
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__SALFORDC__)
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__BORLANDC__)
#  if (__BORLANDC__ < 0x520)
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  else
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__TURBOC__)
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__WATCOMC__)
#  if defined(__386__)
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  else
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__POCC__)
#  if (__POCC__ < 280)
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  elif defined(_MSC_VER)
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  else
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__LCC__)
#  if defined(__e2k__) /* MCST eLbrus C Compiler */
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  else                /* Local (or Little) C Compiler */
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#    define CURL_TYPEOF_CURL_SOCKLEN_T int
#  endif

#elif defined(__SYMBIAN32__)
#  if defined(__EABI__) /* Treat all ARM compilers equally */
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__CW32__)
#    pragma longlong on
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__VC32__)
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T unsigned int

#elif defined(__MWERKS__)
#  define CURL_TYPEOF_CURL_OFF_T     long long
#  define CURL_FORMAT_CURL_OFF_T     "lld"
#  define CURL_FORMAT_CURL_OFF_TU    "llu"
#  define CURL_SUFFIX_CURL_OFF_T     LL
#  define CURL_SUFFIX_CURL_OFF_TU    ULL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(_WIN32_WCE)
#  define CURL_TYPEOF_CURL_OFF_T     __int64
#  define CURL_FORMAT_CURL_OFF_T     "I64d"
#  define CURL_FORMAT_CURL_OFF_TU    "I64u"
#  define CURL_SUFFIX_CURL_OFF_T     i64
#  define CURL_SUFFIX_CURL_OFF_TU    ui64
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__MINGW32__)
#  define CURL_TYPEOF_CURL_OFF_T     long long
#  define CURL_FORMAT_CURL_OFF_T     "I64d"
#  define CURL_FORMAT_CURL_OFF_TU    "I64u"
#  define CURL_SUFFIX_CURL_OFF_T     LL
#  define CURL_SUFFIX_CURL_OFF_TU    ULL
#  define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#  define CURL_PULL_SYS_TYPES_H      1
#  define CURL_PULL_WS2TCPIP_H       1

#elif defined(__VMS)
#  if defined(__VAX)
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  else
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T unsigned int

#elif defined(__OS400__)
#  if defined(__ILEC400__)
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  endif

#elif defined(__MVS__)
#  if defined(__IBMC__) || defined(__IBMCPP__)
#    if defined(_ILP32)
#    elif defined(_LP64)
#    endif
#    if defined(_LONG_LONG)
#      define CURL_TYPEOF_CURL_OFF_T     long long
#      define CURL_FORMAT_CURL_OFF_T     "lld"
#      define CURL_FORMAT_CURL_OFF_TU    "llu"
#      define CURL_SUFFIX_CURL_OFF_T     LL
#      define CURL_SUFFIX_CURL_OFF_TU    ULL
#    elif defined(_LP64)
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    else
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    endif
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  endif

#elif defined(__370__)
#  if defined(__IBMC__) || defined(__IBMCPP__)
#    if defined(_ILP32)
#    elif defined(_LP64)
#    endif
#    if defined(_LONG_LONG)
#      define CURL_TYPEOF_CURL_OFF_T     long long
#      define CURL_FORMAT_CURL_OFF_T     "lld"
#      define CURL_FORMAT_CURL_OFF_TU    "llu"
#      define CURL_SUFFIX_CURL_OFF_T     LL
#      define CURL_SUFFIX_CURL_OFF_TU    ULL
#    elif defined(_LP64)
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    else
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    endif
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  endif

#elif defined(TPF)
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

#elif defined(__TINYC__) /* also known as tcc */
#  define CURL_TYPEOF_CURL_OFF_T     long long
#  define CURL_FORMAT_CURL_OFF_T     "lld"
#  define CURL_FORMAT_CURL_OFF_TU    "llu"
#  define CURL_SUFFIX_CURL_OFF_T     LL
#  define CURL_SUFFIX_CURL_OFF_TU    ULL
#  define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#  define CURL_PULL_SYS_TYPES_H      1
#  define CURL_PULL_SYS_SOCKET_H     1

#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC) /* Oracle Solaris Studio */
#  if !defined(__LP64) && (defined(__ILP32) ||                          \
                           defined(__i386) ||                           \
                           defined(__sparcv8) ||                        \
                           defined(__sparcv8plus))
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__LP64) || \
        defined(__amd64) || defined(__sparcv9)
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#  define CURL_PULL_SYS_TYPES_H      1
#  define CURL_PULL_SYS_SOCKET_H     1

#elif defined(__xlc__) /* IBM xlc compiler */
#  if !defined(_LP64)
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  else
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#  define CURL_PULL_SYS_TYPES_H      1
#  define CURL_PULL_SYS_SOCKET_H     1

/* ===================================== */
/*    KEEP MSVC THE PENULTIMATE ENTRY    */
/* ===================================== */

#elif defined(_MSC_VER)
#  if (_MSC_VER >= 900) && (_INTEGRAL_MAX_BITS >= 64)
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  else
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int

/* ===================================== */
/*    KEEP GENERIC GCC THE LAST ENTRY    */
/* ===================================== */

#elif defined(__GNUC__) && !defined(_SCO_DS)
#  if !defined(__LP64__) &&                                             \
  (defined(__ILP32__) || defined(__i386__) || defined(__hppa__) ||      \
   defined(__ppc__) || defined(__powerpc__) || defined(__arm__) ||      \
   defined(__sparc__) || defined(__mips__) || defined(__sh__) ||        \
   defined(__XTENSA__) ||                                               \
   (defined(__SIZEOF_LONG__) && __SIZEOF_LONG__ == 4)  ||               \
   (defined(__LONG_MAX__) && __LONG_MAX__ == 2147483647L))
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__LP64__) || \
        defined(__x86_64__) || defined(__ppc64__) || defined(__sparc64__) || \
        defined(__e2k__) || \
        (defined(__SIZEOF_LONG__) && __SIZEOF_LONG__ == 8) || \
        (defined(__LONG_MAX__) && __LONG_MAX__ == 9223372036854775807L)
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#  define CURL_PULL_SYS_TYPES_H      1
#  define CURL_PULL_SYS_SOCKET_H     1

#else
/* generic "safe guess" on old 32 bit style */
# define CURL_TYPEOF_CURL_OFF_T     long
# define CURL_FORMAT_CURL_OFF_T     "ld"
# define CURL_FORMAT_CURL_OFF_TU    "lu"
# define CURL_SUFFIX_CURL_OFF_T     L
# define CURL_SUFFIX_CURL_OFF_TU    UL
# define CURL_TYPEOF_CURL_SOCKLEN_T int
#endif

#ifdef _AIX
/* AIX needs <sys/poll.h> */
#define CURL_PULL_SYS_POLL_H
#endif


/* CURL_PULL_WS2TCPIP_H is defined above when inclusion of header file  */
/* ws2tcpip.h is required here to properly make type definitions below. */
#ifdef CURL_PULL_WS2TCPIP_H
#  include <winsock2.h>
#  include <windows.h>
#  include <ws2tcpip.h>
#endif

/* CURL_PULL_SYS_TYPES_H is defined above when inclusion of header file  */
/* sys/types.h is required here to properly make type definitions below. */
#ifdef CURL_PULL_SYS_TYPES_H
#  include <sys/types.h>
#endif

/* CURL_PULL_SYS_SOCKET_H is defined above when inclusion of header file  */
/* sys/socket.h is required here to properly make type definitions below. */
#ifdef CURL_PULL_SYS_SOCKET_H
#  include <sys/socket.h>
#endif

/* CURL_PULL_SYS_POLL_H is defined above when inclusion of header file    */
/* sys/poll.h is required here to properly make type definitions below.   */
#ifdef CURL_PULL_SYS_POLL_H
#  include <sys/poll.h>
#endif

/* Data type definition of curl_socklen_t. */
#ifdef CURL_TYPEOF_CURL_SOCKLEN_T
  typedef CURL_TYPEOF_CURL_SOCKLEN_T curl_socklen_t;
#endif

/* Data type definition of curl_off_t. */

#ifdef CURL_TYPEOF_CURL_OFF_T
  typedef CURL_TYPEOF_CURL_OFF_T curl_off_t;
#endif

/*
 * CURL_ISOCPP and CURL_OFF_T_C definitions are done here in order to allow
 * these to be visible and exported by the external libcurl interface API,
 * while also making them visible to the library internals, simply including
 * curl_setup.h, without actually needing to include curl.h internally.
 * If some day this section would grow big enough, all this should be moved
 * to its own header file.
 */

/*
 * Figure out if we can use the ## preprocessor operator, which is supported
 * by ISO/ANSI C and C++. Some compilers support it without setting __STDC__
 * or  __cplusplus so we need to carefully check for them too.
 */

#if defined(__STDC__) || defined(_MSC_VER) || defined(__cplusplus) || \
  defined(__HP_aCC) || defined(__BORLANDC__) || defined(__LCC__) || \
  defined(__POCC__) || defined(__SALFORDC__) || defined(__HIGHC__) || \
  defined(__ILEC400__)
  /* This compiler is believed to have an ISO compatible preprocessor */
#define CURL_ISOCPP
#else
  /* This compiler is believed NOT to have an ISO compatible preprocessor */
#undef CURL_ISOCPP
#endif

/*
 * Macros for minimum-width signed and unsigned curl_off_t integer constants.
 */

#if defined(__BORLANDC__) && (__BORLANDC__ == 0x0551)
#  define CURLINC_OFF_T_C_HLPR2(x) x
#  define CURLINC_OFF_T_C_HLPR1(x) CURLINC_OFF_T_C_HLPR2(x)
#  define CURL_OFF_T_C(Val)  CURLINC_OFF_T_C_HLPR1(Val) ## \
                             CURLINC_OFF_T_C_HLPR1(CURL_SUFFIX_CURL_OFF_T)
#  define CURL_OFF_TU_C(Val) CURLINC_OFF_T_C_HLPR1(Val) ## \
                             CURLINC_OFF_T_C_HLPR1(CURL_SUFFIX_CURL_OFF_TU)
#else
#  ifdef CURL_ISOCPP
#    define CURLINC_OFF_T_C_HLPR2(Val,Suffix) Val ## Suffix
#  else
#    define CURLINC_OFF_T_C_HLPR2(Val,Suffix) Val/**/Suffix
#  endif
#  define CURLINC_OFF_T_C_HLPR1(Val,Suffix) CURLINC_OFF_T_C_HLPR2(Val,Suffix)
#  define CURL_OFF_T_C(Val)  CURLINC_OFF_T_C_HLPR1(Val,CURL_SUFFIX_CURL_OFF_T)
#  define CURL_OFF_TU_C(Val) CURLINC_OFF_T_C_HLPR1(Val,CURL_SUFFIX_CURL_OFF_TU)
#endif

#endif /* CURLINC_SYSTEM_H */
