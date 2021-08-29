#ifndef HEADER_CURL_CONFIG_WIN32_H
#define HEADER_CURL_CONFIG_WIN32_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

/* ================================================================ */
/*               Hand crafted config file for Windows               */
/* ================================================================ */

/* ---------------------------------------------------------------- */
/*                          HEADER FILES                            */
/* ---------------------------------------------------------------- */

/* Define if you have the <arpa/inet.h> header file. */
/* #define HAVE_ARPA_INET_H 1 */

/* Define if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define if you have the <getopt.h> header file. */
#if defined(__MINGW32__) || defined(__POCC__)
#define HAVE_GETOPT_H 1
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
#define HAVE_INTTYPES_H 1
#endif

/* Define if you have the <io.h> header file. */
#define HAVE_IO_H 1

/* Define if you have the <locale.h> header file. */
#define HAVE_LOCALE_H 1

/* Define if you need <malloc.h> header even with <stdlib.h> header file. */
#if !defined(__SALFORDC__) && !defined(__POCC__)
#define NEED_MALLOC_H 1
#endif

/* Define if you have the <netdb.h> header file. */
/* #define HAVE_NETDB_H 1 */

/* Define if you have the <netinet/in.h> header file. */
/* #define HAVE_NETINET_IN_H 1 */

/* Define if you have the <process.h> header file. */
#ifndef __SALFORDC__
#define HAVE_PROCESS_H 1
#endif

/* Define if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define if you have the <ssl.h> header file. */
/* #define HAVE_SSL_H 1 */

/* Define to 1 if you have the <stdbool.h> header file. */
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
#define HAVE_STDBOOL_H 1
#endif

/* Define if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define if you have the <sys/param.h> header file. */
/* #define HAVE_SYS_PARAM_H 1 */

/* Define if you have the <sys/select.h> header file. */
/* #define HAVE_SYS_SELECT_H 1 */

/* Define if you have the <sys/socket.h> header file. */
/* #define HAVE_SYS_SOCKET_H 1 */

/* Define if you have the <sys/sockio.h> header file. */
/* #define HAVE_SYS_SOCKIO_H 1 */

/* Define if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define if you have the <sys/time.h> header file. */
/* #define HAVE_SYS_TIME_H 1 */

/* Define if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define if you have the <sys/utime.h> header file. */
#ifndef __BORLANDC__
#define HAVE_SYS_UTIME_H 1
#endif

/* Define if you have the <termio.h> header file. */
/* #define HAVE_TERMIO_H 1 */

/* Define if you have the <termios.h> header file. */
/* #define HAVE_TERMIOS_H 1 */

/* Define if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define if you have the <unistd.h> header file. */
#if defined(__MINGW32__) || defined(__WATCOMC__) || defined(__LCC__) || \
    defined(__POCC__)
#define HAVE_UNISTD_H 1
#endif

/* Define if you have the <windows.h> header file. */
#define HAVE_WINDOWS_H 1

/* Define if you have the <winsock2.h> header file. */
#ifndef __SALFORDC__
#define HAVE_WINSOCK2_H 1
#endif

/* Define if you have the <ws2tcpip.h> header file. */
#ifndef __SALFORDC__
#define HAVE_WS2TCPIP_H 1
#endif

/* ---------------------------------------------------------------- */
/*                        OTHER HEADER INFO                         */
/* ---------------------------------------------------------------- */

/* Define if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define if you can safely include both <sys/time.h> and <time.h>. */
/* #define TIME_WITH_SYS_TIME 1 */

/* Define to 1 if bool is an available type. */
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
#define HAVE_BOOL_T 1
#endif

/* ---------------------------------------------------------------- */
/*                             FUNCTIONS                            */
/* ---------------------------------------------------------------- */

/* Define if you have the closesocket function. */
#define HAVE_CLOSESOCKET 1

/* Define if you don't have vprintf but do have _doprnt. */
/* #define HAVE_DOPRNT 1 */

/* Define if you have the ftruncate function. */
/* #define HAVE_FTRUNCATE 1 */

/* Define to 1 if you have the `getpeername' function. */
#define HAVE_GETPEERNAME 1

/* Define to 1 if you have the getsockname function. */
#define HAVE_GETSOCKNAME 1

/* Define if you have the gethostname function. */
#define HAVE_GETHOSTNAME 1

/* Define if you have the getpass function. */
/* #define HAVE_GETPASS 1 */

/* Define if you have the getservbyname function. */
#define HAVE_GETSERVBYNAME 1

/* Define if you have the getprotobyname function. */
#define HAVE_GETPROTOBYNAME

/* Define if you have the gettimeofday function. */
/* #define HAVE_GETTIMEOFDAY 1 */

/* Define if you have the inet_addr function. */
#define HAVE_INET_ADDR 1

/* Define if you have the ioctlsocket function. */
#define HAVE_IOCTLSOCKET 1

/* Define if you have a working ioctlsocket FIONBIO function. */
#define HAVE_IOCTLSOCKET_FIONBIO 1

/* Define if you have the RAND_screen function when using SSL. */
#define HAVE_RAND_SCREEN 1

/* Define if you have the `RAND_status' function when using SSL. */
#define HAVE_RAND_STATUS 1

/* Define if you have the `CRYPTO_cleanup_all_ex_data' function.
   This is present in OpenSSL versions after 0.9.6b */
#define HAVE_CRYPTO_CLEANUP_ALL_EX_DATA 1

/* Define if you have the select function. */
#define HAVE_SELECT 1

/* Define if you have the setlocale function. */
#define HAVE_SETLOCALE 1

/* Define if you have the setmode function. */
#define HAVE_SETMODE 1

/* Define if you have the setvbuf function. */
#define HAVE_SETVBUF 1

/* Define if you have the socket function. */
#define HAVE_SOCKET 1

/* Define if you have the strcasecmp function. */
#ifdef __MINGW32__
#define HAVE_STRCASECMP 1
#endif

/* Define if you have the strdup function. */
#define HAVE_STRDUP 1

/* Define if you have the strftime function. */
#define HAVE_STRFTIME 1

/* Define if you have the stricmp function. */
#define HAVE_STRICMP 1

/* Define if you have the strnicmp function. */
#define HAVE_STRNICMP 1

/* Define if you have the strstr function. */
#define HAVE_STRSTR 1

/* Define if you have the strtoll function. */
#if defined(__MINGW32__) || defined(__WATCOMC__) || defined(__POCC__) || \
    (defined(_MSC_VER) && (_MSC_VER >= 1800))
#define HAVE_STRTOLL 1
#endif

/* Define if you have the utime function. */
#ifndef __BORLANDC__
#define HAVE_UTIME 1
#endif

/* Define if you have the recv function. */
#define HAVE_RECV 1

/* Define to the type of arg 1 for recv. */
#define RECV_TYPE_ARG1 SOCKET

/* Define to the type of arg 2 for recv. */
#define RECV_TYPE_ARG2 char *

/* Define to the type of arg 3 for recv. */
#define RECV_TYPE_ARG3 int

/* Define to the type of arg 4 for recv. */
#define RECV_TYPE_ARG4 int

/* Define to the function return type for recv. */
#define RECV_TYPE_RETV int

/* Define if you have the recvfrom function. */
#define HAVE_RECVFROM 1

/* Define to the type of arg 1 for recvfrom. */
#define RECVFROM_TYPE_ARG1 SOCKET

/* Define to the type pointed by arg 2 for recvfrom. */
#define RECVFROM_TYPE_ARG2 char

/* Define to the type of arg 3 for recvfrom. */
#define RECVFROM_TYPE_ARG3 int

/* Define to the type of arg 4 for recvfrom. */
#define RECVFROM_TYPE_ARG4 int

/* Define to the type pointed by arg 5 for recvfrom. */
#define RECVFROM_TYPE_ARG5 struct sockaddr

/* Define to the type pointed by arg 6 for recvfrom. */
#define RECVFROM_TYPE_ARG6 int

/* Define to the function return type for recvfrom. */
#define RECVFROM_TYPE_RETV int

/* Define if you have the send function. */
#define HAVE_SEND 1

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 SOCKET

/* Define to the type qualifier of arg 2 for send. */
#define SEND_QUAL_ARG2 const

/* Define to the type of arg 2 for send. */
#define SEND_TYPE_ARG2 char *

/* Define to the type of arg 3 for send. */
#define SEND_TYPE_ARG3 int

/* Define to the type of arg 4 for send. */
#define SEND_TYPE_ARG4 int

/* Define to the function return type for send. */
#define SEND_TYPE_RETV int

/* ---------------------------------------------------------------- */
/*                       TYPEDEF REPLACEMENTS                       */
/* ---------------------------------------------------------------- */

/* Define if in_addr_t is not an available 'typedefed' type. */
#define in_addr_t unsigned long

/* Define if ssize_t is not an available 'typedefed' type. */
#ifndef _SSIZE_T_DEFINED
#  if (defined(__WATCOMC__) && (__WATCOMC__ >= 1240)) || \
      defined(__POCC__) || \
      defined(__MINGW32__)
#  elif defined(_WIN64)
#    define _SSIZE_T_DEFINED
#    define ssize_t __int64
#  else
#    define _SSIZE_T_DEFINED
#    define ssize_t int
#  endif
#endif

/* ---------------------------------------------------------------- */
/*                            TYPE SIZES                            */
/* ---------------------------------------------------------------- */

/* Define to the size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* Define to the size of `long double', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE 16

/* Define to the size of `long long', as computed by sizeof. */
/* #define SIZEOF_LONG_LONG 8 */

/* Define to the size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* Define to the size of `long', as computed by sizeof. */
#define SIZEOF_LONG 4

/* Define to the size of `size_t', as computed by sizeof. */
#if defined(_WIN64)
#  define SIZEOF_SIZE_T 8
#else
#  define SIZEOF_SIZE_T 4
#endif

/* Define to the size of `curl_off_t', as computed by sizeof. */
#define SIZEOF_CURL_OFF_T 8

/* ---------------------------------------------------------------- */
/*               BSD-style lwIP TCP/IP stack SPECIFIC               */
/* ---------------------------------------------------------------- */

/* Define to use BSD-style lwIP TCP/IP stack. */
/* #define USE_LWIPSOCK 1 */

#ifdef USE_LWIPSOCK
#  undef USE_WINSOCK
#  undef HAVE_WINSOCK2_H
#  undef HAVE_WS2TCPIP_H
#  undef HAVE_ERRNO_H
#  undef HAVE_GETHOSTNAME
#  undef LWIP_POSIX_SOCKETS_IO_NAMES
#  undef RECV_TYPE_ARG1
#  undef RECV_TYPE_ARG3
#  undef SEND_TYPE_ARG1
#  undef SEND_TYPE_ARG3
#  define HAVE_FREEADDRINFO
#  define HAVE_GETADDRINFO
#  define HAVE_GETHOSTBYNAME
#  define HAVE_GETHOSTBYNAME_R
#  define HAVE_GETHOSTBYNAME_R_6
#  define LWIP_POSIX_SOCKETS_IO_NAMES 0
#  define RECV_TYPE_ARG1 int
#  define RECV_TYPE_ARG3 size_t
#  define SEND_TYPE_ARG1 int
#  define SEND_TYPE_ARG3 size_t
#endif

/* ---------------------------------------------------------------- */
/*                        Watt-32 tcp/ip SPECIFIC                   */
/* ---------------------------------------------------------------- */

#ifdef USE_WATT32
  #include <tcp.h>
  #undef byte
  #undef word
  #undef USE_WINSOCK
  #undef HAVE_WINSOCK2_H
  #undef HAVE_WS2TCPIP_H
  #define HAVE_GETADDRINFO
  #define HAVE_SYS_IOCTL_H
  #define HAVE_SYS_SOCKET_H
  #define HAVE_NETINET_IN_H
  #define HAVE_NETDB_H
  #define HAVE_ARPA_INET_H
  #define HAVE_FREEADDRINFO
  #define SOCKET int
#endif


/* ---------------------------------------------------------------- */
/*                        COMPILER SPECIFIC                         */
/* ---------------------------------------------------------------- */

/* Define to nothing if compiler does not support 'const' qualifier. */
/* #define const */

/* Define to nothing if compiler does not support 'volatile' qualifier. */
/* #define volatile */

/* Windows should not have HAVE_GMTIME_R defined */
/* #undef HAVE_GMTIME_R */

/* Define if the compiler supports C99 variadic macro style. */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define HAVE_VARIADIC_MACROS_C99 1
#endif

/* Define if the compiler supports the 'long long' data type. */
#if defined(__MINGW32__) || defined(__WATCOMC__)      || \
    (defined(_MSC_VER)     && (_MSC_VER     >= 1310)) || \
    (defined(__BORLANDC__) && (__BORLANDC__ >= 0x561))
#define HAVE_LONGLONG 1
#endif

/* Define to avoid VS2005 complaining about portable C functions. */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define _CRT_SECURE_NO_DEPRECATE 1
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif

/* mingw-w64, mingw using >= MSVCR80, and visual studio >= 2005 (MSVCR80)
   all default to 64-bit time_t unless _USE_32BIT_TIME_T is defined */
#ifdef __MINGW32__
#  include <_mingw.h>
#endif
#if defined(__MINGW64_VERSION_MAJOR) || \
    (defined(__MINGW32__) && (__MSVCRT_VERSION__ >= 0x0800)) || \
    (defined(_MSC_VER) && (_MSC_VER >= 1400))
#  ifndef _USE_32BIT_TIME_T
#    define SIZEOF_TIME_T 8
#  else
#    define SIZEOF_TIME_T 4
#  endif
#endif

/* Define some minimum and default build targets for Visual Studio */
#if defined(_MSC_VER)
   /* Officially, Microsoft's Windows SDK versions 6.X does not support Windows
      2000 as a supported build target. VS2008 default installations provides
      an embedded Windows SDK v6.0A along with the claim that Windows 2000 is a
      valid build target for VS2008. Popular belief is that binaries built with
      VS2008 using Windows SDK versions v6.X and Windows 2000 as a build target
      are functional. */
#  define VS2008_MIN_TARGET 0x0500

   /* The minimum build target for VS2012 is Vista unless Update 1 is installed
      and the v110_xp toolset is chosen. */
#  if defined(_USING_V110_SDK71_)
#    define VS2012_MIN_TARGET 0x0501
#  else
#    define VS2012_MIN_TARGET 0x0600
#  endif

   /* VS2008 default build target is Windows Vista. We override default target
      to be Windows XP. */
#  define VS2008_DEF_TARGET 0x0501

   /* VS2012 default build target is Windows Vista unless Update 1 is installed
      and the v110_xp toolset is chosen. */
#  if defined(_USING_V110_SDK71_)
#    define VS2012_DEF_TARGET 0x0501
#  else
#    define VS2012_DEF_TARGET 0x0600
#  endif
#endif

/* VS2008 default target settings and minimum build target check. */
#if defined(_MSC_VER) && (_MSC_VER >= 1500) && (_MSC_VER <= 1600)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT VS2008_DEF_TARGET
#  endif
#  ifndef WINVER
#    define WINVER VS2008_DEF_TARGET
#  endif
#  if (_WIN32_WINNT < VS2008_MIN_TARGET) || (WINVER < VS2008_MIN_TARGET)
#    error VS2008 does not support Windows build targets prior to Windows 2000
#  endif
#endif

/* VS2012 default target settings and minimum build target check. */
#if defined(_MSC_VER) && (_MSC_VER >= 1700)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT VS2012_DEF_TARGET
#  endif
#  ifndef WINVER
#    define WINVER VS2012_DEF_TARGET
#  endif
#  if (_WIN32_WINNT < VS2012_MIN_TARGET) || (WINVER < VS2012_MIN_TARGET)
#    if defined(_USING_V110_SDK71_)
#      error VS2012 does not support Windows build targets prior to Windows XP
#    else
#      error VS2012 does not support Windows build targets prior to Windows \
Vista
#    endif
#  endif
#endif

/* When no build target is specified Pelles C 5.00 and later default build
   target is Windows Vista. We override default target to be Windows 2000. */
#if defined(__POCC__) && (__POCC__ >= 500)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0500
#  endif
#  ifndef WINVER
#    define WINVER 0x0500
#  endif
#endif

/* Availability of freeaddrinfo, getaddrinfo, and if_nametoindex
   functions is quite convoluted, compiler dependent and even build target
   dependent. */
#if defined(HAVE_WS2TCPIP_H)
#  if defined(__POCC__)
#    define HAVE_FREEADDRINFO           1
#    define HAVE_GETADDRINFO            1
#    define HAVE_GETADDRINFO_THREADSAFE 1
#  elif defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0501)
#    define HAVE_FREEADDRINFO           1
#    define HAVE_GETADDRINFO            1
#    define HAVE_GETADDRINFO_THREADSAFE 1
#  elif defined(_MSC_VER) && (_MSC_VER >= 1200)
#    define HAVE_FREEADDRINFO           1
#    define HAVE_GETADDRINFO            1
#    define HAVE_GETADDRINFO_THREADSAFE 1
#  endif
#endif

#if defined(__POCC__)
#  ifndef _MSC_VER
#    error Microsoft extensions /Ze compiler option is required
#  endif
#  ifndef __POCC__OLDNAMES
#    error Compatibility names /Go compiler option is required
#  endif
#endif

/* ---------------------------------------------------------------- */
/*                          STRUCT RELATED                          */
/* ---------------------------------------------------------------- */

/* Define if you have struct sockaddr_storage. */
#if !defined(__SALFORDC__) && !defined(__BORLANDC__)
#define HAVE_STRUCT_SOCKADDR_STORAGE 1
#endif

/* Define if you have struct timeval. */
#define HAVE_STRUCT_TIMEVAL 1

/* Define if struct sockaddr_in6 has the sin6_scope_id member. */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

#if defined(HAVE_WINSOCK2_H) && defined(_WIN32_WINNT) && \
    (_WIN32_WINNT >= 0x0600)
#define HAVE_STRUCT_POLLFD 1
#endif

/* ---------------------------------------------------------------- */
/*                        LARGE FILE SUPPORT                        */
/* ---------------------------------------------------------------- */

#if defined(_MSC_VER) && !defined(_WIN32_WCE)
#  if (_MSC_VER >= 900) && (_INTEGRAL_MAX_BITS >= 64)
#    define USE_WIN32_LARGE_FILES
#  else
#    define USE_WIN32_SMALL_FILES
#  endif
#endif

#if defined(__MINGW32__) && !defined(USE_WIN32_LARGE_FILES)
#  define USE_WIN32_LARGE_FILES
#endif

#if defined(__WATCOMC__) && !defined(USE_WIN32_LARGE_FILES)
#  define USE_WIN32_LARGE_FILES
#endif

#if defined(__POCC__)
#  undef USE_WIN32_LARGE_FILES
#endif

#if !defined(USE_WIN32_LARGE_FILES) && !defined(USE_WIN32_SMALL_FILES)
#  define USE_WIN32_SMALL_FILES
#endif

/* ---------------------------------------------------------------- */
/*                       DNS RESOLVER SPECIALTY                     */
/* ---------------------------------------------------------------- */

/*
 * Undefine both USE_ARES and USE_THREADS_WIN32 for synchronous DNS.
 */

/* Define to enable c-ares asynchronous DNS lookups. */
/* #define USE_ARES 1 */

/* Default define to enable threaded asynchronous DNS lookups. */
#if !defined(USE_SYNC_DNS) && !defined(USE_ARES) && \
    !defined(USE_THREADS_WIN32)
#  define USE_THREADS_WIN32 1
#endif

#if defined(USE_ARES) && defined(USE_THREADS_WIN32)
#  error "Only one DNS lookup specialty may be defined at most"
#endif

/* ---------------------------------------------------------------- */
/*                           LDAP SUPPORT                           */
/* ---------------------------------------------------------------- */

#if defined(CURL_HAS_NOVELL_LDAPSDK) || defined(CURL_HAS_MOZILLA_LDAPSDK)
#undef USE_WIN32_LDAP
#define HAVE_LDAP_SSL_H 1
#define HAVE_LDAP_URL_PARSE 1
#elif defined(CURL_HAS_OPENLDAP_LDAPSDK)
#undef USE_WIN32_LDAP
#define HAVE_LDAP_URL_PARSE 1
#else
#undef HAVE_LDAP_URL_PARSE
#define HAVE_LDAP_SSL 1
#define USE_WIN32_LDAP 1
#endif

#if defined(__WATCOMC__) && defined(USE_WIN32_LDAP)
#if __WATCOMC__ < 1280
#define WINBERAPI  __declspec(cdecl)
#define WINLDAPAPI __declspec(cdecl)
#endif
#endif

#if defined(__POCC__) && defined(USE_WIN32_LDAP)
#  define CURL_DISABLE_LDAP 1
#endif

/* Define to use the Windows crypto library. */
#if !defined(CURL_WINDOWS_APP)
#define USE_WIN32_CRYPTO
#endif

/* Define to use Unix sockets. */
#define USE_UNIX_SOCKETS

/* ---------------------------------------------------------------- */
/*                       ADDITIONAL DEFINITIONS                     */
/* ---------------------------------------------------------------- */

/* Define cpu-machine-OS */
#undef OS
#if defined(_M_IX86) || defined(__i386__) /* x86 (MSVC or gcc) */
#define OS "i386-pc-win32"
#elif defined(_M_X64) || defined(__x86_64__) /* x86_64 (MSVC >=2005 or gcc) */
#define OS "x86_64-pc-win32"
#elif defined(_M_IA64) || defined(__ia64__) /* Itanium */
#define OS "ia64-pc-win32"
#elif defined(_M_ARM_NT) || defined(__arm__) /* ARMv7-Thumb2 (Windows RT) */
#define OS "thumbv7a-pc-win32"
#elif defined(_M_ARM64) || defined(__aarch64__) /* ARM64 (Windows 10) */
#define OS "aarch64-pc-win32"
#else
#define OS "unknown-pc-win32"
#endif

/* Name of package */
#define PACKAGE "curl"

/* If you want to build curl with the built-in manual */
#define USE_MANUAL 1

#if defined(__POCC__) || defined(USE_IPV6)
#  define ENABLE_IPV6 1
#endif

#endif /* HEADER_CURL_CONFIG_WIN32_H */
