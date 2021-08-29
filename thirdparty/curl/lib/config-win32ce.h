#ifndef HEADER_CURL_CONFIG_WIN32CE_H
#define HEADER_CURL_CONFIG_WIN32CE_H
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
/*  lib/config-win32ce.h - Hand crafted config file for windows ce  */
/* ================================================================ */

/* ---------------------------------------------------------------- */
/*                          HEADER FILES                            */
/* ---------------------------------------------------------------- */

/* Define if you have the <arpa/inet.h> header file.  */
/* #define HAVE_ARPA_INET_H 1 */

/* Define if you have the <assert.h> header file.  */
/* #define HAVE_ASSERT_H 1 */

/* Define if you have the <errno.h> header file.  */
/* #define HAVE_ERRNO_H 1 */

/* Define if you have the <fcntl.h> header file.  */
#define HAVE_FCNTL_H 1

/* Define if you have the <getopt.h> header file.  */
/* #define HAVE_GETOPT_H 1 */

/* Define if you have the <io.h> header file.  */
#define HAVE_IO_H 1

/* Define if you need the malloc.h header file even with stdlib.h  */
#define NEED_MALLOC_H 1

/* Define if you have the <netdb.h> header file.  */
/* #define HAVE_NETDB_H 1 */

/* Define if you have the <netinet/in.h> header file.  */
/* #define HAVE_NETINET_IN_H 1 */

/* Define if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define if you have the <ssl.h> header file.  */
/* #define HAVE_SSL_H 1 */

/* Define if you have the <stdlib.h> header file.  */
#define HAVE_STDLIB_H 1

/* Define if you have the <process.h> header file.  */
/* #define HAVE_PROCESS_H 1 */

/* Define if you have the <sys/param.h> header file.  */
/* #define HAVE_SYS_PARAM_H 1 */

/* Define if you have the <sys/select.h> header file.  */
/* #define HAVE_SYS_SELECT_H 1 */

/* Define if you have the <sys/socket.h> header file.  */
/* #define HAVE_SYS_SOCKET_H 1 */

/* Define if you have the <sys/sockio.h> header file.  */
/* #define HAVE_SYS_SOCKIO_H 1 */

/* Define if you have the <sys/stat.h> header file.  */
#define HAVE_SYS_STAT_H 1

/* Define if you have the <sys/time.h> header file */
/* #define HAVE_SYS_TIME_H 1 */

/* Define if you have the <sys/types.h> header file.  */
/* #define HAVE_SYS_TYPES_H 1 */

/* Define if you have the <sys/utime.h> header file */
#define HAVE_SYS_UTIME_H 1

/* Define if you have the <termio.h> header file.  */
/* #define HAVE_TERMIO_H 1 */

/* Define if you have the <termios.h> header file.  */
/* #define HAVE_TERMIOS_H 1 */

/* Define if you have the <time.h> header file.  */
#define HAVE_TIME_H 1

/* Define if you have the <unistd.h> header file.  */
#if defined(__MINGW32__) || defined(__WATCOMC__) || defined(__LCC__)
#define HAVE_UNISTD_H 1
#endif

/* Define if you have the <windows.h> header file.  */
#define HAVE_WINDOWS_H 1

/* Define if you have the <winsock2.h> header file.  */
#define HAVE_WINSOCK2_H 1

/* Define if you have the <ws2tcpip.h> header file.  */
#define HAVE_WS2TCPIP_H 1

/* ---------------------------------------------------------------- */
/*                        OTHER HEADER INFO                         */
/* ---------------------------------------------------------------- */

/* Define if you have the ANSI C header files.  */
#define STDC_HEADERS 1

/* Define if you can safely include both <sys/time.h> and <time.h>.  */
/* #define TIME_WITH_SYS_TIME 1 */

/* ---------------------------------------------------------------- */
/*                             FUNCTIONS                            */
/* ---------------------------------------------------------------- */

/* Define if you have the closesocket function.  */
#define HAVE_CLOSESOCKET 1

/* Define if you don't have vprintf but do have _doprnt.  */
/* #define HAVE_DOPRNT 1 */

/* Define if you have the gethostname function.  */
#define HAVE_GETHOSTNAME 1

/* Define if you have the getpass function.  */
/* #define HAVE_GETPASS 1 */

/* Define if you have the getservbyname function.  */
#define HAVE_GETSERVBYNAME 1

/* Define if you have the gettimeofday function.  */
/*  #define HAVE_GETTIMEOFDAY 1 */

/* Define if you have the inet_addr function.  */
#define HAVE_INET_ADDR 1

/* Define if you have the ioctlsocket function. */
#define HAVE_IOCTLSOCKET 1

/* Define if you have a working ioctlsocket FIONBIO function. */
#define HAVE_IOCTLSOCKET_FIONBIO 1

/* Define if you have the RAND_screen function when using SSL  */
#define HAVE_RAND_SCREEN 1

/* Define if you have the `RAND_status' function when using SSL. */
#define HAVE_RAND_STATUS 1

/* Define if you have the select function.  */
#define HAVE_SELECT 1

/* Define if you have the setvbuf function.  */
#define HAVE_SETVBUF 1

/* Define if you have the socket function.  */
#define HAVE_SOCKET 1

/* Define if you have the strcasecmp function.  */
/* #define HAVE_STRCASECMP 1 */

/* Define if you have the strdup function.  */
/* #define HAVE_STRDUP 1 */

/* Define if you have the strftime function.  */
/* #define HAVE_STRFTIME 1 */

/* Define if you have the stricmp function. */
/* #define HAVE_STRICMP 1 */

/* Define if you have the strnicmp function. */
/* #define HAVE_STRNICMP 1 */

/* Define if you have the strstr function.  */
#define HAVE_STRSTR 1

/* Define if you have the strtoll function.  */
#if defined(__MINGW32__) || defined(__WATCOMC__)
#define HAVE_STRTOLL 1
#endif

/* Define if you have the utime function */
#define HAVE_UTIME 1

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

/* Define this if in_addr_t is not an available 'typedefed' type */
#define in_addr_t unsigned long

/* Define ssize_t if it is not an available 'typedefed' type */
#if (defined(__WATCOMC__) && (__WATCOMC__ >= 1240)) || defined(__POCC__)
#elif defined(_WIN64)
#define ssize_t __int64
#else
#define ssize_t int
#endif

/* ---------------------------------------------------------------- */
/*                            TYPE SIZES                            */
/* ---------------------------------------------------------------- */

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `long double', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE 16

/* The size of `long long', as computed by sizeof. */
/* #define SIZEOF_LONG_LONG 8 */

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* Define to the size of `long', as computed by sizeof. */
#define SIZEOF_LONG 4

/* The size of `size_t', as computed by sizeof. */
#if defined(_WIN64)
#  define SIZEOF_SIZE_T 8
#else
#  define SIZEOF_SIZE_T 4
#endif

/* ---------------------------------------------------------------- */
/*                          STRUCT RELATED                          */
/* ---------------------------------------------------------------- */

/* Define this if you have struct sockaddr_storage */
/* #define HAVE_STRUCT_SOCKADDR_STORAGE 1 */

/* Define this if you have struct timeval */
#define HAVE_STRUCT_TIMEVAL 1

/* Define this if struct sockaddr_in6 has the sin6_scope_id member */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* ---------------------------------------------------------------- */
/*                        COMPILER SPECIFIC                         */
/* ---------------------------------------------------------------- */

/* Undef keyword 'const' if it does not work.  */
/* #undef const */

/* Define to avoid VS2005 complaining about portable C functions */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define _CRT_SECURE_NO_DEPRECATE 1
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif

/* VS2005 and later default size for time_t is 64-bit, unless */
/* _USE_32BIT_TIME_T has been defined to get a 32-bit time_t. */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#  ifndef _USE_32BIT_TIME_T
#    define SIZEOF_TIME_T 8
#  else
#    define SIZEOF_TIME_T 4
#  endif
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

#if !defined(USE_WIN32_LARGE_FILES) && !defined(USE_WIN32_SMALL_FILES)
#  define USE_WIN32_SMALL_FILES
#endif

/* ---------------------------------------------------------------- */
/*                           LDAP SUPPORT                           */
/* ---------------------------------------------------------------- */

#define USE_WIN32_LDAP 1
#undef HAVE_LDAP_URL_PARSE

/* ---------------------------------------------------------------- */
/*                       ADDITIONAL DEFINITIONS                     */
/* ---------------------------------------------------------------- */

/* Define cpu-machine-OS */
#undef OS
#define OS "i386-pc-win32ce"

/* Name of package */
#define PACKAGE "curl"

/* ---------------------------------------------------------------- */
/*                       WinCE                                      */
/* ---------------------------------------------------------------- */

#ifndef UNICODE
#  define UNICODE
#endif

#ifndef _UNICODE
#  define _UNICODE
#endif

#define CURL_DISABLE_FILE 1
#define CURL_DISABLE_TELNET 1
#define CURL_DISABLE_LDAP 1

#define ENOSPC 1
#define ENOMEM 2
#define EAGAIN 3

extern int stat(const char *path, struct stat *buffer);

#endif /* HEADER_CURL_CONFIG_WIN32CE_H */
