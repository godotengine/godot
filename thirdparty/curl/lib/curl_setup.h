#ifndef HEADER_CURL_SETUP_H
#define HEADER_CURL_SETUP_H
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

#if defined(BUILDING_LIBCURL) && !defined(CURL_NO_OLDIES)
#define CURL_NO_OLDIES
#endif

/*
 * Disable Visual Studio warnings:
 * 4127 "conditional expression is constant"
 */
#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

/*
 * Define WIN32 when build target is Win32 API
 */

#if (defined(_WIN32) || defined(__WIN32__)) && !defined(WIN32)
#define WIN32
#endif

#ifdef WIN32
/*
 * Don't include unneeded stuff in Windows headers to avoid compiler
 * warnings and macro clashes.
 * Make sure to define this macro before including any Windows headers.
 */
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOGDI
#    define NOGDI
#  endif
/* Detect Windows App environment which has a restricted access
 * to the Win32 APIs. */
# if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0602)) || \
  defined(WINAPI_FAMILY)
#  include <winapifamily.h>
#  if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) &&  \
     !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#    define CURL_WINDOWS_APP
#  endif
# endif
#endif

/*
 * Include configuration script results or hand-crafted
 * configuration file for platforms which lack config tool.
 */

#ifdef HAVE_CONFIG_H

#include "curl_config.h"

#else /* HAVE_CONFIG_H */

#ifdef _WIN32_WCE
#  include "config-win32ce.h"
#else
#  ifdef WIN32
#    include "config-win32.h"
#  endif
#endif

#if defined(macintosh) && defined(__MRC__)
#  include "config-mac.h"
#endif

#ifdef __riscos__
#  include "config-riscos.h"
#endif

#ifdef __AMIGA__
#  include "config-amigaos.h"
#endif

#ifdef __OS400__
#  include "config-os400.h"
#endif

#ifdef TPF
#  include "config-tpf.h"
#endif

#ifdef __VXWORKS__
#  include "config-vxworks.h"
#endif

#ifdef __PLAN9__
#  include "config-plan9.h"
#endif

#endif /* HAVE_CONFIG_H */

/* ================================================================ */
/* Definition of preprocessor macros/symbols which modify compiler  */
/* behavior or generated code characteristics must be done here,   */
/* as appropriate, before any system header file is included. It is */
/* also possible to have them defined in the config file included   */
/* before this point. As a result of all this we frown inclusion of */
/* system header files in our config files, avoid this at any cost. */
/* ================================================================ */

/*
 * AIX 4.3 and newer needs _THREAD_SAFE defined to build
 * proper reentrant code. Others may also need it.
 */

#ifdef NEED_THREAD_SAFE
#  ifndef _THREAD_SAFE
#    define _THREAD_SAFE
#  endif
#endif

/*
 * Tru64 needs _REENTRANT set for a few function prototypes and
 * things to appear in the system header files. Unixware needs it
 * to build proper reentrant code. Others may also need it.
 */

#ifdef NEED_REENTRANT
#  ifndef _REENTRANT
#    define _REENTRANT
#  endif
#endif

/* Solaris needs this to get a POSIX-conformant getpwuid_r */
#if defined(sun) || defined(__sun)
#  ifndef _POSIX_PTHREAD_SEMANTICS
#    define _POSIX_PTHREAD_SEMANTICS 1
#  endif
#endif

/* ================================================================ */
/*  If you need to include a system header file for your platform,  */
/*  please, do it beyond the point further indicated in this file.  */
/* ================================================================ */

#include <curl/curl.h>

/*
 * Disable other protocols when http is the only one desired.
 */

#ifdef HTTP_ONLY
#  ifndef CURL_DISABLE_DICT
#    define CURL_DISABLE_DICT
#  endif
#  ifndef CURL_DISABLE_FILE
#    define CURL_DISABLE_FILE
#  endif
#  ifndef CURL_DISABLE_FTP
#    define CURL_DISABLE_FTP
#  endif
#  ifndef CURL_DISABLE_GOPHER
#    define CURL_DISABLE_GOPHER
#  endif
#  ifndef CURL_DISABLE_IMAP
#    define CURL_DISABLE_IMAP
#  endif
#  ifndef CURL_DISABLE_LDAP
#    define CURL_DISABLE_LDAP
#  endif
#  ifndef CURL_DISABLE_LDAPS
#    define CURL_DISABLE_LDAPS
#  endif
#  ifndef CURL_DISABLE_MQTT
#    define CURL_DISABLE_MQTT
#  endif
#  ifndef CURL_DISABLE_POP3
#    define CURL_DISABLE_POP3
#  endif
#  ifndef CURL_DISABLE_RTSP
#    define CURL_DISABLE_RTSP
#  endif
#  ifndef CURL_DISABLE_SMB
#    define CURL_DISABLE_SMB
#  endif
#  ifndef CURL_DISABLE_SMTP
#    define CURL_DISABLE_SMTP
#  endif
#  ifndef CURL_DISABLE_TELNET
#    define CURL_DISABLE_TELNET
#  endif
#  ifndef CURL_DISABLE_TFTP
#    define CURL_DISABLE_TFTP
#  endif
#endif

/*
 * When http is disabled rtsp is not supported.
 */

#if defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_RTSP)
#  define CURL_DISABLE_RTSP
#endif

/* ================================================================ */
/* No system header file shall be included in this file before this */
/* point. The only allowed ones are those included from curl/system.h */
/* ================================================================ */

/*
 * OS/400 setup file includes some system headers.
 */

#ifdef __OS400__
#  include "setup-os400.h"
#endif

/*
 * VMS setup file includes some system headers.
 */

#ifdef __VMS
#  include "setup-vms.h"
#endif

/*
 * Windows setup file includes some system headers.
 */

#ifdef HAVE_WINDOWS_H
#  include "setup-win32.h"
#endif

/*
 * Use getaddrinfo to resolve the IPv4 address literal. If the current network
 * interface doesn't support IPv4, but supports IPv6, NAT64, and DNS64,
 * performing this task will result in a synthesized IPv6 address.
 */
#if defined(__APPLE__) && !defined(USE_ARES)
#include <TargetConditionals.h>
#define USE_RESOLVE_ON_IPS 1
#  if defined(TARGET_OS_OSX) && TARGET_OS_OSX
#    define CURL_OSX_CALL_COPYPROXIES 1
#  endif
#endif

#ifdef USE_LWIPSOCK
#  include <lwip/init.h>
#  include <lwip/sockets.h>
#  include <lwip/netdb.h>
#endif

#ifdef HAVE_EXTRA_STRICMP_H
#  include <extra/stricmp.h>
#endif

#ifdef HAVE_EXTRA_STRDUP_H
#  include <extra/strdup.h>
#endif

#ifdef TPF
#  include <strings.h>    /* for bzero, strcasecmp, and strncasecmp */
#  include <string.h>     /* for strcpy and strlen */
#  include <stdlib.h>     /* for rand and srand */
#  include <sys/socket.h> /* for select and ioctl*/
#  include <netdb.h>      /* for in_addr_t definition */
#  include <tpf/sysapi.h> /* for tpf_process_signals */
   /* change which select is used for libcurl */
#  define select(a,b,c,d,e) tpf_select_libcurl(a,b,c,d,e)
#endif

#ifdef __VXWORKS__
#  include <sockLib.h>    /* for generic BSD socket functions */
#  include <ioLib.h>      /* for basic I/O interface functions */
#endif

#ifdef __AMIGA__
#  include <exec/types.h>
#  include <exec/execbase.h>
#  include <proto/exec.h>
#  include <proto/dos.h>
#  include <unistd.h>
#  ifdef HAVE_PROTO_BSDSOCKET_H
#    include <proto/bsdsocket.h> /* ensure bsdsocket.library use */
#    define select(a,b,c,d,e) WaitSelect(a,b,c,d,e,0)
#  endif
/*
 * In clib2 arpa/inet.h warns that some prototypes may clash
 * with bsdsocket.library. This avoids the definition of those.
 */
#  define __NO_NET_API
#endif

#include <stdio.h>
#ifdef HAVE_ASSERT_H
#include <assert.h>
#endif

#ifdef __TANDEM /* for nsr-tandem-nsk systems */
#include <floss.h>
#endif

#ifndef STDC_HEADERS /* no standard C headers! */
#include <curl/stdcheaders.h>
#endif

#ifdef __POCC__
#  include <sys/types.h>
#  include <unistd.h>
#  define sys_nerr EILSEQ
#endif

/*
 * Salford-C kludge section (mostly borrowed from wxWidgets).
 */
#ifdef __SALFORDC__
  #pragma suppress 353             /* Possible nested comments */
  #pragma suppress 593             /* Define not used */
  #pragma suppress 61              /* enum has no name */
  #pragma suppress 106             /* unnamed, unused parameter */
  #include <clib.h>
#endif

/*
 * Large file (>2Gb) support using WIN32 functions.
 */

#ifdef USE_WIN32_LARGE_FILES
#  include <io.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  undef  lseek
#  define lseek(fdes,offset,whence)  _lseeki64(fdes, offset, whence)
#  undef  fstat
#  define fstat(fdes,stp)            _fstati64(fdes, stp)
#  undef  stat
#  define stat(fname,stp)            curlx_win32_stat(fname, stp)
#  define struct_stat                struct _stati64
#  define LSEEK_ERROR                (__int64)-1
#  define open                       curlx_win32_open
#  define fopen(fname,mode)          curlx_win32_fopen(fname, mode)
#  define access(fname,mode)         curlx_win32_access(fname, mode)
   int curlx_win32_open(const char *filename, int oflag, ...);
   int curlx_win32_stat(const char *path, struct_stat *buffer);
   FILE *curlx_win32_fopen(const char *filename, const char *mode);
   int curlx_win32_access(const char *path, int mode);
#endif

/*
 * Small file (<2Gb) support using WIN32 functions.
 */

#ifdef USE_WIN32_SMALL_FILES
#  include <io.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  ifndef _WIN32_WCE
#    undef  lseek
#    define lseek(fdes,offset,whence)  _lseek(fdes, (long)offset, whence)
#    define fstat(fdes,stp)            _fstat(fdes, stp)
#    define stat(fname,stp)            curlx_win32_stat(fname, stp)
#    define struct_stat                struct _stat
#    define open                       curlx_win32_open
#    define fopen(fname,mode)          curlx_win32_fopen(fname, mode)
#    define access(fname,mode)         curlx_win32_access(fname, mode)
     int curlx_win32_stat(const char *path, struct_stat *buffer);
     int curlx_win32_open(const char *filename, int oflag, ...);
     FILE *curlx_win32_fopen(const char *filename, const char *mode);
     int curlx_win32_access(const char *path, int mode);
#  endif
#  define LSEEK_ERROR                (long)-1
#endif

#ifndef struct_stat
#  define struct_stat struct stat
#endif

#ifndef LSEEK_ERROR
#  define LSEEK_ERROR (off_t)-1
#endif

#ifndef SIZEOF_TIME_T
/* assume default size of time_t to be 32 bit */
#define SIZEOF_TIME_T 4
#endif

/*
 * Default sizeof(off_t) in case it hasn't been defined in config file.
 */

#ifndef SIZEOF_OFF_T
#  if defined(__VMS) && !defined(__VAX)
#    if defined(_LARGEFILE)
#      define SIZEOF_OFF_T 8
#    endif
#  elif defined(__OS400__) && defined(__ILEC400__)
#    if defined(_LARGE_FILES)
#      define SIZEOF_OFF_T 8
#    endif
#  elif defined(__MVS__) && defined(__IBMC__)
#    if defined(_LP64) || defined(_LARGE_FILES)
#      define SIZEOF_OFF_T 8
#    endif
#  elif defined(__370__) && defined(__IBMC__)
#    if defined(_LP64) || defined(_LARGE_FILES)
#      define SIZEOF_OFF_T 8
#    endif
#  endif
#  ifndef SIZEOF_OFF_T
#    define SIZEOF_OFF_T 4
#  endif
#endif

#if (SIZEOF_CURL_OFF_T == 4)
#  define CURL_OFF_T_MAX CURL_OFF_T_C(0x7FFFFFFF)
#else
   /* assume SIZEOF_CURL_OFF_T == 8 */
#  define CURL_OFF_T_MAX CURL_OFF_T_C(0x7FFFFFFFFFFFFFFF)
#endif
#define CURL_OFF_T_MIN (-CURL_OFF_T_MAX - CURL_OFF_T_C(1))

#if (SIZEOF_TIME_T == 4)
#  ifdef HAVE_TIME_T_UNSIGNED
#  define TIME_T_MAX UINT_MAX
#  define TIME_T_MIN 0
#  else
#  define TIME_T_MAX INT_MAX
#  define TIME_T_MIN INT_MIN
#  endif
#else
#  ifdef HAVE_TIME_T_UNSIGNED
#  define TIME_T_MAX 0xFFFFFFFFFFFFFFFF
#  define TIME_T_MIN 0
#  else
#  define TIME_T_MAX 0x7FFFFFFFFFFFFFFF
#  define TIME_T_MIN (-TIME_T_MAX - 1)
#  endif
#endif

#ifndef SIZE_T_MAX
/* some limits.h headers have this defined, some don't */
#if defined(SIZEOF_SIZE_T) && (SIZEOF_SIZE_T > 4)
#define SIZE_T_MAX 18446744073709551615U
#else
#define SIZE_T_MAX 4294967295U
#endif
#endif

#ifndef SSIZE_T_MAX
/* some limits.h headers have this defined, some don't */
#if defined(SIZEOF_SIZE_T) && (SIZEOF_SIZE_T > 4)
#define SSIZE_T_MAX 9223372036854775807
#else
#define SSIZE_T_MAX 2147483647
#endif
#endif

/*
 * Arg 2 type for gethostname in case it hasn't been defined in config file.
 */

#ifndef GETHOSTNAME_TYPE_ARG2
#  ifdef USE_WINSOCK
#    define GETHOSTNAME_TYPE_ARG2 int
#  else
#    define GETHOSTNAME_TYPE_ARG2 size_t
#  endif
#endif

/* Below we define some functions. They should

   4. set the SIGALRM signal timeout
   5. set dir/file naming defines
   */

#ifdef WIN32

#  define DIR_CHAR      "\\"

#else /* WIN32 */

#  ifdef MSDOS  /* Watt-32 */

#    include <sys/ioctl.h>
#    define select(n,r,w,x,t) select_s(n,r,w,x,t)
#    define ioctl(x,y,z) ioctlsocket(x,y,(char *)(z))
#    include <tcp.h>
#    ifdef word
#      undef word
#    endif
#    ifdef byte
#      undef byte
#    endif

#  endif /* MSDOS */

#  ifdef __minix
     /* Minix 3 versions up to at least 3.1.3 are missing these prototypes */
     extern char *strtok_r(char *s, const char *delim, char **last);
     extern struct tm *gmtime_r(const time_t * const timep, struct tm *tmp);
#  endif

#  define DIR_CHAR      "/"

#  ifndef fileno /* sunos 4 have this as a macro! */
     int fileno(FILE *stream);
#  endif

#endif /* WIN32 */

/*
 * msvc 6.0 requires PSDK in order to have INET6_ADDRSTRLEN
 * defined in ws2tcpip.h as well as to provide IPv6 support.
 * Does not apply if lwIP is used.
 */

#if defined(_MSC_VER) && !defined(__POCC__) && !defined(USE_LWIPSOCK)
#  if !defined(HAVE_WS2TCPIP_H) || \
     ((_MSC_VER < 1300) && !defined(INET6_ADDRSTRLEN))
#    undef HAVE_GETADDRINFO_THREADSAFE
#    undef HAVE_FREEADDRINFO
#    undef HAVE_GETADDRINFO
#    undef ENABLE_IPV6
#  endif
#endif

/* ---------------------------------------------------------------- */
/*             resolver specialty compile-time defines              */
/*         CURLRES_* defines to use in the host*.c sources          */
/* ---------------------------------------------------------------- */

/*
 * lcc-win32 doesn't have _beginthreadex(), lacks threads support.
 */

#if defined(__LCC__) && defined(WIN32)
#  undef USE_THREADS_POSIX
#  undef USE_THREADS_WIN32
#endif

/*
 * MSVC threads support requires a multi-threaded runtime library.
 * _beginthreadex() is not available in single-threaded ones.
 */

#if defined(_MSC_VER) && !defined(__POCC__) && !defined(_MT)
#  undef USE_THREADS_POSIX
#  undef USE_THREADS_WIN32
#endif

/*
 * Mutually exclusive CURLRES_* definitions.
 */

#if defined(ENABLE_IPV6) && defined(HAVE_GETADDRINFO)
#  define CURLRES_IPV6
#else
#  define CURLRES_IPV4
#endif

#ifdef USE_ARES
#  define CURLRES_ASYNCH
#  define CURLRES_ARES
/* now undef the stock libc functions just to avoid them being used */
#  undef HAVE_GETADDRINFO
#  undef HAVE_FREEADDRINFO
#  undef HAVE_GETHOSTBYNAME
#elif defined(USE_THREADS_POSIX) || defined(USE_THREADS_WIN32)
#  define CURLRES_ASYNCH
#  define CURLRES_THREADED
#else
#  define CURLRES_SYNCH
#endif

/* ---------------------------------------------------------------- */

/*
 * msvc 6.0 does not have struct sockaddr_storage and
 * does not define IPPROTO_ESP in winsock2.h. But both
 * are available if PSDK is properly installed.
 */

#if defined(_MSC_VER) && !defined(__POCC__)
#  if !defined(HAVE_WINSOCK2_H) || ((_MSC_VER < 1300) && !defined(IPPROTO_ESP))
#    undef HAVE_STRUCT_SOCKADDR_STORAGE
#  endif
#endif

/*
 * Intentionally fail to build when using msvc 6.0 without PSDK installed.
 * The brave of heart can circumvent this, defining ALLOW_MSVC6_WITHOUT_PSDK
 * in lib/config-win32.h although absolutely discouraged and unsupported.
 */

#if defined(_MSC_VER) && !defined(__POCC__)
#  if !defined(HAVE_WINDOWS_H) || ((_MSC_VER < 1300) && !defined(_FILETIME_))
#    if !defined(ALLOW_MSVC6_WITHOUT_PSDK)
#      error MSVC 6.0 requires "February 2003 Platform SDK" a.k.a. \
             "Windows Server 2003 PSDK"
#    else
#      define CURL_DISABLE_LDAP 1
#    endif
#  endif
#endif

#ifdef NETWARE
int netware_init(void);
#ifndef __NOVELL_LIBC__
#include <sys/bsdskt.h>
#include <sys/timeval.h>
#endif
#endif

#if defined(HAVE_LIBIDN2) && defined(HAVE_IDN2_H) && !defined(USE_WIN32_IDN)
/* The lib and header are present */
#define USE_LIBIDN2
#endif

#if defined(USE_LIBIDN2) && defined(USE_WIN32_IDN)
#error "Both libidn2 and WinIDN are enabled, choose one."
#endif

#define LIBIDN_REQUIRED_VERSION "0.4.1"

#if defined(USE_GNUTLS) || defined(USE_OPENSSL) || defined(USE_NSS) || \
    defined(USE_MBEDTLS) || \
    defined(USE_WOLFSSL) || defined(USE_SCHANNEL) || \
    defined(USE_SECTRANSP) || defined(USE_GSKIT) || defined(USE_MESALINK) || \
    defined(USE_BEARSSL) || defined(USE_RUSTLS)
#define USE_SSL    /* SSL support has been enabled */
#endif

/* Single point where USE_SPNEGO definition might be defined */
#if !defined(CURL_DISABLE_CRYPTO_AUTH) && \
    (defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI))
#define USE_SPNEGO
#endif

/* Single point where USE_KERBEROS5 definition might be defined */
#if !defined(CURL_DISABLE_CRYPTO_AUTH) && \
    (defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI))
#define USE_KERBEROS5
#endif

/* Single point where USE_NTLM definition might be defined */
#if !defined(CURL_DISABLE_CRYPTO_AUTH) && !defined(CURL_DISABLE_NTLM)
#  if defined(USE_OPENSSL) || defined(USE_MBEDTLS) ||                       \
      defined(USE_GNUTLS) || defined(USE_NSS) || defined(USE_SECTRANSP) ||  \
      defined(USE_OS400CRYPTO) || defined(USE_WIN32_CRYPTO) ||              \
      (defined(USE_WOLFSSL) && defined(HAVE_WOLFSSL_DES_ECB_ENCRYPT))
#    define USE_CURL_NTLM_CORE
#  endif
#  if defined(USE_CURL_NTLM_CORE) || defined(USE_WINDOWS_SSPI)
#    define USE_NTLM
#  endif
#endif

#ifdef CURL_WANTS_CA_BUNDLE_ENV
#error "No longer supported. Set CURLOPT_CAINFO at runtime instead."
#endif

#if defined(USE_LIBSSH2) || defined(USE_LIBSSH) || defined(USE_WOLFSSH)
#define USE_SSH
#endif

/*
 * Provide a mechanism to silence picky compilers, such as gcc 4.6+.
 * Parameters should of course normally not be unused, but for example when
 * we have multiple implementations of the same interface it may happen.
 */

#if defined(__GNUC__) && ((__GNUC__ >= 3) || \
  ((__GNUC__ == 2) && defined(__GNUC_MINOR__) && (__GNUC_MINOR__ >= 7)))
#  define UNUSED_PARAM __attribute__((__unused__))
#  define WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#  define UNUSED_PARAM /*NOTHING*/
#  define WARN_UNUSED_RESULT
#endif

/*
 * Include macros and defines that should only be processed once.
 */

#ifndef HEADER_CURL_SETUP_ONCE_H
#include "curl_setup_once.h"
#endif

/*
 * Definition of our NOP statement Object-like macro
 */

#ifndef Curl_nop_stmt
#  define Curl_nop_stmt do { } while(0)
#endif

/*
 * Ensure that Winsock and lwIP TCP/IP stacks are not mixed.
 */

#if defined(__LWIP_OPT_H__) || defined(LWIP_HDR_OPT_H)
#  if defined(SOCKET) || \
     defined(USE_WINSOCK) || \
     defined(HAVE_WINSOCK2_H) || \
     defined(HAVE_WS2TCPIP_H)
#    error "WinSock and lwIP TCP/IP stack definitions shall not coexist!"
#  endif
#endif

/*
 * shutdown() flags for systems that don't define them
 */

#ifndef SHUT_RD
#define SHUT_RD 0x00
#endif

#ifndef SHUT_WR
#define SHUT_WR 0x01
#endif

#ifndef SHUT_RDWR
#define SHUT_RDWR 0x02
#endif

/* Define S_ISREG if not defined by system headers, f.e. MSVC */
#if !defined(S_ISREG) && defined(S_IFMT) && defined(S_IFREG)
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif

/* Define S_ISDIR if not defined by system headers, f.e. MSVC */
#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

/* In Windows the default file mode is text but an application can override it.
Therefore we specify it explicitly. https://github.com/curl/curl/pull/258
*/
#if defined(WIN32) || defined(MSDOS)
#define FOPEN_READTEXT "rt"
#define FOPEN_WRITETEXT "wt"
#define FOPEN_APPENDTEXT "at"
#elif defined(__CYGWIN__)
/* Cygwin has specific behavior we need to address when WIN32 is not defined.
https://cygwin.com/cygwin-ug-net/using-textbinary.html
For write we want our output to have line endings of LF and be compatible with
other Cygwin utilities. For read we want to handle input that may have line
endings either CRLF or LF so 't' is appropriate.
*/
#define FOPEN_READTEXT "rt"
#define FOPEN_WRITETEXT "w"
#define FOPEN_APPENDTEXT "a"
#else
#define FOPEN_READTEXT "r"
#define FOPEN_WRITETEXT "w"
#define FOPEN_APPENDTEXT "a"
#endif

/* WinSock destroys recv() buffer when send() failed.
 * Enabled automatically for Windows and for Cygwin as Cygwin sockets are
 * wrappers for WinSock sockets. https://github.com/curl/curl/issues/657
 * Define DONT_USE_RECV_BEFORE_SEND_WORKAROUND to force disable workaround.
 */
#if !defined(DONT_USE_RECV_BEFORE_SEND_WORKAROUND)
#  if defined(WIN32) || defined(__CYGWIN__)
#    define USE_RECV_BEFORE_SEND_WORKAROUND
#  endif
#else  /* DONT_USE_RECV_BEFORE_SEND_WORKAROUND */
#  ifdef USE_RECV_BEFORE_SEND_WORKAROUND
#    undef USE_RECV_BEFORE_SEND_WORKAROUND
#  endif
#endif /* DONT_USE_RECV_BEFORE_SEND_WORKAROUND */

/* for systems that don't detect this in configure */
#ifndef CURL_SA_FAMILY_T
#  if defined(HAVE_SA_FAMILY_T)
#    define CURL_SA_FAMILY_T sa_family_t
#  elif defined(HAVE_ADDRESS_FAMILY)
#    define CURL_SA_FAMILY_T ADDRESS_FAMILY
#  else
/* use a sensible default */
#    define CURL_SA_FAMILY_T unsigned short
#  endif
#endif

/* Some convenience macros to get the larger/smaller value out of two given.
   We prefix with CURL to prevent name collisions. */
#define CURLMAX(x,y) ((x)>(y)?(x):(y))
#define CURLMIN(x,y) ((x)<(y)?(x):(y))

/* Some versions of the Android SDK is missing the declaration */
#if defined(HAVE_GETPWUID_R) && defined(HAVE_DECL_GETPWUID_R_MISSING)
struct passwd;
int getpwuid_r(uid_t uid, struct passwd *pwd, char *buf,
               size_t buflen, struct passwd **result);
#endif

#ifdef DEBUGBUILD
#define UNITTEST
#else
#define UNITTEST static
#endif

#if defined(USE_NGHTTP2) || defined(USE_HYPER)
#define USE_HTTP2
#endif

#if defined(USE_NGTCP2) || defined(USE_QUICHE)
#define ENABLE_QUIC
#endif

#if defined(USE_UNIX_SOCKETS) && defined(WIN32)
#  if defined(__MINGW32__) && !defined(LUP_SECURE)
     typedef u_short ADDRESS_FAMILY; /* Classic mingw, 11y+ old mingw-w64 */
#  endif
#  if !defined(UNIX_PATH_MAX)
     /* Replicating logic present in afunix.h
        (distributed with newer Windows 10 SDK versions only) */
#    define UNIX_PATH_MAX 108
     /* !checksrc! disable TYPEDEFSTRUCT 1 */
     typedef struct sockaddr_un {
       ADDRESS_FAMILY sun_family;
       char sun_path[UNIX_PATH_MAX];
     } SOCKADDR_UN, *PSOCKADDR_UN;
#    define WIN32_SOCKADDR_UN
#  endif
#endif

#endif /* HEADER_CURL_SETUP_H */
