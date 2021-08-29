#ifndef HEADER_CURL_CONFIG_OS400_H
#define HEADER_CURL_CONFIG_OS400_H
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
/*                Hand crafted config file for OS/400               */
/* ================================================================ */

#pragma enum(int)

#undef PACKAGE

/* Version number of this archive. */
#undef VERSION

/* Define if you have the getpass function.  */
#undef HAVE_GETPASS

/* Define cpu-machine-OS */
#define OS "OS/400"

/* OS400 supports a 3-argument ASCII version of gethostbyaddr_r(), but its
 *  prototype is incompatible with the "standard" one (1st argument is not
 *  const). However, getaddrinfo() is supported (ASCII version defined as
 *  a local wrapper in setup-os400.h) in a threadsafe way: we can then
 *  configure getaddrinfo() as such and get rid of gethostbyname_r() without
 *  loss of threadsafeness. */
#undef HAVE_GETHOSTBYNAME_R
#undef HAVE_GETHOSTBYNAME_R_3
#undef HAVE_GETHOSTBYNAME_R_5
#undef HAVE_GETHOSTBYNAME_R_6
#define HAVE_GETADDRINFO
#define HAVE_GETADDRINFO_THREADSAFE

/* Define if you need the _REENTRANT define for some functions */
#undef NEED_REENTRANT

/* Define if you have the Kerberos4 libraries (including -ldes) */
#undef HAVE_KRB4

/* Define if you want to enable IPv6 support */
#define ENABLE_IPV6

/* Define if struct sockaddr_in6 has the sin6_scope_id member */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* Define this to 'int' if ssize_t is not an available typedefed type */
#undef ssize_t

/* Define this as a suitable file to read random data from */
#undef RANDOM_FILE

/* Define this to your Entropy Gathering Daemon socket pathname */
#undef EGD_SOCKET

/* Define to 1 if you have the alarm function. */
#define HAVE_ALARM 1

/* Define if you have the <alloca.h> header file. */
#undef HAVE_ALLOCA_H

/* Define if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H

/* Define if you have the `closesocket' function. */
#undef HAVE_CLOSESOCKET

/* Define if you have the <errno.h> header file. */
#define HAVE_ERRNO_H

/* Define if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H

/* Define if you have the `geteuid' function. */
#define HAVE_GETEUID

/* Define if you have the `gethostname' function. */
#define HAVE_GETHOSTNAME

/* Define if you have the <getopt.h> header file. */
#undef HAVE_GETOPT_H

/* Define if you have the `getpass_r' function. */
#undef HAVE_GETPASS_R

/* Define to 1 if you have the getpeername function. */
#define HAVE_GETPEERNAME 1

/* Define if you have the `getpwuid' function. */
#define HAVE_GETPWUID

/* Define if you have the `getservbyname' function. */
#define HAVE_GETSERVBYNAME

/* Define to 1 if you have the getsockname function. */
#define HAVE_GETSOCKNAME 1

/* Define if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY

/* Define if you have the `timeval' struct. */
#define HAVE_STRUCT_TIMEVAL

/* Define if you have the `inet_addr' function. */
#define HAVE_INET_ADDR

/* Define if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H

/* Define if you have the <io.h> header file. */
#undef HAVE_IO_H

/* Define if you have the `krb_get_our_ip_for_realm' function. */
#undef HAVE_KRB_GET_OUR_IP_FOR_REALM

/* Define if you have the <krb.h> header file. */
#undef HAVE_KRB_H

/* Define if you have the `nsl' library (-lnsl). */
#undef HAVE_LIBNSL

/* Define if you have the `resolv' library (-lresolv). */
#undef HAVE_LIBRESOLV

/* Define if you have the `resolve' library (-lresolve). */
#undef HAVE_LIBRESOLVE

/* Define if you have the `socket' library (-lsocket). */
#undef HAVE_LIBSOCKET

/* Define if you have GSS API. */
#define HAVE_GSSAPI

/* Define if you have the GNU gssapi libraries */
#undef HAVE_GSSGNU

/* Define if you have the Heimdal gssapi libraries */
#define HAVE_GSSHEIMDAL

/* Define if you have the MIT gssapi libraries */
#undef HAVE_GSSMIT

/* Define if you have the `ucb' library (-lucb). */
#undef HAVE_LIBUCB

/* Define if you have the `localtime_r' function. */
#define HAVE_LOCALTIME_R

/* Define if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H

/* Define if you need the malloc.h header file even with stdlib.h  */
/* #define NEED_MALLOC_H 1 */

/* Define if you have the <memory.h> header file. */
#undef HAVE_MEMORY_H

/* Define if you have the <netdb.h> header file. */
#define HAVE_NETDB_H

/* Define if you have the <netinet/if_ether.h> header file. */
#undef HAVE_NETINET_IF_ETHER_H

/* Define if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H

/* Define if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H

/* Define if you have the <openssl/crypto.h> header file. */
#undef HAVE_OPENSSL_CRYPTO_H

/* Define if you have the <openssl/err.h> header file. */
#undef HAVE_OPENSSL_ERR_H

/* Define if you have the <openssl/pem.h> header file. */
#undef HAVE_OPENSSL_PEM_H

/* Define if you have the <openssl/rsa.h> header file. */
#undef HAVE_OPENSSL_RSA_H

/* Define if you have the <openssl/ssl.h> header file. */
#undef HAVE_OPENSSL_SSL_H

/* Define if you have the <openssl/x509.h> header file. */
#undef HAVE_OPENSSL_X509_H

/* Define if you have the <pem.h> header file. */
#undef HAVE_PEM_H

/* Define if you have the <pwd.h> header file. */
#define HAVE_PWD_H

/* Define if you have the `RAND_egd' function. */
#undef HAVE_RAND_EGD

/* Define if you have the `RAND_screen' function. */
#undef HAVE_RAND_SCREEN

/* Define if you have the `RAND_status' function. */
#undef HAVE_RAND_STATUS

/* Define if you have the `select' function. */
#define HAVE_SELECT

/* Define if you have the `setvbuf' function. */
#define HAVE_SETVBUF

/* Define if you have the `sigaction' function. */
#define HAVE_SIGACTION

/* Define if you have the `signal' function. */
#undef HAVE_SIGNAL

/* Define if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H

/* Define if you have the `socket' function. */
#define HAVE_SOCKET

/* Define if you have the <ssl.h> header file. */
#undef HAVE_SSL_H

/* Define if you have the <stdint.h> header file. */
#undef HAVE_STDINT_H

/* Define if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H


/* The following define is needed on OS400 to enable strcmpi(), stricmp() and
   strdup(). */
#define __cplusplus__strings__

/* Define if you have the `strcasecmp' function. */
#undef HAVE_STRCASECMP

/* Define if you have the `strcmpi' function. */
#define HAVE_STRCMPI

/* Define if you have the `stricmp' function. */
#define HAVE_STRICMP

/* Define if you have the `strdup' function. */
#define HAVE_STRDUP


/* Define if you have the `strftime' function. */
#define HAVE_STRFTIME

/* Define if you have the <strings.h> header file. */
#define HAVE_STRINGS_H

/* Define if you have the <string.h> header file. */
#define HAVE_STRING_H

/* Define if you have the <stropts.h> header file. */
#undef HAVE_STROPTS_H

/* Define if you have the `strstr' function. */
#define HAVE_STRSTR

/* Define if you have the `strtok_r' function. */
#define HAVE_STRTOK_R

/* Define if you have the `strtoll' function. */
#undef HAVE_STRTOLL             /* Allows ASCII compile on V5R1. */

/* Define if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H

/* Define if you have the <sys/select.h> header file. */
#undef HAVE_SYS_SELECT_H

/* Define if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H

/* Define if you have the <sys/sockio.h> header file. */
#undef HAVE_SYS_SOCKIO_H

/* Define if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H

/* Define if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H

/* Define if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H

/* Define if you have the <sys/un.h> header file. */
#define HAVE_SYS_UN_H

/* Define if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H

/* Define if you have the <termios.h> header file. */
#undef HAVE_TERMIOS_H

/* Define if you have the <termio.h> header file. */
#undef HAVE_TERMIO_H

/* Define if you have the <time.h> header file. */
#define HAVE_TIME_H

/* Define if you have the `uname' function. */
#undef HAVE_UNAME

/* Define if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H

/* Define if you have the <x509.h> header file. */
#undef HAVE_X509_H

/* Name of package */
#undef PACKAGE

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT              4

/* The size of a `long double', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE      8

/* Define if the compiler supports the 'long long' data type. */
#define HAVE_LONGLONG

/* The size of a `long long', as computed by sizeof. */
#define SIZEOF_LONG_LONG        8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT            2

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG             4

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T           4

/* The size of `curl_off_t', as computed by sizeof. */
#define SIZEOF_CURL_OFF_T       8

/* Whether long long constants must be suffixed by LL. */

#define HAVE_LL

/* Define this if you have struct sockaddr_storage */
#define HAVE_STRUCT_SOCKADDR_STORAGE

/* Define if you have the ANSI C header files. */
#define STDC_HEADERS

/* Define if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME

/* Define to enable HTTP3 support (experimental, requires NGTCP2 or QUICHE) */
#undef ENABLE_QUIC

/* Version number of package */
#undef VERSION

/* Number of bits in a file offset, on hosts where this is settable. */
#undef _FILE_OFFSET_BITS

/* Define for large files, on AIX-style hosts. */
#define _LARGE_FILES

/* Define to empty if `const' does not conform to ANSI C. */
#undef const

/* type to use in place of in_addr_t if not defined */
#define in_addr_t       unsigned long

/* Define to `unsigned' if <sys/types.h> does not define. */
#undef size_t

/* Define if you have the ioctl function. */
#define HAVE_IOCTL

/* Define if you have a working ioctl FIONBIO function. */
#define HAVE_IOCTL_FIONBIO

/* Define if you have a working ioctl SIOCGIFADDR function. */
#define HAVE_IOCTL_SIOCGIFADDR

/* To disable LDAP */
#undef CURL_DISABLE_LDAP

/* Definition to make a library symbol externally visible. */
#define CURL_EXTERN_SYMBOL

/* Define if you have the ldap_url_parse procedure. */
/* #define HAVE_LDAP_URL_PARSE */    /* Disabled because of an IBM bug. */

/* Define if you have the recv function. */
#define HAVE_RECV

/* Define to the type of arg 1 for recv. */
#define RECV_TYPE_ARG1 int

/* Define to the type of arg 2 for recv. */
#define RECV_TYPE_ARG2 char *

/* Define to the type of arg 3 for recv. */
#define RECV_TYPE_ARG3 int

/* Define to the type of arg 4 for recv. */
#define RECV_TYPE_ARG4 int

/* Define to the function return type for recv. */
#define RECV_TYPE_RETV int

/* Define if you have the recvfrom function. */
#define HAVE_RECVFROM

/* Define to the type of arg 1 for recvfrom. */
#define RECVFROM_TYPE_ARG1 int

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
#define HAVE_SEND

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 int

/* Define to the type qualifier of arg 2 for send. */
#define SEND_QUAL_ARG2

/* Define to the type of arg 2 for send. */
#define SEND_TYPE_ARG2 char *

/* Define to the type of arg 3 for send. */
#define SEND_TYPE_ARG3 int

/* Define to the type of arg 4 for send. */
#define SEND_TYPE_ARG4 int

/* Define to the function return type for send. */
#define SEND_TYPE_RETV int

/* Define to use the GSKit package. */
#define USE_GSKIT

/* Define to use the OS/400 crypto library. */
#define USE_OS400CRYPTO

/* Define to use Unix sockets. */
#define USE_UNIX_SOCKETS

/* Use the system keyring as the default CA bundle. */
#define CURL_CA_BUNDLE  "/QIBM/UserData/ICSS/Cert/Server/DEFAULT.KDB"

/* ---------------------------------------------------------------- */
/*                       ADDITIONAL DEFINITIONS                     */
/* ---------------------------------------------------------------- */

/* The following must be defined BEFORE system header files inclusion. */

#define __ptr128                       /* No teraspace. */
#define qadrt_use_fputc_inline         /* Generate fputc() wrapper inline. */
#define qadrt_use_fread_inline         /* Generate fread() wrapper inline. */
#define qadrt_use_fwrite_inline        /* Generate fwrite() wrapper inline. */

#endif /* HEADER_CURL_CONFIG_OS400_H */
