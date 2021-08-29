#ifndef HEADER_CURL_CONFIG_RISCOS_H
#define HEADER_CURL_CONFIG_RISCOS_H
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
/*               Hand crafted config file for RISC OS               */
/* ================================================================ */

/* Name of this package! */
#undef PACKAGE

/* Version number of this archive. */
#undef VERSION

/* Define if you have the getpass function.  */
#undef HAVE_GETPASS

/* Define cpu-machine-OS */
#define OS "ARM-RISC OS"

/* Define if you want the built-in manual */
#define USE_MANUAL

/* Define if you have the gethostbyname_r() function with 3 arguments */
#undef HAVE_GETHOSTBYNAME_R_3

/* Define if you have the gethostbyname_r() function with 5 arguments */
#undef HAVE_GETHOSTBYNAME_R_5

/* Define if you have the gethostbyname_r() function with 6 arguments */
#undef HAVE_GETHOSTBYNAME_R_6

/* Define if you need the _REENTRANT define for some functions */
#undef NEED_REENTRANT

/* Define if you have the Kerberos4 libraries (including -ldes) */
#undef HAVE_KRB4

/* Define if you want to enable IPv6 support */
#undef ENABLE_IPV6

/* Define if struct sockaddr_in6 has the sin6_scope_id member */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* Define this to 'int' if ssize_t is not an available typedefed type */
#undef ssize_t

/* Define this as a suitable file to read random data from */
#undef RANDOM_FILE

/* Define this to your Entropy Gathering Daemon socket pathname */
#undef EGD_SOCKET

/* Define if you want to enable IPv6 support */
#undef ENABLE_IPV6

/* Define if you have the alarm function. */
#define HAVE_ALARM

/* Define if you have the <alloca.h> header file. */
#define HAVE_ALLOCA_H

/* Define if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H

/* Define if you have the `closesocket' function. */
#undef HAVE_CLOSESOCKET

/* Define if you have the <errno.h> header file. */
#define HAVE_ERRNO_H

/* Define if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H

/* Define if you have the `ftruncate' function. */
#define HAVE_FTRUNCATE

/* Define if getaddrinfo exists and works */
#define HAVE_GETADDRINFO

/* Define if you have the `geteuid' function. */
#undef HAVE_GETEUID

/* Define if you have the `gethostbyname_r' function. */
#undef HAVE_GETHOSTBYNAME_R

/* Define if you have the `gethostname' function. */
#define HAVE_GETHOSTNAME

/* Define if you have the <getopt.h> header file. */
#define HAVE_GETOPT_H

/* Define if you have the `getpass_r' function. */
#undef HAVE_GETPASS_R

/* Define if you have the `getpwuid' function. */
#undef HAVE_GETPWUID

/* Define if you have the `getservbyname' function. */
#undef HAVE_GETSERVBYNAME

/* Define if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY

/* Define if you have the `timeval' struct. */
#define HAVE_STRUCT_TIMEVAL

/* Define if you have the `inet_addr' function. */
#undef HAVE_INET_ADDR

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

/* Define if you have the `ucb' library (-lucb). */
#undef HAVE_LIBUCB

/* Define if you have the `localtime_r' function. */
#undef HAVE_LOCALTIME_R

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
#undef HAVE_PWD_H

/* Define if you have the `RAND_egd' function. */
#undef HAVE_RAND_EGD

/* Define if you have the `RAND_screen' function. */
#undef HAVE_RAND_SCREEN

/* Define if you have the `RAND_status' function. */
#undef HAVE_RAND_STATUS

/* Define if you have the `select' function. */
#define HAVE_SELECT

/* Define if you have the `setvbuf' function. */
#undef HAVE_SETVBUF

/* Define if you have the `sigaction' function. */
#undef HAVE_SIGACTION

/* Define if you have the `signal' function. */
#define HAVE_SIGNAL

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

/* Define if you have the `strcasecmp' function. */
#undef HAVE_STRCASECMP

/* Define if you have the `strcmpi' function. */
#undef HAVE_STRCMPI

/* Define if you have the `strdup' function. */
#define HAVE_STRDUP

/* Define if you have the `strftime' function. */
#define HAVE_STRFTIME

/* Define if you have the `stricmp' function. */
#define HAVE_STRICMP

/* Define if you have the <strings.h> header file. */
#undef HAVE_STRINGS_H

/* Define if you have the <string.h> header file. */
#define HAVE_STRING_H

/* Define if you have the `strstr' function. */
#define HAVE_STRSTR

/* Define if you have the `strtok_r' function. */
#undef HAVE_STRTOK_R

/* Define if you have the `strtoll' function. */
#undef HAVE_STRTOLL

/* Define if you have the <sys/param.h> header file. */
#undef HAVE_SYS_PARAM_H

/* Define if you have the <sys/select.h> header file. */
#undef HAVE_SYS_SELECT_H

/* Define if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H

/* Define if you have the <sys/sockio.h> header file. */
#undef HAVE_SYS_SOCKIO_H

/* Define if you have the <sys/stat.h> header file. */
#undef HAVE_SYS_STAT_H

/* Define if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H

/* Define if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H

/* Define if you have the <termios.h> header file. */
#define HAVE_TERMIOS_H

/* Define if you have the <termio.h> header file. */
#undef HAVE_TERMIO_H

/* Define if you have the <time.h> header file. */
#undef HAVE_TIME_H

/* Define if you have the `uname' function. */
#define HAVE_UNAME

/* Define if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H

/* Define if you have the <x509.h> header file. */
#undef HAVE_X509_H

/* Name of package */
#undef PACKAGE

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `long double', as computed by sizeof. */
#undef SIZEOF_LONG_DOUBLE

/* The size of `long long', as computed by sizeof. */
#undef SIZEOF_LONG_LONG

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 4

/* Define if you have the ANSI C header files. */
#undef STDC_HEADERS

/* Define if you can safely include both <sys/time.h> and <time.h>. */
#undef TIME_WITH_SYS_TIME

/* Version number of package */
#undef VERSION

/* Define if on AIX 3.
   System headers sometimes define this.
   We just want to avoid a redefinition error message.  */
#ifndef _ALL_SOURCE
# undef _ALL_SOURCE
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
#undef _FILE_OFFSET_BITS

/* Define for large files, on AIX-style hosts. */
#undef _LARGE_FILES

/* Define to empty if `const' does not conform to ANSI C. */
#undef const

/* Define to `unsigned' if <sys/types.h> does not define. */
#undef size_t

/* Define to `int' if <sys/types.h> does not define. */
#undef ssize_t

/* Define if you have the ioctl function. */
#define HAVE_IOCTL

/* Define if you have a working ioctl FIONBIO function. */
#define HAVE_IOCTL_FIONBIO

/* to disable LDAP */
#define CURL_DISABLE_LDAP

/* Define if you have the recv function. */
#define HAVE_RECV 1

/* Define to the type of arg 1 for recv. */
#define RECV_TYPE_ARG1 int

/* Define to the type of arg 2 for recv. */
#define RECV_TYPE_ARG2 void *

/* Define to the type of arg 3 for recv. */
#define RECV_TYPE_ARG3 size_t

/* Define to the type of arg 4 for recv. */
#define RECV_TYPE_ARG4 int

/* Define to the function return type for recv. */
#define RECV_TYPE_RETV ssize_t

/* Define 1 if you have the recvfrom function. */
#define HAVE_RECVFROM 1

/* Define to the type of arg 1 for recvfrom. */
#define RECVFROM_TYPE_ARG1 int

/* Define to the type pointed by arg 2 for recvfrom. */
#define RECVFROM_TYPE_ARG2 void

/* Define if the type pointed by arg 2 for recvfrom is void. */
#define RECVFROM_TYPE_ARG2_IS_VOID

/* Define to the type of arg 3 for recvfrom. */
#define RECVFROM_TYPE_ARG3 size_t

/* Define to the type of arg 4 for recvfrom. */
#define RECVFROM_TYPE_ARG4 int

/* Define to the type pointed by arg 5 for recvfrom. */
#define RECVFROM_TYPE_ARG5 struct sockaddr

/* Define to the type pointed by arg 6 for recvfrom. */
#define RECVFROM_TYPE_ARG6 int

/* Define to the function return type for recvfrom. */
#define RECVFROM_TYPE_RETV ssize_t

/* Define if you have the send function. */
#define HAVE_SEND 1

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 int

/* Define to the type qualifier of arg 2 for send. */
#define SEND_QUAL_ARG2 const

/* Define to the type of arg 2 for send. */
#define SEND_TYPE_ARG2 void *

/* Define to the type of arg 3 for send. */
#define SEND_TYPE_ARG3 size_t

/* Define to the type of arg 4 for send. */
#define SEND_TYPE_ARG4 int

/* Define to the function return type for send. */
#define SEND_TYPE_RETV ssize_t

#endif /* HEADER_CURL_CONFIG_RISCOS_H */
