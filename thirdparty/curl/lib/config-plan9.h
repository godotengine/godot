#ifndef HEADER_CURL_CONFIG_PLAN9_H
#define HEADER_CURL_CONFIG_PLAN9_H
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

#define BUILDING_LIBCURL 1
#define CURL_CA_BUNDLE "/sys/lib/tls/ca.pem"
#define CURL_CA_PATH "/sys/lib/tls"
#define CURL_STATICLIB 1
#define ENABLE_IPV6 1
#define CURL_DISABLE_LDAP 1

#define NEED_REENTRANT 1
#define OS "plan9"
#define PACKAGE "curl"
#define PACKAGE_NAME "curl"
#define PACKAGE_BUGREPORT "a suitable mailing list: https://curl.se/mail/"
#define PACKAGE_STRING "curl -"
#define PACKAGE_TARNAME "curl"
#define PACKAGE_VERSION "-"
#define RANDOM_FILE "/dev/random"
#define VERSION "0.0.0" /* TODO */

#define STDC_HEADERS 1

#ifdef _BITS64
#error not implement
#else
#define SIZEOF_INT 4
#define SIZEOF_SHORT 2
#define SIZEOF_LONG 4
#define SIZEOF_OFF_T 8
#define SIZEOF_CURL_OFF_T 4 /* curl_off_t = timediff_t = int */
#define SIZEOF_SIZE_T 4
#define SIZEOF_TIME_T 4
#endif

#define HAVE_RECV 1
#define RECV_TYPE_ARG1 int
#define RECV_TYPE_ARG2 void *
#define RECV_TYPE_ARG3 int
#define RECV_TYPE_ARG4 int
#define RECV_TYPE_RETV int

#define HAVE_RECVFROM 1
#define RECVFROM_TYPE_ARG1 int
#define RECVFROM_TYPE_ARG2 void
#define RECVFROM_TYPE_ARG2_IS_VOID 1
#define RECVFROM_TYPE_ARG3 int
#define RECVFROM_TYPE_ARG4 int
#define RECVFROM_TYPE_ARG5 void
#define RECVFROM_TYPE_ARG5_IS_VOID 1
#define RECVFROM_TYPE_ARG6 int
#define RECVFROM_TYPE_ARG6_IS_VOID 1
#define RECVFROM_TYPE_RETV int

#define HAVE_SELECT 1
#define SELECT_TYPE_ARG1 int
#define SELECT_TYPE_ARG234 fd_set *
#define SELECT_TYPE_ARG5 struct timeval *
#define SELECT_TYPE_RETV int

#define HAVE_SEND 1
#define SEND_TYPE_ARG1 int
#define SEND_TYPE_ARG2 void *
#define SEND_QUAL_ARG2
#define SEND_TYPE_ARG3 int
#define SEND_TYPE_ARG4 int
#define SEND_TYPE_RETV int

#define HAVE_ALARM 1
#define HAVE_ARPA_INET_H 1
#define HAVE_ASSERT_H 1
#define HAVE_BASENAME 1
#define HAVE_BOOL_T 1
#define HAVE_CRYPTO_CLEANUP_ALL_EX_DATA 1
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL 1
#define HAVE_FCNTL_H 1
#define HAVE_FREEADDRINFO 1
#define HAVE_FTRUNCATE 1
#define HAVE_GETADDRINFO 1
#define HAVE_GETEUID 1
#define HAVE_GETHOSTBYNAME 1
#define HAVE_GETHOSTNAME 1
#define HAVE_GETPPID 1
#define HAVE_GETPROTOBYNAME 1
#define HAVE_GETPWUID 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_GMTIME_R 1
#define HAVE_INET_ADDR 1
#define HAVE_INET_NTOP 1
#define HAVE_INET_PTON 1
#define HAVE_INTTYPES_H 1
#define HAVE_IOCTL 1
#define HAVE_LIBGEN_H 1
#define HAVE_LIBZ 1
#define HAVE_LL 1
#define HAVE_LOCALE_H 1
#define HAVE_LOCALTIME_R 1
#define HAVE_LONGLONG 1
#define HAVE_NETDB_H 1
#define HAVE_NETINET_IN_H 1
#define HAVE_NETINET_TCP_H 1
#define HAVE_PWD_H 1
#define HAVE_SYS_SELECT_H 1

#define USE_OPENSSL 1
#define HAVE_OPENSSL_CRYPTO_H 1
#define HAVE_OPENSSL_ERR_H 1
#define HAVE_OPENSSL_PEM_H 1
#define HAVE_OPENSSL_PKCS12_H 1
#define HAVE_OPENSSL_RSA_H 1
#define HAVE_OPENSSL_SSL_H 1
#define HAVE_OPENSSL_X509_H 1

#define HAVE_PIPE 1
#define HAVE_POLL 1
#define HAVE_POLL_FINE 1
#define HAVE_POLL_H 1
#define HAVE_PTHREAD_H 1
#define HAVE_RAND_STATUS 1
#define HAVE_SETJMP_H 1
#define HAVE_SETLOCALE 1

#define HAVE_SETSOCKOPT 1
#define HAVE_SOCK_OPTS 1 /* for /sys/include/ape/sys/socket.h */

#define HAVE_SIGACTION 1
#define HAVE_SIGNAL 1
#define HAVE_SIGNAL_H 1
#define HAVE_SIGSETJMP 1
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1
#define HAVE_SOCKET 1
#define HAVE_SSL_GET_SHUTDOWN 1
#define HAVE_STDBOOL_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRCASECMP 1
#define HAVE_STRDUP 1
#define HAVE_STRING_H 1
#define HAVE_STRSTR 1
#define HAVE_STRTOK_R 1
#define HAVE_STRTOLL 1
#define HAVE_STRUCT_TIMEVAL 1
#define HAVE_SYS_IOCTL_H 1
#define HAVE_SYS_PARAM_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_UIO_H 1
#define HAVE_SYS_UN_H 1
#define HAVE_TERMIOS_H 1
#define HAVE_TIME_H 1
#define HAVE_UNAME 1
#define HAVE_UNISTD_H 1
#define HAVE_UTIME 1
#define HAVE_UTIME_H 1
#define HAVE_WRITEV 1

#define HAVE_ZLIB_H 1

#define HAVE_POSIX_STRERROR_R 1
#define HAVE_STRERROR_R 1
#define STRERROR_R_TYPE_ARG3 int

#define TIME_WITH_SYS_TIME 1
#define USE_MANUAL 1

#define __attribute__(x)

#ifndef __cplusplus
#undef inline
#endif

#endif /* HEADER_CURL_CONFIG_PLAN9_H */
