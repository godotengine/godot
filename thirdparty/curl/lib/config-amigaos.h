#ifndef HEADER_CURL_CONFIG_AMIGAOS_H
#define HEADER_CURL_CONFIG_AMIGAOS_H
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
/*               Hand crafted config file for AmigaOS               */
/* ================================================================ */

#ifdef __AMIGA__ /* Any AmigaOS flavour */

#define HAVE_ARPA_INET_H 1
#define HAVE_CLOSESOCKET_CAMEL 1
#define HAVE_ERRNO_H 1
#define HAVE_INET_ADDR 1
#define HAVE_INTTYPES_H 1
#define HAVE_IOCTLSOCKET_CAMEL 1
#define HAVE_IOCTLSOCKET_CAMEL_FIONBIO 1
#define HAVE_LIBZ 1
#define HAVE_LONGLONG 1
#define HAVE_MALLOC_H 1
#define HAVE_MEMORY_H 1
#define HAVE_NETDB_H 1
#define HAVE_NETINET_IN_H 1
#define HAVE_NET_IF_H 1
#define HAVE_OPENSSL_CRYPTO_H 1
#define HAVE_OPENSSL_ERR_H 1
#define HAVE_OPENSSL_PEM_H 1
#define HAVE_OPENSSL_RSA_H 1
#define HAVE_OPENSSL_SSL_H 1
#define HAVE_OPENSSL_X509_H 1
#define HAVE_PWD_H 1
#define HAVE_RAND_EGD 1
#define HAVE_RAND_STATUS 1
#define HAVE_SELECT 1
#define HAVE_SETJMP_H 1
#define HAVE_SIGNAL 1
#define HAVE_SIGNAL_H 1
#define HAVE_SOCKET 1
#define HAVE_STRCASECMP 1
#define HAVE_STRDUP 1
#define HAVE_STRFTIME 1
#define HAVE_STRICMP 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
#define HAVE_STRSTR 1
#define HAVE_STRUCT_TIMEVAL 1
#define HAVE_SYS_PARAM_H 1
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_SOCKIO_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_TIME_H 1
#define HAVE_UNAME 1
#define HAVE_UNISTD_H 1
#define HAVE_UTIME 1
#define HAVE_UTIME_H 1
#define HAVE_WRITABLE_ARGV 1
#define HAVE_ZLIB_H 1
#define HAVE_SYS_IOCTL_H 1

#define NEED_MALLOC_H 1

#define SIZEOF_INT 4
#define SIZEOF_SHORT 2
#define SIZEOF_SIZE_T 4

#define USE_MANUAL 1
#define USE_OPENSSL 1
#define CURL_DISABLE_LDAP 1

#define OS "AmigaOS"

#define PACKAGE "curl"
#define PACKAGE_BUGREPORT "a suitable mailing list: https://curl.se/mail/"
#define PACKAGE_NAME "curl"
#define PACKAGE_STRING "curl -"
#define PACKAGE_TARNAME "curl"
#define PACKAGE_VERSION "-"
#define CURL_CA_BUNDLE "s:curl-ca-bundle.crt"

#define SELECT_TYPE_ARG1 int
#define SELECT_TYPE_ARG234 (fd_set *)
#define SELECT_TYPE_ARG5 (struct timeval *)

#define STDC_HEADERS 1
#define TIME_WITH_SYS_TIME 1

#define in_addr_t int

#ifndef F_OK
#  define F_OK 0
#endif

#ifndef O_RDONLY
#  define O_RDONLY 0x0000
#endif

#ifndef LONG_MAX
#  define LONG_MAX 0x7fffffffL
#endif

#ifndef LONG_MIN
#  define LONG_MIN (-0x7fffffffL-1)
#endif

#define HAVE_RECV 1
#define RECV_TYPE_ARG1 long
#define RECV_TYPE_ARG2 char *
#define RECV_TYPE_ARG3 long
#define RECV_TYPE_ARG4 long
#define RECV_TYPE_RETV long

#define HAVE_RECVFROM 1
#define RECVFROM_TYPE_ARG1 long
#define RECVFROM_TYPE_ARG2 char
#define RECVFROM_TYPE_ARG3 long
#define RECVFROM_TYPE_ARG4 long
#define RECVFROM_TYPE_ARG5 struct sockaddr
#define RECVFROM_TYPE_ARG6 long
#define RECVFROM_TYPE_RETV long

#define HAVE_SEND 1
#define SEND_TYPE_ARG1 int
#define SEND_QUAL_ARG2 const
#define SEND_TYPE_ARG2 char *
#define SEND_TYPE_ARG3 int
#define SEND_TYPE_ARG4 int
#define SEND_TYPE_RETV int

#endif /* __AMIGA__ */
#endif /* HEADER_CURL_CONFIG_AMIGAOS_H */
