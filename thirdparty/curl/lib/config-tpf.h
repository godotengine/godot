#ifndef HEADER_CURL_CONFIG_TPF_H
#define HEADER_CURL_CONFIG_TPF_H
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
/*                 Hand crafted config file for TPF                 */
/* ================================================================ */

/* ---------------------------------------------------------------- */
/*            FEATURES, FUNCTIONS, and DEFINITIONS                  */
/* ---------------------------------------------------------------- */

/* NOTE: Refer also to the .mak file for some of the flags below */

/* to disable cookies support */
/* #undef CURL_DISABLE_COOKIES */

/* to disable cryptographic authentication */
/* #undef CURL_DISABLE_CRYPTO_AUTH */

/* to disable DICT */
/* #undef CURL_DISABLE_DICT */

/* to disable FILE */
/* #undef CURL_DISABLE_FILE */

/* to disable FTP */
/* #undef CURL_DISABLE_FTP */

/* to disable HTTP */
/* #undef CURL_DISABLE_HTTP */

/* to disable LDAP */
/* #undef CURL_DISABLE_LDAP */

/* to disable TELNET */
/* #undef CURL_DISABLE_TELNET */

/* to disable TFTP */
/* #undef CURL_DISABLE_TFTP */

/* to disable verbose strings */
/* #undef CURL_DISABLE_VERBOSE_STRINGS */

/* lber dynamic library file */
/* #undef DL_LBER_FILE */

/* ldap dynamic library file */
/* #undef DL_LDAP_FILE */

/* your Entropy Gathering Daemon socket pathname */
/* #undef EGD_SOCKET */

/* Define if you want to enable IPv6 support */
/* #undef ENABLE_IPV6 */

/* Define if struct sockaddr_in6 has the sin6_scope_id member */
/* #undef HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID */

/* Define to 1 if you have the alarm function. */
#define HAVE_ALARM 1

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Define to 1 if you have the <arpa/tftp.h> header file. */
/* #undef HAVE_ARPA_TFTP_H */

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the `basename' function. */
#define HAVE_BASENAME 1

/* Define to 1 if you have the `closesocket' function. */
/* #undef HAVE_CLOSESOCKET */

/* Define to 1 if you have the `CRYPTO_cleanup_all_ex_data' function. */
/* #undef HAVE_CRYPTO_CLEANUP_ALL_EX_DATA */
#define HAVE_CRYPTO_CLEANUP_ALL_EX_DATA 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the fcntl function. */
#define HAVE_FCNTL 1

/* Define to 1 if you have a working fcntl O_NONBLOCK function. */
#define HAVE_FCNTL_O_NONBLOCK 1

/* Define to 1 if you have the `ftruncate' function. */
#define HAVE_FTRUNCATE 1

/* Define if getaddrinfo exists and works */
/* #undef HAVE_GETADDRINFO */

/* Define to 1 if you have the `geteuid' function. */
#define HAVE_GETEUID 1

/* If you have gethostbyname */
#define HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the `gethostbyname_r' function. */
/* #undef HAVE_GETHOSTBYNAME_R */

/* gethostbyname_r() takes 3 args */
/* #undef HAVE_GETHOSTBYNAME_R_3 */

/* gethostbyname_r() takes 5 args */
/* #undef HAVE_GETHOSTBYNAME_R_5 */

/* gethostbyname_r() takes 6 args */
/* #undef HAVE_GETHOSTBYNAME_R_6 1 */

/* Define to 1 if you have the `getpass_r' function. */
/* #undef HAVE_GETPASS_R */

/* Define to 1 if you have the `getprotobyname' function. */
/* #undef HAVE_GETPROTOBYNAME */

/* Define to 1 if you have the `getpwuid' function. */
#define HAVE_GETPWUID 1

/* Define to 1 if you have the `getrlimit' function. */
/* #undef HAVE_GETRLIMIT */

/* Define to 1 if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY 1

/* we have a glibc-style strerror_r() */
/* #undef HAVE_GLIBC_STRERROR_R */
#define HAVE_GLIBC_STRERROR_R 1

/* Define to 1 if you have the `gmtime_r' function. */
#define HAVE_GMTIME_R 1

/* if you have the gssapi libraries */
/* #undef HAVE_GSSAPI */

/* if you have the GNU gssapi libraries */
/* #undef HAVE_GSSGNU */

/* if you have the Heimdal gssapi libraries */
/* #undef HAVE_GSSHEIMDAL */

/* if you have the MIT gssapi libraries */
/* #undef HAVE_GSSMIT */

/* Define to 1 if you have the `iconv' functions. */
#define HAVE_ICONV 1

/* Define to 1 if you have the `idna_strerror' function. */
/* #undef HAVE_IDNA_STRERROR */

/* Define to 1 if you have the `idn_free' function. */
/* #undef HAVE_IDN_FREE */

/* Define to 1 if you have the <idn-free.h> header file. */
/* #undef HAVE_IDN_FREE_H */

/* Define to 1 if you have the `inet_addr' function. */
#define HAVE_INET_ADDR 1

/* Define to 1 if you have a IPv6 capable working inet_ntop function. */
/* #undef HAVE_INET_NTOP */

/* Define to 1 if you have a IPv6 capable working inet_pton function. */
/* #undef HAVE_INET_PTON */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the ioctl function. */
#define HAVE_IOCTL 1

/* Define to 1 if you have a working ioctl FIONBIO function. */
#define HAVE_IOCTL_FIONBIO 1

/* Define to 1 if you have the ioctlsocket function. */
/* #undef HAVE_IOCTLSOCKET */

/* Define to 1 if you have a working ioctlsocket FIONBIO function. */
/* #undef HAVE_IOCTLSOCKET_FIONBIO */

/* Define to 1 if you have the IoctlSocket camel case function. */
/* #undef HAVE_IOCTLSOCKET_CAMEL */

/* Define to 1 if you have a working IoctlSocket camel case FIONBIO
   function. */
/* #undef HAVE_IOCTLSOCKET_CAMEL_FIONBIO */

/* Define to 1 if you have the <io.h> header file. */
/* #undef HAVE_IO_H */

/* if you have the Kerberos4 libraries (including -ldes) */
/* #undef HAVE_KRB4 */

/* Define to 1 if you have the `krb_get_our_ip_for_realm' function. */
/* #undef HAVE_KRB_GET_OUR_IP_FOR_REALM */

/* Define to 1 if you have the <krb.h> header file. */
/* #undef HAVE_KRB_H */

/* Define to 1 if you have the <libgen.h> header file. */
/* #undef HAVE_LIBGEN_H 1 */

/* Define to 1 if you have the `idn' library (-lidn). */
/* #undef HAVE_LIBIDN */

/* Define to 1 if you have the `resolv' library (-lresolv). */
/* #undef HAVE_LIBRESOLV */

/* Define to 1 if you have the `resolve' library (-lresolve). */
/* #undef HAVE_LIBRESOLVE */

/* Define to 1 if you have the `socket' library (-lsocket). */
/* #undef HAVE_LIBSOCKET */

/* if zlib is available */
/* #undef HAVE_LIBZ */

/* if your compiler supports LL */
#define HAVE_LL 1

/* Define to 1 if you have the <locale.h> header file. */
#define HAVE_LOCALE_H 1

/* Define to 1 if you have the `localtime_r' function. */
#define HAVE_LOCALTIME_R 1

/* Define to 1 if the compiler supports the 'long long' data type. */
#define HAVE_LONGLONG 1

/* Define to 1 if you need the malloc.h header file even with stdlib.h  */
/* #undef NEED_MALLOC_H */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
/* undef HAVE_NETINET_TCP_H */

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1

/* Define if NI_WITHSCOPEID exists and works */
/* #undef HAVE_NI_WITHSCOPEID */

/* we have no strerror_r() proto */
/* #undef HAVE_NO_STRERROR_R_DECL */

/* Define to 1 if you have the <openssl/crypto.h> header file. */
/* #undef HAVE_OPENSSL_CRYPTO_H */
#define HAVE_OPENSSL_CRYPTO_H 1

/* Define to 1 if you have the <openssl/err.h> header file. */
/* #undef HAVE_OPENSSL_ERR_H */
#define HAVE_OPENSSL_ERR_H 1

/* Define to 1 if you have the <openssl/pem.h> header file. */
/* #undef HAVE_OPENSSL_PEM_H */
#define HAVE_OPENSSL_PEM_H 1

/* Define to 1 if you have the <openssl/pkcs12.h> header file. */
/* #undef HAVE_OPENSSL_PKCS12_H */
#define HAVE_OPENSSL_PKCS12_H 1

/* Define to 1 if you have the <openssl/rsa.h> header file. */
/* #undef HAVE_OPENSSL_RSA_H */
#define HAVE_OPENSSL_RSA_H 1

/* Define to 1 if you have the <openssl/ssl.h> header file. */
/* #undef HAVE_OPENSSL_SSL_H */
#define HAVE_OPENSSL_SSL_H 1

/* Define to 1 if you have the <openssl/x509.h> header file. */
/* #undef HAVE_OPENSSL_X509_H */
#define HAVE_OPENSSL_X509_H 1

/* Define to 1 if you have the <pem.h> header file. */
/* #undef HAVE_PEM_H */
#define HAVE_PEM_H 1

/* Define to 1 if you have the `pipe' function. */
#define HAVE_PIPE 1

/* Define to 1 if you have the `poll' function. */
/* #undef HAVE_POLL */

/* If you have a fine poll */
/* #undef HAVE_POLL_FINE */

/* we have a POSIX-style strerror_r() */
/* #undef HAVE_POSIX_STRERROR_R */

/* Define to 1 if you have the <pwd.h> header file. */
#define HAVE_PWD_H 1

/* Define to 1 if you have the `RAND_egd' function. */
/* #undef HAVE_RAND_EGD */
#define HAVE_RAND_EGD 1

/* Define to 1 if you have the `RAND_screen' function. */
/* #undef HAVE_RAND_SCREEN */

/* Define to 1 if you have the `RAND_status' function. */
/* #undef HAVE_RAND_STATUS */
#define HAVE_RAND_STATUS 1

/* Define to 1 if you have the `select' function. */
#define HAVE_SELECT 1

/* Define to 1 if you have the <setjmp.h> header file. */
#define HAVE_SETJMP_H 1

/* Define to 1 if you have the `setlocale' function. */
#define HAVE_SETLOCALE 1

/* Define to 1 if you have the `setrlimit' function. */
#define HAVE_SETRLIMIT 1

/* Define to 1 if you have the setsockopt function. */
/* #undef HAVE_SETSOCKOPT */

/* Define to 1 if you have a working setsockopt SO_NONBLOCK function. */
/* #undef HAVE_SETSOCKOPT_SO_NONBLOCK */

/* Define to 1 if you have the `sigaction' function. */
#define HAVE_SIGACTION 1

/* Define to 1 if you have the `siginterrupt' function. */
/* #undef HAVE_SIGINTERRUPT */

/* Define to 1 if you have the `signal' function. */
#define HAVE_SIGNAL 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* If you have sigsetjmp */
/* #undef HAVE_SIGSETJMP */

/* Define to 1 if you have the `socket' function. */
#define HAVE_SOCKET 1

/* Define to 1 if you have the <ssl.h> header file. */
/* #undef HAVE_SSL_H */
#define HAVE_SSL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strcasecmp' function. */
#define HAVE_STRCASECMP 1

/* Define to 1 if you have the `strcmpi' function. */
/* #undef HAVE_STRCMPI */

/* Define to 1 if you have the `strdup' function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the `strerror_r' function. */
#define HAVE_STRERROR_R 1

/* Define to 1 if you have the `stricmp' function. */
/* #undef HAVE_STRICMP */
#define HAVE_STRICMP 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strstr' function. */
#define HAVE_STRSTR 1

/* Define to 1 if you have the `strtok_r' function. */
#define HAVE_STRTOK_R 1

/* Define to 1 if you have the `strtoll' function. */
#define HAVE_STRTOLL 1

/* if struct sockaddr_storage is defined */
/* #undef HAVE_STRUCT_SOCKADDR_STORAGE */

/* Define this if you have struct timeval */
#define HAVE_STRUCT_TIMEVAL 1

/* Define to 1 if you have the <sys/filio.h> header file. */
#define HAVE_SYS_FILIO_H 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/poll.h> header file. */
/* #undef HAVE_SYS_POLL_H */

/* Define to 1 if you have the <sys/resource.h> header file. */
#define HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/select.h> header file. */
#define HAVE_SYS_SELECT_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/sockio.h> header file. */
/* #undef HAVE_SYS_SOCKIO_H */
#define HAVE_SYS_SOCKIO_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/utime.h> header file. */
/* #undef HAVE_SYS_UTIME_H */

/* Define to 1 if you have the <termios.h> header file. */
/* #undef HAVE_TERMIOS_H */

/* Define to 1 if you have the <termio.h> header file. */
/* #undef HAVE_TERMIO_H */

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <tld.h> header file. */
/* #undef HAVE_TLD_H */

/* Define to 1 if you have the `tld_strerror' function. */
/* #undef HAVE_TLD_STRERROR */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `utime' function. */
#define HAVE_UTIME 1

/* Define to 1 if you have the <utime.h> header file. */
#define HAVE_UTIME_H 1

/* Define to 1 if you have the <winsock2.h> header file. */
/* #undef HAVE_WINSOCK2_H */

/* Define this symbol if your OS supports changing the contents of argv */
/* #undef HAVE_WRITABLE_ARGV */

/* Define to 1 if you have the ws2tcpip.h header file. */
/* #undef HAVE_WS2TCPIP_H */

/* Define to 1 if you have the <x509.h> header file. */
/* #undef HAVE_X509_H */

/* if you have the zlib.h header file */
/* #undef HAVE_ZLIB_H */

/* Define to 1 if _REENTRANT preprocessor symbol must be defined. */
/* #undef NEED_REENTRANT */

/* Define to 1 if _THREAD_SAFE preprocessor symbol must be defined. */
/* #undef NEED_THREAD_SAFE */

/* cpu-machine-OS */
#define OS "s390x-ibm-tpf"

/* Name of package */
#define PACKAGE "curl"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT \
  "a suitable curl mailing list => https://curl.se/mail/"

/* Define to the full name of this package. */
#define PACKAGE_NAME "curl"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "curl -"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "curl"

/* Define to the version of this package. */
#define PACKAGE_VERSION "-"

/* a suitable file to read random data from */
/* #undef RANDOM_FILE */

/* Define to the type of arg 1 for `select'. */
#define SELECT_TYPE_ARG1 int

/* Define to the type of args 2, 3 and 4 for `select'. */
#define SELECT_TYPE_ARG234 (fd_set *)

/* Define to the type of arg 5 for `select'. */
#define SELECT_TYPE_ARG5 (struct timeval *)

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `off_t', as computed by sizeof. */
#define SIZEOF_OFF_T 8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* Define to the size of `long', as computed by sizeof. */
#define SIZEOF_LONG 8

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 8

/* The size of `time_t', as computed by sizeof. */
#define SIZEOF_TIME_T 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* Define if you want to enable ares support */
/* #undef USE_ARES */

/* if GnuTLS is enabled */
/* #undef USE_GNUTLS */

/* If you want to build curl with the built-in manual */
/* #undef USE_MANUAL */

/* if OpenSSL is in use */
/* #undef USE_OPENSSL */

/* if SSL is enabled */
/* #undef USE_OPENSSL */

/* to enable SSPI support */
/* #undef USE_WINDOWS_SSPI */

/* Version number of package */
#define VERSION "not-used"

/* Define to avoid automatic inclusion of winsock.h */
/* #undef WIN32_LEAN_AND_MEAN */

/* Define to 1 if on AIX 3.
   System headers sometimes define this.
   We just want to avoid a redefinition error message.  */
#ifndef _ALL_SOURCE
/* # undef _ALL_SOURCE */
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* type to use in place of in_addr_t if not defined */
/* #undef in_addr_t */

/* Define to `unsigned' if <sys/types.h> does not define. */
/* #undef size_t */

/* the signed version of size_t */
/* #undef ssize_t */

/* Define to 1 if you have the recv function. */
#define HAVE_RECV 1

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

/* Define to 1 if you have the recvfrom function. */
#define HAVE_RECVFROM 1

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

/* Define to 1 if you have the send function. */
#define HAVE_SEND 1

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 int

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

#define CURL_DOES_CONVERSIONS
#ifndef CURL_ICONV_CODESET_OF_HOST
#define CURL_ICONV_CODESET_OF_HOST "IBM-1047"
#endif


#endif /* HEADER_CURL_CONFIG_TPF_H */
