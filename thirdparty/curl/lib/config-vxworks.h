#ifndef HEADER_CURL_CONFIG_VXWORKS_H
#define HEADER_CURL_CONFIG_VXWORKS_H
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

/* =============================================================== */
/*               Hand crafted config file for VxWorks              */
/* =============================================================== */

/* Location of default ca bundle */
/* #undef CURL_CA_BUNDLE */

/* Location of default ca path */
/* #undef CURL_CA_PATH */

/* to disable cookies support */
/* #undef CURL_DISABLE_COOKIES */

/* to disable cryptographic authentication */
/* #undef CURL_DISABLE_CRYPTO_AUTH */

/* to disable DICT */
/* #undef CURL_DISABLE_DICT */

/* to disable FILE */
/* #undef CURL_DISABLE_FILE */

/* to disable FTP */
#define CURL_DISABLE_FTP 1

/* to disable HTTP */
/* #undef CURL_DISABLE_HTTP */

/* to disable LDAP */
#define CURL_DISABLE_LDAP 1

/* to disable LDAPS */
#define CURL_DISABLE_LDAPS 1

/* to disable NTLM authentication */
#define CURL_DISABLE_NTLM 1

/* to disable proxies */
/* #undef CURL_DISABLE_PROXY */

/* to disable TELNET */
#define CURL_DISABLE_TELNET 1

/* to disable TFTP */
#define CURL_DISABLE_TFTP 1

/* to disable verbose strings */
/* #undef CURL_DISABLE_VERBOSE_STRINGS */

/* Definition to make a library symbol externally visible. */
/* #undef CURL_EXTERN_SYMBOL */

/* Use Windows LDAP implementation */
/* #undef USE_WIN32_LDAP */

/* your Entropy Gathering Daemon socket pathname */
/* #undef EGD_SOCKET */

/* Define if you want to enable IPv6 support */
#define ENABLE_IPV6 1

/* Define to 1 if you have the alarm function. */
#define HAVE_ALARM 1

/* Define to 1 if you have the <alloca.h> header file. */
#define HAVE_ALLOCA_H 1

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Define to 1 if you have the <arpa/tftp.h> header file. */
/* #undef HAVE_ARPA_TFTP_H */

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the `basename' function. */
/* #undef HAVE_BASENAME */

/* Define to 1 if bool is an available type. */
#define HAVE_BOOL_T 1

/* Define to 1 if you have the clock_gettime function and monotonic timer. */
/* #undef HAVE_CLOCK_GETTIME_MONOTONIC */

/* Define to 1 if you have the `closesocket' function. */
/* #undef HAVE_CLOSESOCKET */

/* Define to 1 if you have the `CRYPTO_cleanup_all_ex_data' function. */
#define HAVE_CRYPTO_CLEANUP_ALL_EX_DATA 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the fcntl function. */
#define HAVE_FCNTL 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have a working fcntl O_NONBLOCK function. */
#define HAVE_FCNTL_O_NONBLOCK 1

/* Define to 1 if you have the freeaddrinfo function. */
#define HAVE_FREEADDRINFO 1

/* Define to 1 if you have the ftruncate function. */
#define HAVE_FTRUNCATE 1

/* Define to 1 if you have a working getaddrinfo function. */
#define HAVE_GETADDRINFO 1

/* Define to 1 if you have the `geteuid' function. */
/* #undef HAVE_GETEUID */

/* Define to 1 if you have the gethostbyname function. */
#define HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the gethostbyname_r function. */
/* #undef HAVE_GETHOSTBYNAME_R */

/* gethostbyname_r() takes 3 args */
/* #undef HAVE_GETHOSTBYNAME_R_3 */

/* gethostbyname_r() takes 5 args */
/* #undef HAVE_GETHOSTBYNAME_R_5 */

/* gethostbyname_r() takes 6 args */
/* #undef HAVE_GETHOSTBYNAME_R_6 */

/* Define to 1 if you have the gethostname function. */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if you have a working getifaddrs function. */
/* #undef HAVE_GETIFADDRS */

/* Define to 1 if you have the `getpass_r' function. */
/* #undef HAVE_GETPASS_R */

/* Define to 1 if you have the `getppid' function. */
#define HAVE_GETPPID 1

/* Define to 1 if you have the `getprotobyname' function. */
#define HAVE_GETPROTOBYNAME 1

/* Define to 1 if you have the `getpwuid' function. */
/* #undef HAVE_GETPWUID */

/* Define to 1 if you have the `getrlimit' function. */
#define HAVE_GETRLIMIT 1

/* Define to 1 if you have the `gettimeofday' function. */
/* #undef HAVE_GETTIMEOFDAY */

/* Define to 1 if you have a working glibc-style strerror_r function. */
/* #undef HAVE_GLIBC_STRERROR_R */

/* Define to 1 if you have a working gmtime_r function. */
#define HAVE_GMTIME_R 1

/* if you have the gssapi libraries */
/* #undef HAVE_GSSAPI */

/* Define to 1 if you have the <gssapi/gssapi_generic.h> header file. */
/* #undef HAVE_GSSAPI_GSSAPI_GENERIC_H */

/* Define to 1 if you have the <gssapi/gssapi.h> header file. */
/* #undef HAVE_GSSAPI_GSSAPI_H */

/* Define to 1 if you have the <gssapi/gssapi_krb5.h> header file. */
/* #undef HAVE_GSSAPI_GSSAPI_KRB5_H */

/* if you have the GNU gssapi libraries */
/* #undef HAVE_GSSGNU */

/* if you have the Heimdal gssapi libraries */
/* #undef HAVE_GSSHEIMDAL */

/* if you have the MIT gssapi libraries */
/* #undef HAVE_GSSMIT */

/* Define to 1 if you have the `idna_strerror' function. */
/* #undef HAVE_IDNA_STRERROR */

/* Define to 1 if you have the `idn_free' function. */
/* #undef HAVE_IDN_FREE */

/* Define to 1 if you have the <idn-free.h> header file. */
/* #undef HAVE_IDN_FREE_H */

/* Define to 1 if you have the <ifaddrs.h> header file. */
/* #undef HAVE_IFADDRS_H */

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

/* Define to 1 if you have the ioctlsocket function. */
/* #undef HAVE_IOCTLSOCKET */

/* Define to 1 if you have the IoctlSocket camel case function. */
/* #undef HAVE_IOCTLSOCKET_CAMEL */

/* Define to 1 if you have a working IoctlSocket camel case FIONBIO function.
   */
/* #undef HAVE_IOCTLSOCKET_CAMEL_FIONBIO */

/* Define to 1 if you have a working ioctlsocket FIONBIO function. */
/* #undef HAVE_IOCTLSOCKET_FIONBIO */

/* Define to 1 if you have a working ioctl FIONBIO function. */
#define HAVE_IOCTL_FIONBIO 1

/* Define to 1 if you have a working ioctl SIOCGIFADDR function. */
#define HAVE_IOCTL_SIOCGIFADDR 1

/* Define to 1 if you have the <io.h> header file. */
#define HAVE_IO_H 1

/* if you have the Kerberos4 libraries (including -ldes) */
/* #undef HAVE_KRB4 */

/* Define to 1 if you have the `krb_get_our_ip_for_realm' function. */
/* #undef HAVE_KRB_GET_OUR_IP_FOR_REALM */

/* Define to 1 if you have the <krb.h> header file. */
/* #undef HAVE_KRB_H */

/* Define to 1 if you have the lber.h header file. */
/* #undef HAVE_LBER_H */

/* Define to 1 if you have the ldapssl.h header file. */
/* #undef HAVE_LDAPSSL_H */

/* Define to 1 if you have the ldap.h header file. */
/* #undef HAVE_LDAP_H */

/* Use LDAPS implementation */
/* #undef HAVE_LDAP_SSL */

/* Define to 1 if you have the ldap_ssl.h header file. */
/* #undef HAVE_LDAP_SSL_H */

/* Define to 1 if you have the `ldap_url_parse' function. */
/* #undef HAVE_LDAP_URL_PARSE */

/* Define to 1 if you have the <libgen.h> header file. */
/* #undef HAVE_LIBGEN_H */

/* Define to 1 if you have the `idn' library (-lidn). */
/* #undef HAVE_LIBIDN */

/* Define to 1 if you have the `resolv' library (-lresolv). */
/* #undef HAVE_LIBRESOLV */

/* Define to 1 if you have the `resolve' library (-lresolve). */
/* #undef HAVE_LIBRESOLVE */

/* Define to 1 if you have the `socket' library (-lsocket). */
/* #undef HAVE_LIBSOCKET */

/* Define to 1 if you have the `ssh2' library (-lssh2). */
/* #undef HAVE_LIBSSH2 */

/* Define to 1 if you have the <libssh2.h> header file. */
/* #undef HAVE_LIBSSH2_H */

/* Define to 1 if you have the `libssh2_version' function. */
/* #undef HAVE_LIBSSH2_VERSION */

/* if zlib is available */
#define HAVE_LIBZ 1

/* if your compiler supports LL */
#define HAVE_LL 1

/* Define to 1 if you have the <locale.h> header file. */
#define HAVE_LOCALE_H 1

/* Define to 1 if you have a working localtime_r function. */
#define HAVE_LOCALTIME_R 1

/* Define to 1 if the compiler supports the 'long long' data type. */
#define HAVE_LONGLONG 1

/* Define to 1 if you have the malloc.h header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the memory.h header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the MSG_NOSIGNAL flag. */
/* #undef HAVE_MSG_NOSIGNAL */

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#define HAVE_NETINET_TCP_H 1

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1

/* Define to 1 if NI_WITHSCOPEID exists and works. */
/* #undef HAVE_NI_WITHSCOPEID */

/* if you have an old MIT gssapi library, lacking GSS_C_NT_HOSTBASED_SERVICE
   */
/* #undef HAVE_OLD_GSSMIT */

/* Define to 1 if you have the <openssl/crypto.h> header file. */
#define HAVE_OPENSSL_CRYPTO_H 1

/* Define to 1 if you have the <openssl/err.h> header file. */
#define HAVE_OPENSSL_ERR_H 1

/* Define to 1 if you have the <openssl/pem.h> header file. */
#define HAVE_OPENSSL_PEM_H 1

/* Define to 1 if you have the <openssl/pkcs12.h> header file. */
#define HAVE_OPENSSL_PKCS12_H 1

/* Define to 1 if you have the <openssl/rsa.h> header file. */
#define HAVE_OPENSSL_RSA_H 1

/* Define to 1 if you have the <openssl/ssl.h> header file. */
#define HAVE_OPENSSL_SSL_H 1

/* Define to 1 if you have the <openssl/x509.h> header file. */
#define HAVE_OPENSSL_X509_H 1

/* Define to 1 if you have the <pem.h> header file. */
/* #undef HAVE_PEM_H */

/* Define to 1 if you have the `pipe' function. */
#define HAVE_PIPE 1

/* Define to 1 if you have a working poll function. */
/* #undef HAVE_POLL */

/* If you have a fine poll */
/* #undef HAVE_POLL_FINE */

/* Define to 1 if you have the <poll.h> header file. */
/* #undef HAVE_POLL_H */

/* Define to 1 if you have a working POSIX-style strerror_r function. */
/* #undef HAVE_POSIX_STRERROR_R */

/* Define to 1 if you have the <pwd.h> header file. */
/* #undef HAVE_PWD_H */

/* Define to 1 if you have the `RAND_egd' function. */
#define HAVE_RAND_EGD 1

/* Define to 1 if you have the `RAND_screen' function. */
/* #undef HAVE_RAND_SCREEN */

/* Define to 1 if you have the `RAND_status' function. */
#define HAVE_RAND_STATUS 1

/* Define to 1 if you have the recv function. */
#define HAVE_RECV 1

/* Define to 1 if you have the recvfrom function. */
#define HAVE_RECVFROM 1

/* Define to 1 if you have the select function. */
#define HAVE_SELECT 1

/* Define to 1 if you have the send function. */
#define HAVE_SEND 1

/* Define to 1 if you have the <setjmp.h> header file. */
#define HAVE_SETJMP_H 1

/* Define to 1 if you have the `setlocale' function. */
#define HAVE_SETLOCALE 1

/* Define to 1 if you have the `setmode' function. */
#define HAVE_SETMODE 1

/* Define to 1 if you have the `setrlimit' function. */
#define HAVE_SETRLIMIT 1

/* Define to 1 if you have the setsockopt function. */
#define HAVE_SETSOCKOPT 1

/* Define to 1 if you have a working setsockopt SO_NONBLOCK function. */
/* #undef HAVE_SETSOCKOPT_SO_NONBLOCK */

/* Define to 1 if you have the sigaction function. */
#define HAVE_SIGACTION 1

/* Define to 1 if you have the siginterrupt function. */
#define HAVE_SIGINTERRUPT 1

/* Define to 1 if you have the signal function. */
#define HAVE_SIGNAL 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the sigsetjmp function or macro. */
/* #undef HAVE_SIGSETJMP */

/* Define to 1 if struct sockaddr_in6 has the sin6_scope_id member */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* Define to 1 if you have the `socket' function. */
#define HAVE_SOCKET 1

/* Define to 1 if you have the <ssl.h> header file. */
/* #undef HAVE_SSL_H */

/* Define to 1 if you have the <stdbool.h> header file. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
/* #undef HAVE_STDINT_H */

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the strcasecmp function. */
#define HAVE_STRCASECMP 1

/* Define to 1 if you have the strcmpi function. */
/* #undef HAVE_STRCMPI */

/* Define to 1 if you have the strdup function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the strerror_r function. */
#define HAVE_STRERROR_R 1

/* Define to 1 if you have the stricmp function. */
/* #undef HAVE_STRICMP */

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the strncmpi function. */
/* #undef HAVE_STRNCMPI */

/* Define to 1 if you have the strnicmp function. */
/* #undef HAVE_STRNICMP */

/* Define to 1 if you have the <stropts.h> header file. */
/* #undef HAVE_STROPTS_H */

/* Define to 1 if you have the strstr function. */
#define HAVE_STRSTR 1

/* Define to 1 if you have the strtok_r function. */
#define HAVE_STRTOK_R 1

/* Define to 1 if you have the strtoll function. */
/* #undef HAVE_STRTOLL */

/* if struct sockaddr_storage is defined */
#define HAVE_STRUCT_SOCKADDR_STORAGE 1

/* Define to 1 if you have the timeval struct. */
#define HAVE_STRUCT_TIMEVAL 1

/* Define to 1 if you have the <sys/filio.h> header file. */
/* #undef HAVE_SYS_FILIO_H */

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
/* #undef HAVE_SYS_PARAM_H */

/* Define to 1 if you have the <sys/poll.h> header file. */
/* #undef HAVE_SYS_POLL_H */

/* Define to 1 if you have the <sys/resource.h> header file. */
#define HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/select.h> header file. */
/* #undef HAVE_SYS_SELECT_H */

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/sockio.h> header file. */
/* #undef HAVE_SYS_SOCKIO_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
/* #undef HAVE_SYS_TIME_H */

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#define HAVE_SYS_UIO_H 1

/* Define to 1 if you have the <sys/un.h> header file. */
#define HAVE_SYS_UN_H 1

/* Define to 1 if you have the <sys/utime.h> header file. */
#define HAVE_SYS_UTIME_H 1

/* Define to 1 if you have the <termios.h> header file. */
#define HAVE_TERMIOS_H 1

/* Define to 1 if you have the <termio.h> header file. */
#define HAVE_TERMIO_H 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <tld.h> header file. */
/* #undef HAVE_TLD_H */

/* Define to 1 if you have the `tld_strerror' function. */
/* #undef HAVE_TLD_STRERROR */

/* Define to 1 if you have the `uname' function. */
#define HAVE_UNAME 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `utime' function. */
#define HAVE_UTIME 1

/* Define to 1 if you have the <utime.h> header file. */
#define HAVE_UTIME_H 1

/* Define to 1 if compiler supports C99 variadic macro style. */
#define HAVE_VARIADIC_MACROS_C99 1

/* Define to 1 if compiler supports old gcc variadic macro style. */
#define HAVE_VARIADIC_MACROS_GCC 1

/* Define to 1 if you have a working vxworks-style strerror_r function. */
#define HAVE_VXWORKS_STRERROR_R 1

/* Define to 1 if you have the winber.h header file. */
/* #undef HAVE_WINBER_H */

/* Define to 1 if you have the windows.h header file. */
/* #undef HAVE_WINDOWS_H */

/* Define to 1 if you have the winldap.h header file. */
/* #undef HAVE_WINLDAP_H */

/* Define to 1 if you have the winsock2.h header file. */
/* #undef HAVE_WINSOCK2_H */

/* Define this symbol if your OS supports changing the contents of argv */
#define HAVE_WRITABLE_ARGV 1

/* Define to 1 if you have the writev function. */
#define HAVE_WRITEV 1

/* Define to 1 if you have the ws2tcpip.h header file. */
/* #undef HAVE_WS2TCPIP_H */

/* Define to 1 if you have the <x509.h> header file. */
/* #undef HAVE_X509_H */

/* if you have the zlib.h header file */
#define HAVE_ZLIB_H 1

/* Define to 1 if you need the lber.h header file even with ldap.h */
/* #undef NEED_LBER_H */

/* Define to 1 if you need the malloc.h header file even with stdlib.h */
/* #undef NEED_MALLOC_H */

/* Define to 1 if you need the memory.h header file even with stdlib.h */
/* #undef NEED_MEMORY_H */

/* Define to 1 if _REENTRANT preprocessor symbol must be defined. */
/* #undef NEED_REENTRANT */

/* Define to 1 if _THREAD_SAFE preprocessor symbol must be defined. */
/* #undef NEED_THREAD_SAFE */

/* Define to 1 if the open function requires three arguments. */
#define OPEN_NEEDS_ARG3 1

/* cpu-machine-OS */
#define OS "unknown-unknown-vxworks"

/* Name of package */
#define PACKAGE "curl"

/* a suitable file to read random data from */
#define RANDOM_FILE "/dev/urandom"

/* Define to the type of arg 1 for recvfrom. */
#define RECVFROM_TYPE_ARG1 int

/* Define to the type pointed by arg 2 for recvfrom. */
#define RECVFROM_TYPE_ARG2 void

/* Define to 1 if the type pointed by arg 2 for recvfrom is void. */
#define RECVFROM_TYPE_ARG2_IS_VOID 1

/* Define to the type of arg 3 for recvfrom. */
#define RECVFROM_TYPE_ARG3 size_t

/* Define to the type of arg 4 for recvfrom. */
#define RECVFROM_TYPE_ARG4 int

/* Define to the type pointed by arg 5 for recvfrom. */
#define RECVFROM_TYPE_ARG5 struct sockaddr

/* Define to 1 if the type pointed by arg 5 for recvfrom is void. */
/* #undef RECVFROM_TYPE_ARG5_IS_VOID */

/* Define to the type pointed by arg 6 for recvfrom. */
#define RECVFROM_TYPE_ARG6 socklen_t

/* Define to 1 if the type pointed by arg 6 for recvfrom is void. */
/* #undef RECVFROM_TYPE_ARG6_IS_VOID */

/* Define to the function return type for recvfrom. */
#define RECVFROM_TYPE_RETV int

/* Define to the type of arg 1 for recv. */
#define RECV_TYPE_ARG1 int

/* Define to the type of arg 2 for recv. */
#define RECV_TYPE_ARG2 void *

/* Define to the type of arg 3 for recv. */
#define RECV_TYPE_ARG3 size_t

/* Define to the type of arg 4 for recv. */
#define RECV_TYPE_ARG4 int

/* Define to the function return type for recv. */
#define RECV_TYPE_RETV int

/* Define to the type qualifier of arg 5 for select. */
#define SELECT_QUAL_ARG5

/* Define to the type of arg 1 for select. */
#define SELECT_TYPE_ARG1 int

/* Define to the type of args 2, 3 and 4 for select. */
#define SELECT_TYPE_ARG234 fd_set *

/* Define to the type of arg 5 for select. */
#define SELECT_TYPE_ARG5 struct timeval *

/* Define to the function return type for select. */
#define SELECT_TYPE_RETV int

/* Define to the type qualifier of arg 2 for send. */
#define SEND_QUAL_ARG2 const

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 int

/* Define to the type of arg 2 for send. */
#define SEND_TYPE_ARG2 void *

/* Define to the type of arg 3 for send. */
#define SEND_TYPE_ARG3 size_t

/* Define to the type of arg 4 for send. */
#define SEND_TYPE_ARG4 int

/* Define to the function return type for send. */
#define SEND_TYPE_RETV int

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG 4

/* The size of `off_t', as computed by sizeof. */
#define SIZEOF_OFF_T 8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 4

/* The size of `time_t', as computed by sizeof. */
#define SIZEOF_TIME_T 4

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to the type of arg 3 for strerror_r. */
/* #undef STRERROR_R_TYPE_ARG3 */

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
/* #undef TIME_WITH_SYS_TIME */

/* Define if you want to enable c-ares support */
/* #undef USE_ARES */

/* if GnuTLS is enabled */
/* #undef USE_GNUTLS */

/* if libSSH2 is in use */
/* #undef USE_LIBSSH2 */

/* If you want to build curl with the built-in manual */
#define USE_MANUAL 1

/* if NSS is enabled */
/* #undef USE_NSS */

/* if OpenSSL is in use */
#define USE_OPENSSL 1

/* Define to 1 if you are building a Windows target without large file
   support. */
/* #undef USE_WIN32_LARGE_FILES */

/* to enable SSPI support */
/* #undef USE_WINDOWS_SSPI */

/* Define to 1 if using yaSSL in OpenSSL compatibility mode. */
/* #undef USE_YASSLEMUL */

/* Define to avoid automatic inclusion of winsock.h */
/* #undef WIN32_LEAN_AND_MEAN */

/* Define to 1 if OS is AIX. */
#ifndef _ALL_SOURCE
/* #  undef _ALL_SOURCE */
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Type to use in place of in_addr_t when system does not provide it. */
/* #undef in_addr_t */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */

/* the signed version of size_t */
/* #undef ssize_t */

#endif /* HEADER_CURL_CONFIG_VXWORKS_H */
