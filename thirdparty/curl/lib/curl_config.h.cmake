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
/* lib/curl_config.h.in.  Generated somehow by cmake.  */

/* when building libcurl itself */
#cmakedefine BUILDING_LIBCURL 1

/* Location of default ca bundle */
#cmakedefine CURL_CA_BUNDLE "${CURL_CA_BUNDLE}"

/* define "1" to use built-in ca store of TLS backend */
#cmakedefine CURL_CA_FALLBACK 1

/* Location of default ca path */
#cmakedefine CURL_CA_PATH "${CURL_CA_PATH}"

/* disables alt-svc */
#cmakedefine CURL_DISABLE_ALTSVC 1

/* disables cookies support */
#cmakedefine CURL_DISABLE_COOKIES 1

/* disables cryptographic authentication */
#cmakedefine CURL_DISABLE_CRYPTO_AUTH 1

/* disables DICT */
#cmakedefine CURL_DISABLE_DICT 1

/* disables DNS-over-HTTPS */
#cmakedefine CURL_DISABLE_DOH 1

/* disables FILE */
#cmakedefine CURL_DISABLE_FILE 1

/* disables FTP */
#cmakedefine CURL_DISABLE_FTP 1

/* disables GOPHER */
#cmakedefine CURL_DISABLE_GOPHER 1

/* disables HSTS support */
#cmakedefine CURL_DISABLE_HSTS 1

/* disables HTTP */
#cmakedefine CURL_DISABLE_HTTP 1

/* disables IMAP */
#cmakedefine CURL_DISABLE_IMAP 1

/* disables LDAP */
#cmakedefine CURL_DISABLE_LDAP 1

/* disables LDAPS */
#cmakedefine CURL_DISABLE_LDAPS 1

/* disables --libcurl option from the curl tool */
#cmakedefine CURL_DISABLE_LIBCURL_OPTION 1

/* disables MIME support */
#cmakedefine CURL_DISABLE_MIME 1

/* disables MQTT */
#cmakedefine CURL_DISABLE_MQTT 1

/* disables netrc parser */
#cmakedefine CURL_DISABLE_NETRC 1

/* disables NTLM support */
#cmakedefine CURL_DISABLE_NTLM 1

/* disables date parsing */
#cmakedefine CURL_DISABLE_PARSEDATE 1

/* disables POP3 */
#cmakedefine CURL_DISABLE_POP3 1

/* disables built-in progress meter */
#cmakedefine CURL_DISABLE_PROGRESS_METER 1

/* disables proxies */
#cmakedefine CURL_DISABLE_PROXY 1

/* disables RTSP */
#cmakedefine CURL_DISABLE_RTSP 1

/* disables SMB */
#cmakedefine CURL_DISABLE_SMB 1

/* disables SMTP */
#cmakedefine CURL_DISABLE_SMTP 1

/* disables use of socketpair for curl_multi_poll */
#cmakedefine CURL_DISABLE_SOCKETPAIR 1

/* disables TELNET */
#cmakedefine CURL_DISABLE_TELNET 1

/* disables TFTP */
#cmakedefine CURL_DISABLE_TFTP 1

/* disables verbose strings */
#cmakedefine CURL_DISABLE_VERBOSE_STRINGS 1

/* to make a symbol visible */
#cmakedefine CURL_EXTERN_SYMBOL ${CURL_EXTERN_SYMBOL}
/* Ensure using CURL_EXTERN_SYMBOL is possible */
#ifndef CURL_EXTERN_SYMBOL
#define CURL_EXTERN_SYMBOL
#endif

/* Allow SMB to work on Windows */
#cmakedefine USE_WIN32_CRYPTO 1

/* Use Windows LDAP implementation */
#cmakedefine USE_WIN32_LDAP 1

/* when not building a shared library */
#cmakedefine CURL_STATICLIB 1

/* your Entropy Gathering Daemon socket pathname */
#cmakedefine EGD_SOCKET ${EGD_SOCKET}

/* Define if you want to enable IPv6 support */
#cmakedefine ENABLE_IPV6 1

/* Define to 1 if you have the alarm function. */
#cmakedefine HAVE_ALARM 1

/* Define to 1 if you have the <alloca.h> header file. */
#cmakedefine HAVE_ALLOCA_H 1

/* Define to 1 if you have the <arpa/inet.h> header file. */
#cmakedefine HAVE_ARPA_INET_H 1

/* Define to 1 if you have the <arpa/tftp.h> header file. */
#cmakedefine HAVE_ARPA_TFTP_H 1

/* Define to 1 if you have the <assert.h> header file. */
#cmakedefine HAVE_ASSERT_H 1

/* Define to 1 if you have the `basename' function. */
#cmakedefine HAVE_BASENAME 1

/* Define to 1 if bool is an available type. */
#cmakedefine HAVE_BOOL_T 1

/* Define to 1 if you have the __builtin_available function. */
#cmakedefine HAVE_BUILTIN_AVAILABLE 1

/* Define to 1 if you have the clock_gettime function and monotonic timer. */
#cmakedefine HAVE_CLOCK_GETTIME_MONOTONIC 1

/* Define to 1 if you have the `closesocket' function. */
#cmakedefine HAVE_CLOSESOCKET 1

/* Define to 1 if you have the `CRYPTO_cleanup_all_ex_data' function. */
#cmakedefine HAVE_CRYPTO_CLEANUP_ALL_EX_DATA 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#cmakedefine HAVE_DLFCN_H 1

/* Define to 1 if you have the <errno.h> header file. */
#cmakedefine HAVE_ERRNO_H 1

/* Define to 1 if you have the fcntl function. */
#cmakedefine HAVE_FCNTL 1

/* Define to 1 if you have the <fcntl.h> header file. */
#cmakedefine HAVE_FCNTL_H 1

/* Define to 1 if you have a working fcntl O_NONBLOCK function. */
#cmakedefine HAVE_FCNTL_O_NONBLOCK 1

/* Define to 1 if you have the freeaddrinfo function. */
#cmakedefine HAVE_FREEADDRINFO 1

/* Define to 1 if you have the ftruncate function. */
#cmakedefine HAVE_FTRUNCATE 1

/* Define to 1 if you have a working getaddrinfo function. */
#cmakedefine HAVE_GETADDRINFO 1

/* Define to 1 if you have the `geteuid' function. */
#cmakedefine HAVE_GETEUID 1

/* Define to 1 if you have the `getppid' function. */
#cmakedefine HAVE_GETPPID 1

/* Define to 1 if you have the gethostbyname function. */
#cmakedefine HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the gethostbyname_r function. */
#cmakedefine HAVE_GETHOSTBYNAME_R 1

/* gethostbyname_r() takes 3 args */
#cmakedefine HAVE_GETHOSTBYNAME_R_3 1

/* gethostbyname_r() takes 5 args */
#cmakedefine HAVE_GETHOSTBYNAME_R_5 1

/* gethostbyname_r() takes 6 args */
#cmakedefine HAVE_GETHOSTBYNAME_R_6 1

/* Define to 1 if you have the gethostname function. */
#cmakedefine HAVE_GETHOSTNAME 1

/* Define to 1 if you have a working getifaddrs function. */
#cmakedefine HAVE_GETIFADDRS 1

/* Define to 1 if you have the `getpass_r' function. */
#cmakedefine HAVE_GETPASS_R 1

/* Define to 1 if you have the `getppid' function. */
#cmakedefine HAVE_GETPPID 1

/* Define to 1 if you have the `getprotobyname' function. */
#cmakedefine HAVE_GETPROTOBYNAME 1

/* Define to 1 if you have the `getpeername' function. */
#cmakedefine HAVE_GETPEERNAME 1

/* Define to 1 if you have the `getsockname' function. */
#cmakedefine HAVE_GETSOCKNAME 1

/* Define to 1 if you have the `if_nametoindex' function. */
#cmakedefine HAVE_IF_NAMETOINDEX 1

/* Define to 1 if you have the `getpwuid' function. */
#cmakedefine HAVE_GETPWUID 1

/* Define to 1 if you have the `getpwuid_r' function. */
#cmakedefine HAVE_GETPWUID_R 1

/* Define to 1 if you have the `getrlimit' function. */
#cmakedefine HAVE_GETRLIMIT 1

/* Define to 1 if you have the `gettimeofday' function. */
#cmakedefine HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have a working glibc-style strerror_r function. */
#cmakedefine HAVE_GLIBC_STRERROR_R 1

/* Define to 1 if you have a working gmtime_r function. */
#cmakedefine HAVE_GMTIME_R 1

/* if you have the gssapi libraries */
#cmakedefine HAVE_GSSAPI 1

/* Define to 1 if you have the <gssapi/gssapi_generic.h> header file. */
#cmakedefine HAVE_GSSAPI_GSSAPI_GENERIC_H 1

/* Define to 1 if you have the <gssapi/gssapi.h> header file. */
#cmakedefine HAVE_GSSAPI_GSSAPI_H 1

/* Define to 1 if you have the <gssapi/gssapi_krb5.h> header file. */
#cmakedefine HAVE_GSSAPI_GSSAPI_KRB5_H 1

/* if you have the GNU gssapi libraries */
#cmakedefine HAVE_GSSGNU 1

/* if you have the Heimdal gssapi libraries */
#cmakedefine HAVE_GSSHEIMDAL 1

/* if you have the MIT gssapi libraries */
#cmakedefine HAVE_GSSMIT 1

/* Define to 1 if you have the `idna_strerror' function. */
#cmakedefine HAVE_IDNA_STRERROR 1

/* Define to 1 if you have the `idn_free' function. */
#cmakedefine HAVE_IDN_FREE 1

/* Define to 1 if you have the <idn-free.h> header file. */
#cmakedefine HAVE_IDN_FREE_H 1

/* Define to 1 if you have the <ifaddrs.h> header file. */
#cmakedefine HAVE_IFADDRS_H 1

/* Define to 1 if you have the `inet_addr' function. */
#cmakedefine HAVE_INET_ADDR 1

/* Define to 1 if you have a IPv6 capable working inet_ntop function. */
#cmakedefine HAVE_INET_NTOP 1

/* Define to 1 if you have a IPv6 capable working inet_pton function. */
#cmakedefine HAVE_INET_PTON 1

/* Define to 1 if symbol `sa_family_t' exists */
#cmakedefine HAVE_SA_FAMILY_T 1

/* Define to 1 if symbol `ADDRESS_FAMILY' exists */
#cmakedefine HAVE_ADDRESS_FAMILY 1

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine HAVE_INTTYPES_H 1

/* Define to 1 if you have the ioctl function. */
#cmakedefine HAVE_IOCTL 1

/* Define to 1 if you have the ioctlsocket function. */
#cmakedefine HAVE_IOCTLSOCKET 1

/* Define to 1 if you have the IoctlSocket camel case function. */
#cmakedefine HAVE_IOCTLSOCKET_CAMEL 1

/* Define to 1 if you have a working IoctlSocket camel case FIONBIO function.
   */
#cmakedefine HAVE_IOCTLSOCKET_CAMEL_FIONBIO 1

/* Define to 1 if you have a working ioctlsocket FIONBIO function. */
#cmakedefine HAVE_IOCTLSOCKET_FIONBIO 1

/* Define to 1 if you have a working ioctl FIONBIO function. */
#cmakedefine HAVE_IOCTL_FIONBIO 1

/* Define to 1 if you have a working ioctl SIOCGIFADDR function. */
#cmakedefine HAVE_IOCTL_SIOCGIFADDR 1

/* Define to 1 if you have the <io.h> header file. */
#cmakedefine HAVE_IO_H 1

/* if you have the Kerberos4 libraries (including -ldes) */
#cmakedefine HAVE_KRB4 1

/* Define to 1 if you have the `krb_get_our_ip_for_realm' function. */
#cmakedefine HAVE_KRB_GET_OUR_IP_FOR_REALM 1

/* Define to 1 if you have the <krb.h> header file. */
#cmakedefine HAVE_KRB_H 1

/* Define to 1 if you have the lber.h header file. */
#cmakedefine HAVE_LBER_H 1

/* Define to 1 if you have the ldapssl.h header file. */
#cmakedefine HAVE_LDAPSSL_H 1

/* Define to 1 if you have the ldap.h header file. */
#cmakedefine HAVE_LDAP_H 1

/* Use LDAPS implementation */
#cmakedefine HAVE_LDAP_SSL 1

/* Define to 1 if you have the ldap_ssl.h header file. */
#cmakedefine HAVE_LDAP_SSL_H 1

/* Define to 1 if you have the `ldap_url_parse' function. */
#cmakedefine HAVE_LDAP_URL_PARSE 1

/* Define to 1 if you have the <libgen.h> header file. */
#cmakedefine HAVE_LIBGEN_H 1

/* Define to 1 if you have the `idn2' library (-lidn2). */
#cmakedefine HAVE_LIBIDN2 1

/* Define to 1 if you have the idn2.h header file. */
#cmakedefine HAVE_IDN2_H 1

/* Define to 1 if you have the `resolv' library (-lresolv). */
#cmakedefine HAVE_LIBRESOLV 1

/* Define to 1 if you have the `resolve' library (-lresolve). */
#cmakedefine HAVE_LIBRESOLVE 1

/* Define to 1 if you have the `socket' library (-lsocket). */
#cmakedefine HAVE_LIBSOCKET 1

/* Define to 1 if you have the `ssh2' library (-lssh2). */
#cmakedefine HAVE_LIBSSH2 1

/* Define to 1 if you have the <libssh2.h> header file. */
#cmakedefine HAVE_LIBSSH2_H 1

/* Define to 1 if you have the <libssh/libssh.h> header file. */
#cmakedefine HAVE_LIBSSH_LIBSSH_H 1

/* if zlib is available */
#cmakedefine HAVE_LIBZ 1

/* if brotli is available */
#cmakedefine HAVE_BROTLI 1

/* if zstd is available */
#cmakedefine HAVE_ZSTD 1

/* if your compiler supports LL */
#cmakedefine HAVE_LL 1

/* Define to 1 if you have the <locale.h> header file. */
#cmakedefine HAVE_LOCALE_H 1

/* Define to 1 if you have a working localtime_r function. */
#cmakedefine HAVE_LOCALTIME_R 1

/* Define to 1 if the compiler supports the 'long long' data type. */
#cmakedefine HAVE_LONGLONG 1

/* Define to 1 if you have the malloc.h header file. */
#cmakedefine HAVE_MALLOC_H 1

/* Define to 1 if you have the <memory.h> header file. */
#cmakedefine HAVE_MEMORY_H 1

/* Define to 1 if you have the MSG_NOSIGNAL flag. */
#cmakedefine HAVE_MSG_NOSIGNAL 1

/* Define to 1 if you have the <netdb.h> header file. */
#cmakedefine HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#cmakedefine HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#cmakedefine HAVE_NETINET_TCP_H 1

/* Define to 1 if you have the <linux/tcp.h> header file. */
#cmakedefine HAVE_LINUX_TCP_H 1

/* Define to 1 if you have the <net/if.h> header file. */
#cmakedefine HAVE_NET_IF_H 1

/* Define to 1 if NI_WITHSCOPEID exists and works. */
#cmakedefine HAVE_NI_WITHSCOPEID 1

/* if you have an old MIT gssapi library, lacking GSS_C_NT_HOSTBASED_SERVICE */
#cmakedefine HAVE_OLD_GSSMIT 1

/* Define to 1 if you have the <openssl/crypto.h> header file. */
#cmakedefine HAVE_OPENSSL_CRYPTO_H 1

/* Define to 1 if you have the <openssl/err.h> header file. */
#cmakedefine HAVE_OPENSSL_ERR_H 1

/* Define to 1 if you have the <openssl/pem.h> header file. */
#cmakedefine HAVE_OPENSSL_PEM_H 1

/* Define to 1 if you have the <openssl/pkcs12.h> header file. */
#cmakedefine HAVE_OPENSSL_PKCS12_H 1

/* Define to 1 if you have the <openssl/rsa.h> header file. */
#cmakedefine HAVE_OPENSSL_RSA_H 1

/* Define to 1 if you have the <openssl/ssl.h> header file. */
#cmakedefine HAVE_OPENSSL_SSL_H 1

/* Define to 1 if you have the <openssl/x509.h> header file. */
#cmakedefine HAVE_OPENSSL_X509_H 1

/* Define to 1 if you have the <pem.h> header file. */
#cmakedefine HAVE_PEM_H 1

/* Define to 1 if you have the `pipe' function. */
#cmakedefine HAVE_PIPE 1

/* Define to 1 if you have a working poll function. */
#cmakedefine HAVE_POLL 1

/* If you have a fine poll */
#cmakedefine HAVE_POLL_FINE 1

/* Define to 1 if you have the <poll.h> header file. */
#cmakedefine HAVE_POLL_H 1

/* Define to 1 if you have a working POSIX-style strerror_r function. */
#cmakedefine HAVE_POSIX_STRERROR_R 1

/* Define to 1 if you have the <pthread.h> header file */
#cmakedefine HAVE_PTHREAD_H 1

/* Define to 1 if you have the <pwd.h> header file. */
#cmakedefine HAVE_PWD_H 1

/* Define to 1 if you have the `RAND_egd' function. */
#cmakedefine HAVE_RAND_EGD 1

/* Define to 1 if you have the `RAND_screen' function. */
#cmakedefine HAVE_RAND_SCREEN 1

/* Define to 1 if you have the `RAND_status' function. */
#cmakedefine HAVE_RAND_STATUS 1

/* Define to 1 if you have the recv function. */
#cmakedefine HAVE_RECV 1

/* Define to 1 if you have the recvfrom function. */
#cmakedefine HAVE_RECVFROM 1

/* Define to 1 if you have the select function. */
#cmakedefine HAVE_SELECT 1

/* Define to 1 if you have the send function. */
#cmakedefine HAVE_SEND 1

/* Define to 1 if you have the 'fsetxattr' function. */
#cmakedefine HAVE_FSETXATTR 1

/* fsetxattr() takes 5 args */
#cmakedefine HAVE_FSETXATTR_5 1

/* fsetxattr() takes 6 args */
#cmakedefine HAVE_FSETXATTR_6 1

/* Define to 1 if you have the <setjmp.h> header file. */
#cmakedefine HAVE_SETJMP_H 1

/* Define to 1 if you have the `setlocale' function. */
#cmakedefine HAVE_SETLOCALE 1

/* Define to 1 if you have the `setmode' function. */
#cmakedefine HAVE_SETMODE 1

/* Define to 1 if you have the `setrlimit' function. */
#cmakedefine HAVE_SETRLIMIT 1

/* Define to 1 if you have the setsockopt function. */
#cmakedefine HAVE_SETSOCKOPT 1

/* Define to 1 if you have a working setsockopt SO_NONBLOCK function. */
#cmakedefine HAVE_SETSOCKOPT_SO_NONBLOCK 1

/* Define to 1 if you have the sigaction function. */
#cmakedefine HAVE_SIGACTION 1

/* Define to 1 if you have the siginterrupt function. */
#cmakedefine HAVE_SIGINTERRUPT 1

/* Define to 1 if you have the signal function. */
#cmakedefine HAVE_SIGNAL 1

/* Define to 1 if you have the <signal.h> header file. */
#cmakedefine HAVE_SIGNAL_H 1

/* Define to 1 if you have the sigsetjmp function or macro. */
#cmakedefine HAVE_SIGSETJMP 1

/* Define to 1 if struct sockaddr_in6 has the sin6_scope_id member */
#cmakedefine HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* Define to 1 if you have the `socket' function. */
#cmakedefine HAVE_SOCKET 1

/* Define to 1 if you have the <ssl.h> header file. */
#cmakedefine HAVE_SSL_H 1

/* Define to 1 if you have the <stdbool.h> header file. */
#cmakedefine HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#cmakedefine HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#cmakedefine HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#cmakedefine HAVE_STDLIB_H 1

/* Define to 1 if you have the strcasecmp function. */
#cmakedefine HAVE_STRCASECMP 1

/* Define to 1 if you have the strcasestr function. */
#cmakedefine HAVE_STRCASESTR 1

/* Define to 1 if you have the strcmpi function. */
#cmakedefine HAVE_STRCMPI 1

/* Define to 1 if you have the strdup function. */
#cmakedefine HAVE_STRDUP 1

/* Define to 1 if you have the strerror_r function. */
#cmakedefine HAVE_STRERROR_R 1

/* Define to 1 if you have the stricmp function. */
#cmakedefine HAVE_STRICMP 1

/* Define to 1 if you have the <strings.h> header file. */
#cmakedefine HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#cmakedefine HAVE_STRING_H 1

/* Define to 1 if you have the strncmpi function. */
#cmakedefine HAVE_STRNCMPI 1

/* Define to 1 if you have the strnicmp function. */
#cmakedefine HAVE_STRNICMP 1

/* Define to 1 if you have the <stropts.h> header file. */
#cmakedefine HAVE_STROPTS_H 1

/* Define to 1 if you have the strstr function. */
#cmakedefine HAVE_STRSTR 1

/* Define to 1 if you have the strtok_r function. */
#cmakedefine HAVE_STRTOK_R 1

/* Define to 1 if you have the strtoll function. */
#cmakedefine HAVE_STRTOLL 1

/* if struct sockaddr_storage is defined */
#cmakedefine HAVE_STRUCT_SOCKADDR_STORAGE 1

/* Define to 1 if you have the timeval struct. */
#cmakedefine HAVE_STRUCT_TIMEVAL 1

/* Define to 1 if you have the <sys/filio.h> header file. */
#cmakedefine HAVE_SYS_FILIO_H 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#cmakedefine HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#cmakedefine HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/poll.h> header file. */
#cmakedefine HAVE_SYS_POLL_H 1

/* Define to 1 if you have the <sys/resource.h> header file. */
#cmakedefine HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/select.h> header file. */
#cmakedefine HAVE_SYS_SELECT_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#cmakedefine HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/sockio.h> header file. */
#cmakedefine HAVE_SYS_SOCKIO_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#cmakedefine HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#cmakedefine HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#cmakedefine HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#cmakedefine HAVE_SYS_UIO_H 1

/* Define to 1 if you have the <sys/un.h> header file. */
#cmakedefine HAVE_SYS_UN_H 1

/* Define to 1 if you have the <sys/utime.h> header file. */
#cmakedefine HAVE_SYS_UTIME_H 1

/* Define to 1 if you have the <termios.h> header file. */
#cmakedefine HAVE_TERMIOS_H 1

/* Define to 1 if you have the <termio.h> header file. */
#cmakedefine HAVE_TERMIO_H 1

/* Define to 1 if you have the <time.h> header file. */
#cmakedefine HAVE_TIME_H 1

/* Define to 1 if you have the <tld.h> header file. */
#cmakedefine HAVE_TLD_H 1

/* Define to 1 if you have the `tld_strerror' function. */
#cmakedefine HAVE_TLD_STRERROR 1

/* Define to 1 if you have the `uname' function. */
#cmakedefine HAVE_UNAME 1

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine HAVE_UNISTD_H 1

/* Define to 1 if you have the `utime' function. */
#cmakedefine HAVE_UTIME 1

/* Define to 1 if you have the `utimes' function. */
#cmakedefine HAVE_UTIMES 1

/* Define to 1 if you have the <utime.h> header file. */
#cmakedefine HAVE_UTIME_H 1

/* Define to 1 if compiler supports C99 variadic macro style. */
#cmakedefine HAVE_VARIADIC_MACROS_C99 1

/* Define to 1 if compiler supports old gcc variadic macro style. */
#cmakedefine HAVE_VARIADIC_MACROS_GCC 1

/* Define to 1 if you have the winber.h header file. */
#cmakedefine HAVE_WINBER_H 1

/* Define to 1 if you have the windows.h header file. */
#cmakedefine HAVE_WINDOWS_H 1

/* Define to 1 if you have the winldap.h header file. */
#cmakedefine HAVE_WINLDAP_H 1

/* Define to 1 if you have the winsock2.h header file. */
#cmakedefine HAVE_WINSOCK2_H 1

/* Define this symbol if your OS supports changing the contents of argv */
#cmakedefine HAVE_WRITABLE_ARGV 1

/* Define to 1 if you have the writev function. */
#cmakedefine HAVE_WRITEV 1

/* Define to 1 if you have the ws2tcpip.h header file. */
#cmakedefine HAVE_WS2TCPIP_H 1

/* Define to 1 if you have the <x509.h> header file. */
#cmakedefine HAVE_X509_H 1

/* Define if you have the <process.h> header file. */
#cmakedefine HAVE_PROCESS_H 1

/* if you have the zlib.h header file */
#cmakedefine HAVE_ZLIB_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#cmakedefine LT_OBJDIR ${LT_OBJDIR}

/* If you lack a fine basename() prototype */
#cmakedefine NEED_BASENAME_PROTO 1

/* Define to 1 if you need the lber.h header file even with ldap.h */
#cmakedefine NEED_LBER_H 1

/* Define to 1 if you need the malloc.h header file even with stdlib.h */
#cmakedefine NEED_MALLOC_H 1

/* Define to 1 if _REENTRANT preprocessor symbol must be defined. */
#cmakedefine NEED_REENTRANT 1

/* cpu-machine-OS */
#cmakedefine OS ${OS}

/* Name of package */
#cmakedefine PACKAGE ${PACKAGE}

/* Define to the address where bug reports for this package should be sent. */
#cmakedefine PACKAGE_BUGREPORT ${PACKAGE_BUGREPORT}

/* Define to the full name of this package. */
#cmakedefine PACKAGE_NAME ${PACKAGE_NAME}

/* Define to the full name and version of this package. */
#cmakedefine PACKAGE_STRING ${PACKAGE_STRING}

/* Define to the one symbol short name of this package. */
#cmakedefine PACKAGE_TARNAME ${PACKAGE_TARNAME}

/* Define to the version of this package. */
#cmakedefine PACKAGE_VERSION ${PACKAGE_VERSION}

/* a suitable file to read random data from */
#cmakedefine RANDOM_FILE "${RANDOM_FILE}"

/* Define to the type of arg 1 for recvfrom. */
#cmakedefine RECVFROM_TYPE_ARG1 ${RECVFROM_TYPE_ARG1}

/* Define to the type pointed by arg 2 for recvfrom. */
#cmakedefine RECVFROM_TYPE_ARG2 ${RECVFROM_TYPE_ARG2}

/* Define to 1 if the type pointed by arg 2 for recvfrom is void. */
#cmakedefine RECVFROM_TYPE_ARG2_IS_VOID 1

/* Define to the type of arg 3 for recvfrom. */
#cmakedefine RECVFROM_TYPE_ARG3 ${RECVFROM_TYPE_ARG3}

/* Define to the type of arg 4 for recvfrom. */
#cmakedefine RECVFROM_TYPE_ARG4 ${RECVFROM_TYPE_ARG4}

/* Define to the type pointed by arg 5 for recvfrom. */
#cmakedefine RECVFROM_TYPE_ARG5 ${RECVFROM_TYPE_ARG5}

/* Define to 1 if the type pointed by arg 5 for recvfrom is void. */
#cmakedefine RECVFROM_TYPE_ARG5_IS_VOID 1

/* Define to the type pointed by arg 6 for recvfrom. */
#cmakedefine RECVFROM_TYPE_ARG6 ${RECVFROM_TYPE_ARG6}

/* Define to 1 if the type pointed by arg 6 for recvfrom is void. */
#cmakedefine RECVFROM_TYPE_ARG6_IS_VOID 1

/* Define to the function return type for recvfrom. */
#cmakedefine RECVFROM_TYPE_RETV ${RECVFROM_TYPE_RETV}

/* Define to the type of arg 1 for recv. */
#cmakedefine RECV_TYPE_ARG1 ${RECV_TYPE_ARG1}

/* Define to the type of arg 2 for recv. */
#cmakedefine RECV_TYPE_ARG2 ${RECV_TYPE_ARG2}

/* Define to the type of arg 3 for recv. */
#cmakedefine RECV_TYPE_ARG3 ${RECV_TYPE_ARG3}

/* Define to the type of arg 4 for recv. */
#cmakedefine RECV_TYPE_ARG4 ${RECV_TYPE_ARG4}

/* Define to the function return type for recv. */
#cmakedefine RECV_TYPE_RETV ${RECV_TYPE_RETV}

/* Define to the type qualifier of arg 5 for select. */
#cmakedefine SELECT_QUAL_ARG5 ${SELECT_QUAL_ARG5}

/* Define to the type of arg 1 for select. */
#cmakedefine SELECT_TYPE_ARG1 ${SELECT_TYPE_ARG1}

/* Define to the type of args 2, 3 and 4 for select. */
#cmakedefine SELECT_TYPE_ARG234 ${SELECT_TYPE_ARG234}

/* Define to the type of arg 5 for select. */
#cmakedefine SELECT_TYPE_ARG5 ${SELECT_TYPE_ARG5}

/* Define to the function return type for select. */
#cmakedefine SELECT_TYPE_RETV ${SELECT_TYPE_RETV}

/* Define to the type qualifier of arg 2 for send. */
#cmakedefine SEND_QUAL_ARG2 ${SEND_QUAL_ARG2}

/* Define to the type of arg 1 for send. */
#cmakedefine SEND_TYPE_ARG1 ${SEND_TYPE_ARG1}

/* Define to the type of arg 2 for send. */
#cmakedefine SEND_TYPE_ARG2 ${SEND_TYPE_ARG2}

/* Define to the type of arg 3 for send. */
#cmakedefine SEND_TYPE_ARG3 ${SEND_TYPE_ARG3}

/* Define to the type of arg 4 for send. */
#cmakedefine SEND_TYPE_ARG4 ${SEND_TYPE_ARG4}

/* Define to the function return type for send. */
#cmakedefine SEND_TYPE_RETV ${SEND_TYPE_RETV}

/*
 Note: SIZEOF_* variables are fetched with CMake through check_type_size().
 As per CMake documentation on CheckTypeSize, C preprocessor code is
 generated by CMake into SIZEOF_*_CODE. This is what we use in the
 following statements.

 Reference: https://cmake.org/cmake/help/latest/module/CheckTypeSize.html
*/

/* The size of `int', as computed by sizeof. */
${SIZEOF_INT_CODE}

/* The size of `short', as computed by sizeof. */
${SIZEOF_SHORT_CODE}

/* The size of `long', as computed by sizeof. */
${SIZEOF_LONG_CODE}

/* The size of `off_t', as computed by sizeof. */
${SIZEOF_OFF_T_CODE}

/* The size of `curl_off_t', as computed by sizeof. */
${SIZEOF_CURL_OFF_T_CODE}

/* The size of `size_t', as computed by sizeof. */
${SIZEOF_SIZE_T_CODE}

/* The size of `time_t', as computed by sizeof. */
${SIZEOF_TIME_T_CODE}

/* Define to 1 if you have the ANSI C header files. */
#cmakedefine STDC_HEADERS 1

/* Define to the type of arg 3 for strerror_r. */
#cmakedefine STRERROR_R_TYPE_ARG3 ${STRERROR_R_TYPE_ARG3}

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#cmakedefine TIME_WITH_SYS_TIME 1

/* Define if you want to enable c-ares support */
#cmakedefine USE_ARES 1

/* Define if you want to enable POSIX threaded DNS lookup */
#cmakedefine USE_THREADS_POSIX 1

/* Define if you want to enable WIN32 threaded DNS lookup */
#cmakedefine USE_THREADS_WIN32 1

/* if GnuTLS is enabled */
#cmakedefine USE_GNUTLS 1

/* if Secure Transport is enabled */
#cmakedefine USE_SECTRANSP 1

/* if mbedTLS is enabled */
#cmakedefine USE_MBEDTLS 1

/* if BearSSL is enabled */
#cmakedefine USE_BEARSSL 1

/* if WolfSSL is enabled */
#cmakedefine USE_WOLFSSL 1

/* if libSSH is in use */
#cmakedefine USE_LIBSSH 1

/* if libSSH2 is in use */
#cmakedefine USE_LIBSSH2 1

/* If you want to build curl with the built-in manual */
#cmakedefine USE_MANUAL 1

/* if NSS is enabled */
#cmakedefine USE_NSS 1

/* if you have the PK11_CreateManagedGenericObject function */
#cmakedefine HAVE_PK11_CREATEMANAGEDGENERICOBJECT 1

/* if you want to use OpenLDAP code instead of legacy ldap implementation */
#cmakedefine USE_OPENLDAP 1

/* if OpenSSL is in use */
#cmakedefine USE_OPENSSL 1

/* Define to 1 if you don't want the OpenSSL configuration to be loaded
   automatically */
#cmakedefine CURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG 1

/* to enable NGHTTP2  */
#cmakedefine USE_NGHTTP2 1

/* to enable NGTCP2 */
#cmakedefine USE_NGTCP2 1

/* to enable NGHTTP3  */
#cmakedefine USE_NGHTTP3 1

/* to enable quiche */
#cmakedefine USE_QUICHE 1

/* Define to 1 if you have the quiche_conn_set_qlog_fd function. */
#cmakedefine HAVE_QUICHE_CONN_SET_QLOG_FD 1

/* if Unix domain sockets are enabled  */
#cmakedefine USE_UNIX_SOCKETS

/* Define to 1 if you are building a Windows target with large file support. */
#cmakedefine USE_WIN32_LARGE_FILES 1

/* to enable SSPI support */
#cmakedefine USE_WINDOWS_SSPI 1

/* to enable Windows SSL  */
#cmakedefine USE_SCHANNEL 1

/* enable multiple SSL backends */
#cmakedefine CURL_WITH_MULTI_SSL 1

/* Define to 1 if using yaSSL in OpenSSL compatibility mode. */
#cmakedefine USE_YASSLEMUL 1

/* Version number of package */
#cmakedefine VERSION ${VERSION}

/* Define to 1 if OS is AIX. */
#ifndef _ALL_SOURCE
#  undef _ALL_SOURCE
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
#cmakedefine _FILE_OFFSET_BITS ${_FILE_OFFSET_BITS}

/* Define for large files, on AIX-style hosts. */
#cmakedefine _LARGE_FILES ${_LARGE_FILES}

/* define this if you need it to compile thread-safe code */
#cmakedefine _THREAD_SAFE ${_THREAD_SAFE}

/* Define to empty if `const' does not conform to ANSI C. */
#cmakedefine const ${const}

/* Type to use in place of in_addr_t when system does not provide it. */
#cmakedefine in_addr_t ${in_addr_t}

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#undef inline
#endif

/* Define to `unsigned int' if <sys/types.h> does not define. */
#cmakedefine size_t ${size_t}

/* the signed version of size_t */
#cmakedefine ssize_t ${ssize_t}

/* Define to 1 if you have the mach_absolute_time function. */
#cmakedefine HAVE_MACH_ABSOLUTE_TIME 1

/* to enable Windows IDN */
#cmakedefine USE_WIN32_IDN 1

/* to make the compiler know the prototypes of Windows IDN APIs */
#cmakedefine WANT_IDN_PROTOTYPES 1
