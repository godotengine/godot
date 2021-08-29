/*****************************************************************************
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
#define BUILDING_LIBCURL 1

/* Location of default ca bundle */
// #define CURL_CA_BUNDLE "${CURL_CA_BUNDLE}"

/* define "1" to use built-in ca store of TLS backend */
// #define CURL_CA_FALLBACK 1

/* Location of default ca path */
// #define CURL_CA_PATH "${CURL_CA_PATH}"

/* to disable cookies support */
// #define CURL_DISABLE_COOKIES 1

/* to disable cryptographic authentication */
#define CURL_DISABLE_CRYPTO_AUTH 1

/* to disable DICT */
#define CURL_DISABLE_DICT 1

/* to disable FILE */
#define CURL_DISABLE_FILE 1

/* to disable FTP */
#define CURL_DISABLE_FTP 1

/* to disable GOPHER */
#define CURL_DISABLE_GOPHER 1

/* to disable IMAP */
#define CURL_DISABLE_IMAP 1

/* to disable HTTP */
// #define CURL_DISABLE_HTTP 1

/* to disable LDAP */
#define CURL_DISABLE_LDAP 1

/* to disable LDAPS */
#define CURL_DISABLE_LDAPS 1

/* to disable MQTT */
#define CURL_DISABLE_MQTT 1

/* to disable POP3 */
#define CURL_DISABLE_POP3 1

/* to disable proxies */
#define CURL_DISABLE_PROXY 1

/* to disable RTSP */
#define CURL_DISABLE_RTSP 1

/* to disable SMB */
#define CURL_DISABLE_SMB 1

/* to disable SMTP */
#define CURL_DISABLE_SMTP 1

/* to disable TELNET */
#define CURL_DISABLE_TELNET 1

/* to disable TFTP */
#define CURL_DISABLE_TFTP 1

/* to disable verbose strings */
#define CURL_DISABLE_VERBOSE_STRINGS 1

// /* to make a symbol visible */
// #define CURL_EXTERN_SYMBOL ${CURL_EXTERN_SYMBOL}
// /* Ensure using CURL_EXTERN_SYMBOL is possible */
// #ifndef CURL_EXTERN_SYMBOL
// #define CURL_EXTERN_SYMBOL
// #endif

/* Allow SMB to work on Windows */
// #define USE_WIN32_CRYPTO 1

/* Use Windows LDAP implementation */
// #define USE_WIN32_LDAP 1

/* when not building a shared library */
#define CURL_STATICLIB 1

/* your Entropy Gathering Daemon socket pathname */
// #define EGD_SOCKET ${EGD_SOCKET}

/* Define if you want to enable IPv6 support */
#define ENABLE_IPV6 1

/* Define to 1 if you have the alarm function. */
#define HAVE_ALARM 1

/* Define to 1 if you have the <alloca.h> header file. */
#define HAVE_ALLOCA_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Define to 1 if you have the <arpa/tftp.h> header file. */
#define HAVE_ARPA_TFTP_H 1
#endif

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the `basename' function. */
#define HAVE_BASENAME 1

/* Define to 1 if bool is an available type. */
#define HAVE_BOOL_T 1

/* Define to 1 if you have the __builtin_available function. */
#define HAVE_BUILTIN_AVAILABLE 1

/* Define to 1 if you have the clock_gettime function and monotonic timer. */
#define HAVE_CLOCK_GETTIME_MONOTONIC 1

// /* Define to 1 if you have the `closesocket' function. */
// #define HAVE_CLOSESOCKET 1

/* Define to 1 if you have the `CRYPTO_cleanup_all_ex_data' function. */
#define HAVE_CRYPTO_CLEANUP_ALL_EX_DATA 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the fcntl function. */
#define HAVE_FCNTL 1

/* Define to 1 if you have the <fcntl.h> header file. */
// #define HAVE_FCNTL_H 1

/* Define to 1 if you have a working fcntl O_NONBLOCK function. */
// #define HAVE_FCNTL_O_NONBLOCK 1

/* Define to 1 if you have the freeaddrinfo function. */
#define HAVE_FREEADDRINFO 1

/* Define to 1 if you have the ftruncate function. */
#define HAVE_FTRUNCATE 1

/* Define to 1 if you have a working getaddrinfo function. */
#define HAVE_GETADDRINFO 1

/* Define to 1 if you have the `geteuid' function. */
#define HAVE_GETEUID 1

/* Define to 1 if you have the `getppid' function. */
#define HAVE_GETPPID 1

/* Define to 1 if you have the gethostbyname function. */
#define HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the gethostbyname_r function. */
#define HAVE_GETHOSTBYNAME_R 1

/* gethostbyname_r() takes 3 args */
#define HAVE_GETHOSTBYNAME_R_3 1

/* gethostbyname_r() takes 5 args */
#define HAVE_GETHOSTBYNAME_R_5 1

/* gethostbyname_r() takes 6 args */
#define HAVE_GETHOSTBYNAME_R_6 1

/* Define to 1 if you have the gethostname function. */
#define HAVE_GETHOSTNAME 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have a working getifaddrs function. */
#define HAVE_GETIFADDRS 1
#endif

/* Define to 1 if you have the `getpass_r' function. */
#define HAVE_GETPASS_R 1

/* Define to 1 if you have the `getppid' function. */
#define HAVE_GETPPID 1

/* Define to 1 if you have the `getprotobyname' function. */
#define HAVE_GETPROTOBYNAME 1

/* Define to 1 if you have the `getpeername' function. */
#define HAVE_GETPEERNAME 1

/* Define to 1 if you have the `getsockname' function. */
#define HAVE_GETSOCKNAME 1

/* Define to 1 if you have the `if_nametoindex' function. */
#define HAVE_IF_NAMETOINDEX 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the `getpwuid' function. */
#define HAVE_GETPWUID 1

/* Define to 1 if you have the `getpwuid_r' function. */
#define HAVE_GETPWUID_R 1
#endif

/* Define to 1 if you have the `getrlimit' function. */
#define HAVE_GETRLIMIT 1

/* Define to 1 if you have the `gettimeofday' function. */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have a working glibc-style strerror_r function. */
#define HAVE_GLIBC_STRERROR_R 1

/* Define to 1 if you have a working gmtime_r function. */
#define HAVE_GMTIME_R 1

/* if you have the gssapi libraries */
// #define HAVE_GSSAPI 1

/* Define to 1 if you have the <gssapi/gssapi_generic.h> header file. */
// #define HAVE_GSSAPI_GSSAPI_GENERIC_H 1

/* Define to 1 if you have the <gssapi/gssapi.h> header file. */
// #define HAVE_GSSAPI_GSSAPI_H 1

/* Define to 1 if you have the <gssapi/gssapi_krb5.h> header file. */
// #define HAVE_GSSAPI_GSSAPI_KRB5_H 1

/* if you have the GNU gssapi libraries */
// #define HAVE_GSSGNU 1

/* if you have the Heimdal gssapi libraries */
// #define HAVE_GSSHEIMDAL 1

/* if you have the MIT gssapi libraries */
// #define HAVE_GSSMIT 1

/* Define to 1 if you have the `idna_strerror' function. */
#define HAVE_IDNA_STRERROR 1

/* Define to 1 if you have the `idn_free' function. */
#define HAVE_IDN_FREE 1

/* Define to 1 if you have the <idn-free.h> header file. */
#define HAVE_IDN_FREE_H 1

#if defined(WINDOWS_ENABLED)
#define USE_WIN32_LARGE_FILES 1
#endif

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <ifaddrs.h> header file. */
#define HAVE_IFADDRS_H 1
#endif

/* Define to 1 if you have the `inet_addr' function. */
#define HAVE_INET_ADDR 1

/* Define to 1 if you have a IPv6 capable working inet_ntop function. */
#define HAVE_INET_NTOP 1

/* Define to 1 if you have a IPv6 capable working inet_pton function. */
#define HAVE_INET_PTON 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if symbol `sa_family_t' exists */
#define HAVE_SA_FAMILY_T 1
#endif

/* Define to 1 if symbol `ADDRESS_FAMILY' exists */
#define HAVE_ADDRESS_FAMILY 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the ioctl function. */
#define HAVE_IOCTL 1

/* Define to 1 if you have the ioctlsocket function. */
#define HAVE_IOCTLSOCKET 1

/* Define to 1 if you have the IoctlSocket camel case function. */
#define HAVE_IOCTLSOCKET_CAMEL 1

/* Define to 1 if you have a working IoctlSocket camel case FIONBIO function.
   */
#define HAVE_IOCTLSOCKET_CAMEL_FIONBIO 1

/* Define to 1 if you have a working ioctlsocket FIONBIO function. */
#define HAVE_IOCTLSOCKET_FIONBIO 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have a working ioctl FIONBIO function. */
#define HAVE_IOCTL_FIONBIO 1
#endif

/* Define to 1 if you have a working ioctl SIOCGIFADDR function. */
#define HAVE_IOCTL_SIOCGIFADDR 1
#endif

/* Define to 1 if you have the <io.h> header file. */
#define HAVE_IO_H 1

/* if you have the Kerberos4 libraries (including -ldes) */
#define HAVE_KRB4 1

/* Define to 1 if you have the `krb_get_our_ip_for_realm' function. */
#define HAVE_KRB_GET_OUR_IP_FOR_REALM 1

/* Define to 1 if you have the <krb.h> header file. */
#define HAVE_KRB_H 1

/* Define to 1 if you have the lber.h header file. */
#define HAVE_LBER_H 1

/* Define to 1 if you have the ldapssl.h header file. */
#define HAVE_LDAPSSL_H 1

/* Define to 1 if you have the ldap.h header file. */
#define HAVE_LDAP_H 1

/* Use LDAPS implementation */
#define HAVE_LDAP_SSL 1

/* Define to 1 if you have the ldap_ssl.h header file. */
#define HAVE_LDAP_SSL_H 1

/* Define to 1 if you have the `ldap_url_parse' function. */
#define HAVE_LDAP_URL_PARSE 1

/* Define to 1 if you have the <libgen.h> header file. */
#define HAVE_LIBGEN_H 1

/* Define to 1 if you have the `idn2' library (-lidn2). */
#define HAVE_LIBIDN2 1

/* Define to 1 if you have the idn2.h header file. */
// #define HAVE_IDN2_H 1

/* Define to 1 if you have the `resolv' library (-lresolv). */
#define HAVE_LIBRESOLV 1

/* Define to 1 if you have the `resolve' library (-lresolve). */
#define HAVE_LIBRESOLVE 1

/* Define to 1 if you have the `socket' library (-lsocket). */
#define HAVE_LIBSOCKET 1

/* Define to 1 if you have the `ssh2' library (-lssh2). */
// #define HAVE_LIBSSH2 1

/* Define to 1 if you have the <libssh2.h> header file. */
// #define HAVE_LIBSSH2_H 1

/* Define to 1 if you have the <libssh/libssh.h> header file. */
// #define HAVE_LIBSSH_LIBSSH_H 1

/* if zlib is available */
#define HAVE_LIBZ 1

/* if brotli is available */
// #define HAVE_BROTLI 1

/* if zstd is available */
#define HAVE_ZSTD 1

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

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

#if !defined(WINDOWS_ENABLED)
#if !defined(OSX_ENABLED)
/* Define to 1 if you have the MSG_NOSIGNAL flag. */
#define HAVE_MSG_NOSIGNAL 1
#endif

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#define HAVE_NETINET_TCP_H 1

#if defined(LINUX_ENABLED)
/* Define to 1 if you have the <linux/tcp.h> header file. */
#define HAVE_LINUX_TCP_H 1
#endif

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1
#endif

/* Define to 1 if NI_WITHSCOPEID exists and works. */
#define HAVE_NI_WITHSCOPEID 1

/* if you have an old MIT gssapi library, lacking GSS_C_NT_HOSTBASED_SERVICE */
#define HAVE_OLD_GSSMIT 1

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
#define HAVE_PEM_H 1

/* Define to 1 if you have the `pipe' function. */
#define HAVE_PIPE 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have a working poll function. */
#define HAVE_POLL 1

/* If you have a fine poll */
#define HAVE_POLL_FINE 1

/* Define to 1 if you have the <poll.h> header file. */
#define HAVE_POLL_H 1
#endif

/* Define to 1 if you have a working POSIX-style strerror_r function. */
#define HAVE_POSIX_STRERROR_R 1

/* Define to 1 if you have the <pthread.h> header file */
#define HAVE_PTHREAD_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <pwd.h> header file. */
#define HAVE_PWD_H 1

/* Define to 1 if you have the `RAND_egd' function. */
#define HAVE_RAND_EGD 1

/* Define to 1 if you have the `RAND_screen' function. */
#define HAVE_RAND_SCREEN 1
#endif

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

/* Define to 1 if you have the 'fsetxattr' function. */
#define HAVE_FSETXATTR 1

/* fsetxattr() takes 5 args */
#define HAVE_FSETXATTR_5 1

/* fsetxattr() takes 6 args */
#define HAVE_FSETXATTR_6 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <setjmp.h> header file. */
#define HAVE_SETJMP_H 1
#endif

/* Define to 1 if you have the `setlocale' function. */
#define HAVE_SETLOCALE 1

/* Define to 1 if you have the `setmode' function. */
#define HAVE_SETMODE 1

/* Define to 1 if you have the `setrlimit' function. */
#define HAVE_SETRLIMIT 1

/* Define to 1 if you have the setsockopt function. */
#define HAVE_SETSOCKOPT 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have a working setsockopt SO_NONBLOCK function. */
#define HAVE_SETSOCKOPT_SO_NONBLOCK 1
#endif

/* Define to 1 if you have the sigaction function. */
#define HAVE_SIGACTION 1

/* Define to 1 if you have the siginterrupt function. */
#define HAVE_SIGINTERRUPT 1

/* Define to 1 if you have the signal function. */
#define HAVE_SIGNAL 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the sigsetjmp function or macro. */
#define HAVE_SIGSETJMP 1
#endif

/* Define to 1 if sig_atomic_t is an available typedef. */
#define HAVE_SIG_ATOMIC_T 1

/* Define to 1 if sig_atomic_t is already defined as volatile. */
#define HAVE_SIG_ATOMIC_T_VOLATILE 1

/* Define to 1 if struct sockaddr_in6 has the sin6_scope_id member */
#define HAVE_SOCKADDR_IN6_SIN6_SCOPE_ID 1

/* Define to 1 if you have the `socket' function. */
#define HAVE_SOCKET 1

/* Define to 1 if you have the <ssl.h> header file. */
#define HAVE_SSL_H 1

/* Define to 1 if you have the <stdbool.h> header file. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the strcasecmp function. */
#define HAVE_STRCASECMP 1

/* Define to 1 if you have the strcasestr function. */
#define HAVE_STRCASESTR 1

/* Define to 1 if you have the strcmpi function. */
#define HAVE_STRCMPI 1

/* Define to 1 if you have the strdup function. */
#define HAVE_STRDUP 1

/* Define to 1 if you have the strerror_r function. */
// #define HAVE_STRERROR_R 1

/* Define to 1 if you have the stricmp function. */
#define HAVE_STRICMP 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the strncmpi function. */
#define HAVE_STRNCMPI 1

/* Define to 1 if you have the strnicmp function. */
#define HAVE_STRNICMP 1

/* Define to 1 if you have the <stropts.h> header file. */
// #define HAVE_STROPTS_H 1

/* Define to 1 if you have the strstr function. */
#define HAVE_STRSTR 1

/* Define to 1 if you have the strtok_r function. */
#define HAVE_STRTOK_R 1

/* Define to 1 if you have the strtoll function. */
#define HAVE_STRTOLL 1

/* if struct sockaddr_storage is defined */
#define HAVE_STRUCT_SOCKADDR_STORAGE 1

/* Define to 1 if you have the timeval struct. */
#define HAVE_STRUCT_TIMEVAL 1

/* Define to 1 if you have the <sys/filio.h> header file. */
#define HAVE_SYS_FILIO_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/poll.h> header file. */
#define HAVE_SYS_POLL_H 1

/* Define to 1 if you have the <sys/resource.h> header file. */
#define HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/select.h> header file. */
#define HAVE_SYS_SELECT_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1
#endif

/* Define to 1 if you have the <sys/sockio.h> header file. */
// #define HAVE_SYS_SOCKIO_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1
#endif

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#define HAVE_SYS_UIO_H 1

#if !defined(WINDOWS_ENABLED)
/* Define to 1 if you have the <sys/un.h> header file. */
#define HAVE_SYS_UN_H 1
#endif

/* Define to 1 if you have the <sys/utime.h> header file. */
#define HAVE_SYS_UTIME_H 1
#endif

/* Define to 1 if you have the <termios.h> header file. */
#define HAVE_TERMIOS_H 1

/* Define to 1 if you have the <termio.h> header file. */
#define HAVE_TERMIO_H 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <tld.h> header file. */
#define HAVE_TLD_H 1

/* Define to 1 if you have the `tld_strerror' function. */
#define HAVE_TLD_STRERROR 1

/* Define to 1 if you have the `uname' function. */
#define HAVE_UNAME 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `utime' function. */
#define HAVE_UTIME 1

/* Define to 1 if you have the `utimes' function. */
#define HAVE_UTIMES 1

/* Define to 1 if you have the <utime.h> header file. */
#define HAVE_UTIME_H 1

/* Define to 1 if compiler supports C99 variadic macro style. */
#define HAVE_VARIADIC_MACROS_C99 1

/* Define to 1 if compiler supports old gcc variadic macro style. */
#define HAVE_VARIADIC_MACROS_GCC 1

/* Define to 1 if you have the winber.h header file. */
#define HAVE_WINBER_H 1

#if defined(WINDOWS_ENABLED)
/* Define to 1 if you have the windows.h header file. */
#define HAVE_WINDOWS_H 1

/* Define to 1 if you have the winldap.h header file. */
// #define HAVE_WINLDAP_H 1

/* Define to 1 if you have the winsock2.h header file. */
#define HAVE_WINSOCK2_H 1

/* Define to 1 if you have the winsock.h header file. */
#define HAVE_WINSOCK_H 1
#endif

/* Define this symbol if your OS supports changing the contents of argv */
#define HAVE_WRITABLE_ARGV 1

/* Define to 1 if you have the writev function. */
#define HAVE_WRITEV 1

/* Define to 1 if you have the ws2tcpip.h header file. */
#define HAVE_WS2TCPIP_H 1

/* Define to 1 if you have the <x509.h> header file. */
#define HAVE_X509_H 1

/* Define if you have the <process.h> header file. */
// #define HAVE_PROCESS_H 1

/* if you have the zlib.h header file */
#define HAVE_ZLIB_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR \
	$ { LT_OBJDIR }

/* If you lack a fine basename() prototype */
#define NEED_BASENAME_PROTO 1

/* Define to 1 if you need the lber.h header file even with ldap.h */
#define NEED_LBER_H 1

// /* Define to 1 if you need the malloc.h header file even with stdlib.h */
// #define NEED_MALLOC_H 1

/* Define to 1 if _REENTRANT preprocessor symbol must be defined. */
#define NEED_REENTRANT 1

/* cpu-machine-OS */
#define OS "OS"

/* Name of package */
#define PACKAGE \
	$ { PACKAGE }

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT \
	$ { PACKAGE_BUGREPORT }

/* Define to the full name of this package. */
#define PACKAGE_NAME \
	$ { PACKAGE_NAME }

/* Define to the full name and version of this package. */
#define PACKAGE_STRING \
	$ { PACKAGE_STRING }

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME \
	$ { PACKAGE_TARNAME }

/* Define to the version of this package. */
#define PACKAGE_VERSION \
	$ { PACKAGE_VERSION }

/* a suitable file to read random data from */
// #define RANDOM_FILE "${RANDOM_FILE}"

/* Define to the type of arg 1 for recvfrom. */
#define RECVFROM_TYPE_ARG1 \
	$ { RECVFROM_TYPE_ARG1 }

/* Define to the type pointed by arg 2 for recvfrom. */
#define RECVFROM_TYPE_ARG2 \
	$ { RECVFROM_TYPE_ARG2 }

/* Define to 1 if the type pointed by arg 2 for recvfrom is void. */
#define RECVFROM_TYPE_ARG2_IS_VOID 1

/* Define to the type of arg 3 for recvfrom. */
#define RECVFROM_TYPE_ARG3 \
	$ { RECVFROM_TYPE_ARG3 }

/* Define to the type of arg 4 for recvfrom. */
#define RECVFROM_TYPE_ARG4 \
	$ { RECVFROM_TYPE_ARG4 }

/* Define to the type pointed by arg 5 for recvfrom. */
#define RECVFROM_TYPE_ARG5 \
	$ { RECVFROM_TYPE_ARG5 }

/* Define to 1 if the type pointed by arg 5 for recvfrom is void. */
#define RECVFROM_TYPE_ARG5_IS_VOID 1

/* Define to the type pointed by arg 6 for recvfrom. */
#define RECVFROM_TYPE_ARG6 \
	$ { RECVFROM_TYPE_ARG6 }

/* Define to 1 if the type pointed by arg 6 for recvfrom is void. */
#define RECVFROM_TYPE_ARG6_IS_VOID 1

/* Define to the function return type for recvfrom. */
#define RECVFROM_TYPE_RETV \
	$ { RECVFROM_TYPE_RETV }

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

// /* Define to the type qualifier of arg 5 for select. */
// #define SELECT_QUAL_ARG5 ${SELECT_QUAL_ARG5}

// /* Define to the type of arg 1 for select. */
// #define SELECT_TYPE_ARG1 ${SELECT_TYPE_ARG1}

// /* Define to the type of args 2, 3 and 4 for select. */
// #define SELECT_TYPE_ARG234 ${SELECT_TYPE_ARG234}

// /* Define to the type of arg 5 for select. */
// #define SELECT_TYPE_ARG5 ${SELECT_TYPE_ARG5}

// /* Define to the function return type for select. */
// #define SELECT_TYPE_RETV ssize_t

/* Define to the type qualifier of arg 2 for send. */
#define SEND_QUAL_ARG2 const

/* Define to the type of arg 1 for send. */
#define SEND_TYPE_ARG1 int

/* Define to the type of arg 2 for send. */
#define SEND_TYPE_ARG2 void *

/* Define to the type of arg 3 for send. */
#define SEND_TYPE_ARG3 size_t

/* Define to the type of arg 4 for send. */
#define SEND_TYPE_ARG4 size_t

/* Define to the function return type for send. */
#define SEND_TYPE_RETV ssize_t

/*
 Note: SIZEOF_* variables are fetched with CMake through check_type_size().
 As per CMake documentation on CheckTypeSize, C preprocessor code is
 generated by CMake into SIZEOF_*_CODE. This is what we use in the
 following statements.

 Reference: https://cmake.org/cmake/help/latest/module/CheckTypeSize.html
*/

/* The size of `int', as computed by sizeof. */
// ${SIZEOF_INT_CODE}

/* The size of `short', as computed by sizeof. */
// ${SIZEOF_SHORT_CODE}

/* The size of `long', as computed by sizeof. */
// ${SIZEOF_LONG_CODE}

/* The size of `off_t', as computed by sizeof. */
// ${SIZEOF_OFF_T_CODE}

/* The size of `curl_off_t', as computed by sizeof. */
#define SIZEOF_CURL_OFF_T 4

/* The size of `size_t', as computed by sizeof. */
// ${SIZEOF_SIZE_T_CODE}

/* The size of `time_t', as computed by sizeof. */
// ${SIZEOF_TIME_T_CODE}

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to the type of arg 3 for strerror_r. */
#define STRERROR_R_TYPE_ARG3 \
	$ { STRERROR_R_TYPE_ARG3 }

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* Define if you want to enable c-ares support */
// #define USE_ARES 1

/* Define if you want to enable POSIX threaded DNS lookup */
#define USE_THREADS_POSIX 1

/* Define if you want to enable WIN32 threaded DNS lookup */
// #define USE_THREADS_WIN32 1

/* if GnuTLS is enabled */
// #define USE_GNUTLS 1

/* if Secure Transport is enabled */
// #define USE_SECTRANSP 1

/* if mbedTLS is enabled */
#define USE_MBEDTLS 1

#define CURL_DISABLE_HSTS 1

/* Define to 1 if you don't want the OpenSSL configuration to be loaded
   automatically */
#define CURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG 1

/* to enable NGHTTP2  */
// #define USE_NGHTTP2 1

/* to enable NGTCP2 */
// #define USE_NGTCP2 1

/* to enable NGHTTP3  */
// #define USE_NGHTTP3 1

/* to enable quiche */
// #define USE_QUICHE 1

/* Define to 1 if you have the quiche_conn_set_qlog_fd function. */
#define HAVE_QUICHE_CONN_SET_QLOG_FD 1

/* if Unix domain sockets are enabled  */
#define USE_UNIX_SOCKETS

/* to disable alt-svc */
#define CURL_DISABLE_ALTSVC 1

/* Define to 1 if you are building a Windows target with large file support. */
// #define USE_WIN32_LARGE_FILES 1

/* to enable SSPI support */
// #define USE_WINDOWS_SSPI 1

/* to enable Windows SSL  */
// #define USE_SCHANNEL 1

/* enable multiple SSL backends */
// #define CURL_WITH_MULTI_SSL 1

/* Define to 1 if using yaSSL in OpenSSL compatibility mode. */
// #define USE_YASSLEMUL 1

/* Version number of package */
// #define VERSION ${VERSION}

/* Define to 1 if OS is AIX. */
// #ifndef _ALL_SOURCE
// #  undef _ALL_SOURCE
// #endif

/* Number of bits in a file offset, on hosts where this is settable. */
// #define _FILE_OFFSET_BITS ${_FILE_OFFSET_BITS}

/* Define for large files, on AIX-style hosts. */
// #define _LARGE_FILES ${_LARGE_FILES}

/* define this if you need it to compile thread-safe code */
// #define _THREAD_SAFE ${_THREAD_SAFE}

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#undef inline
#endif

// /* Define to `unsigned int' if <sys/types.h> does not define. */
// #define size_t ${size_t}

// /* the signed version of size_t */
// #define ssize_t ${ssize_t}

/* Define to 1 if you have the mach_absolute_time function. */
// #define HAVE_MACH_ABSOLUTE_TIME 1

/* to enable Windows IDN */
// #define USE_WIN32_IDN 1

/* to make the compiler know the prototypes of Windows IDN APIs */
// #define WANT_IDN_PROTOTYPES 1
