/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 *  Included from lib/core/private.h if defined(WIN32) || defined(_WIN32)
 */

 #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
 #endif

 #if (WINVER < 0x0501)
  #undef WINVER
  #undef _WIN32_WINNT
  #define WINVER 0x0501
  #define _WIN32_WINNT WINVER
 #endif

 #define LWS_NO_DAEMONIZE
 #define LWS_ERRNO WSAGetLastError()
 #define LWS_EAGAIN WSAEWOULDBLOCK
 #define LWS_EALREADY WSAEALREADY
 #define LWS_EINPROGRESS WSAEINPROGRESS
 #define LWS_EINTR WSAEINTR
 #define LWS_EISCONN WSAEISCONN
 #define LWS_ENOTCONN WSAENOTCONN
 #define LWS_EWOULDBLOCK WSAEWOULDBLOCK
 #define LWS_EADDRINUSE WSAEADDRINUSE
 #define MSG_NOSIGNAL 0
 #define SHUT_RDWR SD_BOTH
 #define SOL_TCP IPPROTO_TCP
 #define SHUT_WR SD_SEND

 #define compatible_close(fd) closesocket(fd)
 #define lws_set_blocking_send(wsi) wsi->sock_send_blocking = 1
 #define LWS_SOCK_INVALID (INVALID_SOCKET)

 #include <winsock2.h>
 #include <ws2tcpip.h>
 #include <windows.h>
 #include <tchar.h>
 #ifdef LWS_HAVE_IN6ADDR_H
  #include <in6addr.h>
 #endif
 #include <mstcpip.h>
 #include <io.h>

 #if !defined(LWS_HAVE_ATOLL)
  #if defined(LWS_HAVE__ATOI64)
   #define atoll _atoi64
  #else
   #warning No atoll or _atoi64 available, using atoi
   #define atoll atoi
  #endif
 #endif

 #ifndef __func__
  #define __func__ __FUNCTION__
 #endif

 #ifdef LWS_HAVE__VSNPRINTF
  #define vsnprintf _vsnprintf
 #endif

/* we don't have an implementation for this on windows... */
int kill(int pid, int sig);
int fork(void);
#ifndef SIGINT
#define SIGINT 2
#endif

#include <gettimeofday.h>

#ifndef BIG_ENDIAN
 #define BIG_ENDIAN    4321  /* to show byte order (taken from gcc) */
#endif
#ifndef LITTLE_ENDIAN
 #define LITTLE_ENDIAN 1234
#endif
#ifndef BYTE_ORDER
 #define BYTE_ORDER LITTLE_ENDIAN
#endif

#undef __P
#ifndef __P
 #if __STDC__
  #define __P(protos) protos
 #else
  #define __P(protos) ()
 #endif
#endif

#ifdef _WIN32
 #ifndef FD_HASHTABLE_MODULUS
  #define FD_HASHTABLE_MODULUS 32
 #endif
#endif

#define lws_plat_socket_offset() (0)

struct lws;
struct lws_context;

#define LWS_FD_HASH(fd) ((fd ^ (fd >> 8) ^ (fd >> 16)) % FD_HASHTABLE_MODULUS)
struct lws_fd_hashtable {
	struct lws **wsi;
	int length;
};


#ifdef LWS_DLL
#ifdef LWS_INTERNAL
#define LWS_EXTERN extern __declspec(dllexport)
#else
#define LWS_EXTERN extern __declspec(dllimport)
#endif
#else
#define LWS_EXTERN
#endif

typedef SOCKET lws_sockfd_type;
typedef HANDLE lws_filefd_type;
#define LWS_WIN32_HANDLE_TYPES

LWS_EXTERN struct lws *
wsi_from_fd(const struct lws_context *context, lws_sockfd_type fd);

LWS_EXTERN int
insert_wsi(struct lws_context *context, struct lws *wsi);

LWS_EXTERN int
delete_from_fd(struct lws_context *context, lws_sockfd_type fd);
