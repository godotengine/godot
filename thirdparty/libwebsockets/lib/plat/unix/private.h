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
 *  Included from lib/core/private.h if no explicit platform
 */

#include <fcntl.h>
#include <strings.h>
#include <unistd.h>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <poll.h>
#include <netdb.h>

#ifndef __cplusplus
#include <errno.h>
#endif
#include <netdb.h>
#include <signal.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/un.h>

#if defined(__APPLE__)
#include <machine/endian.h>
#endif
#if defined(__FreeBSD__)
#include <sys/endian.h>
#endif
#if defined(__linux__)
#include <endian.h>
#endif
#if defined(__QNX__)
	#include <gulliver.h>
	#if defined(__LITTLEENDIAN__)
		#define BYTE_ORDER __LITTLEENDIAN__
		#define LITTLE_ENDIAN __LITTLEENDIAN__
		#define BIG_ENDIAN 4321  /* to show byte order (taken from gcc); for suppres warning that BIG_ENDIAN is not defined. */
	#endif
	#if defined(__BIGENDIAN__)
		#define BYTE_ORDER __BIGENDIAN__
		#define LITTLE_ENDIAN 1234  /* to show byte order (taken from gcc); for suppres warning that LITTLE_ENDIAN is not defined. */
		#define BIG_ENDIAN __BIGENDIAN__
	#endif
#endif

#if defined(__sun) && defined(__GNUC__)

#include <arpa/nameser_compat.h>

#if !defined (BYTE_ORDER)
#define BYTE_ORDER __BYTE_ORDER__
#endif

#if !defined(LITTLE_ENDIAN)
#define LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
#endif

#if !defined(BIG_ENDIAN)
#define BIG_ENDIAN __ORDER_BIG_ENDIAN__
#endif

#endif /* sun + GNUC */

#if !defined(BYTE_ORDER)
#define BYTE_ORDER __BYTE_ORDER
#endif
#if !defined(LITTLE_ENDIAN)
#define LITTLE_ENDIAN __LITTLE_ENDIAN
#endif
#if !defined(BIG_ENDIAN)
#define BIG_ENDIAN __BIG_ENDIAN
#endif

#if defined(LWS_BUILTIN_GETIFADDRS)
#include "./misc/getifaddrs.h"
#else

#if defined(__HAIKU__)
#define _BSD_SOURCE
#endif
#include <ifaddrs.h>

#endif

#if defined (__sun) || defined(__HAIKU__) || defined(__QNX__)
#include <syslog.h>
#else
#include <sys/syslog.h>
#endif

#ifdef __QNX__
# include "netinet/tcp_var.h"
# define TCP_KEEPINTVL TCPCTL_KEEPINTVL
# define TCP_KEEPIDLE  TCPCTL_KEEPIDLE
# define TCP_KEEPCNT   TCPCTL_KEEPCNT
#endif

#define LWS_ERRNO errno
#define LWS_EAGAIN EAGAIN
#define LWS_EALREADY EALREADY
#define LWS_EINPROGRESS EINPROGRESS
#define LWS_EINTR EINTR
#define LWS_EISCONN EISCONN
#define LWS_ENOTCONN ENOTCONN
#define LWS_EWOULDBLOCK EWOULDBLOCK
#define LWS_EADDRINUSE EADDRINUSE
#define lws_set_blocking_send(wsi)
#define LWS_SOCK_INVALID (-1)

#define wsi_from_fd(A,B)  A->lws_lookup[B - lws_plat_socket_offset()]
#define insert_wsi(A,B)   assert(A->lws_lookup[B->desc.sockfd - \
				  lws_plat_socket_offset()] == 0); \
				 A->lws_lookup[B->desc.sockfd - \
				  lws_plat_socket_offset()] = B
#define delete_from_fd(A,B) A->lws_lookup[B - lws_plat_socket_offset()] = 0

#ifndef LWS_NO_FORK
#ifdef LWS_HAVE_SYS_PRCTL_H
#include <sys/prctl.h>
#endif
#endif

#if defined (__ANDROID__)
 #include <syslog.h>
 #include <sys/resource.h>
#endif

#define compatible_close(x) close(x)
#define lws_plat_socket_offset() (0)

/*
 * Mac OSX as well as iOS do not define the MSG_NOSIGNAL flag,
 * but happily have something equivalent in the SO_NOSIGPIPE flag.
 */
#ifdef __APPLE__
#define MSG_NOSIGNAL SO_NOSIGPIPE
#endif

/*
 * Solaris 11.X only supports POSIX 2001, MSG_NOSIGNAL appears in
 * POSIX 2008.
 */
#ifdef __sun
 #define MSG_NOSIGNAL 0
#endif
