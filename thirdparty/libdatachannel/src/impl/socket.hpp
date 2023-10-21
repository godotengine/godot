/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// This header defines types to allow cross-platform socket API usage.

#ifndef RTC_SOCKET_H
#define RTC_SOCKET_H

#ifdef _WIN32

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Windows 7
#endif
#ifndef __MSVCRT_VERSION__
#define __MSVCRT_VERSION__ 0x0601
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
//
#include <iphlpapi.h>
#include <windows.h>

#ifdef __MINGW32__
#include <sys/stat.h>
#include <sys/time.h>
#ifndef IPV6_V6ONLY
#define IPV6_V6ONLY 27
#endif
#endif

#define NO_IFADDRS
#define NO_PMTUDISC

typedef SOCKET socket_t;
typedef SOCKADDR sockaddr;
typedef ULONG ctl_t;
typedef DWORD sockopt_t;
#define sockerrno ((int)WSAGetLastError())
#define IP_DONTFRAG IP_DONTFRAGMENT
#define HOST_NAME_MAX 256

#define poll WSAPoll
typedef ULONG nfds_t;

#define SEADDRINUSE WSAEADDRINUSE
#define SEINTR WSAEINTR
#define SEAGAIN WSAEWOULDBLOCK
#define SEACCES WSAEACCES
#define SEWOULDBLOCK WSAEWOULDBLOCK
#define SEINPROGRESS WSAEINPROGRESS
#define SECONNREFUSED WSAECONNREFUSED
#define SECONNRESET WSAECONNRESET
#define SENETRESET WSAENETRESET

#else // assume POSIX

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef __linux__
#define NO_PMTUDISC
#endif

#ifdef __ANDROID__
#define NO_IFADDRS
#else
#include <ifaddrs.h>
#endif

typedef int socket_t;
typedef int ctl_t;
typedef int sockopt_t;
#define sockerrno errno
#define INVALID_SOCKET -1
#define ioctlsocket ioctl
#define closesocket close

#define SEADDRINUSE EADDRINUSE
#define SEINTR EINTR
#define SEAGAIN EAGAIN
#define SEACCES EACCES
#define SEWOULDBLOCK EWOULDBLOCK
#define SEINPROGRESS EINPROGRESS
#define SECONNREFUSED ECONNREFUSED
#define SECONNRESET ECONNRESET
#define SENETRESET ENETRESET

#endif // _WIN32

#ifndef IN6_IS_ADDR_LOOPBACK
#define IN6_IS_ADDR_LOOPBACK(a)                                                                    \
	(((const uint32_t *)(a))[0] == 0 && ((const uint32_t *)(a))[1] == 0 &&                         \
	 ((const uint32_t *)(a))[2] == 0 && ((const uint32_t *)(a))[3] == htonl(1))
#endif

#ifndef IN6_IS_ADDR_LINKLOCAL
#define IN6_IS_ADDR_LINKLOCAL(a)                                                                   \
	((((const uint32_t *)(a))[0] & htonl(0xffc00000)) == htonl(0xfe800000))
#endif

#ifndef IN6_IS_ADDR_SITELOCAL
#define IN6_IS_ADDR_SITELOCAL(a)                                                                   \
	((((const uint32_t *)(a))[0] & htonl(0xffc00000)) == htonl(0xfec00000))
#endif

#ifndef IN6_IS_ADDR_V4MAPPED
#define IN6_IS_ADDR_V4MAPPED(a)                                                                    \
	((((const uint32_t *)(a))[0] == 0) && (((const uint32_t *)(a))[1] == 0) &&                     \
	 (((const uint32_t *)(a))[2] == htonl(0xFFFF)))
#endif

#endif // JUICE_SOCKET_H
