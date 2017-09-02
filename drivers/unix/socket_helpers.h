/*************************************************************************/
/*  socket_helpers.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef SOCKET_HELPERS_H
#define SOCKET_HELPERS_H

#include <string.h>

#if defined(__MINGW32__) && (!defined(__MINGW64_VERSION_MAJOR) || __MINGW64_VERSION_MAJOR < 4)
// Workaround for mingw-w64 < 4.0
#ifndef IPV6_V6ONLY
#define IPV6_V6ONLY 27
#endif
#endif

// helpers for sockaddr -> IP_Address and back, should work for posix and winsock. All implementations should use this

static size_t _set_sockaddr(struct sockaddr_storage *p_addr, const IP_Address &p_ip, int p_port, IP::Type p_sock_type = IP::TYPE_ANY) {

	memset(p_addr, 0, sizeof(struct sockaddr_storage));

	ERR_FAIL_COND_V(!p_ip.is_valid(), 0);

	// IPv6 socket
	if (p_sock_type == IP::TYPE_IPV6 || p_sock_type == IP::TYPE_ANY) {

		// IPv6 only socket with IPv4 address
		ERR_FAIL_COND_V(p_sock_type == IP::TYPE_IPV6 && p_ip.is_ipv4(), 0);

		struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)p_addr;
		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		copymem(&addr6->sin6_addr.s6_addr, p_ip.get_ipv6(), 16);
		return sizeof(sockaddr_in6);

	} else { // IPv4 socket

		// IPv4 socket with IPv6 address
		ERR_FAIL_COND_V(!p_ip.is_ipv4(), 0);

		struct sockaddr_in *addr4 = (struct sockaddr_in *)p_addr;
		addr4->sin_family = AF_INET;
		addr4->sin_port = htons(p_port); // short, network byte order
		copymem(&addr4->sin_addr.s_addr, p_ip.get_ipv4(), 16);
		return sizeof(sockaddr_in);
	};
};

static size_t _set_listen_sockaddr(struct sockaddr_storage *p_addr, int p_port, IP::Type p_sock_type, const IP_Address p_bind_address) {

	memset(p_addr, 0, sizeof(struct sockaddr_storage));
	if (p_sock_type == IP::TYPE_IPV4) {
		struct sockaddr_in *addr4 = (struct sockaddr_in *)p_addr;
		addr4->sin_family = AF_INET;
		addr4->sin_port = htons(p_port);
		if (p_bind_address.is_valid()) {
			copymem(&addr4->sin_addr.s_addr, p_bind_address.get_ipv4(), 4);
		} else {
			addr4->sin_addr.s_addr = INADDR_ANY;
		}
		return sizeof(sockaddr_in);
	} else {
		struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)p_addr;

		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		if (p_bind_address.is_valid()) {
			copymem(&addr6->sin6_addr.s6_addr, p_bind_address.get_ipv6(), 16);
		} else {
			addr6->sin6_addr = in6addr_any;
		}
		return sizeof(sockaddr_in6);
	};
};

static int _socket_create(IP::Type &p_type, int type, int protocol) {

	ERR_FAIL_COND_V(p_type > IP::TYPE_ANY || p_type < IP::TYPE_NONE, ERR_INVALID_PARAMETER);

	int family = p_type == IP::TYPE_IPV4 ? AF_INET : AF_INET6;
	int sockfd = socket(family, type, protocol);

	if (sockfd == -1 && p_type == IP::TYPE_ANY) {
		// Careful here, changing the referenced parameter so the caller knows that we are using an IPv4 socket
		// in place of a dual stack one, and further calls to _set_sock_addr will work as expected.
		p_type = IP::TYPE_IPV4;
		family = AF_INET;
		sockfd = socket(family, type, protocol);
	}

	ERR_FAIL_COND_V(sockfd == -1, -1);

	if (family == AF_INET6) {
		// Select IPv4 over IPv6 mapping
		int opt = p_type != IP::TYPE_ANY;
		if (setsockopt(sockfd, IPPROTO_IPV6, IPV6_V6ONLY, (const char *)&opt, sizeof(opt)) != 0) {
			WARN_PRINT("Unable to set/unset IPv4 address mapping over IPv6");
		}
	}

	return sockfd;
}

static void _set_ip_addr_port(IP_Address &r_ip, int &r_port, struct sockaddr_storage *p_addr) {

	if (p_addr->ss_family == AF_INET) {

		struct sockaddr_in *addr4 = (struct sockaddr_in *)p_addr;
		r_ip.set_ipv4((uint8_t *)&(addr4->sin_addr.s_addr));

		r_port = ntohs(addr4->sin_port);

	} else if (p_addr->ss_family == AF_INET6) {

		struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)p_addr;
		r_ip.set_ipv6(addr6->sin6_addr.s6_addr);

		r_port = ntohs(addr6->sin6_port);
	};
};

#endif
