/**************************************************************************/
/*  net_socket_posix.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "net_socket_posix.h"

// Some proprietary Unix-derived platforms don't expose Unix sockets
// so this allows skipping this file to reimplement this API differently.
#ifndef UNIX_SOCKET_UNAVAILABLE

#if defined(UNIX_ENABLED)

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef WEB_ENABLED
#include <arpa/inet.h>
#endif

// BSD calls this flag IPV6_JOIN_GROUP
#if !defined(IPV6_ADD_MEMBERSHIP) && defined(IPV6_JOIN_GROUP)
#define IPV6_ADD_MEMBERSHIP IPV6_JOIN_GROUP
#endif
#if !defined(IPV6_DROP_MEMBERSHIP) && defined(IPV6_LEAVE_GROUP)
#define IPV6_DROP_MEMBERSHIP IPV6_LEAVE_GROUP
#endif

// Some custom defines to minimize ifdefs
#define SOCK_EMPTY -1
#define SOCK_BUF(x) x
#define SOCK_CBUF(x) x
#define SOCK_IOCTL ioctl
#define SOCK_CLOSE ::close
#define SOCK_CONNECT(p_sock, p_addr, p_addr_len) ::connect(p_sock, p_addr, p_addr_len)

/* Windows */
#elif defined(WINDOWS_ENABLED)
#include <winsock2.h>
#include <ws2tcpip.h>

#include <mswsock.h>
// Some custom defines to minimize ifdefs
#define SOCK_EMPTY INVALID_SOCKET
#define SOCK_BUF(x) (char *)(x)
#define SOCK_CBUF(x) (const char *)(x)
#define SOCK_IOCTL ioctlsocket
#define SOCK_CLOSE closesocket
// connect is broken on windows under certain conditions, reasons unknown:
// See https://github.com/godotengine/webrtc-native/issues/6
#define SOCK_CONNECT(p_sock, p_addr, p_addr_len) ::WSAConnect(p_sock, p_addr, p_addr_len, nullptr, nullptr, nullptr, nullptr)

// Workaround missing flag in MinGW
#if defined(__MINGW32__) && !defined(SIO_UDP_NETRESET)
#define SIO_UDP_NETRESET _WSAIOW(IOC_VENDOR, 15)
#endif

#endif // UNIX_ENABLED

size_t NetSocketPosix::_set_addr_storage(struct sockaddr_storage *p_addr, const IPAddress &p_ip, uint16_t p_port, IP::Type p_ip_type) {
	memset(p_addr, 0, sizeof(struct sockaddr_storage));
	if (p_ip_type == IP::TYPE_IPV6 || p_ip_type == IP::TYPE_ANY) { // IPv6 socket

		// IPv6 only socket with IPv4 address
		ERR_FAIL_COND_V(!p_ip.is_wildcard() && p_ip_type == IP::TYPE_IPV6 && p_ip.is_ipv4(), 0);

		struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)p_addr;
		addr6->sin6_family = AF_INET6;
		addr6->sin6_port = htons(p_port);
		if (p_ip.is_valid()) {
			memcpy(&addr6->sin6_addr.s6_addr, p_ip.get_ipv6(), 16);
		} else {
			addr6->sin6_addr = in6addr_any;
		}
		return sizeof(sockaddr_in6);
	} else { // IPv4 socket

		// IPv4 socket with IPv6 address
		ERR_FAIL_COND_V(!p_ip.is_wildcard() && !p_ip.is_ipv4(), 0);

		struct sockaddr_in *addr4 = (struct sockaddr_in *)p_addr;
		addr4->sin_family = AF_INET;
		addr4->sin_port = htons(p_port); // short, network byte order

		if (p_ip.is_valid()) {
			memcpy(&addr4->sin_addr.s_addr, p_ip.get_ipv4(), 4);
		} else {
			addr4->sin_addr.s_addr = INADDR_ANY;
		}

		return sizeof(sockaddr_in);
	}
}

void NetSocketPosix::_set_ip_port(struct sockaddr_storage *p_addr, IPAddress *r_ip, uint16_t *r_port) {
	if (p_addr->ss_family == AF_INET) {
		struct sockaddr_in *addr4 = (struct sockaddr_in *)p_addr;
		if (r_ip) {
			r_ip->set_ipv4((uint8_t *)&(addr4->sin_addr.s_addr));
		}
		if (r_port) {
			*r_port = ntohs(addr4->sin_port);
		}
	} else if (p_addr->ss_family == AF_INET6) {
		struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)p_addr;
		if (r_ip) {
			r_ip->set_ipv6(addr6->sin6_addr.s6_addr);
		}
		if (r_port) {
			*r_port = ntohs(addr6->sin6_port);
		}
	}
}

NetSocket *NetSocketPosix::_create_func() {
	return memnew(NetSocketPosix);
}

void NetSocketPosix::make_default() {
#if defined(WINDOWS_ENABLED)
	if (_create == nullptr) {
		WSADATA data;
		WSAStartup(MAKEWORD(2, 2), &data);
	}
#endif
	_create = _create_func;
}

void NetSocketPosix::cleanup() {
#if defined(WINDOWS_ENABLED)
	if (_create != nullptr) {
		WSACleanup();
	}
	_create = nullptr;
#endif
}

NetSocketPosix::NetSocketPosix() :
		_sock(SOCK_EMPTY) {
}

NetSocketPosix::~NetSocketPosix() {
	close();
}

// Silence a warning reported in GH-27594.
// EAGAIN and EWOULDBLOCK have the same value on most platforms, but it's not guaranteed.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif

NetSocketPosix::NetError NetSocketPosix::_get_socket_error() const {
#if defined(WINDOWS_ENABLED)
	int err = WSAGetLastError();
	if (err == WSAEISCONN) {
		return ERR_NET_IS_CONNECTED;
	}
	if (err == WSAEINPROGRESS || err == WSAEALREADY) {
		return ERR_NET_IN_PROGRESS;
	}
	if (err == WSAEWOULDBLOCK) {
		return ERR_NET_WOULD_BLOCK;
	}
	if (err == WSAEADDRINUSE || err == WSAEADDRNOTAVAIL) {
		return ERR_NET_ADDRESS_INVALID_OR_UNAVAILABLE;
	}
	if (err == WSAEACCES) {
		return ERR_NET_UNAUTHORIZED;
	}
	if (err == WSAEMSGSIZE || err == WSAENOBUFS) {
		return ERR_NET_BUFFER_TOO_SMALL;
	}
	print_verbose("Socket error: " + itos(err));
	return ERR_NET_OTHER;
#else
	if (errno == EISCONN) {
		return ERR_NET_IS_CONNECTED;
	}
	if (errno == EINPROGRESS || errno == EALREADY) {
		return ERR_NET_IN_PROGRESS;
	}
	if (errno == EAGAIN || errno == EWOULDBLOCK) {
		return ERR_NET_WOULD_BLOCK;
	}
	if (errno == EADDRINUSE || errno == EINVAL || errno == EADDRNOTAVAIL) {
		return ERR_NET_ADDRESS_INVALID_OR_UNAVAILABLE;
	}
	if (errno == EACCES) {
		return ERR_NET_UNAUTHORIZED;
	}
	if (errno == ENOBUFS) {
		return ERR_NET_BUFFER_TOO_SMALL;
	}
	print_verbose("Socket error: " + itos(errno));
	return ERR_NET_OTHER;
#endif
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

bool NetSocketPosix::_can_use_ip(const IPAddress &p_ip, const bool p_for_bind) const {
	if (p_for_bind && !(p_ip.is_valid() || p_ip.is_wildcard())) {
		return false;
	} else if (!p_for_bind && !p_ip.is_valid()) {
		return false;
	}
	// Check if socket support this IP type.
	IP::Type type = p_ip.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;
	return !(_ip_type != IP::TYPE_ANY && !p_ip.is_wildcard() && _ip_type != type);
}

_FORCE_INLINE_ Error NetSocketPosix::_change_multicast_group(IPAddress p_ip, String p_if_name, bool p_add) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!_can_use_ip(p_ip, false), ERR_INVALID_PARAMETER);

	// Need to force level and af_family to IP(v4) when using dual stacking and provided multicast group is IPv4
	IP::Type type = _ip_type == IP::TYPE_ANY && p_ip.is_ipv4() ? IP::TYPE_IPV4 : _ip_type;
	// This needs to be the proper level for the multicast group, no matter if the socket is dual stacking.
	int level = type == IP::TYPE_IPV4 ? IPPROTO_IP : IPPROTO_IPV6;
	int ret = -1;

	IPAddress if_ip;
	uint32_t if_v6id = 0;
	HashMap<String, IP::Interface_Info> if_info;
	IP::get_singleton()->get_local_interfaces(&if_info);
	for (KeyValue<String, IP::Interface_Info> &E : if_info) {
		IP::Interface_Info &c = E.value;
		if (c.name != p_if_name) {
			continue;
		}

		if_v6id = (uint32_t)c.index.to_int();
		if (type == IP::TYPE_IPV6) {
			break; // IPv6 uses index.
		}

		for (const IPAddress &F : c.ip_addresses) {
			if (!F.is_ipv4()) {
				continue; // Wrong IP type
			}
			if_ip = F;
			break;
		}
		break;
	}

	if (level == IPPROTO_IP) {
		ERR_FAIL_COND_V(!if_ip.is_valid(), ERR_INVALID_PARAMETER);
		struct ip_mreq greq;
		int sock_opt = p_add ? IP_ADD_MEMBERSHIP : IP_DROP_MEMBERSHIP;
		memcpy(&greq.imr_multiaddr, p_ip.get_ipv4(), 4);
		memcpy(&greq.imr_interface, if_ip.get_ipv4(), 4);
		ret = setsockopt(_sock, level, sock_opt, (const char *)&greq, sizeof(greq));
	} else {
		struct ipv6_mreq greq;
		int sock_opt = p_add ? IPV6_ADD_MEMBERSHIP : IPV6_DROP_MEMBERSHIP;
		memcpy(&greq.ipv6mr_multiaddr, p_ip.get_ipv6(), 16);
		greq.ipv6mr_interface = if_v6id;
		ret = setsockopt(_sock, level, sock_opt, (const char *)&greq, sizeof(greq));
	}
	ERR_FAIL_COND_V(ret != 0, FAILED);

	return OK;
}

void NetSocketPosix::_set_socket(SOCKET_TYPE p_sock, IP::Type p_ip_type, bool p_is_stream) {
	_sock = p_sock;
	_ip_type = p_ip_type;
	_is_stream = p_is_stream;
	// Disable descriptor sharing with subprocesses.
	_set_close_exec_enabled(true);
}

void NetSocketPosix::_set_close_exec_enabled(bool p_enabled) {
#ifndef WINDOWS_ENABLED
	// Enable close on exec to avoid sharing with subprocesses. Off by default on Windows.
	int opts = fcntl(_sock, F_GETFD);
	fcntl(_sock, F_SETFD, opts | FD_CLOEXEC);
#endif
}

Error NetSocketPosix::open(Type p_sock_type, IP::Type &ip_type) {
	ERR_FAIL_COND_V(is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(ip_type > IP::TYPE_ANY || ip_type < IP::TYPE_NONE, ERR_INVALID_PARAMETER);

#if defined(__OpenBSD__)
	// OpenBSD does not support dual stacking, fallback to IPv4 only.
	if (ip_type == IP::TYPE_ANY) {
		ip_type = IP::TYPE_IPV4;
	}
#endif

	int family = ip_type == IP::TYPE_IPV4 ? AF_INET : AF_INET6;
	int protocol = p_sock_type == TYPE_TCP ? IPPROTO_TCP : IPPROTO_UDP;
	int type = p_sock_type == TYPE_TCP ? SOCK_STREAM : SOCK_DGRAM;
	_sock = socket(family, type, protocol);

	if (_sock == SOCK_EMPTY && ip_type == IP::TYPE_ANY) {
		// Careful here, changing the referenced parameter so the caller knows that we are using an IPv4 socket
		// in place of a dual stack one, and further calls to _set_sock_addr will work as expected.
		ip_type = IP::TYPE_IPV4;
		family = AF_INET;
		_sock = socket(family, type, protocol);
	}

	ERR_FAIL_COND_V(_sock == SOCK_EMPTY, FAILED);
	_ip_type = ip_type;

	if (family == AF_INET6) {
		// Select IPv4 over IPv6 mapping
		set_ipv6_only_enabled(ip_type != IP::TYPE_ANY);
	}

	if (protocol == IPPROTO_UDP) {
		// Make sure to disable broadcasting for UDP sockets.
		// Depending on the OS, this option might or might not be enabled by default. Let's normalize it.
		set_broadcasting_enabled(false);
	}

	_is_stream = p_sock_type == TYPE_TCP;

	// Disable descriptor sharing with subprocesses.
	_set_close_exec_enabled(true);

#if defined(WINDOWS_ENABLED)
	if (!_is_stream) {
		// Disable windows feature/bug reporting WSAECONNRESET/WSAENETRESET when
		// recv/recvfrom and an ICMP reply was received from a previous send/sendto.
		unsigned long disable = 0;
		if (ioctlsocket(_sock, SIO_UDP_CONNRESET, &disable) == SOCKET_ERROR) {
			print_verbose("Unable to turn off UDP WSAECONNRESET behavior on Windows");
		}
		if (ioctlsocket(_sock, SIO_UDP_NETRESET, &disable) == SOCKET_ERROR) {
			// This feature seems not to be supported on wine.
			print_verbose("Unable to turn off UDP WSAENETRESET behavior on Windows");
		}
	}
#endif
#if defined(SO_NOSIGPIPE)
	// Disable SIGPIPE (should only be relevant to stream sockets, but seems to affect UDP too on iOS)
	int par = 1;
	if (setsockopt(_sock, SOL_SOCKET, SO_NOSIGPIPE, SOCK_CBUF(&par), sizeof(int)) != 0) {
		print_verbose("Unable to turn off SIGPIPE on socket");
	}
#endif
	return OK;
}

void NetSocketPosix::close() {
	if (_sock != SOCK_EMPTY) {
		SOCK_CLOSE(_sock);
	}

	_sock = SOCK_EMPTY;
	_ip_type = IP::TYPE_NONE;
	_is_stream = false;
}

Error NetSocketPosix::bind(IPAddress p_addr, uint16_t p_port) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!_can_use_ip(p_addr, true), ERR_INVALID_PARAMETER);

	sockaddr_storage addr;
	size_t addr_size = _set_addr_storage(&addr, p_addr, p_port, _ip_type);

	if (::bind(_sock, (struct sockaddr *)&addr, addr_size) != 0) {
		NetError err = _get_socket_error();
		print_verbose("Failed to bind socket. Error: " + itos(err));
		close();
		return ERR_UNAVAILABLE;
	}

	return OK;
}

Error NetSocketPosix::listen(int p_max_pending) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	if (::listen(_sock, p_max_pending) != 0) {
		_get_socket_error();
		print_verbose("Failed to listen from socket.");
		close();
		return FAILED;
	}

	return OK;
}

Error NetSocketPosix::connect_to_host(IPAddress p_host, uint16_t p_port) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!_can_use_ip(p_host, false), ERR_INVALID_PARAMETER);

	struct sockaddr_storage addr;
	size_t addr_size = _set_addr_storage(&addr, p_host, p_port, _ip_type);

	if (SOCK_CONNECT(_sock, (struct sockaddr *)&addr, addr_size) != 0) {
		NetError err = _get_socket_error();

		switch (err) {
			// We are already connected
			case ERR_NET_IS_CONNECTED:
				return OK;
			// Still waiting to connect, try again in a while
			case ERR_NET_WOULD_BLOCK:
			case ERR_NET_IN_PROGRESS:
				return ERR_BUSY;
			default:
				print_verbose("Connection to remote host failed!");
				close();
				return FAILED;
		}
	}

	return OK;
}

Error NetSocketPosix::poll(PollType p_type, int p_timeout) const {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

#if defined(WINDOWS_ENABLED)
	bool ready = false;
	fd_set rd, wr, ex;
	fd_set *rdp = nullptr;
	fd_set *wrp = nullptr;
	FD_ZERO(&rd);
	FD_ZERO(&wr);
	FD_ZERO(&ex);
	FD_SET(_sock, &ex);
	struct timeval timeout = { p_timeout / 1000, (p_timeout % 1000) * 1000 };
	// For blocking operation, pass nullptr timeout pointer to select.
	struct timeval *tp = nullptr;
	if (p_timeout >= 0) {
		//  If timeout is non-negative, we want to specify the timeout instead.
		tp = &timeout;
	}

	switch (p_type) {
		case POLL_TYPE_IN:
			FD_SET(_sock, &rd);
			rdp = &rd;
			break;
		case POLL_TYPE_OUT:
			FD_SET(_sock, &wr);
			wrp = &wr;
			break;
		case POLL_TYPE_IN_OUT:
			FD_SET(_sock, &rd);
			FD_SET(_sock, &wr);
			rdp = &rd;
			wrp = &wr;
	}
	int ret = select(1, rdp, wrp, &ex, tp);

	if (ret == SOCKET_ERROR) {
		return FAILED;
	}

	if (ret == 0) {
		return ERR_BUSY;
	}

	if (FD_ISSET(_sock, &ex)) {
		_get_socket_error();
		print_verbose("Exception when polling socket.");
		return FAILED;
	}

	if (rdp && FD_ISSET(_sock, rdp)) {
		ready = true;
	}
	if (wrp && FD_ISSET(_sock, wrp)) {
		ready = true;
	}

	return ready ? OK : ERR_BUSY;
#else
	struct pollfd pfd;
	pfd.fd = _sock;
	pfd.events = POLLIN;
	pfd.revents = 0;

	switch (p_type) {
		case POLL_TYPE_IN:
			pfd.events = POLLIN;
			break;
		case POLL_TYPE_OUT:
			pfd.events = POLLOUT;
			break;
		case POLL_TYPE_IN_OUT:
			pfd.events = POLLOUT | POLLIN;
	}

	int ret = ::poll(&pfd, 1, p_timeout);

	if (ret < 0 || pfd.revents & POLLERR) {
		_get_socket_error();
		print_verbose("Error when polling socket.");
		return FAILED;
	}

	if (ret == 0) {
		return ERR_BUSY;
	}

	return OK;
#endif
}

Error NetSocketPosix::recv(uint8_t *p_buffer, int p_len, int &r_read) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	r_read = ::recv(_sock, SOCK_BUF(p_buffer), p_len, 0);

	if (r_read < 0) {
		NetError err = _get_socket_error();
		if (err == ERR_NET_WOULD_BLOCK) {
			return ERR_BUSY;
		}

		if (err == ERR_NET_BUFFER_TOO_SMALL) {
			return ERR_OUT_OF_MEMORY;
		}

		return FAILED;
	}

	return OK;
}

Error NetSocketPosix::recvfrom(uint8_t *p_buffer, int p_len, int &r_read, IPAddress &r_ip, uint16_t &r_port, bool p_peek) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	struct sockaddr_storage from;
	socklen_t len = sizeof(struct sockaddr_storage);
	memset(&from, 0, len);

	r_read = ::recvfrom(_sock, SOCK_BUF(p_buffer), p_len, p_peek ? MSG_PEEK : 0, (struct sockaddr *)&from, &len);

	if (r_read < 0) {
		NetError err = _get_socket_error();
		if (err == ERR_NET_WOULD_BLOCK) {
			return ERR_BUSY;
		}

		if (err == ERR_NET_BUFFER_TOO_SMALL) {
			return ERR_OUT_OF_MEMORY;
		}

		return FAILED;
	}

	if (from.ss_family == AF_INET) {
		struct sockaddr_in *sin_from = (struct sockaddr_in *)&from;
		r_ip.set_ipv4((uint8_t *)&sin_from->sin_addr);
		r_port = ntohs(sin_from->sin_port);
	} else if (from.ss_family == AF_INET6) {
		struct sockaddr_in6 *s6_from = (struct sockaddr_in6 *)&from;
		r_ip.set_ipv6((uint8_t *)&s6_from->sin6_addr);
		r_port = ntohs(s6_from->sin6_port);
	} else {
		// Unsupported socket family, should never happen.
		ERR_FAIL_V(FAILED);
	}

	return OK;
}

Error NetSocketPosix::send(const uint8_t *p_buffer, int p_len, int &r_sent) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	int flags = 0;
#ifdef MSG_NOSIGNAL
	if (_is_stream) {
		flags = MSG_NOSIGNAL;
	}
#endif
	r_sent = ::send(_sock, SOCK_CBUF(p_buffer), p_len, flags);

	if (r_sent < 0) {
		NetError err = _get_socket_error();
		if (err == ERR_NET_WOULD_BLOCK) {
			return ERR_BUSY;
		}
		if (err == ERR_NET_BUFFER_TOO_SMALL) {
			return ERR_OUT_OF_MEMORY;
		}

		return FAILED;
	}

	return OK;
}

Error NetSocketPosix::sendto(const uint8_t *p_buffer, int p_len, int &r_sent, IPAddress p_ip, uint16_t p_port) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	struct sockaddr_storage addr;
	size_t addr_size = _set_addr_storage(&addr, p_ip, p_port, _ip_type);
	r_sent = ::sendto(_sock, SOCK_CBUF(p_buffer), p_len, 0, (struct sockaddr *)&addr, addr_size);

	if (r_sent < 0) {
		NetError err = _get_socket_error();
		if (err == ERR_NET_WOULD_BLOCK) {
			return ERR_BUSY;
		}
		if (err == ERR_NET_BUFFER_TOO_SMALL) {
			return ERR_OUT_OF_MEMORY;
		}

		return FAILED;
	}

	return OK;
}

Error NetSocketPosix::set_broadcasting_enabled(bool p_enabled) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);
	// IPv6 has no broadcast support.
	if (_ip_type == IP::TYPE_IPV6) {
		return ERR_UNAVAILABLE;
	}

	int par = p_enabled ? 1 : 0;
	if (setsockopt(_sock, SOL_SOCKET, SO_BROADCAST, SOCK_CBUF(&par), sizeof(int)) != 0) {
		WARN_PRINT("Unable to change broadcast setting");
		return FAILED;
	}
	return OK;
}

void NetSocketPosix::set_blocking_enabled(bool p_enabled) {
	ERR_FAIL_COND(!is_open());

	int ret = 0;
#if defined(WINDOWS_ENABLED)
	unsigned long par = p_enabled ? 0 : 1;
	ret = SOCK_IOCTL(_sock, FIONBIO, &par);
#else
	int opts = fcntl(_sock, F_GETFL);
	if (p_enabled) {
		ret = fcntl(_sock, F_SETFL, opts & ~O_NONBLOCK);
	} else {
		ret = fcntl(_sock, F_SETFL, opts | O_NONBLOCK);
	}
#endif

	if (ret != 0) {
		WARN_PRINT("Unable to change non-block mode");
	}
}

void NetSocketPosix::set_ipv6_only_enabled(bool p_enabled) {
	ERR_FAIL_COND(!is_open());
	// This option is only available in IPv6 sockets.
	ERR_FAIL_COND(_ip_type == IP::TYPE_IPV4);

	int par = p_enabled ? 1 : 0;
	if (setsockopt(_sock, IPPROTO_IPV6, IPV6_V6ONLY, SOCK_CBUF(&par), sizeof(int)) != 0) {
		WARN_PRINT("Unable to change IPv4 address mapping over IPv6 option");
	}
}

void NetSocketPosix::set_tcp_no_delay_enabled(bool p_enabled) {
	ERR_FAIL_COND(!is_open());
	ERR_FAIL_COND(!_is_stream); // Not TCP

	int par = p_enabled ? 1 : 0;
	if (setsockopt(_sock, IPPROTO_TCP, TCP_NODELAY, SOCK_CBUF(&par), sizeof(int)) < 0) {
		ERR_PRINT("Unable to set TCP no delay option");
	}
}

void NetSocketPosix::set_reuse_address_enabled(bool p_enabled) {
	ERR_FAIL_COND(!is_open());

// On Windows, enabling SO_REUSEADDR actually would also enable reuse port, very bad on TCP. Denying...
// Windows does not have this option, SO_REUSEADDR in this magical world means SO_REUSEPORT
#ifndef WINDOWS_ENABLED
	int par = p_enabled ? 1 : 0;
	if (setsockopt(_sock, SOL_SOCKET, SO_REUSEADDR, SOCK_CBUF(&par), sizeof(int)) < 0) {
		WARN_PRINT("Unable to set socket REUSEADDR option!");
	}
#endif
}

void NetSocketPosix::set_reuse_port_enabled(bool p_enabled) {
	ERR_FAIL_COND(!is_open());

// See comment above...
#ifdef WINDOWS_ENABLED
#define SO_REUSEPORT SO_REUSEADDR
#endif
	int par = p_enabled ? 1 : 0;
	if (setsockopt(_sock, SOL_SOCKET, SO_REUSEPORT, SOCK_CBUF(&par), sizeof(int)) < 0) {
		WARN_PRINT("Unable to set socket REUSEPORT option!");
	}
}

bool NetSocketPosix::is_open() const {
	return _sock != SOCK_EMPTY;
}

int NetSocketPosix::get_available_bytes() const {
	ERR_FAIL_COND_V(!is_open(), -1);

	unsigned long len;
	int ret = SOCK_IOCTL(_sock, FIONREAD, &len);
	if (ret == -1) {
		_get_socket_error();
		print_verbose("Error when checking available bytes on socket.");
		return -1;
	}
	return len;
}

Error NetSocketPosix::get_socket_address(IPAddress *r_ip, uint16_t *r_port) const {
	ERR_FAIL_COND_V(!is_open(), FAILED);

	struct sockaddr_storage saddr;
	socklen_t len = sizeof(saddr);
	if (getsockname(_sock, (struct sockaddr *)&saddr, &len) != 0) {
		_get_socket_error();
		print_verbose("Error when reading local socket address.");
		return FAILED;
	}
	_set_ip_port(&saddr, r_ip, r_port);
	return OK;
}

Ref<NetSocket> NetSocketPosix::accept(IPAddress &r_ip, uint16_t &r_port) {
	Ref<NetSocket> out;
	ERR_FAIL_COND_V(!is_open(), out);

	struct sockaddr_storage their_addr;
	socklen_t size = sizeof(their_addr);
	SOCKET_TYPE fd = ::accept(_sock, (struct sockaddr *)&their_addr, &size);
	if (fd == SOCK_EMPTY) {
		_get_socket_error();
		print_verbose("Error when accepting socket connection.");
		return out;
	}

	_set_ip_port(&their_addr, &r_ip, &r_port);

	NetSocketPosix *ns = memnew(NetSocketPosix);
	ns->_set_socket(fd, _ip_type, _is_stream);
	ns->set_blocking_enabled(false);
	return Ref<NetSocket>(ns);
}

Error NetSocketPosix::join_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) {
	return _change_multicast_group(p_multi_address, p_if_name, true);
}

Error NetSocketPosix::leave_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) {
	return _change_multicast_group(p_multi_address, p_if_name, false);
}

#endif // UNIX_SOCKET_UNAVAILABLE
