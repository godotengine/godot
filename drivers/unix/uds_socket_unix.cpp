/**************************************************************************/
/*  uds_socket_unix.cpp                                                   */
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

// Some proprietary Unix-derived platforms don't expose Unix sockets
// so this allows skipping this file to reimplement this API differently.
#if defined(UNIX_ENABLED) && !defined(UNIX_SOCKET_UNAVAILABLE)

#include "uds_socket_unix.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>

socklen_t UDSSocketUnix::_set_sockaddr(struct sockaddr_un *p_addr, const CharString &p_path) {
	memset(p_addr, 0, sizeof(struct sockaddr_un));
	p_addr->sun_family = AF_UNIX;

	// Path must not exceed maximum path length for Unix domain socket
	size_t path_len = p_path.length();
	ERR_FAIL_COND_V(path_len >= sizeof(p_addr->sun_path) - 1, 0);

	// Regular file system socket
	memcpy(p_addr->sun_path, p_path.get_data(), path_len);
	p_addr->sun_path[path_len] = '\0';
	return sizeof(struct sockaddr_un);
}

UDSSocket *UDSSocketUnix::_create_func() {
	return memnew(UDSSocketUnix);
}

void UDSSocketUnix::make_default() {
	_create = _create_func;
}

void UDSSocketUnix::cleanup() {
	_create = nullptr;
}

UDSSocketUnix::UDSSocketUnix() {
}

UDSSocketUnix::~UDSSocketUnix() {
	close();
}

UDSSocketUnix::SocketError UDSSocketUnix::_get_socket_error() const {
	int err = errno;

	switch (err) {
		case EAGAIN:
			return ERR_SOCKET_WOULD_BLOCK;
		case EISCONN:
			return ERR_SOCKET_IS_CONNECTED;
		case EINPROGRESS:
			return ERR_SOCKET_IN_PROGRESS;
		case EADDRINUSE:
		case EADDRNOTAVAIL:
			return ERR_SOCKET_ADDRESS_INVALID_OR_UNAVAILABLE;
		case EPERM:
		case EACCES:
			return ERR_SOCKET_UNAUTHORIZED;
		case ENOBUFS:
			return ERR_SOCKET_BUFFER_TOO_SMALL;
		default:
			print_verbose("Socket error: " + itos(err) + ".");
			return ERR_SOCKET_OTHER;
	}
}

void UDSSocketUnix::_set_close_exec_enabled(bool p_enabled) {
	// Enable close on exec to avoid sharing with subprocesses. Off by default on Windows.
	int opts = fcntl(_sock, F_GETFD);
	fcntl(_sock, F_SETFD, opts | FD_CLOEXEC);
}

Error UDSSocketUnix::open() {
	ERR_FAIL_COND_V(is_open(), ERR_ALREADY_IN_USE);

	_sock = socket(AF_UNIX, SOCK_STREAM, 0);
	ERR_FAIL_COND_V(_sock == -1, FAILED);

	_set_close_exec_enabled(true);

#if defined(SO_NOSIGPIPE)
	// Disable SIGPIPE (should only be relevant to stream sockets, but seems to affect UDP too on iOS).
	int par = 1;
	if (setsockopt(_sock, SOL_SOCKET, SO_NOSIGPIPE, &par, sizeof(int)) != 0) {
		print_verbose("Unable to turn off SIGPIPE on socket.");
	}
#endif

	return OK;
}

void UDSSocketUnix::close() {
	if (_sock != -1) {
		::close(_sock);
		_sock = -1;
	}
	if (unlink_on_close) {
		::unlink(path.get_data());
		unlink_on_close = false;
	}
	path = CharString();
}

Error UDSSocketUnix::bind(const String &p_path) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);

	path = p_path.utf8();

	struct sockaddr_un addr;
	socklen_t addr_size = _set_sockaddr(&addr, path);
	ERR_FAIL_COND_V(addr_size == 0, ERR_INVALID_PARAMETER);

	// If the socket file exists, attempt to remove it.
	if (access(path.get_data(), F_OK) == 0) {
		// Check if it's a socket
		struct stat st;
		if (stat(path.get_data(), &st) == 0) {
			if (S_ISSOCK(st.st_mode)) {
				// It is a socket, try to remove it.
				if (unlink(path.get_data()) != 0) {
					// Failed to remove existing socket file.
					return FAILED;
				}
			} else {
				// It's not a socket, don't remove it.
				return ERR_ALREADY_EXISTS;
			}
		}
	}

	unlink_on_close = true;

	if (::bind(_sock, (struct sockaddr *)&addr, addr_size) != 0) {
		SocketError err = _get_socket_error();
		print_verbose("Failed to bind socket. Error: " + itos(err) + ".");
		close();
		switch (err) {
			case ERR_SOCKET_UNAUTHORIZED:
				return ERR_UNAUTHORIZED;
			default:
				return ERR_UNAVAILABLE;
		}
	}

	return OK;
}

Error UDSSocketUnix::listen(int p_max_pending) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	if (::listen(_sock, p_max_pending) != 0) {
		_get_socket_error();
		print_verbose("Failed to listen from socket.");
		close();
		return FAILED;
	}

	return OK;
}

Error UDSSocketUnix::connect_to_host(const String &p_path) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);

	path = p_path.utf8();
	struct sockaddr_un addr;
	socklen_t addr_size = _set_sockaddr(&addr, path);
	ERR_FAIL_COND_V(addr_size == 0, ERR_INVALID_PARAMETER);

	if (::connect(_sock, (struct sockaddr *)&addr, addr_size) != 0) {
		SocketError err = _get_socket_error();
		switch (err) {
			case ERR_SOCKET_ADDRESS_INVALID_OR_UNAVAILABLE:
				return ERR_INVALID_PARAMETER;
			// Still waiting to connect, try again in a while.
			case ERR_SOCKET_WOULD_BLOCK:
			case ERR_SOCKET_IN_PROGRESS:
				return ERR_BUSY;
			case ERR_SOCKET_UNAUTHORIZED:
				return ERR_UNAUTHORIZED;
			default:
				print_verbose("Connection to remote host failed.");
				close();
				return FAILED;
		}
	}

	return OK;
}

Error UDSSocketUnix::poll(PollType p_type, int p_timeout) const {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

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
}

Error UDSSocketUnix::recv(uint8_t *p_buffer, int p_len, int &r_read) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	r_read = ::recv(_sock, p_buffer, p_len, 0);

	if (r_read < 0) {
		SocketError err = _get_socket_error();
		if (err == ERR_SOCKET_WOULD_BLOCK) {
			return ERR_BUSY;
		}

		if (err == ERR_SOCKET_BUFFER_TOO_SMALL) {
			return ERR_OUT_OF_MEMORY;
		}

		return FAILED;
	}

	return OK;
}

Error UDSSocketUnix::send(const uint8_t *p_buffer, int p_len, int &r_sent) {
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	int flags = 0;
#ifdef MSG_NOSIGNAL
	flags = MSG_NOSIGNAL;
#endif
	r_sent = ::send(_sock, p_buffer, p_len, flags);

	if (r_sent < 0) {
		SocketError err = _get_socket_error();
		if (err == ERR_SOCKET_WOULD_BLOCK) {
			return ERR_BUSY;
		}
		if (err == ERR_SOCKET_BUFFER_TOO_SMALL) {
			return ERR_OUT_OF_MEMORY;
		}

		return FAILED;
	}

	return OK;
}

void UDSSocketUnix::set_blocking_enabled(bool p_enabled) {
	ERR_FAIL_COND(!is_open());

	int ret = 0;
	int opts = fcntl(_sock, F_GETFL);
	if (p_enabled) {
		ret = fcntl(_sock, F_SETFL, opts & ~O_NONBLOCK);
	} else {
		ret = fcntl(_sock, F_SETFL, opts | O_NONBLOCK);
	}

	if (ret != 0) {
		WARN_PRINT("Unable to change non-block mode.");
	}
}

bool UDSSocketUnix::is_open() const {
	return _sock != -1;
}

int UDSSocketUnix::get_available_bytes() const {
	ERR_FAIL_COND_V(!is_open(), -1);

	int len;
	int ret = ioctl(_sock, FIONREAD, &len);
	if (ret == -1) {
		_get_socket_error();
		print_verbose("Error when checking available bytes on socket.");
		return -1;
	}
	return len;
}

Ref<UDSSocket> UDSSocketUnix::accept() {
	Ref<UDSSocket> out;
	ERR_FAIL_COND_V(!is_open(), out);

	struct sockaddr_un addr;
	socklen_t addr_len = sizeof(addr);

	int fd = ::accept(_sock, (struct sockaddr *)&addr, &addr_len);
	if (fd == -1) {
		_get_socket_error();
		print_verbose("Error when accepting socket connection.");
		return out;
	}

	UDSSocketUnix *ret = memnew(UDSSocketUnix);
	ret->_sock = fd;
	ret->set_blocking_enabled(false);
	return Ref<UDSSocket>(ret);
}

#endif // UNIX_ENABLED && !UNIX_SOCKET_UNAVAILABLE
