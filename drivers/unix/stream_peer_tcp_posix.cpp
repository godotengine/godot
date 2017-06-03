/*************************************************************************/
/*  stream_peer_tcp_posix.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifdef UNIX_ENABLED

#include "stream_peer_tcp_posix.h"

#include <errno.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef NO_FCNTL
#ifdef __HAIKU__
#include <fcntl.h>
#else
#include <sys/fcntl.h>
#endif
#else
#include <sys/ioctl.h>
#endif
#include <netinet/in.h>

#include <sys/socket.h>
#ifdef JAVASCRIPT_ENABLED
#include <arpa/inet.h>
#endif

#include <netinet/tcp.h>

#if defined(OSX_ENABLED) || defined(IPHONE_ENABLED)
#define MSG_NOSIGNAL SO_NOSIGPIPE
#endif

#include "drivers/unix/socket_helpers.h"

StreamPeerTCP *StreamPeerTCPPosix::_create() {

	return memnew(StreamPeerTCPPosix);
};

void StreamPeerTCPPosix::make_default() {

	StreamPeerTCP::_create = StreamPeerTCPPosix::_create;
};

Error StreamPeerTCPPosix::_block(int p_sockfd, bool p_read, bool p_write) const {

	struct pollfd pfd;
	pfd.fd = p_sockfd;
	pfd.events = 0;
	if (p_read)
		pfd.events |= POLLIN;
	if (p_write)
		pfd.events |= POLLOUT;
	pfd.revents = 0;

	int ret = poll(&pfd, 1, -1);
	return ret < 0 ? FAILED : OK;
};

Error StreamPeerTCPPosix::_poll_connection() const {

	ERR_FAIL_COND_V(status != STATUS_CONNECTING || sockfd == -1, FAILED);

	struct sockaddr_storage their_addr;
	size_t addr_size = _set_sockaddr(&their_addr, peer_host, peer_port, sock_type);

	if (::connect(sockfd, (struct sockaddr *)&their_addr, addr_size) == -1) {

		if (errno == EISCONN) {
			status = STATUS_CONNECTED;
			return OK;
		};

		if (errno == EINPROGRESS || errno == EALREADY) {
			return OK;
		}

		status = STATUS_ERROR;
		return ERR_CONNECTION_ERROR;
	} else {

		status = STATUS_CONNECTED;
		return OK;
	};

	return OK;
};

void StreamPeerTCPPosix::set_socket(int p_sockfd, IP_Address p_host, int p_port, IP::Type p_sock_type) {

	sock_type = p_sock_type;
	sockfd = p_sockfd;
#ifndef NO_FCNTL
	fcntl(sockfd, F_SETFL, O_NONBLOCK);
#else
	int bval = 1;
	ioctl(sockfd, FIONBIO, &bval);

#endif
	status = STATUS_CONNECTING;

	peer_host = p_host;
	peer_port = p_port;
};

Error StreamPeerTCPPosix::connect_to_host(const IP_Address &p_host, uint16_t p_port) {

	ERR_FAIL_COND_V(!p_host.is_valid(), ERR_INVALID_PARAMETER);

	sock_type = p_host.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;
	sockfd = _socket_create(sock_type, SOCK_STREAM, IPPROTO_TCP);
	if (sockfd == -1) {
		ERR_PRINT("Socket creation failed!");
		disconnect_from_host();
		//perror("socket");
		return FAILED;
	};

#ifndef NO_FCNTL
	fcntl(sockfd, F_SETFL, O_NONBLOCK);
#else
	int bval = 1;
	ioctl(sockfd, FIONBIO, &bval);
#endif

	struct sockaddr_storage their_addr;
	size_t addr_size = _set_sockaddr(&their_addr, p_host, p_port, sock_type);

	errno = 0;
	if (::connect(sockfd, (struct sockaddr *)&their_addr, addr_size) == -1 && errno != EINPROGRESS) {

		ERR_PRINT("Connection to remote host failed!");
		disconnect_from_host();
		return FAILED;
	};

	if (errno == EINPROGRESS) {
		status = STATUS_CONNECTING;
	} else {
		status = STATUS_CONNECTED;
	};

	peer_host = p_host;
	peer_port = p_port;

	return OK;
};

Error StreamPeerTCPPosix::write(const uint8_t *p_data, int p_bytes, int &r_sent, bool p_block) {

	if (status == STATUS_NONE || status == STATUS_ERROR) {

		return FAILED;
	};

	if (status != STATUS_CONNECTED) {

		if (_poll_connection() != OK) {

			return FAILED;
		};

		if (status != STATUS_CONNECTED) {
			r_sent = 0;
			return OK;
		};
	};

	int data_to_send = p_bytes;
	const uint8_t *offset = p_data;
	if (sockfd == -1) return FAILED;
	errno = 0;
	int total_sent = 0;

	while (data_to_send) {

		int sent_amount = send(sockfd, offset, data_to_send, MSG_NOSIGNAL);
		//printf("Sent TCP data of %d bytes, errno %d\n", sent_amount, errno);

		if (sent_amount == -1) {

			if (errno != EAGAIN) {

				perror("shit?");
				disconnect_from_host();
				ERR_PRINT("Server disconnected!\n");
				return FAILED;
			};

			if (!p_block) {
				r_sent = total_sent;
				return OK;
			};

			_block(sockfd, false, true);
		} else {

			data_to_send -= sent_amount;
			offset += sent_amount;
			total_sent += sent_amount;
		};
	}

	r_sent = total_sent;

	return OK;
};

Error StreamPeerTCPPosix::read(uint8_t *p_buffer, int p_bytes, int &r_received, bool p_block) {

	if (!is_connected_to_host()) {

		return FAILED;
	};

	if (status == STATUS_CONNECTING) {

		if (_poll_connection() != OK) {

			return FAILED;
		};

		if (status != STATUS_CONNECTED) {
			r_received = 0;
			return OK;
		};
	};

	int to_read = p_bytes;
	int total_read = 0;
	errno = 0;

	while (to_read) {

		int read = recv(sockfd, p_buffer + total_read, to_read, 0);

		if (read == -1) {

			if (errno != EAGAIN) {

				perror("shit?");
				disconnect_from_host();
				ERR_PRINT("Server disconnected!\n");
				return FAILED;
			};

			if (!p_block) {

				r_received = total_read;
				return OK;
			};
			_block(sockfd, true, false);

		} else if (read == 0) {

			sockfd = -1;
			status = STATUS_NONE;
			peer_port = 0;
			peer_host = IP_Address();
			return ERR_FILE_EOF;

		} else {

			to_read -= read;
			total_read += read;
		};
	};

	r_received = total_read;

	return OK;
};

void StreamPeerTCPPosix::set_nodelay(bool p_enabled) {

	ERR_FAIL_COND(!is_connected_to_host());
	int flag = p_enabled ? 1 : 0;
	setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
}

bool StreamPeerTCPPosix::is_connected_to_host() const {

	if (status == STATUS_NONE || status == STATUS_ERROR) {

		return false;
	};
	if (status != STATUS_CONNECTED) {
		return true;
	};

	return (sockfd != -1);
};

StreamPeerTCP::Status StreamPeerTCPPosix::get_status() const {

	if (status == STATUS_CONNECTING) {
		_poll_connection();
	};

	return status;
};

void StreamPeerTCPPosix::disconnect_from_host() {

	if (sockfd != -1)
		close(sockfd);

	sock_type = IP::TYPE_NONE;
	sockfd = -1;

	status = STATUS_NONE;
	peer_port = 0;
	peer_host = IP_Address();
};

Error StreamPeerTCPPosix::put_data(const uint8_t *p_data, int p_bytes) {

	int total;
	return write(p_data, p_bytes, total, true);
};

Error StreamPeerTCPPosix::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {

	return write(p_data, p_bytes, r_sent, false);
};

Error StreamPeerTCPPosix::get_data(uint8_t *p_buffer, int p_bytes) {

	int total;
	return read(p_buffer, p_bytes, total, true);
};

Error StreamPeerTCPPosix::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {

	return read(p_buffer, p_bytes, r_received, false);
};

int StreamPeerTCPPosix::get_available_bytes() const {

	unsigned long len;
	int ret = ioctl(sockfd, FIONREAD, &len);
	ERR_FAIL_COND_V(ret == -1, 0)
	return len;
}
IP_Address StreamPeerTCPPosix::get_connected_host() const {

	return peer_host;
};

uint16_t StreamPeerTCPPosix::get_connected_port() const {

	return peer_port;
};

StreamPeerTCPPosix::StreamPeerTCPPosix() {

	sock_type = IP::TYPE_NONE;
	sockfd = -1;
	status = STATUS_NONE;
	peer_port = 0;
};

StreamPeerTCPPosix::~StreamPeerTCPPosix() {

	disconnect_from_host();
};

#endif
