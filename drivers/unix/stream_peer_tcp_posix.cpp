/*************************************************************************/
/*  stream_peer_tcp_posix.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
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
	#define MSG_NOSIGNAL    SO_NOSIGPIPE
#endif

static void set_addr_in(struct sockaddr_in& their_addr, const IP_Address& p_host, uint16_t p_port) {

	their_addr.sin_family = AF_INET;    // host byte order
	their_addr.sin_port = htons(p_port);  // short, network byte order
	their_addr.sin_addr = *((struct in_addr*)&p_host.host);
	memset(&(their_addr.sin_zero), '\0', 8);
};

StreamPeerTCP* StreamPeerTCPPosix::_create() {

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

Error StreamPeerTCPPosix::_poll_connection(bool p_block) const {

	ERR_FAIL_COND_V(status != STATUS_CONNECTING || sockfd == -1, FAILED);

	if (p_block) {

		_block(sockfd, false, true);
	};

	struct sockaddr_in their_addr;
	set_addr_in(their_addr, peer_host, peer_port);
	if (::connect(sockfd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1) {

		if (errno == EISCONN) {
			status = STATUS_CONNECTED;
			return OK;
		};

		return OK;
	} else {

		status = STATUS_CONNECTED;
		return OK;
	};

	return OK;
};

void StreamPeerTCPPosix::set_socket(int p_sockfd, IP_Address p_host, int p_port) {

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

Error StreamPeerTCPPosix::connect(const IP_Address& p_host, uint16_t p_port) {

	ERR_FAIL_COND_V( p_host.host == 0, ERR_INVALID_PARAMETER);

	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		ERR_PRINT("Socket creation failed!");
		disconnect();
		//perror("socket");
		return FAILED;
	};

#ifndef NO_FCNTL
	fcntl(sockfd, F_SETFL, O_NONBLOCK);
#else
	int bval = 1;
	ioctl(sockfd, FIONBIO, &bval);
#endif

	struct sockaddr_in their_addr;
	set_addr_in(their_addr, p_host, p_port);

	errno = 0;
	if (::connect(sockfd, (struct sockaddr *)&their_addr,sizeof(struct sockaddr)) == -1 && errno != EINPROGRESS) {

		ERR_PRINT("Connection to remote host failed!");
		disconnect();
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

Error StreamPeerTCPPosix::write(const uint8_t* p_data,int p_bytes, int &r_sent, bool p_block) {

	if (status == STATUS_NONE || status == STATUS_ERROR) {

		return FAILED;
	};

	if (status != STATUS_CONNECTED) {

		if (_poll_connection(p_block) != OK) {

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
				disconnect();
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

Error StreamPeerTCPPosix::read(uint8_t* p_buffer, int p_bytes,int &r_received, bool p_block) {

	if (!is_connected()) {

		return FAILED;
	};

	if (status == STATUS_CONNECTING) {

		if (_poll_connection(p_block) != OK) {

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
				disconnect();
				ERR_PRINT("Server disconnected!\n");
				return FAILED;
			};

			if (!p_block) {

				r_received = total_read;
				return OK;
			};
			_block(sockfd, true, false);

		} else if (read==0) {

			sockfd=-1;
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

	ERR_FAIL_COND(!is_connected());
	int flag=p_enabled?1:0;
	setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));
}

bool StreamPeerTCPPosix::is_connected() const {

	if (status == STATUS_NONE || status == STATUS_ERROR) {

		return false;
	};
	if (status != STATUS_CONNECTED) {
		return true;
	};

	return (sockfd!=-1);
};

StreamPeerTCP::Status StreamPeerTCPPosix::get_status() const {

	if (status == STATUS_CONNECTING) {
		_poll_connection(false);
	};

	return status;
};


void StreamPeerTCPPosix::disconnect() {

	if (sockfd != -1)
		close(sockfd);
	sockfd=-1;

	status = STATUS_NONE;
	peer_port = 0;
	peer_host = IP_Address();
};


Error StreamPeerTCPPosix::put_data(const uint8_t* p_data,int p_bytes) {

	int total;
	return write(p_data, p_bytes, total, true);
};

Error StreamPeerTCPPosix::put_partial_data(const uint8_t* p_data,int p_bytes, int &r_sent) {

	return write(p_data, p_bytes, r_sent, false);
};

Error StreamPeerTCPPosix::get_data(uint8_t* p_buffer, int p_bytes) {

	int total;
	return read(p_buffer, p_bytes, total, true);
};

Error StreamPeerTCPPosix::get_partial_data(uint8_t* p_buffer, int p_bytes,int &r_received) {

	return read(p_buffer, p_bytes, r_received, false);
};

IP_Address StreamPeerTCPPosix::get_connected_host() const {

	return peer_host;
};

uint16_t StreamPeerTCPPosix::get_connected_port() const {

	return peer_port;
};

StreamPeerTCPPosix::StreamPeerTCPPosix() {

	sockfd = -1;
	status = STATUS_NONE;
	peer_port = 0;
};

StreamPeerTCPPosix::~StreamPeerTCPPosix() {

	disconnect();
};

#endif
