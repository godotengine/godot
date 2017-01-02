/*************************************************************************/
/*  tcp_server_posix.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "tcp_server_posix.h"
#include "stream_peer_tcp_posix.h"

#ifdef UNIX_ENABLED

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
#ifdef JAVASCRIPT_ENABLED
#include <arpa/inet.h>
#endif
#include <netinet/in.h>
#include <sys/socket.h>
#include <assert.h>

#include "drivers/unix/socket_helpers.h"

TCP_Server* TCPServerPosix::_create() {

	return memnew(TCPServerPosix);
};

void TCPServerPosix::make_default() {

	TCP_Server::_create = TCPServerPosix::_create;
};

Error TCPServerPosix::listen(uint16_t p_port,const List<String> *p_accepted_hosts) {

	int sockfd;
	sockfd = _socket_create(ip_type, SOCK_STREAM, IPPROTO_TCP);

	ERR_FAIL_COND_V(sockfd == -1, FAILED);

#ifndef NO_FCNTL
	fcntl(sockfd, F_SETFL, O_NONBLOCK);
#else
	int bval = 1;
	ioctl(sockfd, FIONBIO, &bval);
#endif

	int reuse=1;
	if(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(reuse)) < 0) {
		WARN_PRINT("REUSEADDR failed!")
	}

	struct sockaddr_storage addr;
	size_t addr_size = _set_listen_sockaddr(&addr, p_port, ip_type, p_accepted_hosts);

	// automatically fill with my IP TODO: use p_accepted_hosts

	if (bind(sockfd, (struct sockaddr *)&addr, addr_size) != -1) {

		if (::listen(sockfd, 1) == -1) {

			close(sockfd);
			ERR_FAIL_V(FAILED);
		};
	}
	else {
		return ERR_ALREADY_IN_USE;
	};

	if (listen_sockfd != -1) {
		stop();
	};

	listen_sockfd = sockfd;

	return OK;
};

bool TCPServerPosix::is_connection_available() const {

	if (listen_sockfd == -1) {
		return false;
	};

	struct pollfd pfd;
	pfd.fd = listen_sockfd;
	pfd.events = POLLIN;
	pfd.revents = 0;

	int ret = poll(&pfd, 1, 0);
	ERR_FAIL_COND_V(ret < 0, FAILED);

	if (ret && (pfd.revents & POLLIN)) {
		return true;
	};

	return false;
};

Ref<StreamPeerTCP> TCPServerPosix::take_connection() {

	if (!is_connection_available()) {
		return Ref<StreamPeerTCP>();
	};

	struct sockaddr_storage their_addr;
	socklen_t size = sizeof(their_addr);
	int fd = accept(listen_sockfd, (struct sockaddr *)&their_addr, &size);
	ERR_FAIL_COND_V(fd == -1, Ref<StreamPeerTCP>());
#ifndef NO_FCNTL
	fcntl(fd, F_SETFL, O_NONBLOCK);
#else
	int bval = 1;
	ioctl(fd, FIONBIO, &bval);
#endif

	Ref<StreamPeerTCPPosix> conn = memnew(StreamPeerTCPPosix);
	IP_Address ip;

	int port;
	_set_ip_addr_port(ip, port, &their_addr);

	conn->set_socket(fd, ip, port, ip_type);

	return conn;
};

void TCPServerPosix::stop() {

	if (listen_sockfd != -1) {
		int ret = close(listen_sockfd);
		ERR_FAIL_COND(ret!=0);
	};

	listen_sockfd = -1;
};


TCPServerPosix::TCPServerPosix() {

	listen_sockfd = -1;
	ip_type = IP::TYPE_ANY;
};

TCPServerPosix::~TCPServerPosix() {

	stop();
};
#endif
