/*************************************************************************/
/*  tcp_server_winsock.cpp                                               */
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
#include "tcp_server_winsock.h"

#include "stream_peer_winsock.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include "drivers/unix/socket_helpers.h"

extern int winsock_refcount;

TCP_Server *TCPServerWinsock::_create() {

	return memnew(TCPServerWinsock);
};

void TCPServerWinsock::make_default() {

	TCP_Server::_create = TCPServerWinsock::_create;

	if (winsock_refcount == 0) {
		WSADATA data;
		WSAStartup(MAKEWORD(2, 2), &data);
	};
	++winsock_refcount;
};

void TCPServerWinsock::cleanup() {

	--winsock_refcount;
	if (winsock_refcount == 0) {

		WSACleanup();
	};
};

Error TCPServerWinsock::listen(uint16_t p_port, const IP_Address p_bind_address) {

	ERR_FAIL_COND_V(listen_sockfd != -1, ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(!p_bind_address.is_valid() && !p_bind_address.is_wildcard(), ERR_INVALID_PARAMETER);

	int sockfd;
	sock_type = IP::TYPE_ANY;

	// If the bind address is valid use its type as the socket type
	if (p_bind_address.is_valid())
		sock_type = p_bind_address.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;

	sockfd = _socket_create(sock_type, SOCK_STREAM, IPPROTO_TCP);
	ERR_FAIL_COND_V(sockfd == INVALID_SOCKET, FAILED);

	unsigned long par = 1;
	if (ioctlsocket(sockfd, FIONBIO, &par)) {
		perror("setting non-block mode");
		stop();
		return FAILED;
	};

	struct sockaddr_storage my_addr;
	size_t addr_size = _set_listen_sockaddr(&my_addr, p_port, sock_type, p_bind_address);

	int reuse = 1;
	if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(reuse)) < 0) {

		printf("REUSEADDR failed!");
	}

	if (bind(sockfd, (struct sockaddr *)&my_addr, addr_size) != SOCKET_ERROR) {

		if (::listen(sockfd, SOMAXCONN) == SOCKET_ERROR) {

			closesocket(sockfd);
			ERR_FAIL_V(FAILED);
		};
	} else {
		return ERR_ALREADY_IN_USE;
	};

	if (listen_sockfd != INVALID_SOCKET) {

		stop();
	};

	listen_sockfd = sockfd;

	return OK;
};

bool TCPServerWinsock::is_connection_available() const {

	if (listen_sockfd == -1) {
		return false;
	};

	timeval timeout;
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;

	fd_set pfd;
	FD_ZERO(&pfd);
	FD_SET(listen_sockfd, &pfd);

	int ret = select(listen_sockfd + 1, &pfd, NULL, NULL, &timeout);
	ERR_FAIL_COND_V(ret < 0, 0);

	if (ret && (FD_ISSET(listen_sockfd, &pfd))) {

		return true;
	};

	return false;
};

Ref<StreamPeerTCP> TCPServerWinsock::take_connection() {

	if (!is_connection_available()) {
		return NULL;
	};

	struct sockaddr_storage their_addr;
	int sin_size = sizeof(their_addr);
	int fd = accept(listen_sockfd, (struct sockaddr *)&their_addr, &sin_size);
	ERR_FAIL_COND_V(fd == INVALID_SOCKET, NULL);

	Ref<StreamPeerWinsock> conn = memnew(StreamPeerWinsock);
	IP_Address ip;
	int port;
	_set_ip_addr_port(ip, port, &their_addr);

	conn->set_socket(fd, ip, port, sock_type);

	return conn;
};

void TCPServerWinsock::stop() {

	if (listen_sockfd != INVALID_SOCKET) {
		closesocket(listen_sockfd);
	};

	listen_sockfd = -1;
	sock_type = IP::TYPE_NONE;
};

TCPServerWinsock::TCPServerWinsock() {

	listen_sockfd = INVALID_SOCKET;
	sock_type = IP::TYPE_NONE;
};

TCPServerWinsock::~TCPServerWinsock() {

	stop();
};
