/*************************************************************************/
/*  packet_peer_udp_winsock.cpp                                          */
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
#ifdef WINDOWS_ENABLED

#include "packet_peer_udp_winsock.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include "drivers/unix/socket_helpers.h"

int PacketPeerUDPWinsock::get_available_packet_count() const {

	Error err = const_cast<PacketPeerUDPWinsock *>(this)->_poll(false);
	if (err != OK)
		return 0;

	return queue_count;
}

Error PacketPeerUDPWinsock::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {

	Error err = const_cast<PacketPeerUDPWinsock *>(this)->_poll(false);
	if (err != OK)
		return err;
	if (queue_count == 0)
		return ERR_UNAVAILABLE;

	uint32_t size;
	uint8_t type;
	rb.read(&type, 1, true);
	if (type == IP::TYPE_IPV4) {
		uint8_t ip[4];
		rb.read(ip, 4, true);
		packet_ip.set_ipv4(ip);
	} else {
		uint8_t ip[16];
		rb.read(ip, 16, true);
		packet_ip.set_ipv6(ip);
	};
	rb.read((uint8_t *)&packet_port, 4, true);
	rb.read((uint8_t *)&size, 4, true);
	rb.read(packet_buffer, size, true);
	--queue_count;
	*r_buffer = packet_buffer;
	r_buffer_size = size;
	return OK;
}
Error PacketPeerUDPWinsock::put_packet(const uint8_t *p_buffer, int p_buffer_size) {

	ERR_FAIL_COND_V(!peer_addr.is_valid(), ERR_UNCONFIGURED);

	if (sock_type == IP::TYPE_NONE)
		sock_type = peer_addr.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;

	int sock = _get_socket();
	ERR_FAIL_COND_V(sock == -1, FAILED);
	struct sockaddr_storage addr;
	size_t addr_size = _set_sockaddr(&addr, peer_addr, peer_port, sock_type);

	_set_sock_blocking(blocking);

	errno = 0;
	int err;
	while ((err = sendto(sock, (const char *)p_buffer, p_buffer_size, 0, (struct sockaddr *)&addr, addr_size)) != p_buffer_size) {

		if (WSAGetLastError() != WSAEWOULDBLOCK) {
			return FAILED;
		} else if (!blocking) {
			return ERR_UNAVAILABLE;
		}
	}

	return OK;
}

int PacketPeerUDPWinsock::get_max_packet_size() const {

	return 512; // uhm maybe not
}

void PacketPeerUDPWinsock::_set_sock_blocking(bool p_blocking) {

	if (sock_blocking == p_blocking)
		return;

	sock_blocking = p_blocking;
	unsigned long par = sock_blocking ? 0 : 1;
	if (ioctlsocket(sockfd, FIONBIO, &par)) {
		perror("setting non-block mode");
		//close();
		//return -1;
	};
}

Error PacketPeerUDPWinsock::listen(int p_port, const IP_Address &p_bind_address, int p_recv_buffer_size) {

	ERR_FAIL_COND_V(sockfd != -1, ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(!p_bind_address.is_valid() && !p_bind_address.is_wildcard(), ERR_INVALID_PARAMETER);

	sock_type = IP::TYPE_ANY;

	if (p_bind_address.is_valid())
		sock_type = p_bind_address.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;

	int sock = _get_socket();
	if (sock == -1)
		return ERR_CANT_CREATE;

	struct sockaddr_storage addr = { 0 };
	size_t addr_size = _set_listen_sockaddr(&addr, p_port, sock_type, IP_Address());

	if (bind(sock, (struct sockaddr *)&addr, addr_size) == -1) {
		close();
		return ERR_UNAVAILABLE;
	}

	printf("UDP Connection listening on port %i\n", p_port);
	rb.resize(nearest_shift(p_recv_buffer_size));
	return OK;
}

void PacketPeerUDPWinsock::close() {

	if (sockfd != -1)
		::closesocket(sockfd);
	sockfd = -1;
	sock_type = IP::TYPE_NONE;
	rb.resize(16);
	queue_count = 0;
}

Error PacketPeerUDPWinsock::wait() {

	return _poll(true);
}
Error PacketPeerUDPWinsock::_poll(bool p_wait) {

	if (sockfd == -1) {
		return FAILED;
	}

	_set_sock_blocking(p_wait);

	struct sockaddr_storage from = { 0 };
	int len = sizeof(struct sockaddr_storage);
	int ret;
	while ((ret = recvfrom(sockfd, (char *)recv_buffer, MIN((int)sizeof(recv_buffer), MAX(rb.space_left() - 24, 0)), 0, (struct sockaddr *)&from, &len)) > 0) {

		uint32_t port = 0;

		if (from.ss_family == AF_INET) {
			uint8_t type = (uint8_t)IP::TYPE_IPV4;
			rb.write(&type, 1);
			struct sockaddr_in *sin_from = (struct sockaddr_in *)&from;
			rb.write((uint8_t *)&sin_from->sin_addr, 4);
			port = ntohs(sin_from->sin_port);

		} else if (from.ss_family == AF_INET6) {

			uint8_t type = (uint8_t)IP::TYPE_IPV6;
			rb.write(&type, 1);

			struct sockaddr_in6 *s6_from = (struct sockaddr_in6 *)&from;
			rb.write((uint8_t *)&s6_from->sin6_addr, 16);

			port = ntohs(s6_from->sin6_port);

		} else {
			// WARN_PRINT("Ignoring packet with unknown address family");
			uint8_t type = (uint8_t)IP::TYPE_NONE;
			rb.write(&type, 1);
		};

		rb.write((uint8_t *)&port, 4);
		rb.write((uint8_t *)&ret, 4);
		rb.write(recv_buffer, ret);

		len = sizeof(struct sockaddr_storage);
		++queue_count;
		if (p_wait)
			break;
	};

	if (ret == SOCKET_ERROR) {
		int error = WSAGetLastError();

		if (error == WSAEWOULDBLOCK) {
			// Expected when doing non-blocking sockets, retry later.
		} else if (error == WSAECONNRESET) {
			// If the remote target does not accept messages, this error may occur, but is harmless.
			// Once the remote target gets available, this message will disappear for new messages.
		} else {
			close();
			return FAILED;
		}
	}

	if (ret == 0) {
		close();
		return FAILED;
	};

	return OK;
}

bool PacketPeerUDPWinsock::is_listening() const {

	return sockfd != -1;
}

IP_Address PacketPeerUDPWinsock::get_packet_address() const {

	return packet_ip;
}

int PacketPeerUDPWinsock::get_packet_port() const {

	return packet_port;
}

int PacketPeerUDPWinsock::_get_socket() {

	ERR_FAIL_COND_V(sock_type == IP::TYPE_NONE, -1);

	if (sockfd != -1)
		return sockfd;

	sockfd = _socket_create(sock_type, SOCK_DGRAM, IPPROTO_UDP);

	if (sockfd != -1)
		_set_sock_blocking(false);

	return sockfd;
}

void PacketPeerUDPWinsock::set_dest_address(const IP_Address &p_address, int p_port) {

	peer_addr = p_address;
	peer_port = p_port;
}

void PacketPeerUDPWinsock::make_default() {

	PacketPeerUDP::_create = PacketPeerUDPWinsock::_create;
};

PacketPeerUDP *PacketPeerUDPWinsock::_create() {

	return memnew(PacketPeerUDPWinsock);
};

PacketPeerUDPWinsock::PacketPeerUDPWinsock() {

	blocking = true;
	sock_blocking = true;
	sockfd = -1;
	packet_port = 0;
	queue_count = 0;
	peer_port = 0;
	sock_type = IP::TYPE_NONE;
	rb.resize(16);
}

PacketPeerUDPWinsock::~PacketPeerUDPWinsock() {

	close();
}

#endif
