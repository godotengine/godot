/*************************************************************************/
/*  godot.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
/**
 @file  godot.cpp
 @brief ENet Godot specific functions
*/

#include "core/io/ip.h"
#include "core/io/net_socket.h"
#include "core/os/os.h"

// This must be last for windows to compile (tested with MinGW)
#include "enet/enet.h"

static enet_uint32 timeBase = 0;

int enet_initialize(void) {

	return 0;
}

void enet_deinitialize(void) {
}

enet_uint32 enet_host_random_seed(void) {

	return (enet_uint32)OS::get_singleton()->get_unix_time();
}

enet_uint32 enet_time_get(void) {

	return OS::get_singleton()->get_ticks_msec() - timeBase;
}

void enet_time_set(enet_uint32 newTimeBase) {

	timeBase = OS::get_singleton()->get_ticks_msec() - newTimeBase;
}

int enet_address_set_host(ENetAddress *address, const char *name) {

	IP_Address ip = IP::get_singleton()->resolve_hostname(name);
	ERR_FAIL_COND_V(!ip.is_valid(), -1);

	enet_address_set_ip(address, ip.get_ipv6(), 16);
	return 0;
}

void enet_address_set_ip(ENetAddress *address, const uint8_t *ip, size_t size) {

	int len = size > 16 ? 16 : size;
	memset(address->host, 0, 16);
	memcpy(address->host, ip, len);
}

int enet_address_get_host_ip(const ENetAddress *address, char *name, size_t nameLength) {

	return -1;
}

int enet_address_get_host(const ENetAddress *address, char *name, size_t nameLength) {

	return -1;
}

ENetSocket enet_socket_create(ENetSocketType type) {

	NetSocket *socket = NetSocket::create();
	IP::Type ip_type = IP::TYPE_ANY;
	socket->open(NetSocket::TYPE_UDP, ip_type);

	return socket;
}

int enet_socket_bind(ENetSocket socket, const ENetAddress *address) {

	IP_Address ip;
	if (address->wildcard) {
		ip = IP_Address("*");
	} else {
		ip.set_ipv6(address->host);
	}

	NetSocket *sock = (NetSocket *)socket;
	if (sock->bind(ip, address->port) != OK) {
		return -1;
	}
	return 0;
}

void enet_socket_destroy(ENetSocket socket) {
	NetSocket *sock = (NetSocket *)socket;
	sock->close();
	memdelete(sock);
}

int enet_socket_send(ENetSocket socket, const ENetAddress *address, const ENetBuffer *buffers, size_t bufferCount) {

	ERR_FAIL_COND_V(address == NULL, -1);

	NetSocket *sock = (NetSocket *)socket;
	IP_Address dest;
	Error err;
	size_t i = 0;

	dest.set_ipv6(address->host);

	// Create a single packet.
	PoolVector<uint8_t> out;
	PoolVector<uint8_t>::Write w;
	int size = 0;
	int pos = 0;
	for (i = 0; i < bufferCount; i++) {
		size += buffers[i].dataLength;
	}

	out.resize(size);
	w = out.write();
	for (i = 0; i < bufferCount; i++) {
		memcpy(&w[pos], buffers[i].data, buffers[i].dataLength);
		pos += buffers[i].dataLength;
	}

	int sent = 0;
	err = sock->sendto((const uint8_t *)&w[0], size, sent, dest, address->port);
	if (err != OK) {

		if (err == ERR_BUSY) { // Blocking call
			return 0;
		}

		WARN_PRINT("Sending failed!");
		return -1;
	}

	return sent;
}

int enet_socket_receive(ENetSocket socket, ENetAddress *address, ENetBuffer *buffers, size_t bufferCount) {

	ERR_FAIL_COND_V(bufferCount != 1, -1);

	NetSocket *sock = (NetSocket *)socket;

	Error ret = sock->poll(NetSocket::POLL_TYPE_IN, 0);

	if (ret == ERR_BUSY)
		return 0;

	if (ret != OK)
		return -1;

	int read;
	IP_Address ip;

	Error err = sock->recvfrom((uint8_t *)buffers[0].data, buffers[0].dataLength, read, ip, address->port);
	if (err == ERR_BUSY)
		return 0;

	if (err != OK)
		return -1;

	enet_address_set_ip(address, ip.get_ipv6(), 16);

	return read;
}

// Not implemented
int enet_socket_wait(ENetSocket socket, enet_uint32 *condition, enet_uint32 timeout) {

	return 0; // do we need this function?
}

int enet_socket_get_address(ENetSocket socket, ENetAddress *address) {

	return -1; // do we need this function?
}

int enet_socketset_select(ENetSocket maxSocket, ENetSocketSet *readSet, ENetSocketSet *writeSet, enet_uint32 timeout) {

	return -1;
}

int enet_socket_listen(ENetSocket socket, int backlog) {

	return -1;
}

int enet_socket_set_option(ENetSocket socket, ENetSocketOption option, int value) {

	NetSocket *sock = (NetSocket *)socket;

	switch (option) {
		case ENET_SOCKOPT_NONBLOCK: {
			sock->set_blocking_enabled(value ? false : true);
			return 0;
		} break;

		case ENET_SOCKOPT_BROADCAST: {
			sock->set_broadcasting_enabled(value ? true : false);
			return 0;
		} break;

		case ENET_SOCKOPT_REUSEADDR: {
			sock->set_reuse_address_enabled(value ? true : false);
			return 0;
		} break;

		case ENET_SOCKOPT_RCVBUF: {
			return -1;
		} break;

		case ENET_SOCKOPT_SNDBUF: {
			return -1;
		} break;

		case ENET_SOCKOPT_RCVTIMEO: {
			return -1;
		} break;

		case ENET_SOCKOPT_SNDTIMEO: {
			return -1;
		} break;

		case ENET_SOCKOPT_NODELAY: {
			sock->set_tcp_no_delay_enabled(value ? true : false);
			return 0;
		} break;
	}

	return -1;
}

int enet_socket_get_option(ENetSocket socket, ENetSocketOption option, int *value) {

	return -1;
}

int enet_socket_connect(ENetSocket socket, const ENetAddress *address) {

	return -1;
}

ENetSocket enet_socket_accept(ENetSocket socket, ENetAddress *address) {

	return NULL;
}

int enet_socket_shutdown(ENetSocket socket, ENetSocketShutdown how) {

	return -1;
}
