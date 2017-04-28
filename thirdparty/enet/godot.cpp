/*************************************************************************/
/*  godot.cpp                                                            */
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
/**
 @file  godot.cpp
 @brief ENet Godot specific functions
*/

#include "core/io/ip.h"
#include "core/io/packet_peer_udp.h"
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

int enet_socket_bind(ENetSocket socket, const ENetAddress *address) {

	IP_Address ip;
	if (address->wildcard) {
		ip = IP_Address("*");
	} else {
		ip.set_ipv6(address->host);
	}

	PacketPeerUDP *sock = (PacketPeerUDP *)socket;
	if (sock->listen(address->port, ip) != OK) {
		return -1;
	}
	return 0;
}

ENetSocket enet_socket_create(ENetSocketType type) {

	PacketPeerUDP *socket = PacketPeerUDP::create();
	socket->set_blocking_mode(false);

	return socket;
}

void enet_socket_destroy(ENetSocket socket) {
	PacketPeerUDP *sock = (PacketPeerUDP *)socket;
	sock->close();
	memdelete(sock);
}

int enet_socket_send(ENetSocket socket, const ENetAddress *address, const ENetBuffer *buffers, size_t bufferCount) {

	ERR_FAIL_COND_V(address == NULL, -1);

	PacketPeerUDP *sock = (PacketPeerUDP *)socket;
	IP_Address dest;
	Error err;
	size_t i = 0;

	dest.set_ipv6(address->host);
	sock->set_dest_address(dest, address->port);

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

	err = sock->put_packet((const uint8_t *)&w[0], size);
	if (err != OK) {

		if (err == ERR_UNAVAILABLE) { // blocking call
			return 0;
		}

		WARN_PRINT("Sending failed!");
		return -1;
	}

	return size;
}

int enet_socket_receive(ENetSocket socket, ENetAddress *address, ENetBuffer *buffers, size_t bufferCount) {

	ERR_FAIL_COND_V(bufferCount != 1, -1);

	PacketPeerUDP *sock = (PacketPeerUDP *)socket;

	if (sock->get_available_packet_count() == 0) {
		return 0;
	}

	const uint8_t *buffer;
	int buffer_size;
	Error err = sock->get_packet(&buffer, buffer_size);
	if (err)
		return -1;

	copymem(buffers[0].data, buffer, buffer_size);

	enet_address_set_ip(address, sock->get_packet_address().get_ipv6(), 16);
	address->port = sock->get_packet_port();

	return buffer_size;
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
