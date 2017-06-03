/*************************************************************************/
/*  packet_peer_udp_winsock.h                                            */
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
#ifndef PACKET_PEER_UDP_WINSOCK_H
#define PACKET_PEER_UDP_WINSOCK_H

#include "io/packet_peer_udp.h"
#include "ring_buffer.h"

class PacketPeerUDPWinsock : public PacketPeerUDP {

	enum {
		PACKET_BUFFER_SIZE = 65536
	};

	mutable RingBuffer<uint8_t> rb;
	uint8_t recv_buffer[PACKET_BUFFER_SIZE];
	mutable uint8_t packet_buffer[PACKET_BUFFER_SIZE];
	mutable IP_Address packet_ip;
	mutable int packet_port;
	mutable int queue_count;
	int sockfd;
	bool sock_blocking;
	IP::Type sock_type;

	IP_Address peer_addr;
	int peer_port;

	_FORCE_INLINE_ int _get_socket();

	static PacketPeerUDP *_create();

	void _set_sock_blocking(bool p_blocking);

	Error _poll(bool p_wait);

public:
	virtual int get_available_packet_count() const;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) const;
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size);

	virtual int get_max_packet_size() const;

	virtual Error listen(int p_port, IP_Address p_bind_address = IP_Address("*"), int p_recv_buffer_size = 65536);
	virtual void close();
	virtual Error wait();
	virtual bool is_listening() const;

	virtual IP_Address get_packet_address() const;
	virtual int get_packet_port() const;

	virtual void set_dest_address(const IP_Address &p_address, int p_port);

	static void make_default();
	PacketPeerUDPWinsock();
	~PacketPeerUDPWinsock();
};
#endif // PACKET_PEER_UDP_WINSOCK_H
