/*************************************************************************/
/*  packet_peer_udp.h                                                    */
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
#ifndef PACKET_PEER_UDP_H
#define PACKET_PEER_UDP_H

#include "io/ip.h"
#include "io/packet_peer.h"

class PacketPeerUDP : public PacketPeer {
	GDCLASS(PacketPeerUDP, PacketPeer);

protected:
	bool blocking;

	static PacketPeerUDP *(*_create)();
	static void _bind_methods();

	String _get_packet_ip() const;

	Error _set_dest_address(const String &p_address, int p_port);

public:
	void set_blocking_mode(bool p_enable);

	virtual Error listen(int p_port, IP_Address p_bind_address = IP_Address("*"), int p_recv_buffer_size = 65536) = 0;
	virtual void close() = 0;
	virtual Error wait() = 0;
	virtual bool is_listening() const = 0;
	virtual IP_Address get_packet_address() const = 0;
	virtual int get_packet_port() const = 0;
	virtual void set_dest_address(const IP_Address &p_address, int p_port) = 0;

	static Ref<PacketPeerUDP> create_ref();
	static PacketPeerUDP *create();

	PacketPeerUDP();
};

#endif // PACKET_PEER_UDP_H
