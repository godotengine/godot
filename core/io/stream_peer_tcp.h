/*************************************************************************/
/*  stream_peer_tcp.h                                                    */
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
#ifndef STREAM_PEER_TCP_H
#define STREAM_PEER_TCP_H

#include "stream_peer.h"

#include "io/ip.h"
#include "ip_address.h"

class StreamPeerTCP : public StreamPeer {

	GDCLASS(StreamPeerTCP, StreamPeer);
	OBJ_CATEGORY("Networking");

public:
	enum Status {

		STATUS_NONE,
		STATUS_CONNECTING,
		STATUS_CONNECTED,
		STATUS_ERROR,
	};

protected:
	virtual Error _connect(const String &p_address, int p_port);
	static StreamPeerTCP *(*_create)();
	static void _bind_methods();

public:
	virtual Error connect_to_host(const IP_Address &p_host, uint16_t p_port) = 0;

	//read/write from streampeer

	virtual bool is_connected_to_host() const = 0;
	virtual Status get_status() const = 0;
	virtual void disconnect_from_host() = 0;
	virtual IP_Address get_connected_host() const = 0;
	virtual uint16_t get_connected_port() const = 0;
	virtual void set_nodelay(bool p_enabled) = 0;

	static Ref<StreamPeerTCP> create_ref();
	static StreamPeerTCP *create();

	StreamPeerTCP();
	~StreamPeerTCP();
};

VARIANT_ENUM_CAST(StreamPeerTCP::Status);

#endif
