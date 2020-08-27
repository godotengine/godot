/*************************************************************************/
/*  stream_peer_tcp.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/io/ip.h"
#include "core/io/ip_address.h"
#include "core/io/net_socket.h"
#include "core/io/stream_peer.h"

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
	Ref<NetSocket> _sock;
	uint64_t timeout = 0;
	Status status = STATUS_NONE;
	IP_Address peer_host;
	uint16_t peer_port = 0;

	Error _connect(const String &p_address, int p_port);
	Error _poll_connection();
	Error write(const uint8_t *p_data, int p_bytes, int &r_sent, bool p_block);
	Error read(uint8_t *p_buffer, int p_bytes, int &r_received, bool p_block);

	static void _bind_methods();

public:
	void accept_socket(Ref<NetSocket> p_sock, IP_Address p_host, uint16_t p_port);

	Error connect_to_host(const IP_Address &p_host, uint16_t p_port);
	bool is_connected_to_host() const;
	IP_Address get_connected_host() const;
	uint16_t get_connected_port() const;
	void disconnect_from_host();

	int get_available_bytes() const override;
	Status get_status();

	void set_no_delay(bool p_enabled);

	// Poll functions (wait or check for writable, readable)
	Error poll(NetSocket::PollType p_type, int timeout = 0);

	// Read/Write from StreamPeer
	Error put_data(const uint8_t *p_data, int p_bytes) override;
	Error put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) override;
	Error get_data(uint8_t *p_buffer, int p_bytes) override;
	Error get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) override;

	StreamPeerTCP();
	~StreamPeerTCP();
};

VARIANT_ENUM_CAST(StreamPeerTCP::Status);

#endif // STREAM_PEER_TCP_H
