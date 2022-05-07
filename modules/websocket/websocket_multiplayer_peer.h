/*************************************************************************/
/*  websocket_multiplayer_peer.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef WEBSOCKET_MULTIPLAYER_PEER_H
#define WEBSOCKET_MULTIPLAYER_PEER_H

#include "core/error_list.h"
#include "core/io/networked_multiplayer_peer.h"
#include "core/list.h"
#include "websocket_peer.h"

class WebSocketMultiplayerPeer : public NetworkedMultiplayerPeer {
	GDCLASS(WebSocketMultiplayerPeer, NetworkedMultiplayerPeer);

private:
	PoolVector<uint8_t> _make_pkt(uint8_t p_type, int32_t p_from, int32_t p_to, const uint8_t *p_data, uint32_t p_data_size);
	void _store_pkt(int32_t p_source, int32_t p_dest, const uint8_t *p_data, uint32_t p_data_size);
	Error _server_relay(int32_t p_from, int32_t p_to, const uint8_t *p_buffer, uint32_t p_buffer_size);

protected:
	enum {
		SYS_NONE = 0,
		SYS_ADD = 1,
		SYS_DEL = 2,
		SYS_ID = 3,

		PROTO_SIZE = 9
	};

	struct Packet {
		int source;
		int destination;
		uint8_t *data;
		uint32_t size;
	};

	List<Packet> _incoming_packets;
	Map<int, Ref<WebSocketPeer>> _peer_map;
	Packet _current_packet;

	bool _is_multiplayer;
	int _target_peer;
	int _peer_id;
	int _refusing;

	static void _bind_methods();

	void _send_add(int32_t p_peer_id);
	void _send_sys(Ref<WebSocketPeer> p_peer, uint8_t p_type, int32_t p_peer_id);
	void _send_del(int32_t p_peer_id);
	int _gen_unique_id() const;

public:
	/* NetworkedMultiplayerPeer */
	void set_transfer_mode(TransferMode p_mode);
	TransferMode get_transfer_mode() const;
	void set_target_peer(int p_target_peer);
	int get_packet_peer() const;
	int get_unique_id() const;
	virtual bool is_server() const = 0;
	void set_refuse_new_connections(bool p_enable);
	bool is_refusing_new_connections() const;
	virtual ConnectionStatus get_connection_status() const = 0;

	/* PacketPeer */
	virtual int get_available_packet_count() const;
	virtual int get_max_packet_size() const = 0;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size);
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size);

	/* WebSocketPeer */
	virtual Error set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets) = 0;
	virtual Ref<WebSocketPeer> get_peer(int p_peer_id) const = 0;

	void _process_multiplayer(Ref<WebSocketPeer> p_peer, uint32_t p_peer_id);
	void _clear();

	WebSocketMultiplayerPeer();
	~WebSocketMultiplayerPeer();
};

#endif // WEBSOCKET_MULTIPLAYER_PEER_H
