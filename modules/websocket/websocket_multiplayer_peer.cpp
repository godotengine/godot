/**************************************************************************/
/*  websocket_multiplayer_peer.cpp                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "websocket_multiplayer_peer.h"

#include "core/os/os.h"

WebSocketMultiplayerPeer::WebSocketMultiplayerPeer() {
	_is_multiplayer = false;
	_peer_id = 0;
	_target_peer = 0;
	_refusing = false;

	_current_packet.source = 0;
	_current_packet.destination = 0;
	_current_packet.size = 0;
	_current_packet.data = nullptr;
}

WebSocketMultiplayerPeer::~WebSocketMultiplayerPeer() {
	_clear();
}

int WebSocketMultiplayerPeer::_gen_unique_id() const {
	uint32_t hash = 0;

	while (hash == 0 || hash == 1) {
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_ticks_usec());
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_unix_time(), hash);
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_data_path().hash64(), hash);
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)this), hash); //rely on aslr heap
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)&hash), hash); //rely on aslr stack
		hash = hash & 0x7FFFFFFF; // make it compatible with unsigned, since negatie id is used for exclusion
	}

	return hash;
}
void WebSocketMultiplayerPeer::_clear() {
	_peer_map.clear();
	if (_current_packet.data != nullptr) {
		memfree(_current_packet.data);
	}

	for (List<Packet>::Element *E = _incoming_packets.front(); E; E = E->next()) {
		memfree(E->get().data);
		E->get().data = nullptr;
	}

	_incoming_packets.clear();
}

void WebSocketMultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_buffers", "input_buffer_size_kb", "input_max_packets", "output_buffer_size_kb", "output_max_packets"), &WebSocketMultiplayerPeer::set_buffers);
	ClassDB::bind_method(D_METHOD("get_peer", "peer_id"), &WebSocketMultiplayerPeer::get_peer);

	ADD_SIGNAL(MethodInfo("peer_packet", PropertyInfo(Variant::INT, "peer_source")));
}

//
// PacketPeer
//
int WebSocketMultiplayerPeer::get_available_packet_count() const {
	ERR_FAIL_COND_V_MSG(!_is_multiplayer, 0, "Please use get_peer(ID).get_available_packet_count to get available packet count from peers when not using the MultiplayerAPI.");

	return _incoming_packets.size();
}

Error WebSocketMultiplayerPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V_MSG(!_is_multiplayer, ERR_UNCONFIGURED, "Please use get_peer(ID).get_packet/var to communicate with peers when not using the MultiplayerAPI.");

	r_buffer_size = 0;

	if (_current_packet.data != nullptr) {
		memfree(_current_packet.data);
		_current_packet.data = nullptr;
	}

	ERR_FAIL_COND_V(_incoming_packets.size() == 0, ERR_UNAVAILABLE);

	_current_packet = _incoming_packets.front()->get();
	_incoming_packets.pop_front();

	*r_buffer = _current_packet.data;
	r_buffer_size = _current_packet.size;

	return OK;
}

Error WebSocketMultiplayerPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V_MSG(!_is_multiplayer, ERR_UNCONFIGURED, "Please use get_peer(ID).put_packet/var to communicate with peers when not using the MultiplayerAPI.");

	PoolVector<uint8_t> buffer = _make_pkt(SYS_NONE, get_unique_id(), _target_peer, p_buffer, p_buffer_size);

	if (is_server()) {
		return _server_relay(1, _target_peer, &(buffer.read()[0]), buffer.size());
	} else {
		return get_peer(1)->put_packet(&(buffer.read()[0]), buffer.size());
	}
}

//
// NetworkedMultiplayerPeer
//
void WebSocketMultiplayerPeer::set_transfer_mode(TransferMode p_mode) {
	// Websocket uses TCP, reliable
}

NetworkedMultiplayerPeer::TransferMode WebSocketMultiplayerPeer::get_transfer_mode() const {
	// Websocket uses TCP, reliable
	return TRANSFER_MODE_RELIABLE;
}

void WebSocketMultiplayerPeer::set_target_peer(int p_target_peer) {
	_target_peer = p_target_peer;
}

int WebSocketMultiplayerPeer::get_packet_peer() const {
	ERR_FAIL_COND_V_MSG(!_is_multiplayer, 1, "This function is not available when not using the MultiplayerAPI.");
	ERR_FAIL_COND_V(_incoming_packets.size() == 0, 1);

	return _incoming_packets.front()->get().source;
}

int WebSocketMultiplayerPeer::get_unique_id() const {
	return _peer_id;
}

void WebSocketMultiplayerPeer::set_refuse_new_connections(bool p_enable) {
	_refusing = p_enable;
}

bool WebSocketMultiplayerPeer::is_refusing_new_connections() const {
	return _refusing;
}

void WebSocketMultiplayerPeer::_send_sys(Ref<WebSocketPeer> p_peer, uint8_t p_type, int32_t p_peer_id) {
	ERR_FAIL_COND(!p_peer.is_valid());
	ERR_FAIL_COND(!p_peer->is_connected_to_host());

	PoolVector<uint8_t> message = _make_pkt(p_type, 1, 0, (uint8_t *)&p_peer_id, 4);
	p_peer->put_packet(&(message.read()[0]), message.size());
}

PoolVector<uint8_t> WebSocketMultiplayerPeer::_make_pkt(uint8_t p_type, int32_t p_from, int32_t p_to, const uint8_t *p_data, uint32_t p_data_size) {
	PoolVector<uint8_t> out;
	out.resize(PROTO_SIZE + p_data_size);

	PoolVector<uint8_t>::Write w = out.write();
	memcpy(&w[0], &p_type, 1);
	memcpy(&w[1], &p_from, 4);
	memcpy(&w[5], &p_to, 4);
	memcpy(&w[PROTO_SIZE], p_data, p_data_size);

	return out;
}

void WebSocketMultiplayerPeer::_send_add(int32_t p_peer_id) {
	// First of all, confirm the ID!
	_send_sys(get_peer(p_peer_id), SYS_ID, p_peer_id);

	// Then send the server peer (which will trigger connection_succeded in client)
	_send_sys(get_peer(p_peer_id), SYS_ADD, 1);

	for (Map<int, Ref<WebSocketPeer>>::Element *E = _peer_map.front(); E; E = E->next()) {
		int32_t id = E->key();
		if (p_peer_id == id) {
			continue; // Skip the newwly added peer (already confirmed)
		}

		// Send new peer to others
		_send_sys(get_peer(id), SYS_ADD, p_peer_id);
		// Send others to new peer
		_send_sys(get_peer(p_peer_id), SYS_ADD, id);
	}
}

void WebSocketMultiplayerPeer::_send_del(int32_t p_peer_id) {
	for (Map<int, Ref<WebSocketPeer>>::Element *E = _peer_map.front(); E; E = E->next()) {
		int32_t id = E->key();
		if (p_peer_id != id) {
			_send_sys(get_peer(id), SYS_DEL, p_peer_id);
		}
	}
}

void WebSocketMultiplayerPeer::_store_pkt(int32_t p_source, int32_t p_dest, const uint8_t *p_data, uint32_t p_data_size) {
	Packet packet;
	packet.data = (uint8_t *)memalloc(p_data_size);
	packet.size = p_data_size;
	packet.source = p_source;
	packet.destination = p_dest;
	memcpy(packet.data, &p_data[PROTO_SIZE], p_data_size);
	_incoming_packets.push_back(packet);
	emit_signal("peer_packet", p_source);
}

Error WebSocketMultiplayerPeer::_server_relay(int32_t p_from, int32_t p_to, const uint8_t *p_buffer, uint32_t p_buffer_size) {
	if (p_to == 1) {
		return OK; // Will not send to self

	} else if (p_to == 0) {
		for (Map<int, Ref<WebSocketPeer>>::Element *E = _peer_map.front(); E; E = E->next()) {
			if (E->key() != p_from) {
				E->get()->put_packet(p_buffer, p_buffer_size);
			}
		}
		return OK; // Sent to all but sender

	} else if (p_to < 0) {
		for (Map<int, Ref<WebSocketPeer>>::Element *E = _peer_map.front(); E; E = E->next()) {
			if (E->key() != p_from && E->key() != -p_to) {
				E->get()->put_packet(p_buffer, p_buffer_size);
			}
		}
		return OK; // Sent to all but sender and excluded

	} else {
		ERR_FAIL_COND_V(p_to == p_from, FAILED);

		Ref<WebSocketPeer> peer_to = get_peer(p_to);
		ERR_FAIL_COND_V(peer_to.is_null(), FAILED);

		return peer_to->put_packet(p_buffer, p_buffer_size); // Sending to specific peer
	}
}

void WebSocketMultiplayerPeer::_process_multiplayer(Ref<WebSocketPeer> p_peer, uint32_t p_peer_id) {
	ERR_FAIL_COND(!p_peer.is_valid());

	const uint8_t *in_buffer;
	int size = 0;
	int data_size = 0;

	Error err = p_peer->get_packet(&in_buffer, size);

	ERR_FAIL_COND(err != OK);
	ERR_FAIL_COND(size < PROTO_SIZE);

	data_size = size - PROTO_SIZE;

	uint8_t type = 0;
	uint32_t from = 0;
	int32_t to = 0;
	memcpy(&type, in_buffer, 1);
	memcpy(&from, &in_buffer[1], 4);
	memcpy(&to, &in_buffer[5], 4);

	if (is_server()) { // Server can resend

		ERR_FAIL_COND(type != SYS_NONE); // Only server sends sys messages
		ERR_FAIL_COND(from != p_peer_id); // Someone is cheating

		if (to == 1) { // This is for the server

			_store_pkt(from, to, in_buffer, data_size);

		} else if (to == 0) {
			// Broadcast, for us too
			_store_pkt(from, to, in_buffer, data_size);

		} else if (to < 0) {
			// All but one, for us if not excluded
			if (_peer_id != -(int32_t)p_peer_id) {
				_store_pkt(from, to, in_buffer, data_size);
			}
		}
		// Relay if needed (i.e. "to" includes a peer that is not the server)
		_server_relay(from, to, in_buffer, size);

	} else {
		if (type == SYS_NONE) { // Payload message

			_store_pkt(from, to, in_buffer, data_size);
			return;
		}

		// System message
		ERR_FAIL_COND(data_size < 4);
		int id = 0;
		memcpy(&id, &in_buffer[PROTO_SIZE], 4);

		switch (type) {
			case SYS_ADD: // Add peer
				_peer_map[id] = Ref<WebSocketPeer>();
				emit_signal("peer_connected", id);
				if (id == 1) { // We just connected to the server
					emit_signal("connection_succeeded");
				}
				break;

			case SYS_DEL: // Remove peer
				_peer_map.erase(id);
				emit_signal("peer_disconnected", id);
				break;
			case SYS_ID: // Helo, server assigned ID
				_peer_id = id;
				break;
			default:
				ERR_FAIL_MSG("Invalid multiplayer message.");
				break;
		}
	}
}
