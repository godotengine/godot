/**************************************************************************/
/*  networked_multiplayer_custom.cpp                                      */
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

#include "core/io/networked_multiplayer_custom.h"

NetworkedMultiplayerCustom::NetworkedMultiplayerCustom() {
	self_id = 0;
	transfer_mode = TransferMode::TRANSFER_MODE_RELIABLE;
	connection_status = ConnectionStatus::CONNECTION_DISCONNECTED;
	refusing_new_connections = false;
	target_id = 0;
	// Default to a large value.
	max_packet_size = 1 << 24;
}

NetworkedMultiplayerCustom::~NetworkedMultiplayerCustom() {
}

void NetworkedMultiplayerCustom::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "self_peer_id"), &NetworkedMultiplayerCustom::initialize);
	ClassDB::bind_method(D_METHOD("set_max_packet_size", "max_packet_size"), &NetworkedMultiplayerCustom::set_max_packet_size);
	ClassDB::bind_method(D_METHOD("set_connection_status", "connection_status"), &NetworkedMultiplayerCustom::set_connection_status);
	ClassDB::bind_method(D_METHOD("deliver_packet", "buffer", "from_peer_id"), &NetworkedMultiplayerCustom::deliver_packet);

	ADD_SIGNAL(MethodInfo("packet_generated", PropertyInfo(Variant::INT, "peer_id"), PropertyInfo(Variant::POOL_BYTE_ARRAY, "buffer"), PropertyInfo(Variant::INT, "transfer_mode")));
}

//
// PacketPeer
//

Error NetworkedMultiplayerCustom::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(incoming_packets.size() == 0, Error::ERR_UNAVAILABLE);

	current_packet = incoming_packets.front()->get();
	incoming_packets.pop_front();

	*r_buffer = current_packet.data.read().ptr();
	r_buffer_size = current_packet.data.size();

	return Error::OK;
}

Error NetworkedMultiplayerCustom::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	PoolByteArray a;
	a.resize(p_buffer_size);

	PoolByteArray::Write w = a.write();
	memcpy(w.ptr(), p_buffer, p_buffer_size);

	emit_signal("packet_generated", target_id, a, transfer_mode);

	return Error::OK;
}

int NetworkedMultiplayerCustom::get_available_packet_count() const {
	return incoming_packets.size();
}

int NetworkedMultiplayerCustom::get_max_packet_size() const {
	return max_packet_size;
}

//
// NetworkedMultiplayerPeer.
//

void NetworkedMultiplayerCustom::set_transfer_mode(NetworkedMultiplayerPeer::TransferMode p_mode) {
	transfer_mode = p_mode;
}

NetworkedMultiplayerPeer::TransferMode NetworkedMultiplayerCustom::get_transfer_mode() const {
	return transfer_mode;
}

void NetworkedMultiplayerCustom::set_target_peer(int p_target_peer) {
	target_id = p_target_peer;
}

int NetworkedMultiplayerCustom::get_packet_peer() const {
	ERR_FAIL_COND_V(connection_status != ConnectionStatus::CONNECTION_CONNECTED, 1);
	ERR_FAIL_COND_V(incoming_packets.size() == 0, 1);

	return incoming_packets.front()->get().from;
}

bool NetworkedMultiplayerCustom::is_server() const {
	return self_id == 1;
}

void NetworkedMultiplayerCustom::poll() {
}

int NetworkedMultiplayerCustom::get_unique_id() const {
	return self_id;
}

void NetworkedMultiplayerCustom::set_refuse_new_connections(bool p_refuse_new_connections) {
	refusing_new_connections = p_refuse_new_connections;
}

bool NetworkedMultiplayerCustom::is_refusing_new_connections() const {
	return refusing_new_connections;
}

NetworkedMultiplayerPeer::ConnectionStatus NetworkedMultiplayerCustom::get_connection_status() const {
	return connection_status;
}

//
// Custom methods.
//

void NetworkedMultiplayerCustom::initialize(int p_self_id) {
	ERR_FAIL_COND_MSG(connection_status != ConnectionStatus::CONNECTION_CONNECTING,
			"Can only initialize if connection status is CONNECTION_CONNECTING.");
	ERR_FAIL_COND_MSG(p_self_id < 0 || p_self_id > ~(1 << 31),
			"Cannot initialize with invalid unique network id.");

	self_id = p_self_id;
	if (self_id == 1) {
		set_connection_status(ConnectionStatus::CONNECTION_CONNECTED);
	}
}

void NetworkedMultiplayerCustom::set_max_packet_size(int p_max_packet_size) {
	max_packet_size = p_max_packet_size;
}

void NetworkedMultiplayerCustom::set_connection_status(NetworkedMultiplayerPeer::ConnectionStatus p_connection_status) {
	if (connection_status == p_connection_status) {
		return;
	}

	ERR_FAIL_COND_MSG(p_connection_status == ConnectionStatus::CONNECTION_CONNECTING && connection_status != ConnectionStatus::CONNECTION_DISCONNECTED,
			"Can only change connection status to CONNECTION_CONNECTING from CONNECTION_DISCONNECTED.");
	ERR_FAIL_COND_MSG(p_connection_status == ConnectionStatus::CONNECTION_CONNECTED && connection_status != ConnectionStatus::CONNECTION_CONNECTING,
			"Can only change connection status to CONNECTION_CONNECTED from CONNECTION_CONNECTING.");

	if (p_connection_status == ConnectionStatus::CONNECTION_CONNECTED) {
		connection_status = p_connection_status;
		emit_signal("connection_succeeded");
	} else if (p_connection_status == ConnectionStatus::CONNECTION_DISCONNECTED) {
		ConnectionStatus old_connection_status = connection_status;
		connection_status = p_connection_status;

		if (self_id != 1) {
			if (old_connection_status == ConnectionStatus::CONNECTION_CONNECTING) {
				emit_signal("connection_failed");
			} else {
				emit_signal("server_disconnected");
			}
		}
	} else {
		connection_status = p_connection_status;
	}
}

void NetworkedMultiplayerCustom::deliver_packet(const PoolByteArray &p_data, int p_from_peer_id) {
	Packet p = { p_data, p_from_peer_id };
	incoming_packets.push_back(p);
}
