/*************************************************************************/
/*  multiplayer_peer.cpp                                                 */
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

#include "multiplayer_peer.h"

#include "core/os/os.h"

uint32_t MultiplayerPeer::generate_unique_id() const {
	uint32_t hash = 0;

	while (hash == 0 || hash == 1) {
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_ticks_usec());
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_unix_time(), hash);
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_user_data_dir().hash64(), hash);
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)this), hash); // Rely on ASLR heap
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)&hash), hash); // Rely on ASLR stack

		hash = hash & 0x7FFFFFFF; // Make it compatible with unsigned, since negative ID is used for exclusion
	}

	return hash;
}

void MultiplayerPeer::set_transfer_channel(int p_channel) {
	transfer_channel = p_channel;
}

int MultiplayerPeer::get_transfer_channel() const {
	return transfer_channel;
}

void MultiplayerPeer::set_transfer_mode(Multiplayer::TransferMode p_mode) {
	transfer_mode = p_mode;
}

Multiplayer::TransferMode MultiplayerPeer::get_transfer_mode() const {
	return transfer_mode;
}

void MultiplayerPeer::set_refuse_new_connections(bool p_enable) {
	refuse_connections = p_enable;
}

bool MultiplayerPeer::is_refusing_new_connections() const {
	return refuse_connections;
}

void MultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transfer_channel", "channel"), &MultiplayerPeer::set_transfer_channel);
	ClassDB::bind_method(D_METHOD("get_transfer_channel"), &MultiplayerPeer::get_transfer_channel);
	ClassDB::bind_method(D_METHOD("set_transfer_mode", "mode"), &MultiplayerPeer::set_transfer_mode);
	ClassDB::bind_method(D_METHOD("get_transfer_mode"), &MultiplayerPeer::get_transfer_mode);
	ClassDB::bind_method(D_METHOD("set_target_peer", "id"), &MultiplayerPeer::set_target_peer);

	ClassDB::bind_method(D_METHOD("get_packet_peer"), &MultiplayerPeer::get_packet_peer);

	ClassDB::bind_method(D_METHOD("poll"), &MultiplayerPeer::poll);

	ClassDB::bind_method(D_METHOD("get_connection_status"), &MultiplayerPeer::get_connection_status);
	ClassDB::bind_method(D_METHOD("get_unique_id"), &MultiplayerPeer::get_unique_id);
	ClassDB::bind_method(D_METHOD("generate_unique_id"), &MultiplayerPeer::generate_unique_id);

	ClassDB::bind_method(D_METHOD("set_refuse_new_connections", "enable"), &MultiplayerPeer::set_refuse_new_connections);
	ClassDB::bind_method(D_METHOD("is_refusing_new_connections"), &MultiplayerPeer::is_refusing_new_connections);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_connections"), "set_refuse_new_connections", "is_refusing_new_connections");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transfer_mode", PROPERTY_HINT_ENUM, "Unreliable,Unreliable Ordered,Reliable"), "set_transfer_mode", "get_transfer_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transfer_channel", PROPERTY_HINT_RANGE, "0,255,1"), "set_transfer_channel", "get_transfer_channel");

	BIND_ENUM_CONSTANT(CONNECTION_DISCONNECTED);
	BIND_ENUM_CONSTANT(CONNECTION_CONNECTING);
	BIND_ENUM_CONSTANT(CONNECTION_CONNECTED);

	BIND_CONSTANT(TARGET_PEER_BROADCAST);
	BIND_CONSTANT(TARGET_PEER_SERVER);

	ADD_SIGNAL(MethodInfo("peer_connected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_disconnected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("server_disconnected"));
	ADD_SIGNAL(MethodInfo("connection_succeeded"));
	ADD_SIGNAL(MethodInfo("connection_failed"));
}

/*************/

int MultiplayerPeerExtension::get_available_packet_count() const {
	int count;
	if (GDVIRTUAL_CALL(_get_available_packet_count, count)) {
		return count;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_available_packet_count is unimplemented!");
	return -1;
}

Error MultiplayerPeerExtension::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	int err;
	if (GDVIRTUAL_CALL(_get_packet, r_buffer, &r_buffer_size, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_packet_native is unimplemented!");
	return FAILED;
}

Error MultiplayerPeerExtension::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	int err;
	if (GDVIRTUAL_CALL(_put_packet, p_buffer, p_buffer_size, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_put_packet_native is unimplemented!");
	return FAILED;
}

int MultiplayerPeerExtension::get_max_packet_size() const {
	int size;
	if (GDVIRTUAL_CALL(_get_max_packet_size, size)) {
		return size;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_max_packet_size is unimplemented!");
	return 0;
}

void MultiplayerPeerExtension::set_transfer_channel(int p_channel) {
	if (GDVIRTUAL_CALL(_set_transfer_channel, p_channel)) {
		return;
	}
	MultiplayerPeer::set_transfer_channel(p_channel);
}

int MultiplayerPeerExtension::get_transfer_channel() const {
	int channel;
	if (GDVIRTUAL_CALL(_get_transfer_channel, channel)) {
		return channel;
	}
	return MultiplayerPeer::get_transfer_channel();
}

void MultiplayerPeerExtension::set_transfer_mode(Multiplayer::TransferMode p_mode) {
	if (GDVIRTUAL_CALL(_set_transfer_mode, p_mode)) {
		return;
	}
	MultiplayerPeer::set_transfer_mode(p_mode);
}

Multiplayer::TransferMode MultiplayerPeerExtension::get_transfer_mode() const {
	int mode;
	if (GDVIRTUAL_CALL(_get_transfer_mode, mode)) {
		return (Multiplayer::TransferMode)mode;
	}
	return MultiplayerPeer::get_transfer_mode();
}

void MultiplayerPeerExtension::set_target_peer(int p_peer_id) {
	if (GDVIRTUAL_CALL(_set_target_peer, p_peer_id)) {
		return;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_set_target_peer is unimplemented!");
}

int MultiplayerPeerExtension::get_packet_peer() const {
	int peer;
	if (GDVIRTUAL_CALL(_get_packet_peer, peer)) {
		return peer;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_packet_peer is unimplemented!");
	return 0;
}

bool MultiplayerPeerExtension::is_server() const {
	bool server;
	if (GDVIRTUAL_CALL(_is_server, server)) {
		return server;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_is_server is unimplemented!");
	return false;
}

void MultiplayerPeerExtension::poll() {
	int err;
	if (GDVIRTUAL_CALL(_poll, err)) {
		return;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_poll is unimplemented!");
}

int MultiplayerPeerExtension::get_unique_id() const {
	int id;
	if (GDVIRTUAL_CALL(_get_unique_id, id)) {
		return id;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_unique_id is unimplemented!");
	return 0;
}

void MultiplayerPeerExtension::set_refuse_new_connections(bool p_enable) {
	if (GDVIRTUAL_CALL(_set_refuse_new_connections, p_enable)) {
		return;
	}
	MultiplayerPeer::set_refuse_new_connections(p_enable);
}

bool MultiplayerPeerExtension::is_refusing_new_connections() const {
	bool refusing;
	if (GDVIRTUAL_CALL(_is_refusing_new_connections, refusing)) {
		return refusing;
	}
	return MultiplayerPeer::is_refusing_new_connections();
}

MultiplayerPeer::ConnectionStatus MultiplayerPeerExtension::get_connection_status() const {
	int status;
	if (GDVIRTUAL_CALL(_get_connection_status, status)) {
		return (ConnectionStatus)status;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_connection_status is unimplemented!");
	return CONNECTION_DISCONNECTED;
}

void MultiplayerPeerExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_packet, "r_buffer", "r_buffer_size");
	GDVIRTUAL_BIND(_put_packet, "p_buffer", "p_buffer_size");
	GDVIRTUAL_BIND(_get_available_packet_count);
	GDVIRTUAL_BIND(_get_max_packet_size);

	GDVIRTUAL_BIND(_set_transfer_channel, "p_channel");
	GDVIRTUAL_BIND(_get_transfer_channel);

	GDVIRTUAL_BIND(_set_transfer_mode, "p_mode");
	GDVIRTUAL_BIND(_get_transfer_mode);

	GDVIRTUAL_BIND(_set_target_peer, "p_peer");

	GDVIRTUAL_BIND(_get_packet_peer);
	GDVIRTUAL_BIND(_is_server);
	GDVIRTUAL_BIND(_poll);
	GDVIRTUAL_BIND(_get_unique_id);
	GDVIRTUAL_BIND(_set_refuse_new_connections, "p_enable");
	GDVIRTUAL_BIND(_is_refusing_new_connections);
	GDVIRTUAL_BIND(_get_connection_status);
}
