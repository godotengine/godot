/**************************************************************************/
/*  multiplayer_peer.cpp                                                  */
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

#include "multiplayer_peer.h"

#include "core/os/os.h"

uint32_t MultiplayerPeer::generate_unique_id() const {
	uint32_t hash = 0;

	while (hash == 0 || hash == 1) {
		hash = hash_murmur3_one_32(
				(uint32_t)OS::get_singleton()->get_ticks_usec());
		hash = hash_murmur3_one_32(
				(uint32_t)OS::get_singleton()->get_unix_time(), hash);
		hash = hash_murmur3_one_32(
				(uint32_t)OS::get_singleton()->get_user_data_dir().hash64(), hash);
		hash = hash_murmur3_one_32(
				(uint32_t)((uint64_t)this), hash); // Rely on ASLR heap
		hash = hash_murmur3_one_32(
				(uint32_t)((uint64_t)&hash), hash); // Rely on ASLR stack

		hash = hash_fmix32(hash);
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

void MultiplayerPeer::set_transfer_mode(TransferMode p_mode) {
	transfer_mode = p_mode;
}

MultiplayerPeer::TransferMode MultiplayerPeer::get_transfer_mode() const {
	return transfer_mode;
}

void MultiplayerPeer::set_refuse_new_connections(bool p_enable) {
	refuse_connections = p_enable;
}

bool MultiplayerPeer::is_refusing_new_connections() const {
	return refuse_connections;
}

bool MultiplayerPeer::is_server_relay_supported() const {
	return false;
}

void MultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transfer_channel", "channel"), &MultiplayerPeer::set_transfer_channel);
	ClassDB::bind_method(D_METHOD("get_transfer_channel"), &MultiplayerPeer::get_transfer_channel);
	ClassDB::bind_method(D_METHOD("set_transfer_mode", "mode"), &MultiplayerPeer::set_transfer_mode);
	ClassDB::bind_method(D_METHOD("get_transfer_mode"), &MultiplayerPeer::get_transfer_mode);
	ClassDB::bind_method(D_METHOD("set_target_peer", "id"), &MultiplayerPeer::set_target_peer);

	ClassDB::bind_method(D_METHOD("get_packet_peer"), &MultiplayerPeer::get_packet_peer);
	ClassDB::bind_method(D_METHOD("get_packet_channel"), &MultiplayerPeer::get_packet_channel);
	ClassDB::bind_method(D_METHOD("get_packet_mode"), &MultiplayerPeer::get_packet_mode);

	ClassDB::bind_method(D_METHOD("poll"), &MultiplayerPeer::poll);
	ClassDB::bind_method(D_METHOD("close"), &MultiplayerPeer::close);
	ClassDB::bind_method(D_METHOD("disconnect_peer", "peer", "force"), &MultiplayerPeer::disconnect_peer, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_connection_status"), &MultiplayerPeer::get_connection_status);
	ClassDB::bind_method(D_METHOD("get_unique_id"), &MultiplayerPeer::get_unique_id);
	ClassDB::bind_method(D_METHOD("generate_unique_id"), &MultiplayerPeer::generate_unique_id);

	ClassDB::bind_method(D_METHOD("set_refuse_new_connections", "enable"), &MultiplayerPeer::set_refuse_new_connections);
	ClassDB::bind_method(D_METHOD("is_refusing_new_connections"), &MultiplayerPeer::is_refusing_new_connections);

	ClassDB::bind_method(D_METHOD("is_server_relay_supported"), &MultiplayerPeer::is_server_relay_supported);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_connections"), "set_refuse_new_connections", "is_refusing_new_connections");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transfer_mode", PROPERTY_HINT_ENUM, "Unreliable,Unreliable Ordered,Reliable"), "set_transfer_mode", "get_transfer_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transfer_channel", PROPERTY_HINT_RANGE, "0,255,1"), "set_transfer_channel", "get_transfer_channel");

	BIND_ENUM_CONSTANT(CONNECTION_DISCONNECTED);
	BIND_ENUM_CONSTANT(CONNECTION_CONNECTING);
	BIND_ENUM_CONSTANT(CONNECTION_CONNECTED);

	BIND_CONSTANT(TARGET_PEER_BROADCAST);
	BIND_CONSTANT(TARGET_PEER_SERVER);

	BIND_ENUM_CONSTANT(TRANSFER_MODE_UNRELIABLE);
	BIND_ENUM_CONSTANT(TRANSFER_MODE_UNRELIABLE_ORDERED);
	BIND_ENUM_CONSTANT(TRANSFER_MODE_RELIABLE);

	ADD_SIGNAL(MethodInfo("peer_connected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_disconnected", PropertyInfo(Variant::INT, "id")));
}

/*************/

Error MultiplayerPeerExtension::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	Error err;
	if (GDVIRTUAL_CALL(_get_packet, r_buffer, &r_buffer_size, err)) {
		return err;
	}
	if (GDVIRTUAL_IS_OVERRIDDEN(_get_packet_script)) {
		if (!GDVIRTUAL_CALL(_get_packet_script, script_buffer)) {
			return FAILED;
		}

		if (script_buffer.size() == 0) {
			return Error::ERR_UNAVAILABLE;
		}

		*r_buffer = script_buffer.ptr();
		r_buffer_size = script_buffer.size();

		return Error::OK;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_get_packet_native is unimplemented!");
	return FAILED;
}

Error MultiplayerPeerExtension::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	Error err;
	if (GDVIRTUAL_CALL(_put_packet, p_buffer, p_buffer_size, err)) {
		return err;
	}
	if (GDVIRTUAL_IS_OVERRIDDEN(_put_packet_script)) {
		PackedByteArray a;
		a.resize(p_buffer_size);
		memcpy(a.ptrw(), p_buffer, p_buffer_size);

		if (!GDVIRTUAL_CALL(_put_packet_script, a, err)) {
			return FAILED;
		}
		return err;
	}
	WARN_PRINT_ONCE("MultiplayerPeerExtension::_put_packet_native is unimplemented!");
	return FAILED;
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

bool MultiplayerPeerExtension::is_server_relay_supported() const {
	bool can_relay;
	if (GDVIRTUAL_CALL(_is_server_relay_supported, can_relay)) {
		return can_relay;
	}
	return MultiplayerPeer::is_server_relay_supported();
}

void MultiplayerPeerExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_packet, "r_buffer", "r_buffer_size");
	GDVIRTUAL_BIND(_put_packet, "p_buffer", "p_buffer_size");
	GDVIRTUAL_BIND(_get_available_packet_count);
	GDVIRTUAL_BIND(_get_max_packet_size);

	GDVIRTUAL_BIND(_get_packet_script)
	GDVIRTUAL_BIND(_put_packet_script, "p_buffer");

	GDVIRTUAL_BIND(_get_packet_channel);
	GDVIRTUAL_BIND(_get_packet_mode);

	GDVIRTUAL_BIND(_set_transfer_channel, "p_channel");
	GDVIRTUAL_BIND(_get_transfer_channel);

	GDVIRTUAL_BIND(_set_transfer_mode, "p_mode");
	GDVIRTUAL_BIND(_get_transfer_mode);

	GDVIRTUAL_BIND(_set_target_peer, "p_peer");

	GDVIRTUAL_BIND(_get_packet_peer);
	GDVIRTUAL_BIND(_is_server);
	GDVIRTUAL_BIND(_poll);
	GDVIRTUAL_BIND(_close);
	GDVIRTUAL_BIND(_disconnect_peer, "p_peer", "p_force");
	GDVIRTUAL_BIND(_get_unique_id);
	GDVIRTUAL_BIND(_set_refuse_new_connections, "p_enable");
	GDVIRTUAL_BIND(_is_refusing_new_connections);
	GDVIRTUAL_BIND(_is_server_relay_supported);
	GDVIRTUAL_BIND(_get_connection_status);

	ADD_PROPERTY_DEFAULT("transfer_mode", TRANSFER_MODE_RELIABLE);
	ADD_PROPERTY_DEFAULT("transfer_channel", 0);
}
