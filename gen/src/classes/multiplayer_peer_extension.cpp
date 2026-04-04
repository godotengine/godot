/**************************************************************************/
/*  multiplayer_peer_extension.cpp                                        */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/multiplayer_peer_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Error MultiplayerPeerExtension::_get_packet(const uint8_t **r_buffer, int32_t *r_buffer_size) {
	return Error(0);
}

Error MultiplayerPeerExtension::_put_packet(const uint8_t *p_buffer, int32_t p_buffer_size) {
	return Error(0);
}

int32_t MultiplayerPeerExtension::_get_available_packet_count() const {
	return 0;
}

int32_t MultiplayerPeerExtension::_get_max_packet_size() const {
	return 0;
}

PackedByteArray MultiplayerPeerExtension::_get_packet_script() {
	return PackedByteArray();
}

Error MultiplayerPeerExtension::_put_packet_script(const PackedByteArray &p_buffer) {
	return Error(0);
}

int32_t MultiplayerPeerExtension::_get_packet_channel() const {
	return 0;
}

MultiplayerPeer::TransferMode MultiplayerPeerExtension::_get_packet_mode() const {
	return MultiplayerPeer::TransferMode(0);
}

void MultiplayerPeerExtension::_set_transfer_channel(int32_t p_channel) {}

int32_t MultiplayerPeerExtension::_get_transfer_channel() const {
	return 0;
}

void MultiplayerPeerExtension::_set_transfer_mode(MultiplayerPeer::TransferMode p_mode) {}

MultiplayerPeer::TransferMode MultiplayerPeerExtension::_get_transfer_mode() const {
	return MultiplayerPeer::TransferMode(0);
}

void MultiplayerPeerExtension::_set_target_peer(int32_t p_peer) {}

int32_t MultiplayerPeerExtension::_get_packet_peer() const {
	return 0;
}

bool MultiplayerPeerExtension::_is_server() const {
	return false;
}

void MultiplayerPeerExtension::_poll() {}

void MultiplayerPeerExtension::_close() {}

void MultiplayerPeerExtension::_disconnect_peer(int32_t p_peer, bool p_force) {}

int32_t MultiplayerPeerExtension::_get_unique_id() const {
	return 0;
}

void MultiplayerPeerExtension::_set_refuse_new_connections(bool p_enable) {}

bool MultiplayerPeerExtension::_is_refusing_new_connections() const {
	return false;
}

bool MultiplayerPeerExtension::_is_server_relay_supported() const {
	return false;
}

MultiplayerPeer::ConnectionStatus MultiplayerPeerExtension::_get_connection_status() const {
	return MultiplayerPeer::ConnectionStatus(0);
}

} // namespace godot
