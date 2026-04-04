/**************************************************************************/
/*  multiplayer_peer.hpp                                                  */
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

#pragma once

#include <godot_cpp/classes/packet_peer.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MultiplayerPeer : public PacketPeer {
	GDEXTENSION_CLASS(MultiplayerPeer, PacketPeer)

public:
	enum ConnectionStatus {
		CONNECTION_DISCONNECTED = 0,
		CONNECTION_CONNECTING = 1,
		CONNECTION_CONNECTED = 2,
	};

	enum TransferMode {
		TRANSFER_MODE_UNRELIABLE = 0,
		TRANSFER_MODE_UNRELIABLE_ORDERED = 1,
		TRANSFER_MODE_RELIABLE = 2,
	};

	static const int TARGET_PEER_BROADCAST = 0;
	static const int TARGET_PEER_SERVER = 1;

	void set_transfer_channel(int32_t p_channel);
	int32_t get_transfer_channel() const;
	void set_transfer_mode(MultiplayerPeer::TransferMode p_mode);
	MultiplayerPeer::TransferMode get_transfer_mode() const;
	void set_target_peer(int32_t p_id);
	int32_t get_packet_peer() const;
	int32_t get_packet_channel() const;
	MultiplayerPeer::TransferMode get_packet_mode() const;
	void poll();
	void close();
	void disconnect_peer(int32_t p_peer, bool p_force = false);
	MultiplayerPeer::ConnectionStatus get_connection_status() const;
	int32_t get_unique_id() const;
	uint32_t generate_unique_id() const;
	void set_refuse_new_connections(bool p_enable);
	bool is_refusing_new_connections() const;
	bool is_server_relay_supported() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PacketPeer::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(MultiplayerPeer::ConnectionStatus);
VARIANT_ENUM_CAST(MultiplayerPeer::TransferMode);

