/**************************************************************************/
/*  e_net_packet_peer.hpp                                                 */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/packet_peer.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedByteArray;

class ENetPacketPeer : public PacketPeer {
	GDEXTENSION_CLASS(ENetPacketPeer, PacketPeer)

public:
	enum PeerState {
		STATE_DISCONNECTED = 0,
		STATE_CONNECTING = 1,
		STATE_ACKNOWLEDGING_CONNECT = 2,
		STATE_CONNECTION_PENDING = 3,
		STATE_CONNECTION_SUCCEEDED = 4,
		STATE_CONNECTED = 5,
		STATE_DISCONNECT_LATER = 6,
		STATE_DISCONNECTING = 7,
		STATE_ACKNOWLEDGING_DISCONNECT = 8,
		STATE_ZOMBIE = 9,
	};

	enum PeerStatistic {
		PEER_PACKET_LOSS = 0,
		PEER_PACKET_LOSS_VARIANCE = 1,
		PEER_PACKET_LOSS_EPOCH = 2,
		PEER_ROUND_TRIP_TIME = 3,
		PEER_ROUND_TRIP_TIME_VARIANCE = 4,
		PEER_LAST_ROUND_TRIP_TIME = 5,
		PEER_LAST_ROUND_TRIP_TIME_VARIANCE = 6,
		PEER_PACKET_THROTTLE = 7,
		PEER_PACKET_THROTTLE_LIMIT = 8,
		PEER_PACKET_THROTTLE_COUNTER = 9,
		PEER_PACKET_THROTTLE_EPOCH = 10,
		PEER_PACKET_THROTTLE_ACCELERATION = 11,
		PEER_PACKET_THROTTLE_DECELERATION = 12,
		PEER_PACKET_THROTTLE_INTERVAL = 13,
	};

	static const int PACKET_LOSS_SCALE = 65536;
	static const int PACKET_THROTTLE_SCALE = 32;
	static const int FLAG_RELIABLE = 1;
	static const int FLAG_UNSEQUENCED = 2;
	static const int FLAG_UNRELIABLE_FRAGMENT = 8;

	void peer_disconnect(int32_t p_data = 0);
	void peer_disconnect_later(int32_t p_data = 0);
	void peer_disconnect_now(int32_t p_data = 0);
	void ping();
	void ping_interval(int32_t p_ping_interval);
	void reset();
	Error send(int32_t p_channel, const PackedByteArray &p_packet, int32_t p_flags);
	void throttle_configure(int32_t p_interval, int32_t p_acceleration, int32_t p_deceleration);
	void set_timeout(int32_t p_timeout, int32_t p_timeout_min, int32_t p_timeout_max);
	int32_t get_packet_flags() const;
	String get_remote_address() const;
	int32_t get_remote_port() const;
	double get_statistic(ENetPacketPeer::PeerStatistic p_statistic);
	ENetPacketPeer::PeerState get_state() const;
	int32_t get_channels() const;
	bool is_active() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PacketPeer::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ENetPacketPeer::PeerState);
VARIANT_ENUM_CAST(ENetPacketPeer::PeerStatistic);

