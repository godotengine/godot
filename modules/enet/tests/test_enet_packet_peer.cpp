/**************************************************************************/
/*  test_enet_packet_peer.cpp                                             */
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

#include "test_enet_packet_peer.h"

#include "../enet_connection.h"
#include "../enet_packet_peer.h"

namespace TestENetPacketPeer {

Ref<ENetPacketPeer> create_test_peer() {
	Ref<ENetConnection> connection;
	connection.instantiate();

	Error err = connection->create_host();
	if (err != OK) {
		return Ref<ENetPacketPeer>();
	}

	return connection->connect_to_host("127.0.0.1", 54321, 1, 0);
}

TEST_CASE("[ENetPacketPeer] Basic peer properties") {
	Ref<ENetPacketPeer> peer = create_test_peer();

	CHECK(peer->is_active());
	CHECK_EQ(peer->get_state(), ENetPacketPeer::STATE_CONNECTING);
	CHECK_GT(peer->get_max_packet_size(), 0);
	CHECK_EQ(peer->get_available_packet_count(), 0);
	CHECK_GT(peer->get_channels(), 0);
	CHECK_EQ(peer->get_remote_port(), 54321);
}

TEST_CASE("[ENetPacketPeer] Packet operations") {
	Ref<ENetPacketPeer> peer = create_test_peer();

	PackedByteArray data;
	data.push_back(1);
	data.push_back(2);
	data.push_back(3);

	Error err = peer->put_packet(data.ptr(), data.size());
	CHECK_EQ(err, OK);

	const uint8_t *buffer;
	int buffer_size;
	ERR_PRINT_OFF;
	err = peer->get_packet(&buffer, buffer_size);
	ERR_PRINT_ON;
	CHECK_NE(err, OK);
}

TEST_CASE("[ENetPacketPeer] Statistics") {
	Ref<ENetPacketPeer> peer = create_test_peer();

	double rtt = peer->get_statistic(ENetPacketPeer::PEER_ROUND_TRIP_TIME);
	double packet_loss = peer->get_statistic(ENetPacketPeer::PEER_PACKET_LOSS);

	CHECK_GE(rtt, 0);
	CHECK_GE(packet_loss, 0);
}

TEST_CASE("[ENetPacketPeer] Control operations") {
	Ref<ENetPacketPeer> peer = create_test_peer();

	peer->ping();
	peer->throttle_configure(32, 2, 4);
	peer->set_timeout(5000, 1000, 3000);
	peer->peer_disconnect_now(0);
}

} // namespace TestENetPacketPeer
