/**************************************************************************/
/*  test_enet_connection.h                                                */
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

#pragma once

#include "core/os/os.h"
#include "modules/enet/enet_connection.h"

#include "tests/test_macros.h"

namespace TestEnetConnection {
TEST_CASE("[enet] Connection Events") {
	Ref<ENetConnection> server_host;
	Ref<ENetConnection> client_host;
	server_host.instantiate();
	client_host.instantiate();
	CHECK_EQ(server_host->create_host(32, 2, 0), OK); // max 32 peers, 2 channels
	CHECK_EQ(client_host->create_host(1, 2), OK); // Client host, single peer

	ENetConnection::Event server_event;
	ENetConnection::Event client_event;

	// ENet may fail to bind in CI environments; skip instead of failing.
	if (server_host->get_local_port() == 0) {
		print_line("Skipping ENet test: server failed to bind.");
		server_host->destroy();
		return;
	}

	// Attempt client connection
	Ref<ENetPacketPeer> client_peer;
	client_peer = client_host->connect_to_host("127.0.0.1", server_host->get_local_port(), 2);

	const int TIMEOUT_MS = 5000; // max time to wait for event
	const int STEP_MS = 10; // service step duration

	// Connection event
	ENetConnection::EventType type = ENetConnection::EVENT_NONE;
	for (int waited = 0; type != ENetConnection::EVENT_CONNECT && waited < TIMEOUT_MS; waited += STEP_MS) {
		server_host->service(STEP_MS, server_event);
		type = client_host->service(STEP_MS, client_event);
	}
	CHECK_EQ(type, ENetConnection::EVENT_CONNECT);

	// Server -> Client packet via broadcast
	const char *data = "Hello ENet!";
	ENetPacket *p_packet = enet_packet_create(data, strlen(data) + 1, ENET_PACKET_FLAG_RELIABLE);
	server_host->broadcast(0, p_packet);
	type = ENetConnection::EVENT_NONE;
	for (int waited = 0; type != ENetConnection::EVENT_RECEIVE && waited < TIMEOUT_MS; waited += STEP_MS) {
		server_host->service(STEP_MS, server_event);
		type = client_host->service(STEP_MS, client_event);
	}
	CHECK_EQ(type, ENetConnection::EVENT_RECEIVE);
	REQUIRE(client_event.packet != nullptr);
	enet_packet_destroy(client_event.packet);

	// Client -> Server packet via ENet peer
	PackedByteArray payload;
	payload.push_back(99);
	ENetPacket *cli_pkt = enet_packet_create(payload.ptr(), payload.size(), ENET_PACKET_FLAG_RELIABLE);
	client_peer->send(0, cli_pkt);
	type = ENetConnection::EVENT_NONE;
	for (int waited = 0; type != ENetConnection::EVENT_RECEIVE && waited < TIMEOUT_MS; waited += STEP_MS) {
		client_host->service(STEP_MS, client_event);
		type = server_host->service(STEP_MS, server_event);
	}
	CHECK_EQ(type, ENetConnection::EVENT_RECEIVE);
	REQUIRE(server_event.packet != nullptr);
	CHECK_EQ(server_event.packet->data[0], 99);
	enet_packet_destroy(server_event.packet);

	// Disconnect
	client_host->destroy();
	type = ENetConnection::EVENT_NONE;
	for (int waited = 0; type != ENetConnection::EVENT_DISCONNECT && waited < TIMEOUT_MS; waited += STEP_MS) {
		type = server_host->service(STEP_MS, server_event);
	}
	CHECK_EQ(type, ENetConnection::EVENT_DISCONNECT);

	server_host->destroy();
}
} //namespace TestEnetConnection
