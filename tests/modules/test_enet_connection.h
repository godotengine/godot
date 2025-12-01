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

#include "modules/enet/enet_connection.h"

#include "tests/test_macros.h"

namespace TestEnetConnection {
static Ref<ENetConnection> server_host;
static Ref<ENetConnection> client_host;

TEST_CASE("[enet] Connection Events") {
	server_host.instantiate();
	client_host.instantiate();

	CHECK(server_host->create_host(32, 2) == OK); // max 32 peers, 2 channels
	CHECK(client_host->create_host(1, 2) == OK); // Client host, single peer

	Ref<ENetPacketPeer> client_peer = client_host->connect_to_host("127.0.0.1", server_host->get_local_port(), 2);
	CHECK(client_peer.is_valid());

	//Service both sides to process connection
	ENetConnection::Event server_event;
	ENetConnection::Event client_event;

	const int TIMEOUT_MS = 5000; // max time to wait for event
	const int STEP_MS = 10; // service step duration

	SUBCASE("Connection Event") {
		ENetConnection::EventType type = ENetConnection::EVENT_NONE;

		for (int waited = 0; type != ENetConnection::EVENT_CONNECT && waited < TIMEOUT_MS; waited += STEP_MS) {
			type = client_host->service(STEP_MS, client_event);
		}

		CHECK(type == ENetConnection::EVENT_CONNECT);
	}

	SUBCASE("Server -> Client Packet") {
		const char *data = "Hello ENet!";
		size_t size = strlen(data) + 1;
		enet_uint32 flags = ENET_PACKET_FLAG_RELIABLE;
		ENetPacket *p_packet = enet_packet_create(data, size, flags);

		server_host->broadcast(0, p_packet);

		ENetConnection::EventType type = ENetConnection::EVENT_NONE;
		for (int waited = 0; type != ENetConnection::EVENT_RECEIVE && waited < TIMEOUT_MS; waited += STEP_MS) {
			type = client_host->service(STEP_MS, client_event);
		}
		CHECK(type == ENetConnection::EVENT_RECEIVE);
		CHECK(client_event.packet != nullptr);
	}

	SUBCASE("Client -> Server Packet") {
		PackedByteArray p_packet;
		p_packet.push_back(99);

		client_host->socket_send("127.0.0.1", server_host->get_local_port(), p_packet);

		ENetConnection::EventType type = ENetConnection::EVENT_NONE;
		for (int waited = 0; type != ENetConnection::EVENT_RECEIVE && waited < TIMEOUT_MS; waited += STEP_MS) {
			type = server_host->service(STEP_MS, server_event);
		}

		CHECK(type == ENetConnection::EVENT_RECEIVE);
		CHECK(server_event.packet != nullptr);
		CHECK(server_event.packet->data[0] == 99);
	}

	SUBCASE("Disconnect Event") {
		client_host->destroy();

		ENetConnection::EventType type = ENetConnection::EVENT_NONE;
		for (int waited = 0; type != ENetConnection::EVENT_DISCONNECT && waited < TIMEOUT_MS; waited += STEP_MS) {
			type = server_host->service(STEP_MS, server_event);
		}

		CHECK(type == ENetConnection::EVENT_DISCONNECT);
	}

	SUBCASE("Update PeerList after disconnecting") {
		bool still_connected = false;
		List<Ref<ENetPacketPeer>> peers;
		server_host->get_peers(peers);
		for (const Ref<ENetPacketPeer> &peer : peers) {
			if (peer == client_peer) {
				still_connected = true;
				break;
			}
		}
		CHECK(!still_connected);
	}

	server_host->destroy();
}

} //namespace TestEnetConnection