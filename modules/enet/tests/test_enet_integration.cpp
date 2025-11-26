/**************************************************************************/
/*  test_enet_integration.cpp                                             */
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

#include "test_enet_integration.h"

#include "../enet_connection.h"
#include "../enet_packet_peer.h"

namespace TestENetIntegration {

TEST_CASE("[ENet Integration] Basic client-server setup") {
	Ref<ENetConnection> server;
	server.instantiate();

	Error server_err = server->create_host_bound(IPAddress("127.0.0.1"), 0);
	CHECK_EQ(server_err, OK);

	int server_port = server->get_local_port();
	CHECK_GT(server_port, 0);

	Ref<ENetConnection> client;
	client.instantiate();

	Error client_err = client->create_host();
	CHECK_EQ(client_err, OK);

	Ref<ENetPacketPeer> client_peer = client->connect_to_host("127.0.0.1", server_port, 1, 42);
	CHECK(client_peer.is_valid());
	CHECK_EQ(client_peer->get_state(), ENetPacketPeer::STATE_CONNECTING);

	for (int i = 0; i < 50; i++) {
		ENetConnection::Event server_event;
		server->service(10, server_event);

		ENetConnection::Event client_event;
		client->service(10, client_event);
	}

	server->destroy();
	client->destroy();
}

TEST_CASE("[ENet Integration] Multiple clients") {
	Ref<ENetConnection> server;
	server.instantiate();

	Error server_err = server->create_host_bound(IPAddress("127.0.0.1"), 0, 10);
	CHECK_EQ(server_err, OK);

	int server_port = server->get_local_port();
	const int client_count = 3;
	Array clients;

	for (int i = 0; i < client_count; i++) {
		Ref<ENetConnection> client;
		client.instantiate();

		Error err = client->create_host();
		CHECK_EQ(err, OK);

		Ref<ENetPacketPeer> peer = client->connect_to_host("127.0.0.1", server_port, 1, i);
		CHECK(peer.is_valid());

		clients.push_back(client);
	}

	for (int iteration = 0; iteration < 20; iteration++) {
		ENetConnection::Event server_event;
		server->service(10, server_event);

		for (int i = 0; i < client_count; i++) {
			Ref<ENetConnection> client = clients[i];
			ENetConnection::Event client_event;
			client->service(10, client_event);
		}
	}

	for (int i = 0; i < client_count; i++) {
		Ref<ENetConnection> client = clients[i];
		client->destroy();
	}
	server->destroy();
}

TEST_CASE("[ENet Integration] Compression") {
	Ref<ENetConnection> server;
	server.instantiate();

	Error server_err = server->create_host_bound(IPAddress("127.0.0.1"), 0);
	CHECK_EQ(server_err, OK);

	server->compress(ENetConnection::COMPRESS_FASTLZ);
	int server_port = server->get_local_port();

	Ref<ENetConnection> client;
	client.instantiate();

	Error client_err = client->create_host();
	CHECK_EQ(client_err, OK);

	client->compress(ENetConnection::COMPRESS_FASTLZ);

	Ref<ENetPacketPeer> client_peer = client->connect_to_host("127.0.0.1", server_port, 1, 0);
	CHECK(client_peer.is_valid());

	PackedByteArray large_data;
	large_data.resize(1024);
	for (int i = 0; i < 1024; i++) {
		large_data.set(i, i % 256);
	}

	Error send_err = client_peer->put_packet(large_data.ptr(), large_data.size());
	CHECK_EQ(send_err, OK);

	for (int i = 0; i < 20; i++) {
		ENetConnection::Event event;
		server->service(10, event);
		client->service(10, event);
	}

	server->destroy();
	client->destroy();
}

TEST_CASE("[ENet Integration] Statistics") {
	Ref<ENetConnection> server;
	server.instantiate();

	Error server_err = server->create_host_bound(IPAddress("127.0.0.1"), 0);
	CHECK_EQ(server_err, OK);

	int server_port = server->get_local_port();

	Ref<ENetConnection> client;
	client.instantiate();

	Error client_err = client->create_host();
	CHECK_EQ(client_err, OK);

	Ref<ENetPacketPeer> client_peer = client->connect_to_host("127.0.0.1", server_port, 1, 0);
	CHECK(client_peer.is_valid());

	for (int i = 0; i < 5; i++) {
		PackedByteArray data;
		data.push_back(i);
		client_peer->put_packet(data.ptr(), data.size());
	}

	for (int i = 0; i < 30; i++) {
		ENetConnection::Event event;
		server->service(10, event);
		client->service(10, event);
	}

	double server_sent = server->pop_statistic(ENetConnection::HOST_TOTAL_SENT_DATA);
	double client_received = client->pop_statistic(ENetConnection::HOST_TOTAL_RECEIVED_DATA);

	CHECK_GE(server_sent, 0);
	CHECK_GE(client_received, 0);

	server->destroy();
	client->destroy();
}

} // namespace TestENetIntegration
