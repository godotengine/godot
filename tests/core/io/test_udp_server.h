/**************************************************************************/
/*  test_udp_server.h                                                     */
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

#ifndef TEST_UDP_SERVER_H
#define TEST_UDP_SERVER_H

#include "core/io/packet_peer_udp.h"
#include "core/io/udp_server.h"
#include "tests/test_macros.h"

namespace TestUDPServer {

const int port = 12345;
const IPAddress localhost("127.0.0.1");
const uint32_t sleep_duration = 1000;
const uint64_t max_wait_usec = 100000;

Ref<UDPServer> create_server(const IPAddress &p_address, int p_port) {
	Ref<UDPServer> server;
	server.instantiate();

	Error err = server->listen(port, localhost);
	REQUIRE_EQ(Error::OK, err);
	REQUIRE(server->is_listening());
	CHECK_FALSE(server->is_connection_available());
	CHECK_EQ(server->get_max_pending_connections(), 16);

	return server;
}

Ref<PacketPeerUDP> create_client(const IPAddress &p_address, int p_port) {
	Ref<PacketPeerUDP> client;
	client.instantiate();

	Error err = client->connect_to_host(localhost, port);
	REQUIRE_EQ(Error::OK, err);
	CHECK(client->is_bound());
	CHECK(client->is_socket_connected());

	return client;
}

Ref<PacketPeerUDP> accept_connection(Ref<UDPServer> &server) {
	// Required to get the connection properly established.
	uint64_t time = OS::get_singleton()->get_ticks_usec();
	while (!server->is_connection_available() && (OS::get_singleton()->get_ticks_usec() - time) < max_wait_usec) {
		server->poll();
		OS::get_singleton()->delay_usec(sleep_duration);
	}

	CHECK_EQ(server->poll(), Error::OK);
	REQUIRE_MESSAGE(server->is_connection_available(), "No connection available");
	Ref<PacketPeerUDP> client_from_server = server->take_connection();
	REQUIRE_MESSAGE(client_from_server.is_valid(), "A Packet Peer UDP from the UDP Server should not be a null pointer.");
	CHECK(client_from_server->is_bound());
	CHECK(client_from_server->is_socket_connected());

	return client_from_server;
}

TEST_CASE("[UDPServer] Instantiation") {
	Ref<UDPServer> server;
	server.instantiate();

	REQUIRE_MESSAGE(server.is_valid(), "A UDP Server created should not be a null pointer.");
	CHECK_EQ(false, server->is_listening());
}

TEST_CASE("[UDPServer] Accept a connection and receive/send data") {
	Ref<UDPServer> server = create_server(localhost, port);
	Ref<PacketPeerUDP> client = create_client(localhost, port);

	// Sending data from client to server.
	String hello_world = "Hello World!";
	CHECK_EQ(client->put_var(hello_world), Error::OK);

	Variant hello_world_received;
	Ref<PacketPeerUDP> client_from_server = accept_connection(server);
	CHECK_EQ(client_from_server->get_var(hello_world_received), Error::OK);
	CHECK_EQ(String(hello_world_received), hello_world);

	// Sending data from server to client.
	Variant pi = 3.1415;
	CHECK_EQ(client_from_server->put_var(pi), Error::OK);

	uint64_t time = OS::get_singleton()->get_ticks_usec();
	// get_available_packet_count() is the recommended way to call _poll(), because there is no public poll().
	while (client->get_available_packet_count() == 0 && (OS::get_singleton()->get_ticks_usec() - time) < max_wait_usec) {
		server->poll();
		OS::get_singleton()->delay_usec(sleep_duration);
	}

	CHECK_EQ(server->poll(), Error::OK);
	CHECK_GT(client->get_available_packet_count(), 0);

	Variant pi_received;
	CHECK_EQ(client->get_var(pi_received), Error::OK);
	CHECK_EQ(pi_received, pi);

	client->close();
	server->stop();
	CHECK_FALSE(server->is_listening());
}

TEST_CASE("[UDPServer] Handle multiple clients at the same time") {
	Ref<UDPServer> server = create_server(localhost, port);

	Vector<Ref<PacketPeerUDP>> clients;
	for (int i = 0; i < 5; i++) {
		Ref<PacketPeerUDP> c = create_client(localhost, port);

		// Sending data from client to server.
		String hello_client = "Hello " + itos(i);
		CHECK_EQ(c->put_var(hello_client), Error::OK);

		clients.push_back(c);
	}

	for (int i = 0; i < clients.size(); i++) {
		Ref<PacketPeerUDP> cfs = accept_connection(server);

		Variant hello_world_received;
		CHECK_EQ(cfs->get_var(hello_world_received), Error::OK);
		CHECK_EQ(String(hello_world_received), "Hello " + itos(i));

		// Sending data from server to client.
		Variant pi = 3.1415 + i;
		CHECK_EQ(cfs->put_var(pi), Error::OK);
	}

	uint64_t time = OS::get_singleton()->get_ticks_usec();
	for (int i = 0; i < clients.size(); i++) {
		Ref<PacketPeerUDP> c = clients[i];
		// get_available_packet_count() is the recommended way to call _poll(), because there is no public poll().
		// Because `time` is defined outside the for, we will wait the max amount of time only for the first client.
		while (c->get_available_packet_count() == 0 && (OS::get_singleton()->get_ticks_usec() - time) < max_wait_usec) {
			server->poll();
			OS::get_singleton()->delay_usec(sleep_duration);
		}

		// The recommended way to call _poll(), because there is no public poll().
		CHECK_GT(c->get_available_packet_count(), 0);

		Variant pi_received;
		Variant pi = 3.1415 + i;
		CHECK_EQ(c->get_var(pi_received), Error::OK);
		CHECK_EQ(pi_received, pi);
	}

	for (Ref<PacketPeerUDP> &c : clients) {
		c->close();
	}
	server->stop();
}

TEST_CASE("[UDPServer] When stopped shouldn't accept new connections") {
	Ref<UDPServer> server = create_server(localhost, port);
	Ref<PacketPeerUDP> client = create_client(localhost, port);

	// Sending data from client to server.
	String hello_world = "Hello World!";
	CHECK_EQ(client->put_var(hello_world), Error::OK);

	Variant hello_world_received;
	Ref<PacketPeerUDP> client_from_server = accept_connection(server);
	CHECK_EQ(client_from_server->get_var(hello_world_received), Error::OK);
	CHECK_EQ(String(hello_world_received), hello_world);

	client->close();
	server->stop();
	CHECK_FALSE(server->is_listening());

	Ref<PacketPeerUDP> new_client = create_client(localhost, port);
	CHECK_EQ(new_client->put_var(hello_world), Error::OK);

	Error err = server->poll();
	REQUIRE_EQ(Error::ERR_UNCONFIGURED, err);
	CHECK_FALSE(server->is_connection_available());

	uint64_t time = OS::get_singleton()->get_ticks_usec();
	// get_available_packet_count() is the recommended way to call _poll(), because there is no public poll().
	while (new_client->get_available_packet_count() == 0 && (OS::get_singleton()->get_ticks_usec() - time) < max_wait_usec) {
		OS::get_singleton()->delay_usec(sleep_duration);
	}

	int packet_count = new_client->get_available_packet_count();
	CHECK_MESSAGE((packet_count == 0 || packet_count == -1), "Packet count should be 0 or -1.");
}

TEST_CASE("[UDPServer] Should disconnect client") {
	Ref<UDPServer> server = create_server(localhost, port);
	Ref<PacketPeerUDP> client = create_client(localhost, port);

	// Sending data from client to server.
	String hello_world = "Hello World!";
	CHECK_EQ(client->put_var(hello_world), Error::OK);

	Variant hello_world_received;
	Ref<PacketPeerUDP> client_from_server = accept_connection(server);
	CHECK_EQ(client_from_server->get_var(hello_world_received), Error::OK);
	CHECK_EQ(String(hello_world_received), hello_world);

	server->stop();
	CHECK_FALSE(server->is_listening());
	CHECK_FALSE(client_from_server->is_bound());
	CHECK_FALSE(client_from_server->is_socket_connected());

	// Sending data from client to server.
	CHECK_EQ(client->put_var(hello_world), Error::OK);

	uint64_t time = OS::get_singleton()->get_ticks_usec();
	// get_available_packet_count() is the recommended way to call _poll(), because there is no public poll().
	while (client->get_available_packet_count() == 0 && (OS::get_singleton()->get_ticks_usec() - time) < max_wait_usec) {
		OS::get_singleton()->delay_usec(sleep_duration);
	}

	int packet_count = client->get_available_packet_count();
	CHECK_MESSAGE((packet_count == 0 || packet_count == -1), "Packet count should be 0 or -1.");

	client->close();
}

TEST_CASE("[UDPServer] Should drop new connections when pending max connection is reached") {
	Ref<UDPServer> server = create_server(localhost, port);
	server->set_max_pending_connections(3);

	Vector<Ref<PacketPeerUDP>> clients;
	for (int i = 0; i < 5; i++) {
		Ref<PacketPeerUDP> c = create_client(localhost, port);

		// Sending data from client to server.
		String hello_client = "Hello " + itos(i);
		CHECK_EQ(c->put_var(hello_client), Error::OK);

		clients.push_back(c);
	}

	for (int i = 0; i < server->get_max_pending_connections(); i++) {
		Ref<PacketPeerUDP> cfs = accept_connection(server);

		Variant hello_world_received;
		CHECK_EQ(cfs->get_var(hello_world_received), Error::OK);
		CHECK_EQ(String(hello_world_received), "Hello " + itos(i));
	}

	CHECK_EQ(server->poll(), Error::OK);

	REQUIRE_FALSE(server->is_connection_available());
	Ref<PacketPeerUDP> client_from_server = server->take_connection();
	REQUIRE_FALSE_MESSAGE(client_from_server.is_valid(), "A Packet Peer UDP from the UDP Server should be a null pointer because the pending connection was dropped.");

	server->stop();
}

} // namespace TestUDPServer

#endif // TEST_UDP_SERVER_H
