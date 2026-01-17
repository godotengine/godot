/**************************************************************************/
/*  test_uds_server.h                                                     */
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

#include "core/io/file_access.h"
#include "core/io/stream_peer_uds.h"
#include "core/io/uds_server.h"
#include "tests/test_macros.h"

#include <functional>

namespace TestUDSServer {

#ifdef UNIX_ENABLED

const String SOCKET_PATH = "/tmp/godot_test_uds_socket";
const uint32_t SLEEP_DURATION = 1000;
const uint64_t MAX_WAIT_USEC = 2000000;

void wait_for_condition(std::function<bool()> f_test) {
	const uint64_t time = OS::get_singleton()->get_ticks_usec();
	while (!f_test() && (OS::get_singleton()->get_ticks_usec() - time) < MAX_WAIT_USEC) {
		OS::get_singleton()->delay_usec(SLEEP_DURATION);
	}
}

void cleanup_socket_file() {
	// Remove socket file if it exists
	if (FileAccess::exists(SOCKET_PATH)) {
		DirAccess::remove_absolute(SOCKET_PATH);
	}
}

Ref<UDSServer> create_server(const String &p_path) {
	cleanup_socket_file();

	Ref<UDSServer> server;
	server.instantiate();

	REQUIRE_EQ(server->listen(p_path), Error::OK);
	REQUIRE(server->is_listening());
	CHECK_FALSE(server->is_connection_available());

	return server;
}

Ref<StreamPeerUDS> create_client(const String &p_path) {
	Ref<StreamPeerUDS> client;
	client.instantiate();

	Error err = client->connect_to_host(p_path);
	REQUIRE_EQ(err, Error::OK);

	// UDS connections may be immediately connected or in connecting state
	StreamPeerUDS::Status status = client->get_status();
	REQUIRE((status == StreamPeerUDS::STATUS_CONNECTED || status == StreamPeerUDS::STATUS_CONNECTING));

	if (status == StreamPeerUDS::STATUS_CONNECTED) {
		CHECK_EQ(client->get_connected_path(), p_path);
	}

	return client;
}

Ref<StreamPeerUDS> accept_connection(Ref<UDSServer> &p_server) {
	wait_for_condition([&]() {
		return p_server->is_connection_available();
	});

	REQUIRE(p_server->is_connection_available());
	Ref<StreamPeerUDS> client_from_server = p_server->take_connection();
	REQUIRE(client_from_server.is_valid());
	CHECK_EQ(client_from_server->get_status(), StreamPeerUDS::STATUS_CONNECTED);

	return client_from_server;
}

TEST_CASE("[UDSServer] Instantiation") {
	Ref<UDSServer> server;
	server.instantiate();

	REQUIRE(server.is_valid());
	CHECK_FALSE(server->is_listening());
}

TEST_CASE("[UDSServer] Accept a connection and receive/send data") {
	Ref<UDSServer> server = create_server(SOCKET_PATH);
	Ref<StreamPeerUDS> client = create_client(SOCKET_PATH);
	Ref<StreamPeerUDS> client_from_server = accept_connection(server);

	wait_for_condition([&]() {
		return client->poll() != Error::OK || client->get_status() == StreamPeerUDS::STATUS_CONNECTED;
	});

	CHECK_EQ(client->get_status(), StreamPeerUDS::STATUS_CONNECTED);

	// Sending data from client to server.
	const String hello_world = "Hello World!";
	client->put_string(hello_world);
	CHECK_EQ(client_from_server->get_string(), hello_world);

	// Sending data from server to client.
	const float pi = 3.1415;
	client_from_server->put_float(pi);
	CHECK_EQ(client->get_float(), pi);

	client->disconnect_from_host();
	server->stop();
	CHECK_FALSE(server->is_listening());

	cleanup_socket_file();
}

TEST_CASE("[UDSServer] Handle multiple clients at the same time") {
	Ref<UDSServer> server = create_server(SOCKET_PATH);

	Vector<Ref<StreamPeerUDS>> clients;
	for (int i = 0; i < 5; i++) {
		clients.push_back(create_client(SOCKET_PATH));
	}

	Vector<Ref<StreamPeerUDS>> clients_from_server;
	for (int i = 0; i < clients.size(); i++) {
		clients_from_server.push_back(accept_connection(server));
	}

	wait_for_condition([&]() {
		bool should_exit = true;
		for (Ref<StreamPeerUDS> &c : clients) {
			if (c->poll() != Error::OK) {
				return true;
			}
			StreamPeerUDS::Status status = c->get_status();
			if (status != StreamPeerUDS::STATUS_CONNECTED && status != StreamPeerUDS::STATUS_CONNECTING) {
				return true;
			}
			if (status != StreamPeerUDS::STATUS_CONNECTED) {
				should_exit = false;
			}
		}
		return should_exit;
	});

	for (Ref<StreamPeerUDS> &c : clients) {
		REQUIRE_EQ(c->get_status(), StreamPeerUDS::STATUS_CONNECTED);
	}

	// Sending data from each client to server.
	for (int i = 0; i < clients.size(); i++) {
		String hello_client = "Hello " + itos(i);
		clients[i]->put_string(hello_client);
		CHECK_EQ(clients_from_server[i]->get_string(), hello_client);
	}

	for (Ref<StreamPeerUDS> &c : clients) {
		c->disconnect_from_host();
	}
	server->stop();

	cleanup_socket_file();
}

TEST_CASE("[UDSServer] When stopped shouldn't accept new connections") {
	Ref<UDSServer> server = create_server(SOCKET_PATH);
	Ref<StreamPeerUDS> client = create_client(SOCKET_PATH);
	Ref<StreamPeerUDS> client_from_server = accept_connection(server);

	wait_for_condition([&]() {
		return client->poll() != Error::OK || client->get_status() == StreamPeerUDS::STATUS_CONNECTED;
	});

	CHECK_EQ(client->get_status(), StreamPeerUDS::STATUS_CONNECTED);

	// Sending data from client to server.
	const String hello_world = "Hello World!";
	client->put_string(hello_world);
	CHECK_EQ(client_from_server->get_string(), hello_world);

	client->disconnect_from_host();
	server->stop();
	CHECK_FALSE(server->is_listening());

	// Clean up the socket file after server stops
	cleanup_socket_file();

	// Try to connect to non-existent socket
	Ref<StreamPeerUDS> new_client;
	new_client.instantiate();
	Error err = new_client->connect_to_host(SOCKET_PATH);

	// Connection should fail since socket doesn't exist
	CHECK_NE(err, Error::OK);
	CHECK_FALSE(server->is_connection_available());

	cleanup_socket_file();
}

TEST_CASE("[UDSServer] Should disconnect client") {
	Ref<UDSServer> server = create_server(SOCKET_PATH);
	Ref<StreamPeerUDS> client = create_client(SOCKET_PATH);
	Ref<StreamPeerUDS> client_from_server = accept_connection(server);

	wait_for_condition([&]() {
		return client->poll() != Error::OK || client->get_status() == StreamPeerUDS::STATUS_CONNECTED;
	});

	CHECK_EQ(client->get_status(), StreamPeerUDS::STATUS_CONNECTED);

	// Sending data from client to server.
	const String hello_world = "Hello World!";
	client->put_string(hello_world);
	CHECK_EQ(client_from_server->get_string(), hello_world);

	client_from_server->disconnect_from_host();
	server->stop();
	CHECK_FALSE(server->is_listening());

	// Wait for disconnection
	wait_for_condition([&]() {
		return client->poll() != Error::OK || client->get_status() == StreamPeerUDS::STATUS_NONE;
	});

	// Wait for disconnection
	wait_for_condition([&]() {
		return client_from_server->poll() != Error::OK || client_from_server->get_status() == StreamPeerUDS::STATUS_NONE;
	});

	CHECK_EQ(client->get_status(), StreamPeerUDS::STATUS_NONE);
	CHECK_EQ(client_from_server->get_status(), StreamPeerUDS::STATUS_NONE);

	ERR_PRINT_OFF;
	CHECK_EQ(client->get_string(), String());
	CHECK_EQ(client_from_server->get_string(), String());
	ERR_PRINT_ON;

	cleanup_socket_file();
}

TEST_CASE("[UDSServer] Test with different socket paths") {
	// Test with a different socket path
	const String alt_socket_path = "/tmp/godot_test_uds_socket_alt";

	// Clean up before test
	if (FileAccess::exists(alt_socket_path)) {
		DirAccess::remove_absolute(alt_socket_path);
	}

	Ref<UDSServer> server = create_server(alt_socket_path);
	Ref<StreamPeerUDS> client = create_client(alt_socket_path);
	Ref<StreamPeerUDS> client_from_server = accept_connection(server);

	wait_for_condition([&]() {
		return client->poll() != Error::OK || client->get_status() == StreamPeerUDS::STATUS_CONNECTED;
	});

	CHECK_EQ(client->get_status(), StreamPeerUDS::STATUS_CONNECTED);

	// Test data exchange
	const int test_number = 42;
	client->put_32(test_number);
	CHECK_EQ(client_from_server->get_32(), test_number);

	client->disconnect_from_host();
	server->stop();

	// Clean up
	if (FileAccess::exists(alt_socket_path)) {
		DirAccess::remove_absolute(alt_socket_path);
	}
}

#endif

} // namespace TestUDSServer
