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
TEST_CASE("[ENet] Connection Events") {
	Ref<ENetConnection> server_connection;
	server_connection.instantiate();
	CHECK_EQ((server_connection->create_host_bound(IPAddress("127.0.0.1"), 7777, 32, 1, 0, 0)), OK); //32 peers 1 channel

	Ref<ENetConnection> client_connection;
	client_connection.instantiate();
	CHECK_EQ(client_connection->create_host(1, 1, 0, 0), OK);

	client_connection->connect_to_host("127.0.0.1", 7777, 1, 0); //Single channel
	ENetConnection::Event client_event;
	ENetConnection::Event server_event;

	//Connection setup
	bool server_check = false, client_check = false;
	for (int i = 0; i < 5; i++) {
		if (client_connection->service(500, client_event) == ENetConnection::EVENT_CONNECT) {
			client_check = true;
			client_connection->flush();
		}
		if (server_connection->service(500, server_event) == ENetConnection::EVENT_CONNECT) {
			server_check = true;
			break;
		}
	}
	CHECK(client_check);
	CHECK(server_check);

	client_connection->destroy();
	server_connection->destroy();
}
} //namespace TestEnetConnection
