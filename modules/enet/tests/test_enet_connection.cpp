/**************************************************************************/
/*  test_enet_connection.cpp                                              */
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

#include "test_enet_connection.h"

#include "../enet_connection.h"
#include "../enet_packet_peer.h"

namespace TestENetConnection {

Ref<ENetConnection> create_test_connection() {
	Ref<ENetConnection> connection;
	connection.instantiate();
	return connection;
}

TEST_CASE("[ENetConnection] Host creation and destruction") {
	Ref<ENetConnection> connection = create_test_connection();
	Error err = connection->create_host();
	CHECK_EQ(err, OK);
	CHECK_GE(connection->get_max_channels(), 0);
	connection->destroy();
}

TEST_CASE("[ENetConnection] Bound host creation") {
	Ref<ENetConnection> connection = create_test_connection();
	Error err = connection->create_host_bound(IPAddress("127.0.0.1"), 12345);
	CHECK_EQ(err, OK);
	CHECK_EQ(connection->get_local_port(), 12345);
	connection->destroy();
}

TEST_CASE("[ENetConnection] Channel limiting") {
	Ref<ENetConnection> connection = create_test_connection();
	Error err = connection->create_host();
	CHECK_EQ(err, OK);

	connection->channel_limit(16);
	CHECK_EQ(connection->get_max_channels(), 16);

	connection->destroy();
}

TEST_CASE("[ENetConnection] Statistics") {
	Ref<ENetConnection> connection = create_test_connection();
	Error err = connection->create_host();
	CHECK_EQ(err, OK);

	double bytes_sent = connection->pop_statistic(ENetConnection::HOST_TOTAL_SENT_DATA);
	double bytes_received = connection->pop_statistic(ENetConnection::HOST_TOTAL_RECEIVED_DATA);

	CHECK_GE(bytes_sent, 0);
	CHECK_GE(bytes_received, 0);

	connection->destroy();
}

TEST_CASE("[ENetConnection] Service operations") {
	Ref<ENetConnection> connection = create_test_connection();
	Error err = connection->create_host_bound(IPAddress("127.0.0.1"), 0);
	CHECK_EQ(err, OK);

	connection->destroy();
}

TEST_CASE("[ENetConnection] Error handling") {
	Ref<ENetConnection> connection = create_test_connection();

	ERR_PRINT_OFF;
	Error err = connection->create_host_bound(IPAddress(), -1);
	ERR_PRINT_ON;
	CHECK_NE(err, OK);
}

} // namespace TestENetConnection
