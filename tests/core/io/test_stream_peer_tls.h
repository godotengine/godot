/**************************************************************************/
/*  test_stream_peer_tls.h                                                */
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

#include "core/io/stream_peer_tcp.h"
#include "core/io/stream_peer_tls.h"
#include "tests/test_macros.h"

namespace TestStreamPeerTLS {

TEST_CASE("[StreamPeerTLS] Availability and creation") {
	// Factory should report available
	CHECK(StreamPeerTLS::is_available() == true);

	// Create a backend instance using Ref
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();

	// Check that we got a valid object
	CHECK(tls.is_valid());
	CHECK_MESSAGE(tls.is_valid(), "TLS backend should be registered and create() should return a valid object.");
}

TEST_CASE("[StreamPeerTLS] Enum values") {
	CHECK(StreamPeerTLS::STATUS_DISCONNECTED == 0);
	CHECK(StreamPeerTLS::STATUS_HANDSHAKING == 1);
	CHECK(StreamPeerTLS::STATUS_CONNECTED == 2);
	CHECK(StreamPeerTLS::STATUS_ERROR == 3);
	CHECK(StreamPeerTLS::STATUS_ERROR_HOSTNAME_MISMATCH == 4);
}

TEST_CASE("[StreamPeerTLS] Connect with unconnected TCP stream") {
	// Create an unconnected TCP stream peer
	Ref<StreamPeerTCP> tcp = memnew(StreamPeerTCP);
	CHECK(tcp.is_valid());
	CHECK(tcp->get_status() == StreamPeerTCP::STATUS_NONE);

	// Create a TLS stream peer
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();
	CHECK(tls.is_valid());

	// Attempt to connect with an unconnected TCP stream
	Error err = tls->connect_to_stream(tcp, "localhost", Ref<TLSOptions>());
	CHECK(err != OK); // Should fail due to unconnected stream
	StreamPeerTLS::Status status = tls->get_status();
	CHECK(status == StreamPeerTLS::STATUS_DISCONNECTED);
	CHECK_MESSAGE(status != StreamPeerTLS::STATUS_CONNECTED, "Status should not be CONNECTED after failed connect.");
}

TEST_CASE("[StreamPeerTLS] Connect with null TLS options") {
	// Create a TCP stream peer (not connected, but valid for error testing)
	Ref<StreamPeerTCP> tcp = memnew(StreamPeerTCP);
	CHECK(tcp.is_valid());

	// Create a TLS stream peer
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();
	CHECK(tls.is_valid());

	// Attempt to connect with null TLS options
	Error err = tls->connect_to_stream(tcp, "localhost", Ref<TLSOptions>());
	CHECK(err != OK); // Should fail due to invalid TLS configuration
	StreamPeerTLS::Status status = tls->get_status();
	CHECK(status == StreamPeerTLS::STATUS_DISCONNECTED);
	CHECK_MESSAGE(status != StreamPeerTLS::STATUS_CONNECTED, "Status should not be CONNECTED after failed connect with null TLS options.");
}

TEST_CASE("[StreamPeerTLS] Accept stream with null TLS options") {
	// Create a TCP stream peer (not connected, but valid for error testing)
	Ref<StreamPeerTCP> tcp = memnew(StreamPeerTCP);
	CHECK(tcp.is_valid());

	// Create a TLS stream peer
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();
	CHECK(tls.is_valid());

	// Attempt to accept a stream with null TLS options
	Error err = tls->accept_stream(tcp, Ref<TLSOptions>());
	CHECK(err != OK); // Should fail due to invalid TLS configuration
	StreamPeerTLS::Status status = tls->get_status();
	CHECK(status == StreamPeerTLS::STATUS_DISCONNECTED);
	CHECK_MESSAGE(status != StreamPeerTLS::STATUS_CONNECTED, "Status should not be CONNECTED after failed accept.");
}

TEST_CASE("[StreamPeerTLS] Poll on disconnected stream") {
	// Create a TLS stream peer
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();
	CHECK(tls.is_valid());

	// Check initial status
	CHECK(tls->get_status() == StreamPeerTLS::STATUS_DISCONNECTED);

	// Call poll on a disconnected stream
	tls->poll();
	CHECK(tls->get_status() == StreamPeerTLS::STATUS_DISCONNECTED); // Should remain disconnected
}

TEST_CASE("[StreamPeerTLS] Data transfer on disconnected stream") {
	// Create a TLS stream peer
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();
	CHECK(tls.is_valid());

	// Attempt to send data on a disconnected stream
	PackedByteArray data = PackedByteArray();
	data.push_back(42);
	const uint8_t *data_ptr = data.ptr();
	Error err = tls->put_data(data_ptr, data.size());
	CHECK(err != OK); // Should fail due to disconnected state

	// Attempt to receive data on a disconnected stream
	Vector<uint8_t> buffer;
	buffer.resize_initialized(1);
	err = tls->get_data(buffer.ptrw(), 1);
	CHECK(err != OK); // Should fail due to disconnected state
	CHECK(buffer[0] == 0); // Buffer should remain unchanged (no data received)
}

TEST_CASE("[StreamPeerTLS] Disconnect from disconnected stream") {
	// Create a TLS stream peer
	Ref<StreamPeerTLS> tls = StreamPeerTLS::create();
	CHECK(tls.is_valid());

	// Check initial status
	CHECK(tls->get_status() == StreamPeerTLS::STATUS_DISCONNECTED);

	// Call disconnect_from_stream on a disconnected stream
	tls->disconnect_from_stream();
	CHECK(tls->get_status() == StreamPeerTLS::STATUS_DISCONNECTED); // Should remain disconnected
}

} // namespace TestStreamPeerTLS
