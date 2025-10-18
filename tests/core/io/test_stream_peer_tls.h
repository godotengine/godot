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

#include "core/io/stream_peer_tls.h"
#include "tests/test_macros.h"

namespace TestStreamPeerTLS {

TEST_CASE("[StreamPeerTLS] Availability and creation") {
	// Factory should report available
	CHECK(StreamPeerTLS::is_available() == true);

	// Create a backend instance
	StreamPeerTLS *tls = StreamPeerTLS::create();

	// Check that we got a valid object
	CHECK(tls != nullptr);
	CHECK_MESSAGE(tls != nullptr, "TLS backend should be registered and create() should return a valid object.");

	// Clean up the object to prevent ObjectDB leak
	memdelete(tls);
}

TEST_CASE("[StreamPeerTLS] Enum values") {
	CHECK(StreamPeerTLS::STATUS_DISCONNECTED == 0);
	CHECK(StreamPeerTLS::STATUS_HANDSHAKING == 1);
	CHECK(StreamPeerTLS::STATUS_CONNECTED == 2);
	CHECK(StreamPeerTLS::STATUS_ERROR == 3);
	CHECK(StreamPeerTLS::STATUS_ERROR_HOSTNAME_MISMATCH == 4);
}

} // namespace TestStreamPeerTLS
