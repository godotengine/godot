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

#ifndef TEST_STREAM_PEER_TLS_H
#define TEST_STREAM_PEER_TLS_H

#include "core/io/stream_peer_mbedtls.h"
#include "core/io/stream_peer_tls.h"

#include "tests/test_macros.h"

namespace TestStreamPeerTLS {

TEST_CASE("[StreamPeerTLS] Not Available Initially") {
	StreamPeerTLS::_create = nullptr;
	CHECK_FALSE(StreamPeerTLS::is_available());

	Ref<StreamPeerTLS> stream_peer_tls = StreamPeerTLS::create();
	CHECK(stream_peer_tls.is_null());
}

TEST_CASE("[StreamPeerTLS] Available After Initialization") {
	StreamPeerMbedTLS::initialize_tls();
	CHECK(StreamPeerTLS::is_available());

	Ref<StreamPeerTLS> stream_peer_tls = StreamPeerTLS::create();
	CHECK(stream_peer_tls.is_valid());
	// stream_peer_tls should be an instance of StreamPeerMbedTLS
	CHECK(Object::cast_to<StreamPeerMbedTLS>(stream_peer_tls.ptr()));

	StreamPeerMbedTLS::finalize_tls(); // Clean up
}

} // namespace TestStreamPeerTLS

#endif // TEST_STREAM_PEER_TLS_H
