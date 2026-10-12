/**************************************************************************/
/*  test_debug_protocol_v2.h                                              */
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

#include "core/debugger/debug_protocol_v2.h"
#include "tests/test_macros.h"

namespace TestDebugProtocolV2 {

TEST_CASE("[DebugProtocolV2] Round-trip envelope") {
	DebugProtocolV2::Envelope message;
	message.session_id = 123;
	message.generation = 2;
	message.channel = DebugProtocolV2::CHANNEL_DATA;
	message.sequence = 9;
	message.request_id = 8;
	message.stop_id = 7;
	message.type = "stack_result";
	message.payload.push_back("frame");

	DebugProtocolV2::Envelope decoded;
	String error;
	CHECK(DebugProtocolV2::decode(DebugProtocolV2::encode(message), decoded, error) == OK);
	CHECK(error.is_empty());
	CHECK(decoded.session_id == message.session_id);
	CHECK(decoded.generation == message.generation);
	CHECK(decoded.channel == message.channel);
	CHECK(decoded.sequence == message.sequence);
	CHECK(decoded.request_id == message.request_id);
	CHECK(decoded.stop_id == message.stop_id);
	CHECK(decoded.type == message.type);
	CHECK(decoded.payload == message.payload);
}

TEST_CASE("[DebugProtocolV2] Reject malformed envelope") {
	DebugProtocolV2::Envelope decoded;
	String error;
	CHECK(DebugProtocolV2::decode(Variant(), decoded, error) == ERR_INVALID_DATA);

	Array packet = DebugProtocolV2::encode(DebugProtocolV2::Envelope());
	packet[0] = "INVALID";
	CHECK(DebugProtocolV2::decode(packet, decoded, error) == ERR_INVALID_DATA);
	packet[0] = DebugProtocolV2::MAGIC;
	packet[1] = DebugProtocolV2::MAJOR + 1;
	CHECK(DebugProtocolV2::decode(packet, decoded, error) == ERR_INVALID_DATA);
	packet[1] = DebugProtocolV2::MAJOR;
	packet[5] = 3;
	CHECK(DebugProtocolV2::decode(packet, decoded, error) == ERR_INVALID_DATA);
	CHECK_FALSE(error.is_empty());
}

} // namespace TestDebugProtocolV2
