/**************************************************************************/
/*  debug_protocol_v2.cpp                                                 */
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
/* "Software"), to deal in the Software without restriction, including  */
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

#include "debug_protocol_v2.h"

const char *DebugProtocolV2::MAGIC = "GDDBG2";

Array DebugProtocolV2::encode(const Envelope &p_message) {
	Array packet;
	packet.resize(ENVELOPE_FIELD_COUNT);
	packet[0] = MAGIC;
	packet[1] = MAJOR;
	packet[2] = MINOR;
	packet[3] = (int64_t)p_message.session_id;
	packet[4] = (int64_t)p_message.generation;
	packet[5] = (int64_t)p_message.channel;
	packet[6] = (int64_t)p_message.sequence;
	packet[7] = (int64_t)p_message.request_id;
	packet[8] = (int64_t)p_message.stop_id;
	packet[9] = p_message.type;
	packet[10] = p_message.payload;
	return packet;
}

Error DebugProtocolV2::decode(const Variant &p_packet, Envelope &r_message, String &r_error) {
	if (p_packet.get_type() != Variant::ARRAY) {
		r_error = "Debugger v2 packet is not an Array.";
		return ERR_INVALID_DATA;
	}

	const Array packet = p_packet;
	if (packet.size() != ENVELOPE_FIELD_COUNT) {
		r_error = "Debugger v2 packet has an invalid field count.";
		return ERR_INVALID_DATA;
	}

	if (packet[0].get_type() != Variant::STRING || String(packet[0]) != MAGIC) {
		r_error = "Debugger v2 packet has invalid magic.";
		return ERR_INVALID_DATA;
	}
	for (int i = 1; i <= 8; i++) {
		if (packet[i].get_type() != Variant::INT) {
			r_error = "Debugger v2 packet has a non-integer header field.";
			return ERR_INVALID_DATA;
		}
	}
	if (packet[9].get_type() != Variant::STRING || packet[10].get_type() != Variant::ARRAY) {
		r_error = "Debugger v2 packet has an invalid message body.";
		return ERR_INVALID_DATA;
	}

	const int major = packet[1];
	const int minor = packet[2];
	if (major != MAJOR || minor != MINOR) {
		r_error = "Debugger v2 packet has an incompatible protocol version.";
		return ERR_INVALID_DATA;
	}

	const int channel = packet[5];
	if (channel < CHANNEL_CONTROL || channel > CHANNEL_DATA) {
		r_error = "Debugger v2 packet has an invalid channel.";
		return ERR_INVALID_DATA;
	}

	Envelope decoded;
	decoded.session_id = (uint64_t)(int64_t)packet[3];
	decoded.generation = (uint32_t)(int64_t)packet[4];
	decoded.channel = (Channel)channel;
	decoded.sequence = (uint64_t)(int64_t)packet[6];
	decoded.request_id = (uint64_t)(int64_t)packet[7];
	decoded.stop_id = (uint64_t)(int64_t)packet[8];
	decoded.type = packet[9];
	decoded.payload = packet[10];

	r_message = decoded;
	r_error = String();
	return OK;
}

String DebugProtocolV2::channel_to_string(Channel p_channel) {
	switch (p_channel) {
		case CHANNEL_CONTROL:
			return "control";
		case CHANNEL_DATA:
			return "data";
	}
	return "unknown";
}

String DebugProtocolV2::state_to_string(State p_state) {
	switch (p_state) {
		case STATE_DISCONNECTED:
			return "disconnected";
		case STATE_CONNECTING:
			return "connecting";
		case STATE_RUNNING:
			return "running";
		case STATE_PAUSE_REQUESTED:
			return "pause_requested";
		case STATE_PAUSED:
			return "paused";
		case STATE_RESUME_REQUESTED:
			return "resume_requested";
	}
	return "unknown";
}
