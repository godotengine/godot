/**************************************************************************/
/*  debug_protocol_v2.h                                                   */
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

#pragma once

#include "core/error/error_list.h"
#include "core/string/ustring.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

class DebugProtocolV2 {
public:
	static const char *MAGIC;
	static const int MAJOR = 2;
	static const int MINOR = 0;
	static const int ENVELOPE_FIELD_COUNT = 11;

	enum Channel {
		CHANNEL_CONTROL,
		CHANNEL_DATA,
	};

	enum State {
		STATE_DISCONNECTED,
		STATE_CONNECTING,
		STATE_RUNNING,
		STATE_PAUSE_REQUESTED,
		STATE_PAUSED,
		STATE_RESUME_REQUESTED,
	};

	struct Envelope {
		uint64_t session_id = 0;
		uint32_t generation = 0;
		Channel channel = CHANNEL_CONTROL;
		uint64_t sequence = 0;
		uint64_t request_id = 0;
		uint64_t stop_id = 0;
		String type;
		Array payload;
	};

	static Array encode(const Envelope &p_message);
	static Error decode(const Variant &p_packet, Envelope &r_message, String &r_error);
	static String channel_to_string(Channel p_channel);
	static String state_to_string(State p_state);
};
