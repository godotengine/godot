/**************************************************************************/
/*  debug_trace_v2.h                                                      */
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

#include "core/templates/list.h"
#include "debug_protocol_v2.h"

class DebugTraceV2 {
	struct Entry {
		uint64_t timestamp_usec = 0;
		String direction;
		String channel;
		String type;
		uint64_t request_id = 0;
		uint64_t stop_id = 0;
		int payload_bytes = 0;
		uint64_t queue_usec = 0;
		String outcome;
	};

	List<Entry> entries;
	int capacity = 512;

public:
	void record(const String &p_direction, DebugProtocolV2::Channel p_channel, const DebugProtocolV2::Envelope &p_message, int p_payload_bytes, uint64_t p_queue_usec, const String &p_outcome);
	void record_state(DebugProtocolV2::State p_from, DebugProtocolV2::State p_to, const String &p_reason);
	Array export_metadata() const;
	void clear();
};
