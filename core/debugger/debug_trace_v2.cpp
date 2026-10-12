/**************************************************************************/
/*  debug_trace_v2.cpp                                                    */
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

#include "debug_trace_v2.h"

#include "core/os/os.h"

void DebugTraceV2::record(const String &p_direction, DebugProtocolV2::Channel p_channel, const DebugProtocolV2::Envelope &p_message, int p_payload_bytes, uint64_t p_queue_usec, const String &p_outcome) {
	Entry entry;
	entry.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	entry.direction = p_direction;
	entry.channel = DebugProtocolV2::channel_to_string(p_channel);
	entry.type = p_message.type;
	entry.request_id = p_message.request_id;
	entry.stop_id = p_message.stop_id;
	entry.payload_bytes = p_payload_bytes;
	entry.queue_usec = p_queue_usec;
	entry.outcome = p_outcome;
	entries.push_back(entry);
	while (entries.size() > capacity) {
		entries.pop_front();
	}
}

void DebugTraceV2::record_state(DebugProtocolV2::State p_from, DebugProtocolV2::State p_to, const String &p_reason) {
	DebugProtocolV2::Envelope message;
	message.type = DebugProtocolV2::state_to_string(p_from) + "->" + DebugProtocolV2::state_to_string(p_to);
	record("state", DebugProtocolV2::CHANNEL_CONTROL, message, 0, 0, p_reason);
}

Array DebugTraceV2::export_metadata() const {
	Array result;
	for (const List<Entry>::Element *element = entries.front(); element; element = element->next()) {
		const Entry &entry = element->get();
		Array descriptor;
		descriptor.push_back((int64_t)entry.timestamp_usec);
		descriptor.push_back(entry.direction);
		descriptor.push_back(entry.channel);
		descriptor.push_back(entry.type);
		descriptor.push_back((int64_t)entry.request_id);
		descriptor.push_back((int64_t)entry.stop_id);
		descriptor.push_back(entry.payload_bytes);
		descriptor.push_back((int64_t)entry.queue_usec);
		descriptor.push_back(entry.outcome);
		result.push_back(descriptor);
	}
	return result;
}

void DebugTraceV2::clear() {
	entries.clear();
}
