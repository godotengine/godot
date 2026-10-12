/**************************************************************************/
/*  debug_transport_v2.h                                                  */
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

#include "core/io/packet_peer.h"
#include "core/io/stream_peer_tcp.h"
#include "debug_protocol_v2.h"
#include "debug_trace_v2.h"

class DebugTransportV2 {
	struct QueuedData {
		DebugProtocolV2::Envelope message;
		int priority = 0;
		int bytes = 0;
		uint64_t enqueued_usec = 0;
	};

	Ref<StreamPeerTCP> control_tcp;
	Ref<PacketPeerStream> control_peer;
	Ref<StreamPeerTCP> data_tcp;
	Ref<PacketPeerStream> data_peer;

	String reconnect_host;
	uint16_t reconnect_port = 0;
	uint64_t session_id = 0;
	uint32_t generation = 1;
	uint64_t control_sequence = 0;
	uint64_t data_sequence = 0;
	uint32_t heartbeat_interval_msec = 250;
	uint32_t heartbeat_timeout_msec = 1000;
	uint32_t variable_page_size = 100;
	bool reconnecting = false;
	bool reconnect_hello_sent = false;
	bool reconnect_control_ack = false;
	bool reconnect_data_ack = false;
	uint64_t reconnect_started_msec = 0;
	List<QueuedData> data_queue;
	int data_queue_bytes = 0;
	int data_queue_max_bytes = 256 * 1024;
	int data_queue_max_messages = 256;
	uint64_t dropped_data_messages = 0;
	uint64_t dropped_data_bytes = 0;
	DebugTraceV2 trace;

	static uint64_t _make_session_id();
	Error _wait_for_connections(uint32_t p_timeout_msec);
	Error _wait_for_hello_ack(DebugProtocolV2::Channel p_channel, uint32_t p_timeout_msec);
	Error _poll(Ref<PacketPeerStream> p_peer, DebugProtocolV2::Channel p_channel, DebugProtocolV2::Envelope &r_message);
	Error _put_data_immediate(const DebugProtocolV2::Envelope &p_message);

public:
	enum DataPriority {
		DATA_PRIORITY_ACTIVE = 0,
		DATA_PRIORITY_INTERACTIVE = 1,
		DATA_PRIORITY_BACKGROUND = 2,
		DATA_PRIORITY_OUTPUT = 3,
	};

	Error connect_to_host(const String &p_host, uint16_t p_port, uint32_t p_timeout_msec = 5000);
	Error start_reconnect();
	Error poll_reconnect(uint32_t p_timeout_msec = 5000);
	void disconnect();

	Error send_control(const String &p_type, const Array &p_payload = Array(), uint64_t p_request_id = 0, uint64_t p_stop_id = 0);
	Error send_data(const String &p_type, const Array &p_payload = Array(), uint64_t p_request_id = 0, uint64_t p_stop_id = 0, DataPriority p_priority = DATA_PRIORITY_ACTIVE);
	int flush_data(int p_max_packets = 4);
	Error poll_control(DebugProtocolV2::Envelope &r_message);
	Error poll_data(DebugProtocolV2::Envelope &r_message);

	bool is_control_connected() const;
	bool is_data_connected() const;
	bool is_connected() const;
	bool is_reconnecting() const { return reconnecting; }

	uint64_t get_session_id() const { return session_id; }
	uint32_t get_generation() const { return generation; }
	uint32_t get_heartbeat_interval_msec() const { return heartbeat_interval_msec; }
	uint32_t get_heartbeat_timeout_msec() const { return heartbeat_timeout_msec; }
	uint32_t get_variable_page_size() const { return variable_page_size; }
	String get_reconnect_host() const { return reconnect_host; }
	uint16_t get_reconnect_port() const { return reconnect_port; }
	int get_queued_data_bytes() const { return data_queue_bytes; }
	int get_queued_data_messages() const { return data_queue.size(); }
	int get_data_queue_max_bytes() const { return data_queue_max_bytes; }
	uint64_t get_dropped_data_messages() const { return dropped_data_messages; }
	uint64_t get_dropped_data_bytes() const { return dropped_data_bytes; }
	Array export_trace() const { return trace.export_metadata(); }
	void trace_state(DebugProtocolV2::State p_from, DebugProtocolV2::State p_to, const String &p_reason) { trace.record_state(p_from, p_to, p_reason); }

	// Removed after all legacy debugger data messages are migrated in tasks 4-5.
	Ref<PacketPeerStream> get_legacy_data_peer() const { return data_peer; }

	DebugTransportV2();
};
