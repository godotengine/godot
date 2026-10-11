/**************************************************************************/
/*  debugger_session_v2.h                                                 */
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

#ifndef EDITOR_DEBUGGER_SESSION_V2_H
#define EDITOR_DEBUGGER_SESSION_V2_H

#include "core/io/packet_peer.h"
#include "core/io/tcp_server.h"
#include "core/list.h"
#include "scene/debugger/debug_protocol_v2.h"

class EditorDebuggerSessionV2 {
	struct PendingConnection {
		Ref<StreamPeerTCP> tcp;
		Ref<PacketPeerStream> peer;
		uint64_t accepted_msec = 0;
	};

	List<PendingConnection> pending;
	Ref<StreamPeerTCP> candidate_control_tcp;
	Ref<PacketPeerStream> candidate_control_peer;
	Ref<StreamPeerTCP> candidate_data_tcp;
	Ref<PacketPeerStream> candidate_data_peer;
	uint64_t candidate_session_id = 0;
	uint32_t candidate_generation = 0;

	Ref<StreamPeerTCP> control_tcp;
	Ref<PacketPeerStream> control_peer;
	Ref<StreamPeerTCP> data_tcp;
	Ref<PacketPeerStream> data_peer;
	uint64_t session_id = 0;
	uint32_t generation = 0;
	uint64_t control_sequence = 0;
	uint64_t data_sequence = 0;
	bool just_connected = false;
	bool just_disconnected = false;

	Error _accept_hello(const PendingConnection &p_pending, const DebugProtocolV2::Envelope &p_hello);
	Error _send_hello_ack(Ref<PacketPeerStream> p_peer, DebugProtocolV2::Channel p_channel);
	Error _poll(Ref<PacketPeerStream> p_peer, DebugProtocolV2::Channel p_channel, DebugProtocolV2::Envelope &r_message);
	void _activate_candidate();
	void _clear_candidate();

public:
	void poll_server(Ref<TCP_Server> p_server);
	Error send_control(const String &p_type, const Array &p_payload, uint64_t p_request_id, uint64_t p_stop_id);
	Error send_data(const String &p_type, const Array &p_payload, uint64_t p_request_id, uint64_t p_stop_id);
	Error poll_control(DebugProtocolV2::Envelope &r_message);
	Error poll_data(DebugProtocolV2::Envelope &r_message);

	bool is_connected() const;
	bool take_just_connected();
	bool take_just_disconnected();
	uint64_t get_session_id() const { return session_id; }
	uint32_t get_generation() const { return generation; }
	Ref<StreamPeerTCP> get_control_tcp() const { return control_tcp; }
	void reset();
};

#endif // EDITOR_DEBUGGER_SESSION_V2_H
