/**************************************************************************/
/*  debugger_session_v2.cpp                                               */
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

#include "debugger_session_v2.h"

#include "core/os/os.h"

Error EditorDebuggerSessionV2::_send_hello_ack(Ref<PacketPeerStream> p_peer, DebugProtocolV2::Channel p_channel) {
	DebugProtocolV2::Envelope ack;
	ack.session_id = candidate_session_id;
	ack.generation = candidate_generation;
	ack.channel = p_channel;
	ack.sequence = 1;
	ack.request_id = 1;
	ack.type = "hello_ack";
	ack.payload.push_back(250);
	ack.payload.push_back(1000);
	ack.payload.push_back(100);
	return p_peer->put_var(DebugProtocolV2::encode(ack));
}

Error EditorDebuggerSessionV2::_accept_hello(const PendingConnection &p_pending, const DebugProtocolV2::Envelope &p_hello) {
	const bool control_hello = p_hello.channel == DebugProtocolV2::CHANNEL_CONTROL && p_hello.type == "control_hello";
	const bool data_hello = p_hello.channel == DebugProtocolV2::CHANNEL_DATA && p_hello.type == "data_hello";
	if (!control_hello && !data_hello) {
		return ERR_INVALID_DATA;
	}
	if (candidate_session_id == 0) {
		candidate_session_id = p_hello.session_id;
		candidate_generation = p_hello.generation;
	}
	if (candidate_session_id != p_hello.session_id || candidate_generation != p_hello.generation) {
		return ERR_ALREADY_IN_USE;
	}

	if (control_hello) {
		if (candidate_control_tcp.is_valid()) {
			return ERR_ALREADY_IN_USE;
		}
		candidate_control_tcp = p_pending.tcp;
		candidate_control_peer = p_pending.peer;
		return _send_hello_ack(candidate_control_peer, DebugProtocolV2::CHANNEL_CONTROL);
	}
	if (candidate_data_tcp.is_valid()) {
		return ERR_ALREADY_IN_USE;
	}
	candidate_data_tcp = p_pending.tcp;
	candidate_data_peer = p_pending.peer;
	return _send_hello_ack(candidate_data_peer, DebugProtocolV2::CHANNEL_DATA);
}

void EditorDebuggerSessionV2::_activate_candidate() {
	if (control_tcp.is_valid()) {
		control_tcp->disconnect_from_host();
	}
	if (data_tcp.is_valid()) {
		data_tcp->disconnect_from_host();
	}
	control_tcp = candidate_control_tcp;
	control_peer = candidate_control_peer;
	data_tcp = candidate_data_tcp;
	data_peer = candidate_data_peer;
	session_id = candidate_session_id;
	generation = candidate_generation;
	control_sequence = 1;
	data_sequence = 1;
	just_connected = true;
	print_verbose("Editor Debugger v2: paired control and data channels for session " + String::num_int64(session_id) + ".");
	_clear_candidate();
}

void EditorDebuggerSessionV2::_clear_candidate() {
	candidate_control_tcp.unref();
	candidate_control_peer.unref();
	candidate_data_tcp.unref();
	candidate_data_peer.unref();
	candidate_session_id = 0;
	candidate_generation = 0;
}

void EditorDebuggerSessionV2::poll_server(Ref<TCP_Server> p_server) {
	while (p_server.is_valid() && p_server->is_connection_available()) {
		PendingConnection pending_connection;
		pending_connection.tcp = p_server->take_connection();
		if (pending_connection.tcp.is_valid()) {
			pending_connection.tcp->set_no_delay(true);
			pending_connection.peer = Ref<PacketPeerStream>(memnew(PacketPeerStream));
			pending_connection.peer->set_stream_peer(pending_connection.tcp);
			pending_connection.accepted_msec = OS::get_singleton()->get_ticks_msec();
			pending.push_back(pending_connection);
		}
	}

	for (List<PendingConnection>::Element *element = pending.front(); element;) {
		List<PendingConnection>::Element *next = element->next();
		bool remove = false;
		if (!element->get().tcp->is_connected_to_host() || OS::get_singleton()->get_ticks_msec() - element->get().accepted_msec > 5000) {
			remove = true;
		} else if (element->get().peer->get_available_packet_count() > 0) {
			Variant packet;
			DebugProtocolV2::Envelope hello;
			String decode_error;
			Error error = element->get().peer->get_var(packet);
			if (error == OK) {
				error = DebugProtocolV2::decode(packet, hello, decode_error);
			}
			if (error == OK) {
				error = _accept_hello(element->get(), hello);
			}
			if (error != OK) {
				element->get().tcp->disconnect_from_host();
			}
			remove = true;
		}
		if (remove) {
			pending.erase(element);
		}
		element = next;
	}

	if (candidate_control_tcp.is_valid() && candidate_data_tcp.is_valid()) {
		_activate_candidate();
	}
	if ((control_tcp.is_valid() && !control_tcp->is_connected_to_host()) || (data_tcp.is_valid() && !data_tcp->is_connected_to_host())) {
		just_disconnected = true;
		control_tcp.unref();
		control_peer.unref();
		data_tcp.unref();
		data_peer.unref();
	}
}

Error EditorDebuggerSessionV2::_poll(Ref<PacketPeerStream> p_peer, DebugProtocolV2::Channel p_channel, DebugProtocolV2::Envelope &r_message) {
	if (p_peer.is_null() || p_peer->get_available_packet_count() == 0) {
		return ERR_UNAVAILABLE;
	}
	Variant packet;
	Error error = p_peer->get_var(packet);
	if (error != OK) {
		return error;
	}
	DebugProtocolV2::Envelope message;
	String decode_error;
	error = DebugProtocolV2::decode(packet, message, decode_error);
	if (error != OK || message.session_id != session_id || message.generation != generation || message.channel != p_channel) {
		return ERR_INVALID_DATA;
	}
	r_message = message;
	return OK;
}

Error EditorDebuggerSessionV2::send_control(const String &p_type, const Array &p_payload, uint64_t p_request_id, uint64_t p_stop_id) {
	ERR_FAIL_COND_V(!is_connected(), ERR_CONNECTION_ERROR);
	DebugProtocolV2::Envelope message;
	message.session_id = session_id;
	message.generation = generation;
	message.channel = DebugProtocolV2::CHANNEL_CONTROL;
	message.sequence = ++control_sequence;
	message.request_id = p_request_id;
	message.stop_id = p_stop_id;
	message.type = p_type;
	message.payload = p_payload;
	return control_peer->put_var(DebugProtocolV2::encode(message));
}

Error EditorDebuggerSessionV2::send_data(const String &p_type, const Array &p_payload, uint64_t p_request_id, uint64_t p_stop_id) {
	ERR_FAIL_COND_V(!is_connected(), ERR_CONNECTION_ERROR);
	DebugProtocolV2::Envelope message;
	message.session_id = session_id;
	message.generation = generation;
	message.channel = DebugProtocolV2::CHANNEL_DATA;
	message.sequence = ++data_sequence;
	message.request_id = p_request_id;
	message.stop_id = p_stop_id;
	message.type = p_type;
	message.payload = p_payload;
	return data_peer->put_var(DebugProtocolV2::encode(message));
}

Error EditorDebuggerSessionV2::poll_control(DebugProtocolV2::Envelope &r_message) {
	return _poll(control_peer, DebugProtocolV2::CHANNEL_CONTROL, r_message);
}

Error EditorDebuggerSessionV2::poll_data(DebugProtocolV2::Envelope &r_message) {
	return _poll(data_peer, DebugProtocolV2::CHANNEL_DATA, r_message);
}

bool EditorDebuggerSessionV2::is_connected() const {
	return control_tcp.is_valid() && data_tcp.is_valid() && control_tcp->is_connected_to_host() && data_tcp->is_connected_to_host();
}

bool EditorDebuggerSessionV2::take_just_connected() {
	const bool value = just_connected;
	just_connected = false;
	return value;
}

bool EditorDebuggerSessionV2::take_just_disconnected() {
	const bool value = just_disconnected;
	just_disconnected = false;
	return value;
}

void EditorDebuggerSessionV2::reset() {
	for (List<PendingConnection>::Element *element = pending.front(); element; element = element->next()) {
		element->get().tcp->disconnect_from_host();
	}
	pending.clear();
	_clear_candidate();
	if (control_tcp.is_valid()) {
		control_tcp->disconnect_from_host();
	}
	if (data_tcp.is_valid()) {
		data_tcp->disconnect_from_host();
	}
	control_tcp.unref();
	control_peer.unref();
	data_tcp.unref();
	data_peer.unref();
	session_id = 0;
	generation = 0;
	just_connected = false;
	just_disconnected = false;
}
