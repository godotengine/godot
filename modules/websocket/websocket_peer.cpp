/**************************************************************************/
/*  websocket_peer.cpp                                                    */
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

#include "websocket_peer.h"

WebSocketPeer *(*WebSocketPeer::_create)() = nullptr;

WebSocketPeer::WebSocketPeer() {
}

WebSocketPeer::~WebSocketPeer() {
}

void WebSocketPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("connect_to_url", "url", "tls_client_options"), &WebSocketPeer::connect_to_url, DEFVAL(Ref<TLSOptions>()));
	ClassDB::bind_method(D_METHOD("accept_stream", "stream"), &WebSocketPeer::accept_stream);
	ClassDB::bind_method(D_METHOD("send", "message", "write_mode"), &WebSocketPeer::_send_bind, DEFVAL(WRITE_MODE_BINARY));
	ClassDB::bind_method(D_METHOD("send_text", "message"), &WebSocketPeer::send_text);
	ClassDB::bind_method(D_METHOD("was_string_packet"), &WebSocketPeer::was_string_packet);
	ClassDB::bind_method(D_METHOD("poll"), &WebSocketPeer::poll);
	ClassDB::bind_method(D_METHOD("close", "code", "reason"), &WebSocketPeer::close, DEFVAL(1000), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_connected_host"), &WebSocketPeer::get_connected_host);
	ClassDB::bind_method(D_METHOD("get_connected_port"), &WebSocketPeer::get_connected_port);
	ClassDB::bind_method(D_METHOD("get_selected_protocol"), &WebSocketPeer::get_selected_protocol);
	ClassDB::bind_method(D_METHOD("get_requested_url"), &WebSocketPeer::get_requested_url);
	ClassDB::bind_method(D_METHOD("set_no_delay", "enabled"), &WebSocketPeer::set_no_delay);
	ClassDB::bind_method(D_METHOD("get_current_outbound_buffered_amount"), &WebSocketPeer::get_current_outbound_buffered_amount);

	ClassDB::bind_method(D_METHOD("get_ready_state"), &WebSocketPeer::get_ready_state);
	ClassDB::bind_method(D_METHOD("get_close_code"), &WebSocketPeer::get_close_code);
	ClassDB::bind_method(D_METHOD("get_close_reason"), &WebSocketPeer::get_close_reason);

	ClassDB::bind_method(D_METHOD("get_supported_protocols"), &WebSocketPeer::_get_supported_protocols);
	ClassDB::bind_method(D_METHOD("set_supported_protocols", "protocols"), &WebSocketPeer::set_supported_protocols);
	ClassDB::bind_method(D_METHOD("get_handshake_headers"), &WebSocketPeer::_get_handshake_headers);
	ClassDB::bind_method(D_METHOD("set_handshake_headers", "protocols"), &WebSocketPeer::set_handshake_headers);

	ClassDB::bind_method(D_METHOD("get_inbound_buffer_size"), &WebSocketPeer::get_inbound_buffer_size);
	ClassDB::bind_method(D_METHOD("set_inbound_buffer_size", "buffer_size"), &WebSocketPeer::set_inbound_buffer_size);
	ClassDB::bind_method(D_METHOD("get_outbound_buffer_size"), &WebSocketPeer::get_outbound_buffer_size);
	ClassDB::bind_method(D_METHOD("set_outbound_buffer_size", "buffer_size"), &WebSocketPeer::set_outbound_buffer_size);

	ClassDB::bind_method(D_METHOD("set_max_queued_packets", "buffer_size"), &WebSocketPeer::set_max_queued_packets);
	ClassDB::bind_method(D_METHOD("get_max_queued_packets"), &WebSocketPeer::get_max_queued_packets);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "supported_protocols"), "set_supported_protocols", "get_supported_protocols");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "handshake_headers"), "set_handshake_headers", "get_handshake_headers");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "inbound_buffer_size"), "set_inbound_buffer_size", "get_inbound_buffer_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "outbound_buffer_size"), "set_outbound_buffer_size", "get_outbound_buffer_size");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_queued_packets"), "set_max_queued_packets", "get_max_queued_packets");

	BIND_ENUM_CONSTANT(WRITE_MODE_TEXT);
	BIND_ENUM_CONSTANT(WRITE_MODE_BINARY);

	BIND_ENUM_CONSTANT(STATE_CONNECTING);
	BIND_ENUM_CONSTANT(STATE_OPEN);
	BIND_ENUM_CONSTANT(STATE_CLOSING);
	BIND_ENUM_CONSTANT(STATE_CLOSED);
}

Error WebSocketPeer::_send_bind(const PackedByteArray &p_message, WriteMode p_mode) {
	return send(p_message.ptr(), p_message.size(), p_mode);
}

Error WebSocketPeer::send_text(const String &p_text) {
	const CharString cs = p_text.utf8();
	return send((const uint8_t *)cs.ptr(), cs.length(), WRITE_MODE_TEXT);
}

void WebSocketPeer::set_supported_protocols(const Vector<String> &p_protocols) {
	// Strip edges from protocols.
	supported_protocols.resize(p_protocols.size());
	for (int i = 0; i < p_protocols.size(); i++) {
		supported_protocols.write[i] = p_protocols[i].strip_edges();
	}
}

const Vector<String> WebSocketPeer::get_supported_protocols() const {
	return supported_protocols;
}

Vector<String> WebSocketPeer::_get_supported_protocols() const {
	Vector<String> out;
	out.append_array(supported_protocols);
	return out;
}

void WebSocketPeer::set_handshake_headers(const Vector<String> &p_headers) {
	handshake_headers = p_headers;
}

const Vector<String> WebSocketPeer::get_handshake_headers() const {
	return handshake_headers;
}

Vector<String> WebSocketPeer::_get_handshake_headers() const {
	Vector<String> out;
	out.append_array(handshake_headers);
	return out;
}

void WebSocketPeer::set_outbound_buffer_size(int p_buffer_size) {
	outbound_buffer_size = p_buffer_size;
}

int WebSocketPeer::get_outbound_buffer_size() const {
	return outbound_buffer_size;
}

void WebSocketPeer::set_inbound_buffer_size(int p_buffer_size) {
	inbound_buffer_size = p_buffer_size;
}

int WebSocketPeer::get_inbound_buffer_size() const {
	return inbound_buffer_size;
}

void WebSocketPeer::set_max_queued_packets(int p_max_queued_packets) {
	max_queued_packets = p_max_queued_packets;
}

int WebSocketPeer::get_max_queued_packets() const {
	return max_queued_packets;
}
