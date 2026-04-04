/**************************************************************************/
/*  web_socket_peer.hpp                                                   */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/packet_peer.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/tls_options.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedByteArray;
class StreamPeer;

class WebSocketPeer : public PacketPeer {
	GDEXTENSION_CLASS(WebSocketPeer, PacketPeer)

public:
	enum WriteMode {
		WRITE_MODE_TEXT = 0,
		WRITE_MODE_BINARY = 1,
	};

	enum State {
		STATE_CONNECTING = 0,
		STATE_OPEN = 1,
		STATE_CLOSING = 2,
		STATE_CLOSED = 3,
	};

	Error connect_to_url(const String &p_url, const Ref<TLSOptions> &p_tls_client_options = nullptr);
	Error accept_stream(const Ref<StreamPeer> &p_stream);
	Error send(const PackedByteArray &p_message, WebSocketPeer::WriteMode p_write_mode = (WebSocketPeer::WriteMode)1);
	Error send_text(const String &p_message);
	bool was_string_packet() const;
	void poll();
	void close(int32_t p_code = 1000, const String &p_reason = String());
	String get_connected_host() const;
	uint16_t get_connected_port() const;
	String get_selected_protocol() const;
	String get_requested_url() const;
	void set_no_delay(bool p_enabled);
	int32_t get_current_outbound_buffered_amount() const;
	WebSocketPeer::State get_ready_state() const;
	int32_t get_close_code() const;
	String get_close_reason() const;
	PackedStringArray get_supported_protocols() const;
	void set_supported_protocols(const PackedStringArray &p_protocols);
	PackedStringArray get_handshake_headers() const;
	void set_handshake_headers(const PackedStringArray &p_protocols);
	int32_t get_inbound_buffer_size() const;
	void set_inbound_buffer_size(int32_t p_buffer_size);
	int32_t get_outbound_buffer_size() const;
	void set_outbound_buffer_size(int32_t p_buffer_size);
	void set_max_queued_packets(int32_t p_buffer_size);
	int32_t get_max_queued_packets() const;
	void set_heartbeat_interval(double p_interval);
	double get_heartbeat_interval() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PacketPeer::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(WebSocketPeer::WriteMode);
VARIANT_ENUM_CAST(WebSocketPeer::State);

