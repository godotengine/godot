/**************************************************************************/
/*  web_socket_peer.cpp                                                   */
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

#include <godot_cpp/classes/web_socket_peer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/stream_peer.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

namespace godot {

Error WebSocketPeer::connect_to_url(const String &p_url, const Ref<TLSOptions> &p_tls_client_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("connect_to_url")._native_ptr(), 1966198364);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_url, (p_tls_client_options != nullptr ? &p_tls_client_options->_owner : nullptr));
}

Error WebSocketPeer::accept_stream(const Ref<StreamPeer> &p_stream) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("accept_stream")._native_ptr(), 255125695);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_stream != nullptr ? &p_stream->_owner : nullptr));
}

Error WebSocketPeer::send(const PackedByteArray &p_message, WebSocketPeer::WriteMode p_write_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("send")._native_ptr(), 2780360567);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_write_mode_encoded;
	PtrToArg<int64_t>::encode(p_write_mode, &p_write_mode_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_message, &p_write_mode_encoded);
}

Error WebSocketPeer::send_text(const String &p_message) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("send_text")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_message);
}

bool WebSocketPeer::was_string_packet() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("was_string_packet")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void WebSocketPeer::poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("poll")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void WebSocketPeer::close(int32_t p_code, const String &p_reason) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("close")._native_ptr(), 1047156615);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_code_encoded;
	PtrToArg<int64_t>::encode(p_code, &p_code_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_code_encoded, &p_reason);
}

String WebSocketPeer::get_connected_host() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_connected_host")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

uint16_t WebSocketPeer::get_connected_port() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_connected_port")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String WebSocketPeer::get_selected_protocol() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_selected_protocol")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String WebSocketPeer::get_requested_url() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_requested_url")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void WebSocketPeer::set_no_delay(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_no_delay")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

int32_t WebSocketPeer::get_current_outbound_buffered_amount() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_current_outbound_buffered_amount")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

WebSocketPeer::State WebSocketPeer::get_ready_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_ready_state")._native_ptr(), 346482985);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (WebSocketPeer::State(0)));
	return (WebSocketPeer::State)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t WebSocketPeer::get_close_code() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_close_code")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String WebSocketPeer::get_close_reason() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_close_reason")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

PackedStringArray WebSocketPeer::get_supported_protocols() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_supported_protocols")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void WebSocketPeer::set_supported_protocols(const PackedStringArray &p_protocols) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_supported_protocols")._native_ptr(), 4015028928);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_protocols);
}

PackedStringArray WebSocketPeer::get_handshake_headers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_handshake_headers")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void WebSocketPeer::set_handshake_headers(const PackedStringArray &p_protocols) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_handshake_headers")._native_ptr(), 4015028928);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_protocols);
}

int32_t WebSocketPeer::get_inbound_buffer_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_inbound_buffer_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void WebSocketPeer::set_inbound_buffer_size(int32_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_inbound_buffer_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer_size_encoded);
}

int32_t WebSocketPeer::get_outbound_buffer_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_outbound_buffer_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void WebSocketPeer::set_outbound_buffer_size(int32_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_outbound_buffer_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer_size_encoded);
}

void WebSocketPeer::set_max_queued_packets(int32_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_max_queued_packets")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer_size_encoded);
}

int32_t WebSocketPeer::get_max_queued_packets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_max_queued_packets")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void WebSocketPeer::set_heartbeat_interval(double p_interval) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("set_heartbeat_interval")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_interval_encoded;
	PtrToArg<double>::encode(p_interval, &p_interval_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interval_encoded);
}

double WebSocketPeer::get_heartbeat_interval() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebSocketPeer::get_class_static()._native_ptr(), StringName("get_heartbeat_interval")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
