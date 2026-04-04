/**************************************************************************/
/*  http_client.cpp                                                       */
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

#include <godot_cpp/classes/http_client.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/stream_peer.hpp>

namespace godot {

Error HTTPClient::connect_to_host(const String &p_host, int32_t p_port, const Ref<TLSOptions> &p_tls_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("connect_to_host")._native_ptr(), 504540374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_port_encoded;
	PtrToArg<int64_t>::encode(p_port, &p_port_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_host, &p_port_encoded, (p_tls_options != nullptr ? &p_tls_options->_owner : nullptr));
}

void HTTPClient::set_connection(const Ref<StreamPeer> &p_connection) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("set_connection")._native_ptr(), 3281897016);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_connection != nullptr ? &p_connection->_owner : nullptr));
}

Ref<StreamPeer> HTTPClient::get_connection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_connection")._native_ptr(), 2741655269);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StreamPeer>()));
	return Ref<StreamPeer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StreamPeer>(_gde_method_bind, _owner));
}

Error HTTPClient::request_raw(HTTPClient::Method p_method, const String &p_url, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("request_raw")._native_ptr(), 540161961);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_method_encoded;
	PtrToArg<int64_t>::encode(p_method, &p_method_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_method_encoded, &p_url, &p_headers, &p_body);
}

Error HTTPClient::request(HTTPClient::Method p_method, const String &p_url, const PackedStringArray &p_headers, const String &p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("request")._native_ptr(), 3778990155);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_method_encoded;
	PtrToArg<int64_t>::encode(p_method, &p_method_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_method_encoded, &p_url, &p_headers, &p_body);
}

void HTTPClient::close() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("close")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool HTTPClient::has_response() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("has_response")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool HTTPClient::is_response_chunked() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("is_response_chunked")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t HTTPClient::get_response_code() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_response_code")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PackedStringArray HTTPClient::get_response_headers() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_response_headers")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

Dictionary HTTPClient::get_response_headers_as_dictionary() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_response_headers_as_dictionary")._native_ptr(), 2382534195);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

int64_t HTTPClient::get_response_body_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_response_body_length")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PackedByteArray HTTPClient::read_response_body_chunk() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("read_response_body_chunk")._native_ptr(), 2115431945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

void HTTPClient::set_read_chunk_size(int32_t p_bytes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("set_read_chunk_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bytes_encoded;
	PtrToArg<int64_t>::encode(p_bytes, &p_bytes_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bytes_encoded);
}

int32_t HTTPClient::get_read_chunk_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_read_chunk_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void HTTPClient::set_blocking_mode(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("set_blocking_mode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool HTTPClient::is_blocking_mode_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("is_blocking_mode_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

HTTPClient::Status HTTPClient::get_status() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("get_status")._native_ptr(), 1426656811);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HTTPClient::Status(0)));
	return (HTTPClient::Status)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error HTTPClient::poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("poll")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void HTTPClient::set_http_proxy(const String &p_host, int32_t p_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("set_http_proxy")._native_ptr(), 2956805083);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_port_encoded;
	PtrToArg<int64_t>::encode(p_port, &p_port_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_host, &p_port_encoded);
}

void HTTPClient::set_https_proxy(const String &p_host, int32_t p_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("set_https_proxy")._native_ptr(), 2956805083);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_port_encoded;
	PtrToArg<int64_t>::encode(p_port, &p_port_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_host, &p_port_encoded);
}

String HTTPClient::query_string_from_dict(const Dictionary &p_fields) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(HTTPClient::get_class_static()._native_ptr(), StringName("query_string_from_dict")._native_ptr(), 2538086567);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_fields);
}

} // namespace godot
