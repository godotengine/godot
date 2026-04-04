/**************************************************************************/
/*  web_rtc_peer_connection.cpp                                           */
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

#include <godot_cpp/classes/web_rtc_peer_connection.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/web_rtc_data_channel.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void WebRTCPeerConnection::set_default_extension(const StringName &p_extension_class) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("set_default_extension")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_extension_class);
}

Error WebRTCPeerConnection::initialize(const Dictionary &p_configuration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("initialize")._native_ptr(), 2625064318);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_configuration);
}

Ref<WebRTCDataChannel> WebRTCPeerConnection::create_data_channel(const String &p_label, const Dictionary &p_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("create_data_channel")._native_ptr(), 1288557393);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<WebRTCDataChannel>()));
	return Ref<WebRTCDataChannel>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<WebRTCDataChannel>(_gde_method_bind, _owner, &p_label, &p_options));
}

Error WebRTCPeerConnection::create_offer() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("create_offer")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error WebRTCPeerConnection::set_local_description(const String &p_type, const String &p_sdp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("set_local_description")._native_ptr(), 852856452);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_type, &p_sdp);
}

Error WebRTCPeerConnection::set_remote_description(const String &p_type, const String &p_sdp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("set_remote_description")._native_ptr(), 852856452);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_type, &p_sdp);
}

Error WebRTCPeerConnection::add_ice_candidate(const String &p_media, int32_t p_index, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("add_ice_candidate")._native_ptr(), 3958950400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_media, &p_index_encoded, &p_name);
}

Error WebRTCPeerConnection::poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("poll")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void WebRTCPeerConnection::close() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("close")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

WebRTCPeerConnection::ConnectionState WebRTCPeerConnection::get_connection_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("get_connection_state")._native_ptr(), 2275710506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (WebRTCPeerConnection::ConnectionState(0)));
	return (WebRTCPeerConnection::ConnectionState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

WebRTCPeerConnection::GatheringState WebRTCPeerConnection::get_gathering_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("get_gathering_state")._native_ptr(), 4262591401);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (WebRTCPeerConnection::GatheringState(0)));
	return (WebRTCPeerConnection::GatheringState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

WebRTCPeerConnection::SignalingState WebRTCPeerConnection::get_signaling_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCPeerConnection::get_class_static()._native_ptr(), StringName("get_signaling_state")._native_ptr(), 3342956226);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (WebRTCPeerConnection::SignalingState(0)));
	return (WebRTCPeerConnection::SignalingState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
