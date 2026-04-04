/**************************************************************************/
/*  web_rtc_multiplayer_peer.cpp                                          */
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

#include <godot_cpp/classes/web_rtc_multiplayer_peer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/web_rtc_peer_connection.hpp>

namespace godot {

Error WebRTCMultiplayerPeer::create_server(const Array &p_channels_config) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("create_server")._native_ptr(), 2865356025);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_channels_config);
}

Error WebRTCMultiplayerPeer::create_client(int32_t p_peer_id, const Array &p_channels_config) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("create_client")._native_ptr(), 2641732907);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_peer_id_encoded, &p_channels_config);
}

Error WebRTCMultiplayerPeer::create_mesh(int32_t p_peer_id, const Array &p_channels_config) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("create_mesh")._native_ptr(), 2641732907);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_peer_id_encoded, &p_channels_config);
}

Error WebRTCMultiplayerPeer::add_peer(const Ref<WebRTCPeerConnection> &p_peer, int32_t p_peer_id, int32_t p_unreliable_lifetime) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("add_peer")._native_ptr(), 4078953270);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	int64_t p_unreliable_lifetime_encoded;
	PtrToArg<int64_t>::encode(p_unreliable_lifetime, &p_unreliable_lifetime_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_peer != nullptr ? &p_peer->_owner : nullptr), &p_peer_id_encoded, &p_unreliable_lifetime_encoded);
}

void WebRTCMultiplayerPeer::remove_peer(int32_t p_peer_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("remove_peer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_peer_id_encoded);
}

bool WebRTCMultiplayerPeer::has_peer(int32_t p_peer_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("has_peer")._native_ptr(), 3067735520);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_peer_id_encoded);
}

Dictionary WebRTCMultiplayerPeer::get_peer(int32_t p_peer_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("get_peer")._native_ptr(), 3554694381);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_peer_id_encoded);
}

Dictionary WebRTCMultiplayerPeer::get_peers() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WebRTCMultiplayerPeer::get_class_static()._native_ptr(), StringName("get_peers")._native_ptr(), 2382534195);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

} // namespace godot
