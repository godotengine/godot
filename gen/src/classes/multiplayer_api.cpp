/**************************************************************************/
/*  multiplayer_api.cpp                                                   */
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

#include <godot_cpp/classes/multiplayer_api.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/multiplayer_peer.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

bool MultiplayerAPI::has_multiplayer_peer() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("has_multiplayer_peer")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<MultiplayerPeer> MultiplayerAPI::get_multiplayer_peer() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("get_multiplayer_peer")._native_ptr(), 3223692825);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<MultiplayerPeer>()));
	return Ref<MultiplayerPeer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<MultiplayerPeer>(_gde_method_bind, _owner));
}

void MultiplayerAPI::set_multiplayer_peer(const Ref<MultiplayerPeer> &p_peer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("set_multiplayer_peer")._native_ptr(), 3694835298);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_peer != nullptr ? &p_peer->_owner : nullptr));
}

int32_t MultiplayerAPI::get_unique_id() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("get_unique_id")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool MultiplayerAPI::is_server() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("is_server")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t MultiplayerAPI::get_remote_sender_id() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("get_remote_sender_id")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error MultiplayerAPI::poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("poll")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error MultiplayerAPI::rpc(int32_t p_peer, Object *p_object, const StringName &p_method, const Array &p_arguments) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("rpc")._native_ptr(), 2077486355);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_peer_encoded;
	PtrToArg<int64_t>::encode(p_peer, &p_peer_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_peer_encoded, (p_object != nullptr ? &p_object->_owner : nullptr), &p_method, &p_arguments);
}

Error MultiplayerAPI::object_configuration_add(Object *p_object, const Variant &p_configuration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("object_configuration_add")._native_ptr(), 1171879464);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_configuration);
}

Error MultiplayerAPI::object_configuration_remove(Object *p_object, const Variant &p_configuration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("object_configuration_remove")._native_ptr(), 1171879464);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr), &p_configuration);
}

PackedInt32Array MultiplayerAPI::get_peers() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("get_peers")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void MultiplayerAPI::set_default_interface(const StringName &p_interface_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("set_default_interface")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, &p_interface_name);
}

StringName MultiplayerAPI::get_default_interface() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("get_default_interface")._native_ptr(), 2737447660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, nullptr);
}

Ref<MultiplayerAPI> MultiplayerAPI::create_default_interface() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerAPI::get_class_static()._native_ptr(), StringName("create_default_interface")._native_ptr(), 3294156723);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<MultiplayerAPI>()));
	return Ref<MultiplayerAPI>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<MultiplayerAPI>(_gde_method_bind, nullptr));
}

} // namespace godot
