/**************************************************************************/
/*  multiplayer_synchronizer.cpp                                          */
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

#include <godot_cpp/classes/multiplayer_synchronizer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/scene_replication_config.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

void MultiplayerSynchronizer::set_root_path(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_root_path")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

NodePath MultiplayerSynchronizer::get_root_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("get_root_path")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void MultiplayerSynchronizer::set_replication_interval(double p_milliseconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_replication_interval")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_milliseconds_encoded;
	PtrToArg<double>::encode(p_milliseconds, &p_milliseconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_milliseconds_encoded);
}

double MultiplayerSynchronizer::get_replication_interval() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("get_replication_interval")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MultiplayerSynchronizer::set_delta_interval(double p_milliseconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_delta_interval")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_milliseconds_encoded;
	PtrToArg<double>::encode(p_milliseconds, &p_milliseconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_milliseconds_encoded);
}

double MultiplayerSynchronizer::get_delta_interval() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("get_delta_interval")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void MultiplayerSynchronizer::set_replication_config(const Ref<SceneReplicationConfig> &p_config) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_replication_config")._native_ptr(), 3889206742);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_config != nullptr ? &p_config->_owner : nullptr));
}

Ref<SceneReplicationConfig> MultiplayerSynchronizer::get_replication_config() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("get_replication_config")._native_ptr(), 3200254614);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SceneReplicationConfig>()));
	return Ref<SceneReplicationConfig>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SceneReplicationConfig>(_gde_method_bind, _owner));
}

void MultiplayerSynchronizer::set_visibility_update_mode(MultiplayerSynchronizer::VisibilityUpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_visibility_update_mode")._native_ptr(), 3494860300);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

MultiplayerSynchronizer::VisibilityUpdateMode MultiplayerSynchronizer::get_visibility_update_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("get_visibility_update_mode")._native_ptr(), 3352241418);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (MultiplayerSynchronizer::VisibilityUpdateMode(0)));
	return (MultiplayerSynchronizer::VisibilityUpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiplayerSynchronizer::update_visibility(int32_t p_for_peer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("update_visibility")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_for_peer_encoded;
	PtrToArg<int64_t>::encode(p_for_peer, &p_for_peer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_for_peer_encoded);
}

void MultiplayerSynchronizer::set_visibility_public(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_visibility_public")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool MultiplayerSynchronizer::is_visibility_public() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("is_visibility_public")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void MultiplayerSynchronizer::add_visibility_filter(const Callable &p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("add_visibility_filter")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter);
}

void MultiplayerSynchronizer::remove_visibility_filter(const Callable &p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("remove_visibility_filter")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter);
}

void MultiplayerSynchronizer::set_visibility_for(int32_t p_peer, bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("set_visibility_for")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_peer_encoded;
	PtrToArg<int64_t>::encode(p_peer, &p_peer_encoded);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_peer_encoded, &p_visible_encoded);
}

bool MultiplayerSynchronizer::get_visibility_for(int32_t p_peer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerSynchronizer::get_class_static()._native_ptr(), StringName("get_visibility_for")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_peer_encoded;
	PtrToArg<int64_t>::encode(p_peer, &p_peer_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_peer_encoded);
}

} // namespace godot
