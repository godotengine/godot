/**************************************************************************/
/*  scene_replication_config.cpp                                          */
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

#include <godot_cpp/classes/scene_replication_config.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

TypedArray<NodePath> SceneReplicationConfig::get_properties() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("get_properties")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<NodePath>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<NodePath>>(_gde_method_bind, _owner);
}

void SceneReplicationConfig::add_property(const NodePath &p_path, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("add_property")._native_ptr(), 4094619021);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_index_encoded);
}

bool SceneReplicationConfig::has_property(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("has_property")._native_ptr(), 861721659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

void SceneReplicationConfig::remove_property(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("remove_property")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

int32_t SceneReplicationConfig::property_get_index(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_get_index")._native_ptr(), 1382022557);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

bool SceneReplicationConfig::property_get_spawn(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_get_spawn")._native_ptr(), 3456846888);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

void SceneReplicationConfig::property_set_spawn(const NodePath &p_path, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_set_spawn")._native_ptr(), 3868023870);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_enabled_encoded);
}

SceneReplicationConfig::ReplicationMode SceneReplicationConfig::property_get_replication_mode(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_get_replication_mode")._native_ptr(), 2870606336);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SceneReplicationConfig::ReplicationMode(0)));
	return (SceneReplicationConfig::ReplicationMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

void SceneReplicationConfig::property_set_replication_mode(const NodePath &p_path, SceneReplicationConfig::ReplicationMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_set_replication_mode")._native_ptr(), 3200083865);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_mode_encoded);
}

bool SceneReplicationConfig::property_get_sync(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_get_sync")._native_ptr(), 3456846888);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

void SceneReplicationConfig::property_set_sync(const NodePath &p_path, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_set_sync")._native_ptr(), 3868023870);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_enabled_encoded);
}

bool SceneReplicationConfig::property_get_watch(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_get_watch")._native_ptr(), 3456846888);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

void SceneReplicationConfig::property_set_watch(const NodePath &p_path, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneReplicationConfig::get_class_static()._native_ptr(), StringName("property_set_watch")._native_ptr(), 3868023870);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_enabled_encoded);
}

} // namespace godot
