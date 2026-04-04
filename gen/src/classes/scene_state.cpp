/**************************************************************************/
/*  scene_state.cpp                                                       */
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

#include <godot_cpp/classes/scene_state.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/packed_scene.hpp>

namespace godot {

String SceneState::get_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Ref<SceneState> SceneState::get_base_scene_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_base_scene_state")._native_ptr(), 3479783971);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SceneState>()));
	return Ref<SceneState>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SceneState>(_gde_method_bind, _owner));
}

int32_t SceneState::get_node_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

StringName SceneState::get_node_type(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_type")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded);
}

StringName SceneState::get_node_name(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_name")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded);
}

NodePath SceneState::get_node_path(int32_t p_idx, bool p_for_parent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_path")._native_ptr(), 2272487792);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_for_parent_encoded;
	PtrToArg<bool>::encode(p_for_parent, &p_for_parent_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_idx_encoded, &p_for_parent_encoded);
}

NodePath SceneState::get_node_owner_path(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_owner_path")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_idx_encoded);
}

bool SceneState::is_node_instance_placeholder(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("is_node_instance_placeholder")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

String SceneState::get_node_instance_placeholder(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_instance_placeholder")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_idx_encoded);
}

Ref<PackedScene> SceneState::get_node_instance(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_instance")._native_ptr(), 511017218);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PackedScene>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<PackedScene>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PackedScene>(_gde_method_bind, _owner, &p_idx_encoded));
}

PackedStringArray SceneState::get_node_groups(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_groups")._native_ptr(), 647634434);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_idx_encoded);
}

int32_t SceneState::get_node_index(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_index")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

int32_t SceneState::get_node_property_count(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_property_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

StringName SceneState::get_node_property_name(int32_t p_idx, int32_t p_prop_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_property_name")._native_ptr(), 351665558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_prop_idx_encoded;
	PtrToArg<int64_t>::encode(p_prop_idx, &p_prop_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded, &p_prop_idx_encoded);
}

Variant SceneState::get_node_property_value(int32_t p_idx, int32_t p_prop_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_node_property_value")._native_ptr(), 678354945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_prop_idx_encoded;
	PtrToArg<int64_t>::encode(p_prop_idx, &p_prop_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_idx_encoded, &p_prop_idx_encoded);
}

int32_t SceneState::get_connection_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

NodePath SceneState::get_connection_source(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_source")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_idx_encoded);
}

StringName SceneState::get_connection_signal(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_signal")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded);
}

NodePath SceneState::get_connection_target(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_target")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_idx_encoded);
}

StringName SceneState::get_connection_method(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_method")._native_ptr(), 659327637);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_idx_encoded);
}

int32_t SceneState::get_connection_flags(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_flags")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

Array SceneState::get_connection_binds(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_binds")._native_ptr(), 663333327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_idx_encoded);
}

int32_t SceneState::get_connection_unbinds(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SceneState::get_class_static()._native_ptr(), StringName("get_connection_unbinds")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

} // namespace godot
