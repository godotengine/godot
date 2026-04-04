/**************************************************************************/
/*  skeleton_modification2dfabrik.cpp                                     */
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

#include <godot_cpp/classes/skeleton_modification2dfabrik.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void SkeletonModification2DFABRIK::set_target_node(const NodePath &p_target_nodepath) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("set_target_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target_nodepath);
}

NodePath SkeletonModification2DFABRIK::get_target_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("get_target_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void SkeletonModification2DFABRIK::set_fabrik_data_chain_length(int32_t p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("set_fabrik_data_chain_length")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_length_encoded;
	PtrToArg<int64_t>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

int32_t SkeletonModification2DFABRIK::get_fabrik_data_chain_length() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("get_fabrik_data_chain_length")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SkeletonModification2DFABRIK::set_fabrik_joint_bone2d_node(int32_t p_joint_idx, const NodePath &p_bone2d_nodepath) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("set_fabrik_joint_bone2d_node")._native_ptr(), 2761262315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_idx_encoded, &p_bone2d_nodepath);
}

NodePath SkeletonModification2DFABRIK::get_fabrik_joint_bone2d_node(int32_t p_joint_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("get_fabrik_joint_bone2d_node")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_joint_idx_encoded);
}

void SkeletonModification2DFABRIK::set_fabrik_joint_bone_index(int32_t p_joint_idx, int32_t p_bone_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("set_fabrik_joint_bone_index")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_idx_encoded, &p_bone_idx_encoded);
}

int32_t SkeletonModification2DFABRIK::get_fabrik_joint_bone_index(int32_t p_joint_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("get_fabrik_joint_bone_index")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_joint_idx_encoded);
}

void SkeletonModification2DFABRIK::set_fabrik_joint_magnet_position(int32_t p_joint_idx, const Vector2 &p_magnet_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("set_fabrik_joint_magnet_position")._native_ptr(), 163021252);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_idx_encoded, &p_magnet_position);
}

Vector2 SkeletonModification2DFABRIK::get_fabrik_joint_magnet_position(int32_t p_joint_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("get_fabrik_joint_magnet_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_joint_idx_encoded);
}

void SkeletonModification2DFABRIK::set_fabrik_joint_use_target_rotation(int32_t p_joint_idx, bool p_use_target_rotation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("set_fabrik_joint_use_target_rotation")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	int8_t p_use_target_rotation_encoded;
	PtrToArg<bool>::encode(p_use_target_rotation, &p_use_target_rotation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_idx_encoded, &p_use_target_rotation_encoded);
}

bool SkeletonModification2DFABRIK::get_fabrik_joint_use_target_rotation(int32_t p_joint_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DFABRIK::get_class_static()._native_ptr(), StringName("get_fabrik_joint_use_target_rotation")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_joint_idx_encoded;
	PtrToArg<int64_t>::encode(p_joint_idx, &p_joint_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_joint_idx_encoded);
}

} // namespace godot
