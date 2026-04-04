/**************************************************************************/
/*  skeleton_modification2d_two_bone_ik.cpp                               */
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

#include <godot_cpp/classes/skeleton_modification2d_two_bone_ik.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void SkeletonModification2DTwoBoneIK::set_target_node(const NodePath &p_target_nodepath) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_target_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target_nodepath);
}

NodePath SkeletonModification2DTwoBoneIK::get_target_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_target_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_target_minimum_distance(float p_minimum_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_target_minimum_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_minimum_distance_encoded;
	PtrToArg<double>::encode(p_minimum_distance, &p_minimum_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_minimum_distance_encoded);
}

float SkeletonModification2DTwoBoneIK::get_target_minimum_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_target_minimum_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_target_maximum_distance(float p_maximum_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_target_maximum_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_maximum_distance_encoded;
	PtrToArg<double>::encode(p_maximum_distance, &p_maximum_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_maximum_distance_encoded);
}

float SkeletonModification2DTwoBoneIK::get_target_maximum_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_target_maximum_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_flip_bend_direction(bool p_flip_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_flip_bend_direction")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_direction_encoded;
	PtrToArg<bool>::encode(p_flip_direction, &p_flip_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_direction_encoded);
}

bool SkeletonModification2DTwoBoneIK::get_flip_bend_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_flip_bend_direction")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_joint_one_bone2d_node(const NodePath &p_bone2d_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_joint_one_bone2d_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone2d_node);
}

NodePath SkeletonModification2DTwoBoneIK::get_joint_one_bone2d_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_joint_one_bone2d_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_joint_one_bone_idx(int32_t p_bone_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_joint_one_bone_idx")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

int32_t SkeletonModification2DTwoBoneIK::get_joint_one_bone_idx() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_joint_one_bone_idx")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_joint_two_bone2d_node(const NodePath &p_bone2d_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_joint_two_bone2d_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone2d_node);
}

NodePath SkeletonModification2DTwoBoneIK::get_joint_two_bone2d_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_joint_two_bone2d_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void SkeletonModification2DTwoBoneIK::set_joint_two_bone_idx(int32_t p_bone_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("set_joint_two_bone_idx")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_idx_encoded;
	PtrToArg<int64_t>::encode(p_bone_idx, &p_bone_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_idx_encoded);
}

int32_t SkeletonModification2DTwoBoneIK::get_joint_two_bone_idx() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SkeletonModification2DTwoBoneIK::get_class_static()._native_ptr(), StringName("get_joint_two_bone_idx")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
