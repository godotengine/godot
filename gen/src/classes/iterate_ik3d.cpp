/**************************************************************************/
/*  iterate_ik3d.cpp                                                      */
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

#include <godot_cpp/classes/iterate_ik3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/joint_limitation3d.hpp>

namespace godot {

void IterateIK3D::set_max_iterations(int32_t p_max_iterations) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_max_iterations")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_iterations_encoded;
	PtrToArg<int64_t>::encode(p_max_iterations, &p_max_iterations_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_iterations_encoded);
}

int32_t IterateIK3D::get_max_iterations() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_max_iterations")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void IterateIK3D::set_min_distance(double p_min_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_min_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_distance_encoded;
	PtrToArg<double>::encode(p_min_distance, &p_min_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_distance_encoded);
}

double IterateIK3D::get_min_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_min_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void IterateIK3D::set_angular_delta_limit(double p_angular_delta_limit) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_angular_delta_limit")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angular_delta_limit_encoded;
	PtrToArg<double>::encode(p_angular_delta_limit, &p_angular_delta_limit_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_delta_limit_encoded);
}

double IterateIK3D::get_angular_delta_limit() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_angular_delta_limit")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void IterateIK3D::set_deterministic(bool p_deterministic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_deterministic")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_deterministic_encoded;
	PtrToArg<bool>::encode(p_deterministic, &p_deterministic_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_deterministic_encoded);
}

bool IterateIK3D::is_deterministic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("is_deterministic")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void IterateIK3D::set_target_node(int32_t p_index, const NodePath &p_target_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_target_node")._native_ptr(), 2761262315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_target_node);
}

NodePath IterateIK3D::get_target_node(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_target_node")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_index_encoded);
}

void IterateIK3D::set_joint_rotation_axis(int32_t p_index, int32_t p_joint, SkeletonModifier3D::RotationAxis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_joint_rotation_axis")._native_ptr(), 1391134969);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_axis_encoded);
}

SkeletonModifier3D::RotationAxis IterateIK3D::get_joint_rotation_axis(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_joint_rotation_axis")._native_ptr(), 3312594080);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::RotationAxis(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return (SkeletonModifier3D::RotationAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void IterateIK3D::set_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint, const Vector3 &p_axis_vector) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_joint_rotation_axis_vector")._native_ptr(), 2866752138);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_axis_vector);
}

Vector3 IterateIK3D::get_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_joint_rotation_axis_vector")._native_ptr(), 1592972041);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void IterateIK3D::set_joint_limitation(int32_t p_index, int32_t p_joint, const Ref<JointLimitation3D> &p_limitation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_joint_limitation")._native_ptr(), 1194636955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, (p_limitation != nullptr ? &p_limitation->_owner : nullptr));
}

Ref<JointLimitation3D> IterateIK3D::get_joint_limitation(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_joint_limitation")._native_ptr(), 91665146);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<JointLimitation3D>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return Ref<JointLimitation3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<JointLimitation3D>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded));
}

void IterateIK3D::set_joint_limitation_right_axis(int32_t p_index, int32_t p_joint, SkeletonModifier3D::SecondaryDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_joint_limitation_right_axis")._native_ptr(), 3838967147);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_direction_encoded);
}

SkeletonModifier3D::SecondaryDirection IterateIK3D::get_joint_limitation_right_axis(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_joint_limitation_right_axis")._native_ptr(), 623936134);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::SecondaryDirection(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return (SkeletonModifier3D::SecondaryDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void IterateIK3D::set_joint_limitation_right_axis_vector(int32_t p_index, int32_t p_joint, const Vector3 &p_vector) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_joint_limitation_right_axis_vector")._native_ptr(), 2866752138);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_vector);
}

Vector3 IterateIK3D::get_joint_limitation_right_axis_vector(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_joint_limitation_right_axis_vector")._native_ptr(), 1592972041);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void IterateIK3D::set_joint_limitation_rotation_offset(int32_t p_index, int32_t p_joint, const Quaternion &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("set_joint_limitation_rotation_offset")._native_ptr(), 4188936002);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_offset);
}

Quaternion IterateIK3D::get_joint_limitation_rotation_offset(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(IterateIK3D::get_class_static()._native_ptr(), StringName("get_joint_limitation_rotation_offset")._native_ptr(), 2722473700);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

} // namespace godot
