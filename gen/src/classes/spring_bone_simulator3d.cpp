/**************************************************************************/
/*  spring_bone_simulator3d.cpp                                           */
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

#include <godot_cpp/classes/spring_bone_simulator3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/curve.hpp>

namespace godot {

void SpringBoneSimulator3D::set_root_bone_name(int32_t p_index, const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_root_bone_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_name);
}

String SpringBoneSimulator3D::get_root_bone_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_root_bone_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_root_bone(int32_t p_index, int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_root_bone")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_encoded);
}

int32_t SpringBoneSimulator3D::get_root_bone(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_root_bone")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_end_bone_name(int32_t p_index, const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_end_bone_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_name);
}

String SpringBoneSimulator3D::get_end_bone_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_end_bone_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_end_bone(int32_t p_index, int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_end_bone")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_encoded);
}

int32_t SpringBoneSimulator3D::get_end_bone(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_end_bone")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_extend_end_bone(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_extend_end_bone")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool SpringBoneSimulator3D::is_end_bone_extended(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("is_end_bone_extended")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_end_bone_direction(int32_t p_index, SkeletonModifier3D::BoneDirection p_bone_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_end_bone_direction")._native_ptr(), 2838484201);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_direction_encoded;
	PtrToArg<int64_t>::encode(p_bone_direction, &p_bone_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_direction_encoded);
}

SkeletonModifier3D::BoneDirection SpringBoneSimulator3D::get_end_bone_direction(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_end_bone_direction")._native_ptr(), 1843036459);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::BoneDirection(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (SkeletonModifier3D::BoneDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_end_bone_length(int32_t p_index, float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_end_bone_length")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_length_encoded);
}

float SpringBoneSimulator3D::get_end_bone_length(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_end_bone_length")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_center_from(int32_t p_index, SpringBoneSimulator3D::CenterFrom p_center_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_center_from")._native_ptr(), 2551505749);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_center_from_encoded;
	PtrToArg<int64_t>::encode(p_center_from, &p_center_from_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_center_from_encoded);
}

SpringBoneSimulator3D::CenterFrom SpringBoneSimulator3D::get_center_from(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_center_from")._native_ptr(), 2721930813);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SpringBoneSimulator3D::CenterFrom(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (SpringBoneSimulator3D::CenterFrom)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_center_node(int32_t p_index, const NodePath &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_center_node")._native_ptr(), 2761262315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_node_path);
}

NodePath SpringBoneSimulator3D::get_center_node(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_center_node")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_center_bone_name(int32_t p_index, const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_center_bone_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_name);
}

String SpringBoneSimulator3D::get_center_bone_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_center_bone_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_center_bone(int32_t p_index, int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_center_bone")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_encoded);
}

int32_t SpringBoneSimulator3D::get_center_bone(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_center_bone")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_radius(int32_t p_index, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_radius")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_radius_encoded);
}

float SpringBoneSimulator3D::get_radius(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_radius")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_rotation_axis(int32_t p_index, SkeletonModifier3D::RotationAxis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_rotation_axis")._native_ptr(), 1539703856);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_axis_encoded);
}

SkeletonModifier3D::RotationAxis SpringBoneSimulator3D::get_rotation_axis(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_rotation_axis")._native_ptr(), 2844851118);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::RotationAxis(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (SkeletonModifier3D::RotationAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_rotation_axis_vector(int32_t p_index, const Vector3 &p_vector) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_rotation_axis_vector")._native_ptr(), 1530502735);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_vector);
}

Vector3 SpringBoneSimulator3D::get_rotation_axis_vector(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_rotation_axis_vector")._native_ptr(), 711720468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_radius_damping_curve(int32_t p_index, const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_radius_damping_curve")._native_ptr(), 1447180063);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> SpringBoneSimulator3D::get_radius_damping_curve(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_radius_damping_curve")._native_ptr(), 747537754);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner, &p_index_encoded));
}

void SpringBoneSimulator3D::set_stiffness(int32_t p_index, float p_stiffness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_stiffness")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_stiffness_encoded;
	PtrToArg<double>::encode(p_stiffness, &p_stiffness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_stiffness_encoded);
}

float SpringBoneSimulator3D::get_stiffness(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_stiffness")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_stiffness_damping_curve(int32_t p_index, const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_stiffness_damping_curve")._native_ptr(), 1447180063);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> SpringBoneSimulator3D::get_stiffness_damping_curve(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_stiffness_damping_curve")._native_ptr(), 747537754);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner, &p_index_encoded));
}

void SpringBoneSimulator3D::set_drag(int32_t p_index, float p_drag) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_drag")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_drag_encoded;
	PtrToArg<double>::encode(p_drag, &p_drag_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_drag_encoded);
}

float SpringBoneSimulator3D::get_drag(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_drag")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_drag_damping_curve(int32_t p_index, const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_drag_damping_curve")._native_ptr(), 1447180063);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> SpringBoneSimulator3D::get_drag_damping_curve(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_drag_damping_curve")._native_ptr(), 747537754);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner, &p_index_encoded));
}

void SpringBoneSimulator3D::set_gravity(int32_t p_index, float p_gravity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_gravity")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_gravity_encoded;
	PtrToArg<double>::encode(p_gravity, &p_gravity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_gravity_encoded);
}

float SpringBoneSimulator3D::get_gravity(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_gravity")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_gravity_damping_curve(int32_t p_index, const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_gravity_damping_curve")._native_ptr(), 1447180063);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> SpringBoneSimulator3D::get_gravity_damping_curve(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_gravity_damping_curve")._native_ptr(), 747537754);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner, &p_index_encoded));
}

void SpringBoneSimulator3D::set_gravity_direction(int32_t p_index, const Vector3 &p_gravity_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_gravity_direction")._native_ptr(), 1530502735);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_gravity_direction);
}

Vector3 SpringBoneSimulator3D::get_gravity_direction(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_gravity_direction")._native_ptr(), 711720468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_setting_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_setting_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t SpringBoneSimulator3D::get_setting_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_setting_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SpringBoneSimulator3D::clear_settings() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("clear_settings")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SpringBoneSimulator3D::set_individual_config(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_individual_config")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool SpringBoneSimulator3D::is_config_individual(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("is_config_individual")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String SpringBoneSimulator3D::get_joint_bone_name(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_bone_name")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

int32_t SpringBoneSimulator3D::get_joint_bone(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_bone")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_rotation_axis(int32_t p_index, int32_t p_joint, SkeletonModifier3D::RotationAxis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_rotation_axis")._native_ptr(), 1391134969);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_axis_encoded);
}

SkeletonModifier3D::RotationAxis SpringBoneSimulator3D::get_joint_rotation_axis(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_rotation_axis")._native_ptr(), 3312594080);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::RotationAxis(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return (SkeletonModifier3D::RotationAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint, const Vector3 &p_vector) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_rotation_axis_vector")._native_ptr(), 2866752138);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_vector);
}

Vector3 SpringBoneSimulator3D::get_joint_rotation_axis_vector(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_rotation_axis_vector")._native_ptr(), 1592972041);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_radius(int32_t p_index, int32_t p_joint, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_radius")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_radius_encoded);
}

float SpringBoneSimulator3D::get_joint_radius(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_radius")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_stiffness(int32_t p_index, int32_t p_joint, float p_stiffness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_stiffness")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	double p_stiffness_encoded;
	PtrToArg<double>::encode(p_stiffness, &p_stiffness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_stiffness_encoded);
}

float SpringBoneSimulator3D::get_joint_stiffness(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_stiffness")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_drag(int32_t p_index, int32_t p_joint, float p_drag) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_drag")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	double p_drag_encoded;
	PtrToArg<double>::encode(p_drag, &p_drag_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_drag_encoded);
}

float SpringBoneSimulator3D::get_joint_drag(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_drag")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_gravity(int32_t p_index, int32_t p_joint, float p_gravity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_gravity")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	double p_gravity_encoded;
	PtrToArg<double>::encode(p_gravity, &p_gravity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_gravity_encoded);
}

float SpringBoneSimulator3D::get_joint_gravity(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_gravity")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void SpringBoneSimulator3D::set_joint_gravity_direction(int32_t p_index, int32_t p_joint, const Vector3 &p_gravity_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_joint_gravity_direction")._native_ptr(), 2866752138);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_gravity_direction);
}

Vector3 SpringBoneSimulator3D::get_joint_gravity_direction(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_gravity_direction")._native_ptr(), 1592972041);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

int32_t SpringBoneSimulator3D::get_joint_count(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_joint_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_enable_all_child_collisions(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_enable_all_child_collisions")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool SpringBoneSimulator3D::are_all_child_collisions_enabled(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("are_all_child_collisions_enabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_exclude_collision_path(int32_t p_index, int32_t p_collision, const NodePath &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_exclude_collision_path")._native_ptr(), 132481804);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_collision_encoded;
	PtrToArg<int64_t>::encode(p_collision, &p_collision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_collision_encoded, &p_node_path);
}

NodePath SpringBoneSimulator3D::get_exclude_collision_path(int32_t p_index, int32_t p_collision) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_exclude_collision_path")._native_ptr(), 464924783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_collision_encoded;
	PtrToArg<int64_t>::encode(p_collision, &p_collision_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_index_encoded, &p_collision_encoded);
}

void SpringBoneSimulator3D::set_exclude_collision_count(int32_t p_index, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_exclude_collision_count")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_count_encoded);
}

int32_t SpringBoneSimulator3D::get_exclude_collision_count(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_exclude_collision_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::clear_exclude_collisions(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("clear_exclude_collisions")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_collision_path(int32_t p_index, int32_t p_collision, const NodePath &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_collision_path")._native_ptr(), 132481804);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_collision_encoded;
	PtrToArg<int64_t>::encode(p_collision, &p_collision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_collision_encoded, &p_node_path);
}

NodePath SpringBoneSimulator3D::get_collision_path(int32_t p_index, int32_t p_collision) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_collision_path")._native_ptr(), 464924783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_collision_encoded;
	PtrToArg<int64_t>::encode(p_collision, &p_collision_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_index_encoded, &p_collision_encoded);
}

void SpringBoneSimulator3D::set_collision_count(int32_t p_index, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_collision_count")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_count_encoded);
}

int32_t SpringBoneSimulator3D::get_collision_count(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_collision_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::clear_collisions(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("clear_collisions")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void SpringBoneSimulator3D::set_external_force(const Vector3 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_external_force")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

Vector3 SpringBoneSimulator3D::get_external_force() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("get_external_force")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void SpringBoneSimulator3D::set_mutable_bone_axes(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("set_mutable_bone_axes")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SpringBoneSimulator3D::are_bone_axes_mutable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("are_bone_axes_mutable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SpringBoneSimulator3D::reset() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpringBoneSimulator3D::get_class_static()._native_ptr(), StringName("reset")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
