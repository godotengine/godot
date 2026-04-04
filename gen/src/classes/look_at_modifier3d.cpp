/**************************************************************************/
/*  look_at_modifier3d.cpp                                                */
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

#include <godot_cpp/classes/look_at_modifier3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void LookAtModifier3D::set_target_node(const NodePath &p_target_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_target_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target_node);
}

NodePath LookAtModifier3D::get_target_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_target_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_bone_name(const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_bone_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_name);
}

String LookAtModifier3D::get_bone_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_bone_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_bone(int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_bone")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_encoded);
}

int32_t LookAtModifier3D::get_bone() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_bone")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_forward_axis(SkeletonModifier3D::BoneAxis p_forward_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_forward_axis")._native_ptr(), 3199955933);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_forward_axis_encoded;
	PtrToArg<int64_t>::encode(p_forward_axis, &p_forward_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_forward_axis_encoded);
}

SkeletonModifier3D::BoneAxis LookAtModifier3D::get_forward_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_forward_axis")._native_ptr(), 4076020284);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::BoneAxis(0)));
	return (SkeletonModifier3D::BoneAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_rotation_axis(Vector3::Axis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_rotation_axis")._native_ptr(), 1144690656);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis_encoded);
}

Vector3::Axis LookAtModifier3D::get_primary_rotation_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_rotation_axis")._native_ptr(), 3050976882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3::Axis(0)));
	return (Vector3::Axis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_use_secondary_rotation(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_use_secondary_rotation")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool LookAtModifier3D::is_using_secondary_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("is_using_secondary_rotation")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_relative(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_relative")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool LookAtModifier3D::is_relative() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("is_relative")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_origin_safe_margin(float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_origin_safe_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

float LookAtModifier3D::get_origin_safe_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_origin_safe_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_origin_from(LookAtModifier3D::OriginFrom p_origin_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_origin_from")._native_ptr(), 4254695669);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_origin_from_encoded;
	PtrToArg<int64_t>::encode(p_origin_from, &p_origin_from_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_origin_from_encoded);
}

LookAtModifier3D::OriginFrom LookAtModifier3D::get_origin_from() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_origin_from")._native_ptr(), 4057166297);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (LookAtModifier3D::OriginFrom(0)));
	return (LookAtModifier3D::OriginFrom)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_origin_bone_name(const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_origin_bone_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_name);
}

String LookAtModifier3D::get_origin_bone_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_origin_bone_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_origin_bone(int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_origin_bone")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bone_encoded);
}

int32_t LookAtModifier3D::get_origin_bone() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_origin_bone")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_origin_external_node(const NodePath &p_external_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_origin_external_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_external_node);
}

NodePath LookAtModifier3D::get_origin_external_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_origin_external_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_origin_offset(const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_origin_offset")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector3 LookAtModifier3D::get_origin_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_origin_offset")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_duration(float p_duration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_duration")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_duration_encoded);
}

float LookAtModifier3D::get_duration() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_duration")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_transition_type(Tween::TransitionType p_transition_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_transition_type")._native_ptr(), 1058637742);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_transition_type_encoded;
	PtrToArg<int64_t>::encode(p_transition_type, &p_transition_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_transition_type_encoded);
}

Tween::TransitionType LookAtModifier3D::get_transition_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_transition_type")._native_ptr(), 3842314528);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Tween::TransitionType(0)));
	return (Tween::TransitionType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_ease_type(Tween::EaseType p_ease_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_ease_type")._native_ptr(), 1208105857);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_ease_type_encoded;
	PtrToArg<int64_t>::encode(p_ease_type, &p_ease_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ease_type_encoded);
}

Tween::EaseType LookAtModifier3D::get_ease_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_ease_type")._native_ptr(), 631880200);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Tween::EaseType(0)));
	return (Tween::EaseType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_use_angle_limitation(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_use_angle_limitation")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool LookAtModifier3D::is_using_angle_limitation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("is_using_angle_limitation")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_symmetry_limitation(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_symmetry_limitation")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool LookAtModifier3D::is_limitation_symmetry() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("is_limitation_symmetry")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_limit_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_limit_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float LookAtModifier3D::get_primary_limit_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_limit_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_damp_threshold(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_damp_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float LookAtModifier3D::get_primary_damp_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_damp_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_positive_limit_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_positive_limit_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float LookAtModifier3D::get_primary_positive_limit_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_positive_limit_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_positive_damp_threshold(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_positive_damp_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float LookAtModifier3D::get_primary_positive_damp_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_positive_damp_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_negative_limit_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_negative_limit_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float LookAtModifier3D::get_primary_negative_limit_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_negative_limit_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_primary_negative_damp_threshold(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_primary_negative_damp_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float LookAtModifier3D::get_primary_negative_damp_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_primary_negative_damp_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_secondary_limit_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_secondary_limit_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float LookAtModifier3D::get_secondary_limit_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_secondary_limit_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_secondary_damp_threshold(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_secondary_damp_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float LookAtModifier3D::get_secondary_damp_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_secondary_damp_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_secondary_positive_limit_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_secondary_positive_limit_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float LookAtModifier3D::get_secondary_positive_limit_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_secondary_positive_limit_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_secondary_positive_damp_threshold(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_secondary_positive_damp_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float LookAtModifier3D::get_secondary_positive_damp_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_secondary_positive_damp_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_secondary_negative_limit_angle(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_secondary_negative_limit_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

float LookAtModifier3D::get_secondary_negative_limit_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_secondary_negative_limit_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void LookAtModifier3D::set_secondary_negative_damp_threshold(float p_power) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("set_secondary_negative_damp_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_power_encoded);
}

float LookAtModifier3D::get_secondary_negative_damp_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_secondary_negative_damp_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float LookAtModifier3D::get_interpolation_remaining() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("get_interpolation_remaining")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool LookAtModifier3D::is_interpolating() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("is_interpolating")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool LookAtModifier3D::is_target_within_limitation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LookAtModifier3D::get_class_static()._native_ptr(), StringName("is_target_within_limitation")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
