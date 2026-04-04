/**************************************************************************/
/*  bone_twist_disperser3d.cpp                                            */
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

#include <godot_cpp/classes/bone_twist_disperser3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/curve.hpp>

namespace godot {

void BoneTwistDisperser3D::set_setting_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_setting_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t BoneTwistDisperser3D::get_setting_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_setting_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void BoneTwistDisperser3D::clear_settings() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("clear_settings")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void BoneTwistDisperser3D::set_mutable_bone_axes(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_mutable_bone_axes")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool BoneTwistDisperser3D::are_bone_axes_mutable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("are_bone_axes_mutable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void BoneTwistDisperser3D::set_root_bone_name(int32_t p_index, const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_root_bone_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_name);
}

String BoneTwistDisperser3D::get_root_bone_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_root_bone_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_root_bone(int32_t p_index, int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_root_bone")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_encoded);
}

int32_t BoneTwistDisperser3D::get_root_bone(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_root_bone")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_end_bone_name(int32_t p_index, const String &p_bone_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_end_bone_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_name);
}

String BoneTwistDisperser3D::get_end_bone_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_end_bone_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_end_bone(int32_t p_index, int32_t p_bone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_end_bone")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_encoded);
}

int32_t BoneTwistDisperser3D::get_end_bone(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_end_bone")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String BoneTwistDisperser3D::get_reference_bone_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_reference_bone_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t BoneTwistDisperser3D::get_reference_bone(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_reference_bone")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_extend_end_bone(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_extend_end_bone")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool BoneTwistDisperser3D::is_end_bone_extended(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("is_end_bone_extended")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_end_bone_direction(int32_t p_index, SkeletonModifier3D::BoneDirection p_bone_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_end_bone_direction")._native_ptr(), 2838484201);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_bone_direction_encoded;
	PtrToArg<int64_t>::encode(p_bone_direction, &p_bone_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_bone_direction_encoded);
}

SkeletonModifier3D::BoneDirection BoneTwistDisperser3D::get_end_bone_direction(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_end_bone_direction")._native_ptr(), 1843036459);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::BoneDirection(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (SkeletonModifier3D::BoneDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_twist_from_rest(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_twist_from_rest")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool BoneTwistDisperser3D::is_twist_from_rest(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("is_twist_from_rest")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_twist_from(int32_t p_index, const Quaternion &p_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_twist_from")._native_ptr(), 2823819782);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_from);
}

Quaternion BoneTwistDisperser3D::get_twist_from(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_twist_from")._native_ptr(), 476865136);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_disperse_mode(int32_t p_index, BoneTwistDisperser3D::DisperseMode p_disperse_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_disperse_mode")._native_ptr(), 2954194337);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_disperse_mode_encoded;
	PtrToArg<int64_t>::encode(p_disperse_mode, &p_disperse_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_disperse_mode_encoded);
}

BoneTwistDisperser3D::DisperseMode BoneTwistDisperser3D::get_disperse_mode(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_disperse_mode")._native_ptr(), 1326397005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BoneTwistDisperser3D::DisperseMode(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (BoneTwistDisperser3D::DisperseMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_weight_position(int32_t p_index, float p_weight_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_weight_position")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_weight_position_encoded;
	PtrToArg<double>::encode(p_weight_position, &p_weight_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_weight_position_encoded);
}

float BoneTwistDisperser3D::get_weight_position(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_weight_position")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded);
}

void BoneTwistDisperser3D::set_damping_curve(int32_t p_index, const Ref<Curve> &p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_damping_curve")._native_ptr(), 1447180063);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_curve != nullptr ? &p_curve->_owner : nullptr));
}

Ref<Curve> BoneTwistDisperser3D::get_damping_curve(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_damping_curve")._native_ptr(), 747537754);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Curve>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Curve>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Curve>(_gde_method_bind, _owner, &p_index_encoded));
}

String BoneTwistDisperser3D::get_joint_bone_name(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_joint_bone_name")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

int32_t BoneTwistDisperser3D::get_joint_bone(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_joint_bone")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

float BoneTwistDisperser3D::get_joint_twist_amount(int32_t p_index, int32_t p_joint) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_joint_twist_amount")._native_ptr(), 3085491603);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded);
}

void BoneTwistDisperser3D::set_joint_twist_amount(int32_t p_index, int32_t p_joint, float p_twist_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("set_joint_twist_amount")._native_ptr(), 3506521499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_joint_encoded;
	PtrToArg<int64_t>::encode(p_joint, &p_joint_encoded);
	double p_twist_amount_encoded;
	PtrToArg<double>::encode(p_twist_amount, &p_twist_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_joint_encoded, &p_twist_amount_encoded);
}

int32_t BoneTwistDisperser3D::get_joint_count(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(BoneTwistDisperser3D::get_class_static()._native_ptr(), StringName("get_joint_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

} // namespace godot
