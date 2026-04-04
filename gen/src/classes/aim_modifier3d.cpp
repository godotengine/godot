/**************************************************************************/
/*  aim_modifier3d.cpp                                                    */
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

#include <godot_cpp/classes/aim_modifier3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AimModifier3D::set_forward_axis(int32_t p_index, SkeletonModifier3D::BoneAxis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("set_forward_axis")._native_ptr(), 2496831085);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_axis_encoded);
}

SkeletonModifier3D::BoneAxis AimModifier3D::get_forward_axis(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("get_forward_axis")._native_ptr(), 3949866735);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SkeletonModifier3D::BoneAxis(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (SkeletonModifier3D::BoneAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void AimModifier3D::set_use_euler(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("set_use_euler")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool AimModifier3D::is_using_euler(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("is_using_euler")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void AimModifier3D::set_primary_rotation_axis(int32_t p_index, Vector3::Axis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("set_primary_rotation_axis")._native_ptr(), 776736805);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_axis_encoded);
}

Vector3::Axis AimModifier3D::get_primary_rotation_axis(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("get_primary_rotation_axis")._native_ptr(), 4131134770);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3::Axis(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Vector3::Axis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void AimModifier3D::set_use_secondary_rotation(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("set_use_secondary_rotation")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool AimModifier3D::is_using_secondary_rotation(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("is_using_secondary_rotation")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void AimModifier3D::set_relative(int32_t p_index, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("set_relative")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enabled_encoded);
}

bool AimModifier3D::is_relative(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AimModifier3D::get_class_static()._native_ptr(), StringName("is_relative")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

} // namespace godot
