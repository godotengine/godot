/**************************************************************************/
/*  character_body3d.cpp                                                  */
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

#include <godot_cpp/classes/character_body3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/kinematic_collision3d.hpp>

namespace godot {

bool CharacterBody3D::move_and_slide() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("move_and_slide")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::apply_floor_snap() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("apply_floor_snap")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void CharacterBody3D::set_velocity(const Vector3 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_velocity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_velocity);
}

Vector3 CharacterBody3D::get_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_safe_margin(float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_safe_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

float CharacterBody3D::get_safe_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_safe_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_floor_stop_on_slope_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_floor_stop_on_slope_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_floor_stop_on_slope_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_floor_stop_on_slope_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void CharacterBody3D::set_floor_constant_speed_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_floor_constant_speed_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CharacterBody3D::is_floor_constant_speed_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_floor_constant_speed_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_floor_block_on_wall_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_floor_block_on_wall_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CharacterBody3D::is_floor_block_on_wall_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_floor_block_on_wall_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_slide_on_ceiling_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_slide_on_ceiling_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CharacterBody3D::is_slide_on_ceiling_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_slide_on_ceiling_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_platform_floor_layers(uint32_t p_exclude_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_platform_floor_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_exclude_layer_encoded;
	PtrToArg<int64_t>::encode(p_exclude_layer, &p_exclude_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclude_layer_encoded);
}

uint32_t CharacterBody3D::get_platform_floor_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_platform_floor_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_platform_wall_layers(uint32_t p_exclude_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_platform_wall_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_exclude_layer_encoded;
	PtrToArg<int64_t>::encode(p_exclude_layer, &p_exclude_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclude_layer_encoded);
}

uint32_t CharacterBody3D::get_platform_wall_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_platform_wall_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t CharacterBody3D::get_max_slides() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_max_slides")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_max_slides(int32_t p_max_slides) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_max_slides")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_slides_encoded;
	PtrToArg<int64_t>::encode(p_max_slides, &p_max_slides_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_slides_encoded);
}

float CharacterBody3D::get_floor_max_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_floor_max_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_floor_max_angle(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_floor_max_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

float CharacterBody3D::get_floor_snap_length() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_floor_snap_length")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_floor_snap_length(float p_floor_snap_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_floor_snap_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_floor_snap_length_encoded;
	PtrToArg<double>::encode(p_floor_snap_length, &p_floor_snap_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_floor_snap_length_encoded);
}

float CharacterBody3D::get_wall_min_slide_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_wall_min_slide_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_wall_min_slide_angle(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_wall_min_slide_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

Vector3 CharacterBody3D::get_up_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_up_direction")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_up_direction(const Vector3 &p_up_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_up_direction")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_up_direction);
}

void CharacterBody3D::set_motion_mode(CharacterBody3D::MotionMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_motion_mode")._native_ptr(), 2690739026);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

CharacterBody3D::MotionMode CharacterBody3D::get_motion_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_motion_mode")._native_ptr(), 3529553604);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CharacterBody3D::MotionMode(0)));
	return (CharacterBody3D::MotionMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CharacterBody3D::set_platform_on_leave(CharacterBody3D::PlatformOnLeave p_on_leave_apply_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("set_platform_on_leave")._native_ptr(), 1459986142);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_on_leave_apply_velocity_encoded;
	PtrToArg<int64_t>::encode(p_on_leave_apply_velocity, &p_on_leave_apply_velocity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_on_leave_apply_velocity_encoded);
}

CharacterBody3D::PlatformOnLeave CharacterBody3D::get_platform_on_leave() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_platform_on_leave")._native_ptr(), 996491171);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CharacterBody3D::PlatformOnLeave(0)));
	return (CharacterBody3D::PlatformOnLeave)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_on_floor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_on_floor")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_on_floor_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_on_floor_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_on_ceiling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_on_ceiling")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_on_ceiling_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_on_ceiling_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_on_wall() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_on_wall")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody3D::is_on_wall_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("is_on_wall_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Vector3 CharacterBody3D::get_floor_normal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_floor_normal")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 CharacterBody3D::get_wall_normal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_wall_normal")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 CharacterBody3D::get_last_motion() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_last_motion")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 CharacterBody3D::get_position_delta() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_position_delta")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 CharacterBody3D::get_real_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_real_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

float CharacterBody3D::get_floor_angle(const Vector3 &p_up_direction) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_floor_angle")._native_ptr(), 2906300789);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_up_direction);
}

Vector3 CharacterBody3D::get_platform_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_platform_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 CharacterBody3D::get_platform_angular_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_platform_angular_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

int32_t CharacterBody3D::get_slide_collision_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_slide_collision_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<KinematicCollision3D> CharacterBody3D::get_slide_collision(int32_t p_slide_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_slide_collision")._native_ptr(), 107003663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<KinematicCollision3D>()));
	int64_t p_slide_idx_encoded;
	PtrToArg<int64_t>::encode(p_slide_idx, &p_slide_idx_encoded);
	return Ref<KinematicCollision3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<KinematicCollision3D>(_gde_method_bind, _owner, &p_slide_idx_encoded));
}

Ref<KinematicCollision3D> CharacterBody3D::get_last_slide_collision() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody3D::get_class_static()._native_ptr(), StringName("get_last_slide_collision")._native_ptr(), 186875014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<KinematicCollision3D>()));
	return Ref<KinematicCollision3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<KinematicCollision3D>(_gde_method_bind, _owner));
}

} // namespace godot
