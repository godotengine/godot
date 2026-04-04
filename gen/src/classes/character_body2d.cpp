/**************************************************************************/
/*  character_body2d.cpp                                                  */
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

#include <godot_cpp/classes/character_body2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/kinematic_collision2d.hpp>

namespace godot {

bool CharacterBody2D::move_and_slide() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("move_and_slide")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::apply_floor_snap() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("apply_floor_snap")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void CharacterBody2D::set_velocity(const Vector2 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_velocity")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_velocity);
}

Vector2 CharacterBody2D::get_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_velocity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_safe_margin(float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_safe_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

float CharacterBody2D::get_safe_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_safe_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_floor_stop_on_slope_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_floor_stop_on_slope_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_floor_stop_on_slope_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_floor_stop_on_slope_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void CharacterBody2D::set_floor_constant_speed_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_floor_constant_speed_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CharacterBody2D::is_floor_constant_speed_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_floor_constant_speed_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_floor_block_on_wall_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_floor_block_on_wall_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CharacterBody2D::is_floor_block_on_wall_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_floor_block_on_wall_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_slide_on_ceiling_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_slide_on_ceiling_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CharacterBody2D::is_slide_on_ceiling_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_slide_on_ceiling_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_platform_floor_layers(uint32_t p_exclude_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_platform_floor_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_exclude_layer_encoded;
	PtrToArg<int64_t>::encode(p_exclude_layer, &p_exclude_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclude_layer_encoded);
}

uint32_t CharacterBody2D::get_platform_floor_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_platform_floor_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_platform_wall_layers(uint32_t p_exclude_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_platform_wall_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_exclude_layer_encoded;
	PtrToArg<int64_t>::encode(p_exclude_layer, &p_exclude_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclude_layer_encoded);
}

uint32_t CharacterBody2D::get_platform_wall_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_platform_wall_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t CharacterBody2D::get_max_slides() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_max_slides")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_max_slides(int32_t p_max_slides) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_max_slides")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_slides_encoded;
	PtrToArg<int64_t>::encode(p_max_slides, &p_max_slides_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_slides_encoded);
}

float CharacterBody2D::get_floor_max_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_floor_max_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_floor_max_angle(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_floor_max_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

float CharacterBody2D::get_floor_snap_length() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_floor_snap_length")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_floor_snap_length(float p_floor_snap_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_floor_snap_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_floor_snap_length_encoded;
	PtrToArg<double>::encode(p_floor_snap_length, &p_floor_snap_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_floor_snap_length_encoded);
}

float CharacterBody2D::get_wall_min_slide_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_wall_min_slide_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_wall_min_slide_angle(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_wall_min_slide_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

Vector2 CharacterBody2D::get_up_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_up_direction")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_up_direction(const Vector2 &p_up_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_up_direction")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_up_direction);
}

void CharacterBody2D::set_motion_mode(CharacterBody2D::MotionMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_motion_mode")._native_ptr(), 1224392233);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

CharacterBody2D::MotionMode CharacterBody2D::get_motion_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_motion_mode")._native_ptr(), 1160151236);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CharacterBody2D::MotionMode(0)));
	return (CharacterBody2D::MotionMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CharacterBody2D::set_platform_on_leave(CharacterBody2D::PlatformOnLeave p_on_leave_apply_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("set_platform_on_leave")._native_ptr(), 2423324375);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_on_leave_apply_velocity_encoded;
	PtrToArg<int64_t>::encode(p_on_leave_apply_velocity, &p_on_leave_apply_velocity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_on_leave_apply_velocity_encoded);
}

CharacterBody2D::PlatformOnLeave CharacterBody2D::get_platform_on_leave() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_platform_on_leave")._native_ptr(), 4054324341);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CharacterBody2D::PlatformOnLeave(0)));
	return (CharacterBody2D::PlatformOnLeave)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_on_floor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_on_floor")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_on_floor_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_on_floor_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_on_ceiling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_on_ceiling")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_on_ceiling_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_on_ceiling_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_on_wall() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_on_wall")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CharacterBody2D::is_on_wall_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("is_on_wall_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Vector2 CharacterBody2D::get_floor_normal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_floor_normal")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 CharacterBody2D::get_wall_normal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_wall_normal")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 CharacterBody2D::get_last_motion() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_last_motion")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 CharacterBody2D::get_position_delta() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_position_delta")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 CharacterBody2D::get_real_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_real_velocity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float CharacterBody2D::get_floor_angle(const Vector2 &p_up_direction) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_floor_angle")._native_ptr(), 2841063350);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_up_direction);
}

Vector2 CharacterBody2D::get_platform_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_platform_velocity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

int32_t CharacterBody2D::get_slide_collision_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_slide_collision_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<KinematicCollision2D> CharacterBody2D::get_slide_collision(int32_t p_slide_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_slide_collision")._native_ptr(), 860659811);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<KinematicCollision2D>()));
	int64_t p_slide_idx_encoded;
	PtrToArg<int64_t>::encode(p_slide_idx, &p_slide_idx_encoded);
	return Ref<KinematicCollision2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<KinematicCollision2D>(_gde_method_bind, _owner, &p_slide_idx_encoded));
}

Ref<KinematicCollision2D> CharacterBody2D::get_last_slide_collision() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CharacterBody2D::get_class_static()._native_ptr(), StringName("get_last_slide_collision")._native_ptr(), 2161834755);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<KinematicCollision2D>()));
	return Ref<KinematicCollision2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<KinematicCollision2D>(_gde_method_bind, _owner));
}

} // namespace godot
