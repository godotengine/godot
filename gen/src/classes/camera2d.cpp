/**************************************************************************/
/*  camera2d.cpp                                                          */
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

#include <godot_cpp/classes/camera2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>

namespace godot {

void Camera2D::set_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 Camera2D::get_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Camera2D::set_anchor_mode(Camera2D::AnchorMode p_anchor_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_anchor_mode")._native_ptr(), 2050398218);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_anchor_mode_encoded;
	PtrToArg<int64_t>::encode(p_anchor_mode, &p_anchor_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anchor_mode_encoded);
}

Camera2D::AnchorMode Camera2D::get_anchor_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_anchor_mode")._native_ptr(), 155978067);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Camera2D::AnchorMode(0)));
	return (Camera2D::AnchorMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Camera2D::set_ignore_rotation(bool p_ignore) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_ignore_rotation")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_ignore_encoded;
	PtrToArg<bool>::encode(p_ignore, &p_ignore_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ignore_encoded);
}

bool Camera2D::is_ignoring_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_ignoring_rotation")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_process_callback(Camera2D::Camera2DProcessCallback p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_process_callback")._native_ptr(), 4201947462);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Camera2D::Camera2DProcessCallback Camera2D::get_process_callback() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_process_callback")._native_ptr(), 2325344499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Camera2D::Camera2DProcessCallback(0)));
	return (Camera2D::Camera2DProcessCallback)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Camera2D::set_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Camera2D::is_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::make_current() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("make_current")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool Camera2D::is_current() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_current")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_limit_enabled(bool p_limit_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_limit_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_limit_enabled_encoded;
	PtrToArg<bool>::encode(p_limit_enabled, &p_limit_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_limit_enabled_encoded);
}

bool Camera2D::is_limit_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_limit_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_limit(Side p_margin, int32_t p_limit) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_limit")._native_ptr(), 437707142);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	int64_t p_limit_encoded;
	PtrToArg<int64_t>::encode(p_limit, &p_limit_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded, &p_limit_encoded);
}

int32_t Camera2D::get_limit(Side p_margin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_limit")._native_ptr(), 1983885014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_margin_encoded);
}

void Camera2D::set_limit_smoothing_enabled(bool p_limit_smoothing_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_limit_smoothing_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_limit_smoothing_enabled_encoded;
	PtrToArg<bool>::encode(p_limit_smoothing_enabled, &p_limit_smoothing_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_limit_smoothing_enabled_encoded);
}

bool Camera2D::is_limit_smoothing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_limit_smoothing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_drag_vertical_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_drag_vertical_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Camera2D::is_drag_vertical_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_drag_vertical_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_drag_horizontal_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_drag_horizontal_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Camera2D::is_drag_horizontal_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_drag_horizontal_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_drag_vertical_offset(float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_drag_vertical_offset")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

float Camera2D::get_drag_vertical_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_drag_vertical_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera2D::set_drag_horizontal_offset(float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_drag_horizontal_offset")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

float Camera2D::get_drag_horizontal_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_drag_horizontal_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera2D::set_drag_margin(Side p_margin, float p_drag_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_drag_margin")._native_ptr(), 4290182280);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	double p_drag_margin_encoded;
	PtrToArg<double>::encode(p_drag_margin, &p_drag_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded, &p_drag_margin_encoded);
}

float Camera2D::get_drag_margin(Side p_margin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_drag_margin")._native_ptr(), 2869120046);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_margin_encoded);
}

Vector2 Camera2D::get_target_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_target_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Camera2D::get_screen_center_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_screen_center_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float Camera2D::get_screen_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_screen_rotation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera2D::set_zoom(const Vector2 &p_zoom) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_zoom")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zoom);
}

Vector2 Camera2D::get_zoom() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_zoom")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Camera2D::set_custom_viewport(Node *p_viewport) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_custom_viewport")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_viewport != nullptr ? &p_viewport->_owner : nullptr));
}

Node *Camera2D::get_custom_viewport() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_custom_viewport")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

void Camera2D::set_position_smoothing_speed(float p_position_smoothing_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_position_smoothing_speed")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_position_smoothing_speed_encoded;
	PtrToArg<double>::encode(p_position_smoothing_speed, &p_position_smoothing_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position_smoothing_speed_encoded);
}

float Camera2D::get_position_smoothing_speed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_position_smoothing_speed")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera2D::set_position_smoothing_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_position_smoothing_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Camera2D::is_position_smoothing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_position_smoothing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_rotation_smoothing_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_rotation_smoothing_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Camera2D::is_rotation_smoothing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_rotation_smoothing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_rotation_smoothing_speed(float p_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_rotation_smoothing_speed")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_speed_encoded;
	PtrToArg<double>::encode(p_speed, &p_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_speed_encoded);
}

float Camera2D::get_rotation_smoothing_speed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("get_rotation_smoothing_speed")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera2D::force_update_scroll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("force_update_scroll")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Camera2D::reset_smoothing() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("reset_smoothing")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Camera2D::align() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("align")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Camera2D::set_screen_drawing_enabled(bool p_screen_drawing_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_screen_drawing_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_screen_drawing_enabled_encoded;
	PtrToArg<bool>::encode(p_screen_drawing_enabled, &p_screen_drawing_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_screen_drawing_enabled_encoded);
}

bool Camera2D::is_screen_drawing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_screen_drawing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_limit_drawing_enabled(bool p_limit_drawing_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_limit_drawing_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_limit_drawing_enabled_encoded;
	PtrToArg<bool>::encode(p_limit_drawing_enabled, &p_limit_drawing_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_limit_drawing_enabled_encoded);
}

bool Camera2D::is_limit_drawing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_limit_drawing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Camera2D::set_margin_drawing_enabled(bool p_margin_drawing_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("set_margin_drawing_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_margin_drawing_enabled_encoded;
	PtrToArg<bool>::encode(p_margin_drawing_enabled, &p_margin_drawing_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_drawing_enabled_encoded);
}

bool Camera2D::is_margin_drawing_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera2D::get_class_static()._native_ptr(), StringName("is_margin_drawing_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
