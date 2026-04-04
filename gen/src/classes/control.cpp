/**************************************************************************/
/*  control.cpp                                                           */
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

#include <godot_cpp/classes/control.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

void Control::accept_event() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("accept_event")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2 Control::get_minimum_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_minimum_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_combined_minimum_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_combined_minimum_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Control::set_anchors_preset(Control::LayoutPreset p_preset, bool p_keep_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_anchors_preset")._native_ptr(), 509135270);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_preset_encoded;
	PtrToArg<int64_t>::encode(p_preset, &p_preset_encoded);
	int8_t p_keep_offsets_encoded;
	PtrToArg<bool>::encode(p_keep_offsets, &p_keep_offsets_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_preset_encoded, &p_keep_offsets_encoded);
}

void Control::set_offsets_preset(Control::LayoutPreset p_preset, Control::LayoutPresetMode p_resize_mode, int32_t p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_offsets_preset")._native_ptr(), 3724524307);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_preset_encoded;
	PtrToArg<int64_t>::encode(p_preset, &p_preset_encoded);
	int64_t p_resize_mode_encoded;
	PtrToArg<int64_t>::encode(p_resize_mode, &p_resize_mode_encoded);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_preset_encoded, &p_resize_mode_encoded, &p_margin_encoded);
}

void Control::set_anchors_and_offsets_preset(Control::LayoutPreset p_preset, Control::LayoutPresetMode p_resize_mode, int32_t p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_anchors_and_offsets_preset")._native_ptr(), 3724524307);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_preset_encoded;
	PtrToArg<int64_t>::encode(p_preset, &p_preset_encoded);
	int64_t p_resize_mode_encoded;
	PtrToArg<int64_t>::encode(p_resize_mode, &p_resize_mode_encoded);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_preset_encoded, &p_resize_mode_encoded, &p_margin_encoded);
}

void Control::set_anchor(Side p_side, float p_anchor, bool p_keep_offset, bool p_push_opposite_anchor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_anchor")._native_ptr(), 2302782885);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	double p_anchor_encoded;
	PtrToArg<double>::encode(p_anchor, &p_anchor_encoded);
	int8_t p_keep_offset_encoded;
	PtrToArg<bool>::encode(p_keep_offset, &p_keep_offset_encoded);
	int8_t p_push_opposite_anchor_encoded;
	PtrToArg<bool>::encode(p_push_opposite_anchor, &p_push_opposite_anchor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_side_encoded, &p_anchor_encoded, &p_keep_offset_encoded, &p_push_opposite_anchor_encoded);
}

float Control::get_anchor(Side p_side) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_anchor")._native_ptr(), 2869120046);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_side_encoded);
}

void Control::set_offset(Side p_side, float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 4290182280);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_side_encoded, &p_offset_encoded);
}

float Control::get_offset(Side p_offset) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 2869120046);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_offset_encoded);
}

void Control::set_anchor_and_offset(Side p_side, float p_anchor, float p_offset, bool p_push_opposite_anchor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_anchor_and_offset")._native_ptr(), 4031722181);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	double p_anchor_encoded;
	PtrToArg<double>::encode(p_anchor, &p_anchor_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	int8_t p_push_opposite_anchor_encoded;
	PtrToArg<bool>::encode(p_push_opposite_anchor, &p_push_opposite_anchor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_side_encoded, &p_anchor_encoded, &p_offset_encoded, &p_push_opposite_anchor_encoded);
}

void Control::set_begin(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_begin")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

void Control::set_end(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_end")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

void Control::set_position(const Vector2 &p_position, bool p_keep_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_position")._native_ptr(), 2436320129);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_offsets_encoded;
	PtrToArg<bool>::encode(p_keep_offsets, &p_keep_offsets_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_keep_offsets_encoded);
}

void Control::set_size(const Vector2 &p_size, bool p_keep_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 2436320129);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_offsets_encoded;
	PtrToArg<bool>::encode(p_keep_offsets, &p_keep_offsets_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size, &p_keep_offsets_encoded);
}

void Control::reset_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("reset_size")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::set_custom_minimum_size(const Vector2 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_custom_minimum_size")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

void Control::set_global_position(const Vector2 &p_position, bool p_keep_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_global_position")._native_ptr(), 2436320129);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_keep_offsets_encoded;
	PtrToArg<bool>::encode(p_keep_offsets, &p_keep_offsets_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_keep_offsets_encoded);
}

void Control::set_rotation(float p_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_rotation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radians_encoded;
	PtrToArg<double>::encode(p_radians, &p_radians_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radians_encoded);
}

void Control::set_rotation_degrees(float p_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_rotation_degrees")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_degrees_encoded;
	PtrToArg<double>::encode(p_degrees, &p_degrees_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_degrees_encoded);
}

void Control::set_scale(const Vector2 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_scale")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

void Control::set_pivot_offset(const Vector2 &p_pivot_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_pivot_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pivot_offset);
}

void Control::set_pivot_offset_ratio(const Vector2 &p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_pivot_offset_ratio")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio);
}

Vector2 Control::get_begin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_begin")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_end")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float Control::get_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_rotation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Control::get_rotation_degrees() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_rotation_degrees")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Vector2 Control::get_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_scale")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_pivot_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_pivot_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_pivot_offset_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_pivot_offset_ratio")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_combined_pivot_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_combined_pivot_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_custom_minimum_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_custom_minimum_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_parent_area_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_parent_area_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_global_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_global_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Control::get_screen_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_screen_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Rect2 Control::get_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

Rect2 Control::get_global_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_global_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

void Control::set_focus_mode(Control::FocusMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_focus_mode")._native_ptr(), 3232914922);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Control::FocusMode Control::get_focus_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_focus_mode")._native_ptr(), 2132829277);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::FocusMode(0)));
	return (Control::FocusMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Control::FocusMode Control::get_focus_mode_with_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_focus_mode_with_override")._native_ptr(), 2132829277);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::FocusMode(0)));
	return (Control::FocusMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_focus_behavior_recursive(Control::FocusBehaviorRecursive p_focus_behavior_recursive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_focus_behavior_recursive")._native_ptr(), 4256832521);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_focus_behavior_recursive_encoded;
	PtrToArg<int64_t>::encode(p_focus_behavior_recursive, &p_focus_behavior_recursive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_focus_behavior_recursive_encoded);
}

Control::FocusBehaviorRecursive Control::get_focus_behavior_recursive() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_focus_behavior_recursive")._native_ptr(), 2435707181);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::FocusBehaviorRecursive(0)));
	return (Control::FocusBehaviorRecursive)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Control::has_focus(bool p_ignore_hidden_focus) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_focus")._native_ptr(), 3302206351);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_ignore_hidden_focus_encoded;
	PtrToArg<bool>::encode(p_ignore_hidden_focus, &p_ignore_hidden_focus_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_ignore_hidden_focus_encoded);
}

void Control::grab_focus(bool p_hide_focus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("grab_focus")._native_ptr(), 107499316);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hide_focus_encoded;
	PtrToArg<bool>::encode(p_hide_focus, &p_hide_focus_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hide_focus_encoded);
}

void Control::release_focus() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("release_focus")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Control *Control::find_prev_valid_focus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("find_prev_valid_focus")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

Control *Control::find_next_valid_focus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("find_next_valid_focus")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

Control *Control::find_valid_focus_neighbor(Side p_side) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("find_valid_focus_neighbor")._native_ptr(), 1543910170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner, &p_side_encoded);
}

void Control::set_h_size_flags(BitField<Control::SizeFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_h_size_flags")._native_ptr(), 394851643);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags);
}

BitField<Control::SizeFlags> Control::get_h_size_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_h_size_flags")._native_ptr(), 3781367401);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<Control::SizeFlags>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_stretch_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_stretch_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float Control::get_stretch_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_stretch_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Control::set_v_size_flags(BitField<Control::SizeFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_v_size_flags")._native_ptr(), 394851643);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags);
}

BitField<Control::SizeFlags> Control::get_v_size_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_v_size_flags")._native_ptr(), 3781367401);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<Control::SizeFlags>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_theme(const Ref<Theme> &p_theme) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_theme")._native_ptr(), 2326690814);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_theme != nullptr ? &p_theme->_owner : nullptr));
}

Ref<Theme> Control::get_theme() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme")._native_ptr(), 3846893731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Theme>()));
	return Ref<Theme>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Theme>(_gde_method_bind, _owner));
}

void Control::set_theme_type_variation(const StringName &p_theme_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_theme_type_variation")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_theme_type);
}

StringName Control::get_theme_type_variation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_type_variation")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void Control::begin_bulk_theme_override() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("begin_bulk_theme_override")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::end_bulk_theme_override() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("end_bulk_theme_override")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::add_theme_icon_override(const StringName &p_name, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("add_theme_icon_override")._native_ptr(), 1373065600);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void Control::add_theme_stylebox_override(const StringName &p_name, const Ref<StyleBox> &p_stylebox) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("add_theme_stylebox_override")._native_ptr(), 4188838905);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_stylebox != nullptr ? &p_stylebox->_owner : nullptr));
}

void Control::add_theme_font_override(const StringName &p_name, const Ref<Font> &p_font) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("add_theme_font_override")._native_ptr(), 3518018674);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_font != nullptr ? &p_font->_owner : nullptr));
}

void Control::add_theme_font_size_override(const StringName &p_name, int32_t p_font_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("add_theme_font_size_override")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_font_size_encoded);
}

void Control::add_theme_color_override(const StringName &p_name, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("add_theme_color_override")._native_ptr(), 4260178595);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_color);
}

void Control::add_theme_constant_override(const StringName &p_name, int32_t p_constant) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("add_theme_constant_override")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_constant_encoded;
	PtrToArg<int64_t>::encode(p_constant, &p_constant_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_constant_encoded);
}

void Control::remove_theme_icon_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("remove_theme_icon_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Control::remove_theme_stylebox_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("remove_theme_stylebox_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Control::remove_theme_font_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("remove_theme_font_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Control::remove_theme_font_size_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("remove_theme_font_size_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Control::remove_theme_color_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("remove_theme_color_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Control::remove_theme_constant_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("remove_theme_constant_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

Ref<Texture2D> Control::get_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_icon")._native_ptr(), 3163973443);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_name, &p_theme_type));
}

Ref<StyleBox> Control::get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_stylebox")._native_ptr(), 604739069);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StyleBox>()));
	return Ref<StyleBox>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StyleBox>(_gde_method_bind, _owner, &p_name, &p_theme_type));
}

Ref<Font> Control::get_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_font")._native_ptr(), 2826986490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner, &p_name, &p_theme_type));
}

int32_t Control::get_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_font_size")._native_ptr(), 1327056374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

Color Control::get_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_color")._native_ptr(), 2798751242);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

int32_t Control::get_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_constant")._native_ptr(), 1327056374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Control::has_theme_icon_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_icon_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Control::has_theme_stylebox_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_stylebox_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Control::has_theme_font_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_font_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Control::has_theme_font_size_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_font_size_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Control::has_theme_color_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_color_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Control::has_theme_constant_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_constant_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Control::has_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_icon")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Control::has_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_stylebox")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Control::has_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_font")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Control::has_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_font_size")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Control::has_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_color")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Control::has_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("has_theme_constant")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

float Control::get_theme_default_base_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_default_base_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Ref<Font> Control::get_theme_default_font() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_default_font")._native_ptr(), 3229501585);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner));
}

int32_t Control::get_theme_default_font_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_theme_default_font_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Control *Control::get_parent_control() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_parent_control")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

void Control::set_h_grow_direction(Control::GrowDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_h_grow_direction")._native_ptr(), 2022385301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Control::GrowDirection Control::get_h_grow_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_h_grow_direction")._native_ptr(), 3635610155);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::GrowDirection(0)));
	return (Control::GrowDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_v_grow_direction(Control::GrowDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_v_grow_direction")._native_ptr(), 2022385301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Control::GrowDirection Control::get_v_grow_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_v_grow_direction")._native_ptr(), 3635610155);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::GrowDirection(0)));
	return (Control::GrowDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_tooltip_auto_translate_mode(Node::AutoTranslateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_tooltip_auto_translate_mode")._native_ptr(), 776149714);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Node::AutoTranslateMode Control::get_tooltip_auto_translate_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_tooltip_auto_translate_mode")._native_ptr(), 2498906432);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::AutoTranslateMode(0)));
	return (Node::AutoTranslateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_tooltip_text(const String &p_hint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_tooltip_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hint);
}

String Control::get_tooltip_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_tooltip_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String Control::get_tooltip(const Vector2 &p_at_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_tooltip")._native_ptr(), 2895288280);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_at_position);
}

void Control::set_default_cursor_shape(Control::CursorShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_default_cursor_shape")._native_ptr(), 217062046);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

Control::CursorShape Control::get_default_cursor_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_default_cursor_shape")._native_ptr(), 2359535750);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::CursorShape(0)));
	return (Control::CursorShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Control::CursorShape Control::get_cursor_shape(const Vector2 &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_cursor_shape")._native_ptr(), 1395773853);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::CursorShape(0)));
	return (Control::CursorShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position);
}

void Control::set_focus_neighbor(Side p_side, const NodePath &p_neighbor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_focus_neighbor")._native_ptr(), 2024461774);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_side_encoded, &p_neighbor);
}

NodePath Control::get_focus_neighbor(Side p_side) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_focus_neighbor")._native_ptr(), 2757935761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_side_encoded;
	PtrToArg<int64_t>::encode(p_side, &p_side_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_side_encoded);
}

void Control::set_focus_next(const NodePath &p_next) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_focus_next")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_next);
}

NodePath Control::get_focus_next() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_focus_next")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void Control::set_focus_previous(const NodePath &p_previous) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_focus_previous")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_previous);
}

NodePath Control::get_focus_previous() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_focus_previous")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void Control::force_drag(const Variant &p_data, Control *p_preview) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("force_drag")._native_ptr(), 3191844692);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data, (p_preview != nullptr ? &p_preview->_owner : nullptr));
}

void Control::accessibility_drag() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("accessibility_drag")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::accessibility_drop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("accessibility_drop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::set_accessibility_name(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

String Control::get_accessibility_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Control::set_accessibility_description(const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_description")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_description);
}

String Control::get_accessibility_description() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_description")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Control::set_accessibility_live(DisplayServer::AccessibilityLiveMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_live")._native_ptr(), 1720261470);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

DisplayServer::AccessibilityLiveMode Control::get_accessibility_live() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_live")._native_ptr(), 3311037003);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (DisplayServer::AccessibilityLiveMode(0)));
	return (DisplayServer::AccessibilityLiveMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_accessibility_controls_nodes(const TypedArray<NodePath> &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_controls_nodes")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_path);
}

TypedArray<NodePath> Control::get_accessibility_controls_nodes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_controls_nodes")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<NodePath>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<NodePath>>(_gde_method_bind, _owner);
}

void Control::set_accessibility_described_by_nodes(const TypedArray<NodePath> &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_described_by_nodes")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_path);
}

TypedArray<NodePath> Control::get_accessibility_described_by_nodes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_described_by_nodes")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<NodePath>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<NodePath>>(_gde_method_bind, _owner);
}

void Control::set_accessibility_labeled_by_nodes(const TypedArray<NodePath> &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_labeled_by_nodes")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_path);
}

TypedArray<NodePath> Control::get_accessibility_labeled_by_nodes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_labeled_by_nodes")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<NodePath>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<NodePath>>(_gde_method_bind, _owner);
}

void Control::set_accessibility_flow_to_nodes(const TypedArray<NodePath> &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_accessibility_flow_to_nodes")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_path);
}

TypedArray<NodePath> Control::get_accessibility_flow_to_nodes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_accessibility_flow_to_nodes")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<NodePath>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<NodePath>>(_gde_method_bind, _owner);
}

void Control::set_mouse_filter(Control::MouseFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_mouse_filter")._native_ptr(), 3891156122);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_encoded);
}

Control::MouseFilter Control::get_mouse_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_mouse_filter")._native_ptr(), 1572545674);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::MouseFilter(0)));
	return (Control::MouseFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Control::MouseFilter Control::get_mouse_filter_with_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_mouse_filter_with_override")._native_ptr(), 1572545674);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::MouseFilter(0)));
	return (Control::MouseFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_mouse_behavior_recursive(Control::MouseBehaviorRecursive p_mouse_behavior_recursive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_mouse_behavior_recursive")._native_ptr(), 849284636);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mouse_behavior_recursive_encoded;
	PtrToArg<int64_t>::encode(p_mouse_behavior_recursive, &p_mouse_behavior_recursive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mouse_behavior_recursive_encoded);
}

Control::MouseBehaviorRecursive Control::get_mouse_behavior_recursive() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_mouse_behavior_recursive")._native_ptr(), 3779367402);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::MouseBehaviorRecursive(0)));
	return (Control::MouseBehaviorRecursive)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Control::set_force_pass_scroll_events(bool p_force_pass_scroll_events) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_force_pass_scroll_events")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_pass_scroll_events_encoded;
	PtrToArg<bool>::encode(p_force_pass_scroll_events, &p_force_pass_scroll_events_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force_pass_scroll_events_encoded);
}

bool Control::is_force_pass_scroll_events() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("is_force_pass_scroll_events")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Control::set_clip_contents(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_clip_contents")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Control::is_clipping_contents() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("is_clipping_contents")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Control::grab_click_focus() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("grab_click_focus")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::set_drag_forwarding(const Callable &p_drag_func, const Callable &p_can_drop_func, const Callable &p_drop_func) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_drag_forwarding")._native_ptr(), 1076571380);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_drag_func, &p_can_drop_func, &p_drop_func);
}

void Control::set_drag_preview(Control *p_control) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_drag_preview")._native_ptr(), 1496901182);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_control != nullptr ? &p_control->_owner : nullptr));
}

bool Control::is_drag_successful() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("is_drag_successful")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Control::warp_mouse(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("warp_mouse")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

void Control::set_shortcut_context(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_shortcut_context")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Node *Control::get_shortcut_context() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_shortcut_context")._native_ptr(), 3160264692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner);
}

void Control::update_minimum_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("update_minimum_size")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Control::set_layout_direction(Control::LayoutDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_layout_direction")._native_ptr(), 3310692370);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Control::LayoutDirection Control::get_layout_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("get_layout_direction")._native_ptr(), 1546772008);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::LayoutDirection(0)));
	return (Control::LayoutDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Control::is_layout_rtl() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("is_layout_rtl")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Control::set_auto_translate(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_auto_translate")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Control::is_auto_translating() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("is_auto_translating")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Control::set_localize_numeral_system(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("set_localize_numeral_system")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Control::is_localizing_numeral_system() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Control::get_class_static()._native_ptr(), StringName("is_localizing_numeral_system")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Control::_has_point(const Vector2 &p_point) const {
	return false;
}

TypedArray<Vector3i> Control::_structured_text_parser(const Array &p_args, const String &p_text) const {
	return TypedArray<Vector3i>();
}

Vector2 Control::_get_minimum_size() const {
	return Vector2();
}

String Control::_get_tooltip(const Vector2 &p_at_position) const {
	return String();
}

Variant Control::_get_drag_data(const Vector2 &p_at_position) {
	return Variant();
}

bool Control::_can_drop_data(const Vector2 &p_at_position, const Variant &p_data) const {
	return false;
}

void Control::_drop_data(const Vector2 &p_at_position, const Variant &p_data) {}

Object *Control::_make_custom_tooltip(const String &p_for_text) const {
	return nullptr;
}

String Control::_accessibility_get_contextual_info() const {
	return String();
}

String Control::_get_accessibility_container_name(Node *p_node) const {
	return String();
}

void Control::_gui_input(const Ref<InputEvent> &p_event) {}

} // namespace godot
