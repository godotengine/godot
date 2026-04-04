/**************************************************************************/
/*  canvas_item.cpp                                                       */
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

#include <godot_cpp/classes/canvas_item.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/canvas_layer.hpp>
#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/classes/world2d.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

RID CanvasItem::get_canvas_item() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_canvas_item")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void CanvasItem::set_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool CanvasItem::is_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool CanvasItem::is_visible_in_tree() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_visible_in_tree")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::show() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("show")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void CanvasItem::hide() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("hide")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void CanvasItem::queue_redraw() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("queue_redraw")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void CanvasItem::move_to_front() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("move_to_front")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void CanvasItem::set_as_top_level(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_as_top_level")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CanvasItem::is_set_as_top_level() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_set_as_top_level")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_light_mask(int32_t p_light_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_light_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_light_mask_encoded;
	PtrToArg<int64_t>::encode(p_light_mask, &p_light_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_mask_encoded);
}

int32_t CanvasItem::get_light_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_light_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_modulate(const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_modulate")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_modulate);
}

Color CanvasItem::get_modulate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_modulate")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void CanvasItem::set_self_modulate(const Color &p_self_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_self_modulate")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_self_modulate);
}

Color CanvasItem::get_self_modulate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_self_modulate")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void CanvasItem::set_z_index(int32_t p_z_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_z_index")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_z_index_encoded;
	PtrToArg<int64_t>::encode(p_z_index, &p_z_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_z_index_encoded);
}

int32_t CanvasItem::get_z_index() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_z_index")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_z_as_relative(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_z_as_relative")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CanvasItem::is_z_relative() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_z_relative")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_y_sort_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_y_sort_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool CanvasItem::is_y_sort_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_y_sort_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_draw_behind_parent(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_draw_behind_parent")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CanvasItem::is_draw_behind_parent_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_draw_behind_parent_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::draw_line(const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_line")._native_ptr(), 1562330099);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from, &p_to, &p_color, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_dashed_line(const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, float p_width, float p_dash, bool p_aligned, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_dashed_line")._native_ptr(), 3653831622);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	double p_dash_encoded;
	PtrToArg<double>::encode(p_dash, &p_dash_encoded);
	int8_t p_aligned_encoded;
	PtrToArg<bool>::encode(p_aligned, &p_aligned_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from, &p_to, &p_color, &p_width_encoded, &p_dash_encoded, &p_aligned_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_polyline(const PackedVector2Array &p_points, const Color &p_color, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_polyline")._native_ptr(), 3797364428);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_color, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_polyline_colors(const PackedVector2Array &p_points, const PackedColorArray &p_colors, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_polyline_colors")._native_ptr(), 2311979562);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_colors, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_ellipse_arc(const Vector2 &p_center, float p_major, float p_minor, float p_start_angle, float p_end_angle, int32_t p_point_count, const Color &p_color, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_ellipse_arc")._native_ptr(), 936174114);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_major_encoded;
	PtrToArg<double>::encode(p_major, &p_major_encoded);
	double p_minor_encoded;
	PtrToArg<double>::encode(p_minor, &p_minor_encoded);
	double p_start_angle_encoded;
	PtrToArg<double>::encode(p_start_angle, &p_start_angle_encoded);
	double p_end_angle_encoded;
	PtrToArg<double>::encode(p_end_angle, &p_end_angle_encoded);
	int64_t p_point_count_encoded;
	PtrToArg<int64_t>::encode(p_point_count, &p_point_count_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_center, &p_major_encoded, &p_minor_encoded, &p_start_angle_encoded, &p_end_angle_encoded, &p_point_count_encoded, &p_color, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_arc(const Vector2 &p_center, float p_radius, float p_start_angle, float p_end_angle, int32_t p_point_count, const Color &p_color, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_arc")._native_ptr(), 4140652635);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	double p_start_angle_encoded;
	PtrToArg<double>::encode(p_start_angle, &p_start_angle_encoded);
	double p_end_angle_encoded;
	PtrToArg<double>::encode(p_end_angle, &p_end_angle_encoded);
	int64_t p_point_count_encoded;
	PtrToArg<int64_t>::encode(p_point_count, &p_point_count_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_center, &p_radius_encoded, &p_start_angle_encoded, &p_end_angle_encoded, &p_point_count_encoded, &p_color, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_multiline(const PackedVector2Array &p_points, const Color &p_color, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_multiline")._native_ptr(), 3797364428);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_color, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_multiline_colors(const PackedVector2Array &p_points, const PackedColorArray &p_colors, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_multiline_colors")._native_ptr(), 2311979562);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_colors, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_rect(const Rect2 &p_rect, const Color &p_color, bool p_filled, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_rect")._native_ptr(), 2773573813);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_filled_encoded;
	PtrToArg<bool>::encode(p_filled, &p_filled_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rect, &p_color, &p_filled_encoded, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_circle(const Vector2 &p_position, float p_radius, const Color &p_color, bool p_filled, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_circle")._native_ptr(), 3153026596);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	int8_t p_filled_encoded;
	PtrToArg<bool>::encode(p_filled, &p_filled_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_radius_encoded, &p_color, &p_filled_encoded, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_ellipse(const Vector2 &p_position, float p_major, float p_minor, const Color &p_color, bool p_filled, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_ellipse")._native_ptr(), 3790774806);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_major_encoded;
	PtrToArg<double>::encode(p_major, &p_major_encoded);
	double p_minor_encoded;
	PtrToArg<double>::encode(p_minor, &p_minor_encoded);
	int8_t p_filled_encoded;
	PtrToArg<bool>::encode(p_filled, &p_filled_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_major_encoded, &p_minor_encoded, &p_color, &p_filled_encoded, &p_width_encoded, &p_antialiased_encoded);
}

void CanvasItem::draw_texture(const Ref<Texture2D> &p_texture, const Vector2 &p_position, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_texture")._native_ptr(), 520200117);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_position, &p_modulate);
}

void CanvasItem::draw_texture_rect(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_texture_rect")._native_ptr(), 3832805018);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_tile_encoded;
	PtrToArg<bool>::encode(p_tile, &p_tile_encoded);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_rect, &p_tile_encoded, &p_modulate, &p_transpose_encoded);
}

void CanvasItem::draw_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_texture_rect_region")._native_ptr(), 3883821411);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	int8_t p_clip_uv_encoded;
	PtrToArg<bool>::encode(p_clip_uv, &p_clip_uv_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_rect, &p_src_rect, &p_modulate, &p_transpose_encoded, &p_clip_uv_encoded);
}

void CanvasItem::draw_msdf_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, double p_outline, double p_pixel_range, double p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_msdf_texture_rect_region")._native_ptr(), 4219163252);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_outline_encoded;
	PtrToArg<double>::encode(p_outline, &p_outline_encoded);
	double p_pixel_range_encoded;
	PtrToArg<double>::encode(p_pixel_range, &p_pixel_range_encoded);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_rect, &p_src_rect, &p_modulate, &p_outline_encoded, &p_pixel_range_encoded, &p_scale_encoded);
}

void CanvasItem::draw_lcd_texture_rect_region(const Ref<Texture2D> &p_texture, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_lcd_texture_rect_region")._native_ptr(), 3212350954);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_rect, &p_src_rect, &p_modulate);
}

void CanvasItem::draw_style_box(const Ref<StyleBox> &p_style_box, const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_style_box")._native_ptr(), 388176283);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_style_box != nullptr ? &p_style_box->_owner : nullptr), &p_rect);
}

void CanvasItem::draw_primitive(const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_primitive")._native_ptr(), 3288481815);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_colors, &p_uvs, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void CanvasItem::draw_polygon(const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_polygon")._native_ptr(), 974537912);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_colors, &p_uvs, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void CanvasItem::draw_colored_polygon(const PackedVector2Array &p_points, const Color &p_color, const PackedVector2Array &p_uvs, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_colored_polygon")._native_ptr(), 15245644);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_points, &p_color, &p_uvs, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void CanvasItem::draw_string(const Ref<Font> &p_font, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, const Color &p_modulate, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_string")._native_ptr(), 719605945);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_modulate, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void CanvasItem::draw_multiline_string(const Ref<Font> &p_font, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_max_lines, const Color &p_modulate, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_multiline_string")._native_ptr(), 2341488182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_max_lines_encoded;
	PtrToArg<int64_t>::encode(p_max_lines, &p_max_lines_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_max_lines_encoded, &p_modulate, &p_brk_flags, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void CanvasItem::draw_string_outline(const Ref<Font> &p_font, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_size, const Color &p_modulate, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_string_outline")._native_ptr(), 707403449);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_size_encoded, &p_modulate, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void CanvasItem::draw_multiline_string_outline(const Ref<Font> &p_font, const Vector2 &p_pos, const String &p_text, HorizontalAlignment p_alignment, float p_width, int32_t p_font_size, int32_t p_max_lines, int32_t p_size, const Color &p_modulate, BitField<TextServer::LineBreakFlag> p_brk_flags, BitField<TextServer::JustificationFlag> p_justification_flags, TextServer::Direction p_direction, TextServer::Orientation p_orientation, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_multiline_string_outline")._native_ptr(), 3050414441);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_max_lines_encoded;
	PtrToArg<int64_t>::encode(p_max_lines, &p_max_lines_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_pos, &p_text, &p_alignment_encoded, &p_width_encoded, &p_font_size_encoded, &p_max_lines_encoded, &p_size_encoded, &p_modulate, &p_brk_flags, &p_justification_flags, &p_direction_encoded, &p_orientation_encoded, &p_oversampling_encoded);
}

void CanvasItem::draw_char(const Ref<Font> &p_font, const Vector2 &p_pos, const String &p_char, int32_t p_font_size, const Color &p_modulate, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_char")._native_ptr(), 1336210142);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_pos, &p_char, &p_font_size_encoded, &p_modulate, &p_oversampling_encoded);
}

void CanvasItem::draw_char_outline(const Ref<Font> &p_font, const Vector2 &p_pos, const String &p_char, int32_t p_font_size, int32_t p_size, const Color &p_modulate, float p_oversampling) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_char_outline")._native_ptr(), 1846384149);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_font != nullptr ? &p_font->_owner : nullptr), &p_pos, &p_char, &p_font_size_encoded, &p_size_encoded, &p_modulate, &p_oversampling_encoded);
}

void CanvasItem::draw_mesh(const Ref<Mesh> &p_mesh, const Ref<Texture2D> &p_texture, const Transform2D &p_transform, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_mesh")._native_ptr(), 153818295);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr), (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_transform, &p_modulate);
}

void CanvasItem::draw_multimesh(const Ref<MultiMesh> &p_multimesh, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_multimesh")._native_ptr(), 937992368);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_multimesh != nullptr ? &p_multimesh->_owner : nullptr), (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void CanvasItem::draw_set_transform(const Vector2 &p_position, float p_rotation, const Vector2 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_set_transform")._native_ptr(), 288975085);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_rotation_encoded;
	PtrToArg<double>::encode(p_rotation, &p_rotation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_rotation_encoded, &p_scale);
}

void CanvasItem::draw_set_transform_matrix(const Transform2D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_set_transform_matrix")._native_ptr(), 2761652528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform);
}

void CanvasItem::draw_animation_slice(double p_animation_length, double p_slice_begin, double p_slice_end, double p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_animation_slice")._native_ptr(), 3112831842);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_animation_length_encoded;
	PtrToArg<double>::encode(p_animation_length, &p_animation_length_encoded);
	double p_slice_begin_encoded;
	PtrToArg<double>::encode(p_slice_begin, &p_slice_begin_encoded);
	double p_slice_end_encoded;
	PtrToArg<double>::encode(p_slice_end, &p_slice_end_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_animation_length_encoded, &p_slice_begin_encoded, &p_slice_end_encoded, &p_offset_encoded);
}

void CanvasItem::draw_end_animation() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("draw_end_animation")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Transform2D CanvasItem::get_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D CanvasItem::get_global_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_global_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D CanvasItem::get_global_transform_with_canvas() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_global_transform_with_canvas")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D CanvasItem::get_viewport_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_viewport_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Rect2 CanvasItem::get_viewport_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_viewport_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

Transform2D CanvasItem::get_canvas_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_canvas_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D CanvasItem::get_screen_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_screen_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Vector2 CanvasItem::get_local_mouse_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_local_mouse_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 CanvasItem::get_global_mouse_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_global_mouse_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

RID CanvasItem::get_canvas() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_canvas")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

CanvasLayer *CanvasItem::get_canvas_layer_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_canvas_layer_node")._native_ptr(), 2602762519);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<CanvasLayer>(_gde_method_bind, _owner);
}

Ref<World2D> CanvasItem::get_world_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_world_2d")._native_ptr(), 2339128592);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<World2D>()));
	return Ref<World2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<World2D>(_gde_method_bind, _owner));
}

void CanvasItem::set_material(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_material")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> CanvasItem::get_material() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_material")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

void CanvasItem::set_instance_shader_parameter(const StringName &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_instance_shader_parameter")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

Variant CanvasItem::get_instance_shader_parameter(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_instance_shader_parameter")._native_ptr(), 2760726917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

void CanvasItem::set_use_parent_material(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_use_parent_material")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CanvasItem::get_use_parent_material() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_use_parent_material")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_notify_local_transform(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_notify_local_transform")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CanvasItem::is_local_transform_notification_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_local_transform_notification_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_notify_transform(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_notify_transform")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool CanvasItem::is_transform_notification_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("is_transform_notification_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItem::force_update_transform() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("force_update_transform")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2 CanvasItem::make_canvas_position_local(const Vector2 &p_viewport_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("make_canvas_position_local")._native_ptr(), 2656412154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_viewport_point);
}

Ref<InputEvent> CanvasItem::make_input_local(const Ref<InputEvent> &p_event) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("make_input_local")._native_ptr(), 811130057);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<InputEvent>()));
	return Ref<InputEvent>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<InputEvent>(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr)));
}

void CanvasItem::set_visibility_layer(uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_visibility_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

uint32_t CanvasItem::get_visibility_layer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_visibility_layer")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_visibility_layer_bit(uint32_t p_layer, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_visibility_layer_bit")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_enabled_encoded);
}

bool CanvasItem::get_visibility_layer_bit(uint32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_visibility_layer_bit")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void CanvasItem::set_texture_filter(CanvasItem::TextureFilter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_texture_filter")._native_ptr(), 1037999706);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

CanvasItem::TextureFilter CanvasItem::get_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_texture_filter")._native_ptr(), 121960042);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItem::TextureFilter(0)));
	return (CanvasItem::TextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_texture_repeat(CanvasItem::TextureRepeat p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_texture_repeat")._native_ptr(), 1716472974);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

CanvasItem::TextureRepeat CanvasItem::get_texture_repeat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_texture_repeat")._native_ptr(), 2667158319);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItem::TextureRepeat(0)));
	return (CanvasItem::TextureRepeat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItem::set_clip_children_mode(CanvasItem::ClipChildrenMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("set_clip_children_mode")._native_ptr(), 1319393776);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

CanvasItem::ClipChildrenMode CanvasItem::get_clip_children_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItem::get_class_static()._native_ptr(), StringName("get_clip_children_mode")._native_ptr(), 3581808349);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItem::ClipChildrenMode(0)));
	return (CanvasItem::ClipChildrenMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItem::_draw() {}

} // namespace godot
