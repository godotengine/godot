/**************************************************************************/
/*  style_box_flat.cpp                                                    */
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

#include <godot_cpp/classes/style_box_flat.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void StyleBoxFlat::set_bg_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_bg_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color StyleBoxFlat::get_bg_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_bg_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_border_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_border_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color StyleBoxFlat::get_border_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_border_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_border_width_all(int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_border_width_all")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

int32_t StyleBoxFlat::get_border_width_min() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_border_width_min")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_border_width(Side p_margin, int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_border_width")._native_ptr(), 437707142);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded, &p_width_encoded);
}

int32_t StyleBoxFlat::get_border_width(Side p_margin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_border_width")._native_ptr(), 1983885014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_margin_encoded);
}

void StyleBoxFlat::set_border_blend(bool p_blend) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_border_blend")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_blend_encoded;
	PtrToArg<bool>::encode(p_blend, &p_blend_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_blend_encoded);
}

bool StyleBoxFlat::get_border_blend() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_border_blend")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_corner_radius_all(int32_t p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_corner_radius_all")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_radius_encoded;
	PtrToArg<int64_t>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

void StyleBoxFlat::set_corner_radius(Corner p_corner, int32_t p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_corner_radius")._native_ptr(), 2696158768);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_corner_encoded;
	PtrToArg<int64_t>::encode(p_corner, &p_corner_encoded);
	int64_t p_radius_encoded;
	PtrToArg<int64_t>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_corner_encoded, &p_radius_encoded);
}

int32_t StyleBoxFlat::get_corner_radius(Corner p_corner) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_corner_radius")._native_ptr(), 3982397690);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_corner_encoded;
	PtrToArg<int64_t>::encode(p_corner, &p_corner_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_corner_encoded);
}

void StyleBoxFlat::set_expand_margin(Side p_margin, float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_expand_margin")._native_ptr(), 4290182280);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded, &p_size_encoded);
}

void StyleBoxFlat::set_expand_margin_all(float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_expand_margin_all")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

float StyleBoxFlat::get_expand_margin(Side p_margin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_expand_margin")._native_ptr(), 2869120046);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_margin_encoded);
}

void StyleBoxFlat::set_draw_center(bool p_draw_center) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_draw_center")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_center_encoded;
	PtrToArg<bool>::encode(p_draw_center, &p_draw_center_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_center_encoded);
}

bool StyleBoxFlat::is_draw_center_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("is_draw_center_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_skew(const Vector2 &p_skew) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_skew")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skew);
}

Vector2 StyleBoxFlat::get_skew() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_skew")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_shadow_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_shadow_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color StyleBoxFlat::get_shadow_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_shadow_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_shadow_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_shadow_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t StyleBoxFlat::get_shadow_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_shadow_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_shadow_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_shadow_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 StyleBoxFlat::get_shadow_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_shadow_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_anti_aliased(bool p_anti_aliased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_anti_aliased")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_anti_aliased_encoded;
	PtrToArg<bool>::encode(p_anti_aliased, &p_anti_aliased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anti_aliased_encoded);
}

bool StyleBoxFlat::is_anti_aliased() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("is_anti_aliased")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_aa_size(float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_aa_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

float StyleBoxFlat::get_aa_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_aa_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void StyleBoxFlat::set_corner_detail(int32_t p_detail) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("set_corner_detail")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_detail_encoded;
	PtrToArg<int64_t>::encode(p_detail, &p_detail_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_encoded);
}

int32_t StyleBoxFlat::get_corner_detail() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StyleBoxFlat::get_class_static()._native_ptr(), StringName("get_corner_detail")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
