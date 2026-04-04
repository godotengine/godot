/**************************************************************************/
/*  sprite_base3d.cpp                                                     */
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

#include <godot_cpp/classes/sprite_base3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/triangle_mesh.hpp>

namespace godot {

void SpriteBase3D::set_centered(bool p_centered) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_centered")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_centered_encoded;
	PtrToArg<bool>::encode(p_centered, &p_centered_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_centered_encoded);
}

bool SpriteBase3D::is_centered() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("is_centered")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 SpriteBase3D::get_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_flip_h(bool p_flip_h) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_flip_h")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_h_encoded;
	PtrToArg<bool>::encode(p_flip_h, &p_flip_h_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_h_encoded);
}

bool SpriteBase3D::is_flipped_h() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("is_flipped_h")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_flip_v(bool p_flip_v) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_flip_v")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_v_encoded;
	PtrToArg<bool>::encode(p_flip_v, &p_flip_v_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_v_encoded);
}

bool SpriteBase3D::is_flipped_v() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("is_flipped_v")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_modulate(const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_modulate")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_modulate);
}

Color SpriteBase3D::get_modulate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_modulate")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_render_priority(int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_render_priority")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_encoded);
}

int32_t SpriteBase3D::get_render_priority() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_render_priority")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_pixel_size(float p_pixel_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_pixel_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pixel_size_encoded;
	PtrToArg<double>::encode(p_pixel_size, &p_pixel_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixel_size_encoded);
}

float SpriteBase3D::get_pixel_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_pixel_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_axis(Vector3::Axis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_axis")._native_ptr(), 1144690656);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis_encoded);
}

Vector3::Axis SpriteBase3D::get_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_axis")._native_ptr(), 3050976882);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3::Axis(0)));
	return (Vector3::Axis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_draw_flag(SpriteBase3D::DrawFlags p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_draw_flag")._native_ptr(), 1135633219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flag_encoded, &p_enabled_encoded);
}

bool SpriteBase3D::get_draw_flag(SpriteBase3D::DrawFlags p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_draw_flag")._native_ptr(), 1733036628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_flag_encoded);
}

void SpriteBase3D::set_alpha_cut_mode(SpriteBase3D::AlphaCutMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_alpha_cut_mode")._native_ptr(), 227561226);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

SpriteBase3D::AlphaCutMode SpriteBase3D::get_alpha_cut_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_alpha_cut_mode")._native_ptr(), 336003791);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SpriteBase3D::AlphaCutMode(0)));
	return (SpriteBase3D::AlphaCutMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_alpha_scissor_threshold(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_alpha_scissor_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float SpriteBase3D::get_alpha_scissor_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_alpha_scissor_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_alpha_hash_scale(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_alpha_hash_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float SpriteBase3D::get_alpha_hash_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_alpha_hash_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_alpha_antialiasing(BaseMaterial3D::AlphaAntiAliasing p_alpha_aa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_alpha_antialiasing")._native_ptr(), 3212649852);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alpha_aa_encoded;
	PtrToArg<int64_t>::encode(p_alpha_aa, &p_alpha_aa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alpha_aa_encoded);
}

BaseMaterial3D::AlphaAntiAliasing SpriteBase3D::get_alpha_antialiasing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_alpha_antialiasing")._native_ptr(), 2889939400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::AlphaAntiAliasing(0)));
	return (BaseMaterial3D::AlphaAntiAliasing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_alpha_antialiasing_edge(float p_edge) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_alpha_antialiasing_edge")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_edge_encoded;
	PtrToArg<double>::encode(p_edge, &p_edge_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_encoded);
}

float SpriteBase3D::get_alpha_antialiasing_edge() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_alpha_antialiasing_edge")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_billboard_mode(BaseMaterial3D::BillboardMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_billboard_mode")._native_ptr(), 4202036497);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::BillboardMode SpriteBase3D::get_billboard_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_billboard_mode")._native_ptr(), 1283840139);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::BillboardMode(0)));
	return (BaseMaterial3D::BillboardMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SpriteBase3D::set_texture_filter(BaseMaterial3D::TextureFilter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("set_texture_filter")._native_ptr(), 22904437);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

BaseMaterial3D::TextureFilter SpriteBase3D::get_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_texture_filter")._native_ptr(), 3289213076);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BaseMaterial3D::TextureFilter(0)));
	return (BaseMaterial3D::TextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Rect2 SpriteBase3D::get_item_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("get_item_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

Ref<TriangleMesh> SpriteBase3D::generate_triangle_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteBase3D::get_class_static()._native_ptr(), StringName("generate_triangle_mesh")._native_ptr(), 3476533166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TriangleMesh>()));
	return Ref<TriangleMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TriangleMesh>(_gde_method_bind, _owner));
}

} // namespace godot
