/**************************************************************************/
/*  polygon2d.cpp                                                         */
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

#include <godot_cpp/classes/polygon2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void Polygon2D::set_polygon(const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_polygon")._native_ptr(), 1509147220);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_polygon);
}

PackedVector2Array Polygon2D::get_polygon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_polygon")._native_ptr(), 2961356807);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner);
}

void Polygon2D::set_uv(const PackedVector2Array &p_uv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_uv")._native_ptr(), 1509147220);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uv);
}

PackedVector2Array Polygon2D::get_uv() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_uv")._native_ptr(), 2961356807);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner);
}

void Polygon2D::set_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color Polygon2D::get_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void Polygon2D::set_polygons(const Array &p_polygons) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_polygons")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_polygons);
}

Array Polygon2D::get_polygons() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_polygons")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void Polygon2D::set_vertex_colors(const PackedColorArray &p_vertex_colors) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_vertex_colors")._native_ptr(), 3546319833);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertex_colors);
}

PackedColorArray Polygon2D::get_vertex_colors() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_vertex_colors")._native_ptr(), 1392750486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedColorArray()));
	return ::godot::internal::_call_native_mb_ret<PackedColorArray>(_gde_method_bind, _owner);
}

void Polygon2D::set_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> Polygon2D::get_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void Polygon2D::set_texture_offset(const Vector2 &p_texture_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_texture_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture_offset);
}

Vector2 Polygon2D::get_texture_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_texture_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Polygon2D::set_texture_rotation(float p_texture_rotation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_texture_rotation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_texture_rotation_encoded;
	PtrToArg<double>::encode(p_texture_rotation, &p_texture_rotation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture_rotation_encoded);
}

float Polygon2D::get_texture_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_texture_rotation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Polygon2D::set_texture_scale(const Vector2 &p_texture_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_texture_scale")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture_scale);
}

Vector2 Polygon2D::get_texture_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_texture_scale")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Polygon2D::set_invert_enabled(bool p_invert) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_invert_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_invert_encoded);
}

bool Polygon2D::get_invert_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_invert_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Polygon2D::set_antialiased(bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_antialiased")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_antialiased_encoded);
}

bool Polygon2D::get_antialiased() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_antialiased")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Polygon2D::set_invert_border(float p_invert_border) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_invert_border")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_invert_border_encoded;
	PtrToArg<double>::encode(p_invert_border, &p_invert_border_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_invert_border_encoded);
}

float Polygon2D::get_invert_border() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_invert_border")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Polygon2D::set_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 Polygon2D::get_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Polygon2D::add_bone(const NodePath &p_path, const PackedFloat32Array &p_weights) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("add_bone")._native_ptr(), 703042815);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_weights);
}

int32_t Polygon2D::get_bone_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_bone_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

NodePath Polygon2D::get_bone_path(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_bone_path")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_index_encoded);
}

PackedFloat32Array Polygon2D::get_bone_weights(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_bone_weights")._native_ptr(), 1542882410);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner, &p_index_encoded);
}

void Polygon2D::erase_bone(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("erase_bone")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void Polygon2D::clear_bones() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("clear_bones")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Polygon2D::set_bone_path(int32_t p_index, const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_bone_path")._native_ptr(), 2761262315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_path);
}

void Polygon2D::set_bone_weights(int32_t p_index, const PackedFloat32Array &p_weights) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_bone_weights")._native_ptr(), 1345852415);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_weights);
}

void Polygon2D::set_skeleton(const NodePath &p_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_skeleton")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton);
}

NodePath Polygon2D::get_skeleton() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_skeleton")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void Polygon2D::set_internal_vertex_count(int32_t p_internal_vertex_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("set_internal_vertex_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_internal_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_internal_vertex_count, &p_internal_vertex_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_internal_vertex_count_encoded);
}

int32_t Polygon2D::get_internal_vertex_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Polygon2D::get_class_static()._native_ptr(), StringName("get_internal_vertex_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
