/**************************************************************************/
/*  surface_tool.cpp                                                      */
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

#include <godot_cpp/classes/surface_tool.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

void SurfaceTool::set_skin_weight_count(SurfaceTool::SkinWeightCount p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_skin_weight_count")._native_ptr(), 618679515);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

SurfaceTool::SkinWeightCount SurfaceTool::get_skin_weight_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("get_skin_weight_count")._native_ptr(), 1072401130);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SurfaceTool::SkinWeightCount(0)));
	return (SurfaceTool::SkinWeightCount)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SurfaceTool::set_custom_format(int32_t p_channel_index, SurfaceTool::CustomFormat p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_custom_format")._native_ptr(), 4087759856);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_index_encoded;
	PtrToArg<int64_t>::encode(p_channel_index, &p_channel_index_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_index_encoded, &p_format_encoded);
}

SurfaceTool::CustomFormat SurfaceTool::get_custom_format(int32_t p_channel_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("get_custom_format")._native_ptr(), 839863283);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SurfaceTool::CustomFormat(0)));
	int64_t p_channel_index_encoded;
	PtrToArg<int64_t>::encode(p_channel_index, &p_channel_index_encoded);
	return (SurfaceTool::CustomFormat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_channel_index_encoded);
}

void SurfaceTool::begin(Mesh::PrimitiveType p_primitive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("begin")._native_ptr(), 2230304113);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_primitive_encoded;
	PtrToArg<int64_t>::encode(p_primitive, &p_primitive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_primitive_encoded);
}

void SurfaceTool::add_vertex(const Vector3 &p_vertex) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("add_vertex")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertex);
}

void SurfaceTool::set_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void SurfaceTool::set_normal(const Vector3 &p_normal) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_normal")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_normal);
}

void SurfaceTool::set_tangent(const Plane &p_tangent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_tangent")._native_ptr(), 3505987427);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tangent);
}

void SurfaceTool::set_uv(const Vector2 &p_uv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_uv")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uv);
}

void SurfaceTool::set_uv2(const Vector2 &p_uv2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_uv2")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uv2);
}

void SurfaceTool::set_bones(const PackedInt32Array &p_bones) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_bones")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bones);
}

void SurfaceTool::set_weights(const PackedFloat32Array &p_weights) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_weights")._native_ptr(), 2899603908);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_weights);
}

void SurfaceTool::set_custom(int32_t p_channel_index, const Color &p_custom_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_custom")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_index_encoded;
	PtrToArg<int64_t>::encode(p_channel_index, &p_channel_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_index_encoded, &p_custom_color);
}

void SurfaceTool::set_smooth_group(uint32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_smooth_group")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void SurfaceTool::add_triangle_fan(const PackedVector3Array &p_vertices, const PackedVector2Array &p_uvs, const PackedColorArray &p_colors, const PackedVector2Array &p_uv2s, const PackedVector3Array &p_normals, const TypedArray<Plane> &p_tangents) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("add_triangle_fan")._native_ptr(), 2235017613);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices, &p_uvs, &p_colors, &p_uv2s, &p_normals, &p_tangents);
}

void SurfaceTool::add_index(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("add_index")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void SurfaceTool::index() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("index")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SurfaceTool::deindex() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("deindex")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SurfaceTool::generate_normals(bool p_flip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("generate_normals")._native_ptr(), 107499316);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_encoded;
	PtrToArg<bool>::encode(p_flip, &p_flip_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_encoded);
}

void SurfaceTool::generate_tangents() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("generate_tangents")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SurfaceTool::optimize_indices_for_cache() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("optimize_indices_for_cache")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

AABB SurfaceTool::get_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("get_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

PackedInt32Array SurfaceTool::generate_lod(float p_nd_threshold, int32_t p_target_index_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("generate_lod")._native_ptr(), 1938056459);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	double p_nd_threshold_encoded;
	PtrToArg<double>::encode(p_nd_threshold, &p_nd_threshold_encoded);
	int64_t p_target_index_count_encoded;
	PtrToArg<int64_t>::encode(p_target_index_count, &p_target_index_count_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_nd_threshold_encoded, &p_target_index_count_encoded);
}

void SurfaceTool::set_material(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("set_material")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Mesh::PrimitiveType SurfaceTool::get_primitive_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("get_primitive_type")._native_ptr(), 768822145);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Mesh::PrimitiveType(0)));
	return (Mesh::PrimitiveType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SurfaceTool::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void SurfaceTool::create_from(const Ref<Mesh> &p_existing, int32_t p_surface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("create_from")._native_ptr(), 1767024570);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_existing != nullptr ? &p_existing->_owner : nullptr), &p_surface_encoded);
}

void SurfaceTool::create_from_arrays(const Array &p_arrays, Mesh::PrimitiveType p_primitive_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("create_from_arrays")._native_ptr(), 1894639680);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_primitive_type_encoded;
	PtrToArg<int64_t>::encode(p_primitive_type, &p_primitive_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_arrays, &p_primitive_type_encoded);
}

void SurfaceTool::create_from_blend_shape(const Ref<Mesh> &p_existing, int32_t p_surface, const String &p_blend_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("create_from_blend_shape")._native_ptr(), 1306185582);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_existing != nullptr ? &p_existing->_owner : nullptr), &p_surface_encoded, &p_blend_shape);
}

void SurfaceTool::append_from(const Ref<Mesh> &p_existing, int32_t p_surface, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("append_from")._native_ptr(), 2217967155);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_existing != nullptr ? &p_existing->_owner : nullptr), &p_surface_encoded, &p_transform);
}

Ref<ArrayMesh> SurfaceTool::commit(const Ref<ArrayMesh> &p_existing, uint64_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("commit")._native_ptr(), 4107864055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ArrayMesh>()));
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return Ref<ArrayMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ArrayMesh>(_gde_method_bind, _owner, (p_existing != nullptr ? &p_existing->_owner : nullptr), &p_flags_encoded));
}

Array SurfaceTool::commit_to_arrays() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SurfaceTool::get_class_static()._native_ptr(), StringName("commit_to_arrays")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

} // namespace godot
