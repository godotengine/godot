/**************************************************************************/
/*  mesh.cpp                                                              */
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

#include <godot_cpp/classes/mesh.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/concave_polygon_shape3d.hpp>
#include <godot_cpp/classes/convex_polygon_shape3d.hpp>
#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/triangle_mesh.hpp>

namespace godot {

void Mesh::set_lightmap_size_hint(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("set_lightmap_size_hint")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i Mesh::get_lightmap_size_hint() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("get_lightmap_size_hint")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

AABB Mesh::get_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("get_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

PackedVector3Array Mesh::get_faces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("get_faces")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

int32_t Mesh::get_surface_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("get_surface_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Array Mesh::surface_get_arrays(int32_t p_surf_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("surface_get_arrays")._native_ptr(), 663333327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_surf_idx_encoded;
	PtrToArg<int64_t>::encode(p_surf_idx, &p_surf_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_surf_idx_encoded);
}

TypedArray<Array> Mesh::surface_get_blend_shape_arrays(int32_t p_surf_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("surface_get_blend_shape_arrays")._native_ptr(), 663333327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Array>()));
	int64_t p_surf_idx_encoded;
	PtrToArg<int64_t>::encode(p_surf_idx, &p_surf_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Array>>(_gde_method_bind, _owner, &p_surf_idx_encoded);
}

void Mesh::surface_set_material(int32_t p_surf_idx, const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("surface_set_material")._native_ptr(), 3671737478);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surf_idx_encoded;
	PtrToArg<int64_t>::encode(p_surf_idx, &p_surf_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_surf_idx_encoded, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> Mesh::surface_get_material(int32_t p_surf_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("surface_get_material")._native_ptr(), 2897466400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	int64_t p_surf_idx_encoded;
	PtrToArg<int64_t>::encode(p_surf_idx, &p_surf_idx_encoded);
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner, &p_surf_idx_encoded));
}

Ref<Resource> Mesh::create_placeholder() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("create_placeholder")._native_ptr(), 121922552);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner));
}

Ref<ConcavePolygonShape3D> Mesh::create_trimesh_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("create_trimesh_shape")._native_ptr(), 4160111210);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ConcavePolygonShape3D>()));
	return Ref<ConcavePolygonShape3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ConcavePolygonShape3D>(_gde_method_bind, _owner));
}

Ref<ConvexPolygonShape3D> Mesh::create_convex_shape(bool p_clean, bool p_simplify) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("create_convex_shape")._native_ptr(), 2529984628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ConvexPolygonShape3D>()));
	int8_t p_clean_encoded;
	PtrToArg<bool>::encode(p_clean, &p_clean_encoded);
	int8_t p_simplify_encoded;
	PtrToArg<bool>::encode(p_simplify, &p_simplify_encoded);
	return Ref<ConvexPolygonShape3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ConvexPolygonShape3D>(_gde_method_bind, _owner, &p_clean_encoded, &p_simplify_encoded));
}

Ref<Mesh> Mesh::create_outline(float p_margin) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("create_outline")._native_ptr(), 1208642001);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Mesh>()));
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	return Ref<Mesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Mesh>(_gde_method_bind, _owner, &p_margin_encoded));
}

Ref<TriangleMesh> Mesh::generate_triangle_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Mesh::get_class_static()._native_ptr(), StringName("generate_triangle_mesh")._native_ptr(), 3476533166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TriangleMesh>()));
	return Ref<TriangleMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TriangleMesh>(_gde_method_bind, _owner));
}

int32_t Mesh::_get_surface_count() const {
	return 0;
}

int32_t Mesh::_surface_get_array_len(int32_t p_index) const {
	return 0;
}

int32_t Mesh::_surface_get_array_index_len(int32_t p_index) const {
	return 0;
}

Array Mesh::_surface_get_arrays(int32_t p_index) const {
	return Array();
}

TypedArray<Array> Mesh::_surface_get_blend_shape_arrays(int32_t p_index) const {
	return TypedArray<Array>();
}

Dictionary Mesh::_surface_get_lods(int32_t p_index) const {
	return Dictionary();
}

uint32_t Mesh::_surface_get_format(int32_t p_index) const {
	return 0;
}

uint32_t Mesh::_surface_get_primitive_type(int32_t p_index) const {
	return 0;
}

void Mesh::_surface_set_material(int32_t p_index, const Ref<Material> &p_material) {}

Ref<Material> Mesh::_surface_get_material(int32_t p_index) const {
	return Ref<Material>();
}

int32_t Mesh::_get_blend_shape_count() const {
	return 0;
}

StringName Mesh::_get_blend_shape_name(int32_t p_index) const {
	return StringName();
}

void Mesh::_set_blend_shape_name(int32_t p_index, const StringName &p_name) {}

AABB Mesh::_get_aabb() const {
	return AABB();
}

} // namespace godot
