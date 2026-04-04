/**************************************************************************/
/*  primitive_mesh.cpp                                                    */
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

#include <godot_cpp/classes/primitive_mesh.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>

namespace godot {

void PrimitiveMesh::set_material(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("set_material")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> PrimitiveMesh::get_material() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("get_material")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

Array PrimitiveMesh::get_mesh_arrays() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("get_mesh_arrays")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void PrimitiveMesh::set_custom_aabb(const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("set_custom_aabb")._native_ptr(), 259215842);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aabb);
}

AABB PrimitiveMesh::get_custom_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("get_custom_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

void PrimitiveMesh::set_flip_faces(bool p_flip_faces) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("set_flip_faces")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_faces_encoded;
	PtrToArg<bool>::encode(p_flip_faces, &p_flip_faces_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_faces_encoded);
}

bool PrimitiveMesh::get_flip_faces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("get_flip_faces")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PrimitiveMesh::set_add_uv2(bool p_add_uv2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("set_add_uv2")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_add_uv2_encoded;
	PtrToArg<bool>::encode(p_add_uv2, &p_add_uv2_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_add_uv2_encoded);
}

bool PrimitiveMesh::get_add_uv2() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("get_add_uv2")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PrimitiveMesh::set_uv2_padding(float p_uv2_padding) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("set_uv2_padding")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_uv2_padding_encoded;
	PtrToArg<double>::encode(p_uv2_padding, &p_uv2_padding_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uv2_padding_encoded);
}

float PrimitiveMesh::get_uv2_padding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("get_uv2_padding")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PrimitiveMesh::request_update() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PrimitiveMesh::get_class_static()._native_ptr(), StringName("request_update")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Array PrimitiveMesh::_create_mesh_array() const {
	return Array();
}

} // namespace godot
