/**************************************************************************/
/*  gltf_physics_shape.cpp                                                */
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

#include <godot_cpp/classes/gltf_physics_shape.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/collision_shape3d.hpp>
#include <godot_cpp/classes/importer_mesh.hpp>
#include <godot_cpp/classes/shape3d.hpp>

namespace godot {

Ref<GLTFPhysicsShape> GLTFPhysicsShape::from_node(CollisionShape3D *p_shape_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("from_node")._native_ptr(), 3613751275);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFPhysicsShape>()));
	return Ref<GLTFPhysicsShape>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFPhysicsShape>(_gde_method_bind, nullptr, (p_shape_node != nullptr ? &p_shape_node->_owner : nullptr)));
}

CollisionShape3D *GLTFPhysicsShape::to_node(bool p_cache_shapes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("to_node")._native_ptr(), 563689933);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int8_t p_cache_shapes_encoded;
	PtrToArg<bool>::encode(p_cache_shapes, &p_cache_shapes_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<CollisionShape3D>(_gde_method_bind, _owner, &p_cache_shapes_encoded);
}

Ref<GLTFPhysicsShape> GLTFPhysicsShape::from_resource(const Ref<Shape3D> &p_shape_resource) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("from_resource")._native_ptr(), 3845569786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFPhysicsShape>()));
	return Ref<GLTFPhysicsShape>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFPhysicsShape>(_gde_method_bind, nullptr, (p_shape_resource != nullptr ? &p_shape_resource->_owner : nullptr)));
}

Ref<Shape3D> GLTFPhysicsShape::to_resource(bool p_cache_shapes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("to_resource")._native_ptr(), 1913542110);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Shape3D>()));
	int8_t p_cache_shapes_encoded;
	PtrToArg<bool>::encode(p_cache_shapes, &p_cache_shapes_encoded);
	return Ref<Shape3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Shape3D>(_gde_method_bind, _owner, &p_cache_shapes_encoded));
}

Ref<GLTFPhysicsShape> GLTFPhysicsShape::from_dictionary(const Dictionary &p_dictionary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("from_dictionary")._native_ptr(), 2390691823);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFPhysicsShape>()));
	return Ref<GLTFPhysicsShape>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFPhysicsShape>(_gde_method_bind, nullptr, &p_dictionary));
}

Dictionary GLTFPhysicsShape::to_dictionary() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("to_dictionary")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

String GLTFPhysicsShape::get_shape_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_shape_type")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void GLTFPhysicsShape::set_shape_type(const String &p_shape_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_shape_type")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_type);
}

Vector3 GLTFPhysicsShape::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFPhysicsShape::set_size(const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

float GLTFPhysicsShape::get_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFPhysicsShape::set_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float GLTFPhysicsShape::get_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFPhysicsShape::set_height(float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

bool GLTFPhysicsShape::get_is_trigger() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_is_trigger")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFPhysicsShape::set_is_trigger(bool p_is_trigger) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_is_trigger")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_trigger_encoded;
	PtrToArg<bool>::encode(p_is_trigger, &p_is_trigger_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_is_trigger_encoded);
}

int32_t GLTFPhysicsShape::get_mesh_index() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_mesh_index")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFPhysicsShape::set_mesh_index(int32_t p_mesh_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_mesh_index")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mesh_index_encoded;
	PtrToArg<int64_t>::encode(p_mesh_index, &p_mesh_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh_index_encoded);
}

Ref<ImporterMesh> GLTFPhysicsShape::get_importer_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("get_importer_mesh")._native_ptr(), 3161779525);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ImporterMesh>()));
	return Ref<ImporterMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ImporterMesh>(_gde_method_bind, _owner));
}

void GLTFPhysicsShape::set_importer_mesh(const Ref<ImporterMesh> &p_importer_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsShape::get_class_static()._native_ptr(), StringName("set_importer_mesh")._native_ptr(), 2255166972);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_importer_mesh != nullptr ? &p_importer_mesh->_owner : nullptr));
}

} // namespace godot
