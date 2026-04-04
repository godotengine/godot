/**************************************************************************/
/*  gltf_object_model_property.cpp                                        */
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

#include <godot_cpp/classes/gltf_object_model_property.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/expression.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void GLTFObjectModelProperty::append_node_path(const NodePath &p_node_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("append_node_path")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_path);
}

void GLTFObjectModelProperty::append_path_to_property(const NodePath &p_node_path, const StringName &p_prop_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("append_path_to_property")._native_ptr(), 1331931644);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_path, &p_prop_name);
}

GLTFAccessor::GLTFAccessorType GLTFObjectModelProperty::get_accessor_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_accessor_type")._native_ptr(), 1998183368);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GLTFAccessor::GLTFAccessorType(0)));
	return (GLTFAccessor::GLTFAccessorType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<Expression> GLTFObjectModelProperty::get_gltf_to_godot_expression() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_gltf_to_godot_expression")._native_ptr(), 2240072449);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Expression>()));
	return Ref<Expression>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Expression>(_gde_method_bind, _owner));
}

void GLTFObjectModelProperty::set_gltf_to_godot_expression(const Ref<Expression> &p_gltf_to_godot_expr) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_gltf_to_godot_expression")._native_ptr(), 1815845073);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_gltf_to_godot_expr != nullptr ? &p_gltf_to_godot_expr->_owner : nullptr));
}

Ref<Expression> GLTFObjectModelProperty::get_godot_to_gltf_expression() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_godot_to_gltf_expression")._native_ptr(), 2240072449);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Expression>()));
	return Ref<Expression>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Expression>(_gde_method_bind, _owner));
}

void GLTFObjectModelProperty::set_godot_to_gltf_expression(const Ref<Expression> &p_godot_to_gltf_expr) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_godot_to_gltf_expression")._native_ptr(), 1815845073);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_godot_to_gltf_expr != nullptr ? &p_godot_to_gltf_expr->_owner : nullptr));
}

TypedArray<NodePath> GLTFObjectModelProperty::get_node_paths() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_node_paths")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<NodePath>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<NodePath>>(_gde_method_bind, _owner);
}

bool GLTFObjectModelProperty::has_node_paths() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("has_node_paths")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFObjectModelProperty::set_node_paths(const TypedArray<NodePath> &p_node_paths) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_node_paths")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_node_paths);
}

GLTFObjectModelProperty::GLTFObjectModelType GLTFObjectModelProperty::get_object_model_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_object_model_type")._native_ptr(), 1094778507);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GLTFObjectModelProperty::GLTFObjectModelType(0)));
	return (GLTFObjectModelProperty::GLTFObjectModelType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFObjectModelProperty::set_object_model_type(GLTFObjectModelProperty::GLTFObjectModelType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_object_model_type")._native_ptr(), 4108684086);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

TypedArray<PackedStringArray> GLTFObjectModelProperty::get_json_pointers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_json_pointers")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedStringArray>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedStringArray>>(_gde_method_bind, _owner);
}

bool GLTFObjectModelProperty::has_json_pointers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("has_json_pointers")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFObjectModelProperty::set_json_pointers(const TypedArray<PackedStringArray> &p_json_pointers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_json_pointers")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_json_pointers);
}

Variant::Type GLTFObjectModelProperty::get_variant_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("get_variant_type")._native_ptr(), 3416842102);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant::Type(0)));
	return (Variant::Type)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFObjectModelProperty::set_variant_type(Variant::Type p_variant_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_variant_type")._native_ptr(), 2887708385);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_variant_type_encoded;
	PtrToArg<int64_t>::encode(p_variant_type, &p_variant_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_variant_type_encoded);
}

void GLTFObjectModelProperty::set_types(Variant::Type p_variant_type, GLTFObjectModelProperty::GLTFObjectModelType p_obj_model_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFObjectModelProperty::get_class_static()._native_ptr(), StringName("set_types")._native_ptr(), 4150728237);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_variant_type_encoded;
	PtrToArg<int64_t>::encode(p_variant_type, &p_variant_type_encoded);
	int64_t p_obj_model_type_encoded;
	PtrToArg<int64_t>::encode(p_obj_model_type, &p_obj_model_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_variant_type_encoded, &p_obj_model_type_encoded);
}

} // namespace godot
