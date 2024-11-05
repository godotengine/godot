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

#include "gltf_object_model_property.h"

#include "../gltf_template_convert.h"

void GLTFObjectModelProperty::_bind_methods() {
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_UNKNOWN);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_BOOL);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT_ARRAY);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT2);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT3);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT4);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT2X2);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT3X3);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_FLOAT4X4);
	BIND_ENUM_CONSTANT(GLTF_OBJECT_MODEL_TYPE_INT);

	ClassDB::bind_method(D_METHOD("append_node_path", "node_path"), &GLTFObjectModelProperty::append_node_path);
	ClassDB::bind_method(D_METHOD("append_path_to_property", "node_path", "prop_name"), &GLTFObjectModelProperty::append_path_to_property);

	ClassDB::bind_method(D_METHOD("get_accessor_type"), &GLTFObjectModelProperty::get_accessor_type);
	ClassDB::bind_method(D_METHOD("get_gltf_to_godot_expression"), &GLTFObjectModelProperty::get_gltf_to_godot_expression);
	ClassDB::bind_method(D_METHOD("set_gltf_to_godot_expression", "gltf_to_godot_expr"), &GLTFObjectModelProperty::set_gltf_to_godot_expression);
	ClassDB::bind_method(D_METHOD("get_godot_to_gltf_expression"), &GLTFObjectModelProperty::get_godot_to_gltf_expression);
	ClassDB::bind_method(D_METHOD("set_godot_to_gltf_expression", "godot_to_gltf_expr"), &GLTFObjectModelProperty::set_godot_to_gltf_expression);
	ClassDB::bind_method(D_METHOD("get_node_paths"), &GLTFObjectModelProperty::get_node_paths);
	ClassDB::bind_method(D_METHOD("has_node_paths"), &GLTFObjectModelProperty::has_node_paths);
	ClassDB::bind_method(D_METHOD("set_node_paths", "node_paths"), &GLTFObjectModelProperty::set_node_paths);
	ClassDB::bind_method(D_METHOD("get_object_model_type"), &GLTFObjectModelProperty::get_object_model_type);
	ClassDB::bind_method(D_METHOD("set_object_model_type", "type"), &GLTFObjectModelProperty::set_object_model_type);
	ClassDB::bind_method(D_METHOD("get_json_pointers"), &GLTFObjectModelProperty::get_json_pointers_bind);
	ClassDB::bind_method(D_METHOD("has_json_pointers"), &GLTFObjectModelProperty::has_json_pointers);
	ClassDB::bind_method(D_METHOD("set_json_pointers", "json_pointers"), &GLTFObjectModelProperty::set_json_pointers_bind);
	ClassDB::bind_method(D_METHOD("get_variant_type"), &GLTFObjectModelProperty::get_variant_type);
	ClassDB::bind_method(D_METHOD("set_variant_type", "variant_type"), &GLTFObjectModelProperty::set_variant_type);
	ClassDB::bind_method(D_METHOD("set_types", "variant_type", "obj_model_type"), &GLTFObjectModelProperty::set_types);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gltf_to_godot_expression", PROPERTY_HINT_RESOURCE_TYPE, "Expression"), "set_gltf_to_godot_expression", "get_gltf_to_godot_expression"); // Ref<Expression>
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "godot_to_gltf_expression", PROPERTY_HINT_RESOURCE_TYPE, "Expression"), "set_godot_to_gltf_expression", "get_godot_to_gltf_expression"); // Ref<Expression>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "node_paths", PROPERTY_HINT_TYPE_STRING, "NodePath"), "set_node_paths", "get_node_paths"); // TypedArray<NodePath>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "object_model_type"), "set_object_model_type", "get_object_model_type"); // GLTFObjectModelType
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "json_pointers"), "set_json_pointers", "get_json_pointers"); // TypedArray<PackedStringArray>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "variant_type"), "set_variant_type", "get_variant_type"); // Variant::Type
}

void GLTFObjectModelProperty::append_node_path(const NodePath &p_node_path) {
	node_paths.push_back(p_node_path);
}

void GLTFObjectModelProperty::append_path_to_property(const NodePath &p_node_path, const StringName &p_prop_name) {
	Vector<StringName> node_names = p_node_path.get_names();
	Vector<StringName> subpath = p_node_path.get_subnames();
	subpath.append(p_prop_name);
	node_paths.push_back(NodePath(node_names, subpath, false));
}

GLTFAccessor::GLTFAccessorType GLTFObjectModelProperty::get_accessor_type() const {
	switch (object_model_type) {
		case GLTF_OBJECT_MODEL_TYPE_FLOAT2:
			return GLTFAccessor::TYPE_VEC2;
		case GLTF_OBJECT_MODEL_TYPE_FLOAT3:
			return GLTFAccessor::TYPE_VEC3;
		case GLTF_OBJECT_MODEL_TYPE_FLOAT4:
			return GLTFAccessor::TYPE_VEC4;
		case GLTF_OBJECT_MODEL_TYPE_FLOAT2X2:
			return GLTFAccessor::TYPE_MAT2;
		case GLTF_OBJECT_MODEL_TYPE_FLOAT3X3:
			return GLTFAccessor::TYPE_MAT3;
		case GLTF_OBJECT_MODEL_TYPE_FLOAT4X4:
			return GLTFAccessor::TYPE_MAT4;
		default:
			return GLTFAccessor::TYPE_SCALAR;
	}
}

Ref<Expression> GLTFObjectModelProperty::get_gltf_to_godot_expression() const {
	return gltf_to_godot_expr;
}

void GLTFObjectModelProperty::set_gltf_to_godot_expression(Ref<Expression> p_gltf_to_godot_expr) {
	gltf_to_godot_expr = p_gltf_to_godot_expr;
}

Ref<Expression> GLTFObjectModelProperty::get_godot_to_gltf_expression() const {
	return godot_to_gltf_expr;
}

void GLTFObjectModelProperty::set_godot_to_gltf_expression(Ref<Expression> p_godot_to_gltf_expr) {
	godot_to_gltf_expr = p_godot_to_gltf_expr;
}

TypedArray<NodePath> GLTFObjectModelProperty::get_node_paths() const {
	return node_paths;
}

bool GLTFObjectModelProperty::has_node_paths() const {
	return !node_paths.is_empty();
}

void GLTFObjectModelProperty::set_node_paths(TypedArray<NodePath> p_node_paths) {
	node_paths = p_node_paths;
}

GLTFObjectModelProperty::GLTFObjectModelType GLTFObjectModelProperty::get_object_model_type() const {
	return object_model_type;
}

void GLTFObjectModelProperty::set_object_model_type(GLTFObjectModelType p_type) {
	object_model_type = p_type;
}

Vector<PackedStringArray> GLTFObjectModelProperty::get_json_pointers() const {
	return json_pointers;
}

bool GLTFObjectModelProperty::has_json_pointers() const {
	return !json_pointers.is_empty();
}

void GLTFObjectModelProperty::set_json_pointers(const Vector<PackedStringArray> &p_json_pointers) {
	json_pointers = p_json_pointers;
}

TypedArray<PackedStringArray> GLTFObjectModelProperty::get_json_pointers_bind() const {
	return GLTFTemplateConvert::to_array(json_pointers);
}

void GLTFObjectModelProperty::set_json_pointers_bind(const TypedArray<PackedStringArray> &p_json_pointers) {
	GLTFTemplateConvert::set_from_array(json_pointers, p_json_pointers);
}

Variant::Type GLTFObjectModelProperty::get_variant_type() const {
	return variant_type;
}

void GLTFObjectModelProperty::set_variant_type(Variant::Type p_variant_type) {
	variant_type = p_variant_type;
}

void GLTFObjectModelProperty::set_types(Variant::Type p_variant_type, GLTFObjectModelType p_obj_model_type) {
	variant_type = p_variant_type;
	object_model_type = p_obj_model_type;
}
