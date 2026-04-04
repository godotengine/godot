/**************************************************************************/
/*  gltf_object_model_property.hpp                                        */
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

#pragma once

#include <godot_cpp/classes/gltf_accessor.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Expression;
class StringName;

class GLTFObjectModelProperty : public RefCounted {
	GDEXTENSION_CLASS(GLTFObjectModelProperty, RefCounted)

public:
	enum GLTFObjectModelType {
		GLTF_OBJECT_MODEL_TYPE_UNKNOWN = 0,
		GLTF_OBJECT_MODEL_TYPE_BOOL = 1,
		GLTF_OBJECT_MODEL_TYPE_FLOAT = 2,
		GLTF_OBJECT_MODEL_TYPE_FLOAT_ARRAY = 3,
		GLTF_OBJECT_MODEL_TYPE_FLOAT2 = 4,
		GLTF_OBJECT_MODEL_TYPE_FLOAT3 = 5,
		GLTF_OBJECT_MODEL_TYPE_FLOAT4 = 6,
		GLTF_OBJECT_MODEL_TYPE_FLOAT2X2 = 7,
		GLTF_OBJECT_MODEL_TYPE_FLOAT3X3 = 8,
		GLTF_OBJECT_MODEL_TYPE_FLOAT4X4 = 9,
		GLTF_OBJECT_MODEL_TYPE_INT = 10,
	};

	void append_node_path(const NodePath &p_node_path);
	void append_path_to_property(const NodePath &p_node_path, const StringName &p_prop_name);
	GLTFAccessor::GLTFAccessorType get_accessor_type() const;
	Ref<Expression> get_gltf_to_godot_expression() const;
	void set_gltf_to_godot_expression(const Ref<Expression> &p_gltf_to_godot_expr);
	Ref<Expression> get_godot_to_gltf_expression() const;
	void set_godot_to_gltf_expression(const Ref<Expression> &p_godot_to_gltf_expr);
	TypedArray<NodePath> get_node_paths() const;
	bool has_node_paths() const;
	void set_node_paths(const TypedArray<NodePath> &p_node_paths);
	GLTFObjectModelProperty::GLTFObjectModelType get_object_model_type() const;
	void set_object_model_type(GLTFObjectModelProperty::GLTFObjectModelType p_type);
	TypedArray<PackedStringArray> get_json_pointers() const;
	bool has_json_pointers() const;
	void set_json_pointers(const TypedArray<PackedStringArray> &p_json_pointers);
	Variant::Type get_variant_type() const;
	void set_variant_type(Variant::Type p_variant_type);
	void set_types(Variant::Type p_variant_type, GLTFObjectModelProperty::GLTFObjectModelType p_obj_model_type);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GLTFObjectModelProperty::GLTFObjectModelType);

