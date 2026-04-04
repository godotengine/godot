/**************************************************************************/
/*  visual_shader.hpp                                                     */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;
class StringName;
class VisualShaderNode;

class VisualShader : public Shader {
	GDEXTENSION_CLASS(VisualShader, Shader)

public:
	enum Type {
		TYPE_VERTEX = 0,
		TYPE_FRAGMENT = 1,
		TYPE_LIGHT = 2,
		TYPE_START = 3,
		TYPE_PROCESS = 4,
		TYPE_COLLIDE = 5,
		TYPE_START_CUSTOM = 6,
		TYPE_PROCESS_CUSTOM = 7,
		TYPE_SKY = 8,
		TYPE_FOG = 9,
		TYPE_MAX = 10,
	};

	enum VaryingMode {
		VARYING_MODE_VERTEX_TO_FRAG_LIGHT = 0,
		VARYING_MODE_FRAG_TO_LIGHT = 1,
		VARYING_MODE_MAX = 2,
	};

	enum VaryingType {
		VARYING_TYPE_FLOAT = 0,
		VARYING_TYPE_INT = 1,
		VARYING_TYPE_UINT = 2,
		VARYING_TYPE_VECTOR_2D = 3,
		VARYING_TYPE_VECTOR_3D = 4,
		VARYING_TYPE_VECTOR_4D = 5,
		VARYING_TYPE_BOOLEAN = 6,
		VARYING_TYPE_TRANSFORM = 7,
		VARYING_TYPE_MAX = 8,
	};

	static const int NODE_ID_INVALID = -1;
	static const int NODE_ID_OUTPUT = 0;

	void set_mode(Shader::Mode p_mode);
	void add_node(VisualShader::Type p_type, const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int32_t p_id);
	Ref<VisualShaderNode> get_node(VisualShader::Type p_type, int32_t p_id) const;
	void set_node_position(VisualShader::Type p_type, int32_t p_id, const Vector2 &p_position);
	Vector2 get_node_position(VisualShader::Type p_type, int32_t p_id) const;
	PackedInt32Array get_node_list(VisualShader::Type p_type) const;
	int32_t get_valid_node_id(VisualShader::Type p_type) const;
	void remove_node(VisualShader::Type p_type, int32_t p_id);
	void replace_node(VisualShader::Type p_type, int32_t p_id, const StringName &p_new_class);
	bool is_node_connection(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) const;
	bool can_connect_nodes(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) const;
	Error connect_nodes(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port);
	void disconnect_nodes(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port);
	void connect_nodes_forced(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port);
	TypedArray<Dictionary> get_node_connections(VisualShader::Type p_type) const;
	void attach_node_to_frame(VisualShader::Type p_type, int32_t p_id, int32_t p_frame);
	void detach_node_from_frame(VisualShader::Type p_type, int32_t p_id);
	void add_varying(const String &p_name, VisualShader::VaryingMode p_mode, VisualShader::VaryingType p_type);
	void remove_varying(const String &p_name);
	bool has_varying(const String &p_name) const;
	void set_graph_offset(const Vector2 &p_offset);
	Vector2 get_graph_offset() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Shader::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShader::Type);
VARIANT_ENUM_CAST(VisualShader::VaryingMode);
VARIANT_ENUM_CAST(VisualShader::VaryingType);

