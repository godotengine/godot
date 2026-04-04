/**************************************************************************/
/*  animation_node_blend_tree.hpp                                         */
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

#include <godot_cpp/classes/animation_root_node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AnimationNode;

class AnimationNodeBlendTree : public AnimationRootNode {
	GDEXTENSION_CLASS(AnimationNodeBlendTree, AnimationRootNode)

public:
	static const int CONNECTION_OK = 0;
	static const int CONNECTION_ERROR_NO_INPUT = 1;
	static const int CONNECTION_ERROR_NO_INPUT_INDEX = 2;
	static const int CONNECTION_ERROR_NO_OUTPUT = 3;
	static const int CONNECTION_ERROR_SAME_NODE = 4;
	static const int CONNECTION_ERROR_CONNECTION_EXISTS = 5;

	void add_node(const StringName &p_name, const Ref<AnimationNode> &p_node, const Vector2 &p_position = Vector2(0, 0));
	Ref<AnimationNode> get_node(const StringName &p_name) const;
	void remove_node(const StringName &p_name);
	void rename_node(const StringName &p_name, const StringName &p_new_name);
	bool has_node(const StringName &p_name) const;
	void connect_node(const StringName &p_input_node, int32_t p_input_index, const StringName &p_output_node);
	void disconnect_node(const StringName &p_input_node, int32_t p_input_index);
	TypedArray<StringName> get_node_list() const;
	void set_node_position(const StringName &p_name, const Vector2 &p_position);
	Vector2 get_node_position(const StringName &p_name) const;
	void set_graph_offset(const Vector2 &p_offset);
	Vector2 get_graph_offset() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AnimationRootNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

