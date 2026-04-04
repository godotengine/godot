/**************************************************************************/
/*  animation_node_state_machine.hpp                                      */
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
class AnimationNodeStateMachineTransition;

class AnimationNodeStateMachine : public AnimationRootNode {
	GDEXTENSION_CLASS(AnimationNodeStateMachine, AnimationRootNode)

public:
	enum StateMachineType {
		STATE_MACHINE_TYPE_ROOT = 0,
		STATE_MACHINE_TYPE_NESTED = 1,
		STATE_MACHINE_TYPE_GROUPED = 2,
	};

	void add_node(const StringName &p_name, const Ref<AnimationNode> &p_node, const Vector2 &p_position = Vector2(0, 0));
	void replace_node(const StringName &p_name, const Ref<AnimationNode> &p_node);
	Ref<AnimationNode> get_node(const StringName &p_name) const;
	void remove_node(const StringName &p_name);
	void rename_node(const StringName &p_name, const StringName &p_new_name);
	bool has_node(const StringName &p_name) const;
	StringName get_node_name(const Ref<AnimationNode> &p_node) const;
	TypedArray<StringName> get_node_list() const;
	void set_node_position(const StringName &p_name, const Vector2 &p_position);
	Vector2 get_node_position(const StringName &p_name) const;
	bool has_transition(const StringName &p_from, const StringName &p_to) const;
	void add_transition(const StringName &p_from, const StringName &p_to, const Ref<AnimationNodeStateMachineTransition> &p_transition);
	Ref<AnimationNodeStateMachineTransition> get_transition(int32_t p_idx) const;
	StringName get_transition_from(int32_t p_idx) const;
	StringName get_transition_to(int32_t p_idx) const;
	int32_t get_transition_count() const;
	void remove_transition_by_index(int32_t p_idx);
	void remove_transition(const StringName &p_from, const StringName &p_to);
	void set_graph_offset(const Vector2 &p_offset);
	Vector2 get_graph_offset() const;
	void set_state_machine_type(AnimationNodeStateMachine::StateMachineType p_state_machine_type);
	AnimationNodeStateMachine::StateMachineType get_state_machine_type() const;
	void set_allow_transition_to_self(bool p_enable);
	bool is_allow_transition_to_self() const;
	void set_reset_ends(bool p_enable);
	bool are_ends_reset() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AnimationRootNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationNodeStateMachine::StateMachineType);

