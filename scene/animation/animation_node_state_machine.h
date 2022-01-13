/*************************************************************************/
/*  animation_node_state_machine.h                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef ANIMATION_NODE_STATE_MACHINE_H
#define ANIMATION_NODE_STATE_MACHINE_H

#include "scene/animation/animation_tree.h"

class AnimationNodeStateMachineTransition : public Resource {
	GDCLASS(AnimationNodeStateMachineTransition, Resource);

public:
	enum SwitchMode {
		SWITCH_MODE_IMMEDIATE,
		SWITCH_MODE_SYNC,
		SWITCH_MODE_AT_END,
	};

private:
	SwitchMode switch_mode;
	bool auto_advance;
	StringName advance_condition;
	StringName advance_condition_name;
	float xfade;
	bool disabled;
	int priority;

protected:
	static void _bind_methods();

public:
	void set_switch_mode(SwitchMode p_mode);
	SwitchMode get_switch_mode() const;

	void set_auto_advance(bool p_enable);
	bool has_auto_advance() const;

	void set_advance_condition(const StringName &p_condition);
	StringName get_advance_condition() const;

	StringName get_advance_condition_name() const;

	void set_xfade_time(float p_xfade);
	float get_xfade_time() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	void set_priority(int p_priority);
	int get_priority() const;

	AnimationNodeStateMachineTransition();
};

VARIANT_ENUM_CAST(AnimationNodeStateMachineTransition::SwitchMode)

class AnimationNodeStateMachine;

class AnimationNodeStateMachinePlayback : public Resource {
	GDCLASS(AnimationNodeStateMachinePlayback, Resource);

	friend class AnimationNodeStateMachine;

	struct AStarCost {
		float distance;
		StringName prev;
	};

	float len_total;

	float len_current;
	float pos_current;
	int loops_current;

	StringName current;

	StringName fading_from;
	float fading_time;
	float fading_pos;

	Vector<StringName> path;
	bool playing;

	StringName start_request;
	bool start_request_travel;
	bool stop_request;

	bool _travel(AnimationNodeStateMachine *p_state_machine, const StringName &p_travel);

	float process(AnimationNodeStateMachine *p_state_machine, float p_time, bool p_seek);

protected:
	static void _bind_methods();

public:
	void travel(const StringName &p_state);
	void start(const StringName &p_state);
	void stop();
	bool is_playing() const;
	StringName get_current_node() const;
	StringName get_blend_from_node() const;
	Vector<StringName> get_travel_path() const;
	float get_current_play_pos() const;
	float get_current_length() const;

	AnimationNodeStateMachinePlayback();
};

class AnimationNodeStateMachine : public AnimationRootNode {
	GDCLASS(AnimationNodeStateMachine, AnimationRootNode);

private:
	friend class AnimationNodeStateMachinePlayback;

	struct State {
		Ref<AnimationRootNode> node;
		Vector2 position;
	};

	Map<StringName, State> states;

	struct Transition {
		StringName from;
		StringName to;
		Ref<AnimationNodeStateMachineTransition> transition;
	};

	Vector<Transition> transitions;

	StringName playback;

	StringName start_node;
	StringName end_node;

	Vector2 graph_offset;

	void _tree_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual void get_parameter_list(List<PropertyInfo> *r_list) const;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const;

	void add_node(const StringName &p_name, Ref<AnimationNode> p_node, const Vector2 &p_position = Vector2());
	void replace_node(const StringName &p_name, Ref<AnimationNode> p_node);
	Ref<AnimationNode> get_node(const StringName &p_name) const;
	void remove_node(const StringName &p_name);
	void rename_node(const StringName &p_name, const StringName &p_new_name);
	bool has_node(const StringName &p_name) const;
	StringName get_node_name(const Ref<AnimationNode> &p_node) const;
	void get_node_list(List<StringName> *r_nodes) const;

	void set_node_position(const StringName &p_name, const Vector2 &p_position);
	Vector2 get_node_position(const StringName &p_name) const;

	virtual void get_child_nodes(List<ChildNode> *r_child_nodes);

	bool has_transition(const StringName &p_from, const StringName &p_to) const;
	int find_transition(const StringName &p_from, const StringName &p_to) const;
	void add_transition(const StringName &p_from, const StringName &p_to, const Ref<AnimationNodeStateMachineTransition> &p_transition);
	Ref<AnimationNodeStateMachineTransition> get_transition(int p_transition) const;
	StringName get_transition_from(int p_transition) const;
	StringName get_transition_to(int p_transition) const;
	int get_transition_count() const;
	void remove_transition_by_index(int p_transition);
	void remove_transition(const StringName &p_from, const StringName &p_to);

	void set_start_node(const StringName &p_node);
	String get_start_node() const;

	void set_end_node(const StringName &p_node);
	String get_end_node() const;

	void set_graph_offset(const Vector2 &p_offset);
	Vector2 get_graph_offset() const;

	virtual float process(float p_time, bool p_seek);
	virtual String get_caption() const;

	virtual Ref<AnimationNode> get_child_by_name(const StringName &p_name);

	AnimationNodeStateMachine();
};

#endif // ANIMATION_NODE_STATE_MACHINE_H
