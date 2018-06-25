#ifndef ANIMATION_NODE_STATE_MACHINE_H
#define ANIMATION_NODE_STATE_MACHINE_H

#include "scene/animation/animation_tree.h"

class AnimationNodeStateMachineTransition : public Resource {
	GDCLASS(AnimationNodeStateMachineTransition, Resource)
public:
	enum SwitchMode {
		SWITCH_MODE_IMMEDIATE,
		SWITCH_MODE_SYNC,
		SWITCH_MODE_AT_END,
	};

private:
	SwitchMode switch_mode;
	bool auto_advance;
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

	void set_xfade_time(float p_xfade);
	float get_xfade_time() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	void set_priority(int p_priority);
	int get_priority() const;

	AnimationNodeStateMachineTransition();
};

VARIANT_ENUM_CAST(AnimationNodeStateMachineTransition::SwitchMode)

class AnimationNodeStateMachine : public AnimationRootNode {

	GDCLASS(AnimationNodeStateMachine, AnimationRootNode);

private:
	Map<StringName, Ref<AnimationRootNode> > states;

	struct Transition {

		StringName from;
		StringName to;
		Ref<AnimationNodeStateMachineTransition> transition;
	};

	struct AStarCost {
		float distance;
		StringName prev;
	};

	Vector<Transition> transitions;

	float len_total;

	float len_current;
	float pos_current;
	int loops_current;

	bool play_start;
	StringName start_node;
	StringName end_node;

	Vector2 graph_offset;

	StringName current;

	StringName fading_from;
	float fading_time;
	float fading_pos;

	Vector<StringName> path;
	bool playing;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void add_node(const StringName &p_name, Ref<AnimationNode> p_node);
	Ref<AnimationNode> get_node(const StringName &p_name) const;
	void remove_node(const StringName &p_name);
	void rename_node(const StringName &p_name, const StringName &p_new_name);
	bool has_node(const StringName &p_name) const;
	StringName get_node_name(const Ref<AnimationNode> &p_node) const;
	void get_node_list(List<StringName> *r_nodes) const;

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

	bool travel(const StringName &p_state);
	void start(const StringName &p_state);
	void stop();
	bool is_playing() const;
	StringName get_current_node() const;
	StringName get_blend_from_node() const;
	Vector<StringName> get_travel_path() const;
	float get_current_play_pos() const;
	float get_current_length() const;

	virtual void set_tree(AnimationTree *p_player);

	AnimationNodeStateMachine();
};

#endif // ANIMATION_NODE_STATE_MACHINE_H
