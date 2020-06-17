/*************************************************************************/
/*  animation_node_state_machine.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "animation_node_state_machine.h"

/////////////////////////////////////////////////

void AnimationNodeStateMachineTransition::set_switch_mode(SwitchMode p_mode) {
	switch_mode = p_mode;
}

AnimationNodeStateMachineTransition::SwitchMode AnimationNodeStateMachineTransition::get_switch_mode() const {
	return switch_mode;
}

void AnimationNodeStateMachineTransition::set_auto_advance(bool p_enable) {
	auto_advance = p_enable;
}

bool AnimationNodeStateMachineTransition::has_auto_advance() const {
	return auto_advance;
}

void AnimationNodeStateMachineTransition::set_advance_condition(const StringName &p_condition) {
	String cs = p_condition;
	ERR_FAIL_COND(cs.find("/") != -1 || cs.find(":") != -1);
	advance_condition = p_condition;
	if (cs != String()) {
		advance_condition_name = "conditions/" + cs;
	} else {
		advance_condition_name = StringName();
	}
	emit_signal("advance_condition_changed");
}

StringName AnimationNodeStateMachineTransition::get_advance_condition() const {
	return advance_condition;
}

StringName AnimationNodeStateMachineTransition::get_advance_condition_name() const {
	return advance_condition_name;
}

void AnimationNodeStateMachineTransition::set_xfade_time(float p_xfade) {
	ERR_FAIL_COND(p_xfade < 0);
	xfade = p_xfade;
	emit_changed();
}

float AnimationNodeStateMachineTransition::get_xfade_time() const {
	return xfade;
}

void AnimationNodeStateMachineTransition::set_actived(bool p_actived) {
	actived = p_actived;
}

bool AnimationNodeStateMachineTransition::is_actived() const {
	return actived;
}

void AnimationNodeStateMachineTransition::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	emit_changed();
}

bool AnimationNodeStateMachineTransition::is_disabled() const {
	return disabled;
}

void AnimationNodeStateMachineTransition::set_priority(int p_priority) {
	priority = p_priority;
	emit_changed();
}

int AnimationNodeStateMachineTransition::get_priority() const {
	return priority;
}

void AnimationNodeStateMachineTransition::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_switch_mode", "mode"), &AnimationNodeStateMachineTransition::set_switch_mode);
	ClassDB::bind_method(D_METHOD("get_switch_mode"), &AnimationNodeStateMachineTransition::get_switch_mode);

	ClassDB::bind_method(D_METHOD("set_auto_advance", "auto_advance"), &AnimationNodeStateMachineTransition::set_auto_advance);
	ClassDB::bind_method(D_METHOD("has_auto_advance"), &AnimationNodeStateMachineTransition::has_auto_advance);

	ClassDB::bind_method(D_METHOD("set_advance_condition", "name"), &AnimationNodeStateMachineTransition::set_advance_condition);
	ClassDB::bind_method(D_METHOD("get_advance_condition"), &AnimationNodeStateMachineTransition::get_advance_condition);

	ClassDB::bind_method(D_METHOD("set_xfade_time", "secs"), &AnimationNodeStateMachineTransition::set_xfade_time);
	ClassDB::bind_method(D_METHOD("get_xfade_time"), &AnimationNodeStateMachineTransition::get_xfade_time);

	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &AnimationNodeStateMachineTransition::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &AnimationNodeStateMachineTransition::is_disabled);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &AnimationNodeStateMachineTransition::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &AnimationNodeStateMachineTransition::get_priority);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "switch_mode", PROPERTY_HINT_ENUM, "Immediate,Sync,AtEnd"), "set_switch_mode", "get_switch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_advance"), "set_auto_advance", "has_auto_advance");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "advance_condition"), "set_advance_condition", "get_advance_condition");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "xfade_time", PROPERTY_HINT_RANGE, "0,240,0.01"), "set_xfade_time", "get_xfade_time");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "0,32,1"), "set_priority", "get_priority");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");

	BIND_ENUM_CONSTANT(SWITCH_MODE_IMMEDIATE);
	BIND_ENUM_CONSTANT(SWITCH_MODE_SYNC);
	BIND_ENUM_CONSTANT(SWITCH_MODE_AT_END);

	ADD_SIGNAL(MethodInfo("advance_condition_changed"));
}

AnimationNodeStateMachineTransition::AnimationNodeStateMachineTransition() {
	switch_mode = SWITCH_MODE_IMMEDIATE;
	auto_advance = false;
	xfade = 0;
	disabled = false;
	priority = 1;
	actived = false;
}

////////////////////////////////////////////////////////

void AnimationNodeStateMachinePlayback::travel(const StringName &p_state) {
	start_request_travel = true;
	start_request = p_state;
	stop_request = false;
}

void AnimationNodeStateMachinePlayback::start(const StringName &p_state) {
	start_request_travel = false;
	start_request = p_state;
	stop_request = false;
}

void AnimationNodeStateMachinePlayback::stop() {
	stop_request = true;
}

bool AnimationNodeStateMachinePlayback::is_playing() const {
	return playing;
}

StringName AnimationNodeStateMachinePlayback::get_current_node() const {
	return current;
}

StringName AnimationNodeStateMachinePlayback::get_blend_from_node() const {
	return fading_from;
}

Vector<StringName> AnimationNodeStateMachinePlayback::get_travel_path() const {
	return path;
}

float AnimationNodeStateMachinePlayback::get_current_play_pos() const {
	return pos_current;
}

float AnimationNodeStateMachinePlayback::get_current_length() const {
	return len_current;
}

bool AnimationNodeStateMachinePlayback::_travel(AnimationNodeStateMachine *p_state_machine, const StringName &p_travel) {
	ERR_FAIL_COND_V(!playing, false);
	ERR_FAIL_COND_V(!p_state_machine->states.has(p_travel), false);
	ERR_FAIL_COND_V(!p_state_machine->states.has(current), false);

	path.clear(); //a new one will be needed

	if (current == p_travel) {
		return true; //nothing to do
	}

	loops_current = 0; // reset loops, so fade does not happen immediately

	Vector2 current_pos = p_state_machine->states[current].position;
	Vector2 target_pos = p_state_machine->states[p_travel].position;

	Map<StringName, AStarCost> cost_map;

	List<int> open_list;

	//build open list
	for (int i = 0; i < p_state_machine->transitions.size(); i++) {
		if (p_state_machine->transitions[i].local_from() == current) {
			open_list.push_back(i);
			float cost = p_state_machine->states[p_state_machine->transitions[i].local_to()].position.distance_to(current_pos);
			cost *= p_state_machine->transitions[i].transition->get_priority();
			AStarCost ap;
			ap.prev = current;
			ap.distance = cost;
			cost_map[p_state_machine->transitions[i].local_to()] = ap;

			if (p_state_machine->transitions[i].local_to() == p_travel) { //prematurely found it! :D
				path.push_back(p_travel);
				return true;
			}
		}
	}

	//begin astar
	bool found_route = false;
	while (!found_route) {
		if (open_list.size() == 0) {
			return false; //no path found
		}

		//find the last cost transition
		List<int>::Element *least_cost_transition = nullptr;
		float least_cost = 1e20;

		for (List<int>::Element *E = open_list.front(); E; E = E->next()) {
			float cost = cost_map[p_state_machine->transitions[E->get()].local_to()].distance;
			cost += p_state_machine->states[p_state_machine->transitions[E->get()].local_to()].position.distance_to(target_pos);

			if (cost < least_cost) {
				least_cost_transition = E;
				least_cost = cost;
			}
		}

		StringName transition_prev = p_state_machine->transitions[least_cost_transition->get()].local_from();
		StringName transition = p_state_machine->transitions[least_cost_transition->get()].local_to();

		for (int i = 0; i < p_state_machine->transitions.size(); i++) {
			if (p_state_machine->transitions[i].local_from() != transition || p_state_machine->transitions[i].local_to() == transition_prev) {
				continue; //not interested on those
			}

			float distance = p_state_machine->states[p_state_machine->transitions[i].local_from()].position.distance_to(p_state_machine->states[p_state_machine->transitions[i].local_to()].position);
			distance *= p_state_machine->transitions[i].transition->get_priority();
			distance += cost_map[p_state_machine->transitions[i].local_from()].distance;

			if (cost_map.has(p_state_machine->transitions[i].local_to())) {
				//oh this was visited already, can we win the cost?
				if (distance < cost_map[p_state_machine->transitions[i].local_to()].distance) {
					cost_map[p_state_machine->transitions[i].local_to()].distance = distance;
					cost_map[p_state_machine->transitions[i].local_to()].prev = p_state_machine->transitions[i].local_from();
				}
			} else {
				//add to open list
				AStarCost ac;
				ac.prev = p_state_machine->transitions[i].local_from();
				ac.distance = distance;
				cost_map[p_state_machine->transitions[i].local_to()] = ac;

				open_list.push_back(i);

				if (p_state_machine->transitions[i].local_to() == p_travel) {
					found_route = true;
					break;
				}
			}
		}

		if (found_route) {
			break;
		}

		open_list.erase(least_cost_transition);
	}

	//make path
	StringName at = p_travel;
	while (at != current) {
		path.push_back(at);
		at = cost_map[at].prev;
	}

	path.invert();

	return true;
}

float AnimationNodeStateMachinePlayback::process(AnimationNodeStateMachine *p_state_machine, float p_time, bool p_seek) {
	//if not playing and it can restart, then restart
	if (!playing && start_request == StringName()) {
		if (!stop_request && p_state_machine->start_node) {
			start(p_state_machine->start_node);
		} else {
			return 0;
		}
	}

	if (playing && stop_request) {
		stop_request = false;
		playing = false;
		return 0;
	}

	bool play_start = false;

	if (start_request != StringName()) {
		if (start_request_travel) {
			if (!playing) {
				if (!stop_request && p_state_machine->start_node) {
					// can restart, just postpone traveling
					path.clear();
					current = p_state_machine->start_node;
					playing = true;
					play_start = true;
				} else {
					// stopped, invalid state
					String node_name = start_request;
					start_request = StringName(); //clear start request
					ERR_FAIL_V_MSG(0, "Can't travel to '" + node_name + "' if state machine is not playing.");
				}
			} else {
				if (!_travel(p_state_machine, start_request)) {
					// can't travel, then teleport
					path.clear();
					current = start_request;
				}
				start_request = StringName(); //clear start request
			}
		} else {
			// teleport to start
			path.clear();
			current = start_request;
			playing = true;
			play_start = true;
			start_request = StringName(); //clear start request
		}
	}

	bool do_start = (p_seek && p_time == 0) || play_start || current == StringName();

	if (do_start) {
		if (p_state_machine->start_node != StringName() && p_seek && p_time == 0) {
			current = p_state_machine->start_node;
		}

		len_current = p_state_machine->blend_node(current, p_state_machine->states[current].node, 0, true, 1.0, AnimationNode::FILTER_IGNORE, false);
		pos_current = 0;
		loops_current = 0;
	}

	if (!p_state_machine->states.has(current)) {
		playing = false; //current does not exist
		current = StringName();
		return 0;
	}
	float fade_blend = 1.0;

	if (fading_from != StringName()) {
		if (!p_state_machine->states.has(fading_from)) {
			fading_from = StringName();
		} else {
			if (!p_seek) {
				fading_pos += p_time;
			}
			fade_blend = MIN(1.0, fading_pos / fading_time);
			if (fade_blend >= 1.0) {
				fading_from = StringName();
			}
		}
	}

	float rem = p_state_machine->blend_node(current, p_state_machine->states[current].node, p_time, p_seek, fade_blend, AnimationNode::FILTER_IGNORE, false);

	if (fading_from != StringName()) {
		p_state_machine->blend_node(fading_from, p_state_machine->states[fading_from].node, p_time, p_seek, 1.0 - fade_blend, AnimationNode::FILTER_IGNORE, false);
	}

	//guess playback position
	if (rem > len_current) { // weird but ok
		len_current = rem;
	}

	{ //advance and loop check

		float next_pos = len_current - rem;

		if (next_pos < pos_current) {
			loops_current++;
		}
		pos_current = next_pos; //looped
	}

	//find next
	StringName next;
	Ref<AnimationNodeStateMachineTransition> current_tr;
	float next_xfade = 0;
	AnimationNodeStateMachineTransition::SwitchMode switch_mode = AnimationNodeStateMachineTransition::SWITCH_MODE_IMMEDIATE;

	if (path.size()) {
		for (int i = 0; i < p_state_machine->transitions.size(); i++) {
			if (p_state_machine->transitions[i].local_from() == current && p_state_machine->transitions[i].local_to() == path[0]) {
				next_xfade = p_state_machine->transitions[i].transition->get_xfade_time();
				switch_mode = p_state_machine->transitions[i].transition->get_switch_mode();
				next = path[0];
			}
		}
	} else {
		float priority_best = 1e20;
		int auto_advance_to = -1;
		for (int i = 0; i < p_state_machine->transitions.size(); i++) {
			bool auto_advance = false;

			if (p_state_machine->transitions[i].from == current) {
				if (p_state_machine->transitions[i].transition->has_auto_advance()) {
					auto_advance = true;
				}
				StringName advance_condition_name = p_state_machine->transitions[i].transition->get_advance_condition_name();
				if (advance_condition_name != StringName() && bool(p_state_machine->get_parameter(advance_condition_name))) {
					auto_advance = true;
				}
			}

			if (p_state_machine->transitions[i].local_from() == current) {
				if (p_state_machine->transitions[i].transition->is_actived()) {
					auto_advance_to = i;
					break;
				}

				if (auto_advance) {
					if (p_state_machine->transitions[i].transition->get_priority() <= priority_best) {
						priority_best = p_state_machine->transitions[i].transition->get_priority();
						auto_advance_to = i;
					}
				}
			}
		}

		if (auto_advance_to != -1) {
			next = p_state_machine->transitions[auto_advance_to].local_to();
			current_tr = p_state_machine->transitions[auto_advance_to].transition;
			next_xfade = p_state_machine->transitions[auto_advance_to].transition->get_xfade_time();
			switch_mode = p_state_machine->transitions[auto_advance_to].transition->get_switch_mode();
		}
	}

	//if next, see when to transition
	if (next != StringName()) {
		bool goto_next = false;

		if (switch_mode == AnimationNodeStateMachineTransition::SWITCH_MODE_AT_END) {
			goto_next = next_xfade >= (len_current - pos_current) || loops_current > 0;
			if (loops_current > 0) {
				next_xfade = 0;
			}
		} else {
			goto_next = fading_from == StringName();
		}

		if (goto_next) { //loops should be used because fade time may be too small or zero and animation may have looped

			if (current_tr.is_valid()) {
				for (int i = 0; i < p_state_machine->get_transition_count(); i++) {
					p_state_machine->get_transition(i)->set_actived(false);
				}
				current_tr->set_actived(true);
			}

			if (next_xfade) {
				//time to fade, baby
				fading_from = current;
				fading_time = next_xfade;
				fading_pos = 0;
			} else {
				fading_from = StringName();
				fading_pos = 0;
			}

			if (path.size()) { //if it came from path, remove path
				path.remove(0);
			}
			current = next;
			if (switch_mode == AnimationNodeStateMachineTransition::SWITCH_MODE_SYNC) {
				len_current = p_state_machine->blend_node(current, p_state_machine->states[current].node, 0, true, 0, AnimationNode::FILTER_IGNORE, false);
				pos_current = MIN(pos_current, len_current);
				p_state_machine->blend_node(current, p_state_machine->states[current].node, pos_current, true, 0, AnimationNode::FILTER_IGNORE, false);

			} else {
				len_current = p_state_machine->blend_node(current, p_state_machine->states[current].node, 0, true, 0, AnimationNode::FILTER_IGNORE, false);
				pos_current = 0;
			}

			rem = len_current; //so it does not show 0 on transition
			loops_current = 0;
		}
	}

	//compute time left for transitions by using the end node
	if (p_state_machine->end_node != current) {
		rem = p_state_machine->blend_node(p_state_machine->end_node, p_state_machine->states[p_state_machine->end_node].node, 0, true, 0, AnimationNode::FILTER_IGNORE, false);
	}

	return rem;
}

void AnimationNodeStateMachinePlayback::_bind_methods() {
	ClassDB::bind_method(D_METHOD("travel", "to_node"), &AnimationNodeStateMachinePlayback::travel);
	ClassDB::bind_method(D_METHOD("start", "node"), &AnimationNodeStateMachinePlayback::start);
	ClassDB::bind_method(D_METHOD("stop"), &AnimationNodeStateMachinePlayback::stop);
	ClassDB::bind_method(D_METHOD("is_playing"), &AnimationNodeStateMachinePlayback::is_playing);
	ClassDB::bind_method(D_METHOD("get_current_node"), &AnimationNodeStateMachinePlayback::get_current_node);
	ClassDB::bind_method(D_METHOD("get_travel_path"), &AnimationNodeStateMachinePlayback::get_travel_path);
}

AnimationNodeStateMachinePlayback::AnimationNodeStateMachinePlayback() {
	set_local_to_scene(true); //only one per instanced scene

	playing = false;
	len_current = 0;
	fading_time = 0;
	stop_request = false;
	len_total = 0.0;
	pos_current = 0.0;
	loops_current = 0;
	fading_pos = 0.0;
	start_request_travel = false;
}

///////////////////////////////////////////////////////

void AnimationNodeStateMachine::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::OBJECT, playback, PROPERTY_HINT_RESOURCE_TYPE, "AnimationNodeStateMachinePlayback", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
	List<StringName> advance_conditions;
	for (int i = 0; i < transitions.size(); i++) {
		StringName ac = transitions[i].transition->get_advance_condition_name();
		if (ac != StringName() && advance_conditions.find(ac) == nullptr) {
			advance_conditions.push_back(ac);
		}
	}

	advance_conditions.sort_custom<StringName::AlphCompare>();
	for (List<StringName>::Element *E = advance_conditions.front(); E; E = E->next()) {
		r_list->push_back(PropertyInfo(Variant::BOOL, E->get()));
	}
}

Variant AnimationNodeStateMachine::get_parameter_default_value(const StringName &p_parameter) const {
	if (p_parameter == playback) {
		Ref<AnimationNodeStateMachinePlayback> p;
		p.instance();
		return p;
	} else {
		return false; //advance condition
	}
}

void AnimationNodeStateMachine::add_node(const StringName &p_name, Ref<AnimationNode> p_node, const Vector2 &p_position) {
	ERR_FAIL_COND(states.has(p_name));
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(String(p_name).find("/") != -1);

	State state;
	state.node = p_node;
	state.position = p_position;

	states[p_name] = state;

	Ref<AnimationNodeStateMachine> anodesm = p_node;

	if (anodesm.is_valid()) {
		anodesm->state_machine_name = p_name;
		anodesm->prev_state_machine = (Ref<AnimationNodeStateMachine>)this;
	}

	emit_changed();
	emit_signal("tree_changed");

	p_node->connect("tree_changed", callable_mp(this, &AnimationNodeStateMachine::_tree_changed), varray(), CONNECT_REFERENCE_COUNTED);
}

void AnimationNodeStateMachine::replace_node(const StringName &p_name, Ref<AnimationNode> p_node) {
	ERR_FAIL_COND(states.has(p_name) == false);
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(String(p_name).find("/") != -1);

	{
		Ref<AnimationNode> node = states[p_name].node;
		if (node.is_valid()) {
			node->disconnect_compat("tree_changed", this, "_tree_changed");
		}
	}

	states[p_name].node = p_node;

	emit_changed();
	emit_signal("tree_changed");

	p_node->connect_compat("tree_changed", this, "_tree_changed", varray(), CONNECT_REFERENCE_COUNTED);
}

bool AnimationNodeStateMachine::can_edit_node(const StringName &p_name) const {
	if (states.has(p_name)) {
		return !(states[p_name].node->is_class("AnimationNodeStart") || states[p_name].node->is_class("AnimationNodeEnd"));
	}

	return true;
}

Ref<AnimationNode> AnimationNodeStateMachine::get_node(const StringName &p_name) const {
	ERR_FAIL_COND_V(!states.has(p_name), Ref<AnimationNode>());

	return states[p_name].node;
}

StringName AnimationNodeStateMachine::get_node_name(const Ref<AnimationNode> &p_node) const {
	for (Map<StringName, State>::Element *E = states.front(); E; E = E->next()) {
		if (E->get().node == p_node) {
			return E->key();
		}
	}

	ERR_FAIL_V(StringName());
}

void AnimationNodeStateMachine::get_child_nodes(List<ChildNode> *r_child_nodes) {
	Vector<StringName> nodes;

	for (Map<StringName, State>::Element *E = states.front(); E; E = E->next()) {
		nodes.push_back(E->key());
	}

	nodes.sort_custom<StringName::AlphCompare>();

	for (int i = 0; i < nodes.size(); i++) {
		ChildNode cn;
		cn.name = nodes[i];
		cn.node = states[cn.name].node;
		r_child_nodes->push_back(cn);
	}
}

bool AnimationNodeStateMachine::has_node(const StringName &p_name) const {
	return states.has(p_name);
}

void AnimationNodeStateMachine::remove_node(const StringName &p_name) {
	ERR_FAIL_COND(!states.has(p_name));

	if (!can_edit_node(p_name)) {
		return;
	}

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].local_from() == p_name || transitions[i].local_to() == p_name) {
			remove_transition_by_index(i);
			i--;
		}
	}

	{
		Ref<AnimationNode> node = states[p_name].node;
		ERR_FAIL_COND(node.is_null());
		node->disconnect("tree_changed", callable_mp(this, &AnimationNodeStateMachine::_tree_changed));
	}

	states.erase(p_name);

	emit_changed();
	emit_signal("tree_changed");
}

void AnimationNodeStateMachine::rename_node(const StringName &p_name, const StringName &p_new_name) {
	ERR_FAIL_COND(!states.has(p_name));
	ERR_FAIL_COND(states.has(p_new_name));
	ERR_FAIL_COND(!can_edit_node(p_name));

	states[p_new_name] = states[p_name];
	states.erase(p_name);

	Ref<AnimationNodeStateMachine> anodesm = states[p_new_name].node;
	if (anodesm.is_valid()) {
		anodesm->state_machine_name = p_new_name;
	}

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].local_from() == p_name) {
			_rename_transition(transitions[i].from, String(transitions[i].from).replace_first(p_name, p_new_name));
		}

		if (transitions[i].local_to() == p_name) {
			_rename_transition(transitions[i].to, String(transitions[i].to).replace_first(p_name, p_new_name));
		}
	}

	emit_signal("tree_changed");
}

void AnimationNodeStateMachine::_rename_transition(const StringName &p_name, const StringName &p_new_name) {
	if (updating_transitions) {
		return;
	}

	updating_transitions = true;
	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == p_name) {
			Vector<String> path = String(transitions[i].to).split("/");
			if (path.size() > 1) {
				if (path[0] == "..") {
					prev_state_machine->_rename_transition(String(state_machine_name) + "/" + p_name, String(state_machine_name) + "/" + p_new_name);
				} else {
					((Ref<AnimationNodeStateMachine>)states[transitions[i].local_to()].node)->_rename_transition("../" + p_name, "../" + p_new_name);
				}
			}

			transitions.write[i].from = p_new_name;
		}

		if (transitions[i].to == p_name) {
			Vector<String> path = String(transitions[i].from).split("/");
			if (path.size() > 1) {
				if (path[0] == "..") {
					prev_state_machine->_rename_transition(String(state_machine_name) + "/" + p_name, String(state_machine_name) + "/" + p_new_name);
				} else {
					((Ref<AnimationNodeStateMachine>)states[transitions[i].local_from()].node)->_rename_transition("../" + p_name, "../" + p_new_name);
				}
			}

			transitions.write[i].to = p_new_name;
		}

		updating_transitions = false;
	}
}

void AnimationNodeStateMachine::get_node_list(List<StringName> *r_nodes) const {
	List<StringName> nodes;
	for (Map<StringName, State>::Element *E = states.front(); E; E = E->next()) {
		if (E->key() == end_node && !prev_state_machine.is_valid()) {
			continue;
		}

		nodes.push_back(E->key());
	}
	nodes.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E = nodes.front(); E; E = E->next()) {
		r_nodes->push_back(E->get());
	}
}

Ref<AnimationNodeStateMachine> AnimationNodeStateMachine::get_prev_state_machine() const {
	return prev_state_machine;
}

bool AnimationNodeStateMachine::has_transition(const StringName &p_from, const StringName &p_to) const {
	StringName from = _get_shortest_path(p_from);
	StringName to = _get_shortest_path(p_to);

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == from && transitions[i].to == to) {
			return true;
		}
	}
	return false;
}

int AnimationNodeStateMachine::find_transition(const StringName &p_from, const StringName &p_to) const {
	StringName from = _get_shortest_path(p_from);
	StringName to = _get_shortest_path(p_to);

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == from && transitions[i].to == to) {
			return i;
		}
	}
	return -1;
}

bool AnimationNodeStateMachine::_can_connect(const StringName &p_name, Vector<Ref<AnimationNodeStateMachine>> p_parents) const {
	if (p_parents.empty()) {
		Ref<AnimationNodeStateMachine> prev = (Ref<AnimationNodeStateMachine>)this;
		while (prev.is_valid()) {
			p_parents.push_back(prev);
			prev = prev->prev_state_machine;
		}
	}

	if (states.has(p_name)) {
		Ref<AnimationNodeStateMachine> anodesm = states[p_name].node;

		if (anodesm.is_valid() && p_parents.find(anodesm) != -1) {
			return false;
		}

		return true;
	}

	String name = p_name;
	Vector<String> path = name.split("/");

	if (path.size() < 2) {
		return false;
	}

	if (path[0] == "..") {
		if (prev_state_machine.is_valid()) {
			return prev_state_machine->_can_connect(name.replace_first("../", ""), p_parents);
		}
	} else if (states.has(path[0])) {
		Ref<AnimationNodeStateMachine> anodesm = states[path[0]].node;
		if (anodesm.is_valid()) {
			return anodesm->_can_connect(name.replace_first(path[0] + "/", ""), p_parents);
		}
	}

	return false;
}

StringName AnimationNodeStateMachine::_get_shortest_path(const StringName &p_path) const {
	// If p_path is something like StateMachine/../StateMachine2/State1,
	// the result will be StateMachine2/State1. This avoid duplicate
	// transitions when using add_transition. eg, this two calls is the same:
	//
	// add_transition("State1", "StateMachine/../State2", tr)
	// add_transition("State1", "State2", tr)
	//
	// but the second call must be invalid because the transition already exists

	Vector<String> path = String(p_path).split("/");
	Vector<String> new_path;

	for (int i = 0; i < path.size(); i++) {
		if (i > 0 && path[i] == ".." && new_path[i - 1] != "..") {
			new_path.remove(i - 1);
		} else {
			new_path.push_back(path[i]);
		}
	}

	String result;
	for (int i = 0; i < new_path.size(); i++) {
		result += new_path[i] + "/";
	}
	result.remove(result.length() - 1);

	return result;
}

void AnimationNodeStateMachine::add_transition(const StringName &p_from, const StringName &p_to, const Ref<AnimationNodeStateMachineTransition> &p_transition) {
	if (updating_transitions) {
		return;
	}

	StringName from = _get_shortest_path(p_from);
	StringName to = _get_shortest_path(p_to);
	Vector<String> path_from = String(from).split("/");
	Vector<String> path_to = String(to).split("/");

	ERR_FAIL_COND(from == end_node || to == start_node);
	ERR_FAIL_COND(from == to);
	ERR_FAIL_COND(!_can_connect(from));
	ERR_FAIL_COND(!_can_connect(to));
	ERR_FAIL_COND(p_transition.is_null());

	for (int i = 0; i < transitions.size(); i++) {
		ERR_FAIL_COND(transitions[i].from == from && transitions[i].to == to);
	}

	if (path_from.size() > 1 || path_to.size() > 1) {
		ERR_FAIL_COND(path_from[0] == path_to[0]);
	}

	updating_transitions = true;

	Transition tr;
	tr.from = from;
	tr.to = to;
	tr.transition = p_transition;

	tr.transition->connect("advance_condition_changed", callable_mp(this, &AnimationNodeStateMachine::_tree_changed), varray(), CONNECT_REFERENCE_COUNTED);

	transitions.push_back(tr);

	// do recursive
	if (path_from.size() > 1) {
		StringName local_path = String(from).replace_first(path_from[0] + "/", "");
		if (path_from[0] == "..") {
			prev_state_machine->add_transition(local_path, String(state_machine_name) + "/" + to, p_transition);
		} else {
			((Ref<AnimationNodeStateMachine>)states[path_from[0]].node)->add_transition(local_path, "../" + to, p_transition);
		}
	}
	if (path_to.size() > 1) {
		StringName local_path = String(to).replace_first(path_to[0] + "/", "");
		if (path_to[0] == "..") {
			prev_state_machine->add_transition(String(state_machine_name) + "/" + from, local_path, p_transition);
		} else {
			((Ref<AnimationNodeStateMachine>)states[path_to[0]].node)->add_transition("../" + from, local_path, p_transition);
		}
	}

	updating_transitions = false;
}

Ref<AnimationNodeStateMachineTransition> AnimationNodeStateMachine::get_transition(int p_transition) const {
	ERR_FAIL_INDEX_V(p_transition, transitions.size(), Ref<AnimationNodeStateMachineTransition>());
	return transitions[p_transition].transition;
}

StringName AnimationNodeStateMachine::get_transition_from(int p_transition) const {
	ERR_FAIL_INDEX_V(p_transition, transitions.size(), StringName());
	return transitions[p_transition].from;
}

StringName AnimationNodeStateMachine::get_transition_to(int p_transition) const {
	ERR_FAIL_INDEX_V(p_transition, transitions.size(), StringName());
	return transitions[p_transition].to;
}

int AnimationNodeStateMachine::get_transition_count() const {
	return transitions.size();
}

void AnimationNodeStateMachine::remove_transition(const StringName &p_from, const StringName &p_to) {
	StringName from = _get_shortest_path(p_from);
	StringName to = _get_shortest_path(p_to);

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == from && transitions[i].to == to) {
			remove_transition_by_index(i);
			return;
		}
	}
}

void AnimationNodeStateMachine::remove_transition_by_index(int p_transition) {
	ERR_FAIL_INDEX(p_transition, transitions.size());
	Transition tr = transitions[p_transition];
	transitions.write[p_transition].transition->disconnect("advance_condition_changed", callable_mp(this, &AnimationNodeStateMachine::_tree_changed));
	transitions.remove(p_transition);

	Vector<String> path_from = String(tr.from).split("/");
	Vector<String> path_to = String(tr.to).split("/");

	List<Vector<String>> paths;
	paths.push_back(path_from);
	paths.push_back(path_to);

	for (List<Vector<String>>::Element *E = paths.front(); E; E = E->next()) {
		if (E->get()[0].size() > 1) {
			if (E->get()[0] == "..") {
				prev_state_machine->_remove_transition(tr.transition);
			} else if (states.has(E->get()[0])) {
				Ref<AnimationNodeStateMachine> anodesm = states[E->get()[0]].node;

				if (anodesm.is_valid()) {
					anodesm->_remove_transition(tr.transition);
				}
			}
		}
	}
}

void AnimationNodeStateMachine::_remove_transition(Ref<AnimationNodeStateMachineTransition> p_transition) {
	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].transition == p_transition) {
			remove_transition_by_index(i);
			return;
		}
	}
}

void AnimationNodeStateMachine::set_graph_offset(const Vector2 &p_offset) {
	graph_offset = p_offset;
}

Vector2 AnimationNodeStateMachine::get_graph_offset() const {
	return graph_offset;
}

float AnimationNodeStateMachine::process(float p_time, bool p_seek) {
	Ref<AnimationNodeStateMachinePlayback> playback = get_parameter(this->playback);
	ERR_FAIL_COND_V(playback.is_null(), 0.0);

	return playback->process(this, p_time, p_seek);
}

String AnimationNodeStateMachine::get_caption() const {
	return "StateMachine";
}

bool AnimationNodeStateMachine::has_local_transition(const StringName &p_from, const StringName &p_to) const {
	StringName from = _get_shortest_path(p_from);
	StringName to = _get_shortest_path(p_to);

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].local_from() == from && transitions[i].local_to() == to) {
			return true;
		}
	}
	return false;
}

void AnimationNodeStateMachine::_notification(int p_what) {
}

Ref<AnimationNode> AnimationNodeStateMachine::get_child_by_name(const StringName &p_name) {
	return get_node(p_name);
}

bool AnimationNodeStateMachine::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name.begins_with("states/")) {
		String node_name = name.get_slicec('/', 1);
		String what = name.get_slicec('/', 2);

		if (what == "node") {
			Ref<AnimationNode> anode = p_value;
			if (anode.is_valid()) {
				add_node(node_name, p_value);
			}
			return true;
		}

		if (what == "position") {
			if (states.has(node_name)) {
				states[node_name].position = p_value;
			}
			return true;
		}
	} else if (name == "transitions") {
		Array trans = p_value;
		ERR_FAIL_COND_V(trans.size() % 3 != 0, false);

		for (int i = 0; i < trans.size(); i += 3) {
			add_transition(trans[i], trans[i + 1], trans[i + 2]);
		}
		return true;
	} else if (name == "graph_offset") {
		set_graph_offset(p_value);
		return true;
	}

	return false;
}

bool AnimationNodeStateMachine::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name.begins_with("states/")) {
		String node_name = name.get_slicec('/', 1);
		String what = name.get_slicec('/', 2);

		if (what == "node") {
			if (states.has(node_name) && can_edit_node(node_name)) {
				r_ret = states[node_name].node;
				return true;
			}
		}

		if (what == "position") {
			if (states.has(node_name)) {
				r_ret = states[node_name].position;
				return true;
			}
		}
	} else if (name == "transitions") {
		Array trans;
		for (int i = 0; i < transitions.size(); i++) {
			String from = transitions[i].from;
			String to = transitions[i].to;

			if (from.get_slicec('/', 0) == ".." || to.get_slicec('/', 0) == "..") {
				continue;
			}

			trans.push_back(from);
			trans.push_back(to);
			trans.push_back(transitions[i].transition);
		}

		r_ret = trans;
		return true;
	} else if (name == "graph_offset") {
		r_ret = get_graph_offset();
		return true;
	}

	return false;
}

void AnimationNodeStateMachine::_get_property_list(List<PropertyInfo> *p_list) const {
	List<StringName> names;
	for (Map<StringName, State>::Element *E = states.front(); E; E = E->next()) {
		names.push_back(E->key());
	}
	names.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		String name = E->get();
		p_list->push_back(PropertyInfo(Variant::OBJECT, "states/" + name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "AnimationNode", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "states/" + name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "transitions", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::VECTOR2, "graph_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
}

void AnimationNodeStateMachine::set_node_position(const StringName &p_name, const Vector2 &p_position) {
	ERR_FAIL_COND(!states.has(p_name));
	states[p_name].position = p_position;
}

Vector2 AnimationNodeStateMachine::get_node_position(const StringName &p_name) const {
	ERR_FAIL_COND_V(!states.has(p_name), Vector2());
	return states[p_name].position;
}

void AnimationNodeStateMachine::_tree_changed() {
	emit_signal("tree_changed");
}

void AnimationNodeStateMachine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_node", "name", "node", "position"), &AnimationNodeStateMachine::add_node, DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("replace_node", "name", "node"), &AnimationNodeStateMachine::replace_node);
	ClassDB::bind_method(D_METHOD("get_node", "name"), &AnimationNodeStateMachine::get_node);
	ClassDB::bind_method(D_METHOD("remove_node", "name"), &AnimationNodeStateMachine::remove_node);
	ClassDB::bind_method(D_METHOD("rename_node", "name", "new_name"), &AnimationNodeStateMachine::rename_node);
	ClassDB::bind_method(D_METHOD("has_node", "name"), &AnimationNodeStateMachine::has_node);
	ClassDB::bind_method(D_METHOD("get_node_name", "node"), &AnimationNodeStateMachine::get_node_name);

	ClassDB::bind_method(D_METHOD("set_node_position", "name", "position"), &AnimationNodeStateMachine::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "name"), &AnimationNodeStateMachine::get_node_position);

	ClassDB::bind_method(D_METHOD("has_transition", "from", "to"), &AnimationNodeStateMachine::has_transition);
	ClassDB::bind_method(D_METHOD("add_transition", "from", "to", "transition"), &AnimationNodeStateMachine::add_transition);
	ClassDB::bind_method(D_METHOD("get_transition", "idx"), &AnimationNodeStateMachine::get_transition);
	ClassDB::bind_method(D_METHOD("get_transition_from", "idx"), &AnimationNodeStateMachine::get_transition_from);
	ClassDB::bind_method(D_METHOD("get_transition_to", "idx"), &AnimationNodeStateMachine::get_transition_to);
	ClassDB::bind_method(D_METHOD("get_transition_count"), &AnimationNodeStateMachine::get_transition_count);
	ClassDB::bind_method(D_METHOD("remove_transition_by_index", "idx"), &AnimationNodeStateMachine::remove_transition_by_index);
	ClassDB::bind_method(D_METHOD("remove_transition", "from", "to"), &AnimationNodeStateMachine::remove_transition);

	ClassDB::bind_method(D_METHOD("set_graph_offset", "offset"), &AnimationNodeStateMachine::set_graph_offset);
	ClassDB::bind_method(D_METHOD("get_graph_offset"), &AnimationNodeStateMachine::get_graph_offset);
}

AnimationNodeStateMachine::AnimationNodeStateMachine() {
	playback = "playback";
	updating_transitions = false;

	start_node = "Start";
	Ref<AnimationNodeStart> s;
	s.instance();
	State start;
	start.node = s;
	start.position = Vector2(200, 100);
	states[start_node] = start;

	end_node = "End";
	Ref<AnimationNodeEnd> e;
	e.instance();
	State end;
	end.node = e;
	end.position = Vector2(900, 100);
	states[end_node] = end;
}

/////////////////////////////////////////////

void MultiAnimationNodeStateMachineTransition::add_transition(const StringName &p_from, const StringName &p_to, Ref<AnimationNodeStateMachineTransition> p_transition) {
	Transition tr;
	tr.from = p_from;
	tr.to = p_to;
	tr.transition = p_transition;
	transitions.push_back(tr);
}

bool MultiAnimationNodeStateMachineTransition::_set(const StringName &p_name, const Variant &p_property) {
	int index = String(p_name).get_slicec('/', 0).to_int();
	StringName prop = String(p_name).get_slicec('/', 1);

	bool found;
	transitions.write[index].transition->set(prop, p_property, &found);
	if (found) {
		return true;
	}

	return false;
}

bool MultiAnimationNodeStateMachineTransition::_get(const StringName &p_name, Variant &r_property) const {
	int index = String(p_name).get_slicec('/', 0).to_int();
	StringName prop = String(p_name).get_slicec('/', 1);

	if (prop == "transition_path") {
		r_property = String(transitions[index].from) + " -> " + transitions[index].to;
		return true;
	}

	bool found;
	r_property = transitions[index].transition->get(prop, &found);
	if (found) {
		return true;
	}

	return false;
}

void MultiAnimationNodeStateMachineTransition::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < transitions.size(); i++) {
		List<PropertyInfo> plist;
		transitions[i].transition->get_property_list(&plist, true);

		PropertyInfo prop_transition_path;
		prop_transition_path.type = Variant::STRING;
		prop_transition_path.name = itos(i) + "/" + "transition_path";
		p_list->push_back(prop_transition_path);

		for (List<PropertyInfo>::Element *F = plist.front(); F; F = F->next()) {
			if (F->get().name == "script" || F->get().name == "resource_name" || F->get().name == "resource_path" || F->get().name == "resource_local_to_scene") {
				continue;
			}

			if (F->get().usage != PROPERTY_USAGE_DEFAULT) {
				continue;
			}

			PropertyInfo prop = F->get();
			prop.name = itos(i) + "/" + prop.name;

			p_list->push_back(prop);
		}
	}
}
