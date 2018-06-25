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

void AnimationNodeStateMachineTransition::set_xfade_time(float p_xfade) {

	ERR_FAIL_COND(p_xfade < 0);
	xfade = p_xfade;
	emit_changed();
}

float AnimationNodeStateMachineTransition::get_xfade_time() const {
	return xfade;
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

	ClassDB::bind_method(D_METHOD("set_xfade_time", "secs"), &AnimationNodeStateMachineTransition::set_xfade_time);
	ClassDB::bind_method(D_METHOD("get_xfade_time"), &AnimationNodeStateMachineTransition::get_xfade_time);

	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &AnimationNodeStateMachineTransition::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &AnimationNodeStateMachineTransition::is_disabled);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &AnimationNodeStateMachineTransition::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &AnimationNodeStateMachineTransition::get_priority);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "switch_mode", PROPERTY_HINT_ENUM, "Immediate,Sync,AtEnd"), "set_switch_mode", "get_switch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_advance"), "set_auto_advance", "has_auto_advance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "xfade_time", PROPERTY_HINT_RANGE, "0,240,0.01"), "set_xfade_time", "get_xfade_time");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "0,32,1"), "set_priority", "get_priority");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");

	BIND_CONSTANT(SWITCH_MODE_IMMEDIATE);
	BIND_CONSTANT(SWITCH_MODE_SYNC);
	BIND_CONSTANT(SWITCH_MODE_AT_END);
}

AnimationNodeStateMachineTransition::AnimationNodeStateMachineTransition() {

	switch_mode = SWITCH_MODE_IMMEDIATE;
	auto_advance = false;
	xfade = 0;
	disabled = false;
	priority = 1;
}

///////////////////////////////////////////////////////
void AnimationNodeStateMachine::add_node(const StringName &p_name, Ref<AnimationNode> p_node) {

	ERR_FAIL_COND(states.has(p_name));
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_node->get_parent().is_valid());
	ERR_FAIL_COND(p_node->get_tree() != NULL);
	ERR_FAIL_COND(String(p_name).find("/") != -1);
	states[p_name] = p_node;

	p_node->set_parent(this);
	p_node->set_tree(get_tree());

	emit_changed();
}

Ref<AnimationNode> AnimationNodeStateMachine::get_node(const StringName &p_name) const {

	ERR_FAIL_COND_V(!states.has(p_name), Ref<AnimationNode>());

	return states[p_name];
}

StringName AnimationNodeStateMachine::get_node_name(const Ref<AnimationNode> &p_node) const {
	for (Map<StringName, Ref<AnimationRootNode> >::Element *E = states.front(); E; E = E->next()) {
		if (E->get() == p_node) {
			return E->key();
		}
	}

	ERR_FAIL_V(StringName());
}

bool AnimationNodeStateMachine::has_node(const StringName &p_name) const {
	return states.has(p_name);
}
void AnimationNodeStateMachine::remove_node(const StringName &p_name) {

	ERR_FAIL_COND(!states.has(p_name));

	{
		//erase node connections
		Ref<AnimationNode> node = states[p_name];
		for (int i = 0; i < node->get_input_count(); i++) {
			node->set_input_connection(i, StringName());
		}
		node->set_parent(NULL);
		node->set_tree(NULL);
	}

	states.erase(p_name);
	path.erase(p_name);

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == p_name || transitions[i].to == p_name) {
			transitions.remove(i);
			i--;
		}
	}

	if (start_node == p_name) {
		start_node = StringName();
	}

	if (end_node == p_name) {
		end_node = StringName();
	}

	if (playing && current == p_name) {
		stop();
	}
	emit_changed();
}

void AnimationNodeStateMachine::rename_node(const StringName &p_name, const StringName &p_new_name) {

	ERR_FAIL_COND(!states.has(p_name));
	ERR_FAIL_COND(states.has(p_new_name));

	states[p_new_name] = states[p_name];
	states.erase(p_name);

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == p_name) {
			transitions[i].from = p_new_name;
		}

		if (transitions[i].to == p_name) {
			transitions[i].to = p_new_name;
		}
	}

	if (start_node == p_name) {
		start_node = p_new_name;
	}

	if (end_node == p_name) {
		end_node = p_new_name;
	}

	if (playing && current == p_name) {
		current = p_new_name;
	}

	path.clear(); //clear path
}

void AnimationNodeStateMachine::get_node_list(List<StringName> *r_nodes) const {

	List<StringName> nodes;
	for (Map<StringName, Ref<AnimationRootNode> >::Element *E = states.front(); E; E = E->next()) {
		nodes.push_back(E->key());
	}
	nodes.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E = nodes.front(); E; E = E->next()) {
		r_nodes->push_back(E->get());
	}
}

bool AnimationNodeStateMachine::has_transition(const StringName &p_from, const StringName &p_to) const {

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == p_from && transitions[i].to == p_to)
			return true;
	}
	return false;
}

int AnimationNodeStateMachine::find_transition(const StringName &p_from, const StringName &p_to) const {

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == p_from && transitions[i].to == p_to)
			return i;
	}
	return -1;
}

void AnimationNodeStateMachine::add_transition(const StringName &p_from, const StringName &p_to, const Ref<AnimationNodeStateMachineTransition> &p_transition) {

	ERR_FAIL_COND(p_from == p_to);
	ERR_FAIL_COND(!states.has(p_from));
	ERR_FAIL_COND(!states.has(p_to));
	ERR_FAIL_COND(p_transition.is_null());

	for (int i = 0; i < transitions.size(); i++) {
		ERR_FAIL_COND(transitions[i].from == p_from && transitions[i].to == p_to);
	}

	Transition tr;
	tr.from = p_from;
	tr.to = p_to;
	tr.transition = p_transition;

	transitions.push_back(tr);
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

	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == p_from && transitions[i].to == p_to) {
			transitions.remove(i);
			return;
		}
	}

	if (playing) {
		path.clear();
	}
}

void AnimationNodeStateMachine::remove_transition_by_index(int p_transition) {

	transitions.remove(p_transition);
	if (playing) {
		path.clear();
	}
}

void AnimationNodeStateMachine::set_start_node(const StringName &p_node) {

	ERR_FAIL_COND(p_node != StringName() && !states.has(p_node));
	start_node = p_node;
}

String AnimationNodeStateMachine::get_start_node() const {

	return start_node;
}

void AnimationNodeStateMachine::set_end_node(const StringName &p_node) {

	ERR_FAIL_COND(p_node != StringName() && !states.has(p_node));
	end_node = p_node;
}

String AnimationNodeStateMachine::get_end_node() const {

	return end_node;
}

void AnimationNodeStateMachine::set_graph_offset(const Vector2 &p_offset) {
	graph_offset = p_offset;
}

Vector2 AnimationNodeStateMachine::get_graph_offset() const {
	return graph_offset;
}

float AnimationNodeStateMachine::process(float p_time, bool p_seek) {

	//if not playing and it can restart, then restart
	if (!playing) {
		if (start_node) {
			start(start_node);
		} else {
			return 0;
		}
	}

	bool do_start = (p_seek && p_time == 0) || play_start || current == StringName();

	if (do_start) {

		if (start_node != StringName() && p_seek && p_time == 0) {
			current = start_node;
		}

		len_current = blend_node(states[current], 0, true, 1.0, FILTER_IGNORE, false);
		pos_current = 0;
		loops_current = 0;
		play_start = false;
	}

	float fade_blend = 1.0;

	if (fading_from != StringName()) {

		if (!p_seek) {
			fading_pos += p_time;
		}
		fade_blend = MIN(1.0, fading_pos / fading_time);
		if (fade_blend >= 1.0) {
			fading_from = StringName();
		}
	}

	float rem = blend_node(states[current], p_time, p_seek, fade_blend, FILTER_IGNORE, false);

	if (fading_from != StringName()) {

		blend_node(states[fading_from], p_time, p_seek, 1.0 - fade_blend, FILTER_IGNORE, false);
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
	float next_xfade = 0;
	AnimationNodeStateMachineTransition::SwitchMode switch_mode = AnimationNodeStateMachineTransition::SWITCH_MODE_IMMEDIATE;

	if (path.size()) {

		for (int i = 0; i < transitions.size(); i++) {
			if (transitions[i].from == current && transitions[i].to == path[0]) {
				next_xfade = transitions[i].transition->get_xfade_time();
				switch_mode = transitions[i].transition->get_switch_mode();
				next = path[0];
			}
		}
	} else {
		float priority_best = 1e20;
		int auto_advance_to = -1;
		for (int i = 0; i < transitions.size(); i++) {
			if (transitions[i].from == current && transitions[i].transition->has_auto_advance()) {

				if (transitions[i].transition->get_priority() < priority_best) {
					auto_advance_to = i;
				}
			}
		}

		if (auto_advance_to != -1) {
			next = transitions[auto_advance_to].to;
			next_xfade = transitions[auto_advance_to].transition->get_xfade_time();
			switch_mode = transitions[auto_advance_to].transition->get_switch_mode();
		}
	}

	//if next, see when to transition
	if (next != StringName()) {

		bool goto_next = false;

		if (switch_mode == AnimationNodeStateMachineTransition::SWITCH_MODE_IMMEDIATE) {
			goto_next = fading_from == StringName();
		} else {
			goto_next = next_xfade >= (len_current - pos_current) || loops_current > 0;
			if (loops_current > 0) {
				next_xfade = 0;
			}
		}

		if (goto_next) { //loops should be used because fade time may be too small or zero and animation may have looped

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
				len_current = blend_node(states[current], 0, true, 0, FILTER_IGNORE, false);
				pos_current = MIN(pos_current, len_current);
				blend_node(states[current], pos_current, true, 0, FILTER_IGNORE, false);

			} else {
				len_current = blend_node(states[current], 0, true, 0, FILTER_IGNORE, false);
				pos_current = 0;
			}

			rem = len_current; //so it does not show 0 on transition
			loops_current = 0;
		}
	}

	//compute time left for transitions by using the end node

	if (end_node != StringName() && end_node != current) {

		rem = blend_node(states[end_node], 0, true, 0, FILTER_IGNORE, false);
	}

	return rem;
}

bool AnimationNodeStateMachine::travel(const StringName &p_state) {
	ERR_FAIL_COND_V(!playing, false);
	ERR_FAIL_COND_V(!states.has(p_state), false);
	ERR_FAIL_COND_V(!states.has(current), false);

	path.clear(); //a new one will be needed

	if (current == p_state)
		return true; //nothing to do

	loops_current = 0; // reset loops, so fade does not happen immediately

	Vector2 current_pos = states[current]->get_position();
	Vector2 target_pos = states[p_state]->get_position();

	Map<StringName, AStarCost> cost_map;

	List<int> open_list;

	//build open list
	for (int i = 0; i < transitions.size(); i++) {
		if (transitions[i].from == current) {
			open_list.push_back(i);
			float cost = states[transitions[i].to]->get_position().distance_to(current_pos);
			cost *= transitions[i].transition->get_priority();
			AStarCost ap;
			ap.prev = current;
			ap.distance = cost;
			cost_map[transitions[i].to] = ap;

			if (transitions[i].to == p_state) { //prematurely found it! :D
				path.push_back(p_state);
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
		List<int>::Element *least_cost_transition = NULL;
		float least_cost = 1e20;

		for (List<int>::Element *E = open_list.front(); E; E = E->next()) {

			float cost = cost_map[transitions[E->get()].to].distance;
			cost += states[transitions[E->get()].to]->get_position().distance_to(target_pos);

			if (cost < least_cost) {
				least_cost_transition = E;
			}
		}

		StringName transition_prev = transitions[least_cost_transition->get()].from;
		StringName transition = transitions[least_cost_transition->get()].to;

		for (int i = 0; i < transitions.size(); i++) {
			if (transitions[i].from != transition || transitions[i].to == transition_prev) {
				continue; //not interested on those
			}

			float distance = states[transitions[i].from]->get_position().distance_to(states[transitions[i].to]->get_position());
			distance *= transitions[i].transition->get_priority();
			distance += cost_map[transitions[i].from].distance;

			if (cost_map.has(transitions[i].to)) {
				//oh this was visited already, can we win the cost?
				if (distance < cost_map[transitions[i].to].distance) {
					cost_map[transitions[i].to].distance = distance;
					cost_map[transitions[i].to].prev = transitions[i].from;
				}
			} else {
				//add to open list
				AStarCost ac;
				ac.prev = transitions[i].from;
				ac.distance = distance;
				cost_map[transitions[i].to] = ac;

				open_list.push_back(i);

				if (transitions[i].to == p_state) {
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
	StringName at = p_state;
	while (at != current) {
		path.push_back(at);
		at = cost_map[at].prev;
	}

	path.invert();

	return true;
}

void AnimationNodeStateMachine::start(const StringName &p_state) {

	ERR_FAIL_COND(!states.has(p_state));
	path.clear();
	current = p_state;
	playing = true;
	play_start = true;
}
void AnimationNodeStateMachine::stop() {
	playing = false;
	play_start = false;
	current = StringName();
}
bool AnimationNodeStateMachine::is_playing() const {

	return playing;
}
StringName AnimationNodeStateMachine::get_current_node() const {
	if (!playing) {
		return StringName();
	}

	return current;
}

StringName AnimationNodeStateMachine::get_blend_from_node() const {
	if (!playing) {
		return StringName();
	}

	return fading_from;
}

float AnimationNodeStateMachine::get_current_play_pos() const {
	return pos_current;
}
float AnimationNodeStateMachine::get_current_length() const {
	return len_current;
}

Vector<StringName> AnimationNodeStateMachine::get_travel_path() const {
	return path;
}
String AnimationNodeStateMachine::get_caption() const {
	return "StateMachine";
}

void AnimationNodeStateMachine::_notification(int p_what) {
}

void AnimationNodeStateMachine::set_tree(AnimationTree *p_player) {

	AnimationNode::set_tree(p_player);

	for (Map<StringName, Ref<AnimationRootNode> >::Element *E = states.front(); E; E = E->next()) {
		Ref<AnimationRootNode> node = E->get();
		node->set_tree(p_player);
	}
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
				states[node_name]->set_position(p_value);
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
	} else if (name == "start_node") {
		set_start_node(p_value);
		return true;
	} else if (name == "end_node") {
		set_end_node(p_value);
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
			if (states.has(node_name)) {
				r_ret = states[node_name];
				return true;
			}
		}

		if (what == "position") {

			if (states.has(node_name)) {
				r_ret = states[node_name]->get_position();
				return true;
			}
		}
	} else if (name == "transitions") {
		Array trans;
		trans.resize(transitions.size() * 3);

		for (int i = 0; i < transitions.size(); i++) {
			trans[i * 3 + 0] = transitions[i].from;
			trans[i * 3 + 1] = transitions[i].to;
			trans[i * 3 + 2] = transitions[i].transition;
		}

		r_ret = trans;
		return true;
	} else if (name == "start_node") {
		r_ret = get_start_node();
		return true;
	} else if (name == "end_node") {
		r_ret = get_end_node();
		return true;
	} else if (name == "graph_offset") {
		r_ret = get_graph_offset();
		return true;
	}

	return false;
}
void AnimationNodeStateMachine::_get_property_list(List<PropertyInfo> *p_list) const {

	List<StringName> names;
	for (Map<StringName, Ref<AnimationRootNode> >::Element *E = states.front(); E; E = E->next()) {
		names.push_back(E->key());
	}
	names.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		String name = E->get();
		p_list->push_back(PropertyInfo(Variant::OBJECT, "states/" + name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "AnimationNode", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "states/" + name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "transitions", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::STRING, "start_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::STRING, "end_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	p_list->push_back(PropertyInfo(Variant::VECTOR2, "graph_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
}

void AnimationNodeStateMachine::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_node", "name", "node"), &AnimationNodeStateMachine::add_node);
	ClassDB::bind_method(D_METHOD("get_node", "name"), &AnimationNodeStateMachine::get_node);
	ClassDB::bind_method(D_METHOD("remove_node", "name"), &AnimationNodeStateMachine::remove_node);
	ClassDB::bind_method(D_METHOD("rename_node", "name", "new_name"), &AnimationNodeStateMachine::rename_node);
	ClassDB::bind_method(D_METHOD("has_node", "name"), &AnimationNodeStateMachine::has_node);
	ClassDB::bind_method(D_METHOD("get_node_name", "node"), &AnimationNodeStateMachine::get_node_name);

	ClassDB::bind_method(D_METHOD("has_transition", "from", "to"), &AnimationNodeStateMachine::add_transition);
	ClassDB::bind_method(D_METHOD("add_transition", "from", "to", "transition"), &AnimationNodeStateMachine::add_transition);
	ClassDB::bind_method(D_METHOD("get_transition", "idx"), &AnimationNodeStateMachine::get_transition);
	ClassDB::bind_method(D_METHOD("get_transition_from", "idx"), &AnimationNodeStateMachine::get_transition_from);
	ClassDB::bind_method(D_METHOD("get_transition_to", "idx"), &AnimationNodeStateMachine::get_transition_to);
	ClassDB::bind_method(D_METHOD("get_transition_count"), &AnimationNodeStateMachine::get_transition_count);
	ClassDB::bind_method(D_METHOD("remove_transition_by_index", "idx"), &AnimationNodeStateMachine::remove_transition_by_index);
	ClassDB::bind_method(D_METHOD("remove_transition", "from", "to"), &AnimationNodeStateMachine::remove_transition);

	ClassDB::bind_method(D_METHOD("set_start_node", "name"), &AnimationNodeStateMachine::set_start_node);
	ClassDB::bind_method(D_METHOD("get_start_node"), &AnimationNodeStateMachine::get_start_node);

	ClassDB::bind_method(D_METHOD("set_end_node", "name"), &AnimationNodeStateMachine::set_end_node);
	ClassDB::bind_method(D_METHOD("get_end_node"), &AnimationNodeStateMachine::get_end_node);

	ClassDB::bind_method(D_METHOD("set_graph_offset", "name"), &AnimationNodeStateMachine::set_graph_offset);
	ClassDB::bind_method(D_METHOD("get_graph_offset"), &AnimationNodeStateMachine::get_graph_offset);

	ClassDB::bind_method(D_METHOD("travel", "to_node"), &AnimationNodeStateMachine::travel);
	ClassDB::bind_method(D_METHOD("start", "node"), &AnimationNodeStateMachine::start);
	ClassDB::bind_method(D_METHOD("stop"), &AnimationNodeStateMachine::stop);
	ClassDB::bind_method(D_METHOD("is_playing"), &AnimationNodeStateMachine::is_playing);
	ClassDB::bind_method(D_METHOD("get_current_node"), &AnimationNodeStateMachine::get_current_node);
	ClassDB::bind_method(D_METHOD("get_travel_path"), &AnimationNodeStateMachine::get_travel_path);
}

AnimationNodeStateMachine::AnimationNodeStateMachine() {

	play_start = false;

	playing = false;
	len_current = 0;

	fading_time = 0;
}
