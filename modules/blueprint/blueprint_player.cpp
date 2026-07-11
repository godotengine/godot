/**************************************************************************/
/*  blueprint_player.cpp                                                  */
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

#include "blueprint_player.h"

#include "core/config/engine.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/object/class_db.h"
#include "core/string/print_string.h"
#include "core/variant/variant_utility.h"

// Safety limits: a blueprint graph is user data, cycles must not hang the game.
static const int MAX_EXEC_STEPS = 10000;
static const int MAX_DATA_DEPTH = 64;

Variant BlueprintPlayer::_parse_param(const Variant &p_value) {
	if (p_value.get_type() != Variant::STRING) {
		return p_value;
	}
	const String s = p_value;
	const Variant parsed = VariantUtilityFunctions::str_to_var(s);
	if (parsed.get_type() == Variant::NIL && s.strip_edges() != "null") {
		return s; // Not valid Variant syntax: treat as a plain string.
	}
	return parsed;
}

void BlueprintPlayer::_build_node_map() {
	node_map.clear();
	if (blueprint.is_null()) {
		return;
	}
	const Array nodes = blueprint->get_nodes();
	for (int i = 0; i < nodes.size(); i++) {
		const Dictionary node = nodes[i];
		node_map[int(node["id"])] = node;
	}
}

Node *BlueprintPlayer::_resolve_target(const Dictionary &p_node) const {
	const Dictionary params = p_node["params"];
	const String path = params.get("node", "..");
	Node *target = get_node_or_null(NodePath(path.is_empty() ? ".." : path));
	if (!target) {
		ERR_PRINT(vformat("Blueprint: node not found at path \"%s\" (relative to %s).", path, get_path()));
	}
	return target;
}

bool BlueprintPlayer::_has_input_value(const Dictionary &p_node, int p_input_port) const {
	const int node_id = p_node["id"];
	const Array conns = blueprint->get_connections();
	for (int i = 0; i < conns.size(); i++) {
		const Dictionary conn = conns[i];
		if (int(conn["to_node"]) == node_id && int(conn["to_port"]) == p_input_port) {
			return true;
		}
	}
	const BlueprintNodeDef *def = blueprint_get_node_def(String(p_node["type"]));
	if (def && p_input_port < def->inputs.size()) {
		const Dictionary params = p_node["params"];
		const String key = def->inputs[p_input_port].name;
		return params.has(key) && !String(params[key]).is_empty();
	}
	return false;
}

int BlueprintPlayer::_find_exec_target(int p_from_node, int p_from_port) const {
	const Array conns = blueprint->get_connections();
	for (int i = 0; i < conns.size(); i++) {
		const Dictionary conn = conns[i];
		if (int(conn["from_node"]) == p_from_node && int(conn["from_port"]) == p_from_port) {
			const int to_node = conn["to_node"];
			if (!node_map.has(to_node)) {
				continue;
			}
			const Dictionary to_dict = node_map[to_node];
			const BlueprintNodeDef *def = blueprint_get_node_def(String(to_dict["type"]));
			const int to_port = conn["to_port"];
			if (def && to_port < def->inputs.size() && def->inputs[to_port].exec) {
				return to_node;
			}
		}
	}
	return -1;
}

Variant BlueprintPlayer::_eval_input(const Dictionary &p_node, int p_input_port, int p_depth) {
	if (p_depth > MAX_DATA_DEPTH) {
		ERR_PRINT("Blueprint: data connection depth limit reached (cycle?).");
		return Variant();
	}

	const int node_id = p_node["id"];
	const Array conns = blueprint->get_connections();
	for (int i = 0; i < conns.size(); i++) {
		const Dictionary conn = conns[i];
		if (int(conn["to_node"]) == node_id && int(conn["to_port"]) == p_input_port) {
			const int from_id = conn["from_node"];
			if (node_map.has(from_id)) {
				return _eval_output(node_map[from_id], conn["from_port"], p_depth + 1);
			}
		}
	}

	// No connection: fall back to the default stored in the node's params.
	const BlueprintNodeDef *def = blueprint_get_node_def(String(p_node["type"]));
	if (def && p_input_port < def->inputs.size()) {
		const Dictionary params = p_node["params"];
		const String key = def->inputs[p_input_port].name;
		if (params.has(key)) {
			return _parse_param(params[key]);
		}
	}
	return Variant();
}

Variant BlueprintPlayer::_eval_output(const Dictionary &p_node, int p_output_port, int p_depth) {
	const String type = p_node["type"];

	if (type == "constant") {
		const Dictionary params = p_node["params"];
		return _parse_param(params.get("value", Variant()));
	}
	if (type == "add") {
		const Variant a = _eval_input(p_node, 0, p_depth);
		const Variant b = _eval_input(p_node, 1, p_depth);
		Variant result;
		bool valid = false;
		Variant::evaluate(Variant::OP_ADD, a, b, result, valid);
		if (!valid) {
			ERR_PRINT(vformat("Blueprint: cannot add %s and %s.", a.stringify(), b.stringify()));
			return Variant();
		}
		return result;
	}
	if (type == "event_process" && p_output_port == 1) {
		return current_delta;
	}
	if (type == "get_property") {
		Node *target = _resolve_target(p_node);
		if (!target) {
			return Variant();
		}
		const Dictionary params = p_node["params"];
		return target->get(StringName(String(params.get("property", ""))));
	}
	if (type == "call_method" && p_output_port == 1) {
		const int id = p_node["id"];
		return exec_results.has(id) ? exec_results[id] : Variant();
	}
	return Variant();
}

void BlueprintPlayer::_run_chain(const Dictionary &p_event_node, int p_start_port) {
	int cur_id = p_event_node["id"];
	int out_port = p_start_port;

	for (int steps = 0; steps < MAX_EXEC_STEPS; steps++) {
		const int next_id = _find_exec_target(cur_id, out_port);
		if (next_id == -1 || !node_map.has(next_id)) {
			return;
		}
		const Dictionary node = node_map[next_id];
		const String type = node["type"];

		if (type == "print") {
			const Variant value = _eval_input(node, 1, 0);
			const String text = value.get_type() == Variant::STRING ? String(value) : value.stringify();
			print_line(text);
			emit_signal(SNAME("blueprint_print"), text);
			out_port = 0;
		} else if (type == "branch") {
			out_port = _eval_input(node, 1, 0).booleanize() ? 0 : 1;
		} else if (type == "call_method") {
			Node *target = _resolve_target(node);
			out_port = 0;
			if (target) {
				const Dictionary params = node["params"];
				const StringName method = String(params.get("method", ""));
				if (!target->has_method(method)) {
					ERR_PRINT(vformat("Blueprint: node \"%s\" has no method \"%s\".", target->get_name(), method));
				} else {
					Array args;
					// arg2 without arg1 makes no sense: only trailing empty args are dropped.
					if (_has_input_value(node, 1)) {
						args.push_back(_eval_input(node, 1, 0));
						if (_has_input_value(node, 2)) {
							args.push_back(_eval_input(node, 2, 0));
						}
					}
					exec_results[int(node["id"])] = target->callv(method, args);
				}
			}
		} else if (type == "set_property") {
			Node *target = _resolve_target(node);
			out_port = 0;
			if (target) {
				const Dictionary params = node["params"];
				target->set(StringName(String(params.get("property", ""))), _eval_input(node, 1, 0));
			}
		} else {
			return; // Not an executable node: chain ends.
		}
		cur_id = next_id;
	}
	ERR_PRINT("Blueprint: execution step limit reached (cycle?).");
}

void BlueprintPlayer::run_event(const String &p_event_type) {
	if (blueprint.is_null()) {
		return;
	}
	_build_node_map();
	const Array nodes = blueprint->get_nodes();
	for (int i = 0; i < nodes.size(); i++) {
		const Dictionary node = nodes[i];
		if (String(node["type"]) == p_event_type) {
			_run_chain(node, 0);
		}
	}
}

void BlueprintPlayer::_poll_input_events() {
	Input *input = Input::get_singleton();
	const Array nodes = blueprint->get_nodes();
	for (int i = 0; i < nodes.size(); i++) {
		const Dictionary node = nodes[i];
		if (String(node["type"]) != "event_input") {
			continue;
		}
		const Dictionary params = node["params"];
		const String action = params.get("action", "");
		if (!InputMap::get_singleton()->has_action(action)) {
			if (!warned_actions.has(action)) {
				warned_actions.insert(action);
				WARN_PRINT(vformat("Blueprint: input action \"%s\" does not exist in the Input Map (Project Settings).", action));
			}
			continue;
		}
		if (input->is_action_just_pressed(action)) {
			_build_node_map();
			_run_chain(node, 0);
		}
		if (input->is_action_just_released(action)) {
			_build_node_map();
			_run_chain(node, 1);
		}
	}
}

void BlueprintPlayer::_connect_signal_events() {
	const Array nodes = blueprint->get_nodes();
	for (int i = 0; i < nodes.size(); i++) {
		const Dictionary node = nodes[i];
		if (String(node["type"]) != "event_signal") {
			continue;
		}
		Node *target = _resolve_target(node);
		if (!target) {
			continue;
		}
		const Dictionary params = node["params"];
		const StringName signal = String(params.get("signal", ""));
		if (!target->has_signal(signal)) {
			ERR_PRINT(vformat("Blueprint: node \"%s\" has no signal \"%s\".", target->get_name(), signal));
			continue;
		}
		SignalHookup hookup;
		hookup.object_id = target->get_instance_id();
		hookup.signal = signal;
		hookup.callable = Callable(this, "_bp_signal_fired").bind(int(node["id"]));
		target->connect(signal, hookup.callable);
		signal_hookups.push_back(hookup);
	}
}

void BlueprintPlayer::_disconnect_signal_events() {
	for (const SignalHookup &hookup : signal_hookups) {
		Object *obj = ObjectDB::get_instance(hookup.object_id);
		if (obj && obj->is_connected(hookup.signal, hookup.callable)) {
			obj->disconnect(hookup.signal, hookup.callable);
		}
	}
	signal_hookups.clear();
}

Variant BlueprintPlayer::_signal_fired(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	if (p_argcount < 1 || blueprint.is_null()) {
		return Variant();
	}
	// The node id is bound last; signal arguments (if any) come before it.
	const int node_id = *p_args[p_argcount - 1];
	_build_node_map();
	if (node_map.has(node_id)) {
		_run_chain(node_map[node_id], 0);
	}
	return Variant();
}

void BlueprintPlayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			set_process(true);
			if (active && !Engine::get_singleton()->is_editor_hint() && blueprint.is_valid()) {
				_connect_signal_events();
				run_event("event_ready");
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_disconnect_signal_events();
		} break;
		case NOTIFICATION_PROCESS: {
			if (active && !Engine::get_singleton()->is_editor_hint() && blueprint.is_valid()) {
				current_delta = get_process_delta_time();
				run_event("event_process");
				_poll_input_events();
			}
		} break;
	}
}

void BlueprintPlayer::set_blueprint(const Ref<Blueprint> &p_blueprint) {
	blueprint = p_blueprint;
}

Ref<Blueprint> BlueprintPlayer::get_blueprint() const {
	return blueprint;
}

void BlueprintPlayer::set_active(bool p_active) {
	active = p_active;
}

bool BlueprintPlayer::is_active() const {
	return active;
}

void BlueprintPlayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_blueprint", "blueprint"), &BlueprintPlayer::set_blueprint);
	ClassDB::bind_method(D_METHOD("get_blueprint"), &BlueprintPlayer::get_blueprint);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &BlueprintPlayer::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &BlueprintPlayer::is_active);
	ClassDB::bind_method(D_METHOD("run_event", "event_type"), &BlueprintPlayer::run_event);

	{
		MethodInfo mi;
		mi.name = "_bp_signal_fired";
		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_bp_signal_fired", &BlueprintPlayer::_signal_fired, mi);
	}

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blueprint", PROPERTY_HINT_RESOURCE_TYPE, "Blueprint"), "set_blueprint", "get_blueprint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");

	ADD_SIGNAL(MethodInfo("blueprint_print", PropertyInfo(Variant::STRING, "text")));
}
