/*************************************************************************/
/*  character_net_controller.cpp                                         */
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

#include "scene_rewinder.h"

void SceneRewinder::_bind_methods() {

	ClassDB::bind_method(D_METHOD("register_variable", "node", "variable", "on_change_notify"), &SceneRewinder::register_variable, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("unregister_variable", "node", "variable"), &SceneRewinder::unregister_variable);

	ClassDB::bind_method(D_METHOD("get_changed_event_name", "variable"), &SceneRewinder::get_changed_event_name);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "method"), &SceneRewinder::track_variable_changes);
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "method"), &SceneRewinder::untrack_variable_changes);

	ADD_SIGNAL(MethodInfo("sync_process", PropertyInfo(Variant::FLOAT, "delta")));
}

void SceneRewinder::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS:
			ERR_FAIL_COND_MSG(get_process_priority() != INT32_MAX, "The process priority MUST not be changed, is likely there is a better way of doing what you are trying to do, if you really need it please open an issue.");

			process();
			break;
	}
}

SceneRewinder::SceneRewinder() {
	// Always runs this as last.
	const int lowest_priority_number = INT32_MAX;
	set_process_priority(lowest_priority_number);

	set_physics_process_internal(true);
}

void SceneRewinder::register_variable(Node *p_object, StringName p_variable, StringName p_on_change_notify) {
	Vector<VarData> *node_vars = variables.getptr(p_object->get_instance_id());
	if (node_vars == nullptr) {
		variables.set(p_object->get_instance_id(), Vector<VarData>());
		node_vars = variables.getptr(p_object->get_instance_id());
	}

	// Unreachable
	CRASH_COND(node_vars == nullptr);

	if (node_vars->find(p_variable) == -1) {
		const Variant old_val = p_object->get(p_variable);
		node_vars->push_back(VarData(p_variable, old_val));

		if (p_object->has_signal(get_changed_event_name(p_variable)) == false) {

			p_object->add_user_signal(MethodInfo(
					get_changed_event_name(p_variable),
					PropertyInfo(Variant::NIL, "old_value")));
		}
	}

	track_variable_changes(p_object, p_variable, p_on_change_notify);
}

void SceneRewinder::unregister_variable(Node *p_object, StringName p_variable) {
	if (variables.has(p_object->get_instance_id()) == false) return;
	if (variables[p_object->get_instance_id()].find(p_variable) == -1) return;

	// Disconnects the eventual connected methods
	List<Connection> connections;
	p_object->get_signal_connection_list(get_changed_event_name(p_variable), &connections);

	for (List<Connection>::Element *e = connections.front(); e != nullptr; e = e->next()) {
		p_object->disconnect(get_changed_event_name(p_variable), e->get().callable);
	}

	// Remove the user signal
	// TODO there is no way to remove signal yet.

	// Remove the variable from the list
	variables[p_object->get_instance_id()].erase(p_variable);

	// Removes the node if no variables to track.
	if (variables[p_object->get_instance_id()].size() == 0) {
		variables.erase(p_object->get_instance_id());
	}
}

String SceneRewinder::get_changed_event_name(StringName p_variable) {
	return "variable_" + p_variable + "_changed";
}

void SceneRewinder::track_variable_changes(Node *p_object, StringName p_variable, StringName p_method) {
	ERR_FAIL_COND_MSG(variables.has(p_object->get_instance_id()) == false, "You need to register the variable to track its changes.");
	ERR_FAIL_COND_MSG(variables[p_object->get_instance_id()].find(p_variable) == -1, "You need to register the variable to track its changes.");

	if (p_object->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_object, p_method)) == false) {

		p_object->connect(
				get_changed_event_name(p_variable),
				Callable(p_object, p_method));
	}
}

void SceneRewinder::untrack_variable_changes(Node *p_object, StringName p_variable, StringName p_method) {
	if (variables.has(p_object->get_instance_id()) == false) return;
	if (variables[p_object->get_instance_id()].find(p_variable) == -1) return;

	if (p_object->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_object, p_method))) {

		p_object->disconnect(
				get_changed_event_name(p_variable),
				Callable(p_object, p_method));
	}
}

void SceneRewinder::process() {

	// Emit process
	const real_t delta = get_physics_process_delta_time();
	emit_signal("sync_process", delta);

	// Detect changed variables
	Vector<ObjectID> null_objects;

	for (const ObjectID *id = variables.next(nullptr); id != nullptr; id = variables.next(id)) {
		Object *object = ObjectDB::get_instance(*id);

		if (object == nullptr) {
			null_objects.push_back(*id);
			continue;
		}

		VarData *object_vars = variables.get(*id).ptrw();
		for (int i = 0; i < variables.get(*id).size(); i += 1) {
			const Variant old_val = object_vars[i].old_val;
			const Variant new_val = object->get(object_vars[i].name);
			object_vars[i].old_val = new_val;

			if (old_val != new_val) {
				object->emit_signal(get_changed_event_name(object_vars[i].name), old_val);
			}

			// TODO create the snapshot here?
		}
	}

	// Removes the null objects.
	for (int i = 0; i < null_objects.size(); i += 1) {
		variables.erase(null_objects[i]);
	}
}

VarData::VarData() {}

VarData::VarData(StringName p_name) :
		name(p_name) {
}

VarData::VarData(StringName p_name, Variant p_val) :
		name(p_name),
		old_val(p_val) {
}

bool VarData::operator==(const VarData &p_other) const {
	return name == p_other.name;
}
