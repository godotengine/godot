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

#include "character_net_controller.h"

VarData::VarData() :
		id(0),
		enabled(false) {}

VarData::VarData(StringName p_name) :
		id(0),
		name(p_name),
		enabled(false) {
}

VarData::VarData(uint32_t p_id, StringName p_name, Variant p_val, bool p_enabled) :
		id(p_id),
		name(p_name),
		old_val(p_val),
		enabled(p_enabled) {
}

bool VarData::operator==(const VarData &p_other) const {
	return name == p_other.name;
}

NodeData::NodeData() :
		id(0) {
}

NodeData::NodeData(uint32_t p_id, ObjectID p_instance_id) :
		id(p_id),
		instance_id(p_instance_id) {
}

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
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			const int lowest_priority_number = INT32_MAX;
			ERR_FAIL_COND_MSG(get_process_priority() != lowest_priority_number, "The process priority MUST not be changed, is likely there is a better way of doing what you are trying to do, if you really need it please open an issue.");

			process();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			// Unreachable.
			CRASH_COND(get_tree() == NULL);

			reset();

			if (get_tree()->get_network_peer().is_null()) {
				//rewinder = memnew(NoNetRewinder(this));
				//rewinder = memnew(ServerRewinder(this));
				rewinder = memnew(ClientRewinder(this));

			} else if (get_tree()->is_network_server()) {
				rewinder = memnew(ServerRewinder(this));

				get_multiplayer()->connect("network_peer_connected", callable_mp(this, &SceneRewinder::on_peer_connected));
				get_multiplayer()->connect("network_peer_disconnected", callable_mp(this, &SceneRewinder::on_peer_disconnected));
			} else {
				rewinder = memnew(ClientRewinder(this));
			}

			// Always runs this as last.
			const int lowest_priority_number = INT32_MAX;
			set_process_priority(lowest_priority_number);
			set_physics_process_internal(true);

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			reset();

			if (get_tree()->is_network_server()) {
				get_multiplayer()->disconnect("network_peer_connected", callable_mp(this, &SceneRewinder::on_peer_connected));
				get_multiplayer()->disconnect("network_peer_disconnected", callable_mp(this, &SceneRewinder::on_peer_disconnected));
			}

			memdelete(rewinder);
			rewinder = nullptr;

			set_physics_process_internal(false);
		}
	}
}

SceneRewinder::SceneRewinder() :
		rewinder(nullptr),
		node_id(1) {
}

SceneRewinder::~SceneRewinder() {
	if (rewinder) {
		memdelete(rewinder);
		rewinder = nullptr;
	}
}

void SceneRewinder::register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify) {

	ERR_FAIL_COND(p_node == nullptr);

	{
		CharacterNetController *controller = Object::cast_to<CharacterNetController>(p_node);
		if (controller) {
			if (controller->has_scene_rewinder()) {
				ERR_FAIL_COND_MSG(controller->get_scene_rewinder() != this, "This controller is associated with a different scene rewinder.");
			} else {
				// Unreachable.
				CRASH_COND(controllers.find(controller) != -1);
				controller->set_scene_rewinder(this);
				controllers.push_back(controller);
			}
		}
	}

	NodeData *node_data = data.getptr(p_node->get_instance_id());
	if (node_data == nullptr) {
		data.set(p_node->get_instance_id(), NodeData(node_id, p_node->get_instance_id()));
		node_id += 1;
		node_data = data.getptr(p_node->get_instance_id());
	}

	// Unreachable
	CRASH_COND(node_data == nullptr);

	const int id = node_data->vars.find(p_variable);
	if (id == -1) {
		const Variant old_val = p_node->get(p_variable);
		const int var_id = node_data->vars.size() + 1;
		node_data->vars.push_back(VarData(
				var_id,
				p_variable,
				old_val,
				true));
	} else {
		node_data->vars.write[id].enabled = true;
	}

	if (p_node->has_signal(get_changed_event_name(p_variable)) == false) {
		p_node->add_user_signal(MethodInfo(
				get_changed_event_name(p_variable),
				PropertyInfo(Variant::NIL, "old_value")));
	}

	track_variable_changes(p_node, p_variable, p_on_change_notify);
}

void SceneRewinder::unregister_variable(Node *p_node, StringName p_variable) {
	if (data.has(p_node->get_instance_id()) == false) return;
	if (data[p_node->get_instance_id()].vars.find(p_variable) == -1) return;

	{
		CharacterNetController *controller = Object::cast_to<CharacterNetController>(p_node);
		if (controller) {
			ERR_FAIL_COND_MSG(controller->get_scene_rewinder() != this, "This controller is associated with this scene rewinder.");
			controller->set_scene_rewinder(nullptr);
			controllers.erase(controller);
		}
	}

	// Disconnects the eventual connected methods
	List<Connection> connections;
	p_node->get_signal_connection_list(get_changed_event_name(p_variable), &connections);

	for (List<Connection>::Element *e = connections.front(); e != nullptr; e = e->next()) {
		p_node->disconnect(get_changed_event_name(p_variable), e->get().callable);
	}

	// Disable variable, don't remove it to preserve var node IDs.
	int id = data[p_node->get_instance_id()].vars.find(p_variable);
	CRASH_COND(id == -1); // Unreachable
	data[p_node->get_instance_id()].vars.write[id].enabled = false;
}

String SceneRewinder::get_changed_event_name(StringName p_variable) {
	return "variable_" + p_variable + "_changed";
}

void SceneRewinder::track_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
	ERR_FAIL_COND_MSG(data.has(p_node->get_instance_id()) == false, "You need to register the variable to track its changes.");
	ERR_FAIL_COND_MSG(data[p_node->get_instance_id()].vars.find(p_variable) == -1, "You need to register the variable to track its changes.");

	if (p_node->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method)) == false) {

		p_node->connect(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method));
	}
}

void SceneRewinder::untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
	if (data.has(p_node->get_instance_id()) == false) return;
	if (data[p_node->get_instance_id()].vars.find(p_variable) == -1) return;

	if (p_node->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method))) {

		p_node->disconnect(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method));
	}
}

void SceneRewinder::reset() {
	for (const ObjectID *id = data.next(nullptr); id != nullptr; id = data.next(id)) {

		const VarData *object_vars = data.get(*id).vars.ptr();
		for (int i = 0; i < data.get(*id).vars.size(); i += 1) {
			Node *node = static_cast<Node *>(ObjectDB::get_instance(*id));

			if (node != nullptr) {
				// Unregister the variable so the connected variables are
				// correctly removed
				unregister_variable(node, object_vars[i].name);
			}

			// TODO remove signal from the node when it's possible.
		}
	}

	data.clear();
	node_id = 1;
}

void SceneRewinder::process() {

	const real_t delta = get_physics_process_delta_time();
	emit_signal("sync_process", delta);

	// Detect changed variables
	Vector<ObjectID> null_objects;

	for (const ObjectID *key = data.next(nullptr); key != nullptr; key = data.next(key)) {
		Node *node = static_cast<Node *>(ObjectDB::get_instance(*key));

		if (node == nullptr) {
			null_objects.push_back(*key);
			continue;
		} else if (node->is_inside_tree() == false) {
			continue;
		}

		data[*key].cached_node = node;

		VarData *object_vars = data.get(*key).vars.ptrw();
		for (int i = 0; i < data.get(*key).vars.size(); i += 1) {
			if (object_vars->enabled == false) {
				continue;
			}

			const Variant old_val = object_vars[i].old_val;
			const Variant new_val = node->get(object_vars[i].name);
			object_vars[i].old_val = new_val;

			if (old_val != new_val) {
				node->emit_signal(get_changed_event_name(object_vars[i].name), old_val);
			}
		}
	}

	// Removes the null objects.
	for (int i = 0; i < null_objects.size(); i += 1) {
		data.erase(null_objects[i]);
	}

	rewinder->process(delta);
}

void SceneRewinder::on_peer_connected(int p_peer_id) {
	// No check of any kind!
	ServerRewinder *server_rewinder = static_cast<ServerRewinder *>(rewinder);
	server_rewinder->on_peer_connected(p_peer_id);
}

void SceneRewinder::on_peer_disconnected(int p_peer_id) {
	// No check of any kind!
	ServerRewinder *server_rewinder = static_cast<ServerRewinder *>(rewinder);
	server_rewinder->on_peer_disconnected(p_peer_id);
}

PeerData::PeerData() :
		peer(0) {
}

PeerData::PeerData(int p_peer) :
		peer(p_peer) {
}

bool PeerData::operator==(const PeerData &p_other) const {
	return peer == p_other.peer;
}

Rewinder::Rewinder(SceneRewinder *p_node) :
		scene_rewinder(p_node) {
}

NoNetRewinder::NoNetRewinder(SceneRewinder *p_node) :
		Rewinder(p_node) {}

void NoNetRewinder::process(real_t p_delta) {
}

ServerRewinder::ServerRewinder(SceneRewinder *p_node) :
		Rewinder(p_node) {}

void ServerRewinder::on_peer_connected(int p_peer_id) {
	ERR_FAIL_COND_MSG(peers_data.find(p_peer_id) != -1, "This peer is already connected, is likely a bug.");
	peers_data.push_back(p_peer_id);
}

void ServerRewinder::on_peer_disconnected(int p_peer_id) {
	ERR_FAIL_COND_MSG(peers_data.find(p_peer_id) == -1, "This peer is already disconnected, is likely a bug.");
	peers_data.erase(p_peer_id);
}

Variant ServerRewinder::generate_snapshot(int p_peer_index) {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
	// [Snapshot ID,
	//  NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
	//  NODE, VARIABLE, Value, VARIABLE, Value, NIL]
	//
	// Each node ends with a NIL, and the NODE and the VARIABLE are special:
	// - NODE, can be an array of two variables [Node ID, NodePath] or directly
	//         a Node ID. Obviously the array is sent only the first time.
	// - VARIABLE, can be an array with the ID and the variable name, or just
	//              the ID; similarly as is for the NODE the array is send only
	//              the first time.

	// TODO
	//PeerData *data = peers_data.ptrw();
	//PeerData *peer_data = data[p_peer_index];

	// NOTE: in this moment the snapshot is the same for anyone
	Vector<Variant> snapshot_data;

	const int snapshot_id(0);
	snapshot_data.push_back(snapshot_id);

	for (
			const ObjectID *key = scene_rewinder->data.next(nullptr);
			key != nullptr;
			key = scene_rewinder->data.next(key)) {

		if (scene_rewinder->data[*key].cached_node->is_inside_tree() == false) {
			continue;
		}

		// Insert NODE.
		Vector<Variant> node_data;
		node_data.resize(2);
		node_data.write[0] = scene_rewinder->data[*key].id;
		node_data.write[1] = scene_rewinder->data[*key].cached_node->get_path();
		snapshot_data.push_back(node_data);

		// Insert the node variables.
		const int size = scene_rewinder->data[*key].vars.size();
		const VarData *vars = scene_rewinder->data[*key].vars.ptr();
		for (int i = 0; i < size; i += 1) {
			Vector<Variant> var_info;
			var_info.resize(2);
			var_info.write[0] = vars[i].id;
			var_info.write[1] = vars[i].name;
			snapshot_data.push_back(var_info);
			snapshot_data.push_back(vars[i].old_val);
		}

		// Insert NIL.
		snapshot_data.push_back(Variant());
	}

	return snapshot_data;
}

void ServerRewinder::process(real_t p_delta) {
	Variant snapshot = generate_snapshot(0);

	print_line(snapshot);
}

ClientRewinder::ClientRewinder(SceneRewinder *p_node) :
		Rewinder(p_node) {}

void ClientRewinder::process(real_t p_delta) {

	// Store snapshots.
	Snapshot snapshot;
	snapshot.snapshot_id = 0;

	for (
			const ObjectID *key = scene_rewinder->data.next(nullptr);
			key != nullptr;
			key = scene_rewinder->data.next(key)) {

		snapshot.data.push_back(scene_rewinder->data[*key]);
	}

	snapshots.push_back(snapshot);
}
