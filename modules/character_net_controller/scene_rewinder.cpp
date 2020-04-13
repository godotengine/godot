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
#include "scene/main/window.h"

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

	ClassDB::bind_method(D_METHOD("reset"), &SceneRewinder::reset);
	ClassDB::bind_method(D_METHOD("clear"), &SceneRewinder::clear);

	ClassDB::bind_method(D_METHOD("set_server_notify_state_interval", "interval"), &SceneRewinder::set_server_notify_state_interval);
	ClassDB::bind_method(D_METHOD("get_server_notify_state_interval"), &SceneRewinder::get_server_notify_state_interval);

	ClassDB::bind_method(D_METHOD("register_variable", "node", "variable", "on_change_notify"), &SceneRewinder::register_variable, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("unregister_variable", "node", "variable"), &SceneRewinder::unregister_variable);

	ClassDB::bind_method(D_METHOD("get_changed_event_name", "variable"), &SceneRewinder::get_changed_event_name);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "method"), &SceneRewinder::track_variable_changes);
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "method"), &SceneRewinder::untrack_variable_changes);

	ClassDB::bind_method(D_METHOD("_rpc_send_state"), &SceneRewinder::_rpc_send_state);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "server_notify_state_interval", PROPERTY_HINT_RANGE, "0.001,10.0,0.0001"), "set_server_notify_state_interval", "get_server_notify_state_interval");

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

			clear();
			reset();

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			clear();

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
		server_notify_state_interval(1.0),
		rewinder(nullptr),
		node_counter(1),
		generate_id(false) {

	rpc_config("__reset", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("__clear", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_send_state", MultiplayerAPI::RPC_MODE_REMOTE);
}

SceneRewinder::~SceneRewinder() {
	if (rewinder) {
		memdelete(rewinder);
		rewinder = nullptr;
	}
}

void SceneRewinder::set_server_notify_state_interval(real_t p_interval) {
	server_notify_state_interval = p_interval;
}

real_t SceneRewinder::get_server_notify_state_interval() const {
	return server_notify_state_interval;
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
		const uint32_t node_id(generate_id ? node_counter : 0);
		data.set(
				p_node->get_instance_id(),
				NodeData(node_id, p_node->get_instance_id()));
		node_counter += 1;
		node_data = data.getptr(p_node->get_instance_id());
	}

	// Unreachable
	CRASH_COND(node_data == nullptr);

	const int id = node_data->vars.find(p_variable);
	if (id == -1) {
		const Variant old_val = p_node->get(p_variable);
		const int var_id = generate_id ? node_data->vars.size() + 1 : 0;
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
	if (get_tree() == nullptr || !get_tree()->has_network_peer()) {

		__reset();
	} else {

		ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "The reset function must be called on server");
		__reset();
		rpc("__reset");
	}
}

void SceneRewinder::__reset() {

	generate_id = false;

	if (get_tree()) {
		if (get_multiplayer()->is_connected("network_peer_connected", callable_mp(this, &SceneRewinder::on_peer_connected))) {

			get_multiplayer()->disconnect("network_peer_connected", callable_mp(this, &SceneRewinder::on_peer_connected));
			get_multiplayer()->disconnect("network_peer_disconnected", callable_mp(this, &SceneRewinder::on_peer_disconnected));
		}
	}

	if (rewinder) {
		memdelete(rewinder);
		rewinder = nullptr;
	}

	if (get_tree()) {

		if (get_tree()->get_network_peer().is_null()) {
			rewinder = memnew(NoNetRewinder(this));
			generate_id = true;

		} else if (get_tree()->is_network_server()) {
			rewinder = memnew(ServerRewinder(this));
			generate_id = true;

			get_multiplayer()->connect("network_peer_connected", callable_mp(this, &SceneRewinder::on_peer_connected));
			get_multiplayer()->connect("network_peer_disconnected", callable_mp(this, &SceneRewinder::on_peer_disconnected));
		} else {
			rewinder = memnew(ClientRewinder(this));
		}

		// Always runs the SceneRewinder last.
		const int lowest_priority_number = INT32_MAX;
		set_process_priority(lowest_priority_number);
		set_physics_process_internal(true);
	}
}

void SceneRewinder::clear() {
	if (get_tree() == nullptr || !get_tree()->has_network_peer()) {

		__clear();
	} else {

		ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "The clear function must be called on server");
		__clear();
		rpc("__clear");
	}
}

void SceneRewinder::__clear() {
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
	node_counter = 1;

	if (rewinder) {
		rewinder->clear();
	}
}

void SceneRewinder::_rpc_send_state(Variant p_snapshot) {
	ERR_FAIL_COND(get_tree()->is_network_server() == true);

	rewinder->receive_snapshot(p_snapshot);
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

void NoNetRewinder::clear() {
}

void NoNetRewinder::process(real_t p_delta) {
	// Nothing to do?
}

void NoNetRewinder::receive_snapshot(Variant _p_snapshot) {
	// Unreachable
	CRASH_NOW();
}

ServerRewinder::ServerRewinder(SceneRewinder *p_node) :
		Rewinder(p_node),
		state_notifier_timer(0.0) {}

void ServerRewinder::clear() {
}

void ServerRewinder::on_peer_connected(int p_peer_id) {
	ERR_FAIL_COND_MSG(peers_data.find(p_peer_id) != -1, "This peer is already connected, is likely a bug.");
	peers_data.push_back(p_peer_id);
}

void ServerRewinder::on_peer_disconnected(int p_peer_id) {
	ERR_FAIL_COND_MSG(peers_data.find(p_peer_id) == -1, "This peer is already disconnected, is likely a bug.");
	peers_data.erase(p_peer_id);
}

Variant ServerRewinder::generate_snapshot() {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
	//  [NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
	//  NODE, VARIABLE, Value, VARIABLE, Value, NIL]
	//
	// Each node ends with a NIL, and the NODE and the VARIABLE are special:
	// - NODE, can be an array of two variables [Node ID, NodePath] or directly
	//         a Node ID. Obviously the array is sent only the first time.
	// - VARIABLE, can be an array with the ID and the variable name, or just
	//              the ID; similarly as is for the NODE the array is send only
	//              the first time.

	// TODO: in this moment the snapshot is the same for anyone. Optimize.

	Vector<Variant> snapshot_data;

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

	state_notifier_timer += p_delta;
	if (state_notifier_timer >= scene_rewinder->get_server_notify_state_interval()) {
		state_notifier_timer = 0.0;

		Variant snapshot = generate_snapshot();
		scene_rewinder->rpc("_rpc_send_state", snapshot);
	}
}

void ServerRewinder::receive_snapshot(Variant _p_snapshot) {
	// Unreachable
	CRASH_NOW();
}

ClientRewinder::ClientRewinder(SceneRewinder *p_node) :
		Rewinder(p_node) {}

void ClientRewinder::clear() {
	node_id_map.clear();
}

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

void ClientRewinder::receive_snapshot(Variant p_snapshot) {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
	//  [NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
	//  NODE, VARIABLE, Value, VARIABLE, Value, NIL]
	//
	// Each node ends with a NIL, and the NODE and the VARIABLE are special:
	// - NODE, can be an array of two variables [Node ID, NodePath] or directly
	//         a Node ID. Obviously the array is sent only the first time.
	// - VARIABLE, can be an array with the ID and the variable name, or just
	//              the ID; similarly as is for the NODE the array is send only
	//              the first time.

	ERR_FAIL_COND(!p_snapshot.is_array());

	const Vector<Variant> snapshot = p_snapshot;

	Node *node = nullptr;
	StringName variable_name;

	for (int i = 0; i < snapshot.size(); i += 1) {
		Variant v = snapshot[i];
		if (node == nullptr) {

			uint32_t node_id(0);

			if (v.is_array()) {
				const Vector<Variant> node_data = v;
				ERR_FAIL_COND(node_data.size() != 2);
				ERR_FAIL_COND(node_data[0].get_type() != Variant::INT);
				ERR_FAIL_COND(node_data[1].get_type() != Variant::NODE_PATH);

				node_id = node_data[0];
				const NodePath node_path = node_data[1];

				ERR_FAIL_COND(node_id_map.has(node_id) == true);
				ERR_FAIL_COND(node_paths.has(node_id) == true);
				node = scene_rewinder->get_tree()->get_root()->get_node(node_path);

				node_paths.set(node_id, node_path);

				if (node == nullptr) {
					// This node does't exist yet, so skip it entirely.
					for (i += 1; i < snapshot.size(); i += 1) {
						if (snapshot[i].get_type() == Variant::NIL) {
							break;
						}
					}
					continue;
				}

				node_id_map.set(node_id, node->get_instance_id());

			} else {
				ERR_FAIL_COND(v.get_type() != Variant::INT);

				node_id = v;
				if (node_id_map.has(node_id)) {
					Object *obj = ObjectDB::get_instance(node_id_map[node_id]);
					ERR_FAIL_COND(obj == nullptr);
					node = Object::cast_to<Node>(obj);
				} else {
					// The node instance for this node ID was not found, try
					// to find it now.
					ERR_FAIL_COND(node_paths.has(node_id) == false);

					const NodePath node_path = node_paths[node_id];
					node = scene_rewinder->get_tree()->get_root()->get_node(node_path);

					if (node == nullptr) {
						// This node does't exist yet, so skip it entirely, again.
						for (i += 1; i < snapshot.size(); i += 1) {
							if (snapshot[i].get_type() == Variant::NIL) {
								break;
							}
						}
						continue;
					}

					node_id_map.set(node_id, node->get_instance_id());
				}
			}

			ERR_FAIL_COND(node == nullptr);

			if (!scene_rewinder->data.has(node->get_instance_id())) {
				// The node is not yet tracked on client, so just add it.
				scene_rewinder->data.set(
						node->get_instance_id(),
						NodeData(node_id, node->get_instance_id()));
			} else {
				if (scene_rewinder->data[node->get_instance_id()].id == 0) {
					scene_rewinder->data[node->get_instance_id()].id = node_id;
				} else {
					ERR_FAIL_COND(scene_rewinder->data[node->get_instance_id()].id != node_id);
				}
			}

		} else if (variable_name == StringName()) {
			// Take the variable name or check if NIL

			if (v.is_array()) {
				const Vector<Variant> var_data = v;
				ERR_FAIL_COND(var_data.size() != 2);
				ERR_FAIL_COND(var_data[0].get_type() != Variant::INT);
				ERR_FAIL_COND(var_data[1].get_type() != Variant::STRING_NAME);

				//const uint32_t var_id = var_data[0];
				//const NodePath var_name = var_data[1];

			} else if (v.get_type() == Variant::INT) {

			} else if (v.get_type() == Variant::NIL) {
				// NIL found, so this variable is done.
				node = nullptr;
				variable_name = StringName();
			}
		} else {
			// Take the variable value

			// TODO do something here?

			// Just reset the variable name so we can continue iterate.
			variable_name = StringName();
		}
	}
}
