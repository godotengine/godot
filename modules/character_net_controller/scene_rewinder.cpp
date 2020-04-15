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

VarData::VarData(uint32_t p_id) :
		id(p_id),
		enabled(false) {
}

VarData::VarData(StringName p_name) :
		id(0),
		name(p_name),
		enabled(false) {
}

VarData::VarData(uint32_t p_id, StringName p_name, Variant p_val, bool p_enabled) :
		id(p_id),
		name(p_name),
		value(p_val),
		enabled(p_enabled) {
}

bool VarData::operator==(const VarData &p_other) const {
	return name == p_other.name;
}

NodeData::NodeData() :
		id(0),
		is_controller(false) {
}

NodeData::NodeData(uint32_t p_id, ObjectID p_instance_id, bool is_controller) :
		id(p_id),
		instance_id(p_instance_id),
		is_controller(is_controller) {
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

			__clear();
			__reset();

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			__clear();

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
		generate_id(false),
		main_controller(nullptr) {

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

	bool is_controller = false;
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
				is_controller = true;

				if (controller->is_player_controller()) {
					if (main_controller == nullptr) {
						main_controller = controller;
					} else {
						WARN_PRINT("More local player net controllers are not fully tested. Please report any strange behaviour.");
					}
				}
			}
		}
	}

	NodeData *node_data = data.getptr(p_node->get_instance_id());
	if (node_data == nullptr) {
		const uint32_t node_id(generate_id ? node_counter : 0);
		data.set(
				p_node->get_instance_id(),
				NodeData(node_id, p_node->get_instance_id(), is_controller));
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

			if (main_controller == controller) {
				main_controller = nullptr;
			}
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
	if (dynamic_cast<NoNetRewinder *>(rewinder) != nullptr) {

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

	if (get_tree() == nullptr || get_tree()->get_network_peer().is_null()) {
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

void SceneRewinder::clear() {
	if (dynamic_cast<NoNetRewinder *>(rewinder) != nullptr) {

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

			const Variant old_val = object_vars[i].value;
			const Variant new_val = node->get(object_vars[i].name);
			object_vars[i].value = new_val;

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

void NoNetRewinder::process(real_t _p_delta) {
}

void NoNetRewinder::receive_snapshot(Variant _p_snapshot) {
}

ServerRewinder::ServerRewinder(SceneRewinder *p_node) :
		Rewinder(p_node),
		state_notifier_timer(0.0),
		snapshot_count(0) {}

void ServerRewinder::clear() {
	state_notifier_timer = 0.0;
	snapshot_count = 0;
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
	//  [SNAPSHOT ID,
	//  NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
	//  NODE, INPUT ID, VARIABLE, Value, VARIABLE, Value, NIL,
	//  NODE, VARIABLE, Value, VARIABLE, Value, NIL]
	//
	// Each node ends with a NIL, and the NODE and the VARIABLE are special:
	// - NODE, can be an array of two variables [Node ID, NodePath] or directly
	//         a Node ID. Obviously the array is sent only the first time.
	// - INPUT ID, this is optional and is used only when the node is a controller.
	// - VARIABLE, can be an array with the ID and the variable name, or just
	//              the ID; similarly as is for the NODE the array is send only
	//              the first time.

	// TODO: in this moment the snapshot is the same for anyone. Optimize.

	snapshot_count += 1;

	Vector<Variant> snapshot_data;

	snapshot_data.push_back(snapshot_count);

	for (
			const ObjectID *key = scene_rewinder->data.next(nullptr);
			key != nullptr;
			key = scene_rewinder->data.next(key)) {

		const NodeData &node_data = scene_rewinder->data.get(*key);
		if (node_data.cached_node->is_inside_tree() == false) {
			continue;
		}

		// Insert NODE.
		Vector<Variant> snap_node_data;
		snap_node_data.resize(2);
		snap_node_data.write[0] = node_data.id;
		snap_node_data.write[1] = node_data.cached_node->get_path();
		snapshot_data.push_back(snap_node_data);

		// Set the input ID if this is a controller.
		if (node_data.is_controller) {
			CharacterNetController *controller = Object::cast_to<CharacterNetController>(node_data.cached_node);
			CRASH_COND(controller == nullptr); // Unreachable

			if (unlikely(controller->get_current_input_id() == UINT64_MAX)) {
				// The first ID id is not yet arrived.
				snapshot_data.push_back(0);
			} else {
				snapshot_data.push_back(controller->get_current_input_id());
			}
		}

		// Insert the node variables.
		const int size = node_data.vars.size();
		const VarData *vars = node_data.vars.ptr();
		for (int i = 0; i < size; i += 1) {

			if (vars[i].enabled == false) {
				continue;
			}

			Vector<Variant> var_info;
			var_info.resize(2);
			var_info.write[0] = vars[i].id;
			var_info.write[1] = vars[i].name;
			snapshot_data.push_back(var_info);
			snapshot_data.push_back(vars[i].value);
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
		Rewinder(p_node),
		server_snapshot_id(0),
		recovered_snapshot_id(0) {
}

void ClientRewinder::clear() {
	node_id_map.clear();
	node_paths.clear();
	server_snapshot_id = 0;
	recovered_snapshot_id = 0;
	server_snapshot.controllers_input_id.clear();
	server_snapshot.data.clear();
	snapshots.clear();
}

void ClientRewinder::process(real_t p_delta) {

	ERR_FAIL_COND_MSG(scene_rewinder->main_controller == nullptr, "Snapshot creation fail, Make sure to track a NetController.");

	// Store snapshots.
	// TODO what happens during the client input saturated phases?
	Snapshot snapshot;

	// Store the player controller input ID
	snapshot.player_controller_input_id = scene_rewinder->main_controller->get_current_input_id();

	// Store the controllers input ID
	const CharacterNetController *const *controllers_ptr = scene_rewinder->controllers.ptr();
	for (int i = 0; i < scene_rewinder->controllers.size(); i += 1) {
		snapshot.controllers_input_id[controllers_ptr[i]->get_instance_id()] =
				controllers_ptr[i]->get_current_input_id();
	}

	snapshot.data = scene_rewinder->data;
	snapshots.push_back(snapshot);

	process_recovery();
}

void ClientRewinder::receive_snapshot(Variant p_snapshot) {
	parse_snapshot(p_snapshot);
}

void ClientRewinder::process_recovery() {
	if (server_snapshot_id <= recovered_snapshot_id) {
		// Nothing to recover.
		return;
	}

	// Compare the server snapshot with the client snapshot.
	// The scene state is relative to the NetController player input ID,
	// so the comparison is done between two snapshots that have the same player
	// NetController input ID.

	Snapshot s;
	s.player_controller_input_id = 0;

	while (snapshots.empty() == false && snapshots.front().player_controller_input_id <= server_snapshot.player_controller_input_id) {
		s = snapshots.front();
		snapshots.pop_front();
	}

	// Unreachable
	ERR_FAIL_COND(s.player_controller_input_id != server_snapshot.player_controller_input_id);

	// We found the client snapshot. Let's compare.

	recovered_snapshot_id = server_snapshot_id;
}

void ClientRewinder::parse_snapshot(Variant p_snapshot) {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
	//  [SNAPSHOT ID,
	//  NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
	//  NODE, INPUT ID, VARIABLE, Value, VARIABLE, Value, NIL,
	//  NODE, VARIABLE, Value, VARIABLE, Value, NIL]
	//
	// Each node ends with a NIL, and the NODE and the VARIABLE are special:
	// - NODE, can be an array of two variables [Node ID, NodePath] or directly
	//         a Node ID. Obviously the array is sent only the first time.
	// - INPUT ID, this is optional and is used only when the node is a controller.
	// - VARIABLE, can be an array with the ID and the variable name, or just
	//              the ID; similarly as is for the NODE the array is send only
	//              the first time.

	ERR_FAIL_COND_MSG(
			scene_rewinder->main_controller == nullptr,
			"Is not possible to receive server snapshots if you are not tracking any NetController.");
	ERR_FAIL_COND(!p_snapshot.is_array());

	const Vector<Variant> raw_snapshot = p_snapshot;
	const Variant *raw_snapshot_ptr = raw_snapshot.ptr();

	Node *node = nullptr;
	NodeData *rewinder_node_data;
	NodeData *server_snapshot_node_data;
	StringName variable_name;
	int server_snap_variable_index = -1;

	ERR_FAIL_COND(raw_snapshot.size() < 1);
	ERR_FAIL_COND(raw_snapshot_ptr[0].get_type() != Variant::INT);

	const uint64_t snapshot_id = raw_snapshot_ptr[0];

	ERR_FAIL_COND(snapshot_id <= server_snapshot_id);

	server_snapshot_id = snapshot_id;
	server_snapshot.player_controller_input_id = 0;

	for (int i = 1; i < raw_snapshot.size(); i += 1) {
		Variant v = raw_snapshot_ptr[i];
		if (node == nullptr) {
			// Take the node

			uint32_t node_id(0);

			if (v.is_array()) {
				const Vector<Variant> node_data = v;
				ERR_FAIL_COND(node_data.size() != 2);
				ERR_FAIL_COND(node_data[0].get_type() != Variant::INT);
				ERR_FAIL_COND(node_data[1].get_type() != Variant::NODE_PATH);

				node_id = node_data[0];
				const NodePath node_path = node_data[1];

				node = scene_rewinder->get_tree()->get_root()->get_node(node_path);

				node_paths.set(node_id, node_path);

				if (node == nullptr) {
					// This node does't exist yet, so skip it entirely.
					for (i += 1; i < raw_snapshot.size(); i += 1) {
						if (raw_snapshot_ptr[i].get_type() == Variant::NIL) {
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

					if (node_paths.has(node_id) == false) {
						WARN_PRINT("The node with ID `" + itos(node_id) + "` is not know by this peer.");
						// TODO notify the server so it sends a full snapshot, and so fix this issue.
						continue;
					}

					const NodePath node_path = node_paths[node_id];
					node = scene_rewinder->get_tree()->get_root()->get_node(node_path);

					if (node == nullptr) {
						// This node does't exist yet, so skip it entirely, again.
						for (i += 1; i < raw_snapshot.size(); i += 1) {
							if (raw_snapshot_ptr[i].get_type() == Variant::NIL) {
								break;
							}
						}
						continue;
					}

					node_id_map.set(node_id, node->get_instance_id());
				}
			}

			ERR_FAIL_COND(node == nullptr);

			const bool is_controller =
					Object::cast_to<CharacterNetController>(node) != nullptr;

			// Make sure this node is being tracked by the client.
			if (!scene_rewinder->data.has(node->get_instance_id())) {
				scene_rewinder->data.set(
						node->get_instance_id(),
						NodeData(node_id,
								node->get_instance_id(),
								is_controller));
			}
			rewinder_node_data = scene_rewinder->data.getptr(node->get_instance_id());
			rewinder_node_data->id = node_id;

			// Make sure this node is part of the server node.
			if (!server_snapshot.data.has(node->get_instance_id())) {
				server_snapshot.data.set(
						node->get_instance_id(),
						NodeData(node_id,
								node->get_instance_id(),
								is_controller));
			}
			server_snapshot_node_data = server_snapshot.data.getptr(node->get_instance_id());
			server_snapshot_node_data->id = node_id;

			if (is_controller) {
				// This is a controller, take the ID.
				ERR_FAIL_COND(i + 1 >= raw_snapshot.size());
				i += 1;
				const uint64_t input_id = raw_snapshot_ptr[i];
				server_snapshot.controllers_input_id[node->get_instance_id()] = input_id;

				if (node == scene_rewinder->main_controller) {
					// This is the main controller, store the ID in the
					// utility variable.
					server_snapshot.player_controller_input_id = input_id;
				}
			}

		} else if (variable_name == StringName()) {
			// Check if this is the end, or a new variable is submitted.

			if (v.get_type() == Variant::NIL) {
				// NIL found, so this node is done.
				node = nullptr;
				continue;
			}

			// Take the variable name.

			uint32_t var_id;
			if (v.is_array()) {
				const Vector<Variant> var_data = v;
				ERR_FAIL_COND(var_data.size() != 2);
				ERR_FAIL_COND(var_data[0].get_type() != Variant::INT);
				ERR_FAIL_COND(var_data[1].get_type() != Variant::STRING_NAME);

				var_id = var_data[0];
				variable_name = var_data[1];

				const int index = rewinder_node_data->vars.find(variable_name);

				if (index == -1) {
					rewinder_node_data->vars
							.push_back(
									VarData(
											var_id,
											variable_name,
											Variant(),
											false));
				} else {
					rewinder_node_data->vars.write[index].id = var_id;
				}
			} else if (v.get_type() == Variant::INT) {
				var_id = v;

				const int index = rewinder_node_data->vars.find(var_id);
				if (index == -1) {
					WARN_PRINT("The var with ID `" + itos(var_id) + "` is not know by this peer.");
					// TODO please notify the server that this peer need a full snapshot.
					continue;
				}
				variable_name = rewinder_node_data->vars[index].name;
				rewinder_node_data->vars.write[index].id = var_id;
			} else {
				ERR_FAIL_MSG("The snapshot received seems corrupted.");
			}

			server_snap_variable_index = server_snapshot_node_data->vars
												 .find(variable_name);

			if (server_snap_variable_index == -1) {
				server_snap_variable_index = server_snapshot_node_data->vars.size();

				server_snapshot_node_data->vars
						.push_back(
								VarData(
										var_id,
										variable_name,
										Variant(),
										false));

			} else {
				server_snapshot_node_data->vars
						.write[server_snap_variable_index]
						.id = var_id;
			}

		} else {
			// Take the variable value.

			server_snapshot_node_data->vars
					.write[server_snap_variable_index]
					.value = v;

			// Just reset the variable name so we can continue iterate.
			variable_name = StringName();
			server_snap_variable_index = -1;
		}
	}

	ERR_FAIL_COND_MSG(server_snapshot.player_controller_input_id == 0, "The player controller ID was not part of the received snapshot. It will not be considered.");
}
