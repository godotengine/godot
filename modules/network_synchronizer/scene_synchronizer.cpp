/*************************************************************************/
/*  scene_synchronizer.cpp                                               */
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

/**
	@author AndreaCatania
*/

#include "scene_synchronizer.h"

#include "networked_controller.h"
#include "scene/main/window.h"
#include "scene_diff.h"

void SceneSynchronizer::_bind_methods() {
	BIND_CONSTANT(CHANGE)
	BIND_CONSTANT(SYNC_RECOVER)
	BIND_CONSTANT(SYNC_RESET)
	BIND_CONSTANT(SYNC_REWIND)
	BIND_CONSTANT(SYNC_END)
	BIND_CONSTANT(DEFAULT)
	BIND_CONSTANT(ALWAYS)

	ClassDB::bind_method(D_METHOD("reset_synchronizer_mode"), &SceneSynchronizer::reset_synchronizer_mode);
	ClassDB::bind_method(D_METHOD("clear"), &SceneSynchronizer::clear);

	ClassDB::bind_method(D_METHOD("set_server_notify_state_interval", "interval"), &SceneSynchronizer::set_server_notify_state_interval);
	ClassDB::bind_method(D_METHOD("get_server_notify_state_interval"), &SceneSynchronizer::get_server_notify_state_interval);

	ClassDB::bind_method(D_METHOD("set_comparison_float_tolerance", "tolerance"), &SceneSynchronizer::set_comparison_float_tolerance);
	ClassDB::bind_method(D_METHOD("get_comparison_float_tolerance"), &SceneSynchronizer::get_comparison_float_tolerance);

	ClassDB::bind_method(D_METHOD("register_variable", "node", "variable", "on_change_notify", "flags"), &SceneSynchronizer::register_variable, DEFVAL(StringName()), DEFVAL(NetEventFlag::DEFAULT));
	ClassDB::bind_method(D_METHOD("unregister_variable", "node", "variable"), &SceneSynchronizer::unregister_variable);

	ClassDB::bind_method(D_METHOD("set_skip_rewinding", "node", "variable", "skip_rewinding"), &SceneSynchronizer::set_skip_rewinding);

	ClassDB::bind_method(D_METHOD("get_changed_event_name", "variable"), &SceneSynchronizer::get_changed_event_name);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "method", "flags"), &SceneSynchronizer::track_variable_changes, DEFVAL(NetEventFlag::DEFAULT));
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "method"), &SceneSynchronizer::untrack_variable_changes);

	ClassDB::bind_method(D_METHOD("set_node_as_controlled_by", "node", "controller"), &SceneSynchronizer::set_node_as_controlled_by);

	ClassDB::bind_method(D_METHOD("register_process", "node", "function"), &SceneSynchronizer::register_process);
	ClassDB::bind_method(D_METHOD("unregister_process", "node", "function"), &SceneSynchronizer::unregister_process);

	ClassDB::bind_method(D_METHOD("start_tracking_scene_changes", "diff_handle"), &SceneSynchronizer::start_tracking_scene_changes);
	ClassDB::bind_method(D_METHOD("stop_tracking_scene_changes", "diff_handle"), &SceneSynchronizer::stop_tracking_scene_changes);
	ClassDB::bind_method(D_METHOD("pop_scene_changes", "diff_handle"), &SceneSynchronizer::pop_scene_changes);
	ClassDB::bind_method(D_METHOD("apply_scene_changes", "sync_data"), &SceneSynchronizer::apply_scene_changes);

	ClassDB::bind_method(D_METHOD("is_recovered"), &SceneSynchronizer::is_recovered);
	ClassDB::bind_method(D_METHOD("is_resetted"), &SceneSynchronizer::is_resetted);
	ClassDB::bind_method(D_METHOD("is_rewinding"), &SceneSynchronizer::is_rewinding);

	ClassDB::bind_method(D_METHOD("force_state_notify"), &SceneSynchronizer::force_state_notify);

	ClassDB::bind_method(D_METHOD("_on_peer_connected"), &SceneSynchronizer::_on_peer_connected);
	ClassDB::bind_method(D_METHOD("_on_peer_disconnected"), &SceneSynchronizer::_on_peer_disconnected);

	ClassDB::bind_method(D_METHOD("__clear"), &SceneSynchronizer::__clear);
	ClassDB::bind_method(D_METHOD("_rpc_send_state"), &SceneSynchronizer::_rpc_send_state);
	ClassDB::bind_method(D_METHOD("_rpc_notify_need_full_snapshot"), &SceneSynchronizer::_rpc_notify_need_full_snapshot);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "server_notify_state_interval", PROPERTY_HINT_RANGE, "0.001,10.0,0.0001"), "set_server_notify_state_interval", "get_server_notify_state_interval");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "comparison_float_tolerance", PROPERTY_HINT_RANGE, "0.000001,0.01,0.000001"), "set_comparison_float_tolerance", "get_comparison_float_tolerance");
}

void SceneSynchronizer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			// TODO add a signal that allows to not check this each frame.
			if (unlikely(peer_ptr != get_multiplayer()->get_network_peer().ptr())) {
				reset_synchronizer_mode();
			}

			const int lowest_priority_number = INT32_MAX;
			ERR_FAIL_COND_MSG(get_process_priority() != lowest_priority_number, "The process priority MUST not be changed, it's likely there is a better way of doing what you are trying to do, if you really need it please open an issue.");

			process();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			__clear();
			reset_synchronizer_mode();

			get_multiplayer()->connect("network_peer_connected", Callable(this, "_on_peer_connected"));
			get_multiplayer()->connect("network_peer_disconnected", Callable(this, "_on_peer_disconnected"));

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			get_multiplayer()->disconnect("network_peer_connected", Callable(this, "_on_peer_connected"));
			get_multiplayer()->disconnect("network_peer_disconnected", Callable(this, "_on_peer_disconnected"));

			__clear();

			if (synchronizer) {
				memdelete(synchronizer);
				synchronizer = nullptr;
				synchronizer_type = SYNCHRONIZER_TYPE_NULL;
			}

			set_physics_process_internal(false);
		}
	}
}

SceneSynchronizer::SceneSynchronizer() {
	rpc_config("__clear", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_send_state", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_notify_need_full_snapshot", MultiplayerAPI::RPC_MODE_REMOTE);
}

SceneSynchronizer::~SceneSynchronizer() {
	__clear();
	if (synchronizer) {
		memdelete(synchronizer);
		synchronizer = nullptr;
		synchronizer_type = SYNCHRONIZER_TYPE_NULL;
	}
}

void SceneSynchronizer::set_server_notify_state_interval(real_t p_interval) {
	server_notify_state_interval = p_interval;
}

real_t SceneSynchronizer::get_server_notify_state_interval() const {
	return server_notify_state_interval;
}

void SceneSynchronizer::set_comparison_float_tolerance(real_t p_tolerance) {
	comparison_float_tolerance = p_tolerance;
}

real_t SceneSynchronizer::get_comparison_float_tolerance() const {
	return comparison_float_tolerance;
}

void SceneSynchronizer::register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify, NetEventFlag p_flags) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	Ref<NetUtility::NodeData> node_data = register_node(p_node);
	ERR_FAIL_COND(node_data.is_null());

	const int id = node_data->vars.find(p_variable);
	if (id == -1) {
		const Variant old_val = p_node->get(p_variable);
		const int var_id = generate_id ? node_data->vars.size() + 1 : 0;
		node_data->vars.push_back(
				NetUtility::VarData(
						var_id,
						p_variable,
						old_val,
						false,
						true));
	} else {
		NetUtility::VarData *ptr = node_data->vars.ptrw();
		ptr[id].enabled = true;
	}

	if (p_node->has_signal(get_changed_event_name(p_variable)) == false) {
		p_node->add_user_signal(MethodInfo(
				get_changed_event_name(p_variable)));
	}

	if (p_on_change_notify != StringName()) {
		track_variable_changes(p_node, p_variable, p_on_change_notify, p_flags);
	}

	synchronizer->on_variable_added(node_data.ptr(), p_variable);
}

void SceneSynchronizer::unregister_variable(Node *p_node, StringName p_variable) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	Ref<NetUtility::NodeData> nd = find_node_data(p_node->get_instance_id());
	ERR_FAIL_COND(nd.is_null());

	const int64_t index = nd->vars.find(p_variable);
	ERR_FAIL_COND(index == -1);

	// Disconnects the eventual connected methods
	List<Connection> connections;
	p_node->get_signal_connection_list(get_changed_event_name(p_variable), &connections);

	for (List<Connection>::Element *e = connections.front(); e != nullptr; e = e->next()) {
		p_node->disconnect(get_changed_event_name(p_variable), e->get().callable);
	}

	nd->vars.write[index].enabled = false;
}

void SceneSynchronizer::set_skip_rewinding(Node *p_node, StringName p_variable, bool p_skip_rewinding) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	Ref<NetUtility::NodeData> nd = find_node_data(p_node->get_instance_id());
	ERR_FAIL_COND(nd.is_null());

	const int64_t index = nd->vars.find(p_variable);
	ERR_FAIL_COND(index == -1);

	nd->vars.write[index].skip_rewinding = p_skip_rewinding;
}

String SceneSynchronizer::get_changed_event_name(StringName p_variable) {
	return "variable_" + p_variable + "_changed";
}

void SceneSynchronizer::track_variable_changes(Node *p_node, StringName p_variable, StringName p_method, NetEventFlag p_flags) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());
	ERR_FAIL_COND(p_method == StringName());

	Ref<NetUtility::NodeData> nd = find_node_data(p_node->get_instance_id());
	ERR_FAIL_COND_MSG(nd.is_null(), "You need to register the variable to track its changes.");
	ERR_FAIL_COND_MSG(nd->vars.find(p_variable) == -1, "You need to register the variable to track its changes.");

	if (p_node->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method)) == false) {
		p_node->connect(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method));
	}
}

void SceneSynchronizer::untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());
	ERR_FAIL_COND(p_method == StringName());

	Ref<NetUtility::NodeData> nd = find_node_data(p_node->get_instance_id());
	ERR_FAIL_COND(nd.is_null());
	ERR_FAIL_COND(nd->vars.find(p_variable) == -1);

	if (p_node->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method))) {
		p_node->disconnect(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method));
	}
}

void SceneSynchronizer::set_node_as_controlled_by(Node *p_node, Node *p_controller) {
	Ref<NetUtility::NodeData> nd = register_node(p_node);
	ERR_FAIL_COND(nd.is_null());
	ERR_FAIL_COND_MSG(nd->is_controller, "A controller can't be controlled by another controller.");

	if (nd->controlled_by) {
#ifdef DEBUG_ENABLED
		CRASH_COND_MSG(node_data_scene.find(nd) != -1, "There is a bug the same node is added twice into the global_nodes_node_data.");
#endif
		// Put the node back into global.
		node_data_scene.push_back(nd);
		nd->controlled_by->controlled_nodes.erase(nd.ptr());
		nd->controlled_by = nullptr;
	}

	if (p_controller) {
		NetworkedController *c = Object::cast_to<NetworkedController>(p_controller);
		ERR_FAIL_COND_MSG(c == nullptr, "The controller must be a node of type: NetworkedController.");

		Ref<NetUtility::NodeData> controller_node_data = register_node(p_controller);
		ERR_FAIL_COND(controller_node_data == nullptr);
		ERR_FAIL_COND_MSG(controller_node_data->is_controller == false, "The node can be only controlled by a controller.");

#ifdef DEBUG_ENABLED
		CRASH_COND_MSG(controller_node_data->controlled_nodes.find(nd.ptr()) != -1, "There is a bug the same node is added twice into the controlled_nodes.");
#endif
		controller_node_data->controlled_nodes.push_back(nd.ptr());
		node_data_scene.erase(nd);
		nd->controlled_by = controller_node_data.ptr();
	}

#ifdef DEBUG_ENABLED
	// The controller is always registered before a node is marked to be
	// controlled by.
	// So assert that no controlled nodes are into globals.
	for (uint32_t i = 0; i < node_data_scene.size(); i += 1) {
		CRASH_COND(node_data_scene[i]->controlled_by != nullptr);
	}

	// And now make sure that all controlled nodes are into the proper controller.
	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		for (uint32_t y = 0; y < node_data_controllers[i]->controlled_nodes.size(); y += 1) {
			CRASH_COND(node_data_controllers[i]->controlled_nodes[y]->controlled_by != node_data_controllers[i].ptr());
		}
	}
#endif
}

void SceneSynchronizer::register_process(Node *p_node, StringName p_function) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_function == StringName());
	Ref<NetUtility::NodeData> node_data = register_node(p_node);
	ERR_FAIL_COND(node_data.is_null());

	if (node_data->functions.find(p_function) == -1) {
		node_data->functions.push_back(p_function);
	}
}

void SceneSynchronizer::unregister_process(Node *p_node, StringName p_function) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_function == StringName());
	Ref<NetUtility::NodeData> node_data = register_node(p_node);
	ERR_FAIL_COND(node_data.is_null());
	node_data->functions.erase(p_function);
}

void SceneSynchronizer::start_tracking_scene_changes(Object *p_diff_handle) const {
	ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "This function is supposed to be called only on server.");
	SceneDiff *diff = Object::cast_to<SceneDiff>(p_diff_handle);
	ERR_FAIL_COND_MSG(diff == nullptr, "The object is not a SceneDiff class.");

	diff->start_tracking_scene_changes(node_data_scene);
}

void SceneSynchronizer::stop_tracking_scene_changes(Object *p_diff_handle) const {
	ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "This function is supposed to be called only on server.");
	SceneDiff *diff = Object::cast_to<SceneDiff>(p_diff_handle);
	ERR_FAIL_COND_MSG(diff == nullptr, "The object is not a SceneDiff class.");

	diff->stop_tracking_scene_changes(this);
}

Variant SceneSynchronizer::pop_scene_changes(Object *p_diff_handle) const {
	ERR_FAIL_COND_V_MSG(
			synchronizer_type != SYNCHRONIZER_TYPE_SERVER,
			Variant(),
			"This function is supposed to be called only on server.");

	SceneDiff *diff = Object::cast_to<SceneDiff>(p_diff_handle);
	ERR_FAIL_COND_V_MSG(
			diff == nullptr,
			Variant(),
			"The object is not a SceneDiff class.");

	ERR_FAIL_COND_V_MSG(
			diff->is_tracking_in_progress(),
			Variant(),
			"You can't pop the changes while the tracking is still in progress.");

	// Generates a sync_data and returns it.
	Vector<Variant> ret;
	for (
			OAHashMap<uint32_t, NodeDiff>::Iterator node_iter = diff->diff.iter();
			node_iter.valid;
			node_iter = diff->diff.next_iter(node_iter)) {
		if (node_iter.value->var_diff.empty() == false) {
			// Set the node id.
			ret.push_back(*node_iter.key);
			for (
					OAHashMap<uint32_t, Variant>::Iterator var_iter = node_iter.value->var_diff.iter();
					var_iter.valid;
					var_iter = node_iter.value->var_diff.next_iter(var_iter)) {
				ret.push_back(*var_iter.key);
				ret.push_back(*var_iter.value);
			}
			// Close the Node data.
			ret.push_back(Variant());
		}
	}

	// Clear the diff data.
	diff->diff.clear();

	return ret.size() > 0 ? Variant(ret) : Variant();
}

void SceneSynchronizer::apply_scene_changes(Variant p_sync_data) {
	ERR_FAIL_COND_MSG(
			synchronizer_type != SYNCHRONIZER_TYPE_CLIENT,
			"This function is not supposed to be called on server.");

	ClientSynchronizer *client_sync = static_cast<ClientSynchronizer *>(synchronizer);

	client_sync->parse_sync_data(
			p_sync_data,
			this,

			// Parse the Node:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data) {},

			// Parse controller:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id) {},

			// Parse variable:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, StringName p_variable_name, const Variant &p_value) {
				SceneSynchronizer *scene_sync = static_cast<SceneSynchronizer *>(p_user_pointer);

				p_node_data->node->set(p_variable_name, p_value);

				p_node_data->node->emit_signal(scene_sync->get_changed_event_name(p_variable_name));
				scene_sync->synchronizer->on_variable_changed(p_node_data, p_variable_name);
			});
}

bool SceneSynchronizer::is_recovered() const {
	return recover_in_progress;
}

bool SceneSynchronizer::is_resetted() const {
	return reset_in_progress;
}

bool SceneSynchronizer::is_rewinding() const {
	return rewinding_in_progress;
}

void SceneSynchronizer::force_state_notify() {
	ERR_FAIL_COND(synchronizer_type != SYNCHRONIZER_TYPE_SERVER);
	ServerSynchronizer *r = static_cast<ServerSynchronizer *>(synchronizer);
	// + 1.0 is just a ridiculous high number to be sure to avoid float
	// precision error.
	r->state_notifier_timer = get_server_notify_state_interval() + 1.0;
}

void SceneSynchronizer::_on_peer_connected(int p_peer) {
	peer_data.set(p_peer, NetUtility::PeerData());
}

void SceneSynchronizer::_on_peer_disconnected(int p_peer) {
	peer_data.remove(p_peer);
}

void SceneSynchronizer::reset_synchronizer_mode() {
	set_physics_process_internal(false);
	const bool was_generating_ids = generate_id;
	generate_id = false;

	if (synchronizer) {
		memdelete(synchronizer);
		synchronizer = nullptr;
		synchronizer_type = SYNCHRONIZER_TYPE_NULL;
	}

	peer_ptr = get_multiplayer()->get_network_peer().ptr();

	if (get_tree() == nullptr || get_tree()->get_network_peer().is_null()) {
		synchronizer_type = SYNCHRONIZER_TYPE_NONETWORK;
		synchronizer = memnew(NoNetSynchronizer(this));
		generate_id = true;

	} else if (get_tree()->is_network_server()) {
		synchronizer_type = SYNCHRONIZER_TYPE_SERVER;
		synchronizer = memnew(ServerSynchronizer(this));
		generate_id = true;
	} else {
		synchronizer_type = SYNCHRONIZER_TYPE_CLIENT;
		synchronizer = memnew(ClientSynchronizer(this));
	}

	// Always runs the SceneSynchronizer last.
	const int lowest_priority_number = INT32_MAX;
	set_process_priority(lowest_priority_number);
	set_physics_process_internal(true);

	if (was_generating_ids != generate_id) {
		organized_node_data.resize(node_data.size());
		for (uint32_t i = 0; i < node_data.size(); i += 1) {
			if (generate_id) {
				node_data[i]->id = i;
				organized_node_data[i] = node_data[i];
			} else {
				node_data[i]->id = UINT32_MAX;
				organized_node_data[i] = Ref<NetUtility::NodeData>();
			}
		}
	}

	if (synchronizer) {
		// Notify the presence all available nodes and its variables to the synchronizer.
		for (uint32_t i = 0; i < node_data.size(); i += 1) {
			synchronizer->on_node_added(node_data[i].ptr());
			for (int y = 0; y < node_data[i]->vars.size(); y += 1) {
				synchronizer->on_variable_added(node_data[i].ptr(), node_data[i]->vars[y].var.name);
			}
		}
	}
}

void SceneSynchronizer::clear() {
	if (synchronizer_type == SYNCHRONIZER_TYPE_NONETWORK) {
		__clear();
	} else {
		ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "The clear function must be called on server");
		__clear();
		rpc("__clear");
	}
}

void SceneSynchronizer::__clear() {
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		Node *node = static_cast<Node *>(ObjectDB::get_instance(node_data[i]->instance_id));
		if (node != nullptr) {
			for (int y = 0; y < node_data[i]->vars.size(); y += 1) {
				// Unregister the variable so the connected variables are
				// correctly removed
				unregister_variable(node, node_data[i]->vars[y].var.name);
			}
		}
	}

	node_data.clear();
	organized_node_data.clear();
	node_data_controllers.clear();
	node_data_scene.clear();

	if (synchronizer) {
		synchronizer->clear();
	}
}

void SceneSynchronizer::_rpc_send_state(Variant p_snapshot) {
	ERR_FAIL_COND(get_tree()->is_network_server() == true);
	synchronizer->receive_snapshot(p_snapshot);
}

void SceneSynchronizer::_rpc_notify_need_full_snapshot() {
	ERR_FAIL_COND(get_tree()->is_network_server() == false);

	const int sender_peer = get_tree()->get_multiplayer()->get_rpc_sender_id();
	NetUtility::PeerData *pd = peer_data.lookup_ptr(sender_peer);
	ERR_FAIL_COND(pd == nullptr);
	pd->need_full_snapshot = true;
}

void SceneSynchronizer::update_peers() {
	if (peer_dirty == false) {
		return;
	}
	peer_dirty = false;

	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		NetUtility::PeerData *pd = peer_data.lookup_ptr(node_data_controllers[i]->node->get_network_master());
		if (pd) {
			pd->controller_id = node_data_controllers[i]->instance_id;
		}
	}
}

Ref<NetUtility::NodeData> SceneSynchronizer::register_node(Node *p_node) {
	ERR_FAIL_COND_V(p_node == nullptr, nullptr);

	Ref<NetUtility::NodeData> nd = find_node_data(p_node->get_instance_id());
	if (unlikely(nd.is_null())) {
		nd.instance();
		nd->id = UINT32_MAX;
		nd->instance_id = p_node->get_instance_id();
		nd->node = p_node;

		NetworkedController *controller = Object::cast_to<NetworkedController>(p_node);
		if (controller) {
			if (unlikely(controller->has_scene_synchronizer())) {
				ERR_FAIL_V_MSG(nullptr, "This controller already has a synchronizer. This is a bug!");
			}

			nd->is_controller = true;
			controller->set_scene_synchronizer(this);
			peer_dirty = true;
		}

		add_node_data(nd);

		NET_DEBUG_PRINT("New node registered" + (generate_id ? String(" #ID: ") + itos(nd->id) : "") + " : " + p_node->get_path());
	}

	return nd;
}

void SceneSynchronizer::add_node_data(Ref<NetUtility::NodeData> p_node_data) {
	if (generate_id) {
#ifdef DEBUG_ENABLED
		// When generate_id is true, the id must always be undefined.
		CRASH_COND(p_node_data->id != UINT32_MAX);
#endif
		p_node_data->id = node_data.size();
	}

#ifdef DEBUG_ENABLED
	// Make sure the registered nodes have an unique ID.
	// Due to an engine bug, it's possible to have two different nodes with the
	// exact same path:
	//		- Create a scene.
	//		- Add a child with the name `BadChild`.
	//		- Instance the scene into another scene.
	//		- Add a child, under the instanced scene, with the name `BadChild`.
	//	Now you have the scene with two different nodes but same path.
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (node_data[i]->node->get_path() == p_node_data->node->get_path()) {
			NET_DEBUG_ERR("You have two different nodes with the same path: " + p_node_data->node->get_path() + ". This will cause troubles. Fix it.");
			break;
		}
	}
#endif
	node_data.push_back(p_node_data);
	organized_node_data.resize(node_data.size());

	if (generate_id) {
		organized_node_data[p_node_data->id] = p_node_data;
	} else {
		if (p_node_data->id != UINT32_MAX) {
			// This node has an ID, make sure to organize it properly.

#ifdef DEBUG_ENABLED
			// The ID is never more than the action node_data size.
			CRASH_COND(p_node_data->id >= node_data.size());
#endif
			organized_node_data.resize(node_data.size());
			organized_node_data[p_node_data->id] = p_node_data;
		}
	}

	if (p_node_data->is_controller) {
		node_data_controllers.push_back(p_node_data);
	} else {
		node_data_scene.push_back(p_node_data);
	}

	synchronizer->on_node_added(p_node_data.ptr());
}

void SceneSynchronizer::set_node_data_id(Ref<NetUtility::NodeData> p_node_data, NetNodeId p_id) {
#ifdef DEBUG_ENABLED
	CRASH_COND_MSG(generate_id, "This function is not supposed to be called, because this instance is generating the IDs");
#endif
	ERR_FAIL_INDEX(p_id, organized_node_data.size());
	p_node_data->id = p_id;
	organized_node_data[p_id] = p_node_data;
	NET_DEBUG_PRINT("NetNodeId: " + itos(p_id) + " just assigned to: " + p_node_data->node->get_path());
}

bool SceneSynchronizer::vec2_evaluation(const Vector2 &a, const Vector2 &b) const {
	return Math::is_equal_approx(a.x, b.x, comparison_float_tolerance) &&
		   Math::is_equal_approx(a.y, b.y, comparison_float_tolerance);
}

bool SceneSynchronizer::vec3_evaluation(const Vector3 &a, const Vector3 &b) const {
	return Math::is_equal_approx(a.x, b.x, comparison_float_tolerance) &&
		   Math::is_equal_approx(a.y, b.y, comparison_float_tolerance) &&
		   Math::is_equal_approx(a.z, b.z, comparison_float_tolerance);
}

bool SceneSynchronizer::synchronizer_variant_evaluation(
		const Variant &v_1,
		const Variant &v_2) const {
	if (v_1.get_type() != v_2.get_type()) {
		return false;
	}

	// Custom evaluation methods
	switch (v_1.get_type()) {
		case Variant::FLOAT: {
			const real_t a(v_1);
			const real_t b(v_2);
			return Math::is_equal_approx(a, b, comparison_float_tolerance);
		}
		case Variant::VECTOR2: {
			return vec2_evaluation(v_1, v_2);
		}
		case Variant::RECT2: {
			const Rect2 a(v_1);
			const Rect2 b(v_2);
			if (vec2_evaluation(a.position, b.position)) {
				if (vec2_evaluation(a.size, b.size)) {
					return true;
				}
			}
			return false;
		}
		case Variant::TRANSFORM2D: {
			const Transform2D a(v_1);
			const Transform2D b(v_2);
			if (vec2_evaluation(a.elements[0], b.elements[0])) {
				if (vec2_evaluation(a.elements[1], b.elements[1])) {
					if (vec2_evaluation(a.elements[2], b.elements[2])) {
						return true;
					}
				}
			}
			return false;
		}
		case Variant::VECTOR3: {
			return vec3_evaluation(v_1, v_2);
		}
		case Variant::QUAT: {
			const Quat a = v_1;
			const Quat b = v_2;
			const Quat r(a - b); // Element wise subtraction.
			return (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w) <= (comparison_float_tolerance * comparison_float_tolerance);
		}
		case Variant::PLANE: {
			const Plane a(v_1);
			const Plane b(v_2);
			if (Math::is_equal_approx(a.d, b.d, comparison_float_tolerance)) {
				if (vec3_evaluation(a.normal, b.normal)) {
					return true;
				}
			}
			return false;
		}
		case Variant::AABB: {
			const AABB a(v_1);
			const AABB b(v_2);
			if (vec3_evaluation(a.position, b.position)) {
				if (vec3_evaluation(a.size, b.size)) {
					return true;
				}
			}
			return false;
		}
		case Variant::BASIS: {
			const Basis a = v_1;
			const Basis b = v_2;
			if (vec3_evaluation(a.elements[0], b.elements[0])) {
				if (vec3_evaluation(a.elements[1], b.elements[1])) {
					if (vec3_evaluation(a.elements[2], b.elements[2])) {
						return true;
					}
				}
			}
			return false;
		}
		case Variant::TRANSFORM: {
			const Transform a = v_1;
			const Transform b = v_2;
			if (vec3_evaluation(a.origin, b.origin)) {
				if (vec3_evaluation(a.basis.elements[0], b.basis.elements[0])) {
					if (vec3_evaluation(a.basis.elements[1], b.basis.elements[1])) {
						if (vec3_evaluation(a.basis.elements[2], b.basis.elements[2])) {
							return true;
						}
					}
				}
			}
			return false;
		}
		case Variant::ARRAY: {
			const Array a = v_1;
			const Array b = v_2;
			if (a.size() != b.size()) {
				return false;
			}
			for (int i = 0; i < a.size(); i += 1) {
				if (synchronizer_variant_evaluation(a[i], b[i]) == false) {
					return false;
				}
			}
			return true;
		}
		case Variant::DICTIONARY: {
			const Dictionary a = v_1;
			const Dictionary b = v_2;

			if (a.size() != b.size()) {
				return false;
			}

			List<Variant> l;
			a.get_key_list(&l);

			for (const List<Variant>::Element *key = l.front(); key; key = key->next()) {
				if (b.has(key->get()) == false) {
					return false;
				}

				if (synchronizer_variant_evaluation(
							a.get(key->get(), Variant()),
							b.get(key->get(), Variant())) == false) {
					return false;
				}
			}

			return true;
		}
		default:
			return v_1 == v_2;
	}
}

bool SceneSynchronizer::is_client() const {
	return synchronizer_type == SYNCHRONIZER_TYPE_CLIENT;
}

void SceneSynchronizer::validate_nodes() {
	LocalVector<Ref<NetUtility::NodeData>> null_objects;

	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (ObjectDB::get_instance(node_data[i]->instance_id) == nullptr) {
			// Mark for removal.
			null_objects.push_back(node_data[i]);
		}
	}

	// Removes the null objects.
	for (uint32_t i = 0; i < null_objects.size(); i += 1) {
		// Invalidate the `NodeData`.
		null_objects[i]->valid = false;

		if (null_objects[i]->controlled_by) {
			null_objects[i]->controlled_by->controlled_nodes.erase(null_objects[i].ptr());
			null_objects[i]->controlled_by = nullptr;
		}

		if (null_objects[i]->is_controller) {
			peer_dirty = true;
		}

		synchronizer->on_node_removed(null_objects[i].ptr());

		node_data.erase(null_objects[i]);
		node_data_controllers.erase(null_objects[i]);
		node_data_scene.erase(null_objects[i]);
		if (null_objects[i]->id < organized_node_data.size()) {
			// Never resize this vector to keep it sort.
			organized_node_data[null_objects[i]->id].unref();
		}
	}
}

Ref<NetUtility::NodeData> SceneSynchronizer::find_node_data(ObjectID p_object_id) {
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (node_data[i].is_null()) {
			continue;
		}
		if (node_data[i]->instance_id == p_object_id) {
			return node_data[i];
		}
	}
	return nullptr;
}

Ref<NetUtility::NodeData> SceneSynchronizer::get_node_data(NetNodeId p_id) {
	ERR_FAIL_INDEX_V(p_id, organized_node_data.size(), nullptr);
	return organized_node_data[p_id];
}

NetUtility::NodeData *SceneSynchronizer::get_controller_node_data(ControllerID p_controller_id) {
	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		if (node_data_controllers[i]->instance_id == p_controller_id) {
			return node_data_controllers[i].ptr();
		}
	}
	return nullptr;
}

void SceneSynchronizer::process() {
	validate_nodes();
	synchronizer->process();
}

void SceneSynchronizer::pull_node_changes(NetUtility::NodeData *p_node_data) {
	Node *node = p_node_data->node;
	const int var_count = p_node_data->vars.size();
	NetUtility::VarData *object_vars = p_node_data->vars.ptrw();

	for (int i = 0; i < var_count; i += 1) {
		if (object_vars[i].enabled == false) {
			continue;
		}

		const Variant old_val = object_vars[i].var.value;
		const Variant new_val = node->get(object_vars[i].var.name);

		if (!synchronizer_variant_evaluation(old_val, new_val)) {
			object_vars[i].var.value = new_val.duplicate(true);
			node->emit_signal(get_changed_event_name(object_vars[i].var.name));
			synchronizer->on_variable_changed(p_node_data, object_vars[i].var.name);
		}
	}
}

Synchronizer::Synchronizer(SceneSynchronizer *p_node) :
		scene_synchronizer(p_node) {
}

Synchronizer::~Synchronizer() {
}

NoNetSynchronizer::NoNetSynchronizer(SceneSynchronizer *p_node) :
		Synchronizer(p_node) {}

void NoNetSynchronizer::clear() {
}

void NoNetSynchronizer::process() {
	const real_t delta = scene_synchronizer->get_physics_process_delta_time();

	// Process the scene
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i].ptr();
		nd->process(delta);
	}

	// Process the controllers_node_data
	for (uint32_t i = 0; i < scene_synchronizer->node_data_controllers.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data_controllers[i].ptr();
		static_cast<NetworkedController *>(nd->node)->get_nonet_controller()->process(delta);
	}

	// Pull the changes.
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i].ptr();
		scene_synchronizer->pull_node_changes(nd);
	}
}

void NoNetSynchronizer::receive_snapshot(Variant _p_snapshot) {
}

ServerSynchronizer::ServerSynchronizer(SceneSynchronizer *p_node) :
		Synchronizer(p_node) {}

void ServerSynchronizer::clear() {
	state_notifier_timer = 0.0;
	changes.clear();
}

void ServerSynchronizer::process() {
	const real_t delta = scene_synchronizer->get_physics_process_delta_time();

	// Process the scene
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i].ptr();
		nd->process(delta);
	}

	// Process the controllers_node_data
	for (uint32_t i = 0; i < scene_synchronizer->node_data_controllers.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data_controllers[i].ptr();
		static_cast<NetworkedController *>(nd->node)->get_server_controller()->process(delta);
	}

	// Pull the changes.
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i].ptr();
		scene_synchronizer->pull_node_changes(nd);
	}

	process_snapshot_notificator(delta);
}

void ServerSynchronizer::receive_snapshot(Variant _p_snapshot) {
	// Unreachable
	CRASH_NOW();
}

void ServerSynchronizer::on_node_added(NetUtility::NodeData *p_node_data) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_synchronizer->is_recovered());
#endif
	Change *c = changes.lookup_ptr(p_node_data->instance_id);
	if (c) {
		c->not_known_before = true;
	} else {
		Change change;
		change.not_known_before = true;
		changes.set(p_node_data->instance_id, change);
	}
}

void ServerSynchronizer::on_variable_added(NetUtility::NodeData *p_node_data, StringName p_var_name) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_synchronizer->is_recovered());
#endif
	Change *c = changes.lookup_ptr(p_node_data->instance_id);
	if (c) {
		c->vars.insert(p_var_name);
		c->uknown_vars.insert(p_var_name);
	} else {
		Change change;
		change.vars.insert(p_var_name);
		change.uknown_vars.insert(p_var_name);
		changes.set(p_node_data->instance_id, change);
	}
}

void ServerSynchronizer::on_variable_changed(NetUtility::NodeData *p_node_data, StringName p_var_name) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_synchronizer->is_recovered());
#endif
	Change *c = changes.lookup_ptr(p_node_data->instance_id);
	if (c) {
		c->vars.insert(p_var_name);
	} else {
		Change change;
		change.vars.insert(p_var_name);
		changes.set(p_node_data->instance_id, change);
	}
}

void ServerSynchronizer::process_snapshot_notificator(real_t p_delta) {
	if (scene_synchronizer->peer_data.empty()) {
		// No one is listening.
		return;
	}

	// Notify the state if needed
	state_notifier_timer += p_delta;
	const bool notify_state = state_notifier_timer >= scene_synchronizer->get_server_notify_state_interval();

	if (notify_state) {
		state_notifier_timer = 0.0;
	}

	scene_synchronizer->update_peers();

	Vector<Variant> full_global_nodes_snapshot;
	Vector<Variant> delta_global_nodes_snapshot;
	for (
			OAHashMap<int, NetUtility::PeerData>::Iterator peer_it = scene_synchronizer->peer_data.iter();
			peer_it.valid;
			peer_it = scene_synchronizer->peer_data.next_iter(peer_it)) {
		if (peer_it.value->force_notify_snapshot == false && notify_state == false) {
			continue;
		}

		peer_it.value->force_notify_snapshot = false;

		// TODO improve the controller lookup.
		NetUtility::NodeData *nd = scene_synchronizer->get_controller_node_data(peer_it.value->controller_id);
		// TODO well that's not really true.. I may have peers that doesn't have controllers_node_data in a
		// certain moment. Please improve this mechanism trying to just use the
		// node->get_network_master() to get the peer.
		ERR_CONTINUE_MSG(nd == nullptr, "This should never happen. Likely there is a bug.");

		NetworkedController *controller = static_cast<NetworkedController *>(nd->node);
		if (unlikely(controller->is_enabled() == false)) {
			continue;
		}

		Vector<Variant> snap;
		if (peer_it.value->need_full_snapshot) {
			peer_it.value->need_full_snapshot = false;
			if (full_global_nodes_snapshot.size() == 0) {
				full_global_nodes_snapshot = global_nodes_generate_snapshot(true);
			}
			snap = full_global_nodes_snapshot;
			controller_generate_snapshot(nd, true, snap);
		} else {
			if (delta_global_nodes_snapshot.size() == 0) {
				delta_global_nodes_snapshot = global_nodes_generate_snapshot(false);
			}
			snap = delta_global_nodes_snapshot;
			controller_generate_snapshot(nd, false, snap);
		}

		controller->get_server_controller()->notify_send_state();
		scene_synchronizer->rpc_id(*peer_it.key, "_rpc_send_state", snap);
	}

	if (notify_state) {
		// The state got notified, mark this as checkpoint so the next state
		// will contains only the changed things.
		changes.clear();
	}
}

Vector<Variant> ServerSynchronizer::global_nodes_generate_snapshot(bool p_force_full_snapshot) const {
	Vector<Variant> snapshot_data;

	for (uint32_t i = 0; i < scene_synchronizer->node_data_scene.size(); i += 1) {
		const NetUtility::NodeData *node_data = scene_synchronizer->node_data_scene[i].ptr();
		generate_snapshot_node_data(node_data, p_force_full_snapshot, snapshot_data);
	}

	return snapshot_data;
}

void ServerSynchronizer::controller_generate_snapshot(
		const NetUtility::NodeData *p_node_data,
		bool p_force_full_snapshot,
		Vector<Variant> &r_snapshot_result) const {
	CRASH_COND(p_node_data->is_controller == false);

	generate_snapshot_node_data(
			p_node_data,
			p_force_full_snapshot,
			r_snapshot_result);

	for (uint32_t i = 0; i < p_node_data->controlled_nodes.size(); i += 1) {
		generate_snapshot_node_data(
				p_node_data->controlled_nodes[i],
				p_force_full_snapshot,
				r_snapshot_result);
	}
}

void ServerSynchronizer::generate_snapshot_node_data(
		const NetUtility::NodeData *p_node_data,
		bool p_force_full_snapshot,
		Vector<Variant> &r_snapshot_data) const {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
	//  [NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
	//  NODE, INPUT ID, VARIABLE, Value, VARIABLE, Value, NIL,
	//  NODE, VARIABLE, Value, VARIABLE, Value, NIL]
	//
	// Each node ends with a NIL, and the NODE and the VARIABLE are special:
	// - NODE, can be an array of two variables [Net Node ID, NodePath] or directly
	//         a Node ID. Obviously the array is sent only the first time.
	// - INPUT ID, this is optional and is used only when the node is a controller.
	// - VARIABLE, can be an array with the ID and the variable name, or just
	//              the ID; similarly as is for the NODE the array is send only
	//              the first time.

	if (p_node_data->node == nullptr || p_node_data->node->is_inside_tree() == false) {
		return;
	}

	const Change *change = changes.lookup_ptr(p_node_data->instance_id);

	// Insert NODE DATA.
	Variant snap_node_data;
	if (p_force_full_snapshot || (change != nullptr && change->not_known_before)) {
		Vector<Variant> _snap_node_data;
		_snap_node_data.resize(2);
		_snap_node_data.write[0] = p_node_data->id;
		_snap_node_data.write[1] = p_node_data->node->get_path();
		snap_node_data = _snap_node_data;
	} else {
		// This node is already known on clients, just set the node ID.
		snap_node_data = p_node_data->id;
	}

	const bool node_has_changes = p_force_full_snapshot || (change != nullptr && change->vars.empty() == false);

	if (p_node_data->is_controller) {
		NetworkedController *controller = static_cast<NetworkedController *>(p_node_data->node);

		// TODO make sure to skip un-active controllers_node_data.
		//  This may no more needed, since the interpolator got integrated and
		//  the only time the controller is sync is when it's needed.
		if (likely(controller->get_current_input_id() != UINT32_MAX)) {
			// This is a controller, always sync it.
			r_snapshot_data.push_back(snap_node_data);
			r_snapshot_data.push_back(controller->get_current_input_id());
		} else {
			// The first ID id is not yet arrived, so just skip this node.
			return;
		}
	} else {
		if (node_has_changes) {
			r_snapshot_data.push_back(snap_node_data);
		} else {
			// It has no changes, skip this node.
			return;
		}
	}

	if (node_has_changes) {
		// Insert the node variables.
		const int size = p_node_data->vars.size();
		const NetUtility::VarData *vars = &p_node_data->vars[0];
		for (int i = 0; i < size; i += 1) {
			if (vars[i].enabled == false) {
				continue;
			}

			if (p_force_full_snapshot == false && change->vars.has(vars[i].var.name) == false) {
				// This is a delta snapshot and this variable is the same as
				// before. Skip it.
				continue;
			}

			Variant var_info;
			if (p_force_full_snapshot || change->uknown_vars.has(vars[i].var.name)) {
				Vector<Variant> _var_info;
				_var_info.resize(2);
				_var_info.write[0] = vars[i].id;
				_var_info.write[1] = vars[i].var.name;
				var_info = _var_info;
			} else {
				var_info = vars[i].id;
			}

			r_snapshot_data.push_back(var_info);
			r_snapshot_data.push_back(vars[i].var.value);
		}
	}

	// Insert NIL.
	r_snapshot_data.push_back(Variant());
}

ClientSynchronizer::ClientSynchronizer(SceneSynchronizer *p_node) :
		Synchronizer(p_node) {
	clear();
}

void ClientSynchronizer::clear() {
	node_paths.clear();
	last_received_snapshot.input_id = UINT32_MAX;
	last_received_snapshot.node_vars.clear();
	client_snapshots.clear();
	server_snapshots.clear();
}

void ClientSynchronizer::process() {
	if (player_controller_node_data == nullptr) {
		// No player controller, nothing to do.
		return;
	}

	const real_t delta = scene_synchronizer->get_physics_process_delta_time();
	const real_t iteration_per_second = Engine::get_singleton()->get_iterations_per_second();

	NetworkedController *controller = static_cast<NetworkedController *>(player_controller_node_data->node);
	PlayerController *player_controller = controller->get_player_controller();

	// Reset this here, so even when `sub_ticks` is zero (and it's not
	// updated due to process is not called), we can still have the corect
	// data.
	controller->player_set_has_new_input(false);

	// Due to some lag we may want to speed up the input_packet
	// generation, for this reason here I'm performing a sub tick.
	//
	// keep in mind that we are just pretending that the time
	// is advancing faster, for this reason we are still using
	// `delta` to step the controllers_node_data.
	//
	// The dolls may want to speed up too, so to consume the inputs faster
	// and get back in time with the server.
	int sub_ticks = player_controller->calculates_sub_ticks(delta, iteration_per_second);

	while (sub_ticks > 0) {
		// Process the scene.
		for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
			NetUtility::NodeData *nd = scene_synchronizer->node_data[i].ptr();
			nd->process(delta);
		}

		// Process the player controllers_node_data.
		player_controller->process(delta);

		// Pull the changes.
		for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
			NetUtility::NodeData *nd = scene_synchronizer->node_data[i].ptr();
			scene_synchronizer->pull_node_changes(nd);
		}

		if (controller->player_has_new_input()) {
			store_snapshot();
		}

		sub_ticks -= 1;
	}

	scene_synchronizer->recover_in_progress = true;
	process_controllers_recovery(delta);
	scene_synchronizer->recover_in_progress = false;
}

void ClientSynchronizer::receive_snapshot(Variant p_snapshot) {
	// The received snapshot is parsed and stored into the `last_received_snapshot`
	// that contains always the last received snapshot.
	// Later, the snapshot is stored into the server queue.
	// In this way, we are free to pop snapshot from the queue without wondering
	// about losing the data. Indeed the received snapshot is just and
	// incremental update so the last received data is always needed to fully
	// reconstruct it.

	// Parse server snapshot.
	const bool success = parse_snapshot(p_snapshot);

	if (success == false) {
		return;
	}

	// Finalize data.

	store_controllers_snapshot(
			last_received_snapshot,
			server_snapshots);
}

void ClientSynchronizer::on_node_added(NetUtility::NodeData *p_node_data) {
	if (p_node_data->is_controller == false) {
		// Nothing to do.
		return;
	}
	ERR_FAIL_COND_MSG(player_controller_node_data != nullptr, "Only one player controller is supported, at the moment.");
	if (static_cast<NetworkedController *>(p_node_data->node)->is_player_controller()) {
		player_controller_node_data = p_node_data;
	}
}

void ClientSynchronizer::on_node_removed(NetUtility::NodeData *p_node_data) {
	if (player_controller_node_data == p_node_data) {
		player_controller_node_data = nullptr;
	}
}

void ClientSynchronizer::store_snapshot() {
	NetworkedController *controller = static_cast<NetworkedController *>(player_controller_node_data->node);

	if (client_snapshots.size() > 0 && controller->get_current_input_id() <= client_snapshots.back().input_id) {
		NET_DEBUG_ERR("During snapshot creation, for controller " + controller->get_path() + ", was found an ID for an older snapshots. New input ID: " + itos(controller->get_current_input_id()) + " Last saved snapshot input ID: " + itos(client_snapshots.back().input_id) + ". This snapshot is not stored.");
		return;
	}

	client_snapshots.push_back(NetUtility::Snapshot());

	NetUtility::Snapshot &snap = client_snapshots.back();
	snap.input_id = controller->get_current_input_id();

	snap.node_vars.resize(scene_synchronizer->node_data.size());

	// TODO can I just iterate over the `node_data`, instead to iterate each
	//      type of node separately?

	// Store the state of all the global nodes.
	for (uint32_t i = 0; i < scene_synchronizer->node_data_scene.size(); i += 1) {
		const NetUtility::NodeData *node_data = scene_synchronizer->node_data_scene[i].ptr();
		if (node_data->id >= snap.node_vars.size()) {
			// Skip this node, it doesn't have a valid ID.
			ERR_FAIL_COND_MSG(node_data->id != UINT32_MAX, "[BUG], because it's not expected that the client has a node with the NetNodeId bigger than the registered node count.");
			continue;
		}
		snap.node_vars[node_data->id] = node_data->vars;
	}

	if (player_controller_node_data->id >= snap.node_vars.size()) {
		ERR_FAIL_COND_MSG(player_controller_node_data->id != UINT32_MAX, "[BUG], because it's not expected that the client has a node [controller] with the NetNodeId bigger than the registered node count.");
		NET_DEBUG_PRINT("The controller node doesn't have a NetNodeId yet.");
	} else {
		// Store the controller state.
		snap.node_vars[player_controller_node_data->id] = player_controller_node_data->vars;
	}

	// Store the controlled node state.
	for (uint32_t i = 0; i < player_controller_node_data->controlled_nodes.size(); i += 1) {
		const NetUtility::NodeData *node_data = player_controller_node_data->controlled_nodes[i];
		if (node_data->id >= snap.node_vars.size()) {
			// Skip this node, it doesn't have a valid ID.
			ERR_FAIL_COND_MSG(node_data->id != UINT32_MAX, "[BUG], because it's not expected that the client has a node [controller node] with the NetNodeId bigger than the registered node count.");
			continue;
		}
		snap.node_vars[node_data->id] = node_data->vars;
	}
}

void ClientSynchronizer::store_controllers_snapshot(
		const NetUtility::Snapshot &p_snapshot,
		std::deque<NetUtility::Snapshot> &r_snapshot_storage) {
	// Put the parsed snapshot into the queue.

	if (p_snapshot.input_id == UINT32_MAX) {
		// The snapshot doesn't have any info for this controller; Skip it.
		return;
	}

	if (r_snapshot_storage.empty() == false) {
		// Make sure the snapshots are stored in order.
		const uint32_t last_stored_input_id = r_snapshot_storage.back().input_id;
		if (p_snapshot.input_id == last_stored_input_id) {
			// Update the snapshot.
			r_snapshot_storage.back() = p_snapshot;
			return;
		} else {
			ERR_FAIL_COND_MSG(p_snapshot.input_id < last_stored_input_id, "This snapshot (with ID: " + itos(p_snapshot.input_id) + ") is not expected because the last stored id is: " + itos(last_stored_input_id));
		}
	}

	r_snapshot_storage.push_back(p_snapshot);
}

void ClientSynchronizer::process_controllers_recovery(real_t p_delta) {
	// The client is responsible to recover only its local controller, while all
	// the other controllers_node_data (dolls) have their state interpolated. There is
	// no need to check the correctness of the doll state nor the needs to
	// rewind those.
	//
	// The scene, (global nodes), are always in sync with the reference frame
	// of the client.

	NetworkedController *controller = static_cast<NetworkedController *>(player_controller_node_data->node);
	PlayerController *player_controller = controller->get_player_controller();

	// --- Phase one: find the snapshot to check. ---
	if (server_snapshots.empty()) {
		// No snapshots to recover for this controller. Nothing to do.
		return;
	}

#ifdef DEBUG_ENABLED
	if (client_snapshots.empty() == false) {
		// The SceneSynchronizer and the PlayerController are always in sync.
		CRASH_COND(client_snapshots.back().input_id != player_controller->last_known_input());
	}
#endif

	// Find the best recoverable input_id.
	uint32_t checkable_input_id = UINT32_MAX;
	// Find the best snapshot to recover from the one already
	// processed.
	if (client_snapshots.empty() == false) {
		for (
				auto s_snap = server_snapshots.rbegin();
				checkable_input_id == UINT32_MAX && s_snap != server_snapshots.rend();
				++s_snap) {
			for (auto c_snap = client_snapshots.begin(); c_snap != client_snapshots.end(); ++c_snap) {
				if (c_snap->input_id == s_snap->input_id) {
					// Server snapshot also found on client, can be checked.
					checkable_input_id = c_snap->input_id;
					break;
				}
			}
		}
	} else {
		// No client input, this happens when the stream is paused.
		process_paused_controller_recovery(p_delta);
		return;
	}

	if (checkable_input_id == UINT32_MAX) {
		// No snapshot found, nothing to do.
		return;
	}

#ifdef DEBUG_ENABLED
	// Unreachable cause the above check
	CRASH_COND(server_snapshots.empty());
	CRASH_COND(client_snapshots.empty());
#endif

	// Drop all the old server snapshots until the one that we need.
	while (server_snapshots.front().input_id < checkable_input_id) {
		server_snapshots.pop_front();
	}

	// Drop all the old client snapshots until the one that we need.
	while (client_snapshots.front().input_id < checkable_input_id) {
		client_snapshots.pop_front();
	}

#ifdef DEBUG_ENABLED
	// These are unreachable at this point.
	CRASH_COND(server_snapshots.empty());
	CRASH_COND(server_snapshots.front().input_id != checkable_input_id);

	// This is unreachable, because we store all the client shapshots
	// each time a new input is processed. Since the `checkable_input_id`
	// is taken by reading the processed doll inputs, it's guaranteed
	// that here the snapshot exists.
	CRASH_COND(client_snapshots.empty());
	CRASH_COND(client_snapshots.front().input_id != checkable_input_id);
#endif

	// --- Phase two: compare the server snapshot with the client snapshot. ---
	bool need_recover = false;
	bool recover_controller = false;
	LocalVector<NetUtility::NodeData *> nodes_to_recover;
	LocalVector<NetUtility::PostponedRecover> postponed_recover;

	nodes_to_recover.reserve(server_snapshots.front().node_vars.size());
	for (uint32_t net_node_id = 0; net_node_id < server_snapshots.front().node_vars.size(); net_node_id += 1) {
		Ref<NetUtility::NodeData> rew_node_data = scene_synchronizer->get_node_data(net_node_id);
		if (rew_node_data == nullptr) {
			continue;
		}

		bool recover_this_node = false;
		const Vector<NetUtility::VarData> *c_vars = client_snapshots.front().node_vars.lookup_ptr(*s_snap_it.key);
		if (net_node_id >= client_snapshots.front().node_vars.size()) {
			NET_DEBUG_PRINT("Rewind is needed because the client snapshot doesn't contain this node: " + rew_node_data->node->get_path());
			recover_this_node = true;
		} else {
			NetUtility::PostponedRecover rec;

			const bool different = compare_vars(
					rew_node_data.ptr(),
					server_snapshots.front().node_vars[net_node_id],
					client_snapshots.front().node_vars[net_node_id],
					rec.vars);

			if (different) {
				NET_DEBUG_PRINT("Rewind is needed because the node on client is different: " + rew_node_data->node->get_path());
				recover_this_node = true;
			} else if (rec.vars.size() > 0) {
				rec.node_data = rew_node_data.ptr();
				postponed_recover.push_back(rec);
			}
		}

		if (recover_this_node) {
			need_recover = true;
			if (rew_node_data->controlled_by != nullptr ||
					rew_node_data->is_controller) {
				// Controller node.
				recover_controller = true;
			} else {
				nodes_to_recover.push_back(rew_node_data.ptr());
			}
		}
	}

	// Popout the client snapshot.
	client_snapshots.pop_front();

	// --- Phase three: recover and reply. ---

	if (need_recover) {
		NET_DEBUG_PRINT("Recover input: " + itos(checkable_input_id) + " - Last input: " + itos(player_controller->get_stored_input_id(-1)));

		if (recover_controller) {
			// Put the controlled and the controllers_node_data into the nodes to
			// rewind.
			// Note, the controller stuffs are added here to ensure that if the
			// controller need a recover, all its nodes are added; no matter
			// at which point the difference is found.
			nodes_to_recover.reserve(
					nodes_to_recover.size() +
					player_controller_node_data->controlled_nodes.size() +
					1);

			nodes_to_recover.push_back(player_controller_node_data);

			for (
					uint32_t y = 0;
					y < player_controller_node_data->controlled_nodes.size();
					y += 1) {
				nodes_to_recover.push_back(player_controller_node_data->controlled_nodes[y]);
			}
		}

		// Apply the server snapshot so to go back in time till that moment,
		// so to be able to correctly reply the movements.
		scene_synchronizer->reset_in_progress = true;
		for (uint32_t i = 0; i < nodes_to_recover.size(); i += 1) {

			if (nodes_to_recover[i]->id >= server_snapshots.front().node_vars.size()) {
				NET_DEBUG_WARN("The node: " + nodes_to_recover[i]->node->get_path() + " was not found on the server snapshot, this is not supposed to happen a lot.");
				continue;
			}

			Node *node = nodes_to_recover[i]->node;
			const Vector<NetUtility::VarData> s_vars = server_snapshots.front().node_vars[nodes_to_recover[i]->id];
			const NetUtility::VarData *s_vars_ptr = s_vars.ptr();
			NetUtility::VarData *nodes_to_recover_vars_ptr = nodes_to_recover[i]->vars.ptrw();

			NET_DEBUG_PRINT("Full reset node: " + node->get_path());

			for (int v = 0; v < s_vars.size(); v += 1) {
				node->set(s_vars_ptr[v].var.name, s_vars_ptr[v].var.value);

				// Set the value on the synchronizer too.
				const int rew_var_index = nodes_to_recover[i]->vars.find(s_vars_ptr[v].var.name);
				// Unreachable, because when the snapshot is received the
				// algorithm make sure the `scene_synchronizer` is traking the
				// variable.
				CRASH_COND(rew_var_index <= -1);

				NET_DEBUG_PRINT(" |- Variable: " + s_vars_ptr[v].var.name + " New value: " + s_vars_ptr[v].var.value);

				nodes_to_recover_vars_ptr[rew_var_index].var.value = s_vars_ptr[v].var.value.duplicate(true);

				node->emit_signal(
						scene_synchronizer->get_changed_event_name(
								s_vars_ptr[v].var.name));
			}
		}
		scene_synchronizer->reset_in_progress = false;

		// Rewind phase.

		scene_synchronizer->rewinding_in_progress = true;
		const int remaining_inputs = player_controller->notify_input_checked(checkable_input_id);
#ifdef DEBUG_ENABLED
		// Unreachable because the SceneSynchronizer and the PlayerController
		// have the same stored data at this point.
		CRASH_COND(client_snapshots.size() != size_t(remaining_inputs));
#endif

		bool has_next = false;
		for (int i = 0; i < remaining_inputs; i += 1) {
			// Step 1 -- Process the nodes into the scene that need to be
			// processed.
			for (
					uint32_t r = 0;
					r < nodes_to_recover.size();
					r += 1) {
				nodes_to_recover[r]->process(p_delta);
#ifdef DEBUG_ENABLED
				if (nodes_to_recover[r]->functions.size()) {
					NET_DEBUG_PRINT("Rewind, processed node: " + nodes_to_recover[r]->node->get_path());
				}
#endif
			}

			if (recover_controller) {
				// Step 2 -- Process the controller.
				has_next = controller->process_instant(i, p_delta);
				NET_DEBUG_PRINT("Rewind, processed controller: " + controller->get_path());
			}

			// Step 3 -- Pull node changes and Update snapshots.
			for (
					uint32_t r = 0;
					r < nodes_to_recover.size();
					r += 1) {
				scene_synchronizer->pull_node_changes(nodes_to_recover[r]);

				// Update client snapshot.
				if (client_snapshots[i].node_vars.size() <= nodes_to_recover[r]->id) {
					client_snapshots[i].node_vars.resize(nodes_to_recover[r]->id + 1);
				}
				client_snapshots[i].node_vars[nodes_to_recover[r]->id] = nodes_to_recover[r]->vars;
			}
		}

#ifdef DEBUG_ENABLED
		// Unreachable because the above loop consume all instants.
		CRASH_COND(has_next);
#endif

		scene_synchronizer->rewinding_in_progress = false;
	} else {
		// Apply found differences without rewind.
		scene_synchronizer->reset_in_progress = true;
		for (uint32_t i = 0; i < postponed_recover.size(); i += 1) {
			NetUtility::NodeData *rew_node_data = postponed_recover[i].node_data;
			Node *node = rew_node_data->node;
			const NetUtility::Var *vars = postponed_recover[i].vars.ptr();

			NET_DEBUG_PRINT("[Snapshot partial reset] Node: " + node->get_path());

			{
				NetUtility::VarData *rew_node_data_vars_ptr = rew_node_data->vars.ptrw();
				for (int v = 0; v < postponed_recover[i].vars.size(); v += 1) {
					node->set(vars[v].name, vars[v].value);

					// Set the value on the synchronizer too.
					const int rew_var_index = rew_node_data->vars.find(vars[v].name);
					// Unreachable, because when the snapshot is received the
					// algorithm make sure the `scene_synchronizer` is traking the
					// variable.
					CRASH_COND(rew_var_index <= -1);

					rew_node_data_vars_ptr[rew_var_index].var.value = vars[v].value.duplicate(true);

					NET_DEBUG_PRINT(" |- Variable: " + vars[v].name + "; value: " + vars[v].value);
					node->emit_signal(scene_synchronizer->get_changed_event_name(vars[v].name));
				}
			}

			// Update the last client snapshot.
			if (client_snapshots.empty() == false) {
				if (client_snapshots.back().node_vars.size() <= rew_node_data->id) {
					client_snapshots.back().node_vars.resize(rew_node_data->id + 1);
				}
				client_snapshots.back().node_vars[rew_node_data->id] = rew_node_data->vars;
			}
		}
		scene_synchronizer->reset_in_progress = false;

		player_controller->notify_input_checked(checkable_input_id);
	}

	// Popout the server snapshot.
	server_snapshots.pop_front();
}

void ClientSynchronizer::process_paused_controller_recovery(real_t p_delta) {
#ifdef DEBUG_ENABLED
	CRASH_COND(server_snapshots.empty());
	CRASH_COND(client_snapshots.empty() == false);
#endif

	// Drop the snapshots till the newest.
	while (server_snapshots.size() != 1) {
		server_snapshots.pop_front();
	}

#ifdef DEBUG_ENABLED
	CRASH_COND(server_snapshots.empty());
#endif
	scene_synchronizer->recover_in_progress = true;
	for (uint32_t net_node_id = 0; net_node_id < server_snapshots.front().node_vars.size(); net_node_id += 1) {
		Ref<NetUtility::NodeData> rew_node_data = scene_synchronizer->get_node_data(net_node_id);
		if (rew_node_data.is_null()) {
			continue;
		}

		Node *node = rew_node_data->node;

		const NetUtility::VarData *vars_ptr = server_snapshots.front().node_vars[net_node_id].ptr();
		for (int v = 0; v < server_snapshots.front().node_vars[net_node_id].size(); v += 1) {
			if (!scene_synchronizer->synchronizer_variant_evaluation(
						node->get(vars_ptr[v].var.name),
						vars_ptr[v].var.value)) {
				// Different
				node->set(vars_ptr[v].var.name, vars_ptr[v].var.value);
				NET_DEBUG_PRINT("[Snapshot paused controller] Node: " + node->get_path());
				NET_DEBUG_PRINT(" |- Variable: " + vars_ptr[v].var.name + "; value: " + vars_ptr[v].var.value);
				node->emit_signal(scene_synchronizer->get_changed_event_name(vars_ptr[v].var.name));
			}
		}
	}
	scene_synchronizer->recover_in_progress = false;
	server_snapshots.pop_front();
}

bool ClientSynchronizer::parse_sync_data(
		Variant p_sync_data,
		void *p_user_pointer,
		void (*p_node_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data),
		void (*p_controller_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id),
		void (*p_variable_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, StringName p_variable_name, const Variant &p_value)) {
	// The sync data is an array that contains the scene informations.
	// It's used for several things, for this reason this function allows to
	// customize the parsing.
	//
	// The data is composed as follows:
	//  [NODE, VARIABLE, Value, VARIABLE, Value, VARIABLE, value, NIL,
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

	if (p_sync_data.get_type() == Variant::NIL) {
		// Nothing to do.
		return true;
	}

	ERR_FAIL_COND_V(!p_sync_data.is_array(), false);

	const Vector<Variant> raw_snapshot = p_sync_data;
	const Variant *raw_snapshot_ptr = raw_snapshot.ptr();

	NetUtility::NodeData *synchronizer_node_data = nullptr;
	uint32_t var_id = 0;
	StringName variable_name;

	for (int snap_data_index = 0; snap_data_index < raw_snapshot.size(); snap_data_index += 1) {
		const Variant v = raw_snapshot_ptr[snap_data_index];
		if (synchronizer_node_data == nullptr) {
			// Node is null so we expect `v` has the node info.

			bool skip_this_node = false;
			uint32_t net_node_id = UINT32_MAX;
			Node *node = nullptr;
			NodePath node_path;

			if (v.is_array()) {
				// Node info are in verbose form, extract it.

				const Vector<Variant> node_data = v;
				ERR_FAIL_COND_V(node_data.size() != 2, false);
				ERR_FAIL_COND_V_MSG(node_data[0].get_type() != Variant::INT, false, "This snapshot is corrupted.");
				ERR_FAIL_COND_V_MSG(node_data[1].get_type() != Variant::NODE_PATH, false, "This snapshot is corrupted.");

				net_node_id = node_data[0];
				node_path = node_data[1];

				// Associate the ID with the path.
				node_paths.set(net_node_id, node_path);

			} else if (v.get_type() == Variant::INT) {
				// Node info are in short form.
				net_node_id = v;
				Ref<NetUtility::NodeData> nd = scene_synchronizer->get_node_data(net_node_id);
				if (nd.is_null() == false) {
					synchronizer_node_data = nd.ptr();
					goto node_lookup_out;
				}
			} else {
				// The arrived snapshot does't seems to be in the expected form.
				ERR_FAIL_V_MSG(false, "This snapshot is corrupted.");
			}

			if (synchronizer_node_data == nullptr) {
				if (node == nullptr) {
					if (node_path.is_empty()) {
						const NodePath *node_path_ptr = node_paths.lookup_ptr(net_node_id);

						if (node_path_ptr == nullptr) {
							// Was not possible lookup the node_path.
							NET_DEBUG_WARN("The node with ID `" + itos(net_node_id) + "` is not know by this peer, this is not supposed to happen.");
							notify_server_full_snapshot_is_needed();
							skip_this_node = true;
							goto node_lookup_check;
						} else {
							node_path = *node_path_ptr;
						}
					}

					node = scene_synchronizer->get_tree()->get_root()->get_node(node_path);

					if (node == nullptr) {
						// The node doesn't exists.
						NET_DEBUG_ERR("The node " + node_path + " still doesn't exist.");
						skip_this_node = true;
						goto node_lookup_check;
					}
				}

				// Register this node, so to make sure the client is tracking it.
				Ref<NetUtility::NodeData> nd = scene_synchronizer->register_node(node);
				if (nd.is_valid()) {
					// Set the node ID.
					scene_synchronizer->set_node_data_id(nd, net_node_id);
					synchronizer_node_data = nd.ptr();
				} else {
					NET_DEBUG_ERR("[BUG] This node " + node->get_path() + " was not know on this client. Though, was not possible to register it.");
					skip_this_node = true;
				}
			}

		node_lookup_check:
			if (skip_this_node || synchronizer_node_data == nullptr) {
				// This node does't exist; skip it entirely.
				for (snap_data_index += 1; snap_data_index < raw_snapshot.size(); snap_data_index += 1) {
					if (raw_snapshot_ptr[snap_data_index].get_type() == Variant::NIL) {
						break;
					}
				}
				ERR_CONTINUE_MSG(true, "This NetNodeId " + itos(net_node_id) + " doesn't exist on this client.");
			}

		node_lookup_out:

			p_node_parse(p_user_pointer, synchronizer_node_data);

			if (synchronizer_node_data->is_controller) {
				// This is a controller, so the next data is the input ID.
				ERR_FAIL_COND_V(snap_data_index + 1 >= raw_snapshot.size(), false);
				snap_data_index += 1;
				const uint32_t input_id = raw_snapshot_ptr[snap_data_index];
				ERR_FAIL_COND_V_MSG(input_id == UINT32_MAX, false, "The server is always able to send input_id, so this snapshot seems corrupted.");

				p_controller_parse(p_user_pointer, synchronizer_node_data, input_id);
			}

		} else if (variable_name == StringName()) {
			// When the node is known and the `variable_name` not, we expect a
			// new variable or the end pf this node data.

			if (v.get_type() == Variant::NIL) {
				// NIL found, so this node is done.
				synchronizer_node_data = nullptr;
				continue;
			}

			// This is a new variable, so let's take the variable name.

			if (v.is_array()) {
				// The variable info are stored in verbose mode.

				const Vector<Variant> var_data = v;
				ERR_FAIL_COND_V(var_data.size() != 2, false);
				ERR_FAIL_COND_V(var_data[0].get_type() != Variant::INT, false);
				ERR_FAIL_COND_V(var_data[1].get_type() != Variant::STRING_NAME, false);

				var_id = var_data[0];
				variable_name = var_data[1];

				const int64_t index = synchronizer_node_data->vars.find(variable_name);

				if (index == -1) {
					// The variable is not known locally, so just add it so
					// to store the variable ID.
					const bool skip_rewinding = false;
					const bool enabled = false;
					synchronizer_node_data->vars
							.push_back(
									NetUtility::VarData(
											var_id,
											variable_name,
											Variant(),
											skip_rewinding,
											enabled));
				} else {
					// The variable is known, just make sure that it has the
					// same server ID.
					synchronizer_node_data->vars.write[index].id = var_id;
				}
			} else if (v.get_type() == Variant::INT) {
				// The variable is stored in the compact form.

				var_id = v;

				const int64_t index = synchronizer_node_data->find_var_by_id(var_id);
				if (index == -1) {
					NET_DEBUG_PRINT("The var with ID `" + itos(var_id) + "` is not know by this peer, this is not supposed to happen.");

					notify_server_full_snapshot_is_needed();

					// Skip the next data since it should be the value, but we
					// can't store it.
					snap_data_index += 1;
					continue;
				} else {
					variable_name = synchronizer_node_data->vars[index].var.name;
					synchronizer_node_data->vars.write[index].id = var_id;
				}

			} else {
				ERR_FAIL_V_MSG(false, "The snapshot received seems corrupted.");
			}

		} else {
			// The node is known, also the variable name is known, so the value
			// is expected.

			p_variable_parse(
					p_user_pointer,
					synchronizer_node_data,
					var_id,
					variable_name,
					v);

			// Just reset the variable name so we can continue iterate.
			variable_name = StringName();
			var_id = 0;
		}
	}

	return true;
}

bool ClientSynchronizer::parse_snapshot(Variant p_snapshot) {
	need_full_snapshot_notified = false;
	last_received_snapshot.input_id = UINT32_MAX;

	ERR_FAIL_COND_V_MSG(
			player_controller_node_data == nullptr,
			false,
			"Is not possible to receive server snapshots if you are not tracking any NetController.");

	struct ParseData {
		NetUtility::Snapshot &snapshot;
		NetUtility::NodeData *player_controller_node_data;
	};

	ParseData parse_data {
		last_received_snapshot,
		player_controller_node_data
	};

	const bool success = parse_sync_data(
			p_snapshot,
			&parse_data,

			// Parse node:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data) {
				ParseData *pd = static_cast<ParseData *>(p_user_pointer);

				// Make sure this node is part of the server node too.
				if (pd->snapshot.node_vars.size() <= p_node_data->id) {
					pd->snapshot.node_vars.resize(p_node_data->id + 1);
				}
			},

			// Parse controller:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id) {
				ParseData *pd = static_cast<ParseData *>(p_user_pointer);
				if (p_node_data == pd->player_controller_node_data) {
					// This is the main controller, store the input ID.
					pd->snapshot.input_id = p_input_id;
				}
			},

			// Parse variable:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, StringName p_variable_name, const Variant &p_value) {
				ParseData *pd = static_cast<ParseData *>(p_user_pointer);

				int server_snap_variable_index = pd->snapshot.node_vars[p_node_data->id].find(p_variable_name);

				if (server_snap_variable_index == -1) {
					// The server snapshot seems not contains this yet.
					server_snap_variable_index = pd->snapshot.node_vars[p_node_data->id].size();

					const bool skip_rewinding = false;
					const bool enabled = true;
					pd->snapshot.node_vars[p_node_data->id].push_back(
							NetUtility::VarData(
									p_var_id,
									p_variable_name,
									Variant(),
									skip_rewinding,
									enabled));

				} else {
					pd->snapshot.node_vars[p_node_data->id].write[server_snap_variable_index].id =
							p_var_id;
				}

				pd->snapshot.node_vars[p_node_data->id].write[server_snap_variable_index].var.value =
						p_value.duplicate(true);
			});

	if (success == false) {
		return false;
	}

	// We espect that the player_controller is updated by this new snapshot,
	// so make sure it's done so.
	if (unlikely(last_received_snapshot.input_id == UINT32_MAX)) {
		NET_DEBUG_PRINT("Recovery aborted, the player controller (" + player_controller_node_data->node->get_path() + ") was not part of the received snapshot, probably the server doesn't have important informations for this peer. NetUtility::Snapshot:");
		NET_DEBUG_PRINT(p_snapshot);
		return false;
	} else {
		return true;
	}
}

bool ClientSynchronizer::compare_vars(
		const NetUtility::NodeData *p_synchronizer_node_data,
		const Vector<NetUtility::VarData> &p_server_vars,
		const Vector<NetUtility::VarData> &p_client_vars,
		Vector<NetUtility::Var> &r_postponed_recover) {
	const NetUtility::VarData *s_vars = p_server_vars.ptr();
	const NetUtility::VarData *c_vars = p_client_vars.ptr();

#ifdef DEBUG_ENABLED
	bool diff = false;
#endif

	for (int s_var_index = 0; s_var_index < p_server_vars.size(); s_var_index += 1) {
		const int c_var_index = p_client_vars.find(s_vars[s_var_index].var.name);
		if (c_var_index == -1) {
			// Variable not found, this is considered a difference.
			NET_DEBUG_PRINT("Difference found on the var name `" + s_vars[s_var_index].var.name + "`, it was not found on client snapshot. Server value: `" + s_vars[s_var_index].var.value + "`.");
#ifdef DEBUG_ENABLED
			diff = true;
#else
			return true;
#endif
		} else {
			// Variable found compare.
			const bool different = !scene_synchronizer->synchronizer_variant_evaluation(s_vars[s_var_index].var.value, c_vars[c_var_index].var.value);

			if (different) {
				const int index = p_synchronizer_node_data->vars.find(s_vars[s_var_index].var.name);
				if (index < 0 || p_synchronizer_node_data->vars[index].skip_rewinding == false) {
					// The vars are different.
					NET_DEBUG_PRINT("Difference found on var name `" + s_vars[s_var_index].var.name + "` Server value: `" + s_vars[s_var_index].var.value + "` Client value: `" + c_vars[c_var_index].var.value + "`.");
#ifdef DEBUG_ENABLED
					diff = true;
#else
					return true;
#endif
				} else {
					// The vars are different, but this variable don't what to
					// trigger a rewind.
					r_postponed_recover.push_back(s_vars[s_var_index].var);
				}
			}
		}
	}

#ifdef DEBUG_ENABLED
	return diff;
#else
	// The vars are not different.
	return false;
#endif
}

void ClientSynchronizer::notify_server_full_snapshot_is_needed() {
	if (need_full_snapshot_notified) {
		return;
	}

	// Notify the server that a full snapshot is needed.
	need_full_snapshot_notified = true;
	scene_synchronizer->rpc_id(1, "_rpc_notify_need_full_snapshot");
}
