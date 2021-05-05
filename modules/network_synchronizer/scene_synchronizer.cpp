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
	BIND_CONSTANT(END_SYNC)
	BIND_CONSTANT(DEFAULT)
	BIND_CONSTANT(SYNC)
	BIND_CONSTANT(ALWAYS)

	ClassDB::bind_method(D_METHOD("reset_synchronizer_mode"), &SceneSynchronizer::reset_synchronizer_mode);
	ClassDB::bind_method(D_METHOD("clear"), &SceneSynchronizer::clear);

	ClassDB::bind_method(D_METHOD("set_server_notify_state_interval", "interval"), &SceneSynchronizer::set_server_notify_state_interval);
	ClassDB::bind_method(D_METHOD("get_server_notify_state_interval"), &SceneSynchronizer::get_server_notify_state_interval);

	ClassDB::bind_method(D_METHOD("set_comparison_float_tolerance", "tolerance"), &SceneSynchronizer::set_comparison_float_tolerance);
	ClassDB::bind_method(D_METHOD("get_comparison_float_tolerance"), &SceneSynchronizer::get_comparison_float_tolerance);

	ClassDB::bind_method(D_METHOD("register_node", "node"), &SceneSynchronizer::register_node_gdscript);
	ClassDB::bind_method(D_METHOD("unregister_node", "node"), &SceneSynchronizer::unregister_node);
	ClassDB::bind_method(D_METHOD("get_node_id", "node"), &SceneSynchronizer::get_node_id);
	ClassDB::bind_method(D_METHOD("get_node_from_id", "id"), &SceneSynchronizer::get_node_from_id);

	ClassDB::bind_method(D_METHOD("register_variable", "node", "variable", "on_change_notify", "flags"), &SceneSynchronizer::register_variable, DEFVAL(StringName()), DEFVAL(NetEventFlag::DEFAULT));
	ClassDB::bind_method(D_METHOD("unregister_variable", "node", "variable"), &SceneSynchronizer::unregister_variable);
	ClassDB::bind_method(D_METHOD("get_variable_id", "node", "variable"), &SceneSynchronizer::get_variable_id);

	ClassDB::bind_method(D_METHOD("start_node_sync", "node"), &SceneSynchronizer::start_node_sync);
	ClassDB::bind_method(D_METHOD("stop_node_sync", "node"), &SceneSynchronizer::stop_node_sync);
	ClassDB::bind_method(D_METHOD("is_node_sync", "node"), &SceneSynchronizer::is_node_sync);

	ClassDB::bind_method(D_METHOD("set_skip_rewinding", "node", "variable", "skip_rewinding"), &SceneSynchronizer::set_skip_rewinding);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "object", "method", "flags"), &SceneSynchronizer::track_variable_changes, DEFVAL(NetEventFlag::DEFAULT));
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "object", "method"), &SceneSynchronizer::untrack_variable_changes);

	ClassDB::bind_method(D_METHOD("set_node_as_controlled_by", "node", "controller"), &SceneSynchronizer::set_node_as_controlled_by);

	ClassDB::bind_method(D_METHOD("controller_add_dependency", "controller", "node"), &SceneSynchronizer::controller_add_dependency);
	ClassDB::bind_method(D_METHOD("controller_remove_dependency", "controller", "node"), &SceneSynchronizer::controller_remove_dependency);
	ClassDB::bind_method(D_METHOD("controller_get_dependency_count", "controller"), &SceneSynchronizer::controller_get_dependency_count);
	ClassDB::bind_method(D_METHOD("controller_get_dependency", "controller", "index"), &SceneSynchronizer::controller_get_dependency);

	ClassDB::bind_method(D_METHOD("register_process", "node", "function"), &SceneSynchronizer::register_process);
	ClassDB::bind_method(D_METHOD("unregister_process", "node", "function"), &SceneSynchronizer::unregister_process);

	ClassDB::bind_method(D_METHOD("start_tracking_scene_changes", "diff_handle"), &SceneSynchronizer::start_tracking_scene_changes);
	ClassDB::bind_method(D_METHOD("stop_tracking_scene_changes", "diff_handle"), &SceneSynchronizer::stop_tracking_scene_changes);
	ClassDB::bind_method(D_METHOD("pop_scene_changes", "diff_handle"), &SceneSynchronizer::pop_scene_changes);
	ClassDB::bind_method(D_METHOD("apply_scene_changes", "sync_data"), &SceneSynchronizer::apply_scene_changes);

	ClassDB::bind_method(D_METHOD("is_recovered"), &SceneSynchronizer::is_recovered);
	ClassDB::bind_method(D_METHOD("is_resetted"), &SceneSynchronizer::is_resetted);
	ClassDB::bind_method(D_METHOD("is_rewinding"), &SceneSynchronizer::is_rewinding);
	ClassDB::bind_method(D_METHOD("is_end_sync"), &SceneSynchronizer::is_end_sync);

	ClassDB::bind_method(D_METHOD("force_state_notify"), &SceneSynchronizer::force_state_notify);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &SceneSynchronizer::set_enabled);
	ClassDB::bind_method(D_METHOD("set_peer_networking_enable", "peer", "enabled"), &SceneSynchronizer::set_peer_networking_enable);
	ClassDB::bind_method(D_METHOD("get_peer_networking_enable", "peer"), &SceneSynchronizer::is_peer_networking_enable);

	ClassDB::bind_method(D_METHOD("is_server"), &SceneSynchronizer::is_server);
	ClassDB::bind_method(D_METHOD("is_client"), &SceneSynchronizer::is_client);
	ClassDB::bind_method(D_METHOD("is_networked"), &SceneSynchronizer::is_networked);

	ClassDB::bind_method(D_METHOD("_on_peer_connected"), &SceneSynchronizer::_on_peer_connected);
	ClassDB::bind_method(D_METHOD("_on_peer_disconnected"), &SceneSynchronizer::_on_peer_disconnected);

	ClassDB::bind_method(D_METHOD("_on_node_removed"), &SceneSynchronizer::_on_node_removed);

	ClassDB::bind_method(D_METHOD("_rpc_send_state"), &SceneSynchronizer::_rpc_send_state);
	ClassDB::bind_method(D_METHOD("_rpc_notify_need_full_snapshot"), &SceneSynchronizer::_rpc_notify_need_full_snapshot);
	ClassDB::bind_method(D_METHOD("_rpc_set_network_enabled", "enabled"), &SceneSynchronizer::_rpc_set_network_enabled);
	ClassDB::bind_method(D_METHOD("_rpc_notify_peer_status", "enabled"), &SceneSynchronizer::_rpc_notify_peer_status);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "server_notify_state_interval", PROPERTY_HINT_RANGE, "0.001,10.0,0.0001"), "set_server_notify_state_interval", "get_server_notify_state_interval");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "comparison_float_tolerance", PROPERTY_HINT_RANGE, "0.000001,0.01,0.000001"), "set_comparison_float_tolerance", "get_comparison_float_tolerance");

	ADD_SIGNAL(MethodInfo("sync_started"));
	ADD_SIGNAL(MethodInfo("sync_paused"));
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

			clear();
			reset_synchronizer_mode();

			get_multiplayer()->connect("network_peer_connected", Callable(this, "_on_peer_connected"));
			get_multiplayer()->connect("network_peer_disconnected", Callable(this, "_on_peer_disconnected"));

			get_tree()->connect("node_removed", Callable(this, "_on_node_removed"));

			// Make sure to reset all the assigned controllers.
			reset_controllers();

			// Init the peers already connected.
			{
				const Vector<int> peer_ids = get_tree()->get_network_connected_peers();
				const int *peer_ids_ptr = peer_ids.ptr();
				for (int i = 0; i < peer_ids.size(); i += 1) {
					_on_peer_connected(peer_ids_ptr[i]);
				}
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			clear_peers();

			get_multiplayer()->disconnect("network_peer_connected", Callable(this, "_on_peer_connected"));
			get_multiplayer()->disconnect("network_peer_disconnected", Callable(this, "_on_peer_disconnected"));

			get_tree()->disconnect("node_removed", Callable(this, "_on_node_removed"));

			clear();

			if (synchronizer) {
				memdelete(synchronizer);
				synchronizer = nullptr;
				synchronizer_type = SYNCHRONIZER_TYPE_NULL;
			}

			set_physics_process_internal(false);

			// Make sure to reset all the assigned controllers.
			reset_controllers();
		}
	}
}

SceneSynchronizer::SceneSynchronizer() {
	rpc_config("_rpc_send_state", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_notify_need_full_snapshot", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_set_network_enabled", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_notify_peer_status", MultiplayerAPI::RPC_MODE_REMOTE);

	// Avoid too much useless re-allocations.
	event_listener.reserve(100);
}

SceneSynchronizer::~SceneSynchronizer() {
	clear();
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

NetUtility::NodeData *SceneSynchronizer::register_node(Node *p_node) {
	ERR_FAIL_COND_V(p_node == nullptr, nullptr);

	NetUtility::NodeData *nd = find_node_data(p_node);
	if (unlikely(nd == nullptr)) {
		nd = memnew(NetUtility::NodeData);
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
			dirty_peers();
		}

		add_node_data(nd);

		NET_DEBUG_PRINT("New node registered" + (generate_id ? String(" #ID: ") + itos(nd->id) : "") + " : " + p_node->get_path());
	}

	return nd;
}

uint32_t SceneSynchronizer::register_node_gdscript(Node *p_node) {
	NetUtility::NodeData *nd = register_node(p_node);
	if (unlikely(nd == nullptr)) {
		return UINT32_MAX;
	}
	return nd->id;
}

void SceneSynchronizer::unregister_node(Node *p_node) {
	ERR_FAIL_COND(p_node == nullptr);

	NetUtility::NodeData *nd = find_node_data(p_node);
	if (unlikely(nd == nullptr)) {
		// Nothing to do.
		return;
	}

	drop_node_data(nd);
}

uint32_t SceneSynchronizer::get_node_id(Node *p_node) {
	ERR_FAIL_COND_V(p_node == nullptr, UINT32_MAX);
	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND_V_MSG(nd == nullptr, UINT32_MAX, "This node " + p_node->get_path() + " is not yet registered, so there is not an available ID.");
	return nd->id;
}

Node *SceneSynchronizer::get_node_from_id(uint32_t p_id) {
	NetUtility::NodeData *nd = get_node_data(p_id);
	ERR_FAIL_COND_V_MSG(nd == nullptr, nullptr, "The ID " + itos(p_id) + " is not assigned to any node.");
	return nd->node;
}

void SceneSynchronizer::register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify, NetEventFlag p_flags) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	NetUtility::NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);

	const int index = node_data->vars.find(p_variable);
	if (index == -1) {
		// The variable is not yet registered.
		const Variant old_val = p_node->get(p_variable);
		const int var_id = generate_id ? node_data->vars.size() : UINT32_MAX;
		node_data->vars.push_back(
				NetUtility::VarData(
						var_id,
						p_variable,
						old_val,
						false,
						true));
	} else {
		// Make sure the var is active.
		node_data->vars[index].enabled = true;
	}

#ifdef DEBUG_ENABLED
	for (uint32_t v = 0; v < node_data->vars.size(); v += 1) {
		// This can't happen, because the ID is always consecutive, or UINT32_MAX.
		CRASH_COND(node_data->vars[v].id != v && node_data->vars[v].id != UINT32_MAX);
	}
#endif

	if (p_on_change_notify != StringName()) {
		track_variable_changes(p_node, p_variable, p_node, p_on_change_notify, p_flags);
	}

	if (synchronizer) {
		synchronizer->on_variable_added(node_data, p_variable);
	}
}

void SceneSynchronizer::unregister_variable(Node *p_node, StringName p_variable) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND(nd == nullptr);

	const int64_t index = nd->vars.find(p_variable);
	ERR_FAIL_COND(index == -1);

	const NetVarId var_id = index;

	// Never remove the variable values, because the order of the vars matters.
	nd->vars[index].enabled = false;

	for (int i = 0; i < nd->vars[var_id].change_listeners.size(); i += 1) {
		const uint32_t event_index = nd->vars[var_id].change_listeners[i];
		// Just erase the tracked variables without removing the listener to
		// keep the order.
		NetUtility::NodeChangeListener ncl;
		ncl.node_data = nd;
		ncl.var_id = var_id;
		event_listener[event_index].watching_vars.erase(ncl);
	}

	nd->vars[index].change_listeners.clear();
}

void SceneSynchronizer::start_node_sync(const Node *p_node) {
	ERR_FAIL_COND(p_node == nullptr);

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND(nd == nullptr);

	nd->sync_enabled = true;
}

void SceneSynchronizer::stop_node_sync(const Node *p_node) {
	ERR_FAIL_COND(p_node == nullptr);

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND(nd == nullptr);

	nd->sync_enabled = false;
}

bool SceneSynchronizer::is_node_sync(const Node *p_node) const {
	ERR_FAIL_COND_V(p_node == nullptr, false);

	const NetUtility::NodeData *nd = find_node_data(p_node);
	if (nd == nullptr) {
		return false;
	}

	return nd->sync_enabled;
}

uint32_t SceneSynchronizer::get_variable_id(Node *p_node, StringName p_variable) {
	ERR_FAIL_COND_V(p_node == nullptr, UINT32_MAX);
	ERR_FAIL_COND_V(p_variable == StringName(), UINT32_MAX);

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND_V_MSG(nd == nullptr, UINT32_MAX, "This node " + p_node->get_path() + "is not registered.");

	const int64_t index = nd->vars.find(p_variable);
	ERR_FAIL_COND_V_MSG(index == -1, UINT32_MAX, "This variable " + p_node->get_path() + ":" + p_variable + " is not registered.");

	return uint32_t(index);
}

void SceneSynchronizer::set_skip_rewinding(Node *p_node, StringName p_variable, bool p_skip_rewinding) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND(nd == nullptr);

	const int64_t index = nd->vars.find(p_variable);
	ERR_FAIL_COND(index == -1);

	nd->vars[index].skip_rewinding = p_skip_rewinding;
}

void SceneSynchronizer::track_variable_changes(Node *p_node, StringName p_variable, Object *p_object, StringName p_method, NetEventFlag p_flags) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());
	ERR_FAIL_COND(p_method == StringName());

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND_MSG(nd == nullptr, "You need to register the variable to track its changes.");

	const int64_t v = nd->vars.find(p_variable);
	ERR_FAIL_COND_MSG(v == -1, "You need to register the variable to track its changes.");

	const NetVarId var_id = v;

	int64_t index;

	{
		NetUtility::ChangeListener listener;
		listener.object_id = p_object->get_instance_id();
		listener.method = p_method;

		index = event_listener.find(listener);

		if (-1 == index) {
			// Add it.
			listener.flag = p_flags;
			listener.method_argument_count = UINT32_MAX;

			// Search the method and get the argument count.
			List<MethodInfo> methods;
			p_object->get_method_list(&methods);
			for (List<MethodInfo>::Element *e = methods.front(); e != nullptr; e = e->next()) {
				if (e->get().name != p_method) {
					continue;
				}

				listener.method_argument_count = e->get().arguments.size();

				break;
			}
			ERR_FAIL_COND_MSG(listener.method_argument_count == UINT32_MAX, "The method " + p_method + " doesn't exist in this node: " + p_node->get_path());

			index = event_listener.size();
			event_listener.push_back(listener);
		} else {
			ERR_FAIL_COND_MSG(event_listener[index].flag != p_flags, "The event listener is already registered with the flag: " + itos(event_listener[index].flag) + ". You can't specify a different one.");
		}
	}

	NetUtility::NodeChangeListener ncl;
	ncl.node_data = nd;
	ncl.var_id = var_id;

	if (event_listener[index].watching_vars.find(ncl) != -1) {
		return;
	}

	event_listener[index].watching_vars.push_back(ncl);
	nd->vars[var_id].change_listeners.push_back(index);
}

void SceneSynchronizer::untrack_variable_changes(Node *p_node, StringName p_variable, Object *p_object, StringName p_method) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());
	ERR_FAIL_COND(p_method == StringName());

	NetUtility::NodeData *nd = find_node_data(p_node);
	ERR_FAIL_COND_MSG(nd == nullptr, "This not is not registered.");

	const int64_t v = nd->vars.find(p_variable);
	ERR_FAIL_COND_MSG(v == -1, "This variable is not registered.");

	const NetVarId var_id = v;

	NetUtility::ChangeListener listener;
	listener.object_id = p_object->get_instance_id();
	listener.method = p_method;

	const int64_t index = event_listener.find(listener);

	ERR_FAIL_COND_MSG(index == -1, "The variable is not know.");

	NetUtility::NodeChangeListener ncl;
	ncl.node_data = nd;
	ncl.var_id = var_id;

	event_listener[index].watching_vars.erase(ncl);
	nd->vars[var_id].change_listeners.erase(index);

	// Don't remove the listener to preserve the order.
}

void SceneSynchronizer::set_node_as_controlled_by(Node *p_node, Node *p_controller) {
	NetUtility::NodeData *nd = register_node(p_node);
	ERR_FAIL_COND(nd == nullptr);
	ERR_FAIL_COND_MSG(nd->is_controller, "A controller can't be controlled by another controller.");

	if (nd->controlled_by) {
		// Put the node back into global.
		nd->controlled_by->controlled_nodes.erase(nd);
		nd->controlled_by = nullptr;
	}

	if (p_controller) {
		NetworkedController *c = Object::cast_to<NetworkedController>(p_controller);
		ERR_FAIL_COND_MSG(c == nullptr, "The controller must be a node of type: NetworkedController.");

		NetUtility::NodeData *controller_node_data = register_node(p_controller);
		ERR_FAIL_COND(controller_node_data == nullptr);
		ERR_FAIL_COND_MSG(controller_node_data->is_controller == false, "The node can be only controlled by a controller.");

#ifdef DEBUG_ENABLED
		CRASH_COND_MSG(controller_node_data->controlled_nodes.find(nd) != -1, "There is a bug the same node is added twice into the controlled_nodes.");
#endif
		controller_node_data->controlled_nodes.push_back(nd);
		nd->controlled_by = controller_node_data;
	}

#ifdef DEBUG_ENABLED
	// Make sure that all controlled nodes are into the proper controller.
	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		for (uint32_t y = 0; y < node_data_controllers[i]->controlled_nodes.size(); y += 1) {
			CRASH_COND(node_data_controllers[i]->controlled_nodes[y]->controlled_by != node_data_controllers[i]);
		}
	}
#endif
}

void SceneSynchronizer::controller_add_dependency(Node *p_controller, Node *p_node) {
	if (is_client() == false) {
		// Nothing to do.
		return;
	}

	NetUtility::NodeData *controller_nd = find_node_data(p_controller);
	ERR_FAIL_COND_MSG(controller_nd == nullptr, "The passed controller (" + p_controller->get_path() + ") is not registered.");
	ERR_FAIL_COND_MSG(controller_nd->is_controller == false, "The node passed as controller (" + p_controller->get_path() + ") is not a controller.");

	NetUtility::NodeData *node_nd = find_node_data(p_node);
	ERR_FAIL_COND_MSG(node_nd == nullptr, "The passed node (" + p_node->get_path() + ") is not registered.");
	ERR_FAIL_COND_MSG(node_nd->is_controller, "The node (" + p_node->get_path() + ") set as dependency is supposed to be just a node.");
	ERR_FAIL_COND_MSG(node_nd->controlled_by != nullptr, "The node (" + p_node->get_path() + ") set as dependency is supposed to be just a node.");

	const int64_t index = controller_nd->dependency_nodes.find(node_nd);
	if (index == -1) {
		controller_nd->dependency_nodes.push_back(node_nd);
		controller_nd->dependency_nodes_end.push_back(UINT32_MAX);
	} else {
		// We already have this dependency, just make sure we don't delete it.
		controller_nd->dependency_nodes_end[index] = UINT32_MAX;
	}
}

void SceneSynchronizer::controller_remove_dependency(Node *p_controller, Node *p_node) {
	if (is_client() == false) {
		// Nothing to do.
		return;
	}

	NetUtility::NodeData *controller_nd = find_node_data(p_controller);
	ERR_FAIL_COND_MSG(controller_nd == nullptr, "The passed controller (" + p_controller->get_path() + ") is not registered.");
	ERR_FAIL_COND_MSG(controller_nd->is_controller == false, "The node passed as controller (" + p_controller->get_path() + ") is not a controller.");

	NetUtility::NodeData *node_nd = find_node_data(p_node);
	ERR_FAIL_COND_MSG(node_nd == nullptr, "The passed node (" + p_node->get_path() + ") is not registered.");
	ERR_FAIL_COND_MSG(node_nd->is_controller, "The node (" + p_node->get_path() + ") set as dependency is supposed to be just a node.");
	ERR_FAIL_COND_MSG(node_nd->controlled_by != nullptr, "The node (" + p_node->get_path() + ") set as dependency is supposed to be just a node.");

	const int64_t index = controller_nd->dependency_nodes.find(node_nd);
	if (index == -1) {
		// Nothing to do, this node is not a dependency.
		return;
	}

	// Instead to remove the dependency immeditaly we have to postpone it till
	// the server confirms the valitity via state.
	// This operation is required otherwise the dependency is remvoved too early,
	// and an eventual rewind may miss it.
	// The actual removal is performed at the end of the sync.
	controller_nd->dependency_nodes_end[index] =
			static_cast<NetworkedController *>(controller_nd->node)->get_current_input_id();
}

int SceneSynchronizer::controller_get_dependency_count(Node *p_controller) const {
	if (is_client() == false) {
		// Nothing to do.
		return 0;
	}

	const NetUtility::NodeData *controller_nd = find_node_data(p_controller);
	ERR_FAIL_COND_V_MSG(controller_nd == nullptr, 0, "The passed controller (" + p_controller->get_path() + ") is not registered.");
	ERR_FAIL_COND_V_MSG(controller_nd->is_controller == false, 0, "The node passed as controller (" + p_controller->get_path() + ") is not a controller.");
	return controller_nd->dependency_nodes.size();
}

Node *SceneSynchronizer::controller_get_dependency(Node *p_controller, int p_index) {
	if (is_client() == false) {
		// Nothing to do.
		return nullptr;
	}

	NetUtility::NodeData *controller_nd = find_node_data(p_controller);
	ERR_FAIL_COND_V_MSG(controller_nd == nullptr, nullptr, "The passed controller (" + p_controller->get_path() + ") is not registered.");
	ERR_FAIL_COND_V_MSG(controller_nd->is_controller == false, nullptr, "The node passed as controller (" + p_controller->get_path() + ") is not a controller.");
	ERR_FAIL_INDEX_V(p_index, int(controller_nd->dependency_nodes.size()), nullptr);

	return controller_nd->dependency_nodes[p_index]->node;
}

void SceneSynchronizer::register_process(Node *p_node, StringName p_function) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_function == StringName());
	NetUtility::NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);

	if (node_data->functions.find(p_function) == -1) {
		node_data->functions.push_back(p_function);
	}
}

void SceneSynchronizer::unregister_process(Node *p_node, StringName p_function) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_function == StringName());
	NetUtility::NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);
	node_data->functions.erase(p_function);
}

void SceneSynchronizer::start_tracking_scene_changes(Object *p_diff_handle) const {
	ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "This function is supposed to be called only on server.");
	SceneDiff *diff = Object::cast_to<SceneDiff>(p_diff_handle);
	ERR_FAIL_COND_MSG(diff == nullptr, "The object is not a SceneDiff class.");

	diff->start_tracking_scene_changes(organized_node_data);
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
	for (NetNodeId node_id = 0; node_id < diff->diff.size(); node_id += 1) {
		if (diff->diff[node_id].size() == 0) {
			// Nothing to do.
			continue;
		}

		bool node_id_in_ret = false;
		for (NetVarId var_id = 0; var_id < diff->diff[node_id].size(); var_id += 1) {
			if (diff->diff[node_id][var_id].is_different == false) {
				continue;
			}
			if (node_id_in_ret == false) {
				node_id_in_ret = true;
				// Set the node id.
				ret.push_back(node_id);
			}
			ret.push_back(var_id);
			ret.push_back(diff->diff[node_id][var_id].value);
		}
		if (node_id_in_ret) {
			// Close the Node data.
			ret.push_back(Variant());
		}
	}

	// Clear the diff data.
	diff->diff.clear();

	return ret.size() > 0 ? Variant(ret) : Variant();
}

void SceneSynchronizer::apply_scene_changes(Variant p_sync_data) {
	ERR_FAIL_COND_MSG(is_client() == false, "This function is not supposed to be called on server.");

	ClientSynchronizer *client_sync = static_cast<ClientSynchronizer *>(synchronizer);

	change_events_begin(NetEventFlag::CHANGE);

	const bool success = client_sync->parse_sync_data(
			p_sync_data,
			this,

			// Parse the Node:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data) {},

			// Parse controller:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id) {},

			// Parse variable:
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, const Variant &p_value) {
				SceneSynchronizer *scene_sync = static_cast<SceneSynchronizer *>(p_user_pointer);

				const Variant current_val = p_node_data->vars[p_var_id].var.value;

				if (scene_sync->synchronizer_variant_evaluation(current_val, p_value) == false) {
					// There is a difference.
					// Set the new value.
					p_node_data->vars[p_var_id].var.value = p_value;
					p_node_data->node->set(
							p_node_data->vars[p_var_id].var.name,
							p_value);

					// Add an event.
					scene_sync->change_event_add(
							p_node_data,
							p_var_id,
							current_val);
				}
			});

	if (success == false) {
		NET_DEBUG_ERR("Scene changes:");
		NET_DEBUG_ERR(p_sync_data);
	}

	change_events_flush();
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

bool SceneSynchronizer::is_end_sync() const {
	return end_sync;
}

void SceneSynchronizer::force_state_notify() {
	ERR_FAIL_COND(is_server() == false);
	ServerSynchronizer *r = static_cast<ServerSynchronizer *>(synchronizer);
	// + 1.0 is just a ridiculous high number to be sure to avoid float
	// precision error.
	r->state_notifier_timer = get_server_notify_state_interval() + 1.0;
}

void SceneSynchronizer::dirty_peers() {
	peer_dirty = true;
}

void SceneSynchronizer::set_enabled(bool p_enable) {
	ERR_FAIL_COND_MSG(synchronizer_type == SYNCHRONIZER_TYPE_SERVER, "The server is always enabled.");
	if (synchronizer_type == SYNCHRONIZER_TYPE_CLIENT) {
		rpc_id(1, "_rpc_set_network_enabled", p_enable);
		if (p_enable == false) {
			// If the peer want to disable, we can disable it locally
			// immediately. When it wants to enable the networking, the server
			// must be notified so it decides when to start the networking
			// again.
			static_cast<ClientSynchronizer *>(synchronizer)->set_enabled(p_enable);
		}
	} else if (synchronizer_type == SYNCHRONIZER_TYPE_NONETWORK) {
		set_peer_networking_enable(0, p_enable);
	}
}

bool SceneSynchronizer::is_enabled() const {
	ERR_FAIL_COND_V_MSG(synchronizer_type == SYNCHRONIZER_TYPE_SERVER, false, "The server is always enabled.");
	if (likely(synchronizer_type == SYNCHRONIZER_TYPE_CLIENT)) {
		return static_cast<ClientSynchronizer *>(synchronizer)->enabled;
	} else if (synchronizer_type == SYNCHRONIZER_TYPE_NONETWORK) {
		return static_cast<NoNetSynchronizer *>(synchronizer)->enabled;
	} else {
		return true;
	}
}

void SceneSynchronizer::set_peer_networking_enable(int p_peer, bool p_enable) {
	if (synchronizer_type == SYNCHRONIZER_TYPE_SERVER) {
		ERR_FAIL_COND_MSG(p_peer == 1, "Disable the server is not possible.");

		NetUtility::PeerData *pd = peer_data.lookup_ptr(p_peer);
		ERR_FAIL_COND_MSG(pd == nullptr, "The peer: " + itos(p_peer) + " is not know. [bug]");

		if (pd->enabled == p_enable) {
			// Nothing to do.
			return;
		}

		pd->enabled = p_enable;
		// Set to true, so next time this peer connects a full snapshot is sent.
		pd->force_notify_snapshot = true;
		pd->need_full_snapshot = true;

		dirty_peers();

		// Just notify the peer status.
		rpc_id(p_peer, "_rpc_notify_peer_status", p_enable);
	} else {
		ERR_FAIL_COND_MSG(synchronizer_type != SYNCHRONIZER_TYPE_NONETWORK, "At this point no network is expected.");
		static_cast<NoNetSynchronizer *>(synchronizer)->set_enabled(p_enable);
	}
}

bool SceneSynchronizer::is_peer_networking_enable(int p_peer) const {

	if (synchronizer_type == SYNCHRONIZER_TYPE_SERVER) {
		if (p_peer == 1) {
			// Server is always enabled.
			return true;
		}

		NetUtility::PeerData *pd = peer_data.lookup_ptr(p_peer);
		ERR_FAIL_COND_V_MSG(pd == nullptr, false, "The peer: " + itos(p_peer) + " is not know. [bug]");
		return pd->enabled;
	} else {
		ERR_FAIL_COND_V_MSG(synchronizer_type != SYNCHRONIZER_TYPE_NONETWORK, false, "At this point no network is expected.");
		return static_cast<NoNetSynchronizer *>(synchronizer)->is_enabled();
	}
}

void SceneSynchronizer::_on_peer_connected(int p_peer) {
	peer_data.insert(p_peer, NetUtility::PeerData());
	dirty_peers();
}

void SceneSynchronizer::_on_peer_disconnected(int p_peer) {
	peer_data.remove(p_peer);

	// Notify all controllers that this peer is gone.
	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		NetworkedController *c = static_cast<NetworkedController *>(node_data_controllers[i]->node);
		c->controller->deactivate_peer(p_peer);
	}
}

void SceneSynchronizer::_on_node_removed(Node *p_node) {
	unregister_node(p_node);
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
			if (node_data[i] == nullptr) {
				continue;
			}

			// Handle the node ID.
			if (generate_id) {
				node_data[i]->id = i;
				organized_node_data[i] = node_data[i];
			} else {
				node_data[i]->id = UINT32_MAX;
				organized_node_data[i] = nullptr;
			}

			// Handle the variables ID.
			for (uint32_t v = 0; v < node_data[i]->vars.size(); v += 1) {
				if (generate_id) {
					node_data[i]->vars[v].id = v;
				} else {
					node_data[i]->vars[v].id = UINT32_MAX;
				}
			}
		}
	}

	// Notify the presence all available nodes and its variables to the synchronizer.
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		synchronizer->on_node_added(node_data[i]);
		for (uint32_t y = 0; y < node_data[i]->vars.size(); y += 1) {
			synchronizer->on_variable_added(node_data[i], node_data[i]->vars[y].var.name);
		}
	}

	// Reset the controllers.
	reset_controllers();
}

void SceneSynchronizer::clear() {
	// Drop the node_data.
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (node_data[i] != nullptr) {
			drop_node_data(node_data[i]);
		}
	}

	node_data.reset();
	organized_node_data.reset();
	node_data_controllers.reset();
	event_listener.reset();

	// Avoid too much useless re-allocations.
	event_listener.reserve(100);

	if (synchronizer) {
		synchronizer->clear();
	}
}

void SceneSynchronizer::_rpc_send_state(Variant p_snapshot) {
	ERR_FAIL_COND_MSG(is_client() == false, "Only clients are suposed to receive the server snapshot.");
	static_cast<ClientSynchronizer *>(synchronizer)->receive_snapshot(p_snapshot);
}

void SceneSynchronizer::_rpc_notify_need_full_snapshot() {
	ERR_FAIL_COND_MSG(is_server() == false, "Only the server can receive the request to send a full snapshot.");

	const int sender_peer = get_tree()->get_multiplayer()->get_rpc_sender_id();
	NetUtility::PeerData *pd = peer_data.lookup_ptr(sender_peer);
	ERR_FAIL_COND(pd == nullptr);
	pd->need_full_snapshot = true;
}

void SceneSynchronizer::_rpc_set_network_enabled(bool p_enabled) {
	ERR_FAIL_COND_MSG(is_server() == false, "The peer status is supposed to be received by the server.");
	set_peer_networking_enable(
			get_multiplayer()->get_rpc_sender_id(),
			p_enabled);
}

void SceneSynchronizer::_rpc_notify_peer_status(bool p_enabled) {
	ERR_FAIL_COND_MSG(is_client() == false, "The peer status is supposed to be received by the client.");
	static_cast<ClientSynchronizer *>(synchronizer)->set_enabled(p_enabled);
}

void SceneSynchronizer::update_peers() {
#ifdef DEBUG_ENABLED
	// This function is only called on server.
	CRASH_COND(synchronizer_type != SYNCHRONIZER_TYPE_SERVER);
#endif

	if (likely(peer_dirty == false)) {
		return;
	}

	peer_dirty = false;

	for (OAHashMap<int, NetUtility::PeerData>::Iterator it = peer_data.iter();
			it.valid;
			it = peer_data.next_iter(it)) {

		// Validate the peer.
		if (it.value->controller_id != UINT32_MAX) {
			NetUtility::NodeData *nd = get_node_data(it.value->controller_id);
			if (nd == nullptr ||
					nd->is_controller == false ||
					nd->node->get_network_master() != (*it.key)) {
				// Invalidate the controller id
				it.value->controller_id = UINT32_MAX;
			}
		}

		if (it.value->controller_id == UINT32_MAX) {
			// The controller_id is not assigned, search it.
			for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
				if (node_data_controllers[i]->node->get_network_master() == (*it.key)) {
					// Controller found.
					it.value->controller_id = node_data_controllers[i]->id;
					break;
				}
			}
		}

		// Propagate the peer change to controllers.
		for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
			NetworkedController *c = static_cast<NetworkedController *>(node_data_controllers[i]->node);

			if (it.value->controller_id == node_data_controllers[i]->id) {
				// This is the controller owned by this peer.
				c->get_server_controller()->set_enabled(it.value->enabled);
			} else {
				// This is a controller owned by another peer.
				if (it.value->enabled) {
					c->controller->activate_peer(*it.key);
				} else {
					c->controller->deactivate_peer(*it.key);
				}
			}
		}
	}
}

void SceneSynchronizer::clear_peers() {
	peer_data.clear();
	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		NetworkedController *c = static_cast<NetworkedController *>(node_data_controllers[i]->node);
		c->controller->clear_peers();
	}
}

void SceneSynchronizer::change_events_begin(int p_flag) {
#ifdef DEBUG_ENABLED
	// This can't happen because at the end these are reset.
	CRASH_COND(recover_in_progress);
	CRASH_COND(reset_in_progress);
	CRASH_COND(rewinding_in_progress);
	CRASH_COND(end_sync);
#endif
	event_flag = p_flag;
	recover_in_progress = NetEventFlag::SYNC & p_flag;
	reset_in_progress = NetEventFlag::SYNC_RESET & p_flag;
	rewinding_in_progress = NetEventFlag::SYNC_REWIND & p_flag;
	end_sync = NetEventFlag::END_SYNC & p_flag;
}

void SceneSynchronizer::change_event_add(NetUtility::NodeData *p_node_data, NetVarId p_var_id, const Variant &p_old) {
	for (int i = 0; i < p_node_data->vars[p_var_id].change_listeners.size(); i += 1) {
		const uint32_t listener_index = p_node_data->vars[p_var_id].change_listeners[i];
		NetUtility::ChangeListener &listener = event_listener[listener_index];
		if ((listener.flag & event_flag) == 0) {
			// Not listening to this event.
			continue;
		}

		listener.emitted = false;

		NetUtility::NodeChangeListener ncl;
		ncl.node_data = p_node_data;
		ncl.var_id = p_var_id;

		const int64_t index = listener.watching_vars.find(ncl);
#ifdef DEBUG_ENABLED
		// This can't never happen because the `NodeData::change_listeners`
		// tracks the correct listener.
		CRASH_COND(index == -1);
#endif
		listener.watching_vars[index].old_value = p_old;
		listener.watching_vars[index].old_set = true;
	}

	// Notify the synchronizer.
	if (synchronizer) {
		synchronizer->on_variable_changed(
				p_node_data,
				p_var_id,
				p_old,
				event_flag);
	}
}

void SceneSynchronizer::change_events_flush() {
	LocalVector<Variant> vars;
	LocalVector<const Variant *> vars_ptr;

	// TODO this can be optimized by storing the changed listener in a separate
	// vector. This change must be inserted into the `change_event_add`.
	for (uint32_t listener_i = 0; listener_i < event_listener.size(); listener_i += 1) {
		NetUtility::ChangeListener &listener = event_listener[listener_i];
		if (listener.emitted) {
			continue;
		}
		listener.emitted = true;

		Object *obj = ObjectDB::get_instance(listener.object_id);
		if (obj == nullptr) {
			// Setting the flag to 0 so no events trigger this anymore.
			listener.flag = NetEventFlag::EMPTY;
			listener.object_id = ObjectID();
			listener.method = StringName();

			// Make sure this listener is not tracking any variable.
			for (uint32_t wv = 0; wv < listener.watching_vars.size(); wv += 1) {
				NetUtility::NodeData *nd = listener.watching_vars[wv].node_data;
				uint32_t var_id = listener.watching_vars[wv].var_id;
				nd->vars[var_id].change_listeners.erase(listener_i);
			}
			listener.watching_vars.clear();
			continue;
		}

		// Initialize the arguments
		ERR_CONTINUE_MSG(listener.method_argument_count > listener.watching_vars.size(), "This method " + listener.method + " has more arguments than the watched variables. This listener is broken.");

		vars.resize(MIN(listener.watching_vars.size(), listener.method_argument_count));
		vars_ptr.resize(vars.size());
		for (uint32_t v = 0; v < MIN(listener.watching_vars.size(), listener.method_argument_count); v += 1) {
			if (listener.watching_vars[v].old_set) {
				vars[v] = listener.watching_vars[v].old_value;
				listener.watching_vars[v].old_set = false;
			} else {
				// This value is not changed, so just retrive the current one.
				vars[v] = listener.watching_vars[v].node_data->vars[listener.watching_vars[v].var_id].var.value;
			}
			vars_ptr[v] = vars.ptr() + v;
		}

		Callable::CallError e;
		obj->call(listener.method, vars_ptr.ptr(), vars_ptr.size(), e);
	}

	recover_in_progress = false;
	reset_in_progress = false;
	rewinding_in_progress = false;
	end_sync = false;
}

void SceneSynchronizer::add_node_data(NetUtility::NodeData *p_node_data) {
	if (generate_id) {
#ifdef DEBUG_ENABLED
		// When generate_id is true, the id must always be undefined.
		CRASH_COND(p_node_data->id != UINT32_MAX);
#endif
		p_node_data->id = organized_node_data.size();
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

	if (generate_id) {
		organized_node_data.push_back(p_node_data);
	} else {
		if (p_node_data->id != UINT32_MAX) {
			// This node has an ID, make sure to organize it properly.

			if (organized_node_data.size() <= p_node_data->id) {
				expand_organized_node_data_vector((p_node_data->id + 1) - organized_node_data.size());
			}

			organized_node_data[p_node_data->id] = p_node_data;
		}
	}

	if (p_node_data->is_controller) {
		node_data_controllers.push_back(p_node_data);
		reset_controller(p_node_data);
	}

	if (synchronizer) {
		synchronizer->on_node_added(p_node_data);
	}
}

void SceneSynchronizer::drop_node_data(NetUtility::NodeData *p_node_data) {
	if (synchronizer) {
		synchronizer->on_node_removed(p_node_data);
	}

	if (p_node_data->controlled_by) {
		// This node is controlled by another one, remove from that node.
		p_node_data->controlled_by->controlled_nodes.erase(p_node_data);
		p_node_data->controlled_by = nullptr;
	}

	if (p_node_data->is_controller) {
		// This is a controller, make sure to reset the peers.
		static_cast<NetworkedController *>(p_node_data->node)->set_scene_synchronizer(nullptr);
		dirty_peers();
		node_data_controllers.erase(p_node_data);
	}

	node_data.erase(p_node_data);

	if (p_node_data->id < organized_node_data.size()) {
		// Never resize this vector to keep it sort.
		organized_node_data[p_node_data->id] = nullptr;
	}

	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		const int64_t index = node_data_controllers[i]->dependency_nodes.find(p_node_data);
		if (index != -1) {
			node_data_controllers[i]->dependency_nodes.remove_unordered(index);
			node_data_controllers[i]->dependency_nodes_end.remove_unordered(index);
		}
	}

	// Remove this `NodeData` from any event listener.
	for (uint32_t i = 0; i < event_listener.size(); i += 1) {
		while (true) {
			uint32_t index_to_remove = UINT32_MAX;

			// Search.
			for (uint32_t v = 0; v < event_listener[i].watching_vars.size(); v += 1) {
				if (event_listener[i].watching_vars[v].node_data == p_node_data) {
					index_to_remove = v;
					break;
				}
			}

			if (index_to_remove == UINT32_MAX) {
				// Nothing more to do.
				break;
			} else {
				event_listener[i].watching_vars.remove_unordered(index_to_remove);
			}
		}
	}

	memdelete(p_node_data);
}

void SceneSynchronizer::set_node_data_id(NetUtility::NodeData *p_node_data, NetNodeId p_id) {
#ifdef DEBUG_ENABLED
	CRASH_COND_MSG(generate_id, "This function is not supposed to be called, because this instance is generating the IDs");
#endif
	if (organized_node_data.size() <= p_id) {
		expand_organized_node_data_vector((p_id + 1) - organized_node_data.size());
	}
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

bool SceneSynchronizer::is_server() const {
	return synchronizer_type == SYNCHRONIZER_TYPE_SERVER;
}

bool SceneSynchronizer::is_client() const {
	return synchronizer_type == SYNCHRONIZER_TYPE_CLIENT;
}

bool SceneSynchronizer::is_no_network() const {
	return synchronizer_type == SYNCHRONIZER_TYPE_NONETWORK;
}

bool SceneSynchronizer::is_networked() const {
	return is_client() || is_server();
}

#ifdef DEBUG_ENABLED
void SceneSynchronizer::validate_nodes() {
	LocalVector<NetUtility::NodeData *> null_objects;
	null_objects.reserve(node_data.size());

	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (ObjectDB::get_instance(node_data[i]->instance_id) == nullptr) {
			// Mark for removal.
			null_objects.push_back(node_data[i]);
		}
	}

	// Removes the invalidated `NodeData`.
	if (null_objects.size()) {
		NET_DEBUG_ERR("At least one node has been removed from the tree without the SceneSynchronizer noticing. This shouldn't happen.");
		for (uint32_t i = 0; i < null_objects.size(); i += 1) {
			drop_node_data(null_objects[i]);
		}
	}
}
#endif

void SceneSynchronizer::purge_node_dependencies() {
	if (is_client() == false) {
		return;
	}

	// Clear the controller dependencies.
	ClientSynchronizer *client_sync = static_cast<ClientSynchronizer *>(synchronizer);

	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		for (
				int d = 0;
				d < int(node_data_controllers[i]->dependency_nodes_end.size());
				d += 1) {
			if (node_data_controllers[i]->dependency_nodes_end[d] < client_sync->last_checked_input) {
				// This controller dependency can be cleared because the server
				// snapshot check has
				node_data_controllers[i]->dependency_nodes.remove_unordered(d);
				node_data_controllers[i]->dependency_nodes_end.remove_unordered(d);
				d -= 1;
			}
		}
	}
}

void SceneSynchronizer::expand_organized_node_data_vector(uint32_t p_size) {
	const uint32_t from = organized_node_data.size();
	organized_node_data.resize(from + p_size);
	memset(organized_node_data.ptr() + from, 0, sizeof(void *) * p_size);
}

NetUtility::NodeData *SceneSynchronizer::find_node_data(const Node *p_node) {
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (node_data[i] == nullptr) {
			continue;
		}
		if (node_data[i]->instance_id == p_node->get_instance_id()) {
			return node_data[i];
		}
	}
	return nullptr;
}

const NetUtility::NodeData *SceneSynchronizer::find_node_data(const Node *p_node) const {
	for (uint32_t i = 0; i < node_data.size(); i += 1) {
		if (node_data[i] == nullptr) {
			continue;
		}
		if (node_data[i]->instance_id == p_node->get_instance_id()) {
			return node_data[i];
		}
	}
	return nullptr;
}

NetUtility::NodeData *SceneSynchronizer::get_node_data(NetNodeId p_id) {
	ERR_FAIL_INDEX_V(p_id, organized_node_data.size(), nullptr);
	return organized_node_data[p_id];
}

const NetUtility::NodeData *SceneSynchronizer::get_node_data(NetNodeId p_id) const {
	ERR_FAIL_INDEX_V(p_id, organized_node_data.size(), nullptr);
	return organized_node_data[p_id];
}

NetNodeId SceneSynchronizer::get_biggest_node_id() const {
	return organized_node_data.size() == 0 ? UINT32_MAX : organized_node_data.size() - 1;
}

void SceneSynchronizer::reset_controllers() {
	for (uint32_t i = 0; i < node_data_controllers.size(); i += 1) {
		reset_controller(node_data_controllers[i]);
	}
}

void SceneSynchronizer::reset_controller(NetUtility::NodeData *p_controller_nd) {
#ifdef DEBUG_ENABLED
	// This can't happen because the callers make sure the `NodeData` is a
	// controller.
	CRASH_COND(p_controller_nd->is_controller == false);
#endif

	NetworkedController *controller = static_cast<NetworkedController *>(p_controller_nd->node);

	// Reset the controller type.
	if (controller->controller != nullptr) {
		memdelete(controller->controller);
		controller->controller = nullptr;
		controller->controller_type = NetworkedController::CONTROLLER_TYPE_NULL;
		controller->set_physics_process_internal(false);
	}

	if (get_tree() == nullptr) {
		if (synchronizer) {
			synchronizer->on_controller_reset(p_controller_nd);
		}

		// Nothing to do.
		return;
	}

	if (get_tree()->get_network_peer().is_null()) {
		controller->controller_type = NetworkedController::CONTROLLER_TYPE_NONETWORK;
		controller->controller = memnew(NoNetController(controller));
	} else if (get_tree()->is_network_server()) {
		controller->controller_type = NetworkedController::CONTROLLER_TYPE_SERVER;
		controller->controller = memnew(ServerController(controller, controller->get_network_traced_frames()));
	} else if (controller->is_network_master()) {
		controller->controller_type = NetworkedController::CONTROLLER_TYPE_PLAYER;
		controller->controller = memnew(PlayerController(controller));
	} else {
		controller->controller_type = NetworkedController::CONTROLLER_TYPE_DOLL;
		controller->controller = memnew(DollController(controller));
		controller->set_physics_process_internal(true);
	}

	dirty_peers();
	controller->controller->ready();

	if (synchronizer) {
		synchronizer->on_controller_reset(p_controller_nd);
	}
}

void SceneSynchronizer::process() {
#ifdef DEBUG_ENABLED
	validate_nodes();
	// Never triggered because this function is called by `PHYSICS_PROCESS`,
	// notification that is emitted only when the node is in the tree.
	// When the node is in the tree, there is no way that the `synchronizer` is
	// null.
	CRASH_COND(synchronizer == nullptr);
#endif

	synchronizer->process();
	purge_node_dependencies();
}

void SceneSynchronizer::pull_node_changes(NetUtility::NodeData *p_node_data) {
	Node *node = p_node_data->node;

	for (NetVarId var_id = 0; var_id < p_node_data->vars.size(); var_id += 1) {
		if (p_node_data->vars[var_id].enabled == false) {
			continue;
		}

		const Variant old_val = p_node_data->vars[var_id].var.value;
		const Variant new_val = node->get(p_node_data->vars[var_id].var.name);

		if (!synchronizer_variant_evaluation(old_val, new_val)) {
			p_node_data->vars[var_id].var.value = new_val.duplicate(true);
			change_event_add(
					p_node_data,
					var_id,
					old_val);
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
	enabled = true;
}

void NoNetSynchronizer::process() {
	if (unlikely(enabled == false)) {
		return;
	}

	const real_t delta = scene_synchronizer->get_physics_process_delta_time();

	// Process the scene
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i];
		nd->process(delta);
	}

	// Process the controllers_node_data
	for (uint32_t i = 0; i < scene_synchronizer->node_data_controllers.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data_controllers[i];
		static_cast<NetworkedController *>(nd->node)->get_nonet_controller()->process(delta);
	}

	// Pull the changes.
	scene_synchronizer->change_events_begin(NetEventFlag::CHANGE);
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i];
		scene_synchronizer->pull_node_changes(nd);
	}
	scene_synchronizer->change_events_flush();
}

void NoNetSynchronizer::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		// Nothing to do.
		return;
	}

	enabled = p_enabled;

	if (enabled) {
		scene_synchronizer->emit_signal("sync_started");
	} else {
		scene_synchronizer->emit_signal("sync_paused");
	}
}

bool NoNetSynchronizer::is_enabled() const {
	return enabled;
}

ServerSynchronizer::ServerSynchronizer(SceneSynchronizer *p_node) :
		Synchronizer(p_node) {}

void ServerSynchronizer::clear() {
	state_notifier_timer = 0.0;
	// Release the internal memory.
	changes.reset();
}

void ServerSynchronizer::process() {
	scene_synchronizer->update_peers();

	const real_t delta = scene_synchronizer->get_physics_process_delta_time();

	// Process the scene
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i];
		nd->process(delta);
	}

	// Process the controllers_node_data
	for (uint32_t i = 0; i < scene_synchronizer->node_data_controllers.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data_controllers[i];
		static_cast<NetworkedController *>(nd->node)->get_server_controller()->process(delta);
	}

	// Pull the changes.
	scene_synchronizer->change_events_begin(NetEventFlag::CHANGE);
	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		NetUtility::NodeData *nd = scene_synchronizer->node_data[i];
		scene_synchronizer->pull_node_changes(nd);
	}
	scene_synchronizer->change_events_flush();

	process_snapshot_notificator(delta);
}

void ServerSynchronizer::on_node_added(NetUtility::NodeData *p_node_data) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_synchronizer->is_recovered());
	// On server the ID is always known.
	CRASH_COND(p_node_data->id == UINT32_MAX);
#endif

	if (changes.size() <= p_node_data->id) {
		changes.resize(p_node_data->id + 1);
	}

	changes[p_node_data->id].not_known_before = true;
}

void ServerSynchronizer::on_variable_added(NetUtility::NodeData *p_node_data, StringName p_var_name) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_synchronizer->is_recovered());
	// On server the ID is always known.
	CRASH_COND(p_node_data->id == UINT32_MAX);
#endif

	if (changes.size() <= p_node_data->id) {
		changes.resize(p_node_data->id + 1);
	}

	changes[p_node_data->id].vars.insert(p_var_name);
	changes[p_node_data->id].uknown_vars.insert(p_var_name);
}

void ServerSynchronizer::on_variable_changed(NetUtility::NodeData *p_node_data, NetVarId p_var_id, Variant p_old_value, int p_flag) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_synchronizer->is_recovered());
	// On server the ID is always known.
	CRASH_COND(p_node_data->id == UINT32_MAX);
#endif

	if (changes.size() <= p_node_data->id) {
		changes.resize(p_node_data->id + 1);
	}

	changes[p_node_data->id].vars.insert(p_node_data->vars[p_var_id].var.name);
}

void ServerSynchronizer::process_snapshot_notificator(real_t p_delta) {
	if (scene_synchronizer->peer_data.is_empty()) {
		// No one is listening.
		return;
	}

	// Notify the state if needed
	state_notifier_timer += p_delta;
	const bool notify_state = state_notifier_timer >= scene_synchronizer->get_server_notify_state_interval();

	if (notify_state) {
		state_notifier_timer = 0.0;
	}

	Vector<Variant> full_global_nodes_snapshot;
	Vector<Variant> delta_global_nodes_snapshot;
	for (
			OAHashMap<int, NetUtility::PeerData>::Iterator peer_it = scene_synchronizer->peer_data.iter();
			peer_it.valid;
			peer_it = scene_synchronizer->peer_data.next_iter(peer_it)) {
		if (unlikely(peer_it.value->controller_id == UINT32_MAX)) {
			// This peer still does not have a `NetworkedController`.
			continue;
		}
		if (unlikely(peer_it.value->enabled == false)) {
			// This peer is disabled.
			continue;
		}
		if (peer_it.value->force_notify_snapshot == false && notify_state == false) {
			// Nothing to do.
			continue;
		}

		peer_it.value->force_notify_snapshot = false;

		NetUtility::NodeData *nd = scene_synchronizer->get_node_data(peer_it.value->controller_id);
		// TODO well that's not really true. I may have peers that doesn't have controllers_node_data in a
		// certain moment. Please improve this mechanism trying to just use the
		// node->get_network_master() to get the peer.
		ERR_CONTINUE_MSG(nd == nullptr, "This should never happen. Likely there is a bug, NedNodeId: " + itos(peer_it.value->controller_id));
		ERR_CONTINUE_MSG(nd->is_controller == false, "[BUG] A controller il expected, The node " + nd->node->get_path() + " is submitted instead.");

		NetworkedController *controller = static_cast<NetworkedController *>(nd->node);

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

	for (uint32_t i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
		const NetUtility::NodeData *node_data = scene_synchronizer->node_data[i];
		if (node_data == nullptr) {
			continue;
		}
		if (node_data->is_controller || node_data->controlled_by != nullptr) {
			// Skip the controllers.
			continue;
		}
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

	const Change *change = p_node_data->id >= changes.size() ? nullptr : changes.ptr() + p_node_data->id;

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

	const bool node_has_changes = p_force_full_snapshot || (change != nullptr && change->vars.is_empty() == false);

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
		for (uint32_t i = 0; i < p_node_data->vars.size(); i += 1) {
			const NetUtility::VarData &var = p_node_data->vars[i];
			if (var.enabled == false) {
				continue;
			}

			if (p_force_full_snapshot == false && change->vars.has(var.var.name) == false) {
				// This is a delta snapshot and this variable is the same as
				// before. Skip it.
				continue;
			}

			Variant var_info;
			if (p_force_full_snapshot || change->uknown_vars.has(var.var.name)) {
				Vector<Variant> _var_info;
				_var_info.resize(2);
				_var_info.write[0] = var.id;
				_var_info.write[1] = var.var.name;
				var_info = _var_info;
			} else {
				var_info = var.id;
			}

			r_snapshot_data.push_back(var_info);
			r_snapshot_data.push_back(var.var.value);
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
	player_controller_node_data = nullptr;
	node_paths.clear();
	last_received_snapshot.input_id = UINT32_MAX;
	last_received_snapshot.node_vars.clear();
	client_snapshots.clear();
	server_snapshots.clear();
	last_checked_input = 0;
	enabled = true;
	need_full_snapshot_notified = false;
}

void ClientSynchronizer::process() {
	if (unlikely(player_controller_node_data == nullptr || enabled == false)) {
		// No player controller or disabled so nothing to do.
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
			NetUtility::NodeData *nd = scene_synchronizer->node_data[i];
			nd->process(delta);
		}

		// Process the player controllers_node_data.
		player_controller->process(delta);

		// Pull the changes.
		scene_synchronizer->change_events_begin(NetEventFlag::CHANGE);
		for (NetNodeId i = 0; i < scene_synchronizer->node_data.size(); i += 1) {
			NetUtility::NodeData *nd = scene_synchronizer->node_data[i];
			scene_synchronizer->pull_node_changes(nd);
		}
		scene_synchronizer->change_events_flush();

		if (controller->player_has_new_input()) {
			store_snapshot();
		}

		sub_ticks -= 1;
	}

	process_controllers_recovery(delta);

	// Now trigger the END_SYNC event.
	scene_synchronizer->change_events_begin(NetEventFlag::END_SYNC);
	for (const Set<EndSyncEvent>::Element *e = sync_end_events.front();
			e != nullptr;
			e = e->next()) {
		// Check if the values between the variables before the sync and the
		// current one are different.
		if (scene_synchronizer->synchronizer_variant_evaluation(
					e->get().node_data->vars[e->get().var_id].var.value,
					e->get().old_value) == false) {
			// Are different so we need to emit the `END_SYNC`.
			scene_synchronizer->change_event_add(
					e->get().node_data,
					e->get().var_id,
					e->get().old_value);
		}
	}
	sync_end_events.clear();

	scene_synchronizer->change_events_flush();
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
}

void ClientSynchronizer::on_node_removed(NetUtility::NodeData *p_node_data) {
	if (player_controller_node_data == p_node_data) {
		player_controller_node_data = nullptr;
		client_snapshots.clear();
	}
}

void ClientSynchronizer::on_variable_changed(NetUtility::NodeData *p_node_data, NetVarId p_var_id, Variant p_old_value, int p_flag) {
	if (p_flag & NetEventFlag::SYNC) {
		sync_end_events.insert(
				EndSyncEvent{
						p_node_data,
						p_var_id,
						p_old_value });
	}
}

void ClientSynchronizer::on_controller_reset(NetUtility::NodeData *p_node_data) {
#ifdef DEBUG_ENABLED
	CRASH_COND(p_node_data->is_controller == false);
#endif

	if (player_controller_node_data == p_node_data) {
		// Reset the node_data.
		player_controller_node_data = nullptr;
		client_snapshots.clear();
	}

	if (player_controller_node_data != nullptr) {
		NET_DEBUG_ERR("Only one player controller is supported, at the moment. Make sure this is the case.");
	}

	if (static_cast<NetworkedController *>(p_node_data->node)->is_player_controller()) {
		// Set this player controller as active.
		player_controller_node_data = p_node_data;
		client_snapshots.clear();
	}
}

void ClientSynchronizer::store_snapshot() {
	NetworkedController *controller = static_cast<NetworkedController *>(player_controller_node_data->node);

#ifdef DEBUG_ENABLED
	if (unlikely(client_snapshots.size() > 0 && controller->get_current_input_id() <= client_snapshots.back().input_id)) {
		CRASH_NOW_MSG("[FATAL] During snapshot creation, for controller " + controller->get_path() + ", was found an ID for an older snapshots. New input ID: " + itos(controller->get_current_input_id()) + " Last saved snapshot input ID: " + itos(client_snapshots.back().input_id) + ".");
	}
#endif

	client_snapshots.push_back(NetUtility::Snapshot());

	NetUtility::Snapshot &snap = client_snapshots.back();
	snap.input_id = controller->get_current_input_id();

	snap.node_vars.resize(scene_synchronizer->organized_node_data.size());

	// Store the nodes state and skip anything is related to the other
	// controllers.
	for (uint32_t i = 0; i < scene_synchronizer->organized_node_data.size(); i += 1) {
		const NetUtility::NodeData *node_data = scene_synchronizer->organized_node_data[i];

		if (node_data == nullptr) {
			// Nothing to do.
			continue;
		}

		if ((node_data->is_controller || node_data->controlled_by != nullptr) &&
				(node_data != player_controller_node_data && node_data->controlled_by != player_controller_node_data)) {
			// Ignore this controller.
			continue;
		}

		if (node_data->id >= uint32_t(snap.node_vars.size())) {
			// Make sure this ID is valid.
			ERR_FAIL_COND_MSG(node_data->id != UINT32_MAX, "[BUG] It's not expected that the client has a node with the NetNodeId (" + itos(node_data->id) + ") bigger than the registered node count: " + itos(snap.node_vars.size()));
			// Skip this node
			continue;
		}

		Vector<NetUtility::Var> *snap_node_vars = snap.node_vars.ptrw() + node_data->id;
		snap_node_vars->resize(node_data->vars.size());
		NetUtility::Var *vars = snap_node_vars->ptrw();
		for (uint32_t v = 0; v < node_data->vars.size(); v += 1) {
			if (node_data->vars[v].enabled) {
				vars[v] = node_data->vars[v].var;
			} else {
				vars[v].name = StringName();
			}
		}
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

// TODO make this function much simpler.
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
		CRASH_COND_MSG(client_snapshots.back().input_id != player_controller->last_known_input(), "This should not be possible: snapshot input: " + itos(client_snapshots.back().input_id) + " last_know_input: " + itos(player_controller->last_known_input()));
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
	for (uint32_t net_node_id = 0; net_node_id < uint32_t(server_snapshots.front().node_vars.size()); net_node_id += 1) {
		NetUtility::NodeData *rew_node_data = scene_synchronizer->get_node_data(net_node_id);
		if (rew_node_data == nullptr || rew_node_data->sync_enabled == false) {
			continue;
		}

		bool recover_this_node = false;
		if (net_node_id >= uint32_t(client_snapshots.front().node_vars.size())) {
			NET_DEBUG_PRINT("Rewind is needed because the client snapshot doesn't contain this node: " + rew_node_data->node->get_path());
			recover_this_node = true;
		} else {
			NetUtility::PostponedRecover rec;

			const bool different = compare_vars(
					rew_node_data,
					server_snapshots.front().node_vars[net_node_id],
					client_snapshots.front().node_vars[net_node_id],
					rec.vars);

			if (different) {
				NET_DEBUG_PRINT("Rewind is needed because the node on client is different: " + rew_node_data->node->get_path());
				recover_this_node = true;
			} else if (rec.vars.size() > 0) {
				rec.node_data = rew_node_data;
				postponed_recover.push_back(rec);
			}
		}

		if (recover_this_node) {
			need_recover = true;
			if (rew_node_data->controlled_by != nullptr ||
					rew_node_data->is_controller ||
					player_controller_node_data->dependency_nodes.find(rew_node_data) != -1) {
				// Controller node.
				recover_controller = true;
			} else {
				nodes_to_recover.push_back(rew_node_data);
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
					player_controller_node_data->dependency_nodes.size() +
					1);

			nodes_to_recover.push_back(player_controller_node_data);

			for (
					uint32_t y = 0;
					y < player_controller_node_data->controlled_nodes.size();
					y += 1) {
				nodes_to_recover.push_back(player_controller_node_data->controlled_nodes[y]);
			}

			for (
					uint32_t y = 0;
					y < player_controller_node_data->dependency_nodes.size();
					y += 1) {
				nodes_to_recover.push_back(player_controller_node_data->dependency_nodes[y]);
			}
		}

		// Apply the server snapshot so to go back in time till that moment,
		// so to be able to correctly reply the movements.
		scene_synchronizer->change_events_begin(NetEventFlag::SYNC_RECOVER | NetEventFlag::SYNC_RESET);
		for (uint32_t i = 0; i < nodes_to_recover.size(); i += 1) {
			if (nodes_to_recover[i]->id >= uint32_t(server_snapshots.front().node_vars.size())) {
				NET_DEBUG_WARN("The node: " + nodes_to_recover[i]->node->get_path() + " was not found on the server snapshot, this is not supposed to happen a lot.");
				continue;
			}
			if (nodes_to_recover[i]->sync_enabled == false) {
				// Don't sync this node.
				// This check is also here, because the `recover_controller`
				// mechanism, may have insert a no sync node.
				// The check is here because I feel it more clear, here.
				continue;
			}

#ifdef DEBUG_ENABLED
			// The parser make sure to properly initialize the snapshot variable
			// array size. So the following condition is always `false`.
			CRASH_COND(uint32_t(server_snapshots.front().node_vars[nodes_to_recover[i]->id].size()) != nodes_to_recover[i]->vars.size());
#endif

			Node *node = nodes_to_recover[i]->node;
			const Vector<NetUtility::Var> s_vars = server_snapshots.front().node_vars[nodes_to_recover[i]->id];
			const NetUtility::Var *s_vars_ptr = s_vars.ptr();

			NET_DEBUG_PRINT("Full reset node: " + node->get_path());

			for (int v = 0; v < s_vars.size(); v += 1) {
				if (s_vars_ptr[v].name == StringName()) {
					// This variable was not set, skip it.
					continue;
				}

				const Variant current_val = nodes_to_recover[i]->vars[v].var.value;
				nodes_to_recover[i]->vars[v].var.value = s_vars_ptr[v].value.duplicate(true);
				node->set(s_vars_ptr[v].name, s_vars_ptr[v].value);

				NET_DEBUG_PRINT(" |- Variable: " + s_vars_ptr[v].name + " New value: " + s_vars_ptr[v].value);
				scene_synchronizer->change_event_add(
						nodes_to_recover[i],
						v,
						current_val);
			}
		}
		scene_synchronizer->change_events_flush();

		// Rewind phase.

		const int remaining_inputs = player_controller->notify_input_checked(checkable_input_id);
#ifdef DEBUG_ENABLED
		// Unreachable because the SceneSynchronizer and the PlayerController
		// have the same stored data at this point.
		CRASH_COND(client_snapshots.size() != size_t(remaining_inputs));
#endif

#ifdef DEBUG_ENABLED
		// Used to double check all the instants have been processed.
		bool has_next = false;
#endif
		for (int i = 0; i < remaining_inputs; i += 1) {
			scene_synchronizer->change_events_begin(NetEventFlag::SYNC_RECOVER | NetEventFlag::SYNC_REWIND);

			// Step 1 -- Process the scene nodes.
			for (uint32_t r = 0; r < nodes_to_recover.size(); r += 1) {
				if (nodes_to_recover[r]->sync_enabled == false) {
					// This node is not sync.
					continue;
				}
				nodes_to_recover[r]->process(p_delta);
#ifdef DEBUG_ENABLED
				if (nodes_to_recover[r]->functions.size()) {
					NET_DEBUG_PRINT("Rewind, processed node: " + nodes_to_recover[r]->node->get_path());
				}
#endif
			}

			// Step 2 -- Process the controller.
			if (recover_controller && player_controller_node_data->sync_enabled) {
#ifdef DEBUG_ENABLED
				has_next =
#endif
						controller->process_instant(i, p_delta);
				NET_DEBUG_PRINT("Rewind, processed controller: " + controller->get_path());
			}

			// Step 3 -- Pull node changes and Update snapshots.
			for (uint32_t r = 0; r < nodes_to_recover.size(); r += 1) {
				if (nodes_to_recover[r]->sync_enabled == false) {
					// This node is not sync.
					continue;
				}
				// Pull changes
				scene_synchronizer->pull_node_changes(nodes_to_recover[r]);

				// Update client snapshot.
				if (uint32_t(client_snapshots[i].node_vars.size()) <= nodes_to_recover[r]->id) {
					client_snapshots[i].node_vars.resize(nodes_to_recover[r]->id + 1);
				}

				Vector<NetUtility::Var> *snap_node_vars = client_snapshots[i].node_vars.ptrw() + nodes_to_recover[r]->id;
				snap_node_vars->resize(nodes_to_recover[r]->vars.size());

				NetUtility::Var *vars = snap_node_vars->ptrw();
				for (uint32_t v = 0; v < nodes_to_recover[r]->vars.size(); v += 1) {
					vars[v] = nodes_to_recover[r]->vars[v].var;
				}
			}
			scene_synchronizer->change_events_flush();
		}

#ifdef DEBUG_ENABLED
		// Unreachable because the above loop consume all instants.
		CRASH_COND(has_next);
#endif

	} else {
		// Apply found differences without rewind.
		scene_synchronizer->change_events_begin(NetEventFlag::SYNC_RECOVER);
		for (uint32_t i = 0; i < postponed_recover.size(); i += 1) {
			NetUtility::NodeData *rew_node_data = postponed_recover[i].node_data;
			if (rew_node_data->sync_enabled == false) {
				// This node sync is disabled.
				continue;
			}

			Node *node = rew_node_data->node;
			const NetUtility::Var *vars_ptr = postponed_recover[i].vars.ptr();

			NET_DEBUG_PRINT("[Snapshot partial reset] Node: " + node->get_path());

			// Set the value on the synchronizer too.
			for (int v = 0; v < postponed_recover[i].vars.size(); v += 1) {
				// We need to search it because the postponed recovered is not
				// aligned.
				// TODO This array is generated few lines above.
				// Can we store the ID too, so to avoid this search????
				const int rew_var_index = rew_node_data->vars.find(vars_ptr[v].name);
				// Unreachable, because when the snapshot is received the
				// algorithm make sure the `scene_synchronizer` is traking the
				// variable.
				CRASH_COND(rew_var_index <= -1);

				const Variant old_val = rew_node_data->vars[rew_var_index].var.value;
				rew_node_data->vars[rew_var_index].var.value = vars_ptr[v].value.duplicate(true);
				node->set(vars_ptr[v].name, vars_ptr[v].value);

				NET_DEBUG_PRINT(" |- Variable: " + vars_ptr[v].name + "; old value: " + old_val + " new value: " + vars_ptr[v].value);
				scene_synchronizer->change_event_add(
						rew_node_data,
						rew_var_index,
						old_val);
			}

			// Update the last client snapshot.
			if (client_snapshots.empty() == false) {
				if (uint32_t(client_snapshots.back().node_vars.size()) <= rew_node_data->id) {
					client_snapshots.back().node_vars.resize(rew_node_data->id + 1);
				}

				Vector<NetUtility::Var> *snap_node_vars = client_snapshots.back().node_vars.ptrw() + rew_node_data->id;
				snap_node_vars->resize(rew_node_data->vars.size());

				NetUtility::Var *vars = snap_node_vars->ptrw();

				for (uint32_t v = 0; v < rew_node_data->vars.size(); v += 1) {
					vars[v] = rew_node_data->vars[v].var;
				}
			}
		}
		scene_synchronizer->change_events_flush();

		player_controller->notify_input_checked(checkable_input_id);
	}

	// Popout the server snapshot.
	server_snapshots.pop_front();

	last_checked_input = checkable_input_id;
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
	scene_synchronizer->change_events_begin(NetEventFlag::SYNC_RECOVER);
	for (uint32_t net_node_id = 0; net_node_id < uint32_t(server_snapshots.front().node_vars.size()); net_node_id += 1) {
		NetUtility::NodeData *rew_node_data = scene_synchronizer->get_node_data(net_node_id);
		if (rew_node_data == nullptr || rew_node_data->sync_enabled == false) {
			continue;
		}

		Node *node = rew_node_data->node;

		const NetUtility::Var *snap_vars_ptr = server_snapshots.front().node_vars[net_node_id].ptr();
		for (int var_id = 0; var_id < server_snapshots.front().node_vars[net_node_id].size(); var_id += 1) {
			// Note: the snapshot variable array is ordered per var_id.
			const Variant old_val = rew_node_data->vars[var_id].var.value;
			if (!scene_synchronizer->synchronizer_variant_evaluation(
						old_val,
						snap_vars_ptr[var_id].value)) {
				// Different
				rew_node_data->vars[var_id].var.value = snap_vars_ptr[var_id].value;
				node->set(snap_vars_ptr[var_id].name, snap_vars_ptr[var_id].value);
				NET_DEBUG_PRINT("[Snapshot paused controller] Node: " + node->get_path());
				NET_DEBUG_PRINT(" |- Variable: " + snap_vars_ptr[var_id].name + "; value: " + snap_vars_ptr[var_id].value);
				scene_synchronizer->change_event_add(
						rew_node_data,
						var_id,
						old_val);
			}
		}
	}

	server_snapshots.pop_front();

	scene_synchronizer->change_events_flush();
}

bool ClientSynchronizer::parse_sync_data(
		Variant p_sync_data,
		void *p_user_pointer,
		void (*p_node_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data),
		void (*p_controller_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id),
		void (*p_variable_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, const Variant &p_value)) {
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
	uint32_t var_id = UINT32_MAX;

	for (int snap_data_index = 0; snap_data_index < raw_snapshot.size(); snap_data_index += 1) {
		const Variant v = raw_snapshot_ptr[snap_data_index];
		if (synchronizer_node_data == nullptr) {
			// Node is null so we expect `v` has the node info.

			bool skip_this_node = false;
			Node *node = nullptr;
			uint32_t net_node_id = UINT32_MAX;
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
				NetUtility::NodeData *nd = scene_synchronizer->get_node_data(net_node_id);
				if (nd != nullptr) {
					synchronizer_node_data = nd;
					goto node_lookup_out;
				}
			} else {
				// The arrived snapshot does't seems to be in the expected form.
				ERR_FAIL_V_MSG(false, "This snapshot is corrupted. Now the node is expected, " + String(v) + " was submitted instead.");
			}

			if (synchronizer_node_data == nullptr) {
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

				// Register this node, so to make sure the client is tracking it.
				NetUtility::NodeData *nd = scene_synchronizer->register_node(node);
				if (nd != nullptr) {
					// Set the node ID.
					scene_synchronizer->set_node_data_id(nd, net_node_id);
					synchronizer_node_data = nd;
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

#ifdef DEBUG_ENABLED
			// At this point the ID is never UINT32_MAX thanks to the above
			// mechanism.
			CRASH_COND(synchronizer_node_data->id == UINT32_MAX);
#endif

			p_node_parse(p_user_pointer, synchronizer_node_data);

			if (synchronizer_node_data->is_controller) {
				// This is a controller, so the next data is the input ID.
				ERR_FAIL_COND_V(snap_data_index + 1 >= raw_snapshot.size(), false);
				snap_data_index += 1;
				const uint32_t input_id = raw_snapshot_ptr[snap_data_index];
				ERR_FAIL_COND_V_MSG(input_id == UINT32_MAX, false, "The server is always able to send input_id, so this snapshot seems corrupted.");

				p_controller_parse(p_user_pointer, synchronizer_node_data, input_id);
			}

		} else if (var_id == UINT32_MAX) {
			// When the node is known and the `var_id` not, we expect a
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
				StringName variable_name = var_data[1];

				{
					int64_t index = synchronizer_node_data->vars.find(variable_name);
					if (index == -1) {
						// The variable is not known locally, so just add it so
						// to store the variable ID.
						index = synchronizer_node_data->vars.size();

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
						NET_DEBUG_ERR("The variable " + variable_name + " for the node " + synchronizer_node_data->node->get_path() + " was not known on this client. This should never happen, make sure to register the same nodes on the client and server.");
					}

					if (index != var_id) {
						if (synchronizer_node_data[var_id].id != UINT32_MAX) {
							// It's not expected because if index is different to
							// var_id, var_id should have a not yet initialized
							// variable.
							NET_DEBUG_ERR("This snapshot is corrupted. The var_id, at this point, must have a not yet init variable.");
							notify_server_full_snapshot_is_needed();
							return false;
						}

						// Make sure the variable is at the right index.
						SWAP(synchronizer_node_data->vars[index], synchronizer_node_data->vars[var_id]);
					}
				}

				// Make sure the ID is properly assigned.
				synchronizer_node_data->vars[var_id].id = var_id;

			} else if (v.get_type() == Variant::INT) {
				// The variable is stored in the compact form.

				var_id = v;

				if (var_id >= synchronizer_node_data->vars.size() ||
						synchronizer_node_data->vars[var_id].id == UINT32_MAX) {
					NET_DEBUG_PRINT("The var with ID `" + itos(var_id) + "` is not know by this peer, this is not supposed to happen.");

					notify_server_full_snapshot_is_needed();

					// Skip the next data since it's the value of this variable.
					snap_data_index += 1;
					var_id = UINT32_MAX;
					continue;
				}

			} else {
				ERR_FAIL_V_MSG(false, "The snapshot received seems corrupted. The variable is expected but " + String(v) + " received instead.");
			}

		} else {
			// The node is known, also the variable name is known, so the value
			// is expected.

			p_variable_parse(
					p_user_pointer,
					synchronizer_node_data,
					var_id,
					v);

			// Just reset the variable name so we can continue iterate.
			var_id = UINT32_MAX;
		}
	}

	return true;
}

void ClientSynchronizer::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		// Nothing to do.
		return;
       }

	if (p_enabled) {
		// Postpone enabling when the next server snapshot is received.
		want_to_enable = true;
	} else {
		// Disabling happens immediately.
		enabled = false;
		want_to_enable = false;
		scene_synchronizer->emit_signal("sync_paused");
	}
}

bool ClientSynchronizer::parse_snapshot(Variant p_snapshot) {

	if (want_to_enable) {
		if (enabled) {
			NET_DEBUG_ERR("At this point the client is supposed to be disabled. This is a bug that must be solved.");
		}
		// The netwroking is disabled and we can re-enable it.
               	enabled = true;
               	want_to_enable = false;
               	scene_synchronizer->emit_signal("sync_started");
	}

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

	ParseData parse_data{
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
				if (uint32_t(pd->snapshot.node_vars.size()) <= p_node_data->id) {
					pd->snapshot.node_vars.resize(p_node_data->id + 1);
				}

				// Make sure this snapshot has all the variables.
				pd->snapshot.node_vars.write[p_node_data->id].resize(p_node_data->vars.size());
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
			[](void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, const Variant &p_value) {
				ParseData *pd = static_cast<ParseData *>(p_user_pointer);

#ifdef DEBUG_ENABLED
				// This can't be triggered because th `Parse Node` function
				// above make sure to create room for this array.
				CRASH_COND(uint32_t(pd->snapshot.node_vars.size()) <= p_node_data->id);
#endif // ~DEBUG_ENABLED

				if (unlikely(p_node_data->vars.size() != uint32_t(pd->snapshot.node_vars[p_node_data->id].size()))) {
					// This mean the parser just added a new variable.
					// Already notified by the parser.
					pd->snapshot.node_vars.write[p_node_data->id].resize(p_node_data->vars.size());
				}

				pd->snapshot.node_vars.write[p_node_data->id].write[p_var_id].name = p_node_data->vars[p_var_id].var.name;
				pd->snapshot.node_vars.write[p_node_data->id].write[p_var_id].value = p_value.duplicate(true);
			});

	if (success == false) {
		NET_DEBUG_ERR("Snapshot:");
		NET_DEBUG_ERR(p_snapshot);
		return false;
	}

	// We espect that the player_controller is updated by this new snapshot,
	// so make sure it's done so.
	if (unlikely(last_received_snapshot.input_id == UINT32_MAX)) {
		NET_DEBUG_PRINT("Recovery aborted, the player controller (" + player_controller_node_data->node->get_path() + ") was not part of the received snapshot, probably the server doesn't have important informations for this peer. NetUtility::Snapshot:");
		NET_DEBUG_PRINT(p_snapshot);
		return false;
	} else {
		// Success.

		return true;
	}
}

bool ClientSynchronizer::compare_vars(
		const NetUtility::NodeData *p_synchronizer_node_data,
		const Vector<NetUtility::Var> &p_server_vars,
		const Vector<NetUtility::Var> &p_client_vars,
		Vector<NetUtility::Var> &r_postponed_recover) {
	const NetUtility::Var *s_vars = p_server_vars.ptr();
	const NetUtility::Var *c_vars = p_client_vars.ptr();

#ifdef DEBUG_ENABLED
	bool diff = false;
#endif

	if (p_server_vars.size() != p_client_vars.size() ||
			uint32_t(p_server_vars.size()) > p_synchronizer_node_data->vars.size()) {
		NET_DEBUG_PRINT("Difference found: The server has a different variable count (" + itos(p_server_vars.size()) + ") compared to the client (" + itos(p_client_vars.size()) + ").");
		NET_DEBUG_PRINT("Server variables:");
		for (int i = 0; i < p_server_vars.size(); i += 1) {
			NET_DEBUG_PRINT("  |- " + p_server_vars[i].name + ": " + p_server_vars[i].value);
		}
		NET_DEBUG_PRINT("  ------");

		NET_DEBUG_PRINT("Client variables:");
		for (int i = 0; i < p_client_vars.size(); i += 1) {
			NET_DEBUG_PRINT("  |- " + p_client_vars[i].name + ": " + p_client_vars[i].value);
		}
		NET_DEBUG_PRINT("  ------");
		return true;
	}

	for (uint32_t var_index = 0; var_index < uint32_t(p_server_vars.size()); var_index += 1) {
		if (s_vars[var_index].name == StringName()) {
			// This variable was not set, skip the check.
			continue;
		}

		// Compare.
		const bool different =
				// Make sure this variable is set.
				c_vars[var_index].name == StringName() ||
				// Check if the value is different.
				!scene_synchronizer->synchronizer_variant_evaluation(
						s_vars[var_index].value,
						c_vars[var_index].value);

		if (different) {
			if (p_synchronizer_node_data->vars[var_index].skip_rewinding) {
				// The vars are different, but this variable don't what to
				// trigger a rewind.
				r_postponed_recover.push_back(s_vars[var_index]);
			} else {
				// The vars are different.
				NET_DEBUG_PRINT("Difference found on var #" + itos(var_index) + " " + p_synchronizer_node_data->vars[var_index].var.name + " " +
								"Server value: `" + s_vars[var_index].value + "` " +
								"Client value: `" + c_vars[var_index].value + "`.    " +
								"[Server name: `" + s_vars[var_index].name + "` " +
								"Client name: `" + c_vars[var_index].name + "`].");
#ifdef DEBUG_ENABLED
				diff = true;
#else
				return true;
#endif
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
