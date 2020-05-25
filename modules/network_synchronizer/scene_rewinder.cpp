/*************************************************************************/
/*  scene_rewinder.cpp                                                   */
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

#include "scene_rewinder.h"

#include "net_utilities.h"
#include "networked_controller.h"
#include "scene/main/window.h"

// TODO add back the DOLL disabling

VarData::VarData() :
		id(0),
		skip_rewinding(false),
		enabled(false) {}

VarData::VarData(StringName p_name) :
		id(0),
		skip_rewinding(false),
		enabled(false) {
	var.name = p_name;
}

VarData::VarData(uint32_t p_id, StringName p_name, Variant p_val, bool p_skip_rewinding, bool p_enabled) :
		id(p_id),
		skip_rewinding(p_skip_rewinding),
		enabled(p_enabled) {
	var.name = p_name;
	var.value = p_val;
}

bool VarData::operator==(const VarData &p_other) const {
	return var.name == p_other.var.name;
}

NodeData::NodeData() :
		id(0),
		instance_id(uint64_t(0)),
		is_controller(false),
		controlled_by(uint64_t(0)),
		isle_id(uint64_t(0)),
		node(nullptr) {
}

NodeData::NodeData(uint32_t p_id, ObjectID p_instance_id, bool is_controller) :
		id(p_id),
		instance_id(p_instance_id),
		is_controller(is_controller),
		controlled_by(uint64_t(0)),
		isle_id(uint64_t(0)),
		node(nullptr) {
}

int NodeData::find_var_by_id(uint32_t p_id) const {
	if (p_id == 0) {
		return -1;
	}
	const VarData *v = vars.ptr();
	for (int i = 0; i < vars.size(); i += 1) {
		if (v[i].id == p_id) {
			return i;
		}
	}
	return -1;
}

bool NodeData::can_be_part_of_isle(ControllerID p_controller_id, bool p_is_main_controller) const {
	if (instance_id == p_controller_id) {
		return true;
	} else if (controlled_by == p_controller_id) {
		return true;
	} else if (is_controller == false && controlled_by.is_null() && p_is_main_controller) {
		return true;
	} else {
		return false;
	}
}

void NodeData::process(const real_t p_delta) const {
	if (functions.size() <= 0)
		return;

	const Variant var_delta = p_delta;
	const Variant *fake_array_vars = &var_delta;

	const StringName *funcs = functions.ptr();

	Callable::CallError e;
	for (int i = 0; i < functions.size(); i += 1) {
		node->call(funcs[i], &fake_array_vars, 1, e);
	}
}

void SceneRewinder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("reset_synchronizer_mode"), &SceneRewinder::reset_synchronizer_mode);
	ClassDB::bind_method(D_METHOD("clear"), &SceneRewinder::clear);

	ClassDB::bind_method(D_METHOD("set_doll_desync_tolerance", "tolerance"), &SceneRewinder::set_doll_desync_tolerance);
	ClassDB::bind_method(D_METHOD("get_doll_desync_tolerance"), &SceneRewinder::get_doll_desync_tolerance);

	ClassDB::bind_method(D_METHOD("set_server_notify_state_interval", "interval"), &SceneRewinder::set_server_notify_state_interval);
	ClassDB::bind_method(D_METHOD("get_server_notify_state_interval"), &SceneRewinder::get_server_notify_state_interval);

	ClassDB::bind_method(D_METHOD("set_comparison_float_tolerance", "tolerance"), &SceneRewinder::set_comparison_float_tolerance);
	ClassDB::bind_method(D_METHOD("get_comparison_float_tolerance"), &SceneRewinder::get_comparison_float_tolerance);

	ClassDB::bind_method(D_METHOD("register_variable", "node", "variable", "on_change_notify", "skip_rewinding"), &SceneRewinder::register_variable, DEFVAL(StringName()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("unregister_variable", "node", "variable"), &SceneRewinder::unregister_variable);

	ClassDB::bind_method(D_METHOD("get_changed_event_name", "variable"), &SceneRewinder::get_changed_event_name);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "method"), &SceneRewinder::track_variable_changes);
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "method"), &SceneRewinder::untrack_variable_changes);

	ClassDB::bind_method(D_METHOD("set_node_as_controlled_by", "node", "controller"), &SceneRewinder::set_node_as_controlled_by);
	ClassDB::bind_method(D_METHOD("unregister_controller", "controller"), &SceneRewinder::unregister_controller);

	ClassDB::bind_method(D_METHOD("register_process", "node", "function"), &SceneRewinder::register_process);
	ClassDB::bind_method(D_METHOD("unregister_process", "node", "function"), &SceneRewinder::unregister_process);

	ClassDB::bind_method(D_METHOD("is_recovered"), &SceneRewinder::is_recovered);
	ClassDB::bind_method(D_METHOD("is_resetted"), &SceneRewinder::is_resetted);
	ClassDB::bind_method(D_METHOD("is_rewinding"), &SceneRewinder::is_rewinding);

	ClassDB::bind_method(D_METHOD("force_state_notify"), &SceneRewinder::force_state_notify);

	ClassDB::bind_method(D_METHOD("_on_peer_connected"), &SceneRewinder::_on_peer_connected);
	ClassDB::bind_method(D_METHOD("_on_peer_disconnected"), &SceneRewinder::_on_peer_disconnected);

	ClassDB::bind_method(D_METHOD("__clear"), &SceneRewinder::__clear);
	ClassDB::bind_method(D_METHOD("_rpc_send_state"), &SceneRewinder::_rpc_send_state);
	ClassDB::bind_method(D_METHOD("_rpc_notify_need_full_snapshot"), &SceneRewinder::_rpc_notify_need_full_snapshot);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "doll_desync_tolerance", PROPERTY_HINT_RANGE, "1,10000,1"), "set_doll_desync_tolerance", "get_doll_desync_tolerance");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "server_notify_state_interval", PROPERTY_HINT_RANGE, "0.001,10.0,0.0001"), "set_server_notify_state_interval", "get_server_notify_state_interval");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "comparison_float_tolerance", PROPERTY_HINT_RANGE, "0.000001,0.01,0.000001"), "set_comparison_float_tolerance", "get_comparison_float_tolerance");
}

void SceneRewinder::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			// TODO add a signal that allows to not check this each frame.
			if (unlikely(peer_ptr != get_multiplayer()->get_network_peer().ptr())) {
				reset_synchronizer_mode();
			}

			const int lowest_priority_number = INT32_MAX;
			ERR_FAIL_COND_MSG(get_process_priority() != lowest_priority_number, "The process priority MUST not be changed, is likely there is a better way of doing what you are trying to do, if you really need it please open an issue.");

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

			if (rewinder) {
				memdelete(rewinder);
				rewinder = nullptr;
				rewinder_type = REWINDER_TYPE_NULL;
			}

			set_physics_process_internal(false);
		}
	}
}

SceneRewinder::SceneRewinder() :
		doll_desync_tolerance(120),
		server_notify_state_interval(1.0),
		comparison_float_tolerance(0.001),
		rewinder_type(REWINDER_TYPE_NULL),
		rewinder(nullptr),
		recover_in_progress(false),
		reset_in_progress(false),
		rewinding_in_progress(false),
		node_counter(1),
		generate_id(false),
		main_controller_object_id(),
		main_controller(nullptr) {
	rpc_config("__clear", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_send_state", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_notify_need_full_snapshot", MultiplayerAPI::RPC_MODE_REMOTE);
}

SceneRewinder::~SceneRewinder() {
	__clear();
	if (rewinder) {
		memdelete(rewinder);
		rewinder = nullptr;
		rewinder_type = REWINDER_TYPE_NULL;
	}
}

void SceneRewinder::set_doll_desync_tolerance(int p_tolerance) {
	doll_desync_tolerance = p_tolerance;
}

int SceneRewinder::get_doll_desync_tolerance() const {
	return doll_desync_tolerance;
}

void SceneRewinder::set_server_notify_state_interval(real_t p_interval) {
	server_notify_state_interval = p_interval;
}

real_t SceneRewinder::get_server_notify_state_interval() const {
	return server_notify_state_interval;
}

void SceneRewinder::set_comparison_float_tolerance(real_t p_tolerance) {
	comparison_float_tolerance = p_tolerance;
}

real_t SceneRewinder::get_comparison_float_tolerance() const {
	return comparison_float_tolerance;
}

void SceneRewinder::register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify, bool p_skip_rewinding) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);

	const int id = node_data->vars.find(p_variable);
	if (id == -1) {
		const Variant old_val = p_node->get(p_variable);
		const int var_id = generate_id ? node_data->vars.size() + 1 : 0;
		node_data->vars.push_back(VarData(
				var_id,
				p_variable,
				old_val,
				p_skip_rewinding,
				true));
	} else {
		node_data->vars.write[id].skip_rewinding = p_skip_rewinding;
		node_data->vars.write[id].enabled = true;
	}

	if (p_node->has_signal(get_changed_event_name(p_variable)) == false) {
		p_node->add_user_signal(MethodInfo(
				get_changed_event_name(p_variable)));
	}

	if (p_on_change_notify != StringName()) {
		track_variable_changes(p_node, p_variable, p_on_change_notify);
	}

	rewinder->on_variable_added(p_node->get_instance_id(), p_variable);
}

void SceneRewinder::unregister_variable(Node *p_node, StringName p_variable) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());

	NodeData *nd = nodes_data.lookup_ptr(p_node->get_instance_id());
	if (nd == nullptr)
		return;
	if (nd->vars.find(p_variable) == -1)
		return;

	// Disconnects the eventual connected methods
	List<Connection> connections;
	p_node->get_signal_connection_list(get_changed_event_name(p_variable), &connections);

	for (List<Connection>::Element *e = connections.front(); e != nullptr; e = e->next()) {
		p_node->disconnect(get_changed_event_name(p_variable), e->get().callable);
	}

	// Disable variable, don't remove it to preserve var node IDs.
	int id = nd->vars.find(p_variable);
	CRASH_COND(id == -1); // Unreachable
	nd->vars.write[id].enabled = false;
}

String SceneRewinder::get_changed_event_name(StringName p_variable) {
	return "variable_" + p_variable + "_changed";
}

void SceneRewinder::track_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());
	ERR_FAIL_COND(p_method == StringName());

	NodeData *nd = nodes_data.lookup_ptr(p_node->get_instance_id());
	ERR_FAIL_COND_MSG(nd == nullptr, "You need to register the variable to track its changes.");
	ERR_FAIL_COND_MSG(nd->vars.find(p_variable) == -1, "You need to register the variable to track its changes.");

	if (p_node->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method)) == false) {
		p_node->connect(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method));
	}
}

void SceneRewinder::untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_variable == StringName());
	ERR_FAIL_COND(p_method == StringName());

	NodeData *nd = nodes_data.lookup_ptr(p_node->get_instance_id());
	if (nd == nullptr)
		return;
	if (nd->vars.find(p_variable) == -1)
		return;

	if (p_node->is_connected(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method))) {
		p_node->disconnect(
				get_changed_event_name(p_variable),
				Callable(p_node, p_method));
	}
}

void SceneRewinder::set_node_as_controlled_by(Node *p_node, Node *p_controller) {
	ERR_FAIL_COND(p_node == nullptr);
	NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);

	if (node_data->controlled_by.is_null() == false) {
		NodeData *controller_node_data = nodes_data.lookup_ptr(node_data->controlled_by);
		if (controller_node_data) {
			controller_node_data->controlled_nodes.erase(p_node->get_instance_id());
		}
		node_data->controlled_by = ObjectID();
	}

	if (node_data->isle_id.is_null() == false) {
		remove_from_isle(node_data->instance_id, node_data->isle_id);
		node_data->isle_id = uint64_t(0);
	}

	if (p_controller) {
		NetworkedController *c = Object::cast_to<NetworkedController>(p_controller);
		ERR_FAIL_COND(c == nullptr);

		NodeData *controller_node_data = register_node(p_controller);
		ERR_FAIL_COND(controller_node_data == nullptr);
		ERR_FAIL_COND(controller_node_data->is_controller == false);
		controller_node_data->controlled_nodes.push_back(p_node->get_instance_id());
		node_data->controlled_by = p_controller->get_instance_id();

		// This MUST never be false.
		CRASH_COND(node_data->can_be_part_of_isle(p_controller->get_instance_id(), p_controller == main_controller) == false);

		put_into_isle(node_data->instance_id, p_controller->get_instance_id());
		node_data->isle_id = p_controller->get_instance_id();
	}
}

void SceneRewinder::unregister_controller(Node *p_controller) {
	NetworkedController *c = Object::cast_to<NetworkedController>(p_controller);
	ERR_FAIL_COND(c == nullptr);
	_unregister_controller(c);
}

void SceneRewinder::_register_controller(NetworkedController *p_controller) {
	if (p_controller->has_scene_rewinder()) {
		ERR_FAIL_COND_MSG(p_controller->get_scene_rewinder() != this, "This controller is associated with a different scene rewinder.");
	} else {
		NodeData *node_data = nodes_data.lookup_ptr(p_controller->get_instance_id());
		ERR_FAIL_COND(node_data == nullptr);

		IsleData *isle = isle_data.lookup_ptr(p_controller->get_instance_id());
		if (isle == nullptr) {
			IsleData _isle;
			_isle.controller_instance_id = p_controller->get_instance_id();
			_isle.controller = p_controller;
			isle_data.set(p_controller->get_instance_id(), _isle);
			isle = isle_data.lookup_ptr(p_controller->get_instance_id());
		} else {
			isle->nodes.clear();
		}

		// Unreachable.
		CRASH_COND(isle == nullptr);
		CRASH_COND(isle->controller_instance_id.is_null());
		CRASH_COND(isle->controller == nullptr);

		p_controller->set_scene_rewinder(this);

		node_data->is_controller = true;

		if (p_controller->is_player_controller()) {
			if (main_controller == nullptr) {
				main_controller = p_controller;
				main_controller_object_id = main_controller->get_instance_id();
			}
		}

		// Make sure to put all nodes that, have the rights to be in this isle.
		for (
				OAHashMap<ObjectID, NodeData>::Iterator it = nodes_data.iter();
				it.valid;
				it = nodes_data.next_iter(it)) {
			if (it.value->can_be_part_of_isle(p_controller->get_instance_id(), p_controller == main_controller)) {
				isle->nodes.push_back(*it.key);
				it.value->isle_id = p_controller->get_instance_id();
			}
		}
	}
}

void SceneRewinder::_unregister_controller(NetworkedController *p_controller) {
	ERR_FAIL_COND_MSG(p_controller->get_scene_rewinder() != this, "This controller is associated with this scene rewinder.");
	p_controller->set_scene_rewinder(nullptr);
	NodeData *node_data = nodes_data.lookup_ptr(p_controller->get_instance_id());
	if (node_data) {
		node_data->is_controller = false;
	}

	isle_data.remove(p_controller->get_instance_id());

	if (main_controller == p_controller) {
		main_controller = nullptr;
		main_controller_object_id = ObjectID();
	}

	for (
			OAHashMap<ObjectID, NodeData>::Iterator it = nodes_data.iter();
			it.valid;
			it = nodes_data.next_iter(it)) {
		if (it.value->isle_id == p_controller->get_instance_id()) {
			it.value->isle_id = uint64_t(0);
		}
	}
}

void SceneRewinder::register_process(Node *p_node, StringName p_function) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_function == StringName());
	NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);

	if (node_data->functions.find(p_function) == -1) {
		node_data->functions.push_back(p_function);
	}
}

void SceneRewinder::unregister_process(Node *p_node, StringName p_function) {
	ERR_FAIL_COND(p_node == nullptr);
	ERR_FAIL_COND(p_function == StringName());
	NodeData *node_data = register_node(p_node);
	ERR_FAIL_COND(node_data == nullptr);
	node_data->functions.erase(p_function);
}

bool SceneRewinder::is_recovered() const {
	return recover_in_progress;
}

bool SceneRewinder::is_resetted() const {
	return reset_in_progress;
}

bool SceneRewinder::is_rewinding() const {
	return rewinding_in_progress;
}

void SceneRewinder::force_state_notify() {
	ERR_FAIL_COND(rewinder_type != REWINDER_TYPE_SERVER);
	ServerRewinder *r = static_cast<ServerRewinder *>(rewinder);
	// + 1.0 is just a ridiculous high number to be sure to avoid float
	// precision error.
	r->state_notifier_timer = get_server_notify_state_interval() + 1.0;
}

void SceneRewinder::_on_peer_connected(int p_peer) {
	peer_data.set(p_peer, PeerData());
}

void SceneRewinder::_on_peer_disconnected(int p_peer) {
	peer_data.remove(p_peer);
}

void SceneRewinder::reset_synchronizer_mode() {
	set_physics_process_internal(false);
	generate_id = false;

	if (rewinder) {
		memdelete(rewinder);
		rewinder = nullptr;
		rewinder_type = REWINDER_TYPE_NULL;
	}

	peer_ptr = get_multiplayer()->get_network_peer().ptr();

	if (get_tree() == nullptr || get_tree()->get_network_peer().is_null()) {
		rewinder_type = REWINDER_TYPE_NONET;
		rewinder = memnew(NoNetRewinder(this));
		generate_id = true;

	} else if (get_tree()->is_network_server()) {
		rewinder_type = REWINDER_TYPE_SERVER;
		rewinder = memnew(ServerRewinder(this));
		generate_id = true;
	} else {
		rewinder_type = REWINDER_TYPE_CLIENT;
		rewinder = memnew(ClientRewinder(this));
	}

	// Always runs the SceneRewinder last.
	const int lowest_priority_number = INT32_MAX;
	set_process_priority(lowest_priority_number);
	set_physics_process_internal(true);

	if (rewinder) {
		// Notify the presence all available nodes and its variables to the rewinder.
		for (OAHashMap<ObjectID, NodeData>::Iterator it = nodes_data.iter(); it.valid; it = nodes_data.next_iter(it)) {
			rewinder->on_node_added(*it.key);
			for (int i = 0; i < it.value->vars.size(); i += 1) {
				rewinder->on_variable_added(*it.key, it.value->vars[i].var.name);
			}
		}
	}
}

void SceneRewinder::clear() {
	if (rewinder_type == REWINDER_TYPE_NONET) {
		__clear();
	} else {
		ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "The clear function must be called on server");
		__clear();
		rpc("__clear");
	}
}

void SceneRewinder::__clear() {
	for (OAHashMap<ObjectID, NodeData>::Iterator it = nodes_data.iter(); it.valid; it = nodes_data.next_iter(it)) {
		const VarData *object_vars = it.value->vars.ptr();
		for (int i = 0; i < it.value->vars.size(); i += 1) {
			Node *node = static_cast<Node *>(ObjectDB::get_instance(it.value->instance_id));

			if (node != nullptr) {
				// Unregister the variable so the connected variables are
				// correctly removed
				unregister_variable(node, object_vars[i].var.name);
			}
		}

		it.value->vars.clear();
		it.value->controlled_nodes.clear();
		it.value->functions.clear();
		it.value->node = nullptr;
	}

	nodes_data.clear();
	isle_data.clear();
	node_counter = 1;

	if (rewinder) {
		rewinder->clear();
	}
}

void SceneRewinder::_rpc_send_state(Variant p_snapshot) {
	ERR_FAIL_COND(get_tree()->is_network_server() == true);
	rewinder->receive_snapshot(p_snapshot);
}

void SceneRewinder::_rpc_notify_need_full_snapshot() {
	ERR_FAIL_COND(get_tree()->is_network_server() == false);

	const int sender_peer = get_tree()->get_multiplayer()->get_rpc_sender_id();
	PeerData *pd = peer_data.lookup_ptr(sender_peer);
	ERR_FAIL_COND(pd == nullptr);
	pd->need_full_snapshot = true;
}

NodeData *SceneRewinder::register_node(Node *p_node) {
	ERR_FAIL_COND_V(p_node == nullptr, nullptr);

	NodeData *node_data = nodes_data.lookup_ptr(p_node->get_instance_id());
	if (node_data == nullptr) {
		const uint32_t node_id(generate_id ? ++node_counter : 0);
		nodes_data.set(
				p_node->get_instance_id(),
				NodeData(node_id, p_node->get_instance_id(), false));
		node_data = nodes_data.lookup_ptr(p_node->get_instance_id());
		node_data->node = p_node;
		rewinder->on_node_added(p_node->get_instance_id());

		// Register this node as controller if it's a controller.
		NetworkedController *controller = Object::cast_to<NetworkedController>(p_node);
		if (controller) {
			_register_controller(controller);
		}
	}

	return node_data;
}

bool SceneRewinder::vec2_evaluation(const Vector2 a, const Vector2 b) {
	return (a - b).length_squared() <= (comparison_float_tolerance * comparison_float_tolerance);
}

bool SceneRewinder::vec3_evaluation(const Vector3 a, const Vector3 b) {
	return (a - b).length_squared() <= (comparison_float_tolerance * comparison_float_tolerance);
}

bool SceneRewinder::rewinder_variant_evaluation(const Variant &v_1, const Variant &v_2) {
	if (v_1.get_type() != v_2.get_type()) {
		return false;
	}

	// Custom evaluation methods
	switch (v_1.get_type()) {
		case Variant::FLOAT: {
			const real_t a(v_1);
			const real_t b(v_2);
			return ABS(a - b) <= comparison_float_tolerance;
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
			if (ABS(a.d - b.d) <= comparison_float_tolerance) {
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

				if (rewinder_variant_evaluation(
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

bool SceneRewinder::is_client() const {
	return rewinder_type == REWINDER_TYPE_CLIENT;
}

void SceneRewinder::validate_nodes() {
	if (ObjectDB::get_instance(main_controller_object_id) == nullptr) {
		main_controller = nullptr;
		main_controller_object_id = ObjectID();
	}

	std::vector<ObjectID> null_objects;

	for (OAHashMap<ObjectID, NodeData>::Iterator it = nodes_data.iter(); it.valid; it = nodes_data.next_iter(it)) {
		if (ObjectDB::get_instance(it.value->instance_id) == nullptr) {
			null_objects.push_back(it.value->instance_id);
			if (it.value->isle_id.is_null() == false) {
				remove_from_isle(*it.key, it.value->isle_id);
				it.value->isle_id = uint64_t(0);
			}
		} else {
			if (it.value->isle_id.is_null()) {
				if (main_controller && it.value->can_be_part_of_isle(main_controller->get_instance_id(), true)) {
					put_into_isle(*it.key, main_controller->get_instance_id());
					it.value->isle_id = *it.key;
				}
			}
		}
	}

	// Removes the null objects.
	for (size_t i = 0; i < null_objects.size(); i += 1) {
		nodes_data.remove(null_objects[i]);
		isle_data.remove(null_objects[i]);
	}
}

void SceneRewinder::put_into_isle(ObjectID p_node_id, ControllerID p_isle_id) {
	IsleData *isle = isle_data.lookup_ptr(p_isle_id);
	ERR_FAIL_COND(isle == nullptr);
	const int index = isle->nodes.find(p_node_id);
	if (index == -1) {
		isle->nodes.push_back(p_node_id);
	}
}

void SceneRewinder::remove_from_isle(ObjectID p_node_id, ControllerID p_isle_id) {
	IsleData *isle = isle_data.lookup_ptr(p_isle_id);
	ERR_FAIL_COND(isle == nullptr);
	isle->nodes.erase(p_node_id);
}

void SceneRewinder::process() {
	validate_nodes();
	rewinder->process();
}

void SceneRewinder::pull_node_changes(NodeData *p_node_data) {
	Node *node = p_node_data->node;
	const int var_count = p_node_data->vars.size();
	VarData *object_vars = p_node_data->vars.ptrw();

	for (int i = 0; i < var_count; i += 1) {
		if (object_vars[i].enabled == false) {
			continue;
		}

		const Variant old_val = object_vars[i].var.value;
		const Variant new_val = node->get(object_vars[i].var.name);

		if (!rewinder_variant_evaluation(old_val, new_val)) {
			object_vars[i].var.value = new_val;
			node->emit_signal(get_changed_event_name(object_vars[i].var.name));
			rewinder->on_variable_changed(node->get_instance_id(), object_vars[i].var.name);
		}
	}
}

Snapshot::operator String() const {
	String s;
	s += "Snapshot:\n";
	for (
			OAHashMap<ObjectID, uint64_t>::Iterator it = controllers_input_id.iter();
			it.valid;
			it = controllers_input_id.next_iter(it)) {
		s += "\nController: ";
		if (nullptr != ObjectDB::get_instance(*it.key))
			s += static_cast<Node *>(ObjectDB::get_instance(*it.key))->get_path();
		else
			s += " (Object ID): " + itos(*it.key);
		s += " - ";
		s += "input ID: ";
		s += itos(*it.value);
	}

	for (
			OAHashMap<ObjectID, Vector<VarData>>::Iterator it = node_vars.iter();
			it.valid;
			it = node_vars.next_iter(it)) {
		s += "\nNode Data: ";
		if (nullptr != ObjectDB::get_instance(*it.key))
			s += static_cast<Node *>(ObjectDB::get_instance(*it.key))->get_path();
		else
			s += " (Object ID): " + itos(*it.key);
		for (int i = 0; i < it.value->size(); i += 1) {
			s += "\n|- Variable: ";
			s += (*it.value)[i].var.name;
			s += " = ";
			s += String((*it.value)[i].var.value);
		}
	}
	return s;
}

IsleSnapshot::operator String() const {
	String s;
	s += "Input ID: " + itos(input_id) + "; ";
	for (
			OAHashMap<ObjectID, Vector<VarData>>::Iterator it = node_vars.iter();
			it.valid;
			it = node_vars.next_iter(it)) {
		s += "\nNode: ";
		if (nullptr != ObjectDB::get_instance(*it.key))
			s += static_cast<Node *>(ObjectDB::get_instance(*it.key))->get_path();
		else
			s += " (Object ID): " + itos(*it.key);

		for (int i = 0; i < it.value->size(); i += 1) {
			s += "\n|- Variable: ";
			s += (*it.value)[i].var.name;
			s += " = ";
			s += String((*it.value)[i].var.value);
		}
	}
	return s;
}

Rewinder::Rewinder(SceneRewinder *p_node) :
		scene_rewinder(p_node) {
}

Rewinder::~Rewinder() {
}

NoNetRewinder::NoNetRewinder(SceneRewinder *p_node) :
		Rewinder(p_node) {}

void NoNetRewinder::clear() {
}

void NoNetRewinder::process() {
	const real_t delta = scene_rewinder->get_physics_process_delta_time();

	// Process the scene
	for (
			OAHashMap<ControllerID, NodeData>::Iterator it = scene_rewinder->nodes_data.iter();
			it.valid;
			it = scene_rewinder->nodes_data.next_iter(it)) {
		NodeData *node_data = it.value;
		ERR_CONTINUE(node_data == nullptr);

		node_data->process(delta);
	}

	// Process the controllers
	for (
			OAHashMap<ControllerID, IsleData>::Iterator it = scene_rewinder->isle_data.iter();
			it.valid;
			it = scene_rewinder->isle_data.next_iter(it)) {
		NetworkedController *controller = it.value->controller;

		controller->process(delta);
	}

	// Pull the changes.
	for (
			OAHashMap<ControllerID, NodeData>::Iterator it = scene_rewinder->nodes_data.iter();
			it.valid;
			it = scene_rewinder->nodes_data.next_iter(it)) {
		NodeData *node_data = it.value;
		ERR_CONTINUE(node_data == nullptr);

		scene_rewinder->pull_node_changes(node_data);
	}
}

void NoNetRewinder::receive_snapshot(Variant _p_snapshot) {
}

ServerRewinder::ServerRewinder(SceneRewinder *p_node) :
		Rewinder(p_node),
		state_notifier_timer(0.0) {}

void ServerRewinder::clear() {
	state_notifier_timer = 0.0;
	changes.clear();
}

void ServerRewinder::process() {
	const real_t delta = scene_rewinder->get_physics_process_delta_time();

	// Process the scene
	for (
			OAHashMap<ControllerID, NodeData>::Iterator it = scene_rewinder->nodes_data.iter();
			it.valid;
			it = scene_rewinder->nodes_data.next_iter(it)) {
		NodeData *node_data = it.value;
		ERR_CONTINUE(node_data == nullptr);

		node_data->process(delta);
	}

	// Process the controllers
	for (
			OAHashMap<ControllerID, IsleData>::Iterator it = scene_rewinder->isle_data.iter();
			it.valid;
			it = scene_rewinder->isle_data.next_iter(it)) {
		NetworkedController *controller = it.value->controller;

		controller->process(delta);
	}

	// Pull the changes.
	for (
			OAHashMap<ControllerID, NodeData>::Iterator it = scene_rewinder->nodes_data.iter();
			it.valid;
			it = scene_rewinder->nodes_data.next_iter(it)) {
		NodeData *node_data = it.value;
		ERR_CONTINUE(node_data == nullptr);

		scene_rewinder->pull_node_changes(node_data);
	}

	process_snapshot_notificator(delta);
}

void ServerRewinder::receive_snapshot(Variant _p_snapshot) {
	// Unreachable
	CRASH_NOW();
}

void ServerRewinder::on_node_added(ObjectID p_node_id) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_rewinder->is_recovered());
#endif
	Change *c = changes.lookup_ptr(p_node_id);
	if (c) {
		c->not_known_before = true;
	} else {
		Change change;
		change.not_known_before = true;
		changes.set(p_node_id, change);
	}
}

void ServerRewinder::on_variable_added(ObjectID p_node_id, StringName p_var_name) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_rewinder->is_recovered());
#endif
	Change *c = changes.lookup_ptr(p_node_id);
	if (c) {
		c->vars.insert(p_var_name);
		c->uknown_vars.insert(p_var_name);
	} else {
		Change change;
		change.vars.insert(p_var_name);
		change.uknown_vars.insert(p_var_name);
		changes.set(p_node_id, change);
	}
}

void ServerRewinder::on_variable_changed(ObjectID p_node_id, StringName p_var_name) {
#ifdef DEBUG_ENABLED
	// Can't happen on server
	CRASH_COND(scene_rewinder->is_recovered());
#endif
	Change *c = changes.lookup_ptr(p_node_id);
	if (c) {
		c->vars.insert(p_var_name);
	} else {
		Change change;
		change.vars.insert(p_var_name);
		changes.set(p_node_id, change);
	}
}

void ServerRewinder::process_snapshot_notificator(real_t p_delta) {
	if (scene_rewinder->peer_data.empty()) {
		// No one is listening.
		return;
	}

	// Notify the state if needed
	state_notifier_timer += p_delta;
	const bool notify_state = state_notifier_timer >= scene_rewinder->get_server_notify_state_interval();

	if (notify_state) {
		state_notifier_timer = 0.0;
	}

	Variant full_snapshot;
	Variant delta_snapshot;
	for (
			OAHashMap<int, PeerData>::Iterator peer_it = scene_rewinder->peer_data.iter();
			peer_it.valid;
			peer_it = scene_rewinder->peer_data.next_iter(peer_it)) {
		if (peer_it.value->force_notify_snapshot == false && notify_state == false) {
			continue;
		}
		peer_it.value->force_notify_snapshot = false;

		if (peer_it.value->need_full_snapshot) {
			peer_it.value->need_full_snapshot = false;
			if (full_snapshot.is_null()) {
				full_snapshot = generate_snapshot(true);
			}
			scene_rewinder->rpc_id(*peer_it.key, "_rpc_send_state", full_snapshot);
		} else {
			if (delta_snapshot.is_null()) {
				delta_snapshot = generate_snapshot(false);
			}
			scene_rewinder->rpc_id(*peer_it.key, "_rpc_send_state", delta_snapshot);
		}
	}

	if (notify_state) {
		// The state got notified, mark this as checkpoint so the next state
		// will contains only the changed things.
		changes.clear();
	}
}

Variant ServerRewinder::generate_snapshot(bool p_full_snapshot) const {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
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

	Vector<Variant> snapshot_data;

	for (
			OAHashMap<ObjectID, NodeData>::Iterator it = scene_rewinder->nodes_data.iter();
			it.valid;
			it = scene_rewinder->nodes_data.next_iter(it)) {
		const NodeData *node_data = it.value;
		if (node_data->node == nullptr || node_data->node->is_inside_tree() == false) {
			continue;
		}

		const Change *change = changes.lookup_ptr(*it.key);

		// Insert NODE DATA.
		Variant snap_node_data;
		if (p_full_snapshot || (change != nullptr && change->not_known_before)) {
			Vector<Variant> _snap_node_data;
			_snap_node_data.resize(2);
			_snap_node_data.write[0] = node_data->id;
			_snap_node_data.write[1] = node_data->node->get_path();
			snap_node_data = _snap_node_data;
		} else {
			// This node is already known on clients, just set the node ID.
			snap_node_data = node_data->id;
		}

		const bool node_has_changes = p_full_snapshot || (change != nullptr && change->vars.empty() == false);

		if (node_data->is_controller) {
			NetworkedController *controller = Object::cast_to<NetworkedController>(node_data->node);
			CRASH_COND(controller == nullptr); // Unreachable

			// TODO make sure to skip un-active controllers.
			if (likely(controller->get_current_input_id() != UINT64_MAX)) {
				// This is a controller, always sync it.
				snapshot_data.push_back(snap_node_data);
				snapshot_data.push_back(controller->get_current_input_id());
			} else {
				// The first ID id is not yet arrived, so just skip this node.
				continue;
			}
		} else {
			if (node_has_changes) {
				snapshot_data.push_back(snap_node_data);
			} else {
				// It has no changes, skip this node.
				continue;
			}
		}

		if (node_has_changes) {
			// Insert the node variables.
			const int size = node_data->vars.size();
			const VarData *vars = node_data->vars.ptr();
			for (int i = 0; i < size; i += 1) {
				if (vars[i].enabled == false) {
					continue;
				}

				if (p_full_snapshot == false && change->vars.has(vars[i].var.name) == false) {
					// This is a delta snapshot and this variable is the same as
					// before. Skip it.
					continue;
				}

				Variant var_info;
				if (p_full_snapshot || change->uknown_vars.has(vars[i].var.name)) {
					Vector<Variant> _var_info;
					_var_info.resize(2);
					_var_info.write[0] = vars[i].id;
					_var_info.write[1] = vars[i].var.name;
					var_info = _var_info;
				} else {
					var_info = vars[i].id;
				}

				snapshot_data.push_back(var_info);
				snapshot_data.push_back(vars[i].var.value);
			}
		}

		// Insert NIL.
		snapshot_data.push_back(Variant());
	}

	return snapshot_data;
}

ClientRewinder::ClientRewinder(SceneRewinder *p_node) :
		Rewinder(p_node) {
	clear();
}

void ClientRewinder::clear() {
	node_id_map.clear();
	node_paths.clear();
	server_snapshot.controllers_input_id.clear();
	server_snapshot.node_vars.clear();
}

void ClientRewinder::process() {
	const real_t delta = scene_rewinder->get_physics_process_delta_time();
	const real_t iteration_per_second = Engine::get_singleton()->get_iterations_per_second();

	for (
			OAHashMap<ControllerID, IsleData>::Iterator it = scene_rewinder->isle_data.iter();
			it.valid;
			it = scene_rewinder->isle_data.next_iter(it)) {
		NetworkedController *controller = it.value->controller;

		// Due to some lag we may want to speed up the input_packet
		// generation, for this reason here I'm performing a sub tick.
		//
		// keep in mind that we are just pretending that the time
		// is advancing faster, for this reason we are still using
		// `delta` to step the controllers.
		//
		// The dolls may want to speed up too, so to consume the inputs faster
		// and get back in time with the server.
		int sub_ticks = controller->calculates_sub_ticks(
				delta,
				iteration_per_second);

		while (sub_ticks > 0) {
			// Process the nodes of this controller.
			const ObjectID *nodes = it.value->nodes.ptr();
			for (
					int node_i = 0;
					node_i < it.value->nodes.size();
					node_i += 1) {
				NodeData *node_data = scene_rewinder->nodes_data.lookup_ptr(nodes[node_i]);
				ERR_CONTINUE(node_data == nullptr);
				node_data->process(delta);
			}

			// Process the controller
			controller->process(delta);

			// TODO find a way to not iterate this again or avoid the below `look_up`.
			// Iterate all the nodes and compare the isle??
			for (
					int node_i = 0;
					node_i < it.value->nodes.size();
					node_i += 1) {
				NodeData *node_data = scene_rewinder->nodes_data.lookup_ptr(nodes[node_i]);
				ERR_CONTINUE(node_data == nullptr);

				scene_rewinder->pull_node_changes(node_data);
			}

			if (controller->player_has_new_input()) {
				store_snapshot(controller);
			}

			sub_ticks -= 1;
		}
	}

	scene_rewinder->recover_in_progress = true;

	process_controllers_recovery(delta);

	scene_rewinder->recover_in_progress = false;
}

void ClientRewinder::receive_snapshot(Variant p_snapshot) {
	// The received snapshot nodes are stored all separated into the
	// `server_snapshot`. Since the snapshots sends data incrementatlly, storing
	// it in this way simplify the parsing phase.
	// Later, the snapshot nodes are organized into isle. In this way the
	// isle organization can change without too much problems.

	// Parse server snapshot.
	const bool success = parse_snapshot(p_snapshot);

	if (success == false) {
		return;
	}

	// Finalize data.

	store_controllers_snapshot(
			server_snapshot,
			server_controllers_snapshots);
}

void ClientRewinder::store_snapshot(const NetworkedController *p_controller) {
	ERR_FAIL_COND(scene_rewinder->main_controller == nullptr);

	std::deque<IsleSnapshot> *client_snaps = client_controllers_snapshots.lookup_ptr(p_controller->get_instance_id());
	if (client_snaps == nullptr) {
		client_controllers_snapshots.set(p_controller->get_instance_id(), std::deque<IsleSnapshot>());
		client_snaps = client_controllers_snapshots.lookup_ptr(p_controller->get_instance_id());
	}

#ifdef DEBUG_ENABLED
	// Simply unreachable.
	CRASH_COND(client_snaps == nullptr);
#endif

	if (client_snaps->size() > 0) {
		if (p_controller->get_current_input_id() <= client_snaps->back().input_id) {
			NET_DEBUG_WARN("During snapshot creation, for controller " + p_controller->get_path() + ", was found an ID for an older snapshots. New input ID: " + itos(p_controller->get_current_input_id()) + " Last saved snapshot input ID: " + itos(client_snaps->back().input_id) + ". This snapshot is not stored.");
			return;
		}
	}

	IsleData *isle_data = scene_rewinder->isle_data.lookup_ptr(p_controller->get_instance_id());
	ERR_FAIL_COND(isle_data == nullptr);

	client_snaps->push_back(IsleSnapshot());

	IsleSnapshot &snap = client_snaps->back();
	snap.input_id = p_controller->get_current_input_id();

	const ObjectID *nodes = isle_data->nodes.ptr();
	for (
			int node_i = 0;
			node_i < isle_data->nodes.size();
			node_i += 1) {
		const NodeData *node_data = scene_rewinder->nodes_data.lookup_ptr(nodes[node_i]);
		ERR_CONTINUE(node_data == nullptr);

		snap.node_vars.set(node_data->instance_id, node_data->vars);
	}
}

void ClientRewinder::store_controllers_snapshot(
		const Snapshot &p_snapshot,
		OAHashMap<ObjectID, std::deque<IsleSnapshot>> &r_snapshot_storage) {
	// Extract the controllers data from the snapshot and store it in the isle
	// snapshot.
	// The main controller takes with him all world nodes.

	for (
			OAHashMap<ControllerID, IsleData>::Iterator isle_it = scene_rewinder->isle_data.iter();
			isle_it.valid;
			isle_it = scene_rewinder->isle_data.next_iter(isle_it)) {
		const NetworkedController *controller = isle_it.value->controller;

		const uint64_t *input_id = p_snapshot.controllers_input_id.lookup_ptr(controller->get_instance_id());
		if (input_id == nullptr || *input_id == UINT64_MAX) {
			// The snapshot doesn't have any info for this controller; Skip it.
			continue;
		}

		std::deque<IsleSnapshot> *controller_snaps = r_snapshot_storage.lookup_ptr(controller->get_instance_id());
		if (controller_snaps == nullptr) {
			r_snapshot_storage.set(controller->get_instance_id(), std::deque<IsleSnapshot>());
			controller_snaps = r_snapshot_storage.lookup_ptr(controller->get_instance_id());
		}

#ifdef DEBUG_ENABLED
		// Simply unreachable.
		CRASH_COND(controller_snaps == nullptr);
#endif

		if (controller_snaps->empty() == false) {
			// Make sure the snapshots are stored in order.
			const uint64_t last_stored_input_id = controller_snaps->back().input_id;
			ERR_FAIL_COND_MSG((*input_id) <= last_stored_input_id, "This doll snapshot (with ID: " + itos(*input_id) + ") is not expected because the last stored id is: " + itos(last_stored_input_id));
		}

		IsleSnapshot snap;
		snap.input_id = *input_id;

		for (
				OAHashMap<ObjectID, Vector<VarData>>::Iterator it = p_snapshot.node_vars.iter();
				it.valid;
				it = p_snapshot.node_vars.next_iter(it)) {
			const NodeData *node_data = scene_rewinder->nodes_data.lookup_ptr(*it.key);

			if (
					node_data != nullptr &&
					node_data->isle_id == controller->get_instance_id()) {
				// This node is part of this isle.
				snap.node_vars.set(*it.key, *it.value);
			}
		}

		controller_snaps->push_back(snap);
	}
}

void ClientRewinder::process_controllers_recovery(real_t p_delta) {
	// Each controller is responsible for it self and its controlled nodes. This
	// give much more freedom during the recovering & rewinding; and the
	// misalignments are recovered atomically (for better performance and avoid
	// cascading errors).
	//
	// The only exception is the main controller. The client main controller is
	// also responsible for the scene nodes synchronization.
	// The scene may have objects that any Player can interact with, and so
	// change its state.
	// With the above approach, the world scene is always up to day with the
	// client reference frame, so the player always interacts with an up to date
	// version of the node.
	//
	// # Dependency Graph
	// Under some circustances this may not be correct. For example when a doll
	// is too much behind the scene timeline and interacts with the scene.
	//
	// To solve this problem a dependency graph and a dispatcher would be needed,
	// so to check/rewind the nodes with the reference frame of the node it
	// interacts with.
	//
	// While a dependency graph would solve this cases, integrate it would require
	// time, would make much more complex this already complex code, and would
	// introduce some `SceneRewinder` API that would make its usage more difficult.
	//
	// As is now the mechanism should be enough to get a really good result,
	// in allmost all cases. So for now, the dependency graph is not going to
	// happen.
	//
	// # Isle Rewinding
	// Each controller is a separate isle where the recover / rewind is performed.
	// The scene may be really big, and not all the nodes of the scene may
	// require to be recovered / rewinded.
	// This mechanism, establish the nodes that need a recovered, and perform
	// the recovering only on such nodes. Saving so performance.

	for (
			OAHashMap<ControllerID, IsleData>::Iterator isle_it = scene_rewinder->isle_data.iter();
			isle_it.valid;
			isle_it = scene_rewinder->isle_data.next_iter(isle_it)) {
		NetworkedController *controller = isle_it.value->controller;
		bool is_main_controller = controller == scene_rewinder->main_controller;

		// --- Phase one, find snapshot to check. ---

		std::deque<IsleSnapshot> *server_snaps = server_controllers_snapshots.lookup_ptr(controller->get_instance_id());
		if (server_snaps == nullptr || server_snaps->empty()) {
			// No snapshots to recover for this controller. Skip it.
			continue;
		}

		std::deque<IsleSnapshot> *client_snaps = client_controllers_snapshots.lookup_ptr(controller->get_instance_id());

		// Find the best recoverable input_id.
		uint64_t checkable_input_id = UINT64_MAX;
		if (controller->last_known_input() != UINT64_MAX && controller->get_stored_input_id(-1) != UINT64_MAX) {
			int diff = controller->last_known_input() - controller->get_stored_input_id(-1);
			if (diff >= scene_rewinder->get_doll_desync_tolerance()) {
				// This happens to the dolls that may be too behind. Just reset
				// to the newer state possible with a small padding.
				for (
						auto s_snap = server_snaps->rbegin();
						checkable_input_id == UINT64_MAX && s_snap != server_snaps->rend();
						++s_snap) {
					if (s_snap->input_id < controller->last_known_input()) {
						checkable_input_id = s_snap->input_id;
					}
				}
			} else {
				// Find the best snapshot to recover from the one already
				// processed.
				if (client_snaps != nullptr && client_snaps->empty() == false) {
					for (
							auto s_snap = server_snaps->rbegin();
							checkable_input_id == UINT64_MAX && s_snap != server_snaps->rend();
							++s_snap) {
						for (auto c_snap = client_snaps->begin(); c_snap != client_snaps->end(); ++c_snap) {
							if (c_snap->input_id == s_snap->input_id) {
								// This snapshot is present on client, can be checked.
								checkable_input_id = c_snap->input_id;
								break;
							}
						}
					}
				}
			}
		}

		if (checkable_input_id == UINT64_MAX) {
			continue;
		}

#ifdef DEBUG_ENABLED
		// Unreachable cause the above check
		CRASH_COND(server_snaps->empty());
#endif

		while (server_snaps->front().input_id < checkable_input_id) {
			// Drop any older snapshot.
			server_snaps->pop_front();
		}

		// Drop all the client snapshots until the one that we need.
		while (client_snaps != nullptr && client_snaps->empty() == false && client_snaps->front().input_id < checkable_input_id) {
			// Drop any olded snapshot.
			client_snaps->pop_front();
		}

#ifdef DEBUG_ENABLED
		// These are unreachable at this point.
		CRASH_COND(server_snaps->empty());
		CRASH_COND(server_snaps->front().input_id != checkable_input_id);
#endif
		// --- Phase two, check snapshot. ---

		bool need_recover = false;
		bool recover_controller = false;
		std::vector<NodeData *> nodes_to_recover;
		std::vector<PostponedRecover> postponed_recover;

		if (client_snaps == nullptr || client_snaps->empty() || client_snaps->front().input_id != checkable_input_id) {
			// We don't have any snapshot on client for this controller.
			// Just reset all the nodes to the server state.
			NET_DEBUG_PRINT("During recovering was not found any client doll snapshot for this doll: " + controller->get_path() + "; The server snapshot is apllied.");
			need_recover = true;
			recover_controller = true;

			nodes_to_recover.reserve(server_snaps->size());
			for (
					OAHashMap<ObjectID, Vector<VarData>>::Iterator s_snap_it = server_snaps->front().node_vars.iter();
					s_snap_it.valid;
					s_snap_it = server_snaps->front().node_vars.next_iter(s_snap_it)) {
				NodeData *nd = scene_rewinder->nodes_data.lookup_ptr(*s_snap_it.key);
				if (nd == nullptr ||
						nd->controlled_by.is_null() == false ||
						nd->is_controller) {
					// This is a controller or a piece of controller; Skip now, it'll be added later.
					continue;
				}
				nodes_to_recover.push_back(nd);
			}

		} else {
#ifdef DEBUG_ENABLED
			// This is unreachable, because we store all the client shapshots
			// each time a new input is processed. Since the `checkable_input_id`
			// is taken by reading the processed doll inputs, it's guaranteed
			// that here the snapshot exists.
			CRASH_COND(client_snaps->empty());
			CRASH_COND(client_snaps->front().input_id != checkable_input_id);
#endif

			for (
					OAHashMap<ObjectID, Vector<VarData>>::Iterator s_snap_it = server_snaps->front().node_vars.iter();
					s_snap_it.valid;
					s_snap_it = server_snaps->front().node_vars.next_iter(s_snap_it)) {
				NodeData *rew_node_data = scene_rewinder->nodes_data.lookup_ptr(*s_snap_it.key);
				if (rew_node_data == nullptr) {
					continue;
				}

				bool recover_this_node = false;
				if (is_main_controller == false && need_recover) {
					// Shortcut for dolls, that just have controlled nodes; So
					// check all nodes is not needed when a difference is already
					// found.
					recover_this_node = true;
				} else {
					const Vector<VarData> *c_vars = client_snaps->front().node_vars.lookup_ptr(*s_snap_it.key);
					if (c_vars == nullptr) {
						NET_DEBUG_PRINT("Rewind is needed because the client snapshot doesn't contains this node: " + rew_node_data->node->get_path());
						recover_this_node = true;
					} else {
						PostponedRecover rec;

						const bool different = compare_vars(
								rew_node_data,
								*s_snap_it.value,
								*c_vars,
								rec.vars);

						if (different) {
							NET_DEBUG_PRINT("Rewind is needed because the node on client is different: " + rew_node_data->node->get_path());
							recover_this_node = true;
						} else if (rec.vars.size() > 0) {
							rec.node_data = rew_node_data;
							postponed_recover.push_back(rec);
						}
					}
				}

				if (recover_this_node) {
					need_recover = true;
					if (rew_node_data->controlled_by.is_null() == false ||
							rew_node_data->is_controller) {
						// Controller node.
						recover_controller = true;
					} else {
						nodes_to_recover.push_back(rew_node_data);
					}
				}
			}

			// Popout the client snapshot.
			client_snaps->pop_front();
		}

		// --- Phase three, recover and reply. ---

		if (need_recover) {
			if (recover_controller) {
				// Put the controlled and the controllers into the nodes to
				// rewind.
				// Note, the controller stuffs are added here to ensure that if the
				// controller need a recover, all its nodes are added; no matter
				// at which point the difference is found.
				NodeData *nd = scene_rewinder->nodes_data.lookup_ptr(controller->get_instance_id());
				if (nd) {
					nodes_to_recover.reserve(
							nodes_to_recover.size() +
							nd->controlled_nodes.size() +
							1);

					nodes_to_recover.push_back(nd);

					const ObjectID *controlled_nodes = nd->controlled_nodes.ptr();
					for (
							int i = 0;
							i < nd->controlled_nodes.size();
							i += 1) {
						NodeData *node_data = scene_rewinder->nodes_data.lookup_ptr(controlled_nodes[i]);
						if (node_data) {
							nodes_to_recover.push_back(node_data);
						}
					}
				}
			}

			// Apply the server snapshot so to go back in time till that moment,
			// so to be able to correctly reply the movements.
			scene_rewinder->reset_in_progress = true;
			for (
					std::vector<NodeData *>::const_iterator it = nodes_to_recover.begin();
					it != nodes_to_recover.end();
					it += 1) {
				NodeData *rew_node_data = *it;

				VarData *rew_vars = rew_node_data->vars.ptrw();
				Node *node = rew_node_data->node;

				const Vector<VarData> *s_vars = server_snaps->front().node_vars.lookup_ptr(rew_node_data->instance_id);
				if (s_vars == nullptr) {
					NET_DEBUG_WARN("The node: " + rew_node_data->node->get_path() + " was not found on the server snapshot, this is not supposed to happen a lot.");
					continue;
				}
				const VarData *s_vars_ptr = s_vars->ptr();

				NET_DEBUG_PRINT("Full reset node: " + node->get_path());
				for (int i = 0; i < s_vars->size(); i += 1) {
					node->set(s_vars_ptr[i].var.name, s_vars_ptr[i].var.value);

					// Set the value on the rewinder too.
					const int rew_var_index = rew_node_data->vars.find(s_vars_ptr[i].var.name);
					// Unreachable, because when the snapshot is received the
					// algorithm make sure the `scene_rewinder` is traking the
					// variable.
					CRASH_COND(rew_var_index <= -1);

					NET_DEBUG_PRINT(" |- Variable: " + s_vars_ptr[i].var.name + " New value: " + s_vars_ptr[i].var.value);

					rew_vars[rew_var_index].var.value = s_vars_ptr[i].var.value;

					node->emit_signal(
							scene_rewinder->get_changed_event_name(
									s_vars_ptr[i].var.name));
				}
			}
			scene_rewinder->reset_in_progress = false;

			// Rewind phase.

			scene_rewinder->rewinding_in_progress = true;
			const int remaining_inputs =
					controller->notify_input_checked(checkable_input_id);
			if (client_snaps) {
				CRASH_COND(client_snaps->size() != size_t(remaining_inputs));
			} else {
				CRASH_COND(remaining_inputs != 0);
			}

			bool has_next = false;
			for (int i = 0; i < remaining_inputs; i += 1) {
				// Step 1 -- Process the nodes into the scene that need to be
				// processed.
				for (
						std::vector<NodeData *>::iterator it = nodes_to_recover.begin();
						it != nodes_to_recover.end();
						it += 1) {
					(*it)->process(p_delta);
				}

				if (recover_controller) {
					// Step 2 -- Process the controller.
					has_next = controller->process_instant(i, p_delta);
				}

				// Step 3 -- Pull node changes and Update snapshots.
				for (
						std::vector<NodeData *>::iterator it = nodes_to_recover.begin();
						it != nodes_to_recover.end();
						it += 1) {
					NodeData *rew_node_data = *it;

					scene_rewinder->pull_node_changes(rew_node_data);

					// Update client snapshot.
					(*client_snaps)[i].node_vars.set(rew_node_data->instance_id, rew_node_data->vars);
				}
			}

#ifdef DEBUG_ENABLED
			// Unreachable because the above loop consume all instants.
			CRASH_COND(has_next);
#endif

			scene_rewinder->rewinding_in_progress = false;
		} else {
			// Apply found differences without rewind.
			scene_rewinder->reset_in_progress = true;
			for (
					std::vector<PostponedRecover>::const_iterator it = postponed_recover.begin();
					it != postponed_recover.end();
					it += 1) {
				NodeData *rew_node_data = it->node_data;
				Node *node = rew_node_data->node;
				const Var *vars = it->vars.ptr();

				NET_DEBUG_PRINT("[Snapshot partial reset] Node: " + node->get_path());

				for (int v = 0; v < it->vars.size(); v += 1) {
					node->set(vars[v].name, vars[v].value);

					// Set the value on the rewinder too.
					const int rew_var_index = rew_node_data->vars.find(vars[v].name);
					// Unreachable, because when the snapshot is received the
					// algorithm make sure the `scene_rewinder` is traking the
					// variable.
					CRASH_COND(rew_var_index <= -1);

					rew_node_data->vars.write[rew_var_index].var.value = vars[v].value;

					NET_DEBUG_PRINT(" |- Variable: " + vars[v].name + "; value: " + vars[v].value);
					node->emit_signal(scene_rewinder->get_changed_event_name(vars[v].name));
				}

				// Update the last client snapshot.
				if (client_snaps && client_snaps->empty() == false) {
					client_snaps->back().node_vars.set(rew_node_data->instance_id, rew_node_data->vars);
				}
			}
			scene_rewinder->reset_in_progress = false;

			controller->notify_input_checked(checkable_input_id);
		}

		// Popout the server snapshot.
		server_snaps->pop_front();
	}
}

bool ClientRewinder::parse_snapshot(Variant p_snapshot) {
	// The packet data is an array that contains the informations to update the
	// client snapshot.
	//
	// It's composed as follows:
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

	need_full_snapshot_notified = false;

	ERR_FAIL_COND_V_MSG(
			scene_rewinder->main_controller == nullptr,
			false,
			"Is not possible to receive server snapshots if you are not tracking any NetController.");
	ERR_FAIL_COND_V(!p_snapshot.is_array(), false);

	const Vector<Variant> raw_snapshot = p_snapshot;
	const Variant *raw_snapshot_ptr = raw_snapshot.ptr();

	Node *node = nullptr;
	NodeData *rewinder_node_data = nullptr;
	Vector<VarData> *server_snapshot_node_data = nullptr;
	StringName variable_name;
	int server_snap_variable_index = -1;

	uint64_t player_controller_input_id = UINT64_MAX;

	for (int snap_data_index = 0; snap_data_index < raw_snapshot.size(); snap_data_index += 1) {
		const Variant v = raw_snapshot_ptr[snap_data_index];
		if (node == nullptr) {
			// Node is null so we expect `v` has the node info.

			uint32_t node_id(0);

			if (v.is_array()) {
				// Node info are in verbose form, extract it.

				const Vector<Variant> node_data = v;
				ERR_FAIL_COND_V(node_data.size() != 2, false);
				ERR_FAIL_COND_V(node_data[0].get_type() != Variant::INT, false);
				ERR_FAIL_COND_V(node_data[1].get_type() != Variant::NODE_PATH, false);

				node_id = node_data[0];
				const NodePath node_path = node_data[1];

				// Associate the ID with the path.
				node_paths.set(node_id, node_path);

				node = scene_rewinder->get_tree()->get_root()->get_node(node_path);

			} else if (v.get_type() == Variant::INT) {
				// Node info are in short form.

				node_id = v;

				const ObjectID *object_id = node_id_map.lookup_ptr(node_id);
				if (object_id != nullptr) {
					Object *const obj = ObjectDB::get_instance(*object_id);
					node = Object::cast_to<Node>(obj);
					if (node == nullptr) {
						// This node doesn't exist anymore.
						node_id_map.remove(node_id);
					}
				}

				if (node == nullptr) {
					// The node instance for this node ID was not found, try
					// to find it now.

					const NodePath *node_path = node_paths.lookup_ptr(node_id);
					if (node_path == nullptr) {
						NET_DEBUG_PRINT("The node with ID `" + itos(node_id) + "` is not know by this peer, this is not supposed to happen.");
						notify_server_full_snapshot_is_needed();
					} else {
						node = scene_rewinder->get_tree()->get_root()->get_node(*node_path);
					}
				}
			} else {
				// The arrived snapshot does't seems to be in the expected form.
				ERR_FAIL_V_MSG(false, "Snapshot is corrupted.");
			}

			if (node == nullptr) {
				// This node does't exist; skip it entirely.
				for (snap_data_index += 1; snap_data_index < raw_snapshot.size(); snap_data_index += 1) {
					if (raw_snapshot_ptr[snap_data_index].get_type() == Variant::NIL) {
						break;
					}
				}
				continue;

			} else {
				// The node is found, make sure to update the instance ID in
				// case it changed or it doesn't exist.
				node_id_map.set(node_id, node->get_instance_id());
			}

			// Make sure this node is being tracked locally.
			rewinder_node_data = scene_rewinder->nodes_data.lookup_ptr(node->get_instance_id());
			if (rewinder_node_data == nullptr) {
				// This node is not know on this client. Add it.
				const bool is_controller =
						Object::cast_to<NetworkedController>(node) != nullptr;

				scene_rewinder->nodes_data.set(
						node->get_instance_id(),
						NodeData(node_id,
								node->get_instance_id(),
								is_controller));
				rewinder_node_data = scene_rewinder->nodes_data.lookup_ptr(node->get_instance_id());
			}
			// Update the node ID created on the server.
			rewinder_node_data->id = node_id;

			// Make sure this node is part of the server node too.
			server_snapshot_node_data = server_snapshot.node_vars.lookup_ptr(node->get_instance_id());
			if (server_snapshot_node_data == nullptr) {
				server_snapshot.node_vars.set(
						node->get_instance_id(),
						Vector<VarData>());
				server_snapshot_node_data = server_snapshot.node_vars.lookup_ptr(node->get_instance_id());
			}

			if (rewinder_node_data->is_controller) {
				// This is a controller, so the next data is the input ID.
				ERR_FAIL_COND_V(snap_data_index + 1 >= raw_snapshot.size(), false);
				snap_data_index += 1;
				const uint64_t input_id = raw_snapshot_ptr[snap_data_index];
				ERR_FAIL_COND_V_MSG(input_id == UINT64_MAX, false, "The server is always able to send input_id, so this snapshot seems corrupted.");

				server_snapshot.controllers_input_id.set(node->get_instance_id(), input_id);

				if (node == scene_rewinder->main_controller) {
					// This is the main controller, store the ID also in the
					// utility variable.
					player_controller_input_id = input_id;
				}
			}

		} else if (variable_name == StringName()) {
			// When the node is known and the `variable_name` not, we expect a
			// new variable or the end pf this node data.

			if (v.get_type() == Variant::NIL) {
				// NIL found, so this node is done.
				node = nullptr;
				continue;
			}

			// This is a new variable, so let's take the variable name.

			uint32_t var_id;
			if (v.is_array()) {
				// The variable info are stored in verbose mode.

				const Vector<Variant> var_data = v;
				ERR_FAIL_COND_V(var_data.size() != 2, false);
				ERR_FAIL_COND_V(var_data[0].get_type() != Variant::INT, false);
				ERR_FAIL_COND_V(var_data[1].get_type() != Variant::STRING_NAME, false);

				var_id = var_data[0];
				variable_name = var_data[1];

				const int index = rewinder_node_data->vars.find(variable_name);

				if (index == -1) {
					// The variable is not known locally, so just add it so
					// to store the variable ID.
					const bool skip_rewinding = false;
					const bool enabled = false;
					rewinder_node_data->vars
							.push_back(
									VarData(
											var_id,
											variable_name,
											Variant(),
											skip_rewinding,
											enabled));
				} else {
					// The variable is known, just make sure that it has the
					// same server ID.
					rewinder_node_data->vars.write[index].id = var_id;
				}
			} else if (v.get_type() == Variant::INT) {
				// The variable is stored in the compact form.

				var_id = v;

				const int index = rewinder_node_data->find_var_by_id(var_id);
				if (index == -1) {
					NET_DEBUG_PRINT("The var with ID `" + itos(var_id) + "` is not know by this peer, this is not supposed to happen.");

					notify_server_full_snapshot_is_needed();

					// Skip the next data since it should be the value, but we
					// can't store it.
					snap_data_index += 1;
					continue;
				} else {
					variable_name = rewinder_node_data->vars[index].var.name;
					rewinder_node_data->vars.write[index].id = var_id;
				}

			} else {
				ERR_FAIL_V_MSG(false, "The snapshot received seems corrupted.");
			}

			server_snap_variable_index = server_snapshot_node_data->find(variable_name);

			if (server_snap_variable_index == -1) {
				// The server snapshot seems not contains this yet.
				server_snap_variable_index = server_snapshot_node_data->size();

				const bool skip_rewinding = false;
				const bool enabled = true;
				server_snapshot_node_data->push_back(
						VarData(
								var_id,
								variable_name,
								Variant(),
								skip_rewinding,
								enabled));

			} else {
				server_snapshot_node_data->write[server_snap_variable_index].id = var_id;
			}

		} else {
			// The node is known, also the variable name is known, so the value
			// is expected.

			server_snapshot_node_data->write[server_snap_variable_index]
					.var
					.value = v;

			// Just reset the variable name so we can continue iterate.
			variable_name = StringName();
			server_snap_variable_index = -1;
		}
	}

	// We espect that the player_controller is updated by this new snapshot,
	// so make sure it's done so.
	if (player_controller_input_id == UINT64_MAX) {
		NET_DEBUG_PRINT("Recovery aborted, the player controller (" + scene_rewinder->main_controller->get_path() + ") was not part of the received snapshot, probably the server doesn't have important informations for this peer. Snapshot:");
		NET_DEBUG_PRINT(p_snapshot);
		return false;
	} else {
		return true;
	}
}

bool ClientRewinder::compare_vars(
		const NodeData *p_rewinder_node_data,
		const Vector<VarData> &p_server_vars,
		const Vector<VarData> &p_client_vars,
		Vector<Var> &r_postponed_recover) {
	const VarData *s_vars = p_server_vars.ptr();
	const VarData *c_vars = p_client_vars.ptr();

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
			const bool different = !scene_rewinder->rewinder_variant_evaluation(s_vars[s_var_index].var.value, c_vars[c_var_index].var.value);

			if (different) {
				const int index = p_rewinder_node_data->vars.find(s_vars[s_var_index].var.name);
				if (index < 0 || p_rewinder_node_data->vars[index].skip_rewinding == false) {
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

void ClientRewinder::notify_server_full_snapshot_is_needed() {
	if (need_full_snapshot_notified) {
		return;
	}

	// Notify the server that a full snapshot is needed.
	need_full_snapshot_notified = true;
	scene_rewinder->rpc_id(1, "_rpc_notify_need_full_snapshot");
}