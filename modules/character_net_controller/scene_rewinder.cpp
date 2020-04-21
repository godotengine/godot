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

// Don't go below 2 so to take into account internet latency
#define MIN_SNAPSHOTS_SIZE 2

#define MAX_ADDITIONAL_TICK_SPEED 2.0

// 2%
#define TICK_SPEED_CHANGE_NOTIF_THRESHOLD 4

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
		value(p_val),
		enabled(p_enabled) {
}

bool VarData::operator==(const VarData &p_other) const {
	return name == p_other.name;
}

NodeData::NodeData() :
		id(0),
		is_controller(false),
		registered_process_count(-1) {
}

NodeData::NodeData(uint32_t p_id, ObjectID p_instance_id, bool is_controller) :
		id(p_id),
		instance_id(p_instance_id),
		is_controller(is_controller),
		registered_process_count(-1) {
}

int NodeData::find_var_by_id(uint32_t p_id) const {
	if (p_id == 0) {
		return -1;
	}
	const VarData *v = vars.ptr();
	for (int i = 0; i < vars.size(); i += 0) {
		if (v[i].id == p_id) {
			return i;
		}
	}
	return -1;
}

void SceneRewinder::_bind_methods() {

	ClassDB::bind_method(D_METHOD("reset"), &SceneRewinder::reset);
	ClassDB::bind_method(D_METHOD("clear"), &SceneRewinder::clear);

	ClassDB::bind_method(D_METHOD("set_network_traced_frames", "size"), &SceneRewinder::set_network_traced_frames);
	ClassDB::bind_method(D_METHOD("get_network_traced_frames"), &SceneRewinder::get_network_traced_frames);

	ClassDB::bind_method(D_METHOD("set_missing_snapshots_max_tolerance", "tolerance"), &SceneRewinder::set_missing_snapshots_max_tolerance);
	ClassDB::bind_method(D_METHOD("get_missing_snapshots_max_tolerance"), &SceneRewinder::get_missing_snapshots_max_tolerance);

	ClassDB::bind_method(D_METHOD("set_tick_acceleration", "acceleration"), &SceneRewinder::set_tick_acceleration);
	ClassDB::bind_method(D_METHOD("get_tick_acceleration"), &SceneRewinder::get_tick_acceleration);

	ClassDB::bind_method(D_METHOD("set_optimal_size_acceleration", "acceleration"), &SceneRewinder::set_optimal_size_acceleration);
	ClassDB::bind_method(D_METHOD("get_optimal_size_acceleration"), &SceneRewinder::get_optimal_size_acceleration);

	ClassDB::bind_method(D_METHOD("set_server_snapshot_storage_size", "size"), &SceneRewinder::set_server_snapshot_storage_size);
	ClassDB::bind_method(D_METHOD("get_server_snapshot_storage_size"), &SceneRewinder::get_server_snapshot_storage_size);

	ClassDB::bind_method(D_METHOD("set_server_notify_state_interval", "interval"), &SceneRewinder::set_server_notify_state_interval);
	ClassDB::bind_method(D_METHOD("get_server_notify_state_interval"), &SceneRewinder::get_server_notify_state_interval);

	ClassDB::bind_method(D_METHOD("set_comparison_tolerance", "tolerance"), &SceneRewinder::set_comparison_tolerance);
	ClassDB::bind_method(D_METHOD("get_comparison_tolerance"), &SceneRewinder::get_comparison_tolerance);

	ClassDB::bind_method(D_METHOD("register_variable", "node", "variable", "on_change_notify"), &SceneRewinder::register_variable, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("unregister_variable", "node", "variable"), &SceneRewinder::unregister_variable);

	ClassDB::bind_method(D_METHOD("get_changed_event_name", "variable"), &SceneRewinder::get_changed_event_name);

	ClassDB::bind_method(D_METHOD("track_variable_changes", "node", "variable", "method"), &SceneRewinder::track_variable_changes);
	ClassDB::bind_method(D_METHOD("untrack_variable_changes", "node", "variable", "method"), &SceneRewinder::untrack_variable_changes);

	ClassDB::bind_method(D_METHOD("register_process", "node", "function"), &SceneRewinder::register_process);
	ClassDB::bind_method(D_METHOD("unregister_process", "node", "function"), &SceneRewinder::unregister_process);

	ClassDB::bind_method(D_METHOD("is_recovered"), &SceneRewinder::is_recovered);
	ClassDB::bind_method(D_METHOD("is_rewinding"), &SceneRewinder::is_rewinding);

	ClassDB::bind_method(D_METHOD("__clear"), &SceneRewinder::__clear);
	ClassDB::bind_method(D_METHOD("__reset"), &SceneRewinder::__reset);
	ClassDB::bind_method(D_METHOD("_rpc_send_state"), &SceneRewinder::_rpc_send_state);
	ClassDB::bind_method(D_METHOD("_rpc_send_tick_additional_speed"), &SceneRewinder::_rpc_send_tick_additional_speed);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "network_traced_frames", PROPERTY_HINT_RANGE, "100,10000,1"), "set_network_traced_frames", "get_network_traced_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "missing_snapshots_max_tolerance", PROPERTY_HINT_RANGE, "3,50,1"), "set_missing_snapshots_max_tolerance", "get_missing_snapshots_max_tolerance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tick_acceleration", PROPERTY_HINT_RANGE, "0.1,20.0,0.01"), "set_tick_acceleration", "get_tick_acceleration");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "optimal_size_acceleration", PROPERTY_HINT_RANGE, "0.1,20.0,0.01"), "set_optimal_size_acceleration", "get_optimal_size_acceleration");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "server_snapshot_storage_size", PROPERTY_HINT_RANGE, "10,100,1"), "set_server_snapshot_storage_size", "get_server_snapshot_storage_size");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "server_notify_state_interval", PROPERTY_HINT_RANGE, "0.001,10.0,0.0001"), "set_server_notify_state_interval", "get_server_notify_state_interval");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "comparison_tolerance", PROPERTY_HINT_RANGE, "0.000001,0.01,0.000001"), "set_comparison_tolerance", "get_comparison_tolerance");
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
		network_traced_frames(1200),
		missing_snapshots_max_tolerance(4),
		tick_acceleration(2.0),
		optimal_size_acceleration(2.5),
		server_snapshot_storage_size(30),
		server_notify_state_interval(1.0),
		comparison_tolerance(0.0001),
		rewinder(nullptr),
		recover_in_progress(false),
		rewinding_in_progress(false),
		node_counter(1),
		generate_id(false),
		main_controller(nullptr),
		time_bank(0.0),
		tick_additional_speed(0.0) {

	// Adding user signal instead of the traditional signal so it is not visible
	// into the editor and the user is not able to connect it from there.
	// The user is forced to use the function `register_process`.
	add_user_signal(MethodInfo("sync_process", PropertyInfo(Variant::FLOAT, "delta")));

	rpc_config("__reset", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("__clear", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_send_state", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_send_tick_additional_speed", MultiplayerAPI::RPC_MODE_REMOTE);
}

SceneRewinder::~SceneRewinder() {
	if (rewinder) {
		memdelete(rewinder);
		rewinder = nullptr;
	}
}

void SceneRewinder::set_network_traced_frames(int p_size) {
	network_traced_frames = p_size;
}

int SceneRewinder::get_network_traced_frames() const {
	return network_traced_frames;
}

void SceneRewinder::set_missing_snapshots_max_tolerance(int p_tolerance) {
	missing_snapshots_max_tolerance = p_tolerance;
}

int SceneRewinder::get_missing_snapshots_max_tolerance() const {
	return missing_snapshots_max_tolerance;
}

void SceneRewinder::set_tick_acceleration(real_t p_acceleration) {
	tick_acceleration = p_acceleration;
}

real_t SceneRewinder::get_tick_acceleration() const {
	return tick_acceleration;
}

void SceneRewinder::set_optimal_size_acceleration(real_t p_acceleration) {
	optimal_size_acceleration = p_acceleration;
}

real_t SceneRewinder::get_optimal_size_acceleration() const {
	return optimal_size_acceleration;
}

void SceneRewinder::set_server_snapshot_storage_size(int p_size) {
	server_snapshot_storage_size = p_size;
}

int SceneRewinder::get_server_snapshot_storage_size() const {
	return server_snapshot_storage_size;
}

void SceneRewinder::set_server_notify_state_interval(real_t p_interval) {
	server_notify_state_interval = p_interval;
}

real_t SceneRewinder::get_server_notify_state_interval() const {
	return server_notify_state_interval;
}

void SceneRewinder::set_comparison_tolerance(real_t p_tolerance) {
	comparison_tolerance = p_tolerance;
}

real_t SceneRewinder::get_comparison_tolerance() const {
	return comparison_tolerance;
}

void SceneRewinder::register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify) {

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
				true));
	} else {
		node_data->vars.write[id].enabled = true;
	}

	if (p_node->has_signal(get_changed_event_name(p_variable)) == false) {
		p_node->add_user_signal(MethodInfo(
				get_changed_event_name(p_variable)));
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

void SceneRewinder::register_process(Node *p_node, StringName p_function) {
	if (!is_connected("sync_process", Callable(p_node, p_function))) {
		connect("sync_process", Callable(p_node, p_function));

		NodeData *node_data = register_node(p_node);
		ERR_FAIL_COND(node_data == nullptr);
		node_data->registered_process_count += 1;
	}
}

void SceneRewinder::unregister_process(Node *p_node, StringName p_function) {
	if (is_connected("sync_process", Callable(p_node, p_function))) {
		disconnect("sync_process", Callable(p_node, p_function));

		NodeData *node_data = register_node(p_node);
		ERR_FAIL_COND(node_data == nullptr);
		node_data->registered_process_count -= 1;
	}
}

bool SceneRewinder::is_recovered() const {
	return recover_in_progress;
}

bool SceneRewinder::is_rewinding() const {
	return rewinding_in_progress;
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

	set_physics_process_internal(false);
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

void SceneRewinder::_rpc_send_tick_additional_speed(int p_speed) {
	ERR_FAIL_COND(get_tree()->is_network_server() == true);

	tick_additional_speed = (static_cast<real_t>(p_speed) / 100.0) * MAX_ADDITIONAL_TICK_SPEED;
	tick_additional_speed = CLAMP(tick_additional_speed, -MAX_ADDITIONAL_TICK_SPEED, MAX_ADDITIONAL_TICK_SPEED);
	print_line(itos(tick_additional_speed));
}

NodeData *SceneRewinder::register_node(Node *p_node) {
	ERR_FAIL_COND_V(p_node == nullptr, nullptr);

	bool is_controller = false;
	{
		CharacterNetController *controller = Object::cast_to<CharacterNetController>(p_node);
		if (controller) {
			if (controller->has_scene_rewinder()) {
				ERR_FAIL_COND_V_MSG(controller->get_scene_rewinder() != this, nullptr, "This controller is associated with a different scene rewinder.");
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
		const uint32_t node_id(generate_id ? ++node_counter : 0);
		data.set(
				p_node->get_instance_id(),
				NodeData(node_id, p_node->get_instance_id(), is_controller));
		node_data = data.getptr(p_node->get_instance_id());
		node_data->registered_process_count = 0;
	}
	return node_data;
}

bool SceneRewinder::vec2_evaluation(const Vector2 a, const Vector2 b) {
	return (a - b).length_squared() <= (comparison_tolerance * comparison_tolerance);
}

bool SceneRewinder::vec3_evaluation(const Vector3 a, const Vector3 b) {
	return (a - b).length_squared() <= (comparison_tolerance * comparison_tolerance);
}

bool SceneRewinder::rewinder_variant_evaluation(const Variant &v_1, const Variant &v_2) {
	if (v_1.get_type() != v_2.get_type()) {
		return false;
	}

	// Custom evaluation methods
	if (v_1.get_type() == Variant::FLOAT) {
		const real_t a(v_1);
		const real_t b(v_2);
		return ABS(a - b) <= comparison_tolerance;
	} else if (v_1.get_type() == Variant::VECTOR2) {
		return vec2_evaluation(v_1, v_2);
	} else if (v_1.get_type() == Variant::RECT2) {
		const Rect2 a(v_1);
		const Rect2 b(v_2);
		if (vec2_evaluation(a.position, b.position)) {
			if (vec2_evaluation(a.size, b.size)) {
				return true;
			}
		}
		return false;
	} else if (v_1.get_type() == Variant::TRANSFORM2D) {
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
	} else if (v_1.get_type() == Variant::VECTOR3) {
		return vec3_evaluation(v_1, v_2);
	} else if (v_1.get_type() == Variant::QUAT) {
		const Quat a(v_1);
		const Quat b(v_2);
		const Quat r(a - b); // Element wise subtraction.
		return (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w) <= (comparison_tolerance * comparison_tolerance);
	} else if (v_1.get_type() == Variant::PLANE) {
		const Plane a(v_1);
		const Plane b(v_2);
		if (ABS(a.d - b.d) <= comparison_tolerance) {
			if (vec3_evaluation(a.normal, b.normal)) {
				return true;
			}
		}
		return false;
	} else if (v_1.get_type() == Variant::AABB) {
		const AABB a(v_1);
		const AABB b(v_2);
		if (vec3_evaluation(a.position, b.position)) {
			if (vec3_evaluation(a.size, b.size)) {
				return true;
			}
		}
		return false;
	} else if (v_1.get_type() == Variant::BASIS) {
		const Basis a(v_1);
		const Basis b(v_2);
		if (vec3_evaluation(a.elements[0], b.elements[0])) {
			if (vec3_evaluation(a.elements[1], b.elements[1])) {
				if (vec3_evaluation(a.elements[2], b.elements[2])) {
					return true;
				}
			}
		}
		return false;
	} else if (v_1.get_type() == Variant::TRANSFORM) {
		const Transform a(v_1);
		const Transform b(v_2);
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

	// Default evaluation methods
	return v_1 == v_2;
}

void SceneRewinder::process() {

	// Due to some lag we may want to speed up the input_packet
	// generation, for this reason here I'm performing a sub tick.
	//
	// keep in mind that we are just pretending that the time
	// is advancing faster, for this reason we are still using
	// `delta` to step the controllers.

	const real_t delta = get_physics_process_delta_time();
	const real_t pretended_delta = get_pretended_delta();

	time_bank += delta;
	uint32_t sub_ticks = static_cast<uint32_t>(time_bank / pretended_delta);
	time_bank -= static_cast<real_t>(sub_ticks) * pretended_delta;

	bool is_pretended = false;

	// TODO fix the additional time before removing this.
	sub_ticks = 1;

	while (sub_ticks > 0) {

		if (is_pretended == false) {
			// This is a legit iteration, so step all controllers.
			CharacterNetController *const *controllers_raw = controllers.ptr();
			for (int c = 0; c < controllers.size(); c += 1) {
				controllers_raw[c]->process(delta);
			}
		} else {
			// Step only the main controller because we don't want that the dolls
			// are speed up too (This because we don't want to consume client
			// inputs too fast).
			main_controller->process(delta);
		}

		// The world is always processed
		emit_signal("sync_process", delta);

		Vector<ObjectID> null_objects;

		pull_variable_changes(&null_objects);

		// Removes the null objects.
		for (int i = 0; i < null_objects.size(); i += 1) {
			data.erase(null_objects[i]);
		}

		rewinder->process(delta);

		sub_ticks -= 1;
		is_pretended = true;
	}

	rewinder->post_process(delta);
}

real_t SceneRewinder::get_pretended_delta() const {
	return 1.0 / (static_cast<real_t>(Engine::get_singleton()->get_iterations_per_second()) + tick_additional_speed);
}

void SceneRewinder::pull_variable_changes(Vector<ObjectID> *r_null_objects) {
	for (const ObjectID *key = data.next(nullptr); key != nullptr; key = data.next(key)) {

		Node *node = static_cast<Node *>(ObjectDB::get_instance(*key));
		data[*key].cached_node = node;

		if (node == nullptr) {
			if (r_null_objects)
				r_null_objects->push_back(*key);
			continue;
		} else if (node->is_inside_tree() == false) {
			continue;
		}

		const int var_count = data.get(*key).vars.size();
		VarData *object_vars = data.get(*key).vars.ptrw();

		for (int i = 0; i < var_count; i += 1) {
			if (object_vars[i].enabled == false) {
				continue;
			}

			const Variant old_val = object_vars[i].value;
			const Variant new_val = node->get(object_vars[i].name);

			if (!rewinder_variant_evaluation(old_val, new_val)) {
				object_vars[i].value = new_val;
				node->emit_signal(get_changed_event_name(object_vars[i].name));
			}
		}
	}
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
		peer(0),
		optimal_snapshots_size(0.0),
		client_tick_additional_speed(0.0),
		client_tick_additional_speed_compressed(0),
		network_tracer(0) {
}

PeerData::PeerData(int p_peer, int p_traced_frames) :
		peer(p_peer),
		optimal_snapshots_size(0.0),
		client_tick_additional_speed(0.0),
		client_tick_additional_speed_compressed(0),
		network_tracer(p_traced_frames) {
}

bool PeerData::operator==(const PeerData &p_other) const {
	return peer == p_other.peer;
}

void ControllerRewinder::init(
		CharacterNetController *p_controller,
		int p_frames,
		uint64_t p_recovered_snapshot_input_id) {

	recovered_snapshot_input_id = p_recovered_snapshot_input_id;
	const bool rewinding_disabled = recovered_snapshot_input_id == UINT64_MAX;

	controller = p_controller;
	frames_to_skip = p_frames;

	if (rewinding_disabled == false) {
		const int remaining_inputs = controller->forget_input_till(p_recovered_snapshot_input_id);
		finished = remaining_inputs <= 0;
	} else {
		finished = true;
	}
}

void ControllerRewinder::advance(int p_i, real_t p_delta) {
	if (finished || p_i < frames_to_skip) {
		return;
	}

	const bool has_next = controller->replay_process_next_instant(p_i, p_delta);
	if (has_next == false) {
		finished = true;
	}
}

CharacterNetController *ControllerRewinder::get_controller() const {
	return controller;
}

// TODO remove this?
uint64_t ControllerRewinder::get_processed_input_id(int p_i) const {
	const bool rewinding_disabled = recovered_snapshot_input_id == UINT64_MAX;
	if (rewinding_disabled) {
		return UINT64_MAX;
	} else {
		if (p_i >= frames_to_skip) {
			return (p_i - frames_to_skip) + recovered_snapshot_input_id;
		} else {
			const uint64_t frames_to_now = (frames_to_skip - p_i);
			if (frames_to_now > recovered_snapshot_input_id) {
				return recovered_snapshot_input_id - frames_to_now;
			} else {
				return UINT64_MAX;
			}
		}
	}
}

bool ControllerRewinder::has_finished() const {
	return finished;
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

void NoNetRewinder::post_process(real_t _p_delta) {
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
	ERR_FAIL_COND_MSG(peers_data.find(PeerData(p_peer_id, 0)) != -1, "This peer is already connected, is likely a bug.");
	peers_data.push_back(PeerData(p_peer_id, scene_rewinder->get_network_traced_frames()));
}

void ServerRewinder::on_peer_disconnected(int p_peer_id) {
	ERR_FAIL_COND_MSG(peers_data.find(PeerData(p_peer_id, 0)) == -1, "This peer is already disconnected, is likely a bug.");
	peers_data.erase(PeerData(p_peer_id, 0));
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
		if (node_data.cached_node == nullptr || node_data.cached_node->is_inside_tree() == false) {
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
				// The first ID id is not yet arrived, so just put a 0.
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

	adjust_player_tick_rate(p_delta);

	state_notifier_timer += p_delta;
	if (state_notifier_timer >= scene_rewinder->get_server_notify_state_interval()) {
		state_notifier_timer = 0.0;

		Variant snapshot = generate_snapshot();
		scene_rewinder->rpc("_rpc_send_state", snapshot);
	}
}

void ServerRewinder::post_process(real_t p_delta) {
	// Nothing.
}

void ServerRewinder::receive_snapshot(Variant _p_snapshot) {
	// Unreachable
	CRASH_NOW();
}

void ServerRewinder::adjust_player_tick_rate(real_t p_delta) {

	PeerData *peers = peers_data.ptrw();
	CharacterNetController *const *controllers = scene_rewinder->controllers.ptr();

	for (int p = 0; p < peers_data.size(); p += 1) {

		PeerData *peer = peers + p;
		CharacterNetController *controller = nullptr;

		// TODO exist a safe way to not iterate each time?
		for (int c = 0; c < scene_rewinder->controllers.size(); c += 1) {
			if (peer->peer == controllers[c]->get_network_master()) {
				controller = controllers[c];
				break;
			}
		}
		ERR_CONTINUE_MSG(controller == nullptr, "The controller was not found, the controller seems not correctly initialized.");

		if (controller->get_packet_missing()) {
			peer->network_tracer.notify_missing_packet();
		} else {
			peer->network_tracer.notify_packet_arrived();
		}

		const int miss_packets = peer->network_tracer.get_missing_packets();
		const int inputs_count = controller->server_get_inputs_count();

		{
			// The first step to establish the client speed up amount is to define the
			// optimal `frames_inputs` size.
			// This size is increased and decreased using an acceleration, so any speed
			// change is spread across a long period rather a little one.
			const real_t acceleration_level = CLAMP(
					(static_cast<real_t>(miss_packets) -
							static_cast<real_t>(inputs_count)) /
							static_cast<real_t>(scene_rewinder->get_missing_snapshots_max_tolerance()),
					-2.0,
					2.0);
			peer->optimal_snapshots_size += acceleration_level * static_cast<real_t>(scene_rewinder->get_optimal_size_acceleration()) * p_delta;
			peer->optimal_snapshots_size = CLAMP(peer->optimal_snapshots_size, MIN_SNAPSHOTS_SIZE, scene_rewinder->get_server_snapshot_storage_size());
		}

		{
			// The client speed is determined using an acceleration so to have much
			// more control over it, and avoid nervous changes.
			const real_t acceleration_level = CLAMP((peer->optimal_snapshots_size - inputs_count) / scene_rewinder->get_server_snapshot_storage_size(), -1.0, 1.0);
			const real_t acc = acceleration_level * scene_rewinder->get_tick_acceleration() * p_delta;
			const real_t damp = peer->client_tick_additional_speed * -0.9;

			// The damping is fully applyied only if it points in the opposite `acc`
			// direction.
			// I want to cut down the oscilations when the target is the same for a while,
			// but I need to move fast toward new targets when they appear.
			peer->client_tick_additional_speed += acc + damp * ((SGN(acc) * SGN(damp) + 1) / 2.0);
			peer->client_tick_additional_speed = CLAMP(peer->client_tick_additional_speed, -MAX_ADDITIONAL_TICK_SPEED, MAX_ADDITIONAL_TICK_SPEED);

			int new_speed = 100 * (peer->client_tick_additional_speed / MAX_ADDITIONAL_TICK_SPEED);

			if (ABS(peer->client_tick_additional_speed_compressed - new_speed) >= TICK_SPEED_CHANGE_NOTIF_THRESHOLD) {
				peer->client_tick_additional_speed_compressed = new_speed;

				// TODO Send bytes please.
				// TODO consider to send this unreliably each X sec
				scene_rewinder->rpc_id(
						peer->peer,
						"_rpc_send_tick_additional_speed",
						peer->client_tick_additional_speed_compressed);
			}
		}
	}
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

void ClientRewinder::process(real_t _p_delta) {
	store_snapshot();
}

void ClientRewinder::post_process(real_t p_delta) {
	process_recovery(p_delta);
}

void ClientRewinder::receive_snapshot(Variant p_snapshot) {
	parse_snapshot(p_snapshot);
}

void ClientRewinder::store_snapshot() {
	ERR_FAIL_COND_MSG(scene_rewinder->main_controller == nullptr, "Snapshot creation fail, Make sure to track a NetController.");

	// Store snapshots.
	Snapshot snapshot;

	// Store the player controller input ID.
	snapshot.player_controller_input_id = scene_rewinder->main_controller->get_current_input_id();

	// Store the controllers input ID
	const CharacterNetController *const *controllers_ptr = scene_rewinder->controllers.ptr();
	for (int i = 0; i < scene_rewinder->controllers.size(); i += 1) {
		snapshot.controllers_input_id.set(
				controllers_ptr[i]->get_instance_id(),
				controllers_ptr[i]->get_current_input_id());
	}

	// Store the current node data
	snapshot.data = scene_rewinder->data;

	// Archive the snapshot.
	snapshots.push_back(snapshot);
}

void ClientRewinder::update_snapshot(
		int p_snapshot_index,
		ControllerRewinder *p_rewinders,
		int p_rewinder_count) {

	//for (int r = 0; r < p_rewinder_count; r += 1) {
	//	// TODO please check this.
	//	const uint64_t i = p_rewinders[r].get_processed_input_id(p_snapshot_index);
	//	snapshots[p_snapshot_index].controllers_input_id[p_rewinders[r].get_controller()->get_instance_id()] = i;
	//}

	ERR_FAIL_COND(size_t(p_snapshot_index) >= snapshots.size());
	snapshots[p_snapshot_index].data = scene_rewinder->data;
}

void ClientRewinder::process_recovery(real_t p_delta) {
	if (server_snapshot_id <= recovered_snapshot_id) {
		// Nothing to recover.
		return;
	}

	// Drop all the snapshots that are older than the delivered server snapshot.
	while (snapshots.empty() == false && snapshots.front().player_controller_input_id < server_snapshot.player_controller_input_id) {
		snapshots.pop_front();
	}

	if (snapshots.empty()) {
		// Nothing to do.
		return;
	}

	// The next snapshot is the one we are searching for.
	const Snapshot &client_snapshot = snapshots.front();
	ERR_FAIL_COND(client_snapshot.player_controller_input_id != server_snapshot.player_controller_input_id);

	// Client snapshot found.
	scene_rewinder->recover_in_progress = true;

	const bool need_rewind = compare_and_recovery(
			server_snapshot,
			client_snapshot);

	if (need_rewind) {
		scene_rewinder->rewinding_in_progress = true;
		recovery_rewind(server_snapshot, client_snapshot, p_delta);
		scene_rewinder->rewinding_in_progress = false;
	}

	scene_rewinder->recover_in_progress = false;
	recovered_snapshot_id = server_snapshot_id;
	snapshots.pop_front();
}

bool ClientRewinder::compare_and_recovery(
		const Snapshot &p_server_snapshot,
		const Snapshot &p_client_snapshot) {

	// Compare the server snapshot with the client snapshot and apply the server
	// value if the difference found doesn't need a full recovery.
	// Note: the full recovery sets the entire server snapshot, so just return
	// `true` is enough.

	bool need_rewind = false;

	for (
			const ObjectID *key = p_server_snapshot.data.next(nullptr);
			key != nullptr;
			key = p_server_snapshot.data.next(key)) {

		const NodeData &s_data = p_server_snapshot.data.get(*key);
		const VarData *s_vars = s_data.vars.ptr();

		NodeData *rew_node_data = scene_rewinder->data.getptr(s_data.instance_id);
		// Unreachable, because once received the server snapshot, the algorithm
		// make sure that the `scene_rewinder` has the node.
		CRASH_COND(rew_node_data == nullptr);

		Node *node = rew_node_data->cached_node;
		if (node == nullptr) {
			continue;
		}

		VarData *rew_vars = rew_node_data->vars.ptrw();

		// Check if this node is sensible to changes, so a rewind is needed.
		const bool sensible =
				rew_node_data->is_controller ||
				(rew_node_data->registered_process_count > 0);

		const NodeData *c_data = p_client_snapshot.data.getptr(*key);
		if (c_data == nullptr) {
			// The client snapshot at that time was not tracking this node, so
			// it's considered different. Just apply the server values.
			for (int s_var_index = 0; s_var_index < s_data.vars.size(); s_var_index += 1) {
				node->set(s_vars[s_var_index].name, s_vars[s_var_index].value);
				node->emit_signal(scene_rewinder->get_changed_event_name(s_vars[s_var_index].name));

				const int rew_var_index = rew_node_data->vars.find(s_vars[s_var_index].name);
				// Unreachable, because when the snapshot is received the
				// algorithm make sure the `scene_rewinder` is traking the
				// variable.
				CRASH_COND(rew_var_index <= -1);
				rew_vars[rew_var_index].value = s_vars[s_var_index].value;
			}

			if (sensible) {
				need_rewind = true;
			}

		} else {
			// Compare vars.
			bool different = false;
			const VarData *c_vars = c_data->vars.ptr();

			for (int s_var_index = 0; s_var_index < s_data.vars.size(); s_var_index += 1) {
				const int c_var_index = c_data->vars.find(s_vars[s_var_index].name);
				if (c_var_index == -1) {
					// Variable not found, this is considered a difference.
					different = true;
				} else {
					// Variable found compare.
					different = !scene_rewinder->rewinder_variant_evaluation(s_vars[s_var_index].value, c_vars[c_var_index].value);
					if (different) {
						print_line(s_vars[s_var_index].value);
						print_line(c_vars[c_var_index].value);
					}
				}

				if (different) {
					node->set(s_vars[s_var_index].name, s_vars[s_var_index].value);
					node->emit_signal(scene_rewinder->get_changed_event_name(s_vars[s_var_index].name));

					const int rew_var_index = rew_node_data->vars.find(s_vars[s_var_index].name);
					// Unreachable, because when the snapshot is received the
					// algorithm make sure the `scene_rewinder` is traking the
					// variable.
					CRASH_COND(rew_var_index <= -1);
					rew_vars[rew_var_index].value = s_vars[s_var_index].value;

					if (sensible) {
						need_rewind = true;
					}
				}
			}
		}
	}

	return need_rewind;
}

void ClientRewinder::recovery_rewind(const Snapshot &p_server_snapshot, const Snapshot &p_client_snapshot, real_t p_delta) {
	// TODO integrate isle rewinding optimization.

	const int controller_size = scene_rewinder->controllers.size();
	CharacterNetController *const *controllers = scene_rewinder->controllers.ptr();

	// Initialize the rewinders.
	ControllerRewinder *rewinders = memnew_arr(ControllerRewinder, controller_size);
	ControllerRewinder *main_controller_rewinder = nullptr;

	for (int i = 0; i < controller_size; i += 1) {

		int frames_to_skip = 0; // Hard reset by default
		uint64_t rewind_input_id = UINT64_MAX; // Rewind disabled by default.

		const uint64_t *server_input_id = p_server_snapshot
												  .controllers_input_id
												  .getptr(controllers[i]->get_instance_id());

		if (server_input_id != nullptr) {
			const uint64_t siid = *server_input_id;
			rewind_input_id = siid;

			// Check if we can avoid timeline jumps for dolls.
			if (scene_rewinder->main_controller != controllers[i]) {

				const uint64_t *client_input_id = p_client_snapshot
														  .controllers_input_id
														  .getptr(controllers[i]->get_instance_id());

				if (client_input_id != nullptr) {
					const uint64_t ciid = *client_input_id;

					frames_to_skip = siid - ciid;

					// TODO please make this customizable
					// Process the tolerance.
					const int tolerance = 10;
					if (frames_to_skip < 0 || frames_to_skip > tolerance) {
						// Hard reset
						frames_to_skip = 0;
					} else {
						// Fluid timeline navigation allowed.
						// Nothing to do.
					}
				}
			} else {
				main_controller_rewinder = rewinders + i;
			}
		}

		rewinders[i].init(controllers[i], frames_to_skip, rewind_input_id);
	}

	// Unreachable, it's not suppposed to arrive here if the main_rewinder is
	// not set.
	CRASH_COND(main_controller_rewinder == nullptr);

	for (int i = 0; main_controller_rewinder->has_finished() == false; i += 1) {

		// Process the controllers
		for (int c = 0; c < controller_size; c += 1) {
			rewinders[c].advance(i, p_delta);
		}

		// Process the environment
		scene_rewinder->emit_signal("sync_process", p_delta);

		// Update the snapshot.
		scene_rewinder->pull_variable_changes();
		update_snapshot(i + 1, rewinders, controller_size);
	}

#ifdef DEBUG_ENABLED
	for (int c = 0; c < controller_size; c += 1) {
		CRASH_COND(rewinders[c].has_finished() == false);
	}
#endif

	memdelete_arr(rewinders);
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

	// Make sure the Snapshot ID is here.
	ERR_FAIL_COND(raw_snapshot.size() < 1);
	ERR_FAIL_COND(raw_snapshot_ptr[0].get_type() != Variant::INT);

	const uint64_t snapshot_id = raw_snapshot_ptr[0];

	// Make sure this snapshot is expected.
	ERR_FAIL_COND(snapshot_id <= server_snapshot_id);

	server_snapshot_id = snapshot_id;
	// We espect that the player_controller is updated by this new snapshot,
	// at the end we will check if it's 0.
	server_snapshot.player_controller_input_id = 0;

	// Start from 1 to skip the snapshot ID.
	for (int snap_data_index = 1; snap_data_index < raw_snapshot.size(); snap_data_index += 1) {
		const Variant v = raw_snapshot_ptr[snap_data_index];
		if (node == nullptr) {
			// Node is null so we expect `v` has the node info.

			uint32_t node_id(0);

			if (v.is_array()) {
				// Node info are in verbose form, extract it.

				const Vector<Variant> node_data = v;
				ERR_FAIL_COND(node_data.size() != 2);
				ERR_FAIL_COND(node_data[0].get_type() != Variant::INT);
				ERR_FAIL_COND(node_data[1].get_type() != Variant::NODE_PATH);

				node_id = node_data[0];
				const NodePath node_path = node_data[1];

				// Associate the ID with the path.
				node_paths.set(node_id, node_path);

				node = scene_rewinder->get_tree()->get_root()->get_node(node_path);

			} else if (v.get_type() == Variant::INT) {
				// Node info are in short form.

				node_id = v;

				const ObjectID *object_id = node_id_map.getptr(node_id);
				if (object_id != nullptr) {
					Object *const obj = ObjectDB::get_instance(*object_id);
					node = Object::cast_to<Node>(obj);
					if (node == nullptr) {
						// This node doesn't exist anymore.
						node_id_map.erase(node_id);
					}
				}

				if (node == nullptr) {
					// The node instance for this node ID was not found, try
					// to find it now.

					if (node_paths.has(node_id) == false) {
						WARN_PRINT("The node with ID `" + itos(node_id) + "` is not know by this peer, this is not supposed to happen.");
						// TODO notify the server so it sends a full snapshot, and so fix this issue.
					} else {
						const NodePath node_path = node_paths[node_id];
						node = scene_rewinder->get_tree()->get_root()->get_node(node_path);
					}
				}
			} else {
				// The arrived snapshot does't seems to be in the expected form.
				ERR_FAIL_MSG("Snapshot is corrupted.");
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
				// The node is found, make sure to update the instance ID.
				node_id_map.set(node_id, node->get_instance_id());
			}

			const bool is_controller =
					Object::cast_to<CharacterNetController>(node) != nullptr;

			// Make sure this node is being tracked locally.
			rewinder_node_data = scene_rewinder->data.getptr(node->get_instance_id());
			if (rewinder_node_data == nullptr) {
				scene_rewinder->data.set(
						node->get_instance_id(),
						NodeData(node_id,
								node->get_instance_id(),
								is_controller));
				rewinder_node_data = scene_rewinder->data.getptr(node->get_instance_id());
			}
			rewinder_node_data->id = node_id;

			// Make sure this node is part of the server node.
			server_snapshot_node_data = server_snapshot.data.getptr(node->get_instance_id());
			if (server_snapshot_node_data == nullptr) {
				server_snapshot.data.set(
						node->get_instance_id(),
						NodeData(node_id,
								node->get_instance_id(),
								is_controller));
				server_snapshot_node_data = server_snapshot.data.getptr(node->get_instance_id());
			}
			server_snapshot_node_data->id = node_id;

			if (is_controller) {
				// This is a controller, so the next data is the input ID.
				ERR_FAIL_COND(snap_data_index + 1 >= raw_snapshot.size());
				snap_data_index += 1;
				const uint64_t input_id = raw_snapshot_ptr[snap_data_index];
				server_snapshot.controllers_input_id[node->get_instance_id()] = input_id;

				if (node == scene_rewinder->main_controller) {
					// This is the main controller, store the ID also in the
					// utility variable.
					server_snapshot.player_controller_input_id = input_id;
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
				ERR_FAIL_COND(var_data.size() != 2);
				ERR_FAIL_COND(var_data[0].get_type() != Variant::INT);
				ERR_FAIL_COND(var_data[1].get_type() != Variant::STRING_NAME);

				var_id = var_data[0];
				variable_name = var_data[1];

				const int index = rewinder_node_data->vars.find(variable_name);

				if (index == -1) {
					// The variable is not known locally, so just add it so
					// to store the variable ID.
					const bool enabled = false;
					rewinder_node_data->vars
							.push_back(
									VarData(
											var_id,
											variable_name,
											Variant(),
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
					WARN_PRINT("The var with ID `" + itos(var_id) + "` is not know by this peer, this is not supposed to happen.");

					// TODO please notify the server that this peer need a full snapshot.

					// Skip the next data since it should be the value, but we
					// can't store it.
					snap_data_index += 1;
					continue;
				} else {
					variable_name = rewinder_node_data->vars[index].name;
					rewinder_node_data->vars.write[index].id = var_id;
				}

			} else {
				ERR_FAIL_MSG("The snapshot received seems corrupted.");
			}

			server_snap_variable_index = server_snapshot_node_data->vars
												 .find(variable_name);

			if (server_snap_variable_index == -1) {
				// The server snapshot seems not contains this yet.
				server_snap_variable_index = server_snapshot_node_data->vars.size();

				const bool enabled = true;
				server_snapshot_node_data->vars
						.push_back(
								VarData(
										var_id,
										variable_name,
										Variant(),
										enabled));

			} else {
				server_snapshot_node_data->vars
						.write[server_snap_variable_index]
						.id = var_id;
			}

		} else {
			// The node is known, also the variable name is known, so the value
			// is expected.

			server_snapshot_node_data->vars
					.write[server_snap_variable_index]
					.value = v;

			// Just reset the variable name so we can continue iterate.
			variable_name = StringName();
			server_snap_variable_index = -1;
		}
	}

	// Just make sure that the local player input got set as expected.
	ERR_FAIL_COND_MSG(server_snapshot.player_controller_input_id == 0, "The player controller ID was not part of the received snapshot. It will not be considered.");
}
