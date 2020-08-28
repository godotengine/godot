/*************************************************************************/
/*  networked_controller.cpp                                             */
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

#include "networked_controller.h"

#include "core/engine.h"
#include "core/io/marshalls.h"
#include "scene_synchronizer.h"
#include <stdint.h>
#include <algorithm>

// Don't go below 2 so to take into account internet latency
#define MIN_SNAPSHOTS_SIZE 2.0

#define MAX_ADDITIONAL_TICK_SPEED 2.0

// 2%
#define TICK_SPEED_CHANGE_NOTIF_THRESHOLD 4

void NetworkedController::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_player_input_storage_size", "size"), &NetworkedController::set_player_input_storage_size);
	ClassDB::bind_method(D_METHOD("get_player_input_storage_size"), &NetworkedController::get_player_input_storage_size);

	ClassDB::bind_method(D_METHOD("set_max_redundant_inputs", "max_redundand_inputs"), &NetworkedController::set_max_redundant_inputs);
	ClassDB::bind_method(D_METHOD("get_max_redundant_inputs"), &NetworkedController::get_max_redundant_inputs);

	ClassDB::bind_method(D_METHOD("set_tick_speedup_notification_delay", "tick_speedup_notification_delay"), &NetworkedController::set_tick_speedup_notification_delay);
	ClassDB::bind_method(D_METHOD("get_tick_speedup_notification_delay"), &NetworkedController::get_tick_speedup_notification_delay);

	ClassDB::bind_method(D_METHOD("set_network_traced_frames", "size"), &NetworkedController::set_network_traced_frames);
	ClassDB::bind_method(D_METHOD("get_network_traced_frames"), &NetworkedController::get_network_traced_frames);

	ClassDB::bind_method(D_METHOD("set_missing_snapshots_max_tolerance", "tolerance"), &NetworkedController::set_missing_snapshots_max_tolerance);
	ClassDB::bind_method(D_METHOD("get_missing_snapshots_max_tolerance"), &NetworkedController::get_missing_snapshots_max_tolerance);

	ClassDB::bind_method(D_METHOD("set_tick_acceleration", "acceleration"), &NetworkedController::set_tick_acceleration);
	ClassDB::bind_method(D_METHOD("get_tick_acceleration"), &NetworkedController::get_tick_acceleration);

	ClassDB::bind_method(D_METHOD("set_optimal_size_acceleration", "acceleration"), &NetworkedController::set_optimal_size_acceleration);
	ClassDB::bind_method(D_METHOD("get_optimal_size_acceleration"), &NetworkedController::get_optimal_size_acceleration);

	ClassDB::bind_method(D_METHOD("set_server_input_storage_size", "size"), &NetworkedController::set_server_input_storage_size);
	ClassDB::bind_method(D_METHOD("get_server_input_storage_size"), &NetworkedController::get_server_input_storage_size);

	ClassDB::bind_method(D_METHOD("get_current_input_id"), &NetworkedController::get_current_input_id);

	ClassDB::bind_method(D_METHOD("mark_epoch_as_important"), &NetworkedController::mark_epoch_as_important);

	ClassDB::bind_method(D_METHOD("set_doll_peer_active", "peer_id", "active"), &NetworkedController::set_doll_peer_active);
	ClassDB::bind_method(D_METHOD("_on_peer_connection_change", "peer_id"), &NetworkedController::_on_peer_connection_change);

	ClassDB::bind_method(D_METHOD("_rpc_server_send_inputs"), &NetworkedController::_rpc_server_send_inputs);
	ClassDB::bind_method(D_METHOD("_rpc_send_tick_additional_speed"), &NetworkedController::_rpc_send_tick_additional_speed);
	ClassDB::bind_method(D_METHOD("_rpc_doll_notify_connection_status"), &NetworkedController::_rpc_doll_notify_connection_status);
	ClassDB::bind_method(D_METHOD("_rpc_doll_send_epoch"), &NetworkedController::_rpc_doll_send_epoch);

	ClassDB::bind_method(D_METHOD("is_server_controller"), &NetworkedController::is_server_controller);
	ClassDB::bind_method(D_METHOD("is_player_controller"), &NetworkedController::is_player_controller);
	ClassDB::bind_method(D_METHOD("is_doll_controller"), &NetworkedController::is_doll_controller);
	ClassDB::bind_method(D_METHOD("is_nonet_controller"), &NetworkedController::is_nonet_controller);

	BIND_VMETHOD(MethodInfo("collect_inputs", PropertyInfo(Variant::FLOAT, "delta"), PropertyInfo(Variant::OBJECT, "buffer", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer")));
	BIND_VMETHOD(MethodInfo("controller_process", PropertyInfo(Variant::FLOAT, "delta"), PropertyInfo(Variant::OBJECT, "buffer", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "are_inputs_different", PropertyInfo(Variant::OBJECT, "inputs_A", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer"), PropertyInfo(Variant::OBJECT, "inputs_B", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "count_input_size", PropertyInfo(Variant::OBJECT, "inputs", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer")));
	BIND_VMETHOD(MethodInfo("collect_epoch_data", PropertyInfo(Variant::OBJECT, "buffer", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer")));
	BIND_VMETHOD(MethodInfo("setup_interpolator", PropertyInfo(Variant::OBJECT, "interpolator", PROPERTY_HINT_RESOURCE_TYPE, "Interpolator")));
	BIND_VMETHOD(MethodInfo("parse_epoch_data", PropertyInfo(Variant::OBJECT, "interpolator", PROPERTY_HINT_RESOURCE_TYPE, "Interpolator"), PropertyInfo(Variant::OBJECT, "buffer", PROPERTY_HINT_RESOURCE_TYPE, "DataBuffer")));
	BIND_VMETHOD(MethodInfo("epoch_process", PropertyInfo(Variant::FLOAT, "delta"), PropertyInfo(Variant::ARRAY, "interpolated_data")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "input_storage_size", PROPERTY_HINT_RANGE, "100,2000,1"), "set_player_input_storage_size", "get_player_input_storage_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_redundant_inputs", PROPERTY_HINT_RANGE, "0,1000,1"), "set_max_redundant_inputs", "get_max_redundant_inputs");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tick_speedup_notification_delay", PROPERTY_HINT_RANGE, "0.001,2.0,0.001"), "set_tick_speedup_notification_delay", "get_tick_speedup_notification_delay");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "network_traced_frames", PROPERTY_HINT_RANGE, "100,10000,1"), "set_network_traced_frames", "get_network_traced_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "missing_snapshots_max_tolerance", PROPERTY_HINT_RANGE, "3,50,1"), "set_missing_snapshots_max_tolerance", "get_missing_snapshots_max_tolerance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tick_acceleration", PROPERTY_HINT_RANGE, "0.1,20.0,0.01"), "set_tick_acceleration", "get_tick_acceleration");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "optimal_size_acceleration", PROPERTY_HINT_RANGE, "0.1,20.0,0.01"), "set_optimal_size_acceleration", "get_optimal_size_acceleration");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "server_input_storage_size", PROPERTY_HINT_RANGE, "10,100,1"), "set_server_input_storage_size", "get_server_input_storage_size");

	ADD_SIGNAL(MethodInfo("doll_server_comunication_opened"));
	ADD_SIGNAL(MethodInfo("doll_server_comunication_closed"));
}

NetworkedController::NetworkedController() {
	rpc_config("_rpc_server_send_inputs", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_send_tick_additional_speed", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_doll_notify_connection_status", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_doll_send_epoch", MultiplayerAPI::RPC_MODE_REMOTE);
}

void NetworkedController::set_player_input_storage_size(int p_size) {
	player_input_storage_size = p_size;
}

int NetworkedController::get_player_input_storage_size() const {
	return player_input_storage_size;
}

void NetworkedController::set_max_redundant_inputs(int p_max) {
	max_redundant_inputs = p_max;
}

int NetworkedController::get_max_redundant_inputs() const {
	return max_redundant_inputs;
}

void NetworkedController::set_tick_speedup_notification_delay(real_t p_delay) {
	tick_speedup_notification_delay = p_delay;
}

real_t NetworkedController::get_tick_speedup_notification_delay() const {
	return tick_speedup_notification_delay;
}

void NetworkedController::set_network_traced_frames(int p_size) {
	network_traced_frames = p_size;
}

int NetworkedController::get_network_traced_frames() const {
	return network_traced_frames;
}

void NetworkedController::set_missing_snapshots_max_tolerance(int p_tolerance) {
	missing_input_max_tolerance = p_tolerance;
}

int NetworkedController::get_missing_snapshots_max_tolerance() const {
	return missing_input_max_tolerance;
}

void NetworkedController::set_tick_acceleration(real_t p_acceleration) {
	tick_acceleration = p_acceleration;
}

real_t NetworkedController::get_tick_acceleration() const {
	return tick_acceleration;
}

void NetworkedController::set_optimal_size_acceleration(real_t p_acceleration) {
	optimal_size_acceleration = p_acceleration;
}

real_t NetworkedController::get_optimal_size_acceleration() const {
	return optimal_size_acceleration;
}

void NetworkedController::set_server_input_storage_size(int p_size) {
	server_input_storage_size = p_size;
}

int NetworkedController::get_server_input_storage_size() const {
	return server_input_storage_size;
}

uint64_t NetworkedController::get_current_input_id() const {
	return controller->get_current_input_id();
}

void NetworkedController::mark_epoch_as_important() {
	ERR_FAIL_COND_MSG(is_server_controller() == false, "This function must be called only within the function `collect_epoch_data`.");
	static_cast<ServerController *>(controller)->is_epoch_important = true;
}

void NetworkedController::set_doll_peer_active(int p_peer_id, bool p_active) {
	ERR_FAIL_COND_MSG(is_server_controller() == false, "You can set doll activation only on server");
	ERR_FAIL_COND_MSG(p_peer_id == get_network_master(), "This `peer_id` is equal to the Master `peer_id`, which is not allowed.");

	const int index = disabled_doll_peers.find(p_peer_id);
	if (p_active) {
		if (index >= 0) {
			disabled_doll_peers.remove(index);
			update_active_doll_peers();
			rpc_id(p_peer_id, "_rpc_doll_notify_connection_status", true);
		}
	} else {
		if (index == -1) {
			disabled_doll_peers.push_back(p_peer_id);
			update_active_doll_peers();
			rpc_id(p_peer_id, "_rpc_doll_notify_connection_status", false);
		}
	}
}

const LocalVector<int> &NetworkedController::get_active_doll_peers() const {
	return active_doll_peers;
}

void NetworkedController::_on_peer_connection_change(int p_peer_id) {
	update_active_doll_peers();
}

void NetworkedController::update_active_doll_peers() {
	// Unreachable
	CRASH_COND(get_tree()->is_network_server() == false);
	active_doll_peers.clear();
	const Vector<int> peers = get_tree()->get_network_connected_peers();
	for (int i = 0; i < peers.size(); i += 1) {
		const int peer_id = peers[i];
		if (peer_id != get_network_master() && disabled_doll_peers.find(peer_id) == -1) {
			active_doll_peers.push_back(peer_id);
		}
	}
}

bool NetworkedController::process_instant(int p_i, real_t p_delta) {
	ERR_FAIL_COND_V_MSG(is_player_controller() == false, false, "Can be executed only on player controllers.");
	return static_cast<PlayerController *>(controller)->process_instant(p_i, p_delta);
}

ServerController *NetworkedController::get_server_controller() const {
	ERR_FAIL_COND_V_MSG(is_server_controller() == false, nullptr, "This controller is not a server controller.");
	return static_cast<ServerController *>(controller);
}

PlayerController *NetworkedController::get_player_controller() const {
	ERR_FAIL_COND_V_MSG(is_player_controller() == false, nullptr, "This controller is not a player controller.");
	return static_cast<PlayerController *>(controller);
}

DollController *NetworkedController::get_doll_controller() const {
	ERR_FAIL_COND_V_MSG(is_doll_controller() == false, nullptr, "This controller is not a doll controller.");
	return static_cast<DollController *>(controller);
}

NoNetController *NetworkedController::get_nonet_controller() const {
	ERR_FAIL_COND_V_MSG(is_nonet_controller() == false, nullptr, "This controller is not a no net controller.");
	return static_cast<NoNetController *>(controller);
}

bool NetworkedController::is_server_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller_type != CONTROLLER_TYPE_NULL)
		return controller_type == CONTROLLER_TYPE_SERVER;
	return get_tree()->is_network_server();
}

bool NetworkedController::is_player_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller_type != CONTROLLER_TYPE_NULL)
		return controller_type == CONTROLLER_TYPE_PLAYER;
	return get_tree()->is_network_server() == false && is_network_master();
}

bool NetworkedController::is_doll_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller_type != CONTROLLER_TYPE_NULL)
		return controller_type == CONTROLLER_TYPE_DOLL;
	return get_tree()->is_network_server() == false && is_network_master() == false;
}

bool NetworkedController::is_nonet_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller_type != CONTROLLER_TYPE_NULL)
		return controller_type == CONTROLLER_TYPE_NONETWORK;
	return get_tree()->get_network_peer().is_null();
}

void NetworkedController::set_inputs_buffer(const BitArray &p_new_buffer) {
	inputs_buffer.get_buffer_mut().get_bytes_mut() = p_new_buffer.get_bytes();
}

void NetworkedController::set_scene_synchronizer(SceneSynchronizer *p_synchronizer) {
	scene_synchronizer = p_synchronizer;
}

SceneSynchronizer *NetworkedController::get_scene_synchronizer() const {
	return scene_synchronizer;
}

bool NetworkedController::has_scene_synchronizer() const {
	return scene_synchronizer;
}

void NetworkedController::_rpc_server_send_inputs(Vector<uint8_t> p_data) {
	ERR_FAIL_COND(is_server_controller() == false);
	static_cast<ServerController *>(controller)->receive_inputs(p_data);
}

void NetworkedController::_rpc_send_tick_additional_speed(Vector<uint8_t> p_data) {
	ERR_FAIL_COND(is_player_controller() == false);
	ERR_FAIL_COND(p_data.size() != 1);

	const uint8_t speed = p_data[0];
	const real_t additional_speed = MAX_ADDITIONAL_TICK_SPEED * (((static_cast<real_t>(speed) / static_cast<real_t>(UINT8_MAX)) - 0.5) / 0.5);

	PlayerController *player_controller = static_cast<PlayerController *>(controller);
	player_controller->tick_additional_speed = CLAMP(additional_speed, -MAX_ADDITIONAL_TICK_SPEED, MAX_ADDITIONAL_TICK_SPEED);
}

void NetworkedController::_rpc_doll_notify_connection_status(bool p_open) {
	ERR_FAIL_COND_MSG(is_doll_controller() == false, "Only dolls are supposed to receive this function call");

#warning TODO emit a signal.
}

void NetworkedController::_rpc_doll_send_epoch(uint64_t p_epoch, Vector<uint8_t> p_data) {
	ERR_FAIL_COND_MSG(is_doll_controller() == false, "Only dolls are supposed to receive this function call");

	static_cast<DollController *>(controller)->receive_epoch(p_epoch, p_data);
}

void NetworkedController::player_set_has_new_input(bool p_has) {
	has_player_new_input = p_has;
}

bool NetworkedController::player_has_new_input() const {
	return has_player_new_input;
}

void NetworkedController::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: { // TODO consider use the process instead.
			if (Engine::get_singleton()->is_editor_hint())
				return;

			// This can't happen, since only the doll are processed here.
			CRASH_COND(is_doll_controller() == false);
			static_cast<DollController *>(controller)->process(get_physics_process_delta_time());

		} break;
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			// Unreachable.
			CRASH_COND(get_tree() == NULL);

			if (get_tree()->get_network_peer().is_null()) {
				controller_type = CONTROLLER_TYPE_NONETWORK;
				controller = memnew(NoNetController(this));
			} else if (get_tree()->is_network_server()) {
				controller_type = CONTROLLER_TYPE_SERVER;
				controller = memnew(ServerController(this, get_network_traced_frames()));
				get_multiplayer()->connect("network_peer_connected", Callable(this, "_on_peer_connection_change"));
				get_multiplayer()->connect("network_peer_disconnected", Callable(this, "_on_peer_connection_change"));
				update_active_doll_peers();
			} else if (is_network_master()) {
				controller_type = CONTROLLER_TYPE_PLAYER;
				controller = memnew(PlayerController(this));
			} else {
				controller_type = CONTROLLER_TYPE_DOLL;
				controller = memnew(DollController(this));
			}

			ERR_FAIL_COND_MSG(has_method("collect_inputs") == false, "In your script you must inherit the virtual method `collect_inputs` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("controller_process") == false, "In your script you must inherit the virtual method `controller_process` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("are_inputs_different") == false, "In your script you must inherit the virtual method `are_inputs_different` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("count_input_size") == false, "In your script you must inherit the virtual method `count_input_size` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("collect_epoch_data") == false, "In your script you must inherit the virtual method `collect_epoch_data` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("setup_interpolator") == false, "In your script you must inherit the virtual method `setup_interpolator` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("parse_epoch_data") == false, "In your script you must inherit the virtual method `parse_epoch_data` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("epoch_process") == false, "In your script you must inherit the virtual method `epoch_process` to correctly use the `NetworkedController`.");

			controller->ready();

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			memdelete(controller);
			controller = NULL;
			controller_type = CONTROLLER_TYPE_NULL;

			if (get_tree()->is_network_server()) {
				get_multiplayer()->disconnect("network_peer_connected", Callable(this, "_on_peer_connection_change"));
				get_multiplayer()->disconnect("network_peer_disconnected", Callable(this, "_on_peer_connection_change"));
			}
		} break;
	}
}

ServerController::ServerController(
		NetworkedController *p_node,
		int p_traced_frames) :
		Controller(p_node),
		network_tracer(p_traced_frames) {
}

void ServerController::process(real_t p_delta) {
	fetch_next_input();

	if (unlikely(current_input_buffer_id == UINT64_MAX)) {
		// Skip this until the first input arrive.
		return;
	}

	node->get_inputs_buffer_mut().begin_read();
	node->call("controller_process", p_delta, &node->get_inputs_buffer_mut());

	doll_sync(p_delta);

	calculates_player_tick_rate(p_delta);
	adjust_player_tick_rate(p_delta);
}

bool is_remote_frame_A_older(const FrameSnapshotSkinny &p_snap_a, const FrameSnapshotSkinny &p_snap_b) {
	return p_snap_a.id < p_snap_b.id;
}

uint64_t ServerController::last_known_input() const {
	if (snapshots.size() > 0) {
		return snapshots.back().id;
	} else {
		return UINT64_MAX;
	}
}

uint64_t ServerController::get_current_input_id() const {
	return current_input_buffer_id;
}

void ServerController::receive_inputs(Vector<uint8_t> p_data) {
	// The packet is composed as follow:
	// - The following four bytes for the first input ID.
	// - Array of inputs:
	// |-- First byte the amount of times this input is duplicated in the packet.
	// |-- inputs buffer.
	//
	// Let's decode it!

	const int data_len = p_data.size();

	int ofs = 0;

	ERR_FAIL_COND(data_len < 4);
	const uint32_t first_input_id = decode_uint32(p_data.ptr() + ofs);
	ofs += 4;

	uint64_t inserted_input_count = 0;

	// Contains the entire packet and in turn it will be seek to specific location
	// so I will not need to copy chunk of the packet data.
	DataBuffer pir;
	pir.get_buffer_mut().get_bytes_mut() = p_data;
	// TODO this is for 3.2
	//pir.get_buffer_mut().resize_in_bytes(data_len);
	//copymem(pir.get_buffer_mut().get_bytes_mut().ptrw(), p_data.ptr(), data_len);

	while (ofs < data_len) {
		ERR_FAIL_COND_MSG(ofs + 1 > data_len, "The arrived packet size doesn't meet the expected size.");
		// First byte is used for the duplication count.
		const uint8_t duplication = p_data[ofs];
		ofs += 1;

		// Validate input
		pir.seek(ofs * 8);
		const int input_size_in_bits = node->call("count_input_size", &pir);
		// Pad to 8 bits.
		const int input_size =
				Math::ceil((static_cast<float>(input_size_in_bits)) / 8.0);
		ERR_FAIL_COND_MSG(ofs + input_size > data_len, "The arrived packet size doesn't meet the expected size.");

		// The input is valid, populate the buffer.
		for (int sub = 0; sub <= duplication; sub += 1) {
			const uint64_t input_id = first_input_id + inserted_input_count;
			inserted_input_count += 1;

			if (current_input_buffer_id != UINT64_MAX && current_input_buffer_id >= input_id)
				continue;

			FrameSnapshotSkinny rfs;
			rfs.id = input_id;

			const bool found = std::binary_search(
					snapshots.begin(),
					snapshots.end(),
					rfs,
					is_remote_frame_A_older);

			if (!found) {
				rfs.inputs_buffer.get_bytes_mut().resize(input_size);
				copymem(
						rfs.inputs_buffer.get_bytes_mut().ptrw(),
						p_data.ptr() + ofs,
						input_size);

				snapshots.push_back(rfs);

				// Sort the new inserted snapshot.
				std::sort(snapshots.begin(), snapshots.end(), is_remote_frame_A_older);
			}
		}

		// We can now advance the offset.
		ofs += input_size;
	}

	ERR_FAIL_COND_MSG(ofs != data_len, "At the end was detected that the arrived packet has an unexpected size.");
}

int ServerController::get_inputs_count() const {
	return snapshots.size();
}

bool ServerController::fetch_next_input() {
	bool is_new_input = true;
	bool is_packet_missing = false;

	if (unlikely(current_input_buffer_id == UINT64_MAX)) {
		// As initial packet, anything is good.
		if (snapshots.empty() == false) {
			// First input arrived.
			node->set_inputs_buffer(snapshots.front().inputs_buffer);
			current_input_buffer_id = snapshots.front().id;
			snapshots.pop_front();
			// Start traking the packets from this moment on
			network_tracer.reset();
		} else {
			is_new_input = false;
		}
		// Don't notify about missed packets until at least one packet has arrived.
		is_packet_missing = false;
	} else {
		// Search the next packet, the cycle is used to make sure to not stop
		// with older packets arrived too late.

		const uint64_t next_input_id = current_input_buffer_id + 1;

		if (unlikely(snapshots.empty() == true)) {
			// The input buffer is empty!
			is_new_input = false;
			is_packet_missing = true;
			ghost_input_count += 1;
			NET_DEBUG_PRINT("Input buffer is void, i'm using the previous one!");

		} else {
			// The input buffer is not empty, search the new input.
			if (next_input_id == snapshots.front().id) {
				// Wow, the next input is perfect!

				node->set_inputs_buffer(snapshots.front().inputs_buffer);
				current_input_buffer_id = snapshots.front().id;
				snapshots.pop_front();

				ghost_input_count = 0;
				is_packet_missing = false;
			} else {
				// The next packet is not here. This can happen when:
				// - The packet is lost or not yet arrived.
				// - The client for any reason desync with the server.
				//
				// In this cases, the server has the hard task to re-sync.
				//
				// # What it does, then?
				// Initially it see that only 1 packet is missing so it just use
				// the previous one and increase `ghost_inputs_count` to 1.
				//
				// The next iteration, if the packet is not yet arrived the
				// server trys to take the next packet with the `id` less or
				// equal to `next_packet_id + ghost_packet_id`.
				//
				// As you can see the server doesn't lose immediately the hope
				// to find the missing packets, but at the same time deals with
				// it so increases its search pool per each iteration.
				//
				// # Wise input search.
				// Let's consider the case when a set of inputs arrive at the
				// same time, while the server is struggling for the missing packets.
				//
				// In the meanwhile that the packets were chilling on the net,
				// the server were simulating by guessing on their data; this
				// mean that they don't have any longer room to be simulated
				// when they arrive, and the right thing would be just forget
				// about these.
				//
				// The thing is that these can still contain meaningful data, so
				// instead to jump directly to the newest we restart the inputs
				// from the next important packet.
				//
				// For this reason we keep track the amount of missing packets
				// using `ghost_input_count`.

				is_packet_missing = true;
				ghost_input_count += 1;

				const int size = MIN(ghost_input_count, snapshots.size());
				const uint64_t ghost_packet_id = next_input_id + ghost_input_count;

				bool recovered = false;
				FrameSnapshotSkinny pi;

				DataBuffer pir_A = node->get_inputs_buffer();

				for (int i = 0; i < size; i += 1) {
					if (ghost_packet_id < snapshots.front().id) {
						break;
					} else {
						pi = snapshots.front();
						snapshots.pop_front();
						recovered = true;

						// If this input has some important changes compared to the last
						// good input, let's recover to this point otherwise skip it
						// until the last one.
						// Useful to avoid that the server stay too much behind the
						// client.

						DataBuffer pir_B(pi.inputs_buffer);

						pir_A.begin_read();
						pir_B.begin_read();

						const bool is_meaningful = pir_A.get_buffer_size() != pir_B.get_buffer_size() || node->call("are_inputs_different", &pir_A, &pir_B);

						if (is_meaningful) {
							break;
						}
					}
				}

				if (recovered) {
					node->set_inputs_buffer(pi.inputs_buffer);
					current_input_buffer_id = pi.id;
					ghost_input_count = 0;
					NET_DEBUG_PRINT("Packet recovered");
				} else {
					is_new_input = false;
					NET_DEBUG_PRINT("Packet still missing");
				}
			}
		}
	}

	if (is_packet_missing) {
		network_tracer.notify_missing_packet();
	} else {
		network_tracer.notify_packet_arrived();
	}

	return is_new_input;
}

void ServerController::doll_sync(real_t p_delta) {
	// Epoch advances anyway.
	epoch += 1;

	// TODO this should not happen each frame.
	// TODO consider to also use is_epoch_important to estabish if report the packet.
	epoch_state_data.begin_write();
	node->call("collect_epoch_data", &epoch_state_data);
	epoch_state_data.dry();

	// Sent the collected data.
	const LocalVector<int> &peers = node->get_active_doll_peers();
	for (uint32_t i = 0; i < peers.size(); i += 1) {
		if (is_epoch_important) {
			node->rpc_id(
					peers[i],
					"_rpc_doll_send_epoch",
					epoch,
					epoch_state_data.get_buffer().get_bytes());
		} else {
#warning TODO make sure that this peer really need this data.
			node->rpc_unreliable_id(
					peers[i],
					"_rpc_doll_send_epoch",
					epoch,
					epoch_state_data.get_buffer().get_bytes());
		}
	}
	is_epoch_important = false;
}

void ServerController::calculates_player_tick_rate(real_t p_delta) {
	// TODO Improve the algorithm that has the following problems:
	// - Tweaking is really difficult (requires too much tests).
	// - It's too slow to recover packets.
	// - It start doing something only when a packets is marked as missing,
	//     instead it should start healing the connection even before the input is used (and so marked as used).

	const int miss_packets = network_tracer.get_missing_packets();
	const int inputs_count = get_inputs_count();

	{
		// The first step to establish the client speed up amount is to define the
		// optimal `frames_inputs` size.
		// This size is increased and decreased using an acceleration, so any speed
		// change is spread across a long period rather a little one.
		const real_t acceleration_level = CLAMP(
				(static_cast<real_t>(miss_packets) -
						static_cast<real_t>(inputs_count)) /
						static_cast<real_t>(node->get_missing_snapshots_max_tolerance()),
				-2.0,
				2.0);
		optimal_snapshots_size += acceleration_level * node->get_optimal_size_acceleration() * p_delta;
		optimal_snapshots_size = CLAMP(optimal_snapshots_size, MIN_SNAPSHOTS_SIZE, node->get_server_input_storage_size());
	}

	{
		// The client speed is determined using an acceleration so to have much
		// more control over it, and avoid nervous changes.
		const real_t acceleration_level = CLAMP((optimal_snapshots_size - static_cast<real_t>(inputs_count)) / node->get_server_input_storage_size(), -1.0, 1.0);
		const real_t acc = acceleration_level * node->get_tick_acceleration() * p_delta;
		const real_t damp = client_tick_additional_speed * -0.9;

		// The damping is fully applyied only if it points in the opposite `acc`
		// direction.
		// I want to cut down the oscilations when the target is the same for a while,
		// but I need to move fast toward new targets when they appear.
		client_tick_additional_speed += acc + damp * ((SGN(acc) * SGN(damp) + 1) / 2.0);
		client_tick_additional_speed = CLAMP(client_tick_additional_speed, -MAX_ADDITIONAL_TICK_SPEED, MAX_ADDITIONAL_TICK_SPEED);
	}
}

void ServerController::adjust_player_tick_rate(real_t p_delta) {
	const uint8_t new_speed = UINT8_MAX * (((client_tick_additional_speed / MAX_ADDITIONAL_TICK_SPEED) + 1.0) / 2.0);

	additional_speed_notif_timer += p_delta;
	if (additional_speed_notif_timer >= node->get_tick_speedup_notification_delay()) {
		additional_speed_notif_timer = 0.0;

		Vector<uint8_t> packet_data;
		packet_data.push_back(new_speed);

		node->rpc_unreliable_id(
				node->get_network_master(),
				"_rpc_send_tick_additional_speed",
				packet_data);
	}
}

PlayerController::PlayerController(NetworkedController *p_node) :
		Controller(p_node),
		current_input_id(UINT64_MAX),
		input_buffers_counter(0),
		time_bank(0.0),
		tick_additional_speed(0.0) {
}

void PlayerController::process(real_t p_delta) {
	// We need to know if we can accept a new input because in case of bad
	// internet connection we can't keep accumulating inputs forever
	// otherwise the server will differ too much from the client and we
	// introduce virtual lag.
	const bool accept_new_inputs = can_accept_new_inputs();

	if (accept_new_inputs) {
		current_input_id = input_buffers_counter;
		input_buffers_counter += 1;
		node->get_inputs_buffer_mut().begin_write();
		node->call("collect_inputs", p_delta, &node->get_inputs_buffer_mut());
	} else {
		NET_DEBUG_WARN("It's not possible to accept new inputs. Is this lagging?");
	}

	node->get_inputs_buffer_mut().dry();
	node->get_inputs_buffer_mut().begin_read();

	// The physics process is always emitted, because we still need to simulate
	// the character motion even if we don't store the player inputs.
	node->call("controller_process", p_delta, &node->get_inputs_buffer_mut());

	if (accept_new_inputs) {
		store_input_buffer(current_input_id);
		send_frame_input_buffer_to_server();
	}

	node->player_set_has_new_input(accept_new_inputs);
}

int PlayerController::calculates_sub_ticks(real_t p_delta, real_t p_iteration_per_seconds) {
	const real_t pretended_delta = get_pretended_delta(p_iteration_per_seconds);

	time_bank += p_delta;
	const int sub_ticks = static_cast<uint32_t>(time_bank / pretended_delta);
	time_bank -= static_cast<real_t>(sub_ticks) * pretended_delta;
	return sub_ticks;
}

int PlayerController::notify_input_checked(uint64_t p_input_id) {
	// Remove inputs.
	while (frames_snapshot.empty() == false && frames_snapshot.front().id <= p_input_id) {
		frames_snapshot.pop_front();
	}
	// Unreachable, because the next input have always the next `p_input_id` or empty.
	CRASH_COND(frames_snapshot.empty() == false && (p_input_id + 1) != frames_snapshot.front().id);
	return frames_snapshot.size();
}

uint64_t PlayerController::last_known_input() const {
	return get_stored_input_id(-1);
}

uint64_t PlayerController::get_stored_input_id(int p_i) const {
	if (p_i < 0) {
		if (frames_snapshot.empty() == false) {
			return frames_snapshot.back().id;
		} else {
			return UINT64_MAX;
		}
	} else {
		const size_t i = p_i;
		if (i < frames_snapshot.size()) {
			return frames_snapshot[i].id;
		} else {
			return UINT64_MAX;
		}
	}
}

bool PlayerController::process_instant(int p_i, real_t p_delta) {
	const size_t i = p_i;
	if (i < frames_snapshot.size()) {
		DataBuffer ib(frames_snapshot[i].inputs_buffer);
		ib.begin_read();
		node->call("controller_process", p_delta, &ib);
		return (i + 1) < frames_snapshot.size();
	} else {
		return false;
	}
}

uint64_t PlayerController::get_current_input_id() const {
	return current_input_id;
}

real_t PlayerController::get_pretended_delta(real_t p_iteration_per_seconds) const {
	return 1.0 / (p_iteration_per_seconds + tick_additional_speed);
}

void PlayerController::store_input_buffer(uint64_t p_id) {
	FrameSnapshot inputs;
	inputs.id = p_id;
	inputs.inputs_buffer = node->get_inputs_buffer().get_buffer();
	inputs.similarity = UINT64_MAX;
	frames_snapshot.push_back(inputs);
}

void PlayerController::send_frame_input_buffer_to_server() {
	// The packet is composed as follow:
	// - The following four bytes for the first input ID.
	// - Array of inputs:
	// |-- First byte the amount of times this input is duplicated in the packet.
	// |-- input buffer.

	const size_t inputs_count = MIN(frames_snapshot.size(), static_cast<size_t>(node->get_max_redundant_inputs() + 1));
	CRASH_COND(inputs_count < 1); // Unreachable

#define MAKE_ROOM(p_size)                                              \
	if (cached_packet_data.size() < static_cast<size_t>(ofs + p_size)) \
		cached_packet_data.resize(ofs + p_size);

	int ofs = 0;

	// Let's store the ID of the first snapshot.
	MAKE_ROOM(4);
	const uint64_t first_input_id = frames_snapshot[frames_snapshot.size() - inputs_count].id;
	ofs += encode_uint32(first_input_id, cached_packet_data.data() + ofs);

	uint64_t previous_input_id = UINT64_MAX;
	uint64_t previous_input_similarity = UINT64_MAX;
	int previous_buffer_size = 0;
	uint8_t duplication_count = 0;

	DataBuffer pir_A(node->get_inputs_buffer().get_buffer());

	// Compose the packets
	for (size_t i = frames_snapshot.size() - inputs_count; i < frames_snapshot.size(); i += 1) {
		bool is_similar = false;

		if (previous_input_id == UINT64_MAX) {
			// This happens for the first input of the packet.
			// Just write it.
			is_similar = false;
		} else if (duplication_count == UINT8_MAX) {
			// Prevent to overflow the `uint8_t`.
			is_similar = false;
		} else {
			if (frames_snapshot[i].similarity != previous_input_id) {
				if (frames_snapshot[i].similarity == UINT64_MAX) {
					// This input was never compared, let's do it now.
					DataBuffer pir_B(frames_snapshot[i].inputs_buffer);

					pir_A.begin_read();
					pir_B.begin_read();

					const bool are_different = pir_A.get_buffer_size() != pir_B.get_buffer_size() || node->call("are_inputs_different", &pir_A, &pir_B);
					is_similar = are_different == false;

				} else if (frames_snapshot[i].similarity == previous_input_similarity) {
					// This input is similar to the previous one, the thing is
					// that the similarity check was done on an older input.
					// Fortunatelly we are able to compare the similarity id
					// and detect its similarity correctly.
					is_similar = true;
				} else {
					// This input is simply different from the previous one.
					is_similar = false;
				}
			} else {
				// These are the same, let's save some space.
				is_similar = true;
			}
		}

		if (is_similar) {
			// This input is similar to the previous one, so just duplicate it.
			duplication_count += 1;
			// In this way, we don't need to compare these frames again.
			frames_snapshot[i].similarity = previous_input_id;

		} else {
			// This input is different from the previous one, so let's
			// finalize the previous and start another one.

			if (previous_input_id != UINT64_MAX) {
				// We can finally finalize the previous input
				cached_packet_data[ofs - previous_buffer_size - 1] = duplication_count;
			}

			// Resets the duplication count.
			duplication_count = 0;

			// Writes the duplication_count for this new input
			MAKE_ROOM(1);
			cached_packet_data[ofs] = 0;
			ofs += 1;

			// Write the inputs
			const int buffer_size = frames_snapshot[i].inputs_buffer.get_bytes().size();
			MAKE_ROOM(buffer_size);
			copymem(
					cached_packet_data.data() + ofs,
					frames_snapshot[i].inputs_buffer.get_bytes().ptr(),
					buffer_size);
			ofs += buffer_size;

			// Let's see if we can duplicate this input.
			previous_input_id = frames_snapshot[i].id;
			previous_input_similarity = frames_snapshot[i].similarity;
			previous_buffer_size = buffer_size;

			pir_A.get_buffer_mut() = frames_snapshot[i].inputs_buffer;
		}
	}

	// Finalize the last added input_buffer.
	cached_packet_data[ofs - previous_buffer_size - 1] = duplication_count;

	// Make the packet data.
	Vector<uint8_t> packet_data;
	// TODO cache this?
	packet_data.resize(ofs);

	copymem(
			packet_data.ptrw(),
			cached_packet_data.data(),
			ofs);

	const int server_peer_id = 1;
	node->rpc_unreliable_id(server_peer_id, "_rpc_server_send_inputs", packet_data);
}

bool PlayerController::can_accept_new_inputs() const {
	return frames_snapshot.size() < static_cast<size_t>(node->get_player_input_storage_size());
}

DollController::DollController(NetworkedController *p_node) :
		Controller(p_node),
		network_tracer(10) {
}

DollController::~DollController() {
	node->set_physics_process_internal(false);
}

void DollController::ready() {
	interpolator.reset();
	node->call("setup_interpolator", &interpolator);
	interpolator.terminate_init();
	node->set_physics_process_internal(true);
}

void DollController::process(real_t p_delta) {
	const uint64_t frame_epoch = next_epoch(p_delta);

	if (unlikely(frame_epoch == UINT64_MAX)) {
		// Nothing to do.
		return;
	}

	node->call("epoch_process", p_delta, interpolator.pop_epoch(frame_epoch));
}

uint64_t DollController::get_current_input_id() const {
	return current_epoch;
}

void DollController::receive_epoch(uint64_t p_epoch, Vector<uint8_t> p_data) {
	DataBuffer buffer(p_data);
	buffer.begin_read();

	interpolator.begin_write(p_epoch);
	node->call("parse_epoch_data", &interpolator, &buffer);
	interpolator.end_write();
}

uint64_t DollController::next_epoch(real_t p_delta) {
	// This function regulates the epoch ID to process.
	// The epoch is not simply increased by one because we need to make sure
	// to make the client apply the nearest server state while giving some room
	// for the subsequent information to arrive.

	// Step 1, Wait that we have at least two epochs.
	if (unlikely(current_epoch == UINT64_MAX)) {
		// Interpolator is not yet started.
		if (interpolator.known_epochs_count() < 2) {
			// Not ready yet.
			return UINT64_MAX;
		}

#ifdef DEBUG_ENABLED
		// At this point we have 2 epoch, something is always returned at this
		// point.
		CRASH_COND(interpolator.get_youngest_epoch() == UINT64_MAX);
#endif

		// Start epoch interpolation.
		current_epoch = interpolator.get_youngest_epoch();
	}

	// At this point the interpolation is started and the function must
	// return the best epoch id which we have to apply the state.

	// Step 2. Make sure we have something to interpolate with.
	const uint64_t oldest_epoch = interpolator.get_oldest_epoch();
	if (unlikely(oldest_epoch == UINT64_MAX || oldest_epoch <= current_epoch)) {
		network_tracer.notify_missing_packet();
		// Nothing to interpolate with.
		return current_epoch;
	}
	network_tracer.notify_packet_arrived();

#ifdef DEBUG_ENABLED
	// This can't happen because the current_epoch is advances only if it's
	// possible to do so.
	CRASH_COND(oldest_epoch < current_epoch);
#endif

	const uint64_t delta_epoch = oldest_epoch - current_epoch;

	// TODO make this customizable
	const uint64_t min_virtual_delay = 2;
	const uint64_t max_virtual_delay = 60;
	const real_t speed_factor = 0.2;
	const real_t max_missing_epochs = 5.0;
	const real_t max_additional_delay = 1.0;

	// Step 3. The `ideal_virtual_delay` is used to introduce a buffering so
	// to give room to the subsequent information to arrive.
	// The server changes the send rate in a smooth way, so most likely we need
	// to wait the previous time window. However, the internet connection
	// oscilate from time to time, so the `additional_delay` is used to
	// introduce a small (and variable) delay to absorb it.
	const uint64_t additional_delay = MAX(MIN(real_t(network_tracer.get_missing_packets()) / max_missing_epochs, 1.0) * max_additional_delay, min_virtual_delay);

	const uint64_t ideal_virtual_delay = MIN(
			interpolator.epochs_between_last_time_window() + additional_delay,
			max_virtual_delay);

	if (unlikely(delta_epoch > max_virtual_delay)) {
		// This client seems too much behind at this point. Teleport forward.
		current_epoch = interpolator.get_oldest_epoch() - max_virtual_delay;
		advancing_epoch = 0.0;
	}

	if (delta_epoch > ideal_virtual_delay) {
		advancing_epoch += (real_t(delta_epoch - ideal_virtual_delay) / real_t(max_virtual_delay - ideal_virtual_delay)) * speed_factor;
	} else {
		advancing_epoch -= (real_t(ideal_virtual_delay - delta_epoch) / real_t(ideal_virtual_delay)) * speed_factor;
	}

	advancing_epoch += 1.0;

	if (advancing_epoch > 0.0) {
		// Advance the epoch by the the integral amount.
		current_epoch += uint64_t(advancing_epoch);

		// Keep the floating point part.
		advancing_epoch -= uint64_t(advancing_epoch);
	}

	if (unlikely(oldest_epoch <= current_epoch)) {
		// Clamp to oldest_epoch.
		current_epoch = oldest_epoch;
	}

	print_line("TW: " + itos(interpolator.epochs_between_last_time_window()) + " - Ideal virtual delay: " + itos(ideal_virtual_delay) + " - Delta epoch: " + itos(delta_epoch) + " - Advancing: " + rtos(advancing_epoch) + " - Epoch: " + itos(current_epoch));

	return current_epoch;
}

NoNetController::NoNetController(NetworkedController *p_node) :
		Controller(p_node),
		frame_id(0) {
}

void NoNetController::process(real_t p_delta) {
	node->get_inputs_buffer_mut().begin_write();
	node->call("collect_inputs", p_delta, &node->get_inputs_buffer_mut());
	node->get_inputs_buffer_mut().dry();
	node->get_inputs_buffer_mut().begin_read();
	node->call("controller_process", p_delta, &node->get_inputs_buffer_mut());
	frame_id += 1;
}

uint64_t NoNetController::get_current_input_id() const {
	return frame_id;
}
