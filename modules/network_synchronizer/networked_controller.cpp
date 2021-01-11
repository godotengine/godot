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
#include "core/os/os.h"
#include "scene_synchronizer.h"
#include <stdint.h>
#include <algorithm>

#define METADATA_SIZE 1

#define MAX_ADDITIONAL_TICK_SPEED 2.0

// 2%
#define TICK_SPEED_CHANGE_NOTIF_THRESHOLD 4

StringName NetworkedController::sn_rpc_server_send_inputs;
StringName NetworkedController::sn_rpc_send_tick_additional_speed;
StringName NetworkedController::sn_rpc_doll_notify_sync_pause;
StringName NetworkedController::sn_rpc_doll_send_epoch_batch;

StringName NetworkedController::sn_controller_process;
StringName NetworkedController::sn_count_input_size;
StringName NetworkedController::sn_are_inputs_different;
StringName NetworkedController::sn_collect_epoch_data;
StringName NetworkedController::sn_collect_inputs;
StringName NetworkedController::sn_setup_interpolator;
StringName NetworkedController::sn_parse_epoch_data;
StringName NetworkedController::sn_apply_epoch;

void NetworkedController::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_player_input_storage_size", "size"), &NetworkedController::set_player_input_storage_size);
	ClassDB::bind_method(D_METHOD("get_player_input_storage_size"), &NetworkedController::get_player_input_storage_size);

	ClassDB::bind_method(D_METHOD("set_max_redundant_inputs", "max_redundand_inputs"), &NetworkedController::set_max_redundant_inputs);
	ClassDB::bind_method(D_METHOD("get_max_redundant_inputs"), &NetworkedController::get_max_redundant_inputs);

	ClassDB::bind_method(D_METHOD("set_tick_speedup_notification_delay", "tick_speedup_notification_delay"), &NetworkedController::set_tick_speedup_notification_delay);
	ClassDB::bind_method(D_METHOD("get_tick_speedup_notification_delay"), &NetworkedController::get_tick_speedup_notification_delay);

	ClassDB::bind_method(D_METHOD("set_network_traced_frames", "size"), &NetworkedController::set_network_traced_frames);
	ClassDB::bind_method(D_METHOD("get_network_traced_frames"), &NetworkedController::get_network_traced_frames);

	ClassDB::bind_method(D_METHOD("set_min_frames_delay", "val"), &NetworkedController::set_min_frames_delay);
	ClassDB::bind_method(D_METHOD("get_min_frames_delay"), &NetworkedController::get_min_frames_delay);

	ClassDB::bind_method(D_METHOD("set_max_frames_delay", "val"), &NetworkedController::set_max_frames_delay);
	ClassDB::bind_method(D_METHOD("get_max_frames_delay"), &NetworkedController::get_max_frames_delay);

	ClassDB::bind_method(D_METHOD("set_net_sensitivity", "val"), &NetworkedController::set_net_sensitivity);
	ClassDB::bind_method(D_METHOD("get_net_sensitivity"), &NetworkedController::get_net_sensitivity);

	ClassDB::bind_method(D_METHOD("set_tick_acceleration", "acceleration"), &NetworkedController::set_tick_acceleration);
	ClassDB::bind_method(D_METHOD("get_tick_acceleration"), &NetworkedController::get_tick_acceleration);

	ClassDB::bind_method(D_METHOD("set_doll_epoch_collect_rate", "rate"), &NetworkedController::set_doll_epoch_collect_rate);
	ClassDB::bind_method(D_METHOD("get_doll_epoch_collect_rate"), &NetworkedController::get_doll_epoch_collect_rate);

	ClassDB::bind_method(D_METHOD("set_doll_epoch_batch_sync_rate", "rate"), &NetworkedController::set_doll_epoch_batch_sync_rate);
	ClassDB::bind_method(D_METHOD("get_doll_epoch_batch_sync_rate"), &NetworkedController::get_doll_epoch_batch_sync_rate);

	ClassDB::bind_method(D_METHOD("set_doll_min_frames_delay", "traced"), &NetworkedController::set_doll_min_frames_delay);
	ClassDB::bind_method(D_METHOD("get_doll_min_frames_delay"), &NetworkedController::get_doll_min_frames_delay);

	ClassDB::bind_method(D_METHOD("set_doll_max_frames_delay", "sensitivity"), &NetworkedController::set_doll_max_frames_delay);
	ClassDB::bind_method(D_METHOD("get_doll_max_frames_delay"), &NetworkedController::get_doll_max_frames_delay);

	ClassDB::bind_method(D_METHOD("set_doll_interpolation_max_speedup", "speedup"), &NetworkedController::set_doll_interpolation_max_speedup);
	ClassDB::bind_method(D_METHOD("get_doll_interpolation_max_speedup"), &NetworkedController::get_doll_interpolation_max_speedup);

	ClassDB::bind_method(D_METHOD("set_doll_connection_stats_frame_span", "speedup"), &NetworkedController::set_doll_connection_stats_frame_span);
	ClassDB::bind_method(D_METHOD("get_doll_connection_stats_frame_span"), &NetworkedController::get_doll_connection_stats_frame_span);

	ClassDB::bind_method(D_METHOD("get_current_input_id"), &NetworkedController::get_current_input_id);

	ClassDB::bind_method(D_METHOD("player_get_pretended_delta", "iterations_per_seconds"), &NetworkedController::player_get_pretended_delta);

	ClassDB::bind_method(D_METHOD("mark_epoch_as_important"), &NetworkedController::mark_epoch_as_important);

	ClassDB::bind_method(D_METHOD("set_doll_collect_rate_factor", "peer", "factor"), &NetworkedController::set_doll_collect_rate_factor);

	ClassDB::bind_method(D_METHOD("set_doll_peer_active", "peer_id", "active"), &NetworkedController::set_doll_peer_active);

	ClassDB::bind_method(D_METHOD("_rpc_server_send_inputs"), &NetworkedController::_rpc_server_send_inputs);
	ClassDB::bind_method(D_METHOD("_rpc_send_tick_additional_speed"), &NetworkedController::_rpc_send_tick_additional_speed);
	ClassDB::bind_method(D_METHOD("_rpc_doll_notify_sync_pause"), &NetworkedController::_rpc_doll_notify_sync_pause);
	ClassDB::bind_method(D_METHOD("_rpc_doll_send_epoch_batch"), &NetworkedController::_rpc_doll_send_epoch_batch);

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
	BIND_VMETHOD(MethodInfo("apply_epoch", PropertyInfo(Variant::FLOAT, "delta"), PropertyInfo(Variant::ARRAY, "interpolated_data")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "input_storage_size", PROPERTY_HINT_RANGE, "5,2000,1"), "set_player_input_storage_size", "get_player_input_storage_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_redundant_inputs", PROPERTY_HINT_RANGE, "0,1000,1"), "set_max_redundant_inputs", "get_max_redundant_inputs");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tick_speedup_notification_delay", PROPERTY_HINT_RANGE, "0.001,2.0,0.001"), "set_tick_speedup_notification_delay", "get_tick_speedup_notification_delay");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "network_traced_frames", PROPERTY_HINT_RANGE, "1,1000,1"), "set_network_traced_frames", "get_network_traced_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "min_frames_delay", PROPERTY_HINT_RANGE, "0,100,1"), "set_min_frames_delay", "get_min_frames_delay");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_frames_delay", PROPERTY_HINT_RANGE, "0,100,1"), "set_max_frames_delay", "get_max_frames_delay");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "net_sensitivity", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_net_sensitivity", "get_net_sensitivity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tick_acceleration", PROPERTY_HINT_RANGE, "0.1,20.0,0.01"), "set_tick_acceleration", "get_tick_acceleration");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "doll_epoch_collect_rate", PROPERTY_HINT_RANGE, "1,100,1"), "set_doll_epoch_collect_rate", "get_doll_epoch_collect_rate");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "doll_epoch_batch_sync_rate", PROPERTY_HINT_RANGE, "0.01,5.0,0.01"), "set_doll_epoch_batch_sync_rate", "get_doll_epoch_batch_sync_rate");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "doll_min_frames_delay", PROPERTY_HINT_RANGE, "0,10,1"), "set_doll_min_frames_delay", "get_doll_min_frames_delay");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "doll_max_frames_delay", PROPERTY_HINT_RANGE, "0,10,1"), "set_doll_max_frames_delay", "get_doll_max_frames_delay");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "doll_interpolation_max_speedup", PROPERTY_HINT_RANGE, "0.01,5.0,0.01"), "set_doll_interpolation_max_speedup", "get_doll_interpolation_max_speedup");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "doll_connection_stats_frame_span", PROPERTY_HINT_RANGE, "0,1000,1"), "set_doll_connection_stats_frame_span", "get_doll_connection_stats_frame_span");

	ADD_SIGNAL(MethodInfo("doll_sync_started"));
	ADD_SIGNAL(MethodInfo("doll_sync_paused"));
}

NetworkedController::NetworkedController() {
	rpc_config(sn_rpc_server_send_inputs, MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config(sn_rpc_send_tick_additional_speed, MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config(sn_rpc_doll_notify_sync_pause, MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config(sn_rpc_doll_send_epoch_batch, MultiplayerAPI::RPC_MODE_REMOTE);
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

void NetworkedController::set_min_frames_delay(int p_val) {
	min_frames_delay = p_val;
}

int NetworkedController::get_min_frames_delay() const {
	return min_frames_delay;
}

void NetworkedController::set_max_frames_delay(int p_val) {
	max_frames_delay = p_val;
}

int NetworkedController::get_max_frames_delay() const {
	return max_frames_delay;
}

void NetworkedController::set_net_sensitivity(real_t p_val) {
	net_sensitivity = p_val;
}

real_t NetworkedController::get_net_sensitivity() const {
	return net_sensitivity;
}

void NetworkedController::set_tick_acceleration(real_t p_acceleration) {
	tick_acceleration = p_acceleration;
}

real_t NetworkedController::get_tick_acceleration() const {
	return tick_acceleration;
}

void NetworkedController::set_doll_epoch_collect_rate(int p_rate) {
	doll_epoch_collect_rate = MAX(p_rate, 1);
}

int NetworkedController::get_doll_epoch_collect_rate() const {
	return doll_epoch_collect_rate;
}

void NetworkedController::set_doll_epoch_batch_sync_rate(real_t p_rate) {
	doll_epoch_batch_sync_rate = MAX(p_rate, 0.001);
}

real_t NetworkedController::get_doll_epoch_batch_sync_rate() const {
	return doll_epoch_batch_sync_rate;
}

void NetworkedController::set_doll_min_frames_delay(int p_min) {
	doll_min_frames_delay = p_min;
}

int NetworkedController::get_doll_min_frames_delay() const {
	return doll_min_frames_delay;
}

void NetworkedController::set_doll_max_frames_delay(int p_max) {
	doll_max_frames_delay = p_max;
}

int NetworkedController::get_doll_max_frames_delay() const {
	return doll_max_frames_delay;
}

void NetworkedController::set_doll_interpolation_max_speedup(real_t p_speedup) {
	doll_interpolation_max_speedup = p_speedup;
}

real_t NetworkedController::get_doll_interpolation_max_speedup() const {
	return doll_interpolation_max_speedup;
}

void NetworkedController::set_doll_connection_stats_frame_span(int p_span) {
	doll_connection_stats_frame_span = p_span;
}

int NetworkedController::get_doll_connection_stats_frame_span() const {
	return doll_connection_stats_frame_span;
}

void NetworkedController::set_doll_net_sensitivity(real_t p_sensitivity) {
	doll_net_sensitivity = p_sensitivity;
}

real_t NetworkedController::get_doll_net_sensitivity() const {
	return doll_net_sensitivity;
}

uint32_t NetworkedController::get_current_input_id() const {
	return controller->get_current_input_id();
}

real_t NetworkedController::player_get_pretended_delta(uint32_t p_iterations_per_seconds) const {
	ERR_FAIL_COND_V_MSG(is_player_controller() == false, 1.0 / real_t(p_iterations_per_seconds), "This function can be called only on client.");
	return get_player_controller()->get_pretended_delta(p_iterations_per_seconds);
}

void NetworkedController::mark_epoch_as_important() {
	ERR_FAIL_COND_MSG(is_server_controller() == false, "This function must be called only within the function `collect_epoch_data`.");
	get_server_controller()->is_epoch_important = true;
}

void NetworkedController::set_doll_collect_rate_factor(int p_peer, real_t p_factor) {
	ERR_FAIL_COND_MSG(is_server_controller() == false, "This function can be called only on server.");
	ServerController *server_controller = static_cast<ServerController *>(controller);
	const uint32_t pos = server_controller->find_peer(p_peer);
	if (pos == UINT32_MAX) {
		// This peers seems disabled, nothing to do here.
		return;
	}
	server_controller->peers[pos].update_rate_factor = CLAMP(p_factor, 0.001, 1.0);
}

void NetworkedController::set_doll_peer_active(int p_peer_id, bool p_active) {
	ERR_FAIL_COND_MSG(is_server_controller() == false, "You can set doll activation only on server");
	ERR_FAIL_COND_MSG(p_peer_id == get_network_master(), "This `peer_id` is equal to the Master `peer_id`, which is not allowed.");

	ServerController *server_controller = static_cast<ServerController *>(controller);
	const uint32_t pos = server_controller->find_peer(p_peer_id);
	if (pos == UINT32_MAX) {
		// This peers seems disabled, nothing to do here.
		return;
	}
	if (server_controller->peers[pos].active == p_active) {
		// Nothing to do.
		return;
	}

	server_controller->peers[pos].active = p_active;
	server_controller->peers[pos].collect_timer = 0.0;

	if (p_active == false) {
		// Notify the doll only for deactivations. The activations are automatically
		// handled when the first epoch is received.
		rpc_id(p_peer_id, sn_rpc_doll_notify_sync_pause, server_controller->epoch);
	}
}

void NetworkedController::pause_notify_dolls() {
	ERR_FAIL_COND_MSG(is_server_controller() == false, "You can pause the dolls only on server. [BUG]");

	// Notify the dolls this actor is disabled.
	ServerController *server_controller = static_cast<ServerController *>(controller);
	for (uint32_t i = 0; i < server_controller->peers.size(); i += 1) {
		if (server_controller->peers[i].active) {
			// Notify this actor is no more active.
			rpc_id(server_controller->peers[i].peer, sn_rpc_doll_notify_sync_pause, server_controller->epoch);
		}
	}
}

bool NetworkedController::process_instant(int p_i, real_t p_delta) {
	ERR_FAIL_COND_V_MSG(is_player_controller() == false, false, "Can be executed only on player controllers.");
	return static_cast<PlayerController *>(controller)->process_instant(p_i, p_delta);
}

ServerController *NetworkedController::get_server_controller() {
	ERR_FAIL_COND_V_MSG(is_server_controller() == false, nullptr, "This controller is not a server controller.");
	return static_cast<ServerController *>(controller);
}

const ServerController *NetworkedController::get_server_controller() const {
	ERR_FAIL_COND_V_MSG(is_server_controller() == false, nullptr, "This controller is not a server controller.");
	return static_cast<const ServerController *>(controller);
}

PlayerController *NetworkedController::get_player_controller() {
	ERR_FAIL_COND_V_MSG(is_player_controller() == false, nullptr, "This controller is not a player controller.");
	return static_cast<PlayerController *>(controller);
}

const PlayerController *NetworkedController::get_player_controller() const {
	ERR_FAIL_COND_V_MSG(is_player_controller() == false, nullptr, "This controller is not a player controller.");
	return static_cast<const PlayerController *>(controller);
}

DollController *NetworkedController::get_doll_controller() {
	ERR_FAIL_COND_V_MSG(is_doll_controller() == false, nullptr, "This controller is not a doll controller.");
	return static_cast<DollController *>(controller);
}

const DollController *NetworkedController::get_doll_controller() const {
	ERR_FAIL_COND_V_MSG(is_doll_controller() == false, nullptr, "This controller is not a doll controller.");
	return static_cast<const DollController *>(controller);
}

NoNetController *NetworkedController::get_nonet_controller() {
	ERR_FAIL_COND_V_MSG(is_nonet_controller() == false, nullptr, "This controller is not a no net controller.");
	return static_cast<NoNetController *>(controller);
}

const NoNetController *NetworkedController::get_nonet_controller() const {
	ERR_FAIL_COND_V_MSG(is_nonet_controller() == false, nullptr, "This controller is not a no net controller.");
	return static_cast<const NoNetController *>(controller);
}

bool NetworkedController::is_server_controller() const {
	return controller_type == CONTROLLER_TYPE_SERVER;
}

bool NetworkedController::is_player_controller() const {
	return controller_type == CONTROLLER_TYPE_PLAYER;
}

bool NetworkedController::is_doll_controller() const {
	return controller_type == CONTROLLER_TYPE_DOLL;
}

bool NetworkedController::is_nonet_controller() const {
	return controller_type == CONTROLLER_TYPE_NONETWORK;
}

void NetworkedController::set_inputs_buffer(const BitArray &p_new_buffer, uint32_t p_metadata_size_in_bit, uint32_t p_size_in_bit) {
	inputs_buffer.get_buffer_mut().get_bytes_mut() = p_new_buffer.get_bytes();
	inputs_buffer.force_set_size(p_metadata_size_in_bit, p_size_in_bit);
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

void NetworkedController::_rpc_doll_notify_sync_pause(uint32_t p_epoch) {
	ERR_FAIL_COND_MSG(is_doll_controller() == false, "Only dolls are supposed to receive this function call");

	static_cast<DollController *>(controller)->pause(p_epoch);
}

void NetworkedController::_rpc_doll_send_epoch_batch(Vector<uint8_t> p_data) {
	ERR_FAIL_COND_MSG(is_doll_controller() == false, "Only dolls are supposed to receive this function call.");
	ERR_FAIL_COND_MSG(p_data.size() <= 0, "It's not supposed to receive a 0 size data.");

	static_cast<DollController *>(controller)->receive_batch(p_data);
}

void NetworkedController::player_set_has_new_input(bool p_has) {
	has_player_new_input = p_has;
}

bool NetworkedController::player_has_new_input() const {
	return has_player_new_input;
}

void NetworkedController::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

#ifdef DEBUG_ENABLED
			// This can't happen, since only the doll are processed here.
			CRASH_COND(is_doll_controller() == false);
#endif
			static_cast<DollController *>(controller)->process(get_physics_process_delta_time());

		} break;
#ifdef DEBUG_ENABLED
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			ERR_FAIL_COND_MSG(has_method("collect_inputs") == false, "In your script you must inherit the virtual method `collect_inputs` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("controller_process") == false, "In your script you must inherit the virtual method `controller_process` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("are_inputs_different") == false, "In your script you must inherit the virtual method `are_inputs_different` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("count_input_size") == false, "In your script you must inherit the virtual method `count_input_size` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("collect_epoch_data") == false, "In your script you must inherit the virtual method `collect_epoch_data` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("setup_interpolator") == false, "In your script you must inherit the virtual method `setup_interpolator` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("parse_epoch_data") == false, "In your script you must inherit the virtual method `parse_epoch_data` to correctly use the `NetworkedController`.");
			ERR_FAIL_COND_MSG(has_method("apply_epoch") == false, "In your script you must inherit the virtual method `apply_epoch` to correctly use the `NetworkedController`.");

		} break;
#endif
	}
}

ServerController::ServerController(
		NetworkedController *p_node,
		int p_traced_frames) :
		Controller(p_node),
		network_watcher(p_traced_frames, 0) {
}

void ServerController::process(real_t p_delta) {
	if (unlikely(enabled == false)) {
		// Disabled by the SceneSynchronizer.
		return;
	}

	fetch_next_input();

	if (unlikely(current_input_buffer_id == UINT32_MAX)) {
		// Skip this until the first input arrive.
		return;
	}

	node->get_inputs_buffer_mut().begin_read();
	node->get_inputs_buffer_mut().seek(METADATA_SIZE);
	node->call(
			NetworkedController::sn_controller_process,
			p_delta,
			&node->get_inputs_buffer_mut());

	doll_sync(p_delta);

	if (streaming_paused == false) {
		calculates_player_tick_rate(p_delta);
		adjust_player_tick_rate(p_delta);
	}
}

bool is_remote_frame_A_older(const FrameSnapshot &p_snap_a, const FrameSnapshot &p_snap_b) {
	return p_snap_a.id < p_snap_b.id;
}

uint32_t ServerController::last_known_input() const {
	if (snapshots.size() > 0) {
		return snapshots.back().id;
	} else {
		return UINT32_MAX;
	}
}

uint32_t ServerController::get_current_input_id() const {
	return current_input_buffer_id;
}

void ServerController::set_enabled(bool p_enable) {
	if (enabled == p_enable) {
		return;
	}
	enabled = p_enable;
	node->pause_notify_dolls();

	// ~~ On state change, reset everything to avoid accumulate old data. ~~

	// Client inputs reset.
	ghost_input_count = 0;
	last_sent_state_input_id = 0;
	client_tick_additional_speed = 0.0;
	additional_speed_notif_timer = 0.0;
	snapshots.clear();
	input_arrival_time = UINT32_MAX;
	network_watcher.reset(0.0);

	// Doll reset.
	is_epoch_important = false;
	batch_sync_timer = 0.0;
}

void ServerController::clear_peers() {
	peers.clear();
}

void ServerController::activate_peer(int p_peer) {
	// Collects all the dolls.

#ifdef DEBUG_ENABLED
	// Unreachable because this is the server controller.
	CRASH_COND(node->get_tree()->is_network_server() == false);
#endif
	if (p_peer == node->get_network_master()) {
		// This is self, so not a doll.
		return;
	}

	const uint32_t index = find_peer(p_peer);
	if (index == UINT32_MAX) {
		peers.push_back(p_peer);
	}
}

void ServerController::deactivate_peer(int p_peer) {
	const uint32_t index = find_peer(p_peer);
	if (index != UINT32_MAX) {
		peers.remove_unordered(p_peer);
	}
}

void ServerController::receive_inputs(Vector<uint8_t> p_data) {
	// The packet is composed as follow:
	// |- The following four bytes for the first input ID.
	// \- Array of inputs:
	//      |-- First byte the amount of times this input is duplicated in the packet.
	//      |-- inputs buffer.
	//
	// Let's decode it!

	const uint32_t now = OS::get_singleton()->get_ticks_msec();
	// If negative the timer was disabled, so just assume 0.
	network_watcher.push(MAX(0, now - input_arrival_time));
	input_arrival_time = now;

	const int data_len = p_data.size();

	int ofs = 0;

	ERR_FAIL_COND(data_len < 4);
	const uint32_t first_input_id = decode_uint32(p_data.ptr() + ofs);
	ofs += 4;

	uint32_t inserted_input_count = 0;

	// Contains the entire packet and in turn it will be seek to specific location
	// so I will not need to copy chunk of the packet data.
	DataBuffer pir;
	pir.begin_read();
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
		const int input_buffer_offset_bit = ofs * 8;
		pir.force_set_size(input_buffer_offset_bit, (data_len - ofs) * 8);
		pir.seek(input_buffer_offset_bit);
		// Read metadata
		const bool has_data = pir.read_bool();

		const int input_size_in_bits = (has_data ? int(node->call(NetworkedController::sn_count_input_size, &pir)) : 0) + METADATA_SIZE;
		// Pad to 8 bits.
		const int input_size_padded =
				Math::ceil((static_cast<float>(input_size_in_bits)) / 8.0);
		ERR_FAIL_COND_MSG(ofs + input_size_padded > data_len, "The arrived packet size doesn't meet the expected size.");

		// The input is valid, populate the buffer.
		for (int sub = 0; sub <= duplication; sub += 1) {
			const uint32_t input_id = first_input_id + inserted_input_count;
			inserted_input_count += 1;

			if (unlikely(current_input_buffer_id != UINT32_MAX && current_input_buffer_id >= input_id)) {
				// We already have this input, so we don't need it anymore.
				continue;
			}

			FrameSnapshot rfs;
			rfs.id = input_id;

			const bool found = std::binary_search(
					snapshots.begin(),
					snapshots.end(),
					rfs,
					is_remote_frame_A_older);

			if (found == false) {
				rfs.buffer_size_bit = input_size_in_bits;
				rfs.inputs_buffer.get_bytes_mut().resize(input_size_padded);
				copymem(
						rfs.inputs_buffer.get_bytes_mut().ptrw(),
						p_data.ptr() + ofs,
						input_size_padded);

				snapshots.push_back(rfs);

				// Sort the new inserted snapshot.
				std::sort(
						snapshots.begin(),
						snapshots.end(),
						is_remote_frame_A_older);
			}
		}

		// We can now advance the offset.
		ofs += input_size_padded;
	}

#ifdef DEBUG_ENABLED
	if (snapshots.empty() == false && current_input_buffer_id != UINT32_MAX) {
		// At this point is guaranteed that the current_input_buffer_id is never
		// greater than the first item contained by `snapshots`.
		CRASH_COND(current_input_buffer_id >= snapshots.front().id);
	}
#endif

	ERR_FAIL_COND_MSG(ofs != data_len, "At the end was detected that the arrived packet has an unexpected size.");
}

int ServerController::get_inputs_count() const {
	return snapshots.size();
}

bool ServerController::fetch_next_input() {
	bool is_new_input = true;

	if (unlikely(current_input_buffer_id == UINT32_MAX)) {
		// As initial packet, anything is good.
		if (snapshots.empty() == false) {
			// First input arrived.
			node->set_inputs_buffer(snapshots.front().inputs_buffer, METADATA_SIZE, snapshots.front().buffer_size_bit - METADATA_SIZE);
			current_input_buffer_id = snapshots.front().id;
			snapshots.pop_front();
			// Start tracing the packets from this moment on.
			network_watcher.reset(0);
			input_arrival_time = UINT32_MAX;
		} else {
			is_new_input = false;
		}
	} else {
		const uint32_t next_input_id = current_input_buffer_id + 1;

		if (unlikely(streaming_paused)) {
			// Stream is paused.
			if (snapshots.empty() == false &&
					snapshots.front().id >= next_input_id) {
				// A new input is arrived while the streaming is paused.
				const bool is_buffer_void = (snapshots.front().buffer_size_bit - METADATA_SIZE) == 0;
				streaming_paused = is_buffer_void;
				node->set_inputs_buffer(snapshots.front().inputs_buffer, METADATA_SIZE, snapshots.front().buffer_size_bit - METADATA_SIZE);
				current_input_buffer_id = snapshots.front().id;
				is_new_input = true;
				snapshots.pop_front();
				network_watcher.reset(0);
				input_arrival_time = UINT32_MAX;
			} else {
				// No inputs, or we are not yet arrived to the client input,
				// so just pretend the next input is void.
				node->set_inputs_buffer(BitArray(METADATA_SIZE), METADATA_SIZE, 0);
				is_new_input = false;
			}
		} else if (unlikely(snapshots.empty() == true)) {
			// The input buffer is empty; a packet is missing.
			is_new_input = false;
			ghost_input_count += 1;
			NET_DEBUG_PRINT("Input buffer is void, i'm using the previous one!");

		} else {
			// The input buffer is not empty, search the new input.
			if (next_input_id == snapshots.front().id) {
				// Wow, the next input is perfect!
				node->set_inputs_buffer(snapshots.front().inputs_buffer, METADATA_SIZE, snapshots.front().buffer_size_bit - METADATA_SIZE);
				current_input_buffer_id = snapshots.front().id;
				snapshots.pop_front();

				ghost_input_count = 0;
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

				ghost_input_count += 1;

				const int size = MIN(ghost_input_count, snapshots.size());
				const uint32_t ghost_packet_id = next_input_id + ghost_input_count;

				bool recovered = false;
				FrameSnapshot pi;

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
						pir_B.force_set_size(METADATA_SIZE, pi.buffer_size_bit - METADATA_SIZE);

						pir_A.begin_read();
						pir_A.seek(METADATA_SIZE);
						pir_B.begin_read();
						pir_B.seek(METADATA_SIZE);

						const bool is_meaningful = node->call(NetworkedController::sn_are_inputs_different, &pir_A, &pir_B);
						if (is_meaningful) {
							break;
						}
					}
				}

				if (recovered) {
					node->set_inputs_buffer(pi.inputs_buffer, METADATA_SIZE, snapshots.front().buffer_size_bit - METADATA_SIZE);
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

#ifdef DEBUG_ENABLED
	if (snapshots.empty() == false && current_input_buffer_id != UINT32_MAX) {
		// At this point is guaranteed that the current_input_buffer_id is never
		// greater than the first item contained by `snapshots`.
		CRASH_COND(current_input_buffer_id >= snapshots.front().id);
	}
#endif
	return is_new_input;
}

void ServerController::notify_send_state() {
	last_sent_state_input_id = get_current_input_id();
	// If the notified input is a void buffer, the client is allowed to pause
	// the input streaming. So missing packets are just handled as void inputs.
	if (node->get_inputs_buffer().size() == 0) {
		streaming_paused = true;
	}
}

void ServerController::doll_sync(real_t p_delta) {
	// Advance the epoch.
	epoch += 1;
	batch_sync_timer += p_delta;
	const bool send_batch = batch_sync_timer >= node->get_doll_epoch_batch_sync_rate();

	bool epoch_state_collected = false;

	// Process each peer and send the data if needed.
	for (uint32_t i = 0; i < peers.size(); i += 1) {
		if (peers[i].active == false) {
			// Nothing to do on this peer.
			continue;
		}

		peers[i].collect_timer += 1;
		if (
				is_epoch_important ||
				peers[i].collect_timer >= peers[i].collect_threshold) {
			// Resets the timer.
			peers[i].collect_timer -= peers[i].collect_threshold;
			// Since is possible to force send the state update, we need to make
			// sure the timer doesn't go below 0.
			peers[i].collect_timer = MAX(0, peers[i].collect_timer);

			// Prepare the epoch_data cache.
			if (epoch_state_collected == false) {
				epoch_state_data_cache.begin_write(0);
				epoch_state_data_cache.add_int(epoch, DataBuffer::COMPRESSION_LEVEL_1);
				node->call(NetworkedController::sn_collect_epoch_data, &epoch_state_data_cache);
				epoch_state_data_cache.dry();
				epoch_state_collected = true;
			}

			// Store this into epoch batch.
			if (unlikely(epoch_state_data_cache.get_buffer().get_bytes().size() > UINT8_MAX)) {
				// If the packet is more than 255 it can't be sent.
				NET_DEBUG_ERR("The status update is too big, try to staty under 255 bytes per update. This status is dropped.");
			} else {
				peers[i].batch_size += 1 + epoch_state_data_cache.get_buffer().get_bytes().size();
				peers[i].epoch_batch.push_back(epoch_state_data_cache.get_buffer().get_bytes());
			}
		}

		// Send batch data.
		if (send_batch) {
			const uint8_t next_collect_rate =
					MIN(node->get_doll_epoch_collect_rate() /
									peers[i].update_rate_factor,
							UINT8_MAX);

			// Next rate is
			peers[i].collect_threshold = next_collect_rate;

			if (peers[i].epoch_batch.size() > 0) {

				// Add space to allocate the next_collect_rate.
				peers[i].batch_size += 1;

#ifdef DEBUG_ENABLED
				if (peers[i].batch_size >= 1350) {
					NET_DEBUG_WARN("The amount of data collected for this batch is more than 1350 bytes. Please make sure the `doll_sync_timer_rate` is not so big, so to avoid packet fragmentation. Batch size: " + itos(peers[i].batch_size) + " - Epochs into the batch: " + itos(peers[i].epoch_batch.size()));
				}
#endif

				// Prepare the batch data.
				Vector<uint8_t> data;
				data.resize(peers[i].batch_size);
				uint8_t *data_ptr = data.ptrw();
				uint32_t offset = 0;
				data_ptr[offset] = next_collect_rate;
				offset += 1;
				for (uint32_t x = 0; x < peers[i].epoch_batch.size(); x += 1) {
					ERR_CONTINUE_MSG(peers[i].epoch_batch[x].size() > 256, "It's not allowed to send more than 256 bytes per status. This status is dropped.");
					data_ptr[offset] = peers[i].epoch_batch[x].size();
					offset += 1;
					for (int l = 0; l < peers[i].epoch_batch[x].size(); l += 1) {
						data_ptr[offset] = peers[i].epoch_batch[x][l];
						offset += 1;
					}
				}
#ifdef DEBUG_ENABLED
				// This is not supposed to happen because the batch_size is
				// correctly computed.
				CRASH_COND(offset != peers[i].batch_size);
#endif
				peers[i].epoch_batch.clear();
				peers[i].batch_size = 0;

				// Send the data
				node->rpc_unreliable_id(
						peers[i].peer,
						NetworkedController::sn_rpc_doll_send_epoch_batch,
						data);
			}
		}
	}

	if (send_batch) {
		batch_sync_timer = 0.0;
	}

	is_epoch_important = false;
}

void ServerController::calculates_player_tick_rate(real_t p_delta) {
	const real_t min_frames_delay = node->get_min_frames_delay();
	const real_t max_frames_delay = node->get_max_frames_delay();
	const real_t net_sensitivity = node->get_net_sensitivity();

	const uint32_t avg_receive_time = network_watcher.average();
	const real_t deviation_receive_time = real_t(network_watcher.get_deviation(avg_receive_time)) / 1000.0;

	// The network quality can be established just by checking the standard
	// deviation. Stable connections have standard deviation that tend to 0.
	const real_t net_poorness = MIN(
			deviation_receive_time / net_sensitivity,
			1.0);

	const int optimal_frame_delay = Math::lerp(
			min_frames_delay,
			max_frames_delay,
			net_poorness);

	int consecutive_inputs = 0;
	for (uint32_t i = 0; i < snapshots.size(); i += 1) {
		if (snapshots[i].id == (current_input_buffer_id + consecutive_inputs + 1)) {
			consecutive_inputs += 1;
		}
	}

	const real_t distance_to_optimal_count = real_t(optimal_frame_delay - consecutive_inputs);

	const real_t acc = distance_to_optimal_count * node->get_tick_acceleration() * p_delta;
	// Used to avoid oscillations.
	const real_t damp = -(client_tick_additional_speed * 0.95);
	client_tick_additional_speed += acc + damp * ((SGN(acc) * SGN(damp) + 1) / 2.0);
	client_tick_additional_speed = CLAMP(client_tick_additional_speed, -MAX_ADDITIONAL_TICK_SPEED, MAX_ADDITIONAL_TICK_SPEED);

#ifdef DEBUG_ENABLED
	const bool debug = ProjectSettings::get_singleton()->get_setting("NetworkSynchronizer/debug_server_speedup");
	if (debug) {
		print_line("Network poorness: " + rtos(net_poorness) + " optimal frame delay: " + itos(optimal_frame_delay) + " Deviation: " + rtos(deviation_receive_time));
	}
#endif
}

void ServerController::adjust_player_tick_rate(real_t p_delta) {
	additional_speed_notif_timer += p_delta;
	if (additional_speed_notif_timer >= node->get_tick_speedup_notification_delay()) {
		additional_speed_notif_timer = 0.0;

		const uint8_t new_speed = UINT8_MAX * (((client_tick_additional_speed / MAX_ADDITIONAL_TICK_SPEED) + 1.0) / 2.0);

		Vector<uint8_t> packet_data;
		packet_data.push_back(new_speed);

		node->rpc_unreliable_id(
				node->get_network_master(),
				NetworkedController::sn_rpc_send_tick_additional_speed,
				packet_data);
	}
}

uint32_t ServerController::find_peer(int p_peer) const {
	for (uint32_t i = 0; i < peers.size(); i += 1) {
		if (peers[i].peer == p_peer) {
			return i;
		}
	}
	return UINT32_MAX;
}

PlayerController::PlayerController(NetworkedController *p_node) :
		Controller(p_node),
		current_input_id(UINT32_MAX),
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

		node->get_inputs_buffer_mut().begin_write(METADATA_SIZE);

		node->get_inputs_buffer_mut().seek(1);
		node->call(NetworkedController::sn_collect_inputs, p_delta, &node->get_inputs_buffer_mut());

		// Set metadata data.
		node->get_inputs_buffer_mut().seek(0);
		if (node->get_inputs_buffer().size() > 0) {
			node->get_inputs_buffer_mut().add_bool(true);
			streaming_paused = false;
		} else {
			node->get_inputs_buffer_mut().add_bool(false);
		}
	} else {
		NET_DEBUG_WARN("It's not possible to accept new inputs. Is this lagging?");
	}

	node->get_inputs_buffer_mut().dry();
	node->get_inputs_buffer_mut().begin_read();
	node->get_inputs_buffer_mut().seek(METADATA_SIZE); // Skip meta.

	// The physics process is always emitted, because we still need to simulate
	// the character motion even if we don't store the player inputs.
	node->call(NetworkedController::sn_controller_process, p_delta, &node->get_inputs_buffer());

	node->player_set_has_new_input(false);
	if (accept_new_inputs) {
		if (streaming_paused == false) {
			input_buffers_counter += 1;
			store_input_buffer(current_input_id);
			send_frame_input_buffer_to_server();
			node->player_set_has_new_input(true);
		}
	}
}

int PlayerController::calculates_sub_ticks(real_t p_delta, real_t p_iteration_per_seconds) {
	const real_t pretended_delta = get_pretended_delta(p_iteration_per_seconds);

	time_bank += p_delta;
	const int sub_ticks = static_cast<uint32_t>(time_bank / pretended_delta);
	time_bank -= static_cast<real_t>(sub_ticks) * pretended_delta;
	return sub_ticks;
}

int PlayerController::notify_input_checked(uint32_t p_input_id) {
	if (frames_snapshot.empty() || p_input_id < frames_snapshot.front().id || p_input_id > frames_snapshot.back().id) {
		// The received p_input_id is not known, so nothing to do.
		NET_DEBUG_ERR("The received snapshot, with input id: " + itos(p_input_id) + " is not known. This is a bug or someone is trying to hack.");
		return frames_snapshot.size();
	}

	// Remove inputs prior to the known one. We may still need the known one
	// when the stream is paused.
	while (frames_snapshot.empty() == false && frames_snapshot.front().id <= p_input_id) {
		if (frames_snapshot.front().id == p_input_id) {
			streaming_paused = (frames_snapshot.front().buffer_size_bit - METADATA_SIZE) <= 0;
		}
		frames_snapshot.pop_front();
	}

#ifdef DEBUG_ENABLED
	// Unreachable, because the next input have always the next `p_input_id` or empty.
	CRASH_COND(frames_snapshot.empty() == false && (p_input_id + 1) != frames_snapshot.front().id);
#endif

	// Make sure the remaining inputs are 0 sized, if not streaming can't be paused.
	if (streaming_paused) {
		for (auto it = frames_snapshot.begin(); it != frames_snapshot.end(); it += 1) {
			if ((it->buffer_size_bit - METADATA_SIZE) > 0) {
				// Streaming can't be paused.
				streaming_paused = false;
				break;
			}
		}
	}

	return frames_snapshot.size();
}

uint32_t PlayerController::last_known_input() const {
	return get_stored_input_id(-1);
}

uint32_t PlayerController::get_stored_input_id(int p_i) const {
	if (p_i < 0) {
		if (frames_snapshot.empty() == false) {
			return frames_snapshot.back().id;
		} else {
			return UINT32_MAX;
		}
	} else {
		const size_t i = p_i;
		if (i < frames_snapshot.size()) {
			return frames_snapshot[i].id;
		} else {
			return UINT32_MAX;
		}
	}
}

bool PlayerController::process_instant(int p_i, real_t p_delta) {
	const size_t i = p_i;
	if (i < frames_snapshot.size()) {
		DataBuffer ib(frames_snapshot[i].inputs_buffer);
		ib.force_set_size(METADATA_SIZE, frames_snapshot[i].buffer_size_bit - METADATA_SIZE);
		ib.begin_read();
		ib.seek(METADATA_SIZE);
		node->call(NetworkedController::sn_controller_process, p_delta, &ib);
		return (i + 1) < frames_snapshot.size();
	} else {
		return false;
	}
}

uint32_t PlayerController::get_current_input_id() const {
	return current_input_id;
}

real_t PlayerController::get_pretended_delta(real_t p_iteration_per_seconds) const {
	return 1.0 / (p_iteration_per_seconds + tick_additional_speed);
}

void PlayerController::store_input_buffer(uint32_t p_id) {
	FrameSnapshot inputs;
	inputs.id = p_id;
	inputs.inputs_buffer = node->get_inputs_buffer().get_buffer();
	inputs.buffer_size_bit = node->get_inputs_buffer().size() + METADATA_SIZE;
	inputs.similarity = UINT32_MAX;
	frames_snapshot.push_back(inputs);
}

void PlayerController::send_frame_input_buffer_to_server() {
	// The packet is composed as follow:
	// - The following four bytes for the first input ID.
	// - Array of inputs:
	// |-- First byte the amount of times this input is duplicated in the packet.
	// |-- Input buffer.

	const size_t inputs_count = MIN(frames_snapshot.size(), static_cast<size_t>(node->get_max_redundant_inputs() + 1));
	CRASH_COND(inputs_count < 1); // Unreachable

#define MAKE_ROOM(p_size)                                              \
	if (cached_packet_data.size() < static_cast<size_t>(ofs + p_size)) \
		cached_packet_data.resize(ofs + p_size);

	int ofs = 0;

	// Let's store the ID of the first snapshot.
	MAKE_ROOM(4);
	const uint32_t first_input_id = frames_snapshot[frames_snapshot.size() - inputs_count].id;
	ofs += encode_uint32(first_input_id, cached_packet_data.ptr() + ofs);

	uint32_t previous_input_id = UINT32_MAX;
	uint32_t previous_input_similarity = UINT32_MAX;
	int previous_buffer_size = 0;
	uint8_t duplication_count = 0;

	DataBuffer pir_A(node->get_inputs_buffer().get_buffer());

	// Compose the packets
	for (size_t i = frames_snapshot.size() - inputs_count; i < frames_snapshot.size(); i += 1) {
		bool is_similar = false;

		if (previous_input_id == UINT32_MAX) {
			// This happens for the first input of the packet.
			// Just write it.
			is_similar = false;
		} else if (duplication_count == UINT8_MAX) {
			// Prevent to overflow the `uint8_t`.
			is_similar = false;
		} else {
			if (frames_snapshot[i].similarity != previous_input_id) {
				if (frames_snapshot[i].similarity == UINT32_MAX) {
					// This input was never compared, let's do it now.
					DataBuffer pir_B(frames_snapshot[i].inputs_buffer);
					pir_B.force_set_size(METADATA_SIZE, frames_snapshot[i].buffer_size_bit - METADATA_SIZE);

					pir_A.begin_read();
					pir_A.seek(METADATA_SIZE);
					pir_B.begin_read();
					pir_B.seek(METADATA_SIZE);

					const bool are_different = node->call(NetworkedController::sn_are_inputs_different, &pir_A, &pir_B);
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

			if (previous_input_id != UINT32_MAX) {
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
					cached_packet_data.ptr() + ofs,
					frames_snapshot[i].inputs_buffer.get_bytes().ptr(),
					buffer_size);
			ofs += buffer_size;

			// Let's see if we can duplicate this input.
			previous_input_id = frames_snapshot[i].id;
			previous_input_similarity = frames_snapshot[i].similarity;
			previous_buffer_size = buffer_size;

			pir_A.get_buffer_mut() = frames_snapshot[i].inputs_buffer;
			pir_A.force_set_size(METADATA_SIZE, frames_snapshot[i].buffer_size_bit - METADATA_SIZE);
		}
	}

	// Finalize the last added input_buffer.
	cached_packet_data[ofs - previous_buffer_size - 1] = duplication_count;

	// Make the packet data.
	Vector<uint8_t> packet_data;
	packet_data.resize(ofs);

	copymem(
			packet_data.ptrw(),
			cached_packet_data.ptr(),
			ofs);

	const int server_peer_id = 1;
	node->rpc_unreliable_id(
			server_peer_id,
			NetworkedController::sn_rpc_server_send_inputs,
			packet_data);
}

bool PlayerController::can_accept_new_inputs() const {
	return frames_snapshot.size() < static_cast<size_t>(node->get_player_input_storage_size());
}

DollController::DollController(NetworkedController *p_node) :
		Controller(p_node),
		network_watcher(node->get_doll_connection_stats_frame_span(), 0) {
}

DollController::~DollController() {
}

void DollController::ready() {
	interpolator.reset();
	node->call(
			NetworkedController::sn_setup_interpolator,
			&interpolator);
	interpolator.terminate_init();
}

void DollController::process(real_t p_delta) {
	const uint32_t frame_epoch = next_epoch(p_delta);

	if (unlikely(frame_epoch == UINT32_MAX)) {
		// Nothing to do.
		return;
	}

	const real_t fractional_part = advancing_epoch;
	node->call(
			NetworkedController::sn_apply_epoch,
			p_delta,
			interpolator.pop_epoch(frame_epoch, fractional_part));
}

uint32_t DollController::get_current_input_id() const {
	return current_epoch;
}

void DollController::receive_batch(Vector<uint8_t> p_data) {

	// Take the epochs befoe the batch is applied.
	const uint32_t youngest_epoch = interpolator.get_youngest_epoch();
	const uint32_t oldest_epoch = interpolator.get_oldest_epoch();

	int initially_stored_epochs = 0;
	if (youngest_epoch != UINT32_MAX && oldest_epoch != UINT32_MAX) {
		initially_stored_epochs = oldest_epoch - youngest_epoch;
	}

	initially_stored_epochs -= missing_epochs;
	missing_epochs = 0;

	uint32_t batch_young_epoch = UINT32_MAX;

	int buffer_start_position = 0;

	const uint8_t next_collect_rate = p_data[buffer_start_position];
	buffer_start_position += 1;

	while (buffer_start_position < p_data.size()) {
		const int buffer_size = p_data[buffer_start_position];
		const Vector<uint8_t> buffer = p_data.subarray(
				buffer_start_position + 1,
				buffer_start_position + 1 + buffer_size - 1);

		ERR_FAIL_COND(buffer.size() <= 0);

		const uint32_t epoch = receive_epoch(buffer);
		buffer_start_position += 1 + buffer_size;

		batch_young_epoch = MIN(epoch, batch_young_epoch);
	}

	// ~~ Establish the interpolation speed ~~
	if (batch_young_epoch == UINT32_MAX) {
		// This may just be a late arrived batch, so nothing more to do.
		return;
	}

	const real_t net_sentitivity = node->get_doll_net_sensitivity();

	// Establish the connection quality by checking if the batch takes
	// always the same time to arrive.
	const uint32_t now = OS::get_singleton()->get_ticks_msec();
	// If negative the timer was disabled, so just assume 0.
	network_watcher.push(MAX(0, now - batch_receiver_timer));
	batch_receiver_timer = now;

	const uint32_t avg_receive_time = network_watcher.average();
	const real_t deviation_receive_time = real_t(network_watcher.get_deviation(avg_receive_time)) / 1000.0;

	// The network quality can be established just by checking the standard
	// deviation. Stable connections have standard deviation that tend to 0.
	const real_t net_poorness = MIN(
			deviation_receive_time / net_sentitivity,
			1.0);

	const int optimal_frame_delay = Math::lerp(
			node->get_doll_min_frames_delay(),
			node->get_doll_max_frames_delay(),
			net_poorness);

	// TODO cache this?
	const double frames_per_batch = node->get_doll_epoch_batch_sync_rate() * real_t(Engine::get_singleton()->get_iterations_per_second());
	const double next_batch_arrives_in = Math::ceil(double(next_collect_rate) / frames_per_batch) * frames_per_batch;

	const real_t doll_interpolation_max_speedup = node->get_doll_interpolation_max_speedup();
	additional_speed = doll_interpolation_max_speedup * (real_t(initially_stored_epochs - optimal_frame_delay) / next_batch_arrives_in);
	additional_speed = CLAMP(additional_speed, -doll_interpolation_max_speedup, doll_interpolation_max_speedup);

#ifdef DEBUG_ENABLED
	const bool debug = ProjectSettings::get_singleton()->get_setting("NetworkSynchronizer/debug_doll_speedup");
	if (debug) {
		print_line("Net poorness: " + rtos(net_poorness) + " Optimal stored epochs: " + rtos(optimal_frame_delay) + " Deviation: " + rtos(deviation_receive_time));
	}
#endif
}

uint32_t DollController::receive_epoch(Vector<uint8_t> p_data) {
	DataBuffer buffer(p_data);
	buffer.begin_read();
	const uint32_t epoch = buffer.read_int(DataBuffer::COMPRESSION_LEVEL_1);

	if (epoch <= paused_epoch) {
		// The sync is in pause from this epoch, so just discard this received
		// epoch that may just be a late received epoch.
		return UINT32_MAX;
	}

	interpolator.begin_write(epoch);
	node->call(NetworkedController::sn_parse_epoch_data, &interpolator, &buffer);
	interpolator.end_write();

	return epoch;
}

uint32_t DollController::next_epoch(real_t p_delta) {
	// TODO re-describe.
	// This function regulates the epoch ID to process.
	// The epoch is not simply increased by one because we need to make sure
	// to make the client apply the nearest server state while giving some room
	// for the subsequent information to arrive.

	// Step 1, Wait that we have at least two epochs.
	if (unlikely(current_epoch == UINT32_MAX)) {
		// Interpolator is not yet started.
		if (interpolator.known_epochs_count() < 2) {
			// Not ready yet.
			return UINT32_MAX;
		}

#ifdef DEBUG_ENABLED
		// At this point we have 2 epoch, something is always returned at this
		// point.
		CRASH_COND(interpolator.get_youngest_epoch() == UINT32_MAX);
#endif

		// Start epoch interpolation.
		current_epoch = interpolator.get_youngest_epoch();
		node->emit_signal("doll_sync_started");
	}

	// At this point the interpolation is started and the function must
	// return the best epoch id which we have to apply the state.

	// Step 2. Make sure we have something to interpolate with.
	const uint32_t oldest_epoch = interpolator.get_oldest_epoch();
	if (unlikely(oldest_epoch == UINT32_MAX || oldest_epoch <= current_epoch)) {
		missing_epochs += 1;
		// Nothing to interpolate with.
		return current_epoch;
	}

#ifdef DEBUG_ENABLED
	// This can't happen because the current_epoch is advances only if it's
	// possible to do so.
	CRASH_COND(oldest_epoch < current_epoch);
#endif

	const uint64_t max_virtual_delay = 60; // TODO make this a parameter

	if (unlikely((oldest_epoch - current_epoch) > max_virtual_delay)) {
		// This client seems too much behind at this point. Teleport forward.
		const uint32_t youngest_epoch = interpolator.get_youngest_epoch();
		current_epoch = MAX(oldest_epoch - max_virtual_delay, youngest_epoch);
	} else {
		advancing_epoch += 1.0 + additional_speed;
	}

	if (advancing_epoch > 0.0) {
		// Advance the epoch by the the integral amount.
		current_epoch += uint32_t(advancing_epoch);
		// Clamp to the oldest epoch.
		current_epoch = MIN(current_epoch, oldest_epoch);

		// Keep the floating point part.
		advancing_epoch -= uint32_t(advancing_epoch);
	}

	return current_epoch;
}

void DollController::pause(uint32_t p_epoch) {
	paused_epoch = p_epoch;

	interpolator.clear();
	additional_speed = 0.0;
	current_epoch = UINT32_MAX;
	advancing_epoch = 0.0;
	missing_epochs = 0;
	network_watcher.resize(node->get_doll_connection_stats_frame_span(), 0);
	batch_receiver_timer = UINT32_MAX;

	node->emit_signal("doll_sync_paused");
}

NoNetController::NoNetController(NetworkedController *p_node) :
		Controller(p_node),
		frame_id(0) {
}

void NoNetController::process(real_t p_delta) {
	node->get_inputs_buffer_mut().begin_write(0); // No need of meta in this case.
	node->call(NetworkedController::sn_collect_inputs, p_delta, &node->get_inputs_buffer_mut());
	node->get_inputs_buffer_mut().dry();
	node->get_inputs_buffer_mut().begin_read();
	node->call(NetworkedController::sn_controller_process, p_delta, &node->get_inputs_buffer_mut());
	frame_id += 1;
}

uint32_t NoNetController::get_current_input_id() const {
	return frame_id;
}
