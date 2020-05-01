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
#include "scene_rewinder.h"
#include <stdint.h>
#include <algorithm>

void NetworkedController::_bind_methods() {

	BIND_CONSTANT(INPUT_COMPRESSION_LEVEL_0);
	BIND_CONSTANT(INPUT_COMPRESSION_LEVEL_1);
	BIND_CONSTANT(INPUT_COMPRESSION_LEVEL_2);
	BIND_CONSTANT(INPUT_COMPRESSION_LEVEL_3);

	ClassDB::bind_method(D_METHOD("set_player_input_storage_size", "size"), &NetworkedController::set_player_input_storage_size);
	ClassDB::bind_method(D_METHOD("get_player_input_storage_size"), &NetworkedController::get_player_input_storage_size);

	ClassDB::bind_method(D_METHOD("set_max_redundant_inputs", "max_redundand_inputs"), &NetworkedController::set_max_redundant_inputs);
	ClassDB::bind_method(D_METHOD("get_max_redundant_inputs"), &NetworkedController::get_max_redundant_inputs);

	ClassDB::bind_method(D_METHOD("get_current_input_id"), &NetworkedController::get_current_input_id);

	ClassDB::bind_method(D_METHOD("input_buffer_add_bool", "bool", "compression_level"), &NetworkedController::input_buffer_add_bool, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("input_buffer_read_bool", "compression_level"), &NetworkedController::input_buffer_read_bool, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("input_buffer_add_int", "int", "compression_level"), &NetworkedController::input_buffer_add_int, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("input_buffer_read_int", "compression_level"), &NetworkedController::input_buffer_read_int, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("input_buffer_add_unit_real", "unit_real", "compression_level"), &NetworkedController::input_buffer_add_unit_real, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("input_buffer_read_unit_real", "compression_level"), &NetworkedController::input_buffer_read_unit_real, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("input_buffer_add_normalized_vector2", "vector", "compression_level"), &NetworkedController::input_buffer_add_normalized_vector2, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("input_buffer_read_normalized_vector2", "compression_level"), &NetworkedController::input_buffer_read_normalized_vector2, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("input_buffer_add_normalized_vector3", "vector", "compression_level"), &NetworkedController::input_buffer_add_normalized_vector3, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("input_buffer_read_normalized_vector3", "compression_level"), &NetworkedController::input_buffer_read_normalized_vector3, DEFVAL(InputCompressionLevel::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("set_doll_peer_active", "peer_id", "active"), &NetworkedController::set_doll_peer_active);
	ClassDB::bind_method(D_METHOD("_on_peer_connection_change", "peer_id"), &NetworkedController::_on_peer_connection_change);

	ClassDB::bind_method(D_METHOD("_rpc_server_send_inputs"), &NetworkedController::_rpc_server_send_inputs);
	ClassDB::bind_method(D_METHOD("_rpc_doll_send_inputs"), &NetworkedController::_rpc_doll_send_inputs);
	ClassDB::bind_method(D_METHOD("_rpc_doll_notify_connection_status"), &NetworkedController::_rpc_doll_notify_connection_status);

	ClassDB::bind_method(D_METHOD("is_server_controller"), &NetworkedController::is_server_controller);
	ClassDB::bind_method(D_METHOD("is_player_controller"), &NetworkedController::is_player_controller);
	ClassDB::bind_method(D_METHOD("is_doll_controller"), &NetworkedController::is_doll_controller);
	ClassDB::bind_method(D_METHOD("is_nonet_controller"), &NetworkedController::is_nonet_controller);

	BIND_VMETHOD(MethodInfo("collect_inputs", PropertyInfo(Variant::FLOAT, "delta")));
	BIND_VMETHOD(MethodInfo("controller_process", PropertyInfo(Variant::FLOAT, "delta")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "are_inputs_different", PropertyInfo(Variant::OBJECT, "inputs_A", PROPERTY_HINT_TYPE_STRING, "PlayerInputsReference"), PropertyInfo(Variant::OBJECT, "inputs_B", PROPERTY_HINT_TYPE_STRING, "PlayerInputsReference")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "count_input_size", PropertyInfo(Variant::OBJECT, "inputs", PROPERTY_HINT_TYPE_STRING, "PlayerInputsReference")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "input_storage_size", PROPERTY_HINT_RANGE, "100,2000,1"), "set_player_input_storage_size", "get_player_input_storage_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_redundant_inputs", PROPERTY_HINT_RANGE, "0,1000,1"), "set_max_redundant_inputs", "get_max_redundant_inputs");

	ADD_SIGNAL(MethodInfo("doll_server_comunication_opened"));
	ADD_SIGNAL(MethodInfo("doll_server_comunication_closed"));
}

NetworkedController::NetworkedController() :
		player_input_storage_size(300),
		max_redundant_inputs(50),
		controller(nullptr),
		scene_rewinder(nullptr),
		packet_missing(false),
		has_player_new_input(false) {

	rpc_config("_rpc_server_send_inputs", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_doll_send_inputs", MultiplayerAPI::RPC_MODE_REMOTE);
	rpc_config("_rpc_doll_notify_connection_status", MultiplayerAPI::RPC_MODE_REMOTE);
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

uint64_t NetworkedController::get_current_input_id() const {
	return controller->get_current_input_id();
}

bool NetworkedController::input_buffer_add_bool(bool p_input, InputCompressionLevel _p_compression) {
	return inputs_buffer.add_bool(p_input);
}

bool NetworkedController::input_buffer_read_bool(InputCompressionLevel _p_compression) {
	return inputs_buffer.read_bool();
}

int64_t NetworkedController::input_buffer_add_int(int64_t p_input, InputCompressionLevel p_compression) {
	return inputs_buffer.add_int(p_input, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

int64_t NetworkedController::input_buffer_read_int(InputCompressionLevel p_compression) {
	return inputs_buffer.read_int(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

real_t NetworkedController::input_buffer_add_unit_real(real_t p_input, InputCompressionLevel p_compression) {
	return inputs_buffer.add_unit_real(p_input, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

real_t NetworkedController::input_buffer_read_unit_real(InputCompressionLevel p_compression) {
	return inputs_buffer.read_unit_real(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

Vector2 NetworkedController::input_buffer_add_normalized_vector2(Vector2 p_input, InputCompressionLevel p_compression) {
	return inputs_buffer.add_normalized_vector2(p_input, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

Vector2 NetworkedController::input_buffer_read_normalized_vector2(InputCompressionLevel p_compression) {
	return inputs_buffer.read_normalized_vector2(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

Vector3 NetworkedController::input_buffer_add_normalized_vector3(Vector3 p_input, InputCompressionLevel p_compression) {
	return inputs_buffer.add_normalized_vector3(p_input, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

Vector3 NetworkedController::input_buffer_read_normalized_vector3(InputCompressionLevel p_compression) {
	return inputs_buffer.read_normalized_vector3(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

void NetworkedController::set_doll_peer_active(int p_peer_id, bool p_active) {
	ERR_FAIL_COND_MSG(get_tree()->is_network_server() == false, "You can set doll activation only on server");
	ERR_FAIL_COND_MSG(p_peer_id == get_network_master(), "This `peer_id` is equals to the Master `peer_id`, so it's not allowed.");

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

const Vector<int> &NetworkedController::get_active_doll_peers() const {
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

int NetworkedController::notify_input_checked(uint64_t p_input_id) {
	return controller->notify_input_checked(p_input_id);
}

uint64_t NetworkedController::get_stored_input_id(int p_i) const {
	return controller->get_stored_input_id(p_i);
}

bool NetworkedController::process_instant(int p_i, real_t p_delta) {
	return controller->process_instant(p_i, p_delta);
}

bool NetworkedController::is_server_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller)
		return dynamic_cast<ServerController *>(controller);
	return get_tree()->is_network_server();
}

bool NetworkedController::is_player_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller)
		return dynamic_cast<PlayerController *>(controller);
	return get_tree()->is_network_server() == false && is_network_master();
}

bool NetworkedController::is_doll_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller)
		return dynamic_cast<DollController *>(controller);
	return get_tree()->is_network_server() == false && is_network_master() == false;
}

bool NetworkedController::is_nonet_controller() const {
	ERR_FAIL_COND_V(get_tree() == nullptr, false);
	if (controller)
		return dynamic_cast<NoNetController *>(controller);
	return get_tree()->get_network_peer().is_null();
}

void NetworkedController::set_packet_missing(bool p_missing) {
	packet_missing = p_missing;
}

bool NetworkedController::get_packet_missing() const {
	return packet_missing;
}

void NetworkedController::set_inputs_buffer(const BitArray &p_new_buffer) {
	inputs_buffer.get_buffer_mut().get_bytes_mut() = p_new_buffer.get_bytes();
}

void NetworkedController::set_scene_rewinder(SceneRewinder *p_rewinder) {
	scene_rewinder = p_rewinder;
}

SceneRewinder *NetworkedController::get_scene_rewinder() const {
	return scene_rewinder;
}

bool NetworkedController::has_scene_rewinder() const {
	return scene_rewinder;
}

void NetworkedController::_rpc_server_send_inputs(Vector<uint8_t> p_data) {
	ERR_FAIL_COND(get_tree()->is_network_server() == false);

	const Vector<int> &peers = get_active_doll_peers();
	for (int i = 0; i < peers.size(); i += 1) {
		// This is an active doll, Let's send the data.
		const int peer_id = peers[i];
		rpc_unreliable_id(peer_id, "_rpc_doll_send_inputs", p_data);
	}

	controller->receive_inputs(p_data);
}

void NetworkedController::_rpc_doll_send_inputs(Vector<uint8_t> p_data) {
	ERR_FAIL_COND_MSG(get_tree()->is_network_server() == true, "This controller is not supposed to receive this call, make sure the controllers node have the same name across all peers.");
	ERR_FAIL_COND_MSG(is_network_master() == true, "This controller is not supposed to receive this call, make sure the controllers node have the same name across all peers.");

	controller->receive_inputs(p_data);
}

void NetworkedController::_rpc_doll_notify_connection_status(bool p_open) {
	ERR_FAIL_COND_MSG(get_tree()->is_network_server() == true, "This controller is not supposed to receive this call, make sure the controllers node have the same name across all peers.");
	ERR_FAIL_COND_MSG(is_network_master() == true, "This controller is not supposed to receive this call, make sure the controllers node have the same name across all peers.");

	if (p_open) {
		static_cast<DollController *>(controller)->open_flow();
	} else {
		static_cast<DollController *>(controller)->close_flow();
	}
}

void NetworkedController::process(real_t p_delta) {
	if (controller) {
		// This is called by the `sceneRewinder` that it's not aware about the
		// controller state; so check that the controller is not null.
		controller->physics_process(p_delta);
	}
}

int NetworkedController::server_get_inputs_count() const {
	if (controller)
		return is_server_controller() ? static_cast<ServerController *>(controller)->get_inputs_count() : 0;
	return 0;
}

void NetworkedController::player_set_has_new_input(bool p_has) {
	has_player_new_input = p_has;
}

bool NetworkedController::player_has_new_input() const {
	return has_player_new_input;
}

void NetworkedController::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			// Unreachable.
			CRASH_COND(get_tree() == NULL);

			if (get_tree()->get_network_peer().is_null()) {
				controller = memnew(NoNetController(this));
			} else if (get_tree()->is_network_server()) {
				controller = memnew(ServerController(this));
				get_multiplayer()->connect("network_peer_connected", callable_mp(this, &NetworkedController::_on_peer_connection_change));
				get_multiplayer()->connect("network_peer_disconnected", callable_mp(this, &NetworkedController::_on_peer_connection_change));
				update_active_doll_peers();
			} else if (is_network_master()) {
				controller = memnew(PlayerController(this));
			} else {
				controller = memnew(DollController(this));
			}

			ERR_FAIL_COND_MSG(has_method("collect_inputs") == false, "In your script you must inherit the virtual method `collect_inputs` to correctly use the `PlayerNetController`.");
			ERR_FAIL_COND_MSG(has_method("controller_process") == false, "In your script you must inherit the virtual method `controller_process` to correctly use the `PlayerNetController`.");
			ERR_FAIL_COND_MSG(has_method("are_inputs_different") == false, "In your script you must inherit the virtual method `are_inputs_different` to correctly use the `PlayerNetController`.");
			ERR_FAIL_COND_MSG(has_method("count_input_size") == false, "In your script you must inherit the virtual method `count_input_size` to correctly use the `PlayerNetController`.");
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (Engine::get_singleton()->is_editor_hint())
				return;

			memdelete(controller);
			controller = NULL;

			if (get_tree()->is_network_server()) {
				get_multiplayer()->disconnect("network_peer_connected", callable_mp(this, &NetworkedController::_on_peer_connection_change));
				get_multiplayer()->disconnect("network_peer_disconnected", callable_mp(this, &NetworkedController::_on_peer_connection_change));
			}
		} break;
	}
}

void PlayerInputsReference::_bind_methods() {
	ClassDB::bind_method(D_METHOD("read_bool", "compression_level"), &PlayerInputsReference::read_bool, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_int", "compression_level"), &PlayerInputsReference::read_int, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_unit_real", "compression_level"), &PlayerInputsReference::read_unit_real, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector2", "compression_level"), &PlayerInputsReference::read_normalized_vector2, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("read_normalized_vector3", "compression_level"), &PlayerInputsReference::read_normalized_vector3, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("skip_bool", "compression_level"), &PlayerInputsReference::skip_bool, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_int", "compression_level"), &PlayerInputsReference::skip_int, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_unit_real", "compression_level"), &PlayerInputsReference::skip_unit_real, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_normalized_vector2", "compression_level"), &PlayerInputsReference::skip_normalized_vector2, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("skip_normalized_vector3", "compression_level"), &PlayerInputsReference::skip_normalized_vector3, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));

	ClassDB::bind_method(D_METHOD("get_bool_size", "compression_level"), &PlayerInputsReference::get_bool_size, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_int_size", "compression_level"), &PlayerInputsReference::get_int_size, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_unit_real_size", "compression_level"), &PlayerInputsReference::get_unit_real_size, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_normalized_vector2_size", "compression_level"), &PlayerInputsReference::get_normalized_vector2_size, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
	ClassDB::bind_method(D_METHOD("get_normalized_vector3_size", "compression_level"), &PlayerInputsReference::get_normalized_vector3_size, DEFVAL(NetworkedController::INPUT_COMPRESSION_LEVEL_1));
}

int PlayerInputsReference::get_size() const {
	return inputs_buffer.get_buffer_size();
}

bool PlayerInputsReference::read_bool(NetworkedController::InputCompressionLevel _p_compression) {
	return inputs_buffer.read_bool();
}

int64_t PlayerInputsReference::read_int(NetworkedController::InputCompressionLevel p_compression) {
	return inputs_buffer.read_int(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

real_t PlayerInputsReference::read_unit_real(NetworkedController::InputCompressionLevel p_compression) {
	return inputs_buffer.read_unit_real(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

Vector2 PlayerInputsReference::read_normalized_vector2(NetworkedController::InputCompressionLevel p_compression) {
	return inputs_buffer.read_normalized_vector2(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

Vector3 PlayerInputsReference::read_normalized_vector3(NetworkedController::InputCompressionLevel p_compression) {
	return inputs_buffer.read_normalized_vector3(static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

void PlayerInputsReference::skip_bool(NetworkedController::InputCompressionLevel p_compression) {
	const int bits = get_bool_size(p_compression);
	inputs_buffer.skip(bits);
}

void PlayerInputsReference::skip_int(NetworkedController::InputCompressionLevel p_compression) {
	const int bits = get_int_size(p_compression);
	inputs_buffer.skip(bits);
}

void PlayerInputsReference::skip_unit_real(NetworkedController::InputCompressionLevel p_compression) {
	const int bits = get_unit_real_size(p_compression);
	inputs_buffer.skip(bits);
}

void PlayerInputsReference::skip_normalized_vector2(NetworkedController::InputCompressionLevel p_compression) {
	const int bits = get_normalized_vector2_size(p_compression);
	inputs_buffer.skip(bits);
}

void PlayerInputsReference::skip_normalized_vector3(NetworkedController::InputCompressionLevel p_compression) {
	const int bits = get_normalized_vector3_size(p_compression);
	inputs_buffer.skip(bits);
}

int PlayerInputsReference::get_bool_size(NetworkedController::InputCompressionLevel p_compression) const {
	return InputsBuffer::get_bit_taken(InputsBuffer::DATA_TYPE_BOOL, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

int PlayerInputsReference::get_int_size(NetworkedController::InputCompressionLevel p_compression) const {
	return InputsBuffer::get_bit_taken(InputsBuffer::DATA_TYPE_INT, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

int PlayerInputsReference::get_unit_real_size(NetworkedController::InputCompressionLevel p_compression) const {
	return InputsBuffer::get_bit_taken(InputsBuffer::DATA_TYPE_UNIT_REAL, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

int PlayerInputsReference::get_normalized_vector2_size(NetworkedController::InputCompressionLevel p_compression) const {
	return InputsBuffer::get_bit_taken(InputsBuffer::DATA_TYPE_NORMALIZED_VECTOR2, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

int PlayerInputsReference::get_normalized_vector3_size(NetworkedController::InputCompressionLevel p_compression) const {
	return InputsBuffer::get_bit_taken(InputsBuffer::DATA_TYPE_NORMALIZED_VECTOR3, static_cast<InputsBuffer::CompressionLevel>(p_compression));
}

void PlayerInputsReference::begin() {
	inputs_buffer.begin_read();
}

void PlayerInputsReference::set_inputs_buffer(const BitArray &p_new_buffer) {
	inputs_buffer.get_buffer_mut().get_bytes_mut() = p_new_buffer.get_bytes();
}

ServerController::ServerController(NetworkedController *p_node) :
		Controller(p_node),
		current_input_buffer_id(UINT64_MAX),
		ghost_input_count(0) {
}

void ServerController::physics_process(real_t p_delta) {
	fetch_next_input();

	if (unlikely(current_input_buffer_id == UINT64_MAX)) {
		// Skip this until the first input arrive.
		return;
	}

	node->get_inputs_buffer_mut().begin_read();
	node->call("controller_process", p_delta);
}

bool is_remote_frame_A_older(const FrameSnapshotSkinny &p_snap_a, const FrameSnapshotSkinny &p_snap_b) {
	return p_snap_a.id < p_snap_b.id;
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

	BitArray bit_array;
	bit_array.resize_in_bytes(data_len);
	copymem(bit_array.get_bytes_mut().ptrw(), p_data.ptr(), data_len);

	// Contains the entire packet and in turn it will be seek to specific location
	// so I will not need to copy chunk of the packet data.
	PlayerInputsReference pir;
	pir.set_inputs_buffer(bit_array);

	while (ofs < data_len) {

		ERR_FAIL_COND_MSG(ofs + 1 > data_len, "The arrived packet size doesn't meet the expected size.");
		// First byte is used for the duplication count.
		const uint8_t duplication = p_data[ofs];
		ofs += 1;

		// Validate input
		pir.inputs_buffer.seek(ofs * 8);
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

int ServerController::notify_input_checked(uint64_t p_input_id) {
	ERR_PRINT("The method `notify_input_checked` must not be called on server. Be sure why it happened.");
	return 0;
}

uint64_t ServerController::get_stored_input_id(int p_i) const {
	ERR_PRINT("The method `get_input_id` must not be called on server. Be sure why it happened.");
	return UINT64_MAX;
}

bool ServerController::process_instant(int p_i, real_t p_delta) {
	ERR_PRINT("The method `process_instant` must not be called on server. Be sure why it happened.");
	return false;
}

uint64_t ServerController::get_current_input_id() const {
	return current_input_buffer_id;
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
			node->set_inputs_buffer(snapshots.front().inputs_buffer);
			current_input_buffer_id = snapshots.front().id;
			snapshots.pop_front();
		} else {
			is_new_input = false;
		}
		// Never notify a packet is missing untill the first one is not yet
		// arrived.
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
			WARN_PRINT("Input buffer is void, i'm using the previous one!");

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

				PlayerInputsReference pir_A(node->get_inputs_buffer());
				// Copy from the node inputs so to copy the data info
				PlayerInputsReference pir_B(node->get_inputs_buffer());

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

						pir_B.set_inputs_buffer(pi.inputs_buffer);

						pir_A.begin();
						pir_B.begin();

						const bool is_meaningful = pir_A.get_size() != pir_B.get_size() || node->call("are_inputs_different", &pir_A, &pir_B);

						if (is_meaningful) {
							break;
						}
					}
				}

				if (recovered) {
					node->set_inputs_buffer(pi.inputs_buffer);
					current_input_buffer_id = pi.id;
					ghost_input_count = 0;
					WARN_PRINT("Packet recovered");
				} else {
					is_new_input = false;
					WARN_PRINT("Packet still missing");
				}
			}
		}
	}

	node->set_packet_missing(is_packet_missing);

	return is_new_input;
}

PlayerController::PlayerController(NetworkedController *p_node) :
		Controller(p_node),
		current_input_id(UINT64_MAX),
		input_buffers_counter(0) {
}

void PlayerController::physics_process(real_t p_delta) {

	// We need to know if we can accept a new input because in case of bad
	// internet connection we can't keep accumulates inputs up to infinite
	// otherwise the server will differ too much from the client and we
	// introduce virtual lag.
	const bool accept_new_inputs = can_accept_new_inputs();

	if (accept_new_inputs) {
		current_input_id = input_buffers_counter;
		input_buffers_counter += 1;
		node->get_inputs_buffer_mut().begin_write();
		node->call("collect_inputs", p_delta);
	} else {
		WARN_PRINT("It's not possible to accept new inputs. Is this lagging?");
	}

	node->get_inputs_buffer_mut().dry();
	node->get_inputs_buffer_mut().begin_read();

	// The physics process is always emitted, because we still need to simulate
	// the character motion even if we don't store the player inputs.
	node->call("controller_process", p_delta);

	if (accept_new_inputs) {
		store_input_buffer(current_input_id);
		send_frame_input_buffer_to_server();
	}

	node->player_set_has_new_input(accept_new_inputs);
}

void PlayerController::receive_inputs(Vector<uint8_t> p_data) {
	ERR_PRINT("The player is not supposed to receive snapshots. Check why this happened.");
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

		// Set snapshot inputs.
		node->set_inputs_buffer(frames_snapshot[i].inputs_buffer);

		node->get_inputs_buffer_mut().begin_read();
		node->call("controller_process", p_delta);
		return (i + 1) < frames_snapshot.size();
	} else {
		return false;
	}
}

uint64_t PlayerController::get_current_input_id() const {
	return current_input_id;
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

	PlayerInputsReference pir_A(node->get_inputs_buffer());
	// Copy from the node inputs so to copy the data info.
	PlayerInputsReference pir_B(node->get_inputs_buffer());

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
					pir_B.set_inputs_buffer(frames_snapshot[i].inputs_buffer);

					pir_A.begin();
					pir_B.begin();

					const bool are_different = pir_A.get_size() != pir_B.get_size() || node->call("are_inputs_different", &pir_A, &pir_B);
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

			pir_A.set_inputs_buffer(frames_snapshot[i].inputs_buffer);
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
		server_controller(p_node),
		player_controller(p_node),
		is_server_communication_detected(false),
		is_server_state_update_received(false),
		last_checked_input_id(0),
		is_flow_open(true) {
}

void DollController::physics_process(real_t p_delta) {

	// Lock mechanism when the server don't update anymore this doll!
	if (is_flow_open) {
		if (is_server_state_update_received &&
				is_server_communication_detected == false) {
			is_server_communication_detected = true;
			hard_reset_to_server_state();
			node->emit_signal("doll_server_comunication_opened");
		}
	} else {
		// Locked
		node->player_set_has_new_input(false);
		return;
	}

	const bool is_new_input = server_controller.fetch_next_input();
	node->get_inputs_buffer_mut().begin_read();
	node->call("controller_process", p_delta);
	if (is_new_input) {
		player_controller.store_input_buffer(server_controller.current_input_buffer_id);
	}

	node->player_set_has_new_input(is_new_input);

	// Keeps the doll in sync with the server.
	soft_reset_to_server_state();
}

void DollController::receive_inputs(Vector<uint8_t> p_data) {
	if (is_flow_open == false)
		return;
	server_controller.receive_inputs(p_data);
}

int DollController::notify_input_checked(uint64_t p_input_id) {
	if (is_flow_open) {
		last_checked_input_id = p_input_id;
		is_server_state_update_received = true;
	}
	return player_controller.notify_input_checked(p_input_id);
}

uint64_t DollController::get_stored_input_id(int p_i) const {
	return player_controller.get_stored_input_id(p_i);
}

bool DollController::process_instant(int p_i, real_t p_delta) {
	return player_controller.process_instant(p_i, p_delta);
}

uint64_t DollController::get_current_input_id() const {
	return server_controller.current_input_buffer_id;
}

void DollController::open_flow() {
	if (is_flow_open == true)
		return;
	is_flow_open = true;
	is_server_communication_detected = false;
	is_server_state_update_received = false;
}

void DollController::close_flow() {
	if (is_flow_open == false)
		return;
	is_flow_open = false;
	node->emit_signal("doll_server_comunication_closed");
}

void DollController::soft_reset_to_server_state() {

	bool reset = false;

	for (std::deque<FrameSnapshotSkinny>::const_reverse_iterator it = server_controller.snapshots.rbegin();
			it != server_controller.snapshots.rend();
			it += 1) {
		if (it->id == last_checked_input_id) {
			reset = true;
			break;
		}
	}

	if (reset) {
		hard_reset_to_server_state();
	}
}

void DollController::hard_reset_to_server_state() {
	// Reset the doll to server state.

	// Drop the current frame_snapshot since are now useless.
	player_controller.frames_snapshot.clear();

	// Set the server `current_packet_id` to the previous `last_recovered_input_id`
	// so the next frame the input to recover is inserted naturally into
	// the player controller `snapshots` array.
	server_controller.current_input_buffer_id = last_checked_input_id - 1;

	// Drop all the snapshots that are older that the latest state update
	// so to sync server and client back again.
	while (server_controller.snapshots.size() > 0 && server_controller.snapshots.front().id < server_controller.current_input_buffer_id) {
		server_controller.snapshots.pop_front();
	}

	// The next frame the recover mechanism will naturally reset the player
	// state back in sync.
}

NoNetController::NoNetController(NetworkedController *p_node) :
		Controller(p_node),
		frame_id(0) {
}

void NoNetController::physics_process(real_t p_delta) {
	node->get_inputs_buffer_mut().begin_write();
	node->call("collect_inputs", p_delta);
	node->get_inputs_buffer_mut().dry();
	node->get_inputs_buffer_mut().begin_read();
	node->call("controller_process", p_delta);
	frame_id += 1;
}

void NoNetController::receive_inputs(Vector<uint8_t> p_data) {
	// Nothing to do.
}

int NoNetController::notify_input_checked(uint64_t p_input_id) {
	// Nothing to do.
	return 0;
}

uint64_t NoNetController::get_stored_input_id(int p_i) const {
	// Nothing to do.
	return UINT64_MAX;
}

bool NoNetController::process_instant(int p_i, real_t p_delta) {
	// Nothing to do.
	return false;
}

uint64_t NoNetController::get_current_input_id() const {
	return frame_id;
}
