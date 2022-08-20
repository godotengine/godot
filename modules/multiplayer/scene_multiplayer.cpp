/*************************************************************************/
/*  scene_multiplayer.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene_multiplayer.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"

#include <stdint.h>

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#endif

#ifdef DEBUG_ENABLED
void SceneMultiplayer::profile_bandwidth(const String &p_inout, int p_size) {
	if (EngineDebugger::is_profiling("multiplayer")) {
		Array values;
		values.push_back(p_inout);
		values.push_back(OS::get_singleton()->get_ticks_msec());
		values.push_back(p_size);
		EngineDebugger::profiler_add_frame_data("multiplayer", values);
	}
}
#endif

Error SceneMultiplayer::poll() {
	if (!multiplayer_peer.is_valid() || multiplayer_peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED) {
		return ERR_UNCONFIGURED;
	}

	multiplayer_peer->poll();

	if (!multiplayer_peer.is_valid()) { // It's possible that polling might have resulted in a disconnection, so check here.
		return OK;
	}

	while (multiplayer_peer->get_available_packet_count()) {
		int sender = multiplayer_peer->get_packet_peer();
		const uint8_t *packet;
		int len;

		Error err = multiplayer_peer->get_packet(&packet, len);
		ERR_FAIL_COND_V_MSG(err != OK, err, vformat("Error getting packet! %d", err));

		remote_sender_id = sender;
		_process_packet(sender, packet, len);
		remote_sender_id = 0;

		if (!multiplayer_peer.is_valid()) {
			return OK; // It's also possible that a packet or RPC caused a disconnection, so also check here.
		}
	}
	replicator->on_network_process();
	return OK;
}

void SceneMultiplayer::clear() {
	connected_peers.clear();
	packet_cache.clear();
	cache->clear();
}

void SceneMultiplayer::set_root_path(const NodePath &p_path) {
	ERR_FAIL_COND_MSG(!p_path.is_absolute() && !p_path.is_empty(), "SceneMultiplayer root path must be absolute.");
	root_path = p_path;
}

NodePath SceneMultiplayer::get_root_path() const {
	return root_path;
}

void SceneMultiplayer::set_multiplayer_peer(const Ref<MultiplayerPeer> &p_peer) {
	if (p_peer == multiplayer_peer) {
		return; // Nothing to do
	}

	ERR_FAIL_COND_MSG(p_peer.is_valid() && p_peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED,
			"Supplied MultiplayerPeer must be connecting or connected.");

	if (multiplayer_peer.is_valid()) {
		multiplayer_peer->disconnect("peer_connected", callable_mp(this, &SceneMultiplayer::_add_peer));
		multiplayer_peer->disconnect("peer_disconnected", callable_mp(this, &SceneMultiplayer::_del_peer));
		multiplayer_peer->disconnect("connection_succeeded", callable_mp(this, &SceneMultiplayer::_connected_to_server));
		multiplayer_peer->disconnect("connection_failed", callable_mp(this, &SceneMultiplayer::_connection_failed));
		multiplayer_peer->disconnect("server_disconnected", callable_mp(this, &SceneMultiplayer::_server_disconnected));
		clear();
	}

	multiplayer_peer = p_peer;

	if (multiplayer_peer.is_valid()) {
		multiplayer_peer->connect("peer_connected", callable_mp(this, &SceneMultiplayer::_add_peer));
		multiplayer_peer->connect("peer_disconnected", callable_mp(this, &SceneMultiplayer::_del_peer));
		multiplayer_peer->connect("connection_succeeded", callable_mp(this, &SceneMultiplayer::_connected_to_server));
		multiplayer_peer->connect("connection_failed", callable_mp(this, &SceneMultiplayer::_connection_failed));
		multiplayer_peer->connect("server_disconnected", callable_mp(this, &SceneMultiplayer::_server_disconnected));
	}
	replicator->on_reset();
}

Ref<MultiplayerPeer> SceneMultiplayer::get_multiplayer_peer() {
	return multiplayer_peer;
}

void SceneMultiplayer::_process_packet(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(root_path.is_empty(), "Multiplayer root was not initialized. If you are using custom multiplayer, remember to set the root path via SceneMultiplayer.set_root_path before using it.");
	ERR_FAIL_COND_MSG(p_packet_len < 1, "Invalid packet received. Size too small.");

#ifdef DEBUG_ENABLED
	profile_bandwidth("in", p_packet_len);
#endif

	// Extract the `packet_type` from the LSB three bits:
	uint8_t packet_type = p_packet[0] & CMD_MASK;

	switch (packet_type) {
		case NETWORK_COMMAND_SIMPLIFY_PATH: {
			cache->process_simplify_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_CONFIRM_PATH: {
			cache->process_confirm_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_REMOTE_CALL: {
			rpc->process_rpc(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_RAW: {
			_process_raw(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_SPAWN: {
			replicator->on_spawn_receive(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_DESPAWN: {
			replicator->on_despawn_receive(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_SYNC: {
			replicator->on_sync_receive(p_from, p_packet, p_packet_len);
		} break;
	}
}

void SceneMultiplayer::_add_peer(int p_id) {
	connected_peers.insert(p_id);
	cache->on_peer_change(p_id, true);
	replicator->on_peer_change(p_id, true);
	emit_signal(SNAME("peer_connected"), p_id);
}

void SceneMultiplayer::_del_peer(int p_id) {
	replicator->on_peer_change(p_id, false);
	cache->on_peer_change(p_id, false);
	connected_peers.erase(p_id);
	emit_signal(SNAME("peer_disconnected"), p_id);
}

void SceneMultiplayer::_connected_to_server() {
	emit_signal(SNAME("connected_to_server"));
}

void SceneMultiplayer::_connection_failed() {
	emit_signal(SNAME("connection_failed"));
}

void SceneMultiplayer::_server_disconnected() {
	replicator->on_reset();
	emit_signal(SNAME("server_disconnected"));
}

Error SceneMultiplayer::send_bytes(Vector<uint8_t> p_data, int p_to, MultiplayerPeer::TransferMode p_mode, int p_channel) {
	ERR_FAIL_COND_V_MSG(p_data.size() < 1, ERR_INVALID_DATA, "Trying to send an empty raw packet.");
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), ERR_UNCONFIGURED, "Trying to send a raw packet while no multiplayer peer is active.");
	ERR_FAIL_COND_V_MSG(multiplayer_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED, ERR_UNCONFIGURED, "Trying to send a raw packet via a multiplayer peer which is not connected.");

	if (packet_cache.size() < p_data.size() + 1) {
		packet_cache.resize(p_data.size() + 1);
	}

	const uint8_t *r = p_data.ptr();
	packet_cache.write[0] = NETWORK_COMMAND_RAW;
	memcpy(&packet_cache.write[1], &r[0], p_data.size());

	multiplayer_peer->set_target_peer(p_to);
	multiplayer_peer->set_transfer_channel(p_channel);
	multiplayer_peer->set_transfer_mode(p_mode);

	return multiplayer_peer->put_packet(packet_cache.ptr(), p_data.size() + 1);
}

void SceneMultiplayer::_process_raw(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(p_packet_len < 2, "Invalid packet received. Size too small.");

	Vector<uint8_t> out;
	int len = p_packet_len - 1;
	out.resize(len);
	{
		uint8_t *w = out.ptrw();
		memcpy(&w[0], &p_packet[1], len);
	}
	emit_signal(SNAME("peer_packet"), p_from, out);
}

int SceneMultiplayer::get_unique_id() {
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), 0, "No multiplayer peer is assigned. Unable to get unique ID.");
	return multiplayer_peer->get_unique_id();
}

void SceneMultiplayer::set_refuse_new_connections(bool p_refuse) {
	ERR_FAIL_COND_MSG(!multiplayer_peer.is_valid(), "No multiplayer peer is assigned. Unable to set 'refuse_new_connections'.");
	multiplayer_peer->set_refuse_new_connections(p_refuse);
}

bool SceneMultiplayer::is_refusing_new_connections() const {
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), false, "No multiplayer peer is assigned. Unable to get 'refuse_new_connections'.");
	return multiplayer_peer->is_refusing_new_connections();
}

Vector<int> SceneMultiplayer::get_peer_ids() {
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), Vector<int>(), "No multiplayer peer is assigned. Assume no peers are connected.");

	Vector<int> ret;
	for (const int &E : connected_peers) {
		ret.push_back(E);
	}

	return ret;
}

void SceneMultiplayer::set_allow_object_decoding(bool p_enable) {
	allow_object_decoding = p_enable;
}

bool SceneMultiplayer::is_object_decoding_allowed() const {
	return allow_object_decoding;
}

String SceneMultiplayer::get_rpc_md5(const Object *p_obj) {
	return rpc->get_rpc_md5(p_obj);
}

Error SceneMultiplayer::rpcp(Object *p_obj, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount) {
	return rpc->rpcp(p_obj, p_peer_id, p_method, p_arg, p_argcount);
}

Error SceneMultiplayer::object_configuration_add(Object *p_obj, Variant p_config) {
	if (p_obj == nullptr && p_config.get_type() == Variant::NODE_PATH) {
		set_root_path(p_config);
		return OK;
	}
	MultiplayerSpawner *spawner = Object::cast_to<MultiplayerSpawner>(p_config.get_validated_object());
	MultiplayerSynchronizer *sync = Object::cast_to<MultiplayerSynchronizer>(p_config.get_validated_object());
	if (spawner) {
		return replicator->on_spawn(p_obj, p_config);
	} else if (sync) {
		return replicator->on_replication_start(p_obj, p_config);
	}
	return ERR_INVALID_PARAMETER;
}

Error SceneMultiplayer::object_configuration_remove(Object *p_obj, Variant p_config) {
	if (p_obj == nullptr && p_config.get_type() == Variant::NODE_PATH) {
		ERR_FAIL_COND_V(root_path != p_config.operator NodePath(), ERR_INVALID_PARAMETER);
		set_root_path(NodePath());
		return OK;
	}
	MultiplayerSpawner *spawner = Object::cast_to<MultiplayerSpawner>(p_config.get_validated_object());
	MultiplayerSynchronizer *sync = Object::cast_to<MultiplayerSynchronizer>(p_config.get_validated_object());
	if (spawner) {
		return replicator->on_despawn(p_obj, p_config);
	}
	if (sync) {
		return replicator->on_replication_stop(p_obj, p_config);
	}
	return ERR_INVALID_PARAMETER;
}

void SceneMultiplayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &SceneMultiplayer::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &SceneMultiplayer::get_root_path);
	ClassDB::bind_method(D_METHOD("clear"), &SceneMultiplayer::clear);
	ClassDB::bind_method(D_METHOD("set_refuse_new_connections", "refuse"), &SceneMultiplayer::set_refuse_new_connections);
	ClassDB::bind_method(D_METHOD("is_refusing_new_connections"), &SceneMultiplayer::is_refusing_new_connections);
	ClassDB::bind_method(D_METHOD("set_allow_object_decoding", "enable"), &SceneMultiplayer::set_allow_object_decoding);
	ClassDB::bind_method(D_METHOD("is_object_decoding_allowed"), &SceneMultiplayer::is_object_decoding_allowed);
	ClassDB::bind_method(D_METHOD("send_bytes", "bytes", "id", "mode", "channel"), &SceneMultiplayer::send_bytes, DEFVAL(MultiplayerPeer::TARGET_PEER_BROADCAST), DEFVAL(MultiplayerPeer::TRANSFER_MODE_RELIABLE), DEFVAL(0));

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_object_decoding"), "set_allow_object_decoding", "is_object_decoding_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_connections"), "set_refuse_new_connections", "is_refusing_new_connections");
	ADD_PROPERTY_DEFAULT("refuse_new_connections", false);

	ADD_SIGNAL(MethodInfo("peer_packet", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "packet")));
}

SceneMultiplayer::SceneMultiplayer() {
	replicator = Ref<SceneReplicationInterface>(memnew(SceneReplicationInterface(this)));
	rpc = Ref<SceneRPCInterface>(memnew(SceneRPCInterface(this)));
	cache = Ref<SceneCacheInterface>(memnew(SceneCacheInterface(this)));
}

SceneMultiplayer::~SceneMultiplayer() {
	clear();
}
