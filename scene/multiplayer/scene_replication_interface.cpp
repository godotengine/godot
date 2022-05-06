/*************************************************************************/
/*  scene_replication_interface.cpp                                      */
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

#include "scene_replication_interface.h"

#include "core/io/marshalls.h"
#include "scene/main/node.h"
#include "scene/multiplayer/multiplayer_spawner.h"
#include "scene/multiplayer/multiplayer_synchronizer.h"

#define MAKE_ROOM(m_amount)             \
	if (packet_cache.size() < m_amount) \
		packet_cache.resize(m_amount);

MultiplayerReplicationInterface *SceneReplicationInterface::_create(MultiplayerAPI *p_multiplayer) {
	return memnew(SceneReplicationInterface(p_multiplayer));
}

void SceneReplicationInterface::make_default() {
	MultiplayerAPI::create_default_replication_interface = _create;
}

void SceneReplicationInterface::_free_remotes(int p_id) {
	const HashMap<uint32_t, ObjectID> remotes = rep_state->peer_get_remotes(p_id);
	const uint32_t *k = nullptr;
	while ((k = remotes.next(k))) {
		Node *node = rep_state->get_node(remotes.get(*k));
		ERR_CONTINUE(!node);
		node->queue_delete();
	}
}

void SceneReplicationInterface::on_peer_change(int p_id, bool p_connected) {
	if (p_connected) {
		rep_state->on_peer_change(p_id, p_connected);
		for (const ObjectID &oid : rep_state->get_spawned_nodes()) {
			_send_spawn(rep_state->get_node(oid), rep_state->get_spawner(oid), p_id);
		}
		for (const ObjectID &oid : rep_state->get_path_only_nodes()) {
			Node *node = rep_state->get_node(oid);
			MultiplayerSynchronizer *sync = rep_state->get_synchronizer(oid);
			ERR_CONTINUE(!node || !sync);
			if (sync->is_multiplayer_authority()) {
				rep_state->peer_add_node(p_id, oid);
			}
		}
	} else {
		_free_remotes(p_id);
		rep_state->on_peer_change(p_id, p_connected);
	}
}

void SceneReplicationInterface::on_reset() {
	for (int pid : rep_state->get_peers()) {
		_free_remotes(pid);
	}
	rep_state->reset();
}

void SceneReplicationInterface::on_network_process() {
	uint64_t msec = OS::get_singleton()->get_ticks_msec();
	for (int peer : rep_state->get_peers()) {
		_send_sync(peer, msec);
	}
}

Error SceneReplicationInterface::on_spawn(Object *p_obj, Variant p_config) {
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V(!node || p_config.get_type() != Variant::OBJECT, ERR_INVALID_PARAMETER);
	MultiplayerSpawner *spawner = Object::cast_to<MultiplayerSpawner>(p_config.get_validated_object());
	ERR_FAIL_COND_V(!spawner, ERR_INVALID_PARAMETER);
	Error err = rep_state->config_add_spawn(node, spawner);
	ERR_FAIL_COND_V(err != OK, err);
	return _send_spawn(node, spawner, 0);
}

Error SceneReplicationInterface::on_despawn(Object *p_obj, Variant p_config) {
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V(!node || p_config.get_type() != Variant::OBJECT, ERR_INVALID_PARAMETER);
	MultiplayerSpawner *spawner = Object::cast_to<MultiplayerSpawner>(p_config.get_validated_object());
	ERR_FAIL_COND_V(!p_obj || !spawner, ERR_INVALID_PARAMETER);
	Error err = rep_state->config_del_spawn(node, spawner);
	ERR_FAIL_COND_V(err != OK, err);
	return _send_despawn(node, 0);
}

Error SceneReplicationInterface::on_replication_start(Object *p_obj, Variant p_config) {
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V(!node || p_config.get_type() != Variant::OBJECT, ERR_INVALID_PARAMETER);
	MultiplayerSynchronizer *sync = Object::cast_to<MultiplayerSynchronizer>(p_config.get_validated_object());
	ERR_FAIL_COND_V(!sync, ERR_INVALID_PARAMETER);
	rep_state->config_add_sync(node, sync);
	// Try to apply initial state if spawning (hack to apply if before ready).
	if (pending_spawn == p_obj->get_instance_id()) {
		pending_spawn = ObjectID(); // Make sure this only happens once.
		const List<NodePath> props = sync->get_replication_config()->get_spawn_properties();
		Vector<Variant> vars;
		vars.resize(props.size());
		int consumed;
		Error err = MultiplayerAPI::decode_and_decompress_variants(vars, pending_buffer, pending_buffer_size, consumed);
		ERR_FAIL_COND_V(err, err);
		err = MultiplayerSynchronizer::set_state(props, node, vars);
		ERR_FAIL_COND_V(err, err);
	} else if (multiplayer->has_multiplayer_peer() && sync->is_multiplayer_authority()) {
		// Either it's a spawn or a static sync, in any case add it to the list of known nodes.
		rep_state->peer_add_node(0, p_obj->get_instance_id());
	}
	return OK;
}

Error SceneReplicationInterface::on_replication_stop(Object *p_obj, Variant p_config) {
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V(!node || p_config.get_type() != Variant::OBJECT, ERR_INVALID_PARAMETER);
	MultiplayerSynchronizer *sync = Object::cast_to<MultiplayerSynchronizer>(p_config.get_validated_object());
	ERR_FAIL_COND_V(!p_obj || !sync, ERR_INVALID_PARAMETER);
	return rep_state->config_del_sync(node, sync);
}

Error SceneReplicationInterface::_send_raw(const uint8_t *p_buffer, int p_size, int p_peer, bool p_reliable) {
	ERR_FAIL_COND_V(!p_buffer || p_size < 1, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!multiplayer, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!multiplayer->has_multiplayer_peer(), ERR_UNCONFIGURED);

#ifdef DEBUG_ENABLED
	multiplayer->profile_bandwidth("out", p_size);
#endif

	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	peer->set_target_peer(p_peer);
	peer->set_transfer_channel(0);
	peer->set_transfer_mode(p_reliable ? Multiplayer::TRANSFER_MODE_RELIABLE : Multiplayer::TRANSFER_MODE_UNRELIABLE);
	return peer->put_packet(p_buffer, p_size);
}

Error SceneReplicationInterface::_send_spawn(Node *p_node, MultiplayerSpawner *p_spawner, int p_peer) {
	ERR_FAIL_COND_V(p_peer < 0, ERR_BUG);
	ERR_FAIL_COND_V(!multiplayer, ERR_BUG);
	ERR_FAIL_COND_V(!p_spawner || !p_node, ERR_BUG);

	const ObjectID oid = p_node->get_instance_id();
	uint32_t nid = rep_state->ensure_net_id(oid);

	// Prepare custom arg and scene_id
	uint8_t scene_id = p_spawner->get_spawn_id(oid);
	bool is_custom = scene_id == MultiplayerSpawner::INVALID_ID;
	Variant spawn_arg = p_spawner->get_spawn_argument(oid);
	int spawn_arg_size = 0;
	if (is_custom) {
		Error err = MultiplayerAPI::encode_and_compress_variant(spawn_arg, nullptr, spawn_arg_size, false);
		ERR_FAIL_COND_V(err, err);
	}

	// Prepare spawn state.
	int state_size = 0;
	Vector<Variant> state_vars;
	Vector<const Variant *> state_varp;
	MultiplayerSynchronizer *synchronizer = rep_state->get_synchronizer(oid);
	if (synchronizer && synchronizer->get_replication_config().is_valid()) {
		const List<NodePath> props = synchronizer->get_replication_config()->get_spawn_properties();
		Error err = MultiplayerSynchronizer::get_state(props, p_node, state_vars, state_varp);
		ERR_FAIL_COND_V_MSG(err != OK, err, "Unable to retrieve spawn state.");
		err = MultiplayerAPI::encode_and_compress_variants(state_varp.ptrw(), state_varp.size(), nullptr, state_size);
		ERR_FAIL_COND_V_MSG(err != OK, err, "Unable to encode spawn state.");
	}

	// Prepare simplified path.
	NodePath rel_path = multiplayer->get_root_path().rel_path_to(p_spawner->get_path());

	int path_id = 0;
	multiplayer->send_object_cache(p_spawner, rel_path, p_peer, path_id);

	// Encode name and parent ID.
	CharString cname = p_node->get_name().operator String().utf8();
	int nlen = encode_cstring(cname.get_data(), nullptr);
	MAKE_ROOM(1 + 1 + 4 + 4 + 4 + nlen + (is_custom ? 4 + spawn_arg_size : 0) + state_size);
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = (uint8_t)MultiplayerAPI::NETWORK_COMMAND_SPAWN;
	ptr[1] = scene_id;
	int ofs = 2;
	ofs += encode_uint32(path_id, &ptr[ofs]);
	ofs += encode_uint32(nid, &ptr[ofs]);
	ofs += encode_uint32(nlen, &ptr[ofs]);
	ofs += encode_cstring(cname.get_data(), &ptr[ofs]);
	// Write args
	if (is_custom) {
		ofs += encode_uint32(spawn_arg_size, &ptr[ofs]);
		Error err = MultiplayerAPI::encode_and_compress_variant(spawn_arg, &ptr[ofs], spawn_arg_size, false);
		ERR_FAIL_COND_V(err, err);
		ofs += spawn_arg_size;
	}
	// Write state.
	if (state_size) {
		Error err = MultiplayerAPI::encode_and_compress_variants(state_varp.ptrw(), state_varp.size(), &ptr[ofs], state_size);
		ERR_FAIL_COND_V(err, err);
		ofs += state_size;
	}
	Error err = _send_raw(ptr, ofs, p_peer, true);
	ERR_FAIL_COND_V(err, err);
	return rep_state->peer_add_node(p_peer, oid);
}

Error SceneReplicationInterface::_send_despawn(Node *p_node, int p_peer) {
	const ObjectID oid = p_node->get_instance_id();
	MAKE_ROOM(5);
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = (uint8_t)MultiplayerAPI::NETWORK_COMMAND_DESPAWN;
	int ofs = 1;
	uint32_t nid = rep_state->get_net_id(oid);
	ofs += encode_uint32(nid, &ptr[ofs]);
	Error err = _send_raw(ptr, ofs, p_peer, true);
	ERR_FAIL_COND_V(err, err);
	return rep_state->peer_del_node(p_peer, oid);
}

Error SceneReplicationInterface::on_spawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) {
	ERR_FAIL_COND_V_MSG(p_buffer_len < 14, ERR_INVALID_DATA, "Invalid spawn packet received");
	int ofs = 1; // The spawn/despawn command.
	uint8_t scene_id = p_buffer[ofs];
	ofs += 1;
	uint32_t node_target = decode_uint32(&p_buffer[ofs]);
	ofs += 4;
	MultiplayerSpawner *spawner = Object::cast_to<MultiplayerSpawner>(multiplayer->get_cached_object(p_from, node_target));
	ERR_FAIL_COND_V(!spawner, ERR_DOES_NOT_EXIST);
	ERR_FAIL_COND_V(p_from != spawner->get_multiplayer_authority(), ERR_UNAUTHORIZED);

	uint32_t net_id = decode_uint32(&p_buffer[ofs]);
	ofs += 4;
	uint32_t name_len = decode_uint32(&p_buffer[ofs]);
	ofs += 4;
	ERR_FAIL_COND_V_MSG(name_len > uint32_t(p_buffer_len - ofs), ERR_INVALID_DATA, vformat("Invalid spawn packet size: %d, wants: %d", p_buffer_len, ofs + name_len));
	ERR_FAIL_COND_V_MSG(name_len < 1, ERR_INVALID_DATA, "Zero spawn name size.");

	// We need to make sure no trickery happens here, but we want to allow autogenerated ("@") node names.
	const String name = String::utf8((const char *)&p_buffer[ofs], name_len);
	ERR_FAIL_COND_V_MSG(name.validate_node_name() != name, ERR_INVALID_DATA, vformat("Invalid node name received: '%s'. Make sure to add nodes via 'add_child(node, true)' remotely.", name));
	ofs += name_len;

	// Check that we can spawn.
	Node *parent = spawner->get_node_or_null(spawner->get_spawn_path());
	ERR_FAIL_COND_V(!parent, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(parent->has_node(name), ERR_INVALID_DATA);

	Node *node = nullptr;
	if (scene_id == MultiplayerSpawner::INVALID_ID) {
		// Custom spawn.
		ERR_FAIL_COND_V(p_buffer_len - ofs < 4, ERR_INVALID_DATA);
		uint32_t arg_size = decode_uint32(&p_buffer[ofs]);
		ofs += 4;
		ERR_FAIL_COND_V(arg_size > uint32_t(p_buffer_len - ofs), ERR_INVALID_DATA);
		Variant v;
		Error err = MultiplayerAPI::decode_and_decompress_variant(v, &p_buffer[ofs], arg_size, nullptr, false);
		ERR_FAIL_COND_V(err != OK, err);
		ofs += arg_size;
		node = spawner->instantiate_custom(v);
	} else {
		// Scene based spawn.
		node = spawner->instantiate_scene(scene_id);
	}
	ERR_FAIL_COND_V(!node, ERR_UNAUTHORIZED);
	node->set_name(name);
	rep_state->peer_add_remote(p_from, net_id, node, spawner);
	// The initial state will be applied during the sync config (i.e. before _ready).
	int state_len = p_buffer_len - ofs;
	if (state_len) {
		pending_spawn = node->get_instance_id();
		pending_buffer = &p_buffer[ofs];
		pending_buffer_size = state_len;
	}
	parent->add_child(node);
	pending_spawn = ObjectID();
	pending_buffer = nullptr;
	pending_buffer_size = 0;
	return OK;
}

Error SceneReplicationInterface::on_despawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) {
	ERR_FAIL_COND_V_MSG(p_buffer_len < 5, ERR_INVALID_DATA, "Invalid spawn packet received");
	int ofs = 1; // The spawn/despawn command.
	uint32_t net_id = decode_uint32(&p_buffer[ofs]);
	ofs += 4;
	Node *node = nullptr;
	Error err = rep_state->peer_del_remote(p_from, net_id, &node);
	ERR_FAIL_COND_V(err != OK, err);
	ERR_FAIL_COND_V(!node, ERR_BUG);
	if (node->get_parent() != nullptr) {
		node->get_parent()->remove_child(node);
	}
	node->queue_delete();
	return OK;
}

void SceneReplicationInterface::_send_sync(int p_peer, uint64_t p_msec) {
	const Set<ObjectID> &known = rep_state->get_known_nodes(p_peer);
	if (known.is_empty()) {
		return;
	}
	MAKE_ROOM(sync_mtu);
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = MultiplayerAPI::NETWORK_COMMAND_SYNC;
	int ofs = 1;
	ofs += encode_uint16(rep_state->peer_sync_next(p_peer), &ptr[1]);
	// Can only send updates for already notified nodes.
	// This is a lazy implementation, we could optimize much more here with by grouping by replication config.
	for (const ObjectID &oid : known) {
		if (!rep_state->update_sync_time(oid, p_msec)) {
			continue; // nothing to sync.
		}
		MultiplayerSynchronizer *sync = rep_state->get_synchronizer(oid);
		ERR_CONTINUE(!sync);
		Node *node = rep_state->get_node(oid);
		ERR_CONTINUE(!node);
		int size;
		Vector<Variant> vars;
		Vector<const Variant *> varp;
		const List<NodePath> props = sync->get_replication_config()->get_sync_properties();
		Error err = MultiplayerSynchronizer::get_state(props, node, vars, varp);
		ERR_CONTINUE_MSG(err != OK, "Unable to retrieve sync state.");
		err = MultiplayerAPI::encode_and_compress_variants(varp.ptrw(), varp.size(), nullptr, size);
		ERR_CONTINUE_MSG(err != OK, "Unable to encode sync state.");
		// TODO Handle single state above MTU.
		ERR_CONTINUE_MSG(size > 3 + 4 + 4 + sync_mtu, vformat("Node states bigger then MTU will not be sent (%d > %d): %s", size, sync_mtu, node->get_path()));
		if (ofs + 4 + 4 + size > sync_mtu) {
			// Send what we got, and reset write.
			_send_raw(packet_cache.ptr(), ofs, p_peer, false);
			ofs = 3;
		}
		if (size) {
			uint32_t net_id = rep_state->get_net_id(oid);
			if (net_id == 0 || (net_id & 0x80000000)) {
				// First time path based ID.
				NodePath rel_path = multiplayer->get_root_path().rel_path_to(sync->get_path());
				int path_id = 0;
				multiplayer->send_object_cache(sync, rel_path, p_peer, path_id);
				ERR_CONTINUE_MSG(net_id && net_id != (uint32_t(path_id) | 0x80000000), "This should never happen!");
				net_id = path_id;
				rep_state->set_net_id(oid, net_id | 0x80000000);
			}
			ofs += encode_uint32(rep_state->get_net_id(oid), &ptr[ofs]);
			ofs += encode_uint32(size, &ptr[ofs]);
			MultiplayerAPI::encode_and_compress_variants(varp.ptrw(), varp.size(), &ptr[ofs], size);
			ofs += size;
		}
	}
	if (ofs > 3) {
		// Got some left over to send.
		_send_raw(packet_cache.ptr(), ofs, p_peer, false);
	}
}

Error SceneReplicationInterface::on_sync_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) {
	ERR_FAIL_COND_V_MSG(p_buffer_len < 11, ERR_INVALID_DATA, "Invalid sync packet received");
	uint16_t time = decode_uint16(&p_buffer[1]);
	int ofs = 3;
	rep_state->peer_sync_recv(p_from, time);
	while (ofs + 8 < p_buffer_len) {
		uint32_t net_id = decode_uint32(&p_buffer[ofs]);
		ofs += 4;
		uint32_t size = decode_uint32(&p_buffer[ofs]);
		ofs += 4;
		Node *node = nullptr;
		if (net_id & 0x80000000) {
			MultiplayerSynchronizer *sync = Object::cast_to<MultiplayerSynchronizer>(multiplayer->get_cached_object(p_from, net_id & 0x7FFFFFFF));
			ERR_FAIL_COND_V(!sync || sync->get_multiplayer_authority() != p_from, ERR_UNAUTHORIZED);
			node = sync->get_node(sync->get_root_path());
		} else {
			node = rep_state->peer_get_remote(p_from, net_id);
		}
		if (!node) {
			// Not received yet.
			ofs += size;
			continue;
		}
		const ObjectID oid = node->get_instance_id();
		if (!rep_state->update_last_node_sync(oid, time)) {
			// State is too old.
			ofs += size;
			continue;
		}
		MultiplayerSynchronizer *sync = rep_state->get_synchronizer(oid);
		ERR_FAIL_COND_V(!sync, ERR_BUG);
		ERR_FAIL_COND_V(size > uint32_t(p_buffer_len - ofs), ERR_BUG);
		const List<NodePath> props = sync->get_replication_config()->get_sync_properties();
		Vector<Variant> vars;
		vars.resize(props.size());
		int consumed;
		Error err = MultiplayerAPI::decode_and_decompress_variants(vars, &p_buffer[ofs], size, consumed);
		ERR_FAIL_COND_V(err, err);
		err = MultiplayerSynchronizer::set_state(props, node, vars);
		ERR_FAIL_COND_V(err, err);
		ofs += size;
	}
	return OK;
}
