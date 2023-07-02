/**************************************************************************/
/*  scene_saveload_interface.cpp                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "scene_saveload_interface.h"

#include "scene_saveload.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"
#include "scene/main/node.h"
#include "scene/scene_string_names.h"

#ifdef DEBUG_ENABLED
_FORCE_INLINE_ void SceneSaveloadInterface::_profile_node_data(const String &p_what, ObjectID p_id, int p_size) {
	if (EngineDebugger::is_profiling("saveload:saveload")) {
		Array values;
		values.push_back(p_what);
		values.push_back(p_id);
		values.push_back(p_size);
		EngineDebugger::profiler_add_frame_data("saveload:saveload", values);
	}
}
#endif

void SceneSaveloadInterface::on_reset() {
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE(!sync);
		sync->reset();
	}
}

void SceneSaveloadInterface::flush_spawn_queue() {
	// Prevent endless stalling in case of unforeseen spawn errors.
	if (spawn_queue.size()) {
		ERR_PRINT("An error happened during last spawn, this usually means the 'ready' signal was not emitted by the spawned node.");
		for (const ObjectID &oid : spawn_queue) {
			Node *node = get_id_as<Node>(oid);
			ERR_CONTINUE(!node);
			if (node->is_connected(SceneStringNames::get_singleton()->ready, callable_mp(this, &SceneSaveloadInterface::_node_ready))) {
				node->disconnect(SceneStringNames::get_singleton()->ready, callable_mp(this, &SceneSaveloadInterface::_node_ready));
			}
		}
		spawn_queue.clear();
	}
}

void SceneSaveloadInterface::configure_spawn(Node *p_node, const SaveloadSpawner &p_spawner) {
	const ObjectID node_id = p_node->get_instance_id();
	const ObjectID spawner_id = p_spawner.get_instance_id();
	if (!spawn_nodes.has(spawner_id)) {
		spawn_nodes.insert(spawner_id);
	}
	p_node->connect(SceneStringNames::get_singleton()->ready, callable_mp(this, &SceneSaveloadInterface::_node_ready).bind(node_id), Node::CONNECT_ONE_SHOT);
}

void SceneSaveloadInterface::deconfigure_spawn(const SaveloadSpawner &p_spawner) {
	spawn_nodes.erase(p_spawner.get_instance_id());
}

void SceneSaveloadInterface::configure_sync(Node *p_node, const SaveloadSynchronizer &p_syncher) {
	const NodePath root_path = p_syncher.get_root_path();
	Node *root_node = p_syncher.get_root_node();
	ERR_FAIL_COND_MSG(!root_node, vformat("Could not find Synchronizer root node at path %s", root_path));
	ERR_FAIL_COND_MSG((root_node->get_path() != p_node->get_path()) && !(root_node->is_ancestor_of(p_node)), vformat("Synchronizer root at %s is not an ancestor of node at %s", root_node->get_path(), p_node->get_path()));
	sync_nodes.insert(p_syncher.get_instance_id());
}

void SceneSaveloadInterface::deconfigure_sync(const SaveloadSynchronizer &p_syncher) {
	sync_nodes.erase(p_syncher.get_instance_id());
}

void SceneSaveloadInterface::_node_ready(const ObjectID &p_oid) {
//	ERR_FAIL_COND(!spawn_queue.has(p_oid)); // Bug.
//
//	// If we are a nested spawn, we need to wait until the parent is ready.
//	if (p_oid != *(spawn_queue.begin())) {
//		return;
//	}
//
//	for (const ObjectID &oid : spawn_queue) {
//		ERR_CONTINUE(!tracked_nodes.has(oid));
//
//		TrackedNode &tobj = tracked_nodes[oid];
//		MultiplayerSpawner *spawner = get_id_as<MultiplayerSpawner>(tobj.spawner);
//		ERR_CONTINUE(!spawner);
//
//		spawned_nodes.insert(oid);
//		if (multiplayer->has_multiplayer_peer() && spawner->is_multiplayer_authority()) {
//			if (tobj.net_id == 0) {
//				tobj.net_id = ++last_net_id;
//			}
//			_update_spawn_visibility(0, oid);
//		}
//	}
//	spawn_queue.clear();
}

TypedArray<SaveloadSynchronizer> SceneSaveloadInterface::get_sync_nodes() {
	TypedArray<SaveloadSynchronizer> syncs;
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid synchronizer", oid));
		syncs.append(sync);
	}
	return syncs;
}

Dictionary SceneSaveloadInterface::get_sync_state() {
	Dictionary sync_state;
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE(!sync);
		const NodePath node_path = sync->get_root_path();
		Dictionary state_dict = sync->get_state_wrapper();
		sync_state[sync->get_root_path()] = sync->get_state_wrapper();
	}
}

SceneSaveloadInterface::SaveloadState SceneSaveloadInterface::get_saveload_state() {
	SaveloadState saveload_state;
	for (const ObjectID &oid : spawn_nodes) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid spawner", oid));
		SaveloadSpawner::SpawnState spawn_state = spawner->get_spawn_state();
		saveload_state.spawn_states.insert(spawner->get_path(), spawn_state);
	}
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid synchronizer", oid));
		SaveloadSynchronizer::SyncState sync_state = sync->get_sync_state();
		saveload_state.sync_states.insert(sync->get_path(), sync_state);
	}
	return saveload_state;
}

void SceneSaveloadInterface::load_saveload_state(const SaveloadState p_saveload_state) {
	Node *root = SceneTree::get_singleton()->get_current_scene(); //TODO: Is this the right root?
	for (const KeyValue<const NodePath, SaveloadSpawner::SpawnState> &spawn_state : p_saveload_state.spawn_states) {
		//TODO: How should I delete nodes that aren't in the save file?
		SaveloadSpawner *spawn_node = Object::cast_to<SaveloadSpawner>(root->get_node(spawn_state.key));
		ERR_CONTINUE_MSG(!spawn_node, vformat("could not find SaveloadSpawner at path %s", spawn_state.key));
		spawn_node->spawn_all(spawn_state.value);
	}
	for (const KeyValue<const NodePath, SaveloadSynchronizer::SyncState> &sync_state : p_saveload_state.sync_states) {
		SaveloadSynchronizer *sync_node = Object::cast_to<SaveloadSynchronizer>(root->get_node(sync_state.key));
		ERR_CONTINUE_MSG(!sync_node, vformat("could not find SaveloadSynchronizer at path %s", sync_state.key));
		sync_node->synchronize(sync_state.value);
	}
}

Dictionary SceneSaveloadInterface::SaveloadState::to_dict() {
	Dictionary dict;
	Dictionary spawn_dict;
	for (const KeyValue<const NodePath, SaveloadSpawner::SpawnState> &spawn_state : SaveloadState::spawn_states) {
		spawn_dict[spawn_state.key] = spawn_state.value.to_dict();
	}
	Dictionary sync_dict;
	for (const KeyValue<const NodePath, SaveloadSynchronizer::SyncState> &sync_state : SaveloadState::sync_states) {
		sync_dict[sync_state.key] = sync_state.value.to_dict();
	}
	dict[StringName("spawn_states")] = spawn_dict;
	dict[StringName("sync_states")] = sync_dict;
	return dict;
}

SceneSaveloadInterface::SaveloadState::SaveloadState(const Dictionary &p_saveload_dict) {
	Dictionary spawn_states_dict = p_saveload_dict[StringName("spawn_states")];
	Dictionary sync_states_dict = p_saveload_dict[StringName("sync_states")];
	List<Variant> spawn_keys;
	spawn_states_dict.get_key_list(&spawn_keys);
	for (const NodePath spawn_key : spawn_keys) {
		Dictionary spawn_state_as_dict = spawn_states_dict[spawn_key];
		spawn_states.insert(spawn_key, SaveloadSpawner::SpawnState(spawn_state_as_dict));
	}
	List<Variant> sync_keys;
	sync_states_dict.get_key_list(&sync_keys);
	for (const NodePath sync_key : sync_keys) {
		Dictionary sync_state_as_dict = sync_states_dict[sync_key];
		sync_states.insert(sync_key, SaveloadSynchronizer::SyncState(sync_state_as_dict));
	}
}

//void SceneSaveloadInterface::_visibility_changed(int p_peer, ObjectID p_sid) {
//	SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(p_sid);
//	ERR_FAIL_COND(!sync); // Bug.
//	Node *node = sync->get_root_node();
//	ERR_FAIL_COND(!node); // Bug.
//	const ObjectID oid = node->get_instance_id();
//	if (spawned_nodes.has(oid) && p_peer != saveload->get_unique_id()) {
//		_update_spawn_visibility(p_peer, oid);
//	}
//	_update_sync_visibility(p_peer, sync);
//}
//
//bool SceneSaveloadInterface::is_rpc_visible(const ObjectID &p_oid, int p_peer) const {
//	if (!tracked_nodes.has(p_oid)) {
//		return true; // Untracked nodes are always visible to RPCs.
//	}
//	ERR_FAIL_COND_V(p_peer < 0, false);
//	const TrackedNode &tnode = tracked_nodes[p_oid];
//	if (tnode.synchronizers.is_empty()) {
//		return true; // No synchronizers means no visibility restrictions.
//	}
//	if (tnode.remote_peer && uint32_t(p_peer) == tnode.remote_peer) {
//		return true; // RPCs on spawned nodes are always visible to spawner.
//	} else if (spawned_nodes.has(p_oid)) {
//		// It's a spawned node we control, this can be fast.
//		if (p_peer) {
//			return peers_info.has(p_peer) && peers_info[p_peer].spawn_nodes.has(p_oid);
//		} else {
//			for (const KeyValue<int, PeerInfo> &E : peers_info) {
//				if (!E.value.spawn_nodes.has(p_oid)) {
//					return false; // Not public.
//				}
//			}
//			return true; // All peers have this node.
//		}
//	} else {
//		// Cycle object synchronizers to check visibility.
//		for (const ObjectID &sid : tnode.synchronizers) {
//			SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(sid);
//			ERR_CONTINUE(!sync);
//			// RPC visibility is composed using OR when multiple synchronizers are present.
//			// Note that we don't really care about authority here which may lead to unexpected
//			// results when using multiple synchronizers to control the same node.
//			if (sync->is_visible_to(p_peer)) {
//				return true;
//			}
//		}
//		return false; // Not visible.
//	}
//}
//
//Error SceneSaveloadInterface::_update_sync_visibility(int p_peer, SaveloadSynchronizer *p_sync) {
//	ERR_FAIL_COND_V(!p_sync, ERR_BUG);
//	return OK
//}
//
//Error SceneSaveloadInterface::_update_spawn_visibility(int p_peer, const ObjectID &p_oid) {
//	const TrackedNode *tnode = tracked_nodes.getptr(p_oid);
//	ERR_FAIL_COND_V(!tnode, ERR_BUG);
//	SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(tnode->spawner);
//	Node *node = get_id_as<Node>(p_oid);
//	ERR_FAIL_COND_V(!node || !spawner, ERR_BUG);
//	ERR_FAIL_COND_V(!tracked_nodes.has(p_oid), ERR_BUG);
//	const HashSet<ObjectID> synchronizers = tracked_nodes[p_oid].synchronizers;
//	bool is_visible = true;
//	for (const ObjectID &sid : synchronizers) {
//		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(sid);
//		ERR_CONTINUE(!sync);
////		if (!sync->is_multiplayer_authority()) {
////			continue;
////		}
//		// Spawn visibility is composed using OR when multiple synchronizers are present.
//		if (sync->is_visible_to(p_peer)) {
//			is_visible = true;
//			break;
//		}
//		is_visible = false;
//	}
//	// Spawn (and despawn) when needed.
//	HashSet<int> to_spawn;
//	HashSet<int> to_despawn;
//	if (p_peer) {
//		ERR_FAIL_COND_V(!peers_info.has(p_peer), ERR_INVALID_PARAMETER);
//		if (is_visible == peers_info[p_peer].spawn_nodes.has(p_oid)) {
//			return OK;
//		}
//		if (is_visible) {
//			to_spawn.insert(p_peer);
//		} else {
//			to_despawn.insert(p_peer);
//		}
//	} else {
//		// Check visibility for each peers.
//		for (const KeyValue<int, PeerInfo> &E : peers_info) {
//			if (is_visible) {
//				// This is fast, since the the object is visible to everyone, we don't need to check each peer.
//				if (E.value.spawn_nodes.has(p_oid)) {
//					// Already spawned.
//					continue;
//				}
//				to_spawn.insert(E.key);
//			} else {
//				// Need to check visibility for each peer.
//				_update_spawn_visibility(E.key, p_oid);
//			}
//		}
//	}
//	if (to_spawn.size()) {
//		int len = 0;
//		_make_spawn_packet(node, spawner, len);
//		for (int pid : to_spawn) {
//			ERR_CONTINUE(!peers_info.has(pid));
//			int path_id;
//			saveload->get_path_cache()->send_object_cache(spawner, pid, path_id);
//			_send_raw(packet_cache.ptr(), len, pid, true);
//			peers_info[pid].spawn_nodes.insert(p_oid);
//		}
//	}
//	if (to_despawn.size()) {
//		int len = 0;
//		_make_despawn_packet(node, len);
//		for (int pid : to_despawn) {
//			ERR_CONTINUE(!peers_info.has(pid));
//			peers_info[pid].spawn_nodes.erase(p_oid);
//			_send_raw(packet_cache.ptr(), len, pid, true);
//		}
//	}
//	return OK;
//}

//Error SceneSaveloadInterface::_send_raw(const uint8_t *p_buffer, int p_size, int p_peer, bool p_reliable) {
//	ERR_FAIL_COND_V(!p_buffer || p_size < 1, ERR_INVALID_PARAMETER);
//	ERR_FAIL_COND_V(!saveload->has_multiplayer_peer(), ERR_UNCONFIGURED);
//
//	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
//	peer->set_transfer_channel(0);
//	peer->set_transfer_mode(p_reliable ? MultiplayerPeer::TRANSFER_MODE_RELIABLE : MultiplayerPeer::TRANSFER_MODE_UNRELIABLE);
//	return multiplayer->send_command(p_peer, p_buffer, p_size);
//}

//Error SceneSaveloadInterface::_make_spawn_packet(Node *p_node, SaveloadSpawner *p_spawner, int &r_len) {
//	ERR_FAIL_COND_V(!saveload || !p_node || !p_spawner, ERR_BUG);
//
//	const ObjectID oid = p_node->get_instance_id();
//	const TrackedNode *tnode = tracked_nodes.getptr(oid);
//	ERR_FAIL_COND_V(!tnode, ERR_INVALID_PARAMETER);
//
//	uint32_t nid = tnode->net_id;
//	ERR_FAIL_COND_V(!nid, ERR_UNCONFIGURED);
//
//	// Prepare custom arg and scene_id
//	uint8_t scene_id = p_spawner->find_spawnable_scene_index_from_object(oid);
//	bool is_custom = scene_id == SaveloadSpawner::INVALID_ID;
//	Variant spawn_arg = p_spawner->get_spawn_argument(oid);
//	int spawn_arg_size = 0;
//	if (is_custom) {
//		Error err = SaveloadAPI::encode_and_compress_variant(spawn_arg, nullptr, spawn_arg_size, false);
//		ERR_FAIL_COND_V(err, err);
//	}
//
//	// Prepare spawn state.
//	List<NodePath> state_props;
//	List<uint32_t> sync_ids;
//	const HashSet<ObjectID> synchronizers = tnode->synchronizers;
//	for (const ObjectID &sid : synchronizers) {
//		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(sid);
////		if (!sync->is_multiplayer_authority()) {
////			continue;
////		}
//		ERR_CONTINUE(!sync);
//		ERR_FAIL_COND_V(sync->get_saveload_config().is_null(), ERR_BUG);
//		for (const NodePath &prop : sync->get_saveload_config()->get_spawn_properties()) {
//			state_props.push_back(prop);
//		}
////		// Ensure the synchronizer has an ID.
////		if (sync->get_net_id() == 0) {
////			sync->set_net_id(++last_net_id);
////		}
////		sync_ids.push_back(sync->get_net_id());
//	}
//	int state_size = 0;
//	Vector<Variant> state_vars;
//	Vector<const Variant *> state_varp;
//	if (state_props.size()) {
//		Error err = SaveloadSynchronizer::get_state(state_props, p_node, state_vars, state_varp);
//		ERR_FAIL_COND_V_MSG(err != OK, err, "Unable to retrieve spawn state.");
//		err = SaveloadAPI::encode_and_compress_variants(state_varp.ptrw(), state_varp.size(), nullptr, state_size);
//		ERR_FAIL_COND_V_MSG(err != OK, err, "Unable to encode spawn state.");
//	}
//
//	// Encode scene ID, path ID, net ID, node name.
//	int path_id = saveload->get_path_cache()->make_object_cache(p_spawner);
//	CharString cname = p_node->get_name().operator String().utf8();
//	int nlen = encode_cstring(cname.get_data(), nullptr);
//	MAKE_ROOM(1 + 1 + 4 + 4 + 4 + 4 * sync_ids.size() + 4 + nlen + (is_custom ? 4 + spawn_arg_size : 0) + state_size);
//	uint8_t *ptr = packet_cache.ptrw();
//	ptr[0] = (uint8_t)SceneSaveload::SAVELOAD_COMMAND_SPAWN;
//	ptr[1] = scene_id;
//	int ofs = 2;
//	ofs += encode_uint32(path_id, &ptr[ofs]);
//	ofs += encode_uint32(nid, &ptr[ofs]);
//	ofs += encode_uint32(sync_ids.size(), &ptr[ofs]);
//	ofs += encode_uint32(nlen, &ptr[ofs]);
//	for (uint32_t snid : sync_ids) {
//		ofs += encode_uint32(snid, &ptr[ofs]);
//	}
//	ofs += encode_cstring(cname.get_data(), &ptr[ofs]);
//	// Write args
//	if (is_custom) {
//		ofs += encode_uint32(spawn_arg_size, &ptr[ofs]);
//		Error err = SaveloadAPI::encode_and_compress_variant(spawn_arg, &ptr[ofs], spawn_arg_size, false);
//		ERR_FAIL_COND_V(err, err);
//		ofs += spawn_arg_size;
//	}
//	// Write state.
//	if (state_size) {
//		Error err = SaveloadAPI::encode_and_compress_variants(state_varp.ptrw(), state_varp.size(), &ptr[ofs], state_size);
//		ERR_FAIL_COND_V(err, err);
//		ofs += state_size;
//	}
//	r_len = ofs;
//	return OK;
//}

//Error SceneSaveloadInterface::_make_despawn_packet(Node *p_node, int &r_len) {
//	const ObjectID oid = p_node->get_instance_id();
//	const TrackedNode *tnode = tracked_nodes.getptr(oid);
//	ERR_FAIL_COND_V(!tnode, ERR_INVALID_PARAMETER);
//	MAKE_ROOM(5);
//	uint8_t *ptr = packet_cache.ptrw();
//	ptr[0] = (uint8_t)SceneSaveload::SAVELOAD_COMMAND_DESPAWN;
//	int ofs = 1;
//	uint32_t nid = tnode->net_id;
//	ofs += encode_uint32(nid, &ptr[ofs]);
//	r_len = ofs;
//	return OK;
//}

//Error SceneSaveloadInterface::on_despawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) {
//	ERR_FAIL_COND_V_MSG(p_buffer_len < 5, ERR_INVALID_DATA, "Invalid spawn packet received");
//	int ofs = 1; // The spawn/despawn command.
//	uint32_t net_id = decode_uint32(&p_buffer[ofs]);
//	ofs += 4;
//
//	// Untrack remote
//	ERR_FAIL_COND_V(!saveload_info.has(p_from), ERR_UNAUTHORIZED);
//	SaveloadInfo &pinfo = saveload_info[p_from];
//	ERR_FAIL_COND_V(!pinfo.recv_nodes.has(net_id), ERR_UNAUTHORIZED);
//	Node *node = get_id_as<Node>(pinfo.recv_nodes[net_id]);
//	ERR_FAIL_COND_V(!node, ERR_BUG);
//	pinfo.recv_nodes.erase(net_id);
//
//	const ObjectID oid = node->get_instance_id();
//	ERR_FAIL_COND_V(!tracked_nodes.has(oid), ERR_BUG);
//	SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(tracked_nodes[oid].spawner);
//	ERR_FAIL_COND_V(!spawner, ERR_DOES_NOT_EXIST);
////	ERR_FAIL_COND_V(p_from != spawner->get_multiplayer_authority(), ERR_UNAUTHORIZED);
//
//	if (node->get_parent() != nullptr) {
//		node->get_parent()->remove_child(node);
//	}
//	node->queue_free();
//	spawner->emit_signal(SNAME("despawned"), node);
//
//	return OK;
//}

//bool SceneSaveloadInterface::_verify_synchronizer(int p_peer, SaveloadSynchronizer *p_sync, uint32_t &r_net_id) {
//	r_net_id = p_sync->get_net_id();
//	if (r_net_id == 0 || (r_net_id & 0x80000000)) {
//		int path_id = 0;
//		bool verified = saveload->get_path_cache()->send_object_cache(p_sync, p_peer, path_id);
//		ERR_FAIL_COND_V_MSG(path_id < 0, false, "This should never happen!");
//		if (r_net_id == 0) {
//			// First time path based ID.
//			r_net_id = path_id | 0x80000000;
//			p_sync->set_net_id(r_net_id | 0x80000000);
//		}
//		return verified;
//	}
//	return true;
//}

//void SceneSaveloadInterface::_send_delta(int p_peer, const HashSet<ObjectID> p_synchronizers, uint64_t p_usec, const HashMap<ObjectID, uint64_t> p_last_watch_usecs) {
//	MAKE_ROOM(/* header */ 1 + /* element */ 4 + 8 + 4 + delta_mtu);
//	uint8_t *ptr = packet_cache.ptrw();
//	ptr[0] = SceneSaveload::SAVELOAD_COMMAND_SYNC | (1 << SceneSaveload::CMD_FLAG_0_SHIFT);
//	int ofs = 1;
//	for (const ObjectID &oid : p_synchronizers) {
//		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
//		ERR_CONTINUE(!sync || !sync->get_saveload_config().is_valid());
//		uint32_t net_id;
////		if (!_verify_synchronizer(p_peer, sync, net_id)) {
////			continue;
////		}
//		uint64_t last_usec = p_last_watch_usecs.has(oid) ? p_last_watch_usecs[oid] : 0;
//		uint64_t indexes;
//		List<Variant> delta = sync->get_delta_state(p_usec, last_usec, indexes);
//
//		if (!delta.size()) {
//			continue; // Nothing to update.
//		}
//
//		Vector<const Variant *> varp;
//		varp.resize(delta.size());
//		const Variant **vptr = varp.ptrw();
//		int i = 0;
//		for (const Variant &v : delta) {
//			vptr[i] = &v;
//		}
//		int size;
//		Error err = SaveloadAPI::encode_and_compress_variants(vptr, varp.size(), nullptr, size);
//		ERR_CONTINUE_MSG(err != OK, "Unable to encode delta state.");
//
//		ERR_CONTINUE_MSG(size > delta_mtu, vformat("Synchronizer delta bigger than MTU will not be sent (%d > %d): %s", size, delta_mtu, sync->get_path()));
//
//		if (ofs + 4 + 8 + 4 + size > delta_mtu) {
//			// Send what we got, and reset write.
//			//_send_raw(packet_cache.ptr(), ofs, p_peer, true);
//			ofs = 1;
//		}
//		if (size) {
//			ofs += encode_uint32(sync->get_net_id(), &ptr[ofs]);
//			ofs += encode_uint64(indexes, &ptr[ofs]);
//			ofs += encode_uint32(size, &ptr[ofs]);
//			SaveloadAPI::encode_and_compress_variants(vptr, varp.size(), &ptr[ofs], size);
//			ofs += size;
//		}
//#ifdef DEBUG_ENABLED
//		_profile_node_data("delta_out", oid, size);
//#endif
//		saveload_info[p_peer].last_watch_usecs[oid] = p_usec;
//	}
//	if (ofs > 1) {
//		// Got some left over to send.
//		//_send_raw(packet_cache.ptr(), ofs, p_peer, true);
//	}
//}
//
//Error SceneSaveloadInterface::on_delta_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) {
//	int ofs = 1;
//	while (ofs + 4 + 8 + 4 < p_buffer_len) {
//		uint32_t net_id = decode_uint32(&p_buffer[ofs]);
//		ofs += 4;
//		uint64_t indexes = decode_uint64(&p_buffer[ofs]);
//		ofs += 8;
//		uint32_t size = decode_uint32(&p_buffer[ofs]);
//		ofs += 4;
//		ERR_FAIL_COND_V(size > uint32_t(p_buffer_len - ofs), ERR_INVALID_DATA);
//		SaveloadSynchronizer *sync = _find_synchronizer(p_from, net_id);
//		Node *node = sync ? sync->get_root_node() : nullptr;
//		if (!sync || !node) {
//			ofs += size;
//			ERR_CONTINUE_MSG(true, "Ignoring delta for non-authority or invalid synchronizer.");
//		}
//		List<NodePath> props = sync->get_delta_properties(indexes);
//		ERR_FAIL_COND_V(props.size() == 0, ERR_INVALID_DATA);
//		Vector<Variant> vars;
//		vars.resize(props.size());
//		int consumed = 0;
//		Error err = SaveloadAPI::decode_and_decompress_variants(vars, p_buffer + ofs, size, consumed);
//		ERR_FAIL_COND_V(err != OK, err);
//		ERR_FAIL_COND_V(uint32_t(consumed) != size, ERR_INVALID_DATA);
//		err = SaveloadSynchronizer::set_state(props, node, vars);
//		ERR_FAIL_COND_V(err != OK, err);
//		ofs += size;
//		sync->emit_signal(SNAME("delta_synchronized"));
//#ifdef DEBUG_ENABLED
//		_profile_node_data("delta_in", sync->get_instance_id(), size);
//#endif
//	}
//	return OK;
//}

//void SceneSaveloadInterface::_send_sync(int p_peer, const HashSet<ObjectID> p_synchronizers, uint16_t p_sync_net_time, uint64_t p_usec) {
//	MAKE_ROOM(/* header */ 3 + /* element */ 4 + 4 + sync_mtu);
//	uint8_t *ptr = packet_cache.ptrw();
//	ptr[0] = SceneSaveload::NETWORK_COMMAND_SYNC;
//	int ofs = 1;
//	ofs += encode_uint16(p_sync_net_time, &ptr[1]);
//	// Can only send updates for already notified nodes.
//	// This is a lazy implementation, we could optimize much more here with by grouping by replication config.
//	for (const ObjectID &oid : p_synchronizers) {
//		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
//		ERR_CONTINUE(!sync || !sync->get_saveload_config().is_valid());
//		if (!sync->update_outbound_sync_time(p_usec)) {
//			continue; // nothing to sync.
//		}
//
//		Node *node = sync->get_root_node();
//		ERR_CONTINUE(!node);
//		uint32_t net_id = sync->get_net_id();
//		if (!_verify_synchronizer(p_peer, sync, net_id)) {
//			// The path based sync is not yet confirmed, skipping.
//			continue;
//		}
//		int size;
//		Vector<Variant> vars;
//		Vector<const Variant *> varp;
//		const List<NodePath> props = sync->get_saveload_config()->get_sync_properties();
//		Error err = SaveloadSynchronizer::get_state(props, node, vars, varp);
//		ERR_CONTINUE_MSG(err != OK, "Unable to retrieve sync state.");
//		err = SaveloadAPI::encode_and_compress_variants(varp.ptrw(), varp.size(), nullptr, size);
//		ERR_CONTINUE_MSG(err != OK, "Unable to encode sync state.");
//		// TODO Handle single state above MTU.
//		ERR_CONTINUE_MSG(size > sync_mtu, vformat("Node states bigger than MTU will not be sent (%d > %d): %s", size, sync_mtu, node->get_path()));
//		if (ofs + 4 + 4 + size > sync_mtu) {
//			// Send what we got, and reset write.
//			//_send_raw(packet_cache.ptr(), ofs, p_peer, false);
//			ofs = 3;
//		}
//		if (size) {
//			ofs += encode_uint32(sync->get_net_id(), &ptr[ofs]);
//			ofs += encode_uint32(size, &ptr[ofs]);
//			SaveloadAPI::encode_and_compress_variants(varp.ptrw(), varp.size(), &ptr[ofs], size);
//			ofs += size;
//		}
//#ifdef DEBUG_ENABLED
//		_profile_node_data("sync_out", oid, size);
//#endif
//	}
//	if (ofs > 3) {
//		// Got some left over to send.
//		//_send_raw(packet_cache.ptr(), ofs, p_peer, false);
//	}
//}

//void SceneSaveloadInterface::_encode_sync(const HashSet<ObjectID> p_synchronizers, uint16_t p_sync_net_time, uint64_t p_usec) {
//	MAKE_ROOM(/* header */ 3 + /* element */ 4 + 4 + sync_mtu);
//	uint8_t *ptr = packet_cache.ptrw();
//	ptr[0] = SceneSaveload::SAVELOAD_COMMAND_SYNC;
//	int ofs = 1;
//	ofs += encode_uint16(p_sync_net_time, &ptr[1]);
//	// Can only send updates for already notified nodes.
//	// This is a lazy implementation, we could optimize much more here with by grouping by replication config.
//	for (const ObjectID &oid : p_synchronizers) {
//		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
//		ERR_CONTINUE(!sync || !sync->get_saveload_config().is_valid());
//		if (!sync->update_outbound_sync_time(p_usec)) {
//			continue; // nothing to sync.
//		}
//
//		Node *node = sync->get_root_node();
//		ERR_CONTINUE(!node);
//		uint32_t net_id = sync->get_net_id();
////		if (!_verify_synchronizer(p_peer, sync, net_id)) {
////			// The path based sync is not yet confirmed, skipping.
////			continue;
////		}
//		int size;
//		Vector<Variant> vars;
//		Vector<const Variant *> varp;
//		const List<NodePath> props = sync->get_saveload_config()->get_sync_properties();
//		Error err = SaveloadSynchronizer::get_state(props, node, vars, varp);
//		ERR_CONTINUE_MSG(err != OK, "Unable to retrieve sync state.");
//		err = SaveloadAPI::encode_and_compress_variants(varp.ptrw(), varp.size(), nullptr, size);
//		ERR_CONTINUE_MSG(err != OK, "Unable to encode sync state.");
//		// TODO Handle single state above MTU.
//		ERR_CONTINUE_MSG(size > sync_mtu, vformat("Node states bigger than MTU will not be sent (%d > %d): %s", size, sync_mtu, node->get_path()));
//		if (ofs + 4 + 4 + size > sync_mtu) {
//			// Send what we got, and reset write.
//			//_send_raw(packet_cache.ptr(), ofs, p_peer, false);
//			ofs = 3;
//		}
//		if (size) {
//			ofs += encode_uint32(sync->get_net_id(), &ptr[ofs]);
//			ofs += encode_uint32(size, &ptr[ofs]);
//			SaveloadAPI::encode_and_compress_variants(varp.ptrw(), varp.size(), &ptr[ofs], size);
//			ofs += size;
//		}
//#ifdef DEBUG_ENABLED
//		_profile_node_data("sync_out", oid, size);
//#endif
//	}
//	if (ofs > 3) {
//		// Got some left over to send.
//		//_send_raw(packet_cache.ptr(), ofs, p_peer, false);
//	}
//}
//
//Error SceneSaveloadInterface::on_sync_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) {
//	ERR_FAIL_COND_V_MSG(p_buffer_len < 11, ERR_INVALID_DATA, "Invalid sync packet received");
//	bool is_delta = (p_buffer[0] & (1 << SceneSaveload::CMD_FLAG_0_SHIFT)) != 0;
//	if (is_delta) {
//		return on_delta_receive(p_from, p_buffer, p_buffer_len);
//	}
//	uint16_t time = decode_uint16(&p_buffer[1]);
//	int ofs = 3;
//	while (ofs + 8 < p_buffer_len) {
//		uint32_t net_id = decode_uint32(&p_buffer[ofs]);
//		ofs += 4;
//		uint32_t size = decode_uint32(&p_buffer[ofs]);
//		ofs += 4;
//		ERR_FAIL_COND_V(size > uint32_t(p_buffer_len - ofs), ERR_INVALID_DATA);
//		SaveloadSynchronizer *sync = _find_synchronizer(p_from, net_id);
//		if (!sync) {
//			// Not received yet.
//			ofs += size;
//			continue;
//		}
//		Node *node = sync->get_root_node();
//		if (!node) {
//			// Not valid for me.
//			ofs += size;
//			ERR_CONTINUE_MSG(true, "Ignoring sync data from non-authority or for missing node.");
//		}
//		if (!sync->update_inbound_sync_time(time)) {
//			// State is too old.
//			ofs += size;
//			continue;
//		}
//		const List<NodePath> props = sync->get_saveload_config()->get_sync_properties();
//		Vector<Variant> vars;
//		vars.resize(props.size());
//		int consumed;
//		Error err = SaveloadAPI::decode_and_decompress_variants(vars, &p_buffer[ofs], size, consumed);
//		ERR_FAIL_COND_V(err, err);
//		err = SaveloadSynchronizer::set_state(props, node, vars);
//		ERR_FAIL_COND_V(err, err);
//		ofs += size;
//		sync->emit_signal(SNAME("synchronized"));
//#ifdef DEBUG_ENABLED
//		_profile_node_data("sync_in", sync->get_instance_id(), size);
//#endif
//	}
//	return OK;
//}

PackedByteArray SceneSaveloadInterface::encode(Object *p_obj, const StringName section) {
	flush_spawn_queue();
	return packet_cache;
}
Error SceneSaveloadInterface::decode(PackedByteArray p_bytes, Object *p_obj, const StringName section) {
	return ERR_UNAVAILABLE; //TODO: not sure what to do here
}
