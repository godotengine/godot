/**************************************************************************/
/*  scene_saveload_interface.h                                            */
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

#ifndef SCENE_SAVELOAD_INTERFACE_H
#define SCENE_SAVELOAD_INTERFACE_H

#include "core/object/ref_counted.h"

#include "saveload_spawner.h"
#include "saveload_synchronizer.h"

class SceneSaveload;

class SceneSaveloadInterface : public RefCounted {
	GDCLASS(SceneSaveloadInterface, RefCounted);

public:
	struct SaveloadState {
		HashMap<const NodePath, SaveloadSpawner::SpawnState> spawn_states;
		HashMap<const NodePath, SaveloadSynchronizer::SyncState> sync_states;

		Dictionary to_dict();

		SaveloadState() {}
		SaveloadState(const Dictionary &saveload_dict);
	};

private:

	// Replication state.
	SaveloadState saveload_state_cache;
	HashSet<ObjectID> spawn_nodes;
	HashSet<ObjectID> sync_nodes;

	// Pending local spawn information (handles spawning nested nodes during ready).
	HashSet<ObjectID> spawn_queue;

	// Replicator config.
	SceneSaveload *saveload = nullptr;
	PackedByteArray packet_cache;

	void _node_ready(const ObjectID &p_oid);

//	bool _verify_synchronizer(int p_peer, SaveloadSynchronizer *p_sync, uint32_t &r_net_id);
//	SaveloadSynchronizer *_find_synchronizer(int p_peer, uint32_t p_net_ida);
//
//	void _send_delta(int p_peer, const HashSet<ObjectID> p_synchronizers, uint64_t p_usec, const HashMap<ObjectID, uint64_t> p_last_watch_usecs);
//	Error _make_spawn_packet(Node *p_node, SaveloadSpawner *p_spawner, int &r_len);
//	Error _make_despawn_packet(Node *p_node, int &r_len);
//	Error _send_raw(const uint8_t *p_buffer, int p_size, int p_peer, bool p_reliable);

//	void _encode_sync(const HashSet<ObjectID> p_synchronizers, uint16_t p_sync_net_time, uint64_t p_usec);
//	void _encode_spawn(Node *p_node, SaveloadSpawner *p_spawner, int &r_len);
//	void _encode_despawn(Node *p_node, int &r_len);

//	void _visibility_changed(int p_peer, ObjectID p_oid);
//	Error _update_sync_visibility(int p_peer, SaveloadSynchronizer *p_sync);
//	Error _update_spawn_visibility(int p_peer, const ObjectID &p_oid);
//	void _free_remotes(const PeerInfo &p_info);

	template <class T>
	static T *get_id_as(const ObjectID &p_id) {
		return p_id.is_valid() ? Object::cast_to<T>(ObjectDB::get_instance(p_id)) : nullptr;
	}

#ifdef DEBUG_ENABLED
	_FORCE_INLINE_ void _profile_node_data(const String &p_what, ObjectID p_id, int p_size);
#endif

public:
	static void make_default();

	void on_reset();
//	void on_peer_change(int p_id, bool p_connected);

	void configure_spawn(Node *p_node, const SaveloadSpawner &p_spawner);
	void configure_sync(Node *p_node, const SaveloadSynchronizer &p_syncher);

	void deconfigure_spawn(const SaveloadSpawner &p_spawner);
	void deconfigure_sync(const SaveloadSynchronizer &p_syncher);

	TypedArray<SaveloadSynchronizer> get_sync_nodes();
	Dictionary get_sync_state();

	SaveloadState get_saveload_state();
	void load_saveload_state(const SaveloadState p_saveload_state);

	PackedByteArray encode(Object *p_obj, const StringName section = "");
	Error decode(PackedByteArray p_bytes, Object *p_obj, const StringName section = "");

	void flush_spawn_queue();

	SceneSaveloadInterface(SceneSaveload *p_saveload) {
		saveload = p_saveload;
	}
};

#endif // SCENE_SAVELOAD_INTERFACE_H
