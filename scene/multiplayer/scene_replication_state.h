/*************************************************************************/
/*  scene_replication_state.h                                            */
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

#ifndef SCENE_REPLICATON_STATE_H
#define SCENE_REPLICATON_STATE_H

#include "core/object/ref_counted.h"

class MultiplayerSpawner;
class MultiplayerSynchronizer;
class Node;

class SceneReplicationState : public RefCounted {
private:
	struct TrackedNode {
		ObjectID id;
		uint32_t net_id = 0;
		uint32_t remote_peer = 0;
		ObjectID spawner;
		ObjectID synchronizer;
		uint16_t last_sync = 0;
		uint64_t last_sync_msec = 0;

		bool operator==(const ObjectID &p_other) { return id == p_other; }

		Node *get_node() const { return id.is_valid() ? Object::cast_to<Node>(ObjectDB::get_instance(id)) : nullptr; }
		MultiplayerSpawner *get_spawner() const { return spawner.is_valid() ? Object::cast_to<MultiplayerSpawner>(ObjectDB::get_instance(spawner)) : nullptr; }
		MultiplayerSynchronizer *get_synchronizer() const { return synchronizer.is_valid() ? Object::cast_to<MultiplayerSynchronizer>(ObjectDB::get_instance(synchronizer)) : nullptr; }
		TrackedNode() {}
		TrackedNode(const ObjectID &p_id) { id = p_id; }
		TrackedNode(const ObjectID &p_id, uint32_t p_net_id) {
			id = p_id;
			net_id = p_net_id;
		}
	};

	struct PeerInfo {
		RBSet<ObjectID> known_nodes;
		HashMap<uint32_t, ObjectID> recv_nodes;
		uint16_t last_sent_sync = 0;
		uint16_t last_recv_sync = 0;
	};

	RBSet<int> known_peers;
	uint32_t last_net_id = 0;
	HashMap<ObjectID, TrackedNode> tracked_nodes;
	HashMap<int, PeerInfo> peers_info;
	RBSet<ObjectID> spawned_nodes;
	RBSet<ObjectID> path_only_nodes;

	TrackedNode &_track(const ObjectID &p_id);
	void _untrack(const ObjectID &p_id);
	bool is_tracked(const ObjectID &p_id) const { return tracked_nodes.has(p_id); }

public:
	const RBSet<int> get_peers() const { return known_peers; }
	const RBSet<ObjectID> &get_spawned_nodes() const { return spawned_nodes; }
	const RBSet<ObjectID> &get_path_only_nodes() const { return path_only_nodes; }

	MultiplayerSynchronizer *get_synchronizer(const ObjectID &p_id) { return tracked_nodes.has(p_id) ? tracked_nodes[p_id].get_synchronizer() : nullptr; }
	MultiplayerSpawner *get_spawner(const ObjectID &p_id) { return tracked_nodes.has(p_id) ? tracked_nodes[p_id].get_spawner() : nullptr; }
	Node *get_node(const ObjectID &p_id) { return tracked_nodes.has(p_id) ? tracked_nodes[p_id].get_node() : nullptr; }
	bool update_last_node_sync(const ObjectID &p_id, uint16_t p_time);
	bool update_sync_time(const ObjectID &p_id, uint64_t p_msec);

	const RBSet<ObjectID> get_known_nodes(int p_peer);
	uint32_t get_net_id(const ObjectID &p_id) const;
	void set_net_id(const ObjectID &p_id, uint32_t p_net_id);
	uint32_t ensure_net_id(const ObjectID &p_id);

	void reset();
	void on_peer_change(int p_peer, bool p_connected);

	Error config_add_spawn(Node *p_node, MultiplayerSpawner *p_spawner);
	Error config_del_spawn(Node *p_node, MultiplayerSpawner *p_spawner);

	Error config_add_sync(Node *p_node, MultiplayerSynchronizer *p_sync);
	Error config_del_sync(Node *p_node, MultiplayerSynchronizer *p_sync);

	Error peer_add_node(int p_peer, const ObjectID &p_id);
	Error peer_del_node(int p_peer, const ObjectID &p_id);

	const HashMap<uint32_t, ObjectID> peer_get_remotes(int p_peer) const;
	Node *peer_get_remote(int p_peer, uint32_t p_net_id);
	Error peer_add_remote(int p_peer, uint32_t p_net_id, Node *p_node, MultiplayerSpawner *p_spawner);
	Error peer_del_remote(int p_peer, uint32_t p_net_id, Node **r_node);

	uint16_t peer_sync_next(int p_peer);
	void peer_sync_recv(int p_peer, uint16_t p_time);

	SceneReplicationState() {}
};

#endif // SCENE_REPLICATON_STATE_H
