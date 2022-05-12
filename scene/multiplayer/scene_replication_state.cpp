/*************************************************************************/
/*  scene_replication_state.cpp                                          */
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

#include "scene/multiplayer/scene_replication_state.h"

#include "core/multiplayer/multiplayer_api.h"
#include "scene/multiplayer/multiplayer_spawner.h"
#include "scene/multiplayer/multiplayer_synchronizer.h"
#include "scene/scene_string_names.h"

SceneReplicationState::TrackedNode &SceneReplicationState::_track(const ObjectID &p_id) {
	if (!tracked_nodes.has(p_id)) {
		tracked_nodes[p_id] = TrackedNode(p_id);
		Node *node = Object::cast_to<Node>(ObjectDB::get_instance(p_id));
		node->connect(SceneStringNames::get_singleton()->tree_exited, callable_mp(this, &SceneReplicationState::_untrack), varray(p_id), Node::CONNECT_ONESHOT);
	}
	return tracked_nodes[p_id];
}

void SceneReplicationState::_untrack(const ObjectID &p_id) {
	if (tracked_nodes.has(p_id)) {
		uint32_t net_id = tracked_nodes[p_id].net_id;
		uint32_t peer = tracked_nodes[p_id].remote_peer;
		tracked_nodes.erase(p_id);
		// If it was spawned by a remote, remove it from the received nodes.
		if (peer && peers_info.has(peer)) {
			peers_info[peer].recv_nodes.erase(net_id);
		}
		// If we spawned or synced it, we need to remove it from any peer it was sent to.
		if (net_id || peer == 0) {
			for (KeyValue<int, PeerInfo> &E : peers_info) {
				E.value.known_nodes.erase(p_id);
			}
		}
	}
}

const HashMap<uint32_t, ObjectID> SceneReplicationState::peer_get_remotes(int p_peer) const {
	return peers_info.has(p_peer) ? peers_info[p_peer].recv_nodes : HashMap<uint32_t, ObjectID>();
}

bool SceneReplicationState::update_last_node_sync(const ObjectID &p_id, uint16_t p_time) {
	TrackedNode *tnode = tracked_nodes.getptr(p_id);
	ERR_FAIL_COND_V(!tnode, false);
	if (p_time <= tnode->last_sync && tnode->last_sync - p_time < 32767) {
		return false;
	}
	tnode->last_sync = p_time;
	return true;
}

bool SceneReplicationState::update_sync_time(const ObjectID &p_id, uint64_t p_msec) {
	TrackedNode *tnode = tracked_nodes.getptr(p_id);
	ERR_FAIL_COND_V(!tnode, false);
	MultiplayerSynchronizer *sync = get_synchronizer(p_id);
	if (!sync) {
		return false;
	}
	if (tnode->last_sync_msec == p_msec) {
		return true;
	}
	if (p_msec >= tnode->last_sync_msec + sync->get_replication_interval_msec()) {
		tnode->last_sync_msec = p_msec;
		return true;
	}
	return false;
}

const Set<ObjectID> SceneReplicationState::get_known_nodes(int p_peer) {
	ERR_FAIL_COND_V(!peers_info.has(p_peer), Set<ObjectID>());
	return peers_info[p_peer].known_nodes;
}

uint32_t SceneReplicationState::get_net_id(const ObjectID &p_id) const {
	const TrackedNode *tnode = tracked_nodes.getptr(p_id);
	ERR_FAIL_COND_V(!tnode, 0);
	return tnode->net_id;
}

void SceneReplicationState::set_net_id(const ObjectID &p_id, uint32_t p_net_id) {
	TrackedNode *tnode = tracked_nodes.getptr(p_id);
	ERR_FAIL_COND(!tnode);
	tnode->net_id = p_net_id;
}

uint32_t SceneReplicationState::ensure_net_id(const ObjectID &p_id) {
	TrackedNode *tnode = tracked_nodes.getptr(p_id);
	ERR_FAIL_COND_V(!tnode, 0);
	if (tnode->net_id == 0) {
		tnode->net_id = ++last_net_id;
	}
	return tnode->net_id;
}

void SceneReplicationState::on_peer_change(int p_peer, bool p_connected) {
	if (p_connected) {
		peers_info[p_peer] = PeerInfo();
		known_peers.insert(p_peer);
	} else {
		peers_info.erase(p_peer);
		known_peers.erase(p_peer);
	}
}

void SceneReplicationState::reset() {
	peers_info.clear();
	known_peers.clear();
	// Tracked nodes are cleared on deletion, here we only reset the ids so they can be later re-assigned.
	for (KeyValue<ObjectID, TrackedNode> &E : tracked_nodes) {
		TrackedNode &tobj = E.value;
		tobj.net_id = 0;
		tobj.remote_peer = 0;
		tobj.last_sync = 0;
	}
}

Error SceneReplicationState::config_add_spawn(Node *p_node, MultiplayerSpawner *p_spawner) {
	const ObjectID oid = p_node->get_instance_id();
	TrackedNode &tobj = _track(oid);
	ERR_FAIL_COND_V(tobj.spawner != ObjectID(), ERR_ALREADY_IN_USE);
	tobj.spawner = p_spawner->get_instance_id();
	spawned_nodes.insert(oid);
	// The spawner may be notified after the synchronizer.
	path_only_nodes.erase(oid);
	return OK;
}

Error SceneReplicationState::config_del_spawn(Node *p_node, MultiplayerSpawner *p_spawner) {
	const ObjectID oid = p_node->get_instance_id();
	ERR_FAIL_COND_V(!is_tracked(oid), ERR_INVALID_PARAMETER);
	TrackedNode &tobj = _track(oid);
	ERR_FAIL_COND_V(tobj.spawner != p_spawner->get_instance_id(), ERR_INVALID_PARAMETER);
	tobj.spawner = ObjectID();
	spawned_nodes.erase(oid);
	return OK;
}

Error SceneReplicationState::config_add_sync(Node *p_node, MultiplayerSynchronizer *p_sync) {
	const ObjectID oid = p_node->get_instance_id();
	TrackedNode &tobj = _track(oid);
	ERR_FAIL_COND_V(tobj.synchronizer != ObjectID(), ERR_ALREADY_IN_USE);
	tobj.synchronizer = p_sync->get_instance_id();
	// If it doesn't have a spawner, we might need to assign ID for this node using it's path.
	if (tobj.spawner.is_null()) {
		path_only_nodes.insert(oid);
	}
	return OK;
}

Error SceneReplicationState::config_del_sync(Node *p_node, MultiplayerSynchronizer *p_sync) {
	const ObjectID oid = p_node->get_instance_id();
	ERR_FAIL_COND_V(!is_tracked(oid), ERR_INVALID_PARAMETER);
	TrackedNode &tobj = _track(oid);
	ERR_FAIL_COND_V(tobj.synchronizer != p_sync->get_instance_id(), ERR_INVALID_PARAMETER);
	tobj.synchronizer = ObjectID();
	if (path_only_nodes.has(oid)) {
		p_node->disconnect(SceneStringNames::get_singleton()->tree_exited, callable_mp(this, &SceneReplicationState::_untrack));
		_untrack(oid);
		path_only_nodes.erase(oid);
	}
	return OK;
}

Error SceneReplicationState::peer_add_node(int p_peer, const ObjectID &p_id) {
	if (p_peer) {
		ERR_FAIL_COND_V(!peers_info.has(p_peer), ERR_INVALID_PARAMETER);
		peers_info[p_peer].known_nodes.insert(p_id);
	} else {
		for (KeyValue<int, PeerInfo> &E : peers_info) {
			E.value.known_nodes.insert(p_id);
		}
	}
	return OK;
}

Error SceneReplicationState::peer_del_node(int p_peer, const ObjectID &p_id) {
	if (p_peer) {
		ERR_FAIL_COND_V(!peers_info.has(p_peer), ERR_INVALID_PARAMETER);
		peers_info[p_peer].known_nodes.erase(p_id);
	} else {
		for (KeyValue<int, PeerInfo> &E : peers_info) {
			E.value.known_nodes.erase(p_id);
		}
	}
	return OK;
}

Node *SceneReplicationState::peer_get_remote(int p_peer, uint32_t p_net_id) {
	PeerInfo *info = peers_info.getptr(p_peer);
	return info && info->recv_nodes.has(p_net_id) ? Object::cast_to<Node>(ObjectDB::get_instance(info->recv_nodes[p_net_id])) : nullptr;
}

Error SceneReplicationState::peer_add_remote(int p_peer, uint32_t p_net_id, Node *p_node, MultiplayerSpawner *p_spawner) {
	ERR_FAIL_COND_V(!p_node || !p_spawner, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!peers_info.has(p_peer), ERR_UNAVAILABLE);
	PeerInfo &pinfo = peers_info[p_peer];
	ObjectID oid = p_node->get_instance_id();
	TrackedNode &tobj = _track(oid);
	tobj.spawner = p_spawner->get_instance_id();
	tobj.net_id = p_net_id;
	tobj.remote_peer = p_peer;
	tobj.last_sync = pinfo.last_recv_sync;
	// Also track as a remote.
	ERR_FAIL_COND_V(pinfo.recv_nodes.has(p_net_id), ERR_ALREADY_IN_USE);
	pinfo.recv_nodes[p_net_id] = oid;
	return OK;
}

Error SceneReplicationState::peer_del_remote(int p_peer, uint32_t p_net_id, Node **r_node) {
	ERR_FAIL_COND_V(!peers_info.has(p_peer), ERR_UNAUTHORIZED);
	PeerInfo &info = peers_info[p_peer];
	ERR_FAIL_COND_V(!info.recv_nodes.has(p_net_id), ERR_UNAUTHORIZED);
	*r_node = Object::cast_to<Node>(ObjectDB::get_instance(info.recv_nodes[p_net_id]));
	info.recv_nodes.erase(p_net_id);
	return OK;
}

uint16_t SceneReplicationState::peer_sync_next(int p_peer) {
	ERR_FAIL_COND_V(!peers_info.has(p_peer), 0);
	PeerInfo &info = peers_info[p_peer];
	return ++info.last_sent_sync;
}

void SceneReplicationState::peer_sync_recv(int p_peer, uint16_t p_time) {
	ERR_FAIL_COND(!peers_info.has(p_peer));
	peers_info[p_peer].last_recv_sync = p_time;
}
