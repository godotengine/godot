/*************************************************************************/
/*  multiplayer_synchronizer.cpp                                         */
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

#include "multiplayer_synchronizer.h"

#include "core/config/engine.h"
#include "core/multiplayer/multiplayer_api.h"
#include "scene/scene_string_names.h"

void MultiplayerSynchronizer::_bind_methods() {
	// Must be bound for rpc_config to work.
	ClassDB::bind_method(D_METHOD("_rpc_synchronize_reliable", "payload"), &MultiplayerSynchronizer::_rpc_synchronize_reliable);
	ClassDB::bind_method(D_METHOD("_rpc_synchronize", "payload"), &MultiplayerSynchronizer::_rpc_synchronize);
	ClassDB::bind_method(D_METHOD("_rpc_request_synchronize"), &MultiplayerSynchronizer::_rpc_request_synchronize);

	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &MultiplayerSynchronizer::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &MultiplayerSynchronizer::get_root_path);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");

	ClassDB::bind_method(D_METHOD("set_replication_interval", "interval"), &MultiplayerSynchronizer::set_replication_interval);
	ClassDB::bind_method(D_METHOD("get_replication_interval"), &MultiplayerSynchronizer::get_replication_interval);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "replication_interval", PROPERTY_HINT_RANGE, "0,5,0.001,suffix:s"), "set_replication_interval", "get_replication_interval");

	ClassDB::bind_method(D_METHOD("set_replication_config", "config"), &MultiplayerSynchronizer::set_replication_config);
	ClassDB::bind_method(D_METHOD("get_replication_config"), &MultiplayerSynchronizer::get_replication_config);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "replication_config", PROPERTY_HINT_RESOURCE_TYPE, "SceneReplicationConfig"), "set_replication_config", "get_replication_config");
}

void MultiplayerSynchronizer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			set_process_internal(true);

			_update_root_node();

			// Deferred signal to ensure the spawn was handled first.
			get_multiplayer()->connect(SNAME("peer_connected"), callable_mp(this, &MultiplayerSynchronizer::_on_peer_connected), varray(), CONNECT_DEFERRED);
			//get_multiplayer()->connect(SNAME("peer_disconnected"), callable_mp(this, &MultiplayerSynchronizer::_on_peer_disconnected));

			// When this node is added on a client, request synchronize from the server.
			if (get_multiplayer()->has_multiplayer_peer() &&
					get_multiplayer()->get_multiplayer_peer()->get_connection_status() == MultiplayerPeer::CONNECTION_CONNECTED &&
					!is_multiplayer_authority()) {
				rpc_id(get_multiplayer_authority(), SNAME("_rpc_request_synchronize"));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			get_multiplayer()->disconnect(SNAME("peer_connected"), callable_mp(this, &MultiplayerSynchronizer::_on_peer_connected));
			//get_multiplayer()->disconnect(SNAME("peer_disconnected"), callable_mp(this, &MultiplayerSynchronizer::_on_peer_disconnected));
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			_synchronize(0, SYNC);
		} break;
	}
}

bool MultiplayerSynchronizer::_get_path_target(const NodePath &p_path, Node *&r_node, StringName &r_prop) {
	if (p_path.get_name_count() != 0) {
		r_node = root_node->get_node(String(p_path.get_concatenated_names()));
	} else {
		r_node = root_node;
	}
	ERR_FAIL_NULL_V(r_node, false);

	r_prop = p_path.get_concatenated_subnames();
	return true;
}

void MultiplayerSynchronizer::_update_root_node() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif

	root_node = is_inside_tree() ? get_node_or_null(root_path) : nullptr;
}

bool MultiplayerSynchronizer::_create_payload(const SynchronizeAction p_action, Array &r_payload) {
	bool has_values = false;
	r_payload.push_back(p_action);

	List<NodePath> properties;
	switch (p_action) {
		case READY: {
			properties = replication_config->get_spawn_properties();
		} break;
		case SYNC: {
			properties = replication_config->get_sync_properties();
		} break;
	}

	for (const NodePath &path : properties) {
		Node *node;
		StringName prop;
		const bool found_node = _get_path_target(path, node, prop);
		ERR_CONTINUE(!found_node);

		Variant value = node->get(prop);

		r_payload.push_back(value);
		has_values = true;
	}

	return has_values;
}

void MultiplayerSynchronizer::_apply_payload(const Array &p_payload) {
	int index = 0;

	const int action = p_payload[index++];

	List<NodePath> properties;
	switch (action) {
		case READY: {
			properties = replication_config->get_spawn_properties();
		} break;
		case SYNC: {
			properties = replication_config->get_sync_properties();
		} break;
		default: {
			ERR_FAIL_MSG("Invalid action type: " + itos(action));
		} break;
	}

	for (const NodePath &path : properties) {
		Node *node;
		StringName prop;
		const bool found_node = _get_path_target(path, node, prop);
		ERR_CONTINUE(!found_node);

		Variant value = p_payload[index++];
		node->set(prop, value);
	}

	ERR_FAIL_COND_MSG(index != p_payload.size(), "Synchronization failed. The server and client have different configurations!");
}

void MultiplayerSynchronizer::_on_peer_connected(const int p_peer) {
	if (p_peer == get_multiplayer_authority()) {
		rpc_id(get_multiplayer_authority(), SNAME("_rpc_request_synchronize"));
	}
}

// Not needed currently.
/*void MultiplayerSpawner::_on_peer_disconnected(const int p_peer) {
}*/

void MultiplayerSynchronizer::_rpc_synchronize_reliable(const Array &p_payload) {
	ERR_FAIL_COND(p_payload.is_empty());

	_apply_payload(p_payload);
}

void MultiplayerSynchronizer::_rpc_synchronize(const Array &p_payload) {
	ERR_FAIL_COND(p_payload.is_empty());

	_apply_payload(p_payload);
}

void MultiplayerSynchronizer::_rpc_request_synchronize() {
	const int peer = get_multiplayer()->get_remote_sender_id();
	ERR_FAIL_COND(peer == 0);

	const bool erased = watching_peers.erase(peer);
	if (erased) {
#ifdef DEBUG_ENABLED
		WARN_PRINT("Client requested another ready synchronize.");
#endif
	}

	watching_peers.push_back(peer);
	_synchronize(peer, READY);
}

void MultiplayerSynchronizer::_synchronize(const int p_peer, const SynchronizeAction p_action) {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}
#endif

	// Was already synchronized directly by a spawner.
	if (p_action == READY && spawn_synced) {
		return;
	}

	if (!get_multiplayer()->has_multiplayer_peer() ||
			get_multiplayer()->get_multiplayer_peer()->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED ||
			!is_multiplayer_authority()) {
		return;
	}

	ERR_FAIL_NULL(replication_config);

	Array payload;
	bool payload_has_values = _create_payload(p_action, payload);

	if (payload_has_values) {
		switch (p_action) {
			case READY: {
				if (p_peer != 0) {
					rpc_id(p_peer, SNAME("_rpc_synchronize_reliable"), payload);
				} else {
					for (const int peer : watching_peers) {
						rpc_id(peer, SNAME("_rpc_synchronize_reliable"), payload);
					}
				}
			} break;
			case SYNC: {
				if (p_peer != 0) {
					rpc_id(p_peer, SNAME("_rpc_synchronize"), payload);
				} else {
					for (const int peer : watching_peers) {
						rpc_id(peer, SNAME("_rpc_synchronize"), payload);
					}
				}
			} break;
		}
	}
}

void MultiplayerSynchronizer::set_root_path(const NodePath &p_path) {
	root_path = p_path;
}

NodePath MultiplayerSynchronizer::get_root_path() const {
	return root_path;
}

void MultiplayerSynchronizer::set_replication_interval(double p_interval) {
	ERR_FAIL_COND_MSG(p_interval < 0, "Replication interval must be greater or equal to 0.");
	replication_interval = p_interval;
}

double MultiplayerSynchronizer::get_replication_interval() const {
	return replication_interval;
}

void MultiplayerSynchronizer::set_replication_config(Ref<SceneReplicationConfig> p_config) {
	replication_config = p_config;
}

Ref<SceneReplicationConfig> MultiplayerSynchronizer::get_replication_config() {
	return replication_config;
}

MultiplayerSynchronizer::MultiplayerSynchronizer() {
	rpc_config(SNAME("_rpc_synchronize_reliable"), RPCMode::RPC_MODE_AUTHORITY);
	rpc_config(SNAME("_rpc_synchronize"), RPCMode::RPC_MODE_AUTHORITY, false, Multiplayer::TRANSFER_MODE_UNRELIABLE_ORDERED);
	rpc_config(SNAME("_rpc_request_synchronize"), RPCMode::RPC_MODE_ANY_PEER);
}
