/*************************************************************************/
/*  multiplayer_synchronizer.h                                           */
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

#ifndef MULTIPLAYER_SYNCHRONIZER_H
#define MULTIPLAYER_SYNCHRONIZER_H

#include "core/templates/local_vector.h"
#include "core/variant/typed_array.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"

#include "scene_replication_config.h"

class MultiplayerSpawner;

class MultiplayerSynchronizer : public Node {
	GDCLASS(MultiplayerSynchronizer, Node);

	friend class MultiplayerSpawner;

private:
	NodePath root_path = NodePath("..");
	real_t replication_interval = 0.0;
	Ref<SceneReplicationConfig> replication_config;

	Node *root_node = nullptr;
	bool spawn_synced = false;

	List<int> watching_peers;

protected:
	enum SynchronizeAction {
		READY,
		SYNC,
	};

	static void _bind_methods();
	void _notification(int p_what);

	bool _get_path_target(const NodePath &p_path, Node *&r_node, StringName &r_prop);

	void _update_root_node();

	bool _create_payload(const SynchronizeAction p_action, Array &r_payload);
	void _apply_payload(const Array &p_payload);

	void _on_peer_connected(const int p_peer);
	//void _on_peer_disconnected(const int p_peer);

	void _rpc_synchronize_reliable(const Array &p_payload);
	void _rpc_synchronize(const Array &p_payload);
	void _rpc_request_synchronize();

	void _synchronize(const int p_peer, const SynchronizeAction p_action);

public:
	NodePath get_root_path() const;
	void set_root_path(const NodePath &p_path);

	void set_replication_interval(double p_interval);
	double get_replication_interval() const;

	void set_replication_config(Ref<SceneReplicationConfig> p_config);
	Ref<SceneReplicationConfig> get_replication_config();

	MultiplayerSynchronizer();
};

#endif // MULTIPLAYER_SYNCHRONIZER_H
