/*************************************************************************/
/*  scene_replication_interface.h                                        */
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

#ifndef SCENE_REPLICATION_INTERFACE_H
#define SCENE_REPLICATION_INTERFACE_H

#include "scene/main/multiplayer_api.h"

#include "scene_replication_state.h"

class SceneMultiplayer;

class SceneReplicationInterface : public RefCounted {
	GDCLASS(SceneReplicationInterface, RefCounted);

private:
	void _send_sync(int p_peer, uint64_t p_msec);
	Error _make_spawn_packet(Node *p_node, int &r_len);
	Error _make_despawn_packet(Node *p_node, int &r_len);
	Error _send_raw(const uint8_t *p_buffer, int p_size, int p_peer, bool p_reliable);

	void _visibility_changed(int p_peer, ObjectID p_oid);
	Error _update_sync_visibility(int p_peer, const ObjectID &p_oid);
	Error _update_spawn_visibility(int p_peer, const ObjectID &p_oid);
	void _free_remotes(int p_peer);

	Ref<SceneReplicationState> rep_state;
	SceneMultiplayer *multiplayer = nullptr;
	PackedByteArray packet_cache;
	int sync_mtu = 1350; // Highly dependent on underlying protocol.

	// An hack to apply the initial state before ready.
	ObjectID pending_spawn;
	const uint8_t *pending_buffer = nullptr;
	int pending_buffer_size = 0;

public:
	static void make_default();

	void on_reset();
	void on_peer_change(int p_id, bool p_connected);

	Error on_spawn(Object *p_obj, Variant p_config);
	Error on_despawn(Object *p_obj, Variant p_config);
	Error on_replication_start(Object *p_obj, Variant p_config);
	Error on_replication_stop(Object *p_obj, Variant p_config);
	void on_network_process();

	Error on_spawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len);
	Error on_despawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len);
	Error on_sync_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len);

	SceneReplicationInterface(SceneMultiplayer *p_multiplayer) {
		rep_state.instantiate();
		multiplayer = p_multiplayer;
	}
};

#endif // SCENE_REPLICATION_INTERFACE_H
