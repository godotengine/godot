/*************************************************************************/
/*  multiplayer_replicator.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MULTIPLAYER_REPLICATOR_H
#define MULTIPLAYER_REPLICATOR_H

#include "core/io/multiplayer_api.h"
#include "core/variant/typed_array.h"

class MultiplayerReplicator : public Object {
	GDCLASS(MultiplayerReplicator, Object);

public:
	enum {
		SPAWN_CMD_OFFSET = 9,
	};

	enum ReplicationMode {
		REPLICATION_MODE_NONE,
		REPLICATION_MODE_SERVER,
		REPLICATION_MODE_CUSTOM,
	};

	struct SceneConfig {
		ReplicationMode mode;
		List<StringName> properties;
		Callable on_spawn_despawn_send;
		Callable on_spawn_despawn_receive;
	};

protected:
	static void _bind_methods();

private:
	MultiplayerAPI *multiplayer = nullptr;
	Vector<uint8_t> packet_cache;
	Map<ResourceUID::ID, SceneConfig> replications;
	Map<ObjectID, ResourceUID::ID> replicated_nodes;

	Error _encode_state(const List<Variant> &p_variants, uint8_t *p_buffer, int &r_len, bool *r_raw = nullptr);
	Error _decode_state(const List<StringName> &p_cfg, Object *p_obj, const uint8_t *p_buffer, int p_len, int &r_len, bool p_raw = false);
	Error _get_state(const List<StringName> &p_properties, const Object *p_obj, List<Variant> &r_variant);
	Error _spawn_despawn(ResourceUID::ID p_scene_id, Object *p_obj, int p_peer, bool p_spawn);
	Error _send_spawn_despawn(int p_peer_id, const ResourceUID::ID &p_scene_id, const Variant &p_data, bool p_spawn);
	void _process_default_spawn_despawn(int p_from, const ResourceUID::ID &p_scene_id, const uint8_t *p_packet, int p_packet_len, bool p_spawn);
	Error _send_default_spawn_despawn(int p_peer_id, const ResourceUID::ID &p_scene_id, Object *p_obj, const NodePath &p_path, bool p_spawn);

public:
	void clear();

	Error spawn_config(const ResourceUID::ID &p_id, ReplicationMode p_mode, const TypedArray<StringName> &p_props = TypedArray<StringName>(), const Callable &p_on_send = Callable(), const Callable &p_on_recv = Callable());
	Error spawn(ResourceUID::ID p_scene_id, Object *p_obj, int p_peer = 0);
	Error despawn(ResourceUID::ID p_scene_id, Object *p_obj, int p_peer = 0);

	Error send_despawn(int p_peer_id, const ResourceUID::ID &p_scene_id, const Variant &p_data = Variant(), const NodePath &p_path = NodePath());
	Error send_spawn(int p_peer_id, const ResourceUID::ID &p_scene_id, const Variant &p_data = Variant(), const NodePath &p_path = NodePath());
	PackedByteArray encode_state(const ResourceUID::ID &p_scene_id, const Object *p_node);
	Error decode_state(const ResourceUID::ID &p_scene_id, Object *p_node, PackedByteArray p_data);

	// Used by MultiplayerAPI
	void spawn_all(int p_peer);
	void process_spawn_despawn(int p_from, const uint8_t *p_packet, int p_packet_len, bool p_spawn);
	void scene_enter_exit_notify(const String &p_scene, Node *p_node, bool p_enter);

	MultiplayerReplicator(MultiplayerAPI *p_multiplayer) {
		multiplayer = p_multiplayer;
	}
};

VARIANT_ENUM_CAST(MultiplayerReplicator::ReplicationMode);

#endif // MULTIPLAYER_REPLICATOR_H
