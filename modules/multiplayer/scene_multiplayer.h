/*************************************************************************/
/*  scene_multiplayer.h                                                  */
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

#ifndef SCENE_MULTIPLAYER_H
#define SCENE_MULTIPLAYER_H

#include "scene/main/multiplayer_api.h"

#include "scene_cache_interface.h"
#include "scene_replication_interface.h"
#include "scene_rpc_interface.h"

class SceneMultiplayer : public MultiplayerAPI {
	GDCLASS(SceneMultiplayer, MultiplayerAPI);

public:
	enum NetworkCommands {
		NETWORK_COMMAND_REMOTE_CALL = 0,
		NETWORK_COMMAND_SIMPLIFY_PATH,
		NETWORK_COMMAND_CONFIRM_PATH,
		NETWORK_COMMAND_RAW,
		NETWORK_COMMAND_SPAWN,
		NETWORK_COMMAND_DESPAWN,
		NETWORK_COMMAND_SYNC,
	};

	// For each command, the 4 MSB can contain custom flags, as defined by subsystems.
	enum {
		CMD_FLAG_0_SHIFT = 4,
		CMD_FLAG_1_SHIFT = 5,
		CMD_FLAG_2_SHIFT = 6,
		CMD_FLAG_3_SHIFT = 7,
	};

	// This is the mask that will be used to extract the command.
	enum {
		CMD_MASK = 7, // 0x7 -> 0b00001111
	};

private:
	Ref<MultiplayerPeer> multiplayer_peer;
	HashSet<int> connected_peers;
	int remote_sender_id = 0;
	int remote_sender_override = 0;

	Vector<uint8_t> packet_cache;

	NodePath root_path;
	bool allow_object_decoding = false;

	Ref<SceneCacheInterface> cache;
	Ref<SceneReplicationInterface> replicator;
	Ref<SceneRPCInterface> rpc;

protected:
	static void _bind_methods();

	void _process_packet(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_raw(int p_from, const uint8_t *p_packet, int p_packet_len);

	void _add_peer(int p_id);
	void _del_peer(int p_id);
	void _connected_to_server();
	void _connection_failed();
	void _server_disconnected();

public:
	virtual void set_multiplayer_peer(const Ref<MultiplayerPeer> &p_peer) override;
	virtual Ref<MultiplayerPeer> get_multiplayer_peer() override;

	virtual Error poll() override;
	virtual int get_unique_id() override;
	virtual Vector<int> get_peer_ids() override;
	virtual int get_remote_sender_id() override { return remote_sender_override ? remote_sender_override : remote_sender_id; }

	virtual Error rpcp(Object *p_obj, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount) override;

	virtual Error object_configuration_add(Object *p_obj, Variant p_config) override;
	virtual Error object_configuration_remove(Object *p_obj, Variant p_config) override;

	void clear();

	// Usually from object_configuration_add/remove
	void set_root_path(const NodePath &p_path);
	NodePath get_root_path() const;

	Error send_bytes(Vector<uint8_t> p_data, int p_to = MultiplayerPeer::TARGET_PEER_BROADCAST, MultiplayerPeer::TransferMode p_mode = MultiplayerPeer::TRANSFER_MODE_RELIABLE, int p_channel = 0);
	String get_rpc_md5(const Object *p_obj);

	const HashSet<int> get_connected_peers() const { return connected_peers; }

	void set_remote_sender_override(int p_id) { remote_sender_override = p_id; }
	void set_refuse_new_connections(bool p_refuse);
	bool is_refusing_new_connections() const;

	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;

	Ref<SceneCacheInterface> get_path_cache() { return cache; }

#ifdef DEBUG_ENABLED
	void profile_bandwidth(const String &p_inout, int p_size);
#endif

	SceneMultiplayer();
	~SceneMultiplayer();
};

#endif // SCENE_MULTIPLAYER_H
