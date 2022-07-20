/*************************************************************************/
/*  multiplayer_api.h                                                    */
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

#ifndef MULTIPLAYER_API_H
#define MULTIPLAYER_API_H

#include "core/multiplayer/multiplayer.h"
#include "core/multiplayer/multiplayer_peer.h"
#include "core/object/ref_counted.h"

class MultiplayerAPI;

class MultiplayerReplicationInterface : public RefCounted {
	GDCLASS(MultiplayerReplicationInterface, RefCounted);

public:
	virtual void on_peer_change(int p_id, bool p_connected) {}
	virtual void on_reset() {}
	virtual Error on_spawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) { return ERR_UNAVAILABLE; }
	virtual Error on_despawn_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) { return ERR_UNAVAILABLE; }
	virtual Error on_sync_receive(int p_from, const uint8_t *p_buffer, int p_buffer_len) { return ERR_UNAVAILABLE; }
	virtual Error on_spawn(Object *p_obj, Variant p_config) { return ERR_UNAVAILABLE; }
	virtual Error on_despawn(Object *p_obj, Variant p_config) { return ERR_UNAVAILABLE; }
	virtual Error on_replication_start(Object *p_obj, Variant p_config) { return ERR_UNAVAILABLE; }
	virtual Error on_replication_stop(Object *p_obj, Variant p_config) { return ERR_UNAVAILABLE; }
	virtual void on_network_process() {}

	MultiplayerReplicationInterface() {}
};

class MultiplayerRPCInterface : public RefCounted {
	GDCLASS(MultiplayerRPCInterface, RefCounted);

public:
	// Called by Node.rpc
	virtual void rpcp(Object *p_obj, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount) {}
	virtual void process_rpc(int p_from, const uint8_t *p_packet, int p_packet_len) {}
	virtual String get_rpc_md5(const Object *p_obj) const { return String(); }

	MultiplayerRPCInterface() {}
};

class MultiplayerCacheInterface : public RefCounted {
	GDCLASS(MultiplayerCacheInterface, RefCounted);

public:
	virtual void clear() {}
	virtual void on_peer_change(int p_id, bool p_connected) {}
	virtual void process_simplify_path(int p_from, const uint8_t *p_packet, int p_packet_len) {}
	virtual void process_confirm_path(int p_from, const uint8_t *p_packet, int p_packet_len) {}

	// Returns true if all peers have cached path.
	virtual bool send_object_cache(Object *p_obj, NodePath p_path, int p_target, int &p_id) { return false; }
	virtual Object *get_cached_object(int p_from, uint32_t p_cache_id) { return nullptr; }
	virtual bool is_cache_confirmed(NodePath p_path, int p_peer) { return false; }

	MultiplayerCacheInterface() {}
};

class MultiplayerAPI : public RefCounted {
	GDCLASS(MultiplayerAPI, RefCounted);

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

	Ref<MultiplayerCacheInterface> cache;
	Ref<MultiplayerReplicationInterface> replicator;
	Ref<MultiplayerRPCInterface> rpc;

protected:
	static void _bind_methods();

	void _process_packet(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_raw(int p_from, const uint8_t *p_packet, int p_packet_len);

public:
	static MultiplayerReplicationInterface *(*create_default_replication_interface)(MultiplayerAPI *p_multiplayer);
	static MultiplayerRPCInterface *(*create_default_rpc_interface)(MultiplayerAPI *p_multiplayer);
	static MultiplayerCacheInterface *(*create_default_cache_interface)(MultiplayerAPI *p_multiplayer);

	static Error encode_and_compress_variant(const Variant &p_variant, uint8_t *p_buffer, int &r_len, bool p_allow_object_decoding);
	static Error decode_and_decompress_variant(Variant &r_variant, const uint8_t *p_buffer, int p_len, int *r_len, bool p_allow_object_decoding);
	static Error encode_and_compress_variants(const Variant **p_variants, int p_count, uint8_t *p_buffer, int &r_len, bool *r_raw = nullptr, bool p_allow_object_decoding = false);
	static Error decode_and_decompress_variants(Vector<Variant> &r_variants, const uint8_t *p_buffer, int p_len, int &r_len, bool p_raw = false, bool p_allow_object_decoding = false);

	void poll();
	void clear();
	void set_root_path(const NodePath &p_path);
	NodePath get_root_path() const;
	void set_multiplayer_peer(const Ref<MultiplayerPeer> &p_peer);
	Ref<MultiplayerPeer> get_multiplayer_peer() const;

	Error send_bytes(Vector<uint8_t> p_data, int p_to = MultiplayerPeer::TARGET_PEER_BROADCAST, Multiplayer::TransferMode p_mode = Multiplayer::TRANSFER_MODE_RELIABLE, int p_channel = 0);

	// RPC API
	void rpcp(Object *p_obj, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount);
	String get_rpc_md5(const Object *p_obj) const;
	// Replication API
	Error spawn(Object *p_object, Variant p_config);
	Error despawn(Object *p_object, Variant p_config);
	Error replication_start(Object *p_object, Variant p_config);
	Error replication_stop(Object *p_object, Variant p_config);
	// Cache API
	bool send_object_cache(Object *p_obj, NodePath p_path, int p_target, int &p_id);
	Object *get_cached_object(int p_from, uint32_t p_cache_id);
	bool is_cache_confirmed(NodePath p_path, int p_peer);

	void _add_peer(int p_id);
	void _del_peer(int p_id);
	void _connected_to_server();
	void _connection_failed();
	void _server_disconnected();

	bool has_multiplayer_peer() const { return multiplayer_peer.is_valid(); }
	Vector<int> get_peer_ids() const;
	const HashSet<int> get_connected_peers() const { return connected_peers; }
	int get_remote_sender_id() const { return remote_sender_override ? remote_sender_override : remote_sender_id; }
	void set_remote_sender_override(int p_id) { remote_sender_override = p_id; }
	int get_unique_id() const;
	bool is_server() const;
	void set_refuse_new_connections(bool p_refuse);
	bool is_refusing_new_connections() const;

	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;

#ifdef DEBUG_ENABLED
	void profile_bandwidth(const String &p_inout, int p_size);
#endif

	MultiplayerAPI();
	~MultiplayerAPI();
};

#endif // MULTIPLAYER_API_H
