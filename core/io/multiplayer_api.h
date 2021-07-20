/*************************************************************************/
/*  multiplayer_api.h                                                    */
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

#ifndef MULTIPLAYER_API_H
#define MULTIPLAYER_API_H

#include "core/io/multiplayer_peer.h"
#include "core/object/ref_counted.h"

class MultiplayerAPI : public RefCounted {
	GDCLASS(MultiplayerAPI, RefCounted);

public:
	enum RPCMode {
		RPC_MODE_DISABLED, // No rpc for this method, calls to this will be blocked (default)
		RPC_MODE_REMOTE, // Using rpc() on it will call method in all remote peers
		RPC_MODE_MASTER, // Using rpc() on it will call method on wherever the master is, be it local or remote
		RPC_MODE_PUPPET, // Using rpc() on it will call method for all puppets
	};

	struct RPCConfig {
		StringName name;
		RPCMode rpc_mode = RPC_MODE_DISABLED;
		bool sync = false;
		MultiplayerPeer::TransferMode transfer_mode = MultiplayerPeer::TRANSFER_MODE_RELIABLE;
		int channel = 0;

		bool operator==(RPCConfig const &p_other) const {
			return name == p_other.name;
		}
	};

	struct SortRPCConfig {
		StringName::AlphCompare compare;
		bool operator()(const RPCConfig &p_a, const RPCConfig &p_b) const {
			return compare(p_a.name, p_b.name);
		}
	};

private:
	//path sent caches
	struct PathSentCache {
		Map<int, bool> confirmed_peers;
		int id;
	};

	//path get caches
	struct PathGetCache {
		struct NodeInfo {
			NodePath path;
			ObjectID instance;
		};

		Map<int, NodeInfo> nodes;
	};

	Ref<MultiplayerPeer> network_peer;
	int rpc_sender_id = 0;
	Set<int> connected_peers;
	HashMap<NodePath, PathSentCache> path_send_cache;
	Map<int, PathGetCache> path_get_cache;
	int last_send_cache_id;
	Vector<uint8_t> packet_cache;
	Node *root_node = nullptr;
	bool allow_object_decoding = false;

protected:
	static void _bind_methods();

	void _process_packet(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_simplify_path(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_confirm_path(int p_from, const uint8_t *p_packet, int p_packet_len);
	Node *_process_get_node(int p_from, const uint8_t *p_packet, uint32_t p_node_target, int p_packet_len);
	void _process_rpc(Node *p_node, const uint16_t p_rpc_method_id, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset);
	void _process_raw(int p_from, const uint8_t *p_packet, int p_packet_len);

	void _send_rpc(Node *p_from, int p_to, uint16_t p_rpc_id, const RPCConfig &p_config, const StringName &p_name, const Variant **p_arg, int p_argcount);
	bool _send_confirm_path(Node *p_node, NodePath p_path, PathSentCache *psc, int p_target);

	Error _encode_and_compress_variant(const Variant &p_variant, uint8_t *p_buffer, int &r_len);
	Error _decode_and_decompress_variant(Variant &r_variant, const uint8_t *p_buffer, int p_len, int *r_len);

public:
	enum NetworkCommands {
		NETWORK_COMMAND_REMOTE_CALL = 0,
		NETWORK_COMMAND_SIMPLIFY_PATH,
		NETWORK_COMMAND_CONFIRM_PATH,
		NETWORK_COMMAND_RAW,
	};

	enum NetworkNodeIdCompression {
		NETWORK_NODE_ID_COMPRESSION_8 = 0,
		NETWORK_NODE_ID_COMPRESSION_16,
		NETWORK_NODE_ID_COMPRESSION_32,
	};

	enum NetworkNameIdCompression {
		NETWORK_NAME_ID_COMPRESSION_8 = 0,
		NETWORK_NAME_ID_COMPRESSION_16,
	};

	void poll();
	void clear();
	void set_root_node(Node *p_node);
	Node *get_root_node();
	void set_network_peer(const Ref<MultiplayerPeer> &p_peer);
	Ref<MultiplayerPeer> get_network_peer() const;
	Error send_bytes(Vector<uint8_t> p_data, int p_to = MultiplayerPeer::TARGET_PEER_BROADCAST, MultiplayerPeer::TransferMode p_mode = MultiplayerPeer::TRANSFER_MODE_RELIABLE);

	// Called by Node.rpc
	void rpcp(Node *p_node, int p_peer_id, bool p_unreliable, const StringName &p_method, const Variant **p_arg, int p_argcount);

	void _add_peer(int p_id);
	void _del_peer(int p_id);
	void _connected_to_server();
	void _connection_failed();
	void _server_disconnected();

	bool has_network_peer() const { return network_peer.is_valid(); }
	Vector<int> get_network_connected_peers() const;
	int get_rpc_sender_id() const { return rpc_sender_id; }
	int get_network_unique_id() const;
	bool is_network_server() const;
	void set_refuse_new_network_connections(bool p_refuse);
	bool is_refusing_new_network_connections() const;

	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;

	MultiplayerAPI();
	~MultiplayerAPI();
};

VARIANT_ENUM_CAST(MultiplayerAPI::RPCMode);

#endif // MULTIPLAYER_API_H
