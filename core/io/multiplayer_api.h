#ifndef MULTIPLAYER_PROTOCOL_H
#define MULTIPLAYER_PROTOCOL_H

#include "core/io/networked_multiplayer_peer.h"
#include "core/reference.h"

class MultiplayerAPI : public Reference {

	GDCLASS(MultiplayerAPI, Reference);

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

	Ref<NetworkedMultiplayerPeer> network_peer;
	int rpc_sender_id;
	Set<int> connected_peers;
	HashMap<NodePath, PathSentCache> path_send_cache;
	Map<int, PathGetCache> path_get_cache;
	int last_send_cache_id;
	Vector<uint8_t> packet_cache;
	Node *root_node;

protected:
	static void _bind_methods();

	void _process_packet(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_simplify_path(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_confirm_path(int p_from, const uint8_t *p_packet, int p_packet_len);
	Node *_process_get_node(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_rpc(Node *p_node, const StringName &p_name, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset);
	void _process_rset(Node *p_node, const StringName &p_name, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset);

	void _send_rpc(Node *p_from, int p_to, bool p_unreliable, bool p_set, const StringName &p_name, const Variant **p_arg, int p_argcount);
	bool _send_confirm_path(NodePath p_path, PathSentCache *psc, int p_from);

public:
	enum NetworkCommands {
		NETWORK_COMMAND_REMOTE_CALL,
		NETWORK_COMMAND_REMOTE_SET,
		NETWORK_COMMAND_SIMPLIFY_PATH,
		NETWORK_COMMAND_CONFIRM_PATH,
	};

	void poll();
	void clear();
	void set_root_node(Node *p_node);
	void set_network_peer(const Ref<NetworkedMultiplayerPeer> &p_peer);
	Ref<NetworkedMultiplayerPeer> get_network_peer() const;

	// Called by Node.rpc
	void rpcp(Node *p_node, int p_peer_id, bool p_unreliable, const StringName &p_method, const Variant **p_arg, int p_argcount);
	// Called by Node.rset
	void rsetp(Node *p_node, int p_peer_id, bool p_unreliable, const StringName &p_property, const Variant &p_value);

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

	MultiplayerAPI();
	~MultiplayerAPI();
};

#endif // MULTIPLAYER_PROTOCOL_H
