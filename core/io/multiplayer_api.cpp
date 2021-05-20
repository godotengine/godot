/*************************************************************************/
/*  multiplayer_api.cpp                                                  */
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

#include "multiplayer_api.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"
#include "scene/main/node.h"

#include <stdint.h>

#define NODE_ID_COMPRESSION_SHIFT 3
#define NAME_ID_COMPRESSION_SHIFT 5
#define BYTE_ONLY_OR_NO_ARGS_SHIFT 6

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#endif

_FORCE_INLINE_ bool _should_call_local(MultiplayerAPI::RPCMode mode, bool is_master, bool &r_skip_rpc) {
	switch (mode) {
		case MultiplayerAPI::RPC_MODE_DISABLED: {
			// Do nothing.
		} break;
		case MultiplayerAPI::RPC_MODE_REMOTE: {
			// Do nothing. Remote cannot produce a local call.
		} break;
		case MultiplayerAPI::RPC_MODE_MASTERSYNC: {
			if (is_master) {
				r_skip_rpc = true; // I am the master, so skip remote call.
			}
			[[fallthrough]];
		}
		case MultiplayerAPI::RPC_MODE_REMOTESYNC:
		case MultiplayerAPI::RPC_MODE_PUPPETSYNC: {
			// Call it, sync always results in a local call.
			return true;
		} break;
		case MultiplayerAPI::RPC_MODE_MASTER: {
			if (is_master) {
				r_skip_rpc = true; // I am the master, so skip remote call.
			}
			return is_master;
		} break;
		case MultiplayerAPI::RPC_MODE_PUPPET: {
			return !is_master;
		} break;
	}
	return false;
}

_FORCE_INLINE_ bool _can_call_mode(Node *p_node, MultiplayerAPI::RPCMode mode, int p_remote_id) {
	switch (mode) {
		case MultiplayerAPI::RPC_MODE_DISABLED: {
			return false;
		} break;
		case MultiplayerAPI::RPC_MODE_REMOTE:
		case MultiplayerAPI::RPC_MODE_REMOTESYNC: {
			return true;
		} break;
		case MultiplayerAPI::RPC_MODE_MASTERSYNC:
		case MultiplayerAPI::RPC_MODE_MASTER: {
			return p_node->is_network_master();
		} break;
		case MultiplayerAPI::RPC_MODE_PUPPETSYNC:
		case MultiplayerAPI::RPC_MODE_PUPPET: {
			return !p_node->is_network_master() && p_remote_id == p_node->get_network_master();
		} break;
	}

	return false;
}

void MultiplayerAPI::poll() {
	if (!network_peer.is_valid() || network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_DISCONNECTED) {
		return;
	}

	network_peer->poll();

	if (!network_peer.is_valid()) { // It's possible that polling might have resulted in a disconnection, so check here.
		return;
	}

	while (network_peer->get_available_packet_count()) {
		int sender = network_peer->get_packet_peer();
		const uint8_t *packet;
		int len;

		Error err = network_peer->get_packet(&packet, len);
		if (err != OK) {
			ERR_PRINT("Error getting packet!");
			break; // Something is wrong!
		}

		rpc_sender_id = sender;
		_process_packet(sender, packet, len);
		rpc_sender_id = 0;

		if (!network_peer.is_valid()) {
			break; // It's also possible that a packet or RPC caused a disconnection, so also check here.
		}
	}
}

void MultiplayerAPI::clear() {
	connected_peers.clear();
	path_get_cache.clear();
	path_send_cache.clear();
	packet_cache.clear();
	last_send_cache_id = 1;
}

void MultiplayerAPI::set_root_node(Node *p_node) {
	root_node = p_node;
}

Node *MultiplayerAPI::get_root_node() {
	return root_node;
}

void MultiplayerAPI::set_network_peer(const Ref<NetworkedMultiplayerPeer> &p_peer) {
	if (p_peer == network_peer) {
		return; // Nothing to do
	}

	ERR_FAIL_COND_MSG(p_peer.is_valid() && p_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_DISCONNECTED,
			"Supplied NetworkedMultiplayerPeer must be connecting or connected.");

	if (network_peer.is_valid()) {
		network_peer->disconnect("peer_connected", callable_mp(this, &MultiplayerAPI::_add_peer));
		network_peer->disconnect("peer_disconnected", callable_mp(this, &MultiplayerAPI::_del_peer));
		network_peer->disconnect("connection_succeeded", callable_mp(this, &MultiplayerAPI::_connected_to_server));
		network_peer->disconnect("connection_failed", callable_mp(this, &MultiplayerAPI::_connection_failed));
		network_peer->disconnect("server_disconnected", callable_mp(this, &MultiplayerAPI::_server_disconnected));
		clear();
	}

	network_peer = p_peer;

	if (network_peer.is_valid()) {
		network_peer->connect("peer_connected", callable_mp(this, &MultiplayerAPI::_add_peer));
		network_peer->connect("peer_disconnected", callable_mp(this, &MultiplayerAPI::_del_peer));
		network_peer->connect("connection_succeeded", callable_mp(this, &MultiplayerAPI::_connected_to_server));
		network_peer->connect("connection_failed", callable_mp(this, &MultiplayerAPI::_connection_failed));
		network_peer->connect("server_disconnected", callable_mp(this, &MultiplayerAPI::_server_disconnected));
	}
}

Ref<NetworkedMultiplayerPeer> MultiplayerAPI::get_network_peer() const {
	return network_peer;
}

#ifdef DEBUG_ENABLED
void _profile_node_data(const String &p_what, ObjectID p_id) {
	if (EngineDebugger::is_profiling("multiplayer")) {
		Array values;
		values.push_back("node");
		values.push_back(p_id);
		values.push_back(p_what);
		EngineDebugger::profiler_add_frame_data("multiplayer", values);
	}
}

void _profile_bandwidth_data(const String &p_inout, int p_size) {
	if (EngineDebugger::is_profiling("multiplayer")) {
		Array values;
		values.push_back("bandwidth");
		values.push_back(p_inout);
		values.push_back(OS::get_singleton()->get_ticks_msec());
		values.push_back(p_size);
		EngineDebugger::profiler_add_frame_data("multiplayer", values);
	}
}
#endif

// Returns the packet size stripping the node path added when the node is not yet cached.
int get_packet_len(uint32_t p_node_target, int p_packet_len) {
	if (p_node_target & 0x80000000) {
		int ofs = p_node_target & 0x7FFFFFFF;
		return p_packet_len - (p_packet_len - ofs);
	} else {
		return p_packet_len;
	}
}

void MultiplayerAPI::_process_packet(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(root_node == nullptr, "Multiplayer root node was not initialized. If you are using custom multiplayer, remember to set the root node via MultiplayerAPI.set_root_node before using it.");
	ERR_FAIL_COND_MSG(p_packet_len < 1, "Invalid packet received. Size too small.");

#ifdef DEBUG_ENABLED
	_profile_bandwidth_data("in", p_packet_len);
#endif

	// Extract the `packet_type` from the LSB three bits:
	uint8_t packet_type = p_packet[0] & 7;

	switch (packet_type) {
		case NETWORK_COMMAND_SIMPLIFY_PATH: {
			_process_simplify_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_CONFIRM_PATH: {
			_process_confirm_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_REMOTE_CALL:
		case NETWORK_COMMAND_REMOTE_SET: {
			// Extract packet meta
			int packet_min_size = 1;
			int name_id_offset = 1;
			ERR_FAIL_COND_MSG(p_packet_len < packet_min_size, "Invalid packet received. Size too small.");
			// Compute the meta size, which depends on the compression level.
			int node_id_compression = (p_packet[0] & 24) >> NODE_ID_COMPRESSION_SHIFT;
			int name_id_compression = (p_packet[0] & 32) >> NAME_ID_COMPRESSION_SHIFT;

			switch (node_id_compression) {
				case NETWORK_NODE_ID_COMPRESSION_8:
					packet_min_size += 1;
					name_id_offset += 1;
					break;
				case NETWORK_NODE_ID_COMPRESSION_16:
					packet_min_size += 2;
					name_id_offset += 2;
					break;
				case NETWORK_NODE_ID_COMPRESSION_32:
					packet_min_size += 4;
					name_id_offset += 4;
					break;
				default:
					ERR_FAIL_MSG("Was not possible to extract the node id compression mode.");
			}
			switch (name_id_compression) {
				case NETWORK_NAME_ID_COMPRESSION_8:
					packet_min_size += 1;
					break;
				case NETWORK_NAME_ID_COMPRESSION_16:
					packet_min_size += 2;
					break;
				default:
					ERR_FAIL_MSG("Was not possible to extract the name id compression mode.");
			}
			ERR_FAIL_COND_MSG(p_packet_len < packet_min_size, "Invalid packet received. Size too small.");

			uint32_t node_target = 0;
			switch (node_id_compression) {
				case NETWORK_NODE_ID_COMPRESSION_8:
					node_target = p_packet[1];
					break;
				case NETWORK_NODE_ID_COMPRESSION_16:
					node_target = decode_uint16(p_packet + 1);
					break;
				case NETWORK_NODE_ID_COMPRESSION_32:
					node_target = decode_uint32(p_packet + 1);
					break;
				default:
					// Unreachable, checked before.
					CRASH_NOW();
			}

			Node *node = _process_get_node(p_from, p_packet, node_target, p_packet_len);
			ERR_FAIL_COND_MSG(node == nullptr, "Invalid packet received. Requested node was not found.");

			uint16_t name_id = 0;
			switch (name_id_compression) {
				case NETWORK_NAME_ID_COMPRESSION_8:
					name_id = p_packet[name_id_offset];
					break;
				case NETWORK_NAME_ID_COMPRESSION_16:
					name_id = decode_uint16(p_packet + name_id_offset);
					break;
				default:
					// Unreachable, checked before.
					CRASH_NOW();
			}

			const int packet_len = get_packet_len(node_target, p_packet_len);
			if (packet_type == NETWORK_COMMAND_REMOTE_CALL) {
				_process_rpc(node, name_id, p_from, p_packet, packet_len, packet_min_size);

			} else {
				_process_rset(node, name_id, p_from, p_packet, packet_len, packet_min_size);
			}

		} break;

		case NETWORK_COMMAND_RAW: {
			_process_raw(p_from, p_packet, p_packet_len);
		} break;
	}
}

Node *MultiplayerAPI::_process_get_node(int p_from, const uint8_t *p_packet, uint32_t p_node_target, int p_packet_len) {
	Node *node = nullptr;

	if (p_node_target & 0x80000000) {
		// Use full path (not cached yet).

		int ofs = p_node_target & 0x7FFFFFFF;

		ERR_FAIL_COND_V_MSG(ofs >= p_packet_len, nullptr, "Invalid packet received. Size smaller than declared.");

		String paths;
		paths.parse_utf8((const char *)&p_packet[ofs], p_packet_len - ofs);

		NodePath np = paths;

		node = root_node->get_node(np);

		if (!node) {
			ERR_PRINT("Failed to get path from RPC: " + String(np) + ".");
		}
	} else {
		// Use cached path.
		int id = p_node_target;

		Map<int, PathGetCache>::Element *E = path_get_cache.find(p_from);
		ERR_FAIL_COND_V_MSG(!E, nullptr, "Invalid packet received. Requests invalid peer cache.");

		Map<int, PathGetCache::NodeInfo>::Element *F = E->get().nodes.find(id);
		ERR_FAIL_COND_V_MSG(!F, nullptr, "Invalid packet received. Unabled to find requested cached node.");

		PathGetCache::NodeInfo *ni = &F->get();
		// Do proper caching later.

		node = root_node->get_node(ni->path);
		if (!node) {
			ERR_PRINT("Failed to get cached path from RPC: " + String(ni->path) + ".");
		}
	}
	return node;
}

void MultiplayerAPI::_process_rpc(Node *p_node, const uint16_t p_rpc_method_id, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset) {
	ERR_FAIL_COND_MSG(p_offset > p_packet_len, "Invalid packet received. Size too small.");

	// Check that remote can call the RPC on this node.
	StringName name = p_node->get_node_rpc_method(p_rpc_method_id);
	RPCMode rpc_mode = p_node->get_node_rpc_mode_by_id(p_rpc_method_id);
	if (name == StringName() && p_node->get_script_instance()) {
		name = p_node->get_script_instance()->get_rpc_method(p_rpc_method_id);
		rpc_mode = p_node->get_script_instance()->get_rpc_mode_by_id(p_rpc_method_id);
	}
	ERR_FAIL_COND(name == StringName());

	bool can_call = _can_call_mode(p_node, rpc_mode, p_from);
	ERR_FAIL_COND_MSG(!can_call, "RPC '" + String(name) + "' is not allowed on node " + p_node->get_path() + " from: " + itos(p_from) + ". Mode is " + itos((int)rpc_mode) + ", master is " + itos(p_node->get_network_master()) + ".");

	int argc = 0;
	bool byte_only = false;

	const bool byte_only_or_no_args = ((p_packet[0] & 64) >> BYTE_ONLY_OR_NO_ARGS_SHIFT) == 1;
	if (byte_only_or_no_args) {
		if (p_offset < p_packet_len) {
			// This packet contains only bytes.
			argc = 1;
			byte_only = true;
		} else {
			// This rpc calls a method without parameters.
		}
	} else {
		// Normal variant, takes the argument count from the packet.
		ERR_FAIL_COND_MSG(p_offset >= p_packet_len, "Invalid packet received. Size too small.");
		argc = p_packet[p_offset];
		p_offset += 1;
	}

	Vector<Variant> args;
	Vector<const Variant *> argp;
	args.resize(argc);
	argp.resize(argc);

#ifdef DEBUG_ENABLED
	_profile_node_data("in_rpc", p_node->get_instance_id());
#endif

	if (byte_only) {
		Vector<uint8_t> pure_data;
		const int len = p_packet_len - p_offset;
		pure_data.resize(len);
		memcpy(pure_data.ptrw(), &p_packet[p_offset], len);
		args.write[0] = pure_data;
		argp.write[0] = &args[0];
		p_offset += len;
	} else {
		for (int i = 0; i < argc; i++) {
			ERR_FAIL_COND_MSG(p_offset >= p_packet_len, "Invalid packet received. Size too small.");

			int vlen;
			Error err = _decode_and_decompress_variant(args.write[i], &p_packet[p_offset], p_packet_len - p_offset, &vlen);
			ERR_FAIL_COND_MSG(err != OK, "Invalid packet received. Unable to decode RPC argument.");

			argp.write[i] = &args[i];
			p_offset += vlen;
		}
	}

	Callable::CallError ce;

	p_node->call(name, (const Variant **)argp.ptr(), argc, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		String error = Variant::get_call_error_text(p_node, name, (const Variant **)argp.ptr(), argc, ce);
		error = "RPC - " + error;
		ERR_PRINT(error);
	}
}

void MultiplayerAPI::_process_rset(Node *p_node, const uint16_t p_rpc_property_id, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset) {
	ERR_FAIL_COND_MSG(p_offset >= p_packet_len, "Invalid packet received. Size too small.");

	// Check that remote can call the RSET on this node.
	StringName name = p_node->get_node_rset_property(p_rpc_property_id);
	RPCMode rset_mode = p_node->get_node_rset_mode_by_id(p_rpc_property_id);
	if (name == StringName() && p_node->get_script_instance()) {
		name = p_node->get_script_instance()->get_rset_property(p_rpc_property_id);
		rset_mode = p_node->get_script_instance()->get_rset_mode_by_id(p_rpc_property_id);
	}
	ERR_FAIL_COND(name == StringName());

	bool can_call = _can_call_mode(p_node, rset_mode, p_from);
	ERR_FAIL_COND_MSG(!can_call, "RSET '" + String(name) + "' is not allowed on node " + p_node->get_path() + " from: " + itos(p_from) + ". Mode is " + itos((int)rset_mode) + ", master is " + itos(p_node->get_network_master()) + ".");

#ifdef DEBUG_ENABLED
	_profile_node_data("in_rset", p_node->get_instance_id());
#endif

	Variant value;
	Error err = _decode_and_decompress_variant(value, &p_packet[p_offset], p_packet_len - p_offset, nullptr);

	ERR_FAIL_COND_MSG(err != OK, "Invalid packet received. Unable to decode RSET value.");

	bool valid;

	p_node->set(name, value, &valid);
	if (!valid) {
		String error = "Error setting remote property '" + String(name) + "', not found in object of type " + p_node->get_class() + ".";
		ERR_PRINT(error);
	}
}

void MultiplayerAPI::_process_simplify_path(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(p_packet_len < 38, "Invalid packet received. Size too small.");
	int ofs = 1;

	String methods_md5;
	methods_md5.parse_utf8((const char *)(p_packet + ofs), 32);
	ofs += 33;

	int id = decode_uint32(&p_packet[ofs]);
	ofs += 4;

	String paths;
	paths.parse_utf8((const char *)(p_packet + ofs), p_packet_len - ofs);

	NodePath path = paths;

	if (!path_get_cache.has(p_from)) {
		path_get_cache[p_from] = PathGetCache();
	}

	Node *node = root_node->get_node(path);
	ERR_FAIL_COND(node == nullptr);
	const bool valid_rpc_checksum = node->get_rpc_md5() == methods_md5;
	if (valid_rpc_checksum == false) {
		ERR_PRINT("The rpc node checksum failed. Make sure to have the same methods on both nodes. Node path: " + path);
	}

	PathGetCache::NodeInfo ni;
	ni.path = path;

	path_get_cache[p_from].nodes[id] = ni;

	// Encode path to send ack.
	CharString pname = String(path).utf8();
	int len = encode_cstring(pname.get_data(), nullptr);

	Vector<uint8_t> packet;

	packet.resize(1 + 1 + len);
	packet.write[0] = NETWORK_COMMAND_CONFIRM_PATH;
	packet.write[1] = valid_rpc_checksum;
	encode_cstring(pname.get_data(), &packet.write[2]);

	network_peer->set_transfer_mode(NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE);
	network_peer->set_target_peer(p_from);
	network_peer->put_packet(packet.ptr(), packet.size());
}

void MultiplayerAPI::_process_confirm_path(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(p_packet_len < 3, "Invalid packet received. Size too small.");

	const bool valid_rpc_checksum = p_packet[1];

	String paths;
	paths.parse_utf8((const char *)&p_packet[2], p_packet_len - 2);

	NodePath path = paths;

	if (valid_rpc_checksum == false) {
		ERR_PRINT("The rpc node checksum failed. Make sure to have the same methods on both nodes. Node path: " + path);
	}

	PathSentCache *psc = path_send_cache.getptr(path);
	ERR_FAIL_COND_MSG(!psc, "Invalid packet received. Tries to confirm a path which was not found in cache.");

	Map<int, bool>::Element *E = psc->confirmed_peers.find(p_from);
	ERR_FAIL_COND_MSG(!E, "Invalid packet received. Source peer was not found in cache for the given path.");
	E->get() = true;
}

bool MultiplayerAPI::_send_confirm_path(Node *p_node, NodePath p_path, PathSentCache *psc, int p_target) {
	bool has_all_peers = true;
	List<int> peers_to_add; // If one is missing, take note to add it.

	for (Set<int>::Element *E = connected_peers.front(); E; E = E->next()) {
		if (p_target < 0 && E->get() == -p_target) {
			continue; // Continue, excluded.
		}

		if (p_target > 0 && E->get() != p_target) {
			continue; // Continue, not for this peer.
		}

		Map<int, bool>::Element *F = psc->confirmed_peers.find(E->get());

		if (!F || !F->get()) {
			// Path was not cached, or was cached but is unconfirmed.
			if (!F) {
				// Not cached at all, take note.
				peers_to_add.push_back(E->get());
			}

			has_all_peers = false;
		}
	}

	if (peers_to_add.size() > 0) {
		// Those that need to be added, send a message for this.

		// Encode function name.
		const CharString path = String(p_path).utf8();
		const int path_len = encode_cstring(path.get_data(), nullptr);

		// Extract MD5 from rpc methods list.
		const String methods_md5 = p_node->get_rpc_md5();
		const int methods_md5_len = 33; // 32 + 1 for the `0` that is added by the encoder.

		Vector<uint8_t> packet;
		packet.resize(1 + 4 + path_len + methods_md5_len);
		int ofs = 0;

		packet.write[ofs] = NETWORK_COMMAND_SIMPLIFY_PATH;
		ofs += 1;

		ofs += encode_cstring(methods_md5.utf8().get_data(), &packet.write[ofs]);

		ofs += encode_uint32(psc->id, &packet.write[ofs]);

		ofs += encode_cstring(path.get_data(), &packet.write[ofs]);

		for (List<int>::Element *E = peers_to_add.front(); E; E = E->next()) {
			network_peer->set_target_peer(E->get()); // To all of you.
			network_peer->set_transfer_mode(NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE);
			network_peer->put_packet(packet.ptr(), packet.size());

			psc->confirmed_peers.insert(E->get(), false); // Insert into confirmed, but as false since it was not confirmed.
		}
	}

	return has_all_peers;
}

// The variant is compressed and encoded; The first byte contains all the meta
// information and the format is:
// - The first LSB 5 bits are used for the variant type.
// - The next two bits are used to store the encoding mode.
// - The most significant is used to store the boolean value.
#define VARIANT_META_TYPE_MASK 0x1F
#define VARIANT_META_EMODE_MASK 0x60
#define VARIANT_META_BOOL_MASK 0x80
#define ENCODE_8 0 << 5
#define ENCODE_16 1 << 5
#define ENCODE_32 2 << 5
#define ENCODE_64 3 << 5
Error MultiplayerAPI::_encode_and_compress_variant(const Variant &p_variant, uint8_t *r_buffer, int &r_len) {
	// Unreachable because `VARIANT_MAX` == 27 and `ENCODE_VARIANT_MASK` == 31
	CRASH_COND(p_variant.get_type() > VARIANT_META_TYPE_MASK);

	uint8_t *buf = r_buffer;
	r_len = 0;
	uint8_t encode_mode = 0;

	switch (p_variant.get_type()) {
		case Variant::BOOL: {
			if (buf) {
				// We still have 1 free bit in the meta, so let's use it.
				buf[0] = (p_variant.operator bool()) ? (1 << 7) : 0;
				buf[0] |= encode_mode | p_variant.get_type();
			}
			r_len += 1;
		} break;
		case Variant::INT: {
			if (buf) {
				// Reserve the first byte for the meta.
				buf += 1;
			}
			r_len += 1;
			int64_t val = p_variant;
			if (val <= (int64_t)INT8_MAX && val >= (int64_t)INT8_MIN) {
				// Use 8 bit
				encode_mode = ENCODE_8;
				if (buf) {
					buf[0] = val;
				}
				r_len += 1;
			} else if (val <= (int64_t)INT16_MAX && val >= (int64_t)INT16_MIN) {
				// Use 16 bit
				encode_mode = ENCODE_16;
				if (buf) {
					encode_uint16(val, buf);
				}
				r_len += 2;
			} else if (val <= (int64_t)INT32_MAX && val >= (int64_t)INT32_MIN) {
				// Use 32 bit
				encode_mode = ENCODE_32;
				if (buf) {
					encode_uint32(val, buf);
				}
				r_len += 4;
			} else {
				// Use 64 bit
				encode_mode = ENCODE_64;
				if (buf) {
					encode_uint64(val, buf);
				}
				r_len += 8;
			}
			// Store the meta
			if (buf) {
				buf -= 1;
				buf[0] = encode_mode | p_variant.get_type();
			}
		} break;
		default:
			// Any other case is not yet compressed.
			Error err = encode_variant(p_variant, r_buffer, r_len, allow_object_decoding);
			if (err != OK) {
				return err;
			}
			if (r_buffer) {
				// The first byte is not used by the marshalling, so store the type
				// so we know how to decompress and decode this variant.
				r_buffer[0] = p_variant.get_type();
			}
	}

	return OK;
}

Error MultiplayerAPI::_decode_and_decompress_variant(Variant &r_variant, const uint8_t *p_buffer, int p_len, int *r_len) {
	const uint8_t *buf = p_buffer;
	int len = p_len;

	ERR_FAIL_COND_V(len < 1, ERR_INVALID_DATA);
	uint8_t type = buf[0] & VARIANT_META_TYPE_MASK;
	uint8_t encode_mode = buf[0] & VARIANT_META_EMODE_MASK;

	ERR_FAIL_COND_V(type >= Variant::VARIANT_MAX, ERR_INVALID_DATA);

	switch (type) {
		case Variant::BOOL: {
			bool val = (buf[0] & VARIANT_META_BOOL_MASK) > 0;
			r_variant = val;
			if (r_len) {
				*r_len = 1;
			}
		} break;
		case Variant::INT: {
			buf += 1;
			len -= 1;
			if (r_len) {
				*r_len = 1;
			}
			if (encode_mode == ENCODE_8) {
				// 8 bits.
				ERR_FAIL_COND_V(len < 1, ERR_INVALID_DATA);
				int8_t val = buf[0];
				r_variant = val;
				if (r_len) {
					(*r_len) += 1;
				}
			} else if (encode_mode == ENCODE_16) {
				// 16 bits.
				ERR_FAIL_COND_V(len < 2, ERR_INVALID_DATA);
				int16_t val = decode_uint16(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += 2;
				}
			} else if (encode_mode == ENCODE_32) {
				// 32 bits.
				ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
				int32_t val = decode_uint32(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += 4;
				}
			} else {
				// 64 bits.
				ERR_FAIL_COND_V(len < 8, ERR_INVALID_DATA);
				int64_t val = decode_uint64(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += 8;
				}
			}
		} break;
		default:
			Error err = decode_variant(r_variant, p_buffer, p_len, r_len, allow_object_decoding);
			if (err != OK) {
				return err;
			}
	}

	return OK;
}

void MultiplayerAPI::_send_rpc(Node *p_from, int p_to, bool p_unreliable, bool p_set, const StringName &p_name, const Variant **p_arg, int p_argcount) {
	ERR_FAIL_COND_MSG(network_peer.is_null(), "Attempt to remote call/set when networking is not active in SceneTree.");

	ERR_FAIL_COND_MSG(network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_CONNECTING, "Attempt to remote call/set when networking is not connected yet in SceneTree.");

	ERR_FAIL_COND_MSG(network_peer->get_connection_status() == NetworkedMultiplayerPeer::CONNECTION_DISCONNECTED, "Attempt to remote call/set when networking is disconnected.");

	ERR_FAIL_COND_MSG(p_argcount > 255, "Too many arguments >255.");

	if (p_to != 0 && !connected_peers.has(ABS(p_to))) {
		ERR_FAIL_COND_MSG(p_to == network_peer->get_unique_id(), "Attempt to remote call/set yourself! unique ID: " + itos(network_peer->get_unique_id()) + ".");

		ERR_FAIL_MSG("Attempt to remote call unexisting ID: " + itos(p_to) + ".");
	}

	NodePath from_path = (root_node->get_path()).rel_path_to(p_from->get_path());
	ERR_FAIL_COND_MSG(from_path.is_empty(), "Unable to send RPC. Relative path is empty. THIS IS LIKELY A BUG IN THE ENGINE!");

	// See if the path is cached.
	PathSentCache *psc = path_send_cache.getptr(from_path);
	if (!psc) {
		// Path is not cached, create.
		path_send_cache[from_path] = PathSentCache();
		psc = path_send_cache.getptr(from_path);
		psc->id = last_send_cache_id++;
	}

	// See if all peers have cached path (if so, call can be fast).
	const bool has_all_peers = _send_confirm_path(p_from, from_path, psc, p_to);

	// Create base packet, lots of hardcode because it must be tight.

	int ofs = 0;

#define MAKE_ROOM(m_amount)             \
	if (packet_cache.size() < m_amount) \
		packet_cache.resize(m_amount);

	// Encode meta.
	// The meta is composed by a single byte that contains (starting from the least significant bit):
	// - `NetworkCommands` in the first three bits.
	// - `NetworkNodeIdCompression` in the next 2 bits.
	// - `NetworkNameIdCompression` in the next 1 bit.
	// - `byte_only_or_no_args` in the next 1 bit.
	// - So we still have the last bit free!
	uint8_t command_type = p_set ? NETWORK_COMMAND_REMOTE_SET : NETWORK_COMMAND_REMOTE_CALL;
	uint8_t node_id_compression = UINT8_MAX;
	uint8_t name_id_compression = UINT8_MAX;
	bool byte_only_or_no_args = false;

	MAKE_ROOM(1);
	// The meta is composed along the way, so just set 0 for now.
	packet_cache.write[0] = 0;
	ofs += 1;

	// Encode Node ID.
	if (has_all_peers) {
		// Compress the node ID only if all the target peers already know it.
		if (psc->id >= 0 && psc->id <= 255) {
			// We can encode the id in 1 byte
			node_id_compression = NETWORK_NODE_ID_COMPRESSION_8;
			MAKE_ROOM(ofs + 1);
			packet_cache.write[ofs] = static_cast<uint8_t>(psc->id);
			ofs += 1;
		} else if (psc->id >= 0 && psc->id <= 65535) {
			// We can encode the id in 2 bytes
			node_id_compression = NETWORK_NODE_ID_COMPRESSION_16;
			MAKE_ROOM(ofs + 2);
			encode_uint16(static_cast<uint16_t>(psc->id), &(packet_cache.write[ofs]));
			ofs += 2;
		} else {
			// Too big, let's use 4 bytes.
			node_id_compression = NETWORK_NODE_ID_COMPRESSION_32;
			MAKE_ROOM(ofs + 4);
			encode_uint32(psc->id, &(packet_cache.write[ofs]));
			ofs += 4;
		}
	} else {
		// The targets don't know the node yet, so we need to use 32 bits int.
		node_id_compression = NETWORK_NODE_ID_COMPRESSION_32;
		MAKE_ROOM(ofs + 4);
		encode_uint32(psc->id, &(packet_cache.write[ofs]));
		ofs += 4;
	}

	if (p_set) {
		// Take the rpc property ID
		uint16_t property_id = p_from->get_node_rset_property_id(p_name);
		if (property_id == UINT16_MAX && p_from->get_script_instance()) {
			property_id = p_from->get_script_instance()->get_rset_property_id(p_name);
		}
		ERR_FAIL_COND_MSG(property_id == UINT16_MAX, "Unable to take the `property_id` for the property:" + p_name + ". This can only happen if this property is not marked as `remote`.");

		if (property_id <= UINT8_MAX) {
			// The ID fits in 1 byte
			name_id_compression = NETWORK_NAME_ID_COMPRESSION_8;
			MAKE_ROOM(ofs + 1);
			packet_cache.write[ofs] = static_cast<uint8_t>(property_id);
			ofs += 1;
		} else {
			// The ID is larger, let's use 2 bytes
			name_id_compression = NETWORK_NAME_ID_COMPRESSION_16;
			MAKE_ROOM(ofs + 2);
			encode_uint16(property_id, &(packet_cache.write[ofs]));
			ofs += 2;
		}

		// Set argument.
		int len(0);
		Error err = _encode_and_compress_variant(*p_arg[0], nullptr, len);
		ERR_FAIL_COND_MSG(err != OK, "Unable to encode RSET value. THIS IS LIKELY A BUG IN THE ENGINE!");
		MAKE_ROOM(ofs + len);
		_encode_and_compress_variant(*p_arg[0], &(packet_cache.write[ofs]), len);
		ofs += len;

	} else {
		// Take the rpc method ID
		uint16_t method_id = p_from->get_node_rpc_method_id(p_name);
		if (method_id == UINT16_MAX && p_from->get_script_instance()) {
			method_id = p_from->get_script_instance()->get_rpc_method_id(p_name);
		}
		ERR_FAIL_COND_MSG(method_id == UINT16_MAX,
				vformat("Unable to take the `method_id` for the function \"%s\" at path: \"%s\". This happens when the method is not marked as `remote`.", p_name, p_from->get_path()));

		if (method_id <= UINT8_MAX) {
			// The ID fits in 1 byte
			name_id_compression = NETWORK_NAME_ID_COMPRESSION_8;
			MAKE_ROOM(ofs + 1);
			packet_cache.write[ofs] = static_cast<uint8_t>(method_id);
			ofs += 1;
		} else {
			// The ID is larger, let's use 2 bytes
			name_id_compression = NETWORK_NAME_ID_COMPRESSION_16;
			MAKE_ROOM(ofs + 2);
			encode_uint16(method_id, &(packet_cache.write[ofs]));
			ofs += 2;
		}

		if (p_argcount == 0) {
			byte_only_or_no_args = true;
		} else if (p_argcount == 1 && p_arg[0]->get_type() == Variant::PACKED_BYTE_ARRAY) {
			byte_only_or_no_args = true;
			// Special optimization when only the byte vector is sent.
			const Vector<uint8_t> data = *p_arg[0];
			MAKE_ROOM(ofs + data.size());
			memcpy(&(packet_cache.write[ofs]), data.ptr(), sizeof(uint8_t) * data.size());
			ofs += data.size();
		} else {
			// Arguments
			MAKE_ROOM(ofs + 1);
			packet_cache.write[ofs] = p_argcount;
			ofs += 1;
			for (int i = 0; i < p_argcount; i++) {
				int len(0);
				Error err = _encode_and_compress_variant(*p_arg[i], nullptr, len);
				ERR_FAIL_COND_MSG(err != OK, "Unable to encode RPC argument. THIS IS LIKELY A BUG IN THE ENGINE!");
				MAKE_ROOM(ofs + len);
				_encode_and_compress_variant(*p_arg[i], &(packet_cache.write[ofs]), len);
				ofs += len;
			}
		}
	}

	ERR_FAIL_COND(command_type > 7);
	ERR_FAIL_COND(node_id_compression > 3);
	ERR_FAIL_COND(name_id_compression > 1);

	// We can now set the meta
	packet_cache.write[0] = command_type + (node_id_compression << NODE_ID_COMPRESSION_SHIFT) + (name_id_compression << NAME_ID_COMPRESSION_SHIFT) + ((byte_only_or_no_args ? 1 : 0) << BYTE_ONLY_OR_NO_ARGS_SHIFT);

#ifdef DEBUG_ENABLED
	_profile_bandwidth_data("out", ofs);
#endif

	// Take chance and set transfer mode, since all send methods will use it.
	network_peer->set_transfer_mode(p_unreliable ? NetworkedMultiplayerPeer::TRANSFER_MODE_UNRELIABLE : NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE);

	if (has_all_peers) {
		// They all have verified paths, so send fast.
		network_peer->set_target_peer(p_to); // To all of you.
		network_peer->put_packet(packet_cache.ptr(), ofs); // A message with love.
	} else {
		// Unreachable because the node ID is never compressed if the peers doesn't know it.
		CRASH_COND(node_id_compression != NETWORK_NODE_ID_COMPRESSION_32);

		// Not all verified path, so send one by one.

		// Append path at the end, since we will need it for some packets.
		CharString pname = String(from_path).utf8();
		int path_len = encode_cstring(pname.get_data(), nullptr);
		MAKE_ROOM(ofs + path_len);
		encode_cstring(pname.get_data(), &(packet_cache.write[ofs]));

		for (Set<int>::Element *E = connected_peers.front(); E; E = E->next()) {
			if (p_to < 0 && E->get() == -p_to) {
				continue; // Continue, excluded.
			}

			if (p_to > 0 && E->get() != p_to) {
				continue; // Continue, not for this peer.
			}

			Map<int, bool>::Element *F = psc->confirmed_peers.find(E->get());
			ERR_CONTINUE(!F); // Should never happen.

			network_peer->set_target_peer(E->get()); // To this one specifically.

			if (F->get()) {
				// This one confirmed path, so use id.
				encode_uint32(psc->id, &(packet_cache.write[1]));
				network_peer->put_packet(packet_cache.ptr(), ofs);
			} else {
				// This one did not confirm path yet, so use entire path (sorry!).
				encode_uint32(0x80000000 | ofs, &(packet_cache.write[1])); // Offset to path and flag.
				network_peer->put_packet(packet_cache.ptr(), ofs + path_len);
			}
		}
	}
}

void MultiplayerAPI::_add_peer(int p_id) {
	connected_peers.insert(p_id);
	path_get_cache.insert(p_id, PathGetCache());
	emit_signal("network_peer_connected", p_id);
}

void MultiplayerAPI::_del_peer(int p_id) {
	connected_peers.erase(p_id);
	// Cleanup get cache.
	path_get_cache.erase(p_id);
	// Cleanup sent cache.
	// Some refactoring is needed to make this faster and do paths GC.
	List<NodePath> keys;
	path_send_cache.get_key_list(&keys);
	for (List<NodePath>::Element *E = keys.front(); E; E = E->next()) {
		PathSentCache *psc = path_send_cache.getptr(E->get());
		psc->confirmed_peers.erase(p_id);
	}
	emit_signal("network_peer_disconnected", p_id);
}

void MultiplayerAPI::_connected_to_server() {
	emit_signal("connected_to_server");
}

void MultiplayerAPI::_connection_failed() {
	emit_signal("connection_failed");
}

void MultiplayerAPI::_server_disconnected() {
	emit_signal("server_disconnected");
}

void MultiplayerAPI::rpcp(Node *p_node, int p_peer_id, bool p_unreliable, const StringName &p_method, const Variant **p_arg, int p_argcount) {
	ERR_FAIL_COND_MSG(!network_peer.is_valid(), "Trying to call an RPC while no network peer is active.");
	ERR_FAIL_COND_MSG(!p_node->is_inside_tree(), "Trying to call an RPC on a node which is not inside SceneTree.");
	ERR_FAIL_COND_MSG(network_peer->get_connection_status() != NetworkedMultiplayerPeer::CONNECTION_CONNECTED, "Trying to call an RPC via a network peer which is not connected.");

	int node_id = network_peer->get_unique_id();
	bool skip_rpc = node_id == p_peer_id;
	bool call_local_native = false;
	bool call_local_script = false;
	bool is_master = p_node->is_network_master();

	if (p_peer_id == 0 || p_peer_id == node_id || (p_peer_id < 0 && p_peer_id != -node_id)) {
		// Check that send mode can use local call.

		RPCMode rpc_mode = p_node->get_node_rpc_mode(p_method);
		call_local_native = _should_call_local(rpc_mode, is_master, skip_rpc);

		if (call_local_native) {
			// Done below.
		} else if (p_node->get_script_instance()) {
			// Attempt with script.
			rpc_mode = p_node->get_script_instance()->get_rpc_mode(p_method);
			call_local_script = _should_call_local(rpc_mode, is_master, skip_rpc);
		}
	}

	if (!skip_rpc) {
#ifdef DEBUG_ENABLED
		_profile_node_data("out_rpc", p_node->get_instance_id());
#endif

		_send_rpc(p_node, p_peer_id, p_unreliable, false, p_method, p_arg, p_argcount);
	}

	if (call_local_native) {
		int temp_id = rpc_sender_id;
		rpc_sender_id = get_network_unique_id();
		Callable::CallError ce;
		p_node->call(p_method, p_arg, p_argcount, ce);
		rpc_sender_id = temp_id;
		if (ce.error != Callable::CallError::CALL_OK) {
			String error = Variant::get_call_error_text(p_node, p_method, p_arg, p_argcount, ce);
			error = "rpc() aborted in local call:  - " + error + ".";
			ERR_PRINT(error);
			return;
		}
	}

	if (call_local_script) {
		int temp_id = rpc_sender_id;
		rpc_sender_id = get_network_unique_id();
		Callable::CallError ce;
		ce.error = Callable::CallError::CALL_OK;
		p_node->get_script_instance()->call(p_method, p_arg, p_argcount, ce);
		rpc_sender_id = temp_id;
		if (ce.error != Callable::CallError::CALL_OK) {
			String error = Variant::get_call_error_text(p_node, p_method, p_arg, p_argcount, ce);
			error = "rpc() aborted in script local call:  - " + error + ".";
			ERR_PRINT(error);
			return;
		}
	}

	ERR_FAIL_COND_MSG(skip_rpc && !(call_local_native || call_local_script), "RPC '" + p_method + "' on yourself is not allowed by selected mode.");
}

void MultiplayerAPI::rsetp(Node *p_node, int p_peer_id, bool p_unreliable, const StringName &p_property, const Variant &p_value) {
	ERR_FAIL_COND_MSG(!network_peer.is_valid(), "Trying to RSET while no network peer is active.");
	ERR_FAIL_COND_MSG(!p_node->is_inside_tree(), "Trying to RSET on a node which is not inside SceneTree.");
	ERR_FAIL_COND_MSG(network_peer->get_connection_status() != NetworkedMultiplayerPeer::CONNECTION_CONNECTED, "Trying to send an RSET via a network peer which is not connected.");

	int node_id = network_peer->get_unique_id();
	bool is_master = p_node->is_network_master();
	bool skip_rset = node_id == p_peer_id;
	bool set_local = false;

	if (p_peer_id == 0 || p_peer_id == node_id || (p_peer_id < 0 && p_peer_id != -node_id)) {
		// Check that send mode can use local call.
		RPCMode rpc_mode = p_node->get_node_rset_mode(p_property);
		set_local = _should_call_local(rpc_mode, is_master, skip_rset);

		if (set_local) {
			bool valid;
			int temp_id = rpc_sender_id;

			rpc_sender_id = get_network_unique_id();
			p_node->set(p_property, p_value, &valid);
			rpc_sender_id = temp_id;

			if (!valid) {
				String error = "rset() aborted in local set, property not found:  - " + String(p_property) + ".";
				ERR_PRINT(error);
				return;
			}
		} else if (p_node->get_script_instance()) {
			// Attempt with script.
			rpc_mode = p_node->get_script_instance()->get_rset_mode(p_property);

			set_local = _should_call_local(rpc_mode, is_master, skip_rset);

			if (set_local) {
				int temp_id = rpc_sender_id;

				rpc_sender_id = get_network_unique_id();
				bool valid = p_node->get_script_instance()->set(p_property, p_value);
				rpc_sender_id = temp_id;

				if (!valid) {
					String error = "rset() aborted in local script set, property not found:  - " + String(p_property) + ".";
					ERR_PRINT(error);
					return;
				}
			}
		}
	}

	if (skip_rset) {
		ERR_FAIL_COND_MSG(!set_local, "RSET for '" + p_property + "' on yourself is not allowed by selected mode.");
		return;
	}

#ifdef DEBUG_ENABLED
	_profile_node_data("out_rset", p_node->get_instance_id());
#endif

	const Variant *vptr = &p_value;

	_send_rpc(p_node, p_peer_id, p_unreliable, true, p_property, &vptr, 1);
}

Error MultiplayerAPI::send_bytes(Vector<uint8_t> p_data, int p_to, NetworkedMultiplayerPeer::TransferMode p_mode) {
	ERR_FAIL_COND_V_MSG(p_data.size() < 1, ERR_INVALID_DATA, "Trying to send an empty raw packet.");
	ERR_FAIL_COND_V_MSG(!network_peer.is_valid(), ERR_UNCONFIGURED, "Trying to send a raw packet while no network peer is active.");
	ERR_FAIL_COND_V_MSG(network_peer->get_connection_status() != NetworkedMultiplayerPeer::CONNECTION_CONNECTED, ERR_UNCONFIGURED, "Trying to send a raw packet via a network peer which is not connected.");

	MAKE_ROOM(p_data.size() + 1);
	const uint8_t *r = p_data.ptr();
	packet_cache.write[0] = NETWORK_COMMAND_RAW;
	memcpy(&packet_cache.write[1], &r[0], p_data.size());

	network_peer->set_target_peer(p_to);
	network_peer->set_transfer_mode(p_mode);

	return network_peer->put_packet(packet_cache.ptr(), p_data.size() + 1);
}

void MultiplayerAPI::_process_raw(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(p_packet_len < 2, "Invalid packet received. Size too small.");

	Vector<uint8_t> out;
	int len = p_packet_len - 1;
	out.resize(len);
	{
		uint8_t *w = out.ptrw();
		memcpy(&w[0], &p_packet[1], len);
	}
	emit_signal("network_peer_packet", p_from, out);
}

int MultiplayerAPI::get_network_unique_id() const {
	ERR_FAIL_COND_V_MSG(!network_peer.is_valid(), 0, "No network peer is assigned. Unable to get unique network ID.");
	return network_peer->get_unique_id();
}

bool MultiplayerAPI::is_network_server() const {
	// XXX Maybe fail silently? Maybe should actually return true to make development of both local and online multiplayer easier?
	ERR_FAIL_COND_V_MSG(!network_peer.is_valid(), false, "No network peer is assigned. I can't be a server.");
	return network_peer->is_server();
}

void MultiplayerAPI::set_refuse_new_network_connections(bool p_refuse) {
	ERR_FAIL_COND_MSG(!network_peer.is_valid(), "No network peer is assigned. Unable to set 'refuse_new_connections'.");
	network_peer->set_refuse_new_connections(p_refuse);
}

bool MultiplayerAPI::is_refusing_new_network_connections() const {
	ERR_FAIL_COND_V_MSG(!network_peer.is_valid(), false, "No network peer is assigned. Unable to get 'refuse_new_connections'.");
	return network_peer->is_refusing_new_connections();
}

Vector<int> MultiplayerAPI::get_network_connected_peers() const {
	ERR_FAIL_COND_V_MSG(!network_peer.is_valid(), Vector<int>(), "No network peer is assigned. Assume no peers are connected.");

	Vector<int> ret;
	for (Set<int>::Element *E = connected_peers.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}

	return ret;
}

void MultiplayerAPI::set_allow_object_decoding(bool p_enable) {
	allow_object_decoding = p_enable;
}

bool MultiplayerAPI::is_object_decoding_allowed() const {
	return allow_object_decoding;
}

void MultiplayerAPI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_node", "node"), &MultiplayerAPI::set_root_node);
	ClassDB::bind_method(D_METHOD("get_root_node"), &MultiplayerAPI::get_root_node);
	ClassDB::bind_method(D_METHOD("send_bytes", "bytes", "id", "mode"), &MultiplayerAPI::send_bytes, DEFVAL(NetworkedMultiplayerPeer::TARGET_PEER_BROADCAST), DEFVAL(NetworkedMultiplayerPeer::TRANSFER_MODE_RELIABLE));
	ClassDB::bind_method(D_METHOD("has_network_peer"), &MultiplayerAPI::has_network_peer);
	ClassDB::bind_method(D_METHOD("get_network_peer"), &MultiplayerAPI::get_network_peer);
	ClassDB::bind_method(D_METHOD("get_network_unique_id"), &MultiplayerAPI::get_network_unique_id);
	ClassDB::bind_method(D_METHOD("is_network_server"), &MultiplayerAPI::is_network_server);
	ClassDB::bind_method(D_METHOD("get_rpc_sender_id"), &MultiplayerAPI::get_rpc_sender_id);
	ClassDB::bind_method(D_METHOD("set_network_peer", "peer"), &MultiplayerAPI::set_network_peer);
	ClassDB::bind_method(D_METHOD("poll"), &MultiplayerAPI::poll);
	ClassDB::bind_method(D_METHOD("clear"), &MultiplayerAPI::clear);

	ClassDB::bind_method(D_METHOD("get_network_connected_peers"), &MultiplayerAPI::get_network_connected_peers);
	ClassDB::bind_method(D_METHOD("set_refuse_new_network_connections", "refuse"), &MultiplayerAPI::set_refuse_new_network_connections);
	ClassDB::bind_method(D_METHOD("is_refusing_new_network_connections"), &MultiplayerAPI::is_refusing_new_network_connections);
	ClassDB::bind_method(D_METHOD("set_allow_object_decoding", "enable"), &MultiplayerAPI::set_allow_object_decoding);
	ClassDB::bind_method(D_METHOD("is_object_decoding_allowed"), &MultiplayerAPI::is_object_decoding_allowed);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_object_decoding"), "set_allow_object_decoding", "is_object_decoding_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_network_connections"), "set_refuse_new_network_connections", "is_refusing_new_network_connections");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "network_peer", PROPERTY_HINT_RESOURCE_TYPE, "NetworkedMultiplayerPeer", 0), "set_network_peer", "get_network_peer");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "root_node", PROPERTY_HINT_RESOURCE_TYPE, "Node", 0), "set_root_node", "get_root_node");
	ADD_PROPERTY_DEFAULT("refuse_new_network_connections", false);

	ADD_SIGNAL(MethodInfo("network_peer_connected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("network_peer_disconnected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("network_peer_packet", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "packet")));
	ADD_SIGNAL(MethodInfo("connected_to_server"));
	ADD_SIGNAL(MethodInfo("connection_failed"));
	ADD_SIGNAL(MethodInfo("server_disconnected"));

	BIND_ENUM_CONSTANT(RPC_MODE_DISABLED);
	BIND_ENUM_CONSTANT(RPC_MODE_REMOTE);
	BIND_ENUM_CONSTANT(RPC_MODE_MASTER);
	BIND_ENUM_CONSTANT(RPC_MODE_PUPPET);
	BIND_ENUM_CONSTANT(RPC_MODE_REMOTESYNC);
	BIND_ENUM_CONSTANT(RPC_MODE_MASTERSYNC);
	BIND_ENUM_CONSTANT(RPC_MODE_PUPPETSYNC);
}

MultiplayerAPI::MultiplayerAPI() {
	clear();
}

MultiplayerAPI::~MultiplayerAPI() {
	clear();
}
