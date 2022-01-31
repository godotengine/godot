/*************************************************************************/
/*  rpc_manager.cpp                                                      */
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

#include "core/multiplayer/rpc_manager.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"
#include "core/multiplayer/multiplayer_api.h"
#include "scene/main/node.h"

#ifdef DEBUG_ENABLED
_FORCE_INLINE_ void RPCManager::_profile_node_data(const String &p_what, ObjectID p_id) {
	if (EngineDebugger::is_profiling("multiplayer")) {
		Array values;
		values.push_back("node");
		values.push_back(p_id);
		values.push_back(p_what);
		EngineDebugger::profiler_add_frame_data("multiplayer", values);
	}
}
#else
_FORCE_INLINE_ void RPCManager::_profile_node_data(const String &p_what, ObjectID p_id) {}
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

const Multiplayer::RPCConfig _get_rpc_config(const Node *p_node, const StringName &p_method, uint16_t &r_id) {
	const Vector<Multiplayer::RPCConfig> node_config = p_node->get_node_rpc_methods();
	for (int i = 0; i < node_config.size(); i++) {
		if (node_config[i].name == p_method) {
			r_id = ((uint16_t)i) | (1 << 15);
			return node_config[i];
		}
	}
	if (p_node->get_script_instance()) {
		const Vector<Multiplayer::RPCConfig> script_config = p_node->get_script_instance()->get_rpc_methods();
		for (int i = 0; i < script_config.size(); i++) {
			if (script_config[i].name == p_method) {
				r_id = (uint16_t)i;
				return script_config[i];
			}
		}
	}
	return Multiplayer::RPCConfig();
}

const Multiplayer::RPCConfig _get_rpc_config_by_id(Node *p_node, uint16_t p_id) {
	Vector<Multiplayer::RPCConfig> config;
	uint16_t id = p_id;
	if (id & (1 << 15)) {
		id = id & ~(1 << 15);
		config = p_node->get_node_rpc_methods();
	} else if (p_node->get_script_instance()) {
		config = p_node->get_script_instance()->get_rpc_methods();
	}
	if (id < config.size()) {
		return config[id];
	}
	return Multiplayer::RPCConfig();
}

_FORCE_INLINE_ bool _can_call_mode(Node *p_node, Multiplayer::RPCMode mode, int p_remote_id) {
	switch (mode) {
		case Multiplayer::RPC_MODE_DISABLED: {
			return false;
		} break;
		case Multiplayer::RPC_MODE_ANY_PEER: {
			return true;
		} break;
		case Multiplayer::RPC_MODE_AUTHORITY: {
			return !p_node->is_multiplayer_authority() && p_remote_id == p_node->get_multiplayer_authority();
		} break;
	}

	return false;
}

String RPCManager::get_rpc_md5(const Node *p_node) {
	String rpc_list;
	const Vector<Multiplayer::RPCConfig> node_config = p_node->get_node_rpc_methods();
	for (int i = 0; i < node_config.size(); i++) {
		rpc_list += String(node_config[i].name);
	}
	if (p_node->get_script_instance()) {
		const Vector<Multiplayer::RPCConfig> script_config = p_node->get_script_instance()->get_rpc_methods();
		for (int i = 0; i < script_config.size(); i++) {
			rpc_list += String(script_config[i].name);
		}
	}
	return rpc_list.md5_text();
}

Node *RPCManager::_process_get_node(int p_from, const uint8_t *p_packet, uint32_t p_node_target, int p_packet_len) {
	Node *node = nullptr;

	if (p_node_target & 0x80000000) {
		// Use full path (not cached yet).
		int ofs = p_node_target & 0x7FFFFFFF;

		ERR_FAIL_COND_V_MSG(ofs >= p_packet_len, nullptr, "Invalid packet received. Size smaller than declared.");

		String paths;
		paths.parse_utf8((const char *)&p_packet[ofs], p_packet_len - ofs);

		NodePath np = paths;

		node = multiplayer->get_root_node()->get_node(np);

		if (!node) {
			ERR_PRINT("Failed to get path from RPC: " + String(np) + ".");
		}
		return node;
	} else {
		// Use cached path.
		return multiplayer->get_cached_node(p_from, p_node_target);
	}
}

void RPCManager::process_rpc(int p_from, const uint8_t *p_packet, int p_packet_len) {
	// Extract packet meta
	int packet_min_size = 1;
	int name_id_offset = 1;
	ERR_FAIL_COND_MSG(p_packet_len < packet_min_size, "Invalid packet received. Size too small.");
	// Compute the meta size, which depends on the compression level.
	int node_id_compression = (p_packet[0] & NODE_ID_COMPRESSION_FLAG) >> NODE_ID_COMPRESSION_SHIFT;
	int name_id_compression = (p_packet[0] & NAME_ID_COMPRESSION_FLAG) >> NAME_ID_COMPRESSION_SHIFT;

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
	_process_rpc(node, name_id, p_from, p_packet, packet_len, packet_min_size);
}

void RPCManager::_process_rpc(Node *p_node, const uint16_t p_rpc_method_id, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset) {
	ERR_FAIL_COND_MSG(p_offset > p_packet_len, "Invalid packet received. Size too small.");

	// Check that remote can call the RPC on this node.
	const Multiplayer::RPCConfig config = _get_rpc_config_by_id(p_node, p_rpc_method_id);
	ERR_FAIL_COND(config.name == StringName());

	bool can_call = _can_call_mode(p_node, config.rpc_mode, p_from);
	ERR_FAIL_COND_MSG(!can_call, "RPC '" + String(config.name) + "' is not allowed on node " + p_node->get_path() + " from: " + itos(p_from) + ". Mode is " + itos((int)config.rpc_mode) + ", authority is " + itos(p_node->get_multiplayer_authority()) + ".");

	int argc = 0;
	bool byte_only = false;

	const bool byte_only_or_no_args = p_packet[0] & BYTE_ONLY_OR_NO_ARGS_FLAG;
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
			Error err = multiplayer->decode_and_decompress_variant(args.write[i], &p_packet[p_offset], p_packet_len - p_offset, &vlen);
			ERR_FAIL_COND_MSG(err != OK, "Invalid packet received. Unable to decode RPC argument.");

			argp.write[i] = &args[i];
			p_offset += vlen;
		}
	}

	Callable::CallError ce;

	p_node->call(config.name, (const Variant **)argp.ptr(), argc, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		String error = Variant::get_call_error_text(p_node, config.name, (const Variant **)argp.ptr(), argc, ce);
		error = "RPC - " + error;
		ERR_PRINT(error);
	}
}

void RPCManager::_send_rpc(Node *p_from, int p_to, uint16_t p_rpc_id, const Multiplayer::RPCConfig &p_config, const StringName &p_name, const Variant **p_arg, int p_argcount) {
	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	ERR_FAIL_COND_MSG(peer.is_null(), "Attempt to call RPC without active multiplayer peer.");

	ERR_FAIL_COND_MSG(peer->get_connection_status() == MultiplayerPeer::CONNECTION_CONNECTING, "Attempt to call RPC while multiplayer peer is not connected yet.");

	ERR_FAIL_COND_MSG(peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED, "Attempt to call RPC while multiplayer peer is disconnected.");

	ERR_FAIL_COND_MSG(p_argcount > 255, "Too many arguments (>255).");

	if (p_to != 0 && !multiplayer->get_connected_peers().has(ABS(p_to))) {
		ERR_FAIL_COND_MSG(p_to == peer->get_unique_id(), "Attempt to call RPC on yourself! Peer unique ID: " + itos(peer->get_unique_id()) + ".");

		ERR_FAIL_MSG("Attempt to call RPC with unknown peer ID: " + itos(p_to) + ".");
	}

	NodePath from_path = (multiplayer->get_root_node()->get_path()).rel_path_to(p_from->get_path());
	ERR_FAIL_COND_MSG(from_path.is_empty(), "Unable to send RPC. Relative path is empty. THIS IS LIKELY A BUG IN THE ENGINE!");

	// See if all peers have cached path (if so, call can be fast).
	int psc_id;
	const bool has_all_peers = multiplayer->send_confirm_path(p_from, from_path, p_to, psc_id);

	// Create base packet, lots of hardcode because it must be tight.

	int ofs = 0;

#define MAKE_ROOM(m_amount)             \
	if (packet_cache.size() < m_amount) \
		packet_cache.resize(m_amount);

	// Encode meta.
	uint8_t command_type = MultiplayerAPI::NETWORK_COMMAND_REMOTE_CALL;
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
		if (psc_id >= 0 && psc_id <= 255) {
			// We can encode the id in 1 byte
			node_id_compression = NETWORK_NODE_ID_COMPRESSION_8;
			MAKE_ROOM(ofs + 1);
			packet_cache.write[ofs] = static_cast<uint8_t>(psc_id);
			ofs += 1;
		} else if (psc_id >= 0 && psc_id <= 65535) {
			// We can encode the id in 2 bytes
			node_id_compression = NETWORK_NODE_ID_COMPRESSION_16;
			MAKE_ROOM(ofs + 2);
			encode_uint16(static_cast<uint16_t>(psc_id), &(packet_cache.write[ofs]));
			ofs += 2;
		} else {
			// Too big, let's use 4 bytes.
			node_id_compression = NETWORK_NODE_ID_COMPRESSION_32;
			MAKE_ROOM(ofs + 4);
			encode_uint32(psc_id, &(packet_cache.write[ofs]));
			ofs += 4;
		}
	} else {
		// The targets don't know the node yet, so we need to use 32 bits int.
		node_id_compression = NETWORK_NODE_ID_COMPRESSION_32;
		MAKE_ROOM(ofs + 4);
		encode_uint32(psc_id, &(packet_cache.write[ofs]));
		ofs += 4;
	}

	// Encode method ID
	if (p_rpc_id <= UINT8_MAX) {
		// The ID fits in 1 byte
		name_id_compression = NETWORK_NAME_ID_COMPRESSION_8;
		MAKE_ROOM(ofs + 1);
		packet_cache.write[ofs] = static_cast<uint8_t>(p_rpc_id);
		ofs += 1;
	} else {
		// The ID is larger, let's use 2 bytes
		name_id_compression = NETWORK_NAME_ID_COMPRESSION_16;
		MAKE_ROOM(ofs + 2);
		encode_uint16(p_rpc_id, &(packet_cache.write[ofs]));
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
			Error err = multiplayer->encode_and_compress_variant(*p_arg[i], nullptr, len);
			ERR_FAIL_COND_MSG(err != OK, "Unable to encode RPC argument. THIS IS LIKELY A BUG IN THE ENGINE!");
			MAKE_ROOM(ofs + len);
			multiplayer->encode_and_compress_variant(*p_arg[i], &(packet_cache.write[ofs]), len);
			ofs += len;
		}
	}

	ERR_FAIL_COND(command_type > 7);
	ERR_FAIL_COND(node_id_compression > 3);
	ERR_FAIL_COND(name_id_compression > 1);

	// We can now set the meta
	packet_cache.write[0] = command_type + (node_id_compression << NODE_ID_COMPRESSION_SHIFT) + (name_id_compression << NAME_ID_COMPRESSION_SHIFT) + (byte_only_or_no_args ? BYTE_ONLY_OR_NO_ARGS_FLAG : 0);

#ifdef DEBUG_ENABLED
	multiplayer->profile_bandwidth("out", ofs);
#endif

	// Take chance and set transfer mode, since all send methods will use it.
	peer->set_transfer_channel(p_config.channel);
	peer->set_transfer_mode(p_config.transfer_mode);

	if (has_all_peers) {
		// They all have verified paths, so send fast.
		peer->set_target_peer(p_to); // To all of you.
		peer->put_packet(packet_cache.ptr(), ofs); // A message with love.
	} else {
		// Unreachable because the node ID is never compressed if the peers doesn't know it.
		CRASH_COND(node_id_compression != NETWORK_NODE_ID_COMPRESSION_32);

		// Not all verified path, so send one by one.

		// Append path at the end, since we will need it for some packets.
		CharString pname = String(from_path).utf8();
		int path_len = encode_cstring(pname.get_data(), nullptr);
		MAKE_ROOM(ofs + path_len);
		encode_cstring(pname.get_data(), &(packet_cache.write[ofs]));

		for (const int &P : multiplayer->get_connected_peers()) {
			if (p_to < 0 && P == -p_to) {
				continue; // Continue, excluded.
			}

			if (p_to > 0 && P != p_to) {
				continue; // Continue, not for this peer.
			}

			bool confirmed = multiplayer->is_cache_confirmed(from_path, P);

			peer->set_target_peer(P); // To this one specifically.

			if (confirmed) {
				// This one confirmed path, so use id.
				encode_uint32(psc_id, &(packet_cache.write[1]));
				peer->put_packet(packet_cache.ptr(), ofs);
			} else {
				// This one did not confirm path yet, so use entire path (sorry!).
				encode_uint32(0x80000000 | ofs, &(packet_cache.write[1])); // Offset to path and flag.
				peer->put_packet(packet_cache.ptr(), ofs + path_len);
			}
		}
	}
}

void RPCManager::rpcp(Node *p_node, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount) {
	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	ERR_FAIL_COND_MSG(!peer.is_valid(), "Trying to call an RPC while no multiplayer peer is active.");
	ERR_FAIL_COND_MSG(!p_node->is_inside_tree(), "Trying to call an RPC on a node which is not inside SceneTree.");
	ERR_FAIL_COND_MSG(peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED, "Trying to call an RPC via a multiplayer peer which is not connected.");

	int node_id = peer->get_unique_id();
	bool call_local_native = false;
	bool call_local_script = false;
	uint16_t rpc_id = UINT16_MAX;
	const Multiplayer::RPCConfig config = _get_rpc_config(p_node, p_method, rpc_id);
	ERR_FAIL_COND_MSG(config.name == StringName(),
			vformat("Unable to get the RPC configuration for the function \"%s\" at path: \"%s\". This happens when the method is not marked for RPCs.", p_method, p_node->get_path()));
	if (p_peer_id == 0 || p_peer_id == node_id || (p_peer_id < 0 && p_peer_id != -node_id)) {
		if (rpc_id & (1 << 15)) {
			call_local_native = config.call_local;
		} else {
			call_local_script = config.call_local;
		}
	}

	if (p_peer_id != node_id) {
#ifdef DEBUG_ENABLED
		_profile_node_data("out_rpc", p_node->get_instance_id());
#endif

		_send_rpc(p_node, p_peer_id, rpc_id, config, p_method, p_arg, p_argcount);
	}

	if (call_local_native) {
		Callable::CallError ce;

		multiplayer->set_remote_sender_override(peer->get_unique_id());
		p_node->call(p_method, p_arg, p_argcount, ce);
		multiplayer->set_remote_sender_override(0);

		if (ce.error != Callable::CallError::CALL_OK) {
			String error = Variant::get_call_error_text(p_node, p_method, p_arg, p_argcount, ce);
			error = "rpc() aborted in local call:  - " + error + ".";
			ERR_PRINT(error);
			return;
		}
	}

	if (call_local_script) {
		Callable::CallError ce;
		ce.error = Callable::CallError::CALL_OK;

		multiplayer->set_remote_sender_override(peer->get_unique_id());
		p_node->get_script_instance()->call(p_method, p_arg, p_argcount, ce);
		multiplayer->set_remote_sender_override(0);

		if (ce.error != Callable::CallError::CALL_OK) {
			String error = Variant::get_call_error_text(p_node, p_method, p_arg, p_argcount, ce);
			error = "rpc() aborted in script local call:  - " + error + ".";
			ERR_PRINT(error);
			return;
		}
	}

	ERR_FAIL_COND_MSG(p_peer_id == node_id && !config.call_local, "RPC '" + p_method + "' on yourself is not allowed by selected mode.");
}
