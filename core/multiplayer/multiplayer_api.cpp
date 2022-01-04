/*************************************************************************/
/*  multiplayer_api.cpp                                                  */
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

#include "multiplayer_api.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"
#include "core/multiplayer/multiplayer_replicator.h"
#include "core/multiplayer/rpc_manager.h"
#include "scene/main/node.h"

#include <stdint.h>

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#endif

#ifdef DEBUG_ENABLED
void MultiplayerAPI::profile_bandwidth(const String &p_inout, int p_size) {
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

void MultiplayerAPI::poll() {
	if (!multiplayer_peer.is_valid() || multiplayer_peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED) {
		return;
	}

	multiplayer_peer->poll();

	if (!multiplayer_peer.is_valid()) { // It's possible that polling might have resulted in a disconnection, so check here.
		return;
	}

	while (multiplayer_peer->get_available_packet_count()) {
		int sender = multiplayer_peer->get_packet_peer();
		const uint8_t *packet;
		int len;

		Error err = multiplayer_peer->get_packet(&packet, len);
		if (err != OK) {
			ERR_PRINT("Error getting packet!");
			break; // Something is wrong!
		}

		remote_sender_id = sender;
		_process_packet(sender, packet, len);
		remote_sender_id = 0;

		if (!multiplayer_peer.is_valid()) {
			break; // It's also possible that a packet or RPC caused a disconnection, so also check here.
		}
	}
	if (multiplayer_peer.is_valid() && multiplayer_peer->get_connection_status() == MultiplayerPeer::CONNECTION_CONNECTED) {
		replicator->poll();
	}
}

void MultiplayerAPI::clear() {
	replicator->clear();
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

void MultiplayerAPI::set_multiplayer_peer(const Ref<MultiplayerPeer> &p_peer) {
	if (p_peer == multiplayer_peer) {
		return; // Nothing to do
	}

	ERR_FAIL_COND_MSG(p_peer.is_valid() && p_peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED,
			"Supplied MultiplayerPeer must be connecting or connected.");

	if (multiplayer_peer.is_valid()) {
		multiplayer_peer->disconnect("peer_connected", callable_mp(this, &MultiplayerAPI::_add_peer));
		multiplayer_peer->disconnect("peer_disconnected", callable_mp(this, &MultiplayerAPI::_del_peer));
		multiplayer_peer->disconnect("connection_succeeded", callable_mp(this, &MultiplayerAPI::_connected_to_server));
		multiplayer_peer->disconnect("connection_failed", callable_mp(this, &MultiplayerAPI::_connection_failed));
		multiplayer_peer->disconnect("server_disconnected", callable_mp(this, &MultiplayerAPI::_server_disconnected));
		clear();
	}

	multiplayer_peer = p_peer;

	if (multiplayer_peer.is_valid()) {
		multiplayer_peer->connect("peer_connected", callable_mp(this, &MultiplayerAPI::_add_peer));
		multiplayer_peer->connect("peer_disconnected", callable_mp(this, &MultiplayerAPI::_del_peer));
		multiplayer_peer->connect("connection_succeeded", callable_mp(this, &MultiplayerAPI::_connected_to_server));
		multiplayer_peer->connect("connection_failed", callable_mp(this, &MultiplayerAPI::_connection_failed));
		multiplayer_peer->connect("server_disconnected", callable_mp(this, &MultiplayerAPI::_server_disconnected));
	}
}

Ref<MultiplayerPeer> MultiplayerAPI::get_multiplayer_peer() const {
	return multiplayer_peer;
}

void MultiplayerAPI::_process_packet(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(root_node == nullptr, "Multiplayer root node was not initialized. If you are using custom multiplayer, remember to set the root node via MultiplayerAPI.set_root_node before using it.");
	ERR_FAIL_COND_MSG(p_packet_len < 1, "Invalid packet received. Size too small.");

#ifdef DEBUG_ENABLED
	profile_bandwidth("in", p_packet_len);
#endif

	// Extract the `packet_type` from the LSB three bits:
	uint8_t packet_type = p_packet[0] & CMD_MASK;

	switch (packet_type) {
		case NETWORK_COMMAND_SIMPLIFY_PATH: {
			_process_simplify_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_CONFIRM_PATH: {
			_process_confirm_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_REMOTE_CALL: {
			rpc_manager->process_rpc(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_RAW: {
			_process_raw(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_SPAWN: {
			replicator->process_spawn_despawn(p_from, p_packet, p_packet_len, true);
		} break;
		case NETWORK_COMMAND_DESPAWN: {
			replicator->process_spawn_despawn(p_from, p_packet, p_packet_len, false);
		} break;
		case NETWORK_COMMAND_SYNC: {
			replicator->process_sync(p_from, p_packet, p_packet_len);
		} break;
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
	const bool valid_rpc_checksum = rpc_manager->get_rpc_md5(node) == methods_md5;
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

	multiplayer_peer->set_transfer_channel(0);
	multiplayer_peer->set_transfer_mode(Multiplayer::TRANSFER_MODE_RELIABLE);
	multiplayer_peer->set_target_peer(p_from);
	multiplayer_peer->put_packet(packet.ptr(), packet.size());
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
		const String methods_md5 = rpc_manager->get_rpc_md5(p_node);
		const int methods_md5_len = 33; // 32 + 1 for the `0` that is added by the encoder.

		Vector<uint8_t> packet;
		packet.resize(1 + 4 + path_len + methods_md5_len);
		int ofs = 0;

		packet.write[ofs] = NETWORK_COMMAND_SIMPLIFY_PATH;
		ofs += 1;

		ofs += encode_cstring(methods_md5.utf8().get_data(), &packet.write[ofs]);

		ofs += encode_uint32(psc->id, &packet.write[ofs]);

		ofs += encode_cstring(path.get_data(), &packet.write[ofs]);

		for (int &E : peers_to_add) {
			multiplayer_peer->set_target_peer(E); // To all of you.
			multiplayer_peer->set_transfer_channel(0);
			multiplayer_peer->set_transfer_mode(Multiplayer::TRANSFER_MODE_RELIABLE);
			multiplayer_peer->put_packet(packet.ptr(), packet.size());

			psc->confirmed_peers.insert(E, false); // Insert into confirmed, but as false since it was not confirmed.
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
Error MultiplayerAPI::encode_and_compress_variant(const Variant &p_variant, uint8_t *r_buffer, int &r_len) {
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

Error MultiplayerAPI::decode_and_decompress_variant(Variant &r_variant, const uint8_t *p_buffer, int p_len, int *r_len) {
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

void MultiplayerAPI::_add_peer(int p_id) {
	connected_peers.insert(p_id);
	path_get_cache.insert(p_id, PathGetCache());
	if (is_server()) {
		replicator->spawn_all(p_id);
	}
	emit_signal(SNAME("peer_connected"), p_id);
}

void MultiplayerAPI::_del_peer(int p_id) {
	connected_peers.erase(p_id);
	// Cleanup get cache.
	path_get_cache.erase(p_id);
	// Cleanup sent cache.
	// Some refactoring is needed to make this faster and do paths GC.
	List<NodePath> keys;
	path_send_cache.get_key_list(&keys);
	for (const NodePath &E : keys) {
		PathSentCache *psc = path_send_cache.getptr(E);
		psc->confirmed_peers.erase(p_id);
	}
	emit_signal(SNAME("peer_disconnected"), p_id);
}

void MultiplayerAPI::_connected_to_server() {
	emit_signal(SNAME("connected_to_server"));
}

void MultiplayerAPI::_connection_failed() {
	emit_signal(SNAME("connection_failed"));
}

void MultiplayerAPI::_server_disconnected() {
	emit_signal(SNAME("server_disconnected"));
}

Error MultiplayerAPI::send_bytes(Vector<uint8_t> p_data, int p_to, Multiplayer::TransferMode p_mode, int p_channel) {
	ERR_FAIL_COND_V_MSG(p_data.size() < 1, ERR_INVALID_DATA, "Trying to send an empty raw packet.");
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), ERR_UNCONFIGURED, "Trying to send a raw packet while no multiplayer peer is active.");
	ERR_FAIL_COND_V_MSG(multiplayer_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED, ERR_UNCONFIGURED, "Trying to send a raw packet via a multiplayer peer which is not connected.");

	if (packet_cache.size() < p_data.size() + 1) {
		packet_cache.resize(p_data.size() + 1);
	}

	const uint8_t *r = p_data.ptr();
	packet_cache.write[0] = NETWORK_COMMAND_RAW;
	memcpy(&packet_cache.write[1], &r[0], p_data.size());

	multiplayer_peer->set_target_peer(p_to);
	multiplayer_peer->set_transfer_channel(p_channel);
	multiplayer_peer->set_transfer_mode(p_mode);

	return multiplayer_peer->put_packet(packet_cache.ptr(), p_data.size() + 1);
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
	emit_signal(SNAME("peer_packet"), p_from, out);
}

bool MultiplayerAPI::is_cache_confirmed(NodePath p_path, int p_peer) {
	const PathSentCache *psc = path_send_cache.getptr(p_path);
	ERR_FAIL_COND_V(!psc, false);
	const Map<int, bool>::Element *F = psc->confirmed_peers.find(p_peer);
	ERR_FAIL_COND_V(!F, false); // Should never happen.
	return F->get();
}

bool MultiplayerAPI::send_confirm_path(Node *p_node, NodePath p_path, int p_peer_id, int &r_id) {
	// See if the path is cached.
	PathSentCache *psc = path_send_cache.getptr(p_path);
	if (!psc) {
		// Path is not cached, create.
		path_send_cache[p_path] = PathSentCache();
		psc = path_send_cache.getptr(p_path);
		psc->id = last_send_cache_id++;
	}
	r_id = psc->id;

	// See if all peers have cached path (if so, call can be fast).
	return _send_confirm_path(p_node, p_path, psc, p_peer_id);
}

Node *MultiplayerAPI::get_cached_node(int p_from, uint32_t p_node_id) {
	Map<int, PathGetCache>::Element *E = path_get_cache.find(p_from);
	ERR_FAIL_COND_V_MSG(!E, nullptr, vformat("No cache found for peer %d.", p_from));

	Map<int, PathGetCache::NodeInfo>::Element *F = E->get().nodes.find(p_node_id);
	ERR_FAIL_COND_V_MSG(!F, nullptr, vformat("ID %d not found in cache of peer %d.", p_node_id, p_from));

	PathGetCache::NodeInfo *ni = &F->get();
	Node *node = root_node->get_node(ni->path);
	if (!node) {
		ERR_PRINT("Failed to get cached path: " + String(ni->path) + ".");
	}
	return node;
}

int MultiplayerAPI::get_unique_id() const {
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), 0, "No multiplayer peer is assigned. Unable to get unique ID.");
	return multiplayer_peer->get_unique_id();
}

bool MultiplayerAPI::is_server() const {
	return multiplayer_peer.is_valid() && multiplayer_peer->is_server();
}

void MultiplayerAPI::set_refuse_new_connections(bool p_refuse) {
	ERR_FAIL_COND_MSG(!multiplayer_peer.is_valid(), "No multiplayer peer is assigned. Unable to set 'refuse_new_connections'.");
	multiplayer_peer->set_refuse_new_connections(p_refuse);
}

bool MultiplayerAPI::is_refusing_new_connections() const {
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), false, "No multiplayer peer is assigned. Unable to get 'refuse_new_connections'.");
	return multiplayer_peer->is_refusing_new_connections();
}

Vector<int> MultiplayerAPI::get_peer_ids() const {
	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), Vector<int>(), "No multiplayer peer is assigned. Assume no peers are connected.");

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

void MultiplayerAPI::scene_enter_exit_notify(const String &p_scene, Node *p_node, bool p_enter) {
	replicator->scene_enter_exit_notify(p_scene, p_node, p_enter);
}

void MultiplayerAPI::rpcp(Node *p_node, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount) {
	rpc_manager->rpcp(p_node, p_peer_id, p_method, p_arg, p_argcount);
}

void MultiplayerAPI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_node", "node"), &MultiplayerAPI::set_root_node);
	ClassDB::bind_method(D_METHOD("get_root_node"), &MultiplayerAPI::get_root_node);
	ClassDB::bind_method(D_METHOD("send_bytes", "bytes", "id", "mode", "channel"), &MultiplayerAPI::send_bytes, DEFVAL(MultiplayerPeer::TARGET_PEER_BROADCAST), DEFVAL(Multiplayer::TRANSFER_MODE_RELIABLE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("has_multiplayer_peer"), &MultiplayerAPI::has_multiplayer_peer);
	ClassDB::bind_method(D_METHOD("get_multiplayer_peer"), &MultiplayerAPI::get_multiplayer_peer);
	ClassDB::bind_method(D_METHOD("set_multiplayer_peer", "peer"), &MultiplayerAPI::set_multiplayer_peer);
	ClassDB::bind_method(D_METHOD("get_unique_id"), &MultiplayerAPI::get_unique_id);
	ClassDB::bind_method(D_METHOD("is_server"), &MultiplayerAPI::is_server);
	ClassDB::bind_method(D_METHOD("get_remote_sender_id"), &MultiplayerAPI::get_remote_sender_id);
	ClassDB::bind_method(D_METHOD("poll"), &MultiplayerAPI::poll);
	ClassDB::bind_method(D_METHOD("clear"), &MultiplayerAPI::clear);

	ClassDB::bind_method(D_METHOD("get_peers"), &MultiplayerAPI::get_peer_ids);
	ClassDB::bind_method(D_METHOD("set_refuse_new_connections", "refuse"), &MultiplayerAPI::set_refuse_new_connections);
	ClassDB::bind_method(D_METHOD("is_refusing_new_connections"), &MultiplayerAPI::is_refusing_new_connections);
	ClassDB::bind_method(D_METHOD("set_allow_object_decoding", "enable"), &MultiplayerAPI::set_allow_object_decoding);
	ClassDB::bind_method(D_METHOD("is_object_decoding_allowed"), &MultiplayerAPI::is_object_decoding_allowed);
	ClassDB::bind_method(D_METHOD("get_replicator"), &MultiplayerAPI::get_replicator);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_object_decoding"), "set_allow_object_decoding", "is_object_decoding_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_connections"), "set_refuse_new_connections", "is_refusing_new_connections");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "multiplayer_peer", PROPERTY_HINT_RESOURCE_TYPE, "MultiplayerPeer", PROPERTY_USAGE_NONE), "set_multiplayer_peer", "get_multiplayer_peer");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "root_node", PROPERTY_HINT_RESOURCE_TYPE, "Node", PROPERTY_USAGE_NONE), "set_root_node", "get_root_node");
	ADD_PROPERTY_DEFAULT("refuse_new_connections", false);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "replicator", PROPERTY_HINT_RESOURCE_TYPE, "MultiplayerReplicator", PROPERTY_USAGE_NONE), "", "get_replicator");

	ADD_SIGNAL(MethodInfo("peer_connected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_disconnected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_packet", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "packet")));
	ADD_SIGNAL(MethodInfo("connected_to_server"));
	ADD_SIGNAL(MethodInfo("connection_failed"));
	ADD_SIGNAL(MethodInfo("server_disconnected"));
}

MultiplayerAPI::MultiplayerAPI() {
	replicator = memnew(MultiplayerReplicator(this));
	rpc_manager = memnew(RPCManager(this));
	clear();
}

MultiplayerAPI::~MultiplayerAPI() {
	clear();
	memdelete(replicator);
	memdelete(rpc_manager);
}
