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

#include <stdint.h>

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#endif

MultiplayerReplicationInterface *(*MultiplayerAPI::create_default_replication_interface)(MultiplayerAPI *p_multiplayer) = nullptr;
MultiplayerRPCInterface *(*MultiplayerAPI::create_default_rpc_interface)(MultiplayerAPI *p_multiplayer) = nullptr;
MultiplayerCacheInterface *(*MultiplayerAPI::create_default_cache_interface)(MultiplayerAPI *p_multiplayer) = nullptr;

#ifdef DEBUG_ENABLED
void MultiplayerAPI::profile_bandwidth(const String &p_inout, int p_size) {
	if (EngineDebugger::is_profiling("multiplayer")) {
		Array values;
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
			return; // Something is wrong!
		}

		remote_sender_id = sender;
		_process_packet(sender, packet, len);
		remote_sender_id = 0;

		if (!multiplayer_peer.is_valid()) {
			return; // It's also possible that a packet or RPC caused a disconnection, so also check here.
		}
	}
	replicator->on_network_process();
}

void MultiplayerAPI::clear() {
	connected_peers.clear();
	packet_cache.clear();
	cache->clear();
}

void MultiplayerAPI::set_root_path(const NodePath &p_path) {
	ERR_FAIL_COND_MSG(!p_path.is_absolute() && !p_path.is_empty(), "MultiplayerAPI root path must be absolute.");
	root_path = p_path;
}

NodePath MultiplayerAPI::get_root_path() const {
	return root_path;
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
	replicator->on_reset();
}

Ref<MultiplayerPeer> MultiplayerAPI::get_multiplayer_peer() const {
	return multiplayer_peer;
}

void MultiplayerAPI::_process_packet(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(root_path.is_empty(), "Multiplayer root was not initialized. If you are using custom multiplayer, remember to set the root path via MultiplayerAPI.set_root_path before using it.");
	ERR_FAIL_COND_MSG(p_packet_len < 1, "Invalid packet received. Size too small.");

#ifdef DEBUG_ENABLED
	profile_bandwidth("in", p_packet_len);
#endif

	// Extract the `packet_type` from the LSB three bits:
	uint8_t packet_type = p_packet[0] & CMD_MASK;

	switch (packet_type) {
		case NETWORK_COMMAND_SIMPLIFY_PATH: {
			cache->process_simplify_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_CONFIRM_PATH: {
			cache->process_confirm_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_REMOTE_CALL: {
			rpc->process_rpc(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_RAW: {
			_process_raw(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_SPAWN: {
			replicator->on_spawn_receive(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_DESPAWN: {
			replicator->on_despawn_receive(p_from, p_packet, p_packet_len);
		} break;
		case NETWORK_COMMAND_SYNC: {
			replicator->on_sync_receive(p_from, p_packet, p_packet_len);
		} break;
	}
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
Error MultiplayerAPI::encode_and_compress_variant(const Variant &p_variant, uint8_t *r_buffer, int &r_len, bool p_allow_object_decoding) {
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
			Error err = encode_variant(p_variant, r_buffer, r_len, p_allow_object_decoding);
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

Error MultiplayerAPI::decode_and_decompress_variant(Variant &r_variant, const uint8_t *p_buffer, int p_len, int *r_len, bool p_allow_object_decoding) {
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
			Error err = decode_variant(r_variant, p_buffer, p_len, r_len, p_allow_object_decoding);
			if (err != OK) {
				return err;
			}
	}

	return OK;
}

Error MultiplayerAPI::encode_and_compress_variants(const Variant **p_variants, int p_count, uint8_t *p_buffer, int &r_len, bool *r_raw, bool p_allow_object_decoding) {
	r_len = 0;
	int size = 0;

	if (p_count == 0) {
		if (r_raw) {
			*r_raw = true;
		}
		return OK;
	}

	// Try raw encoding optimization.
	if (r_raw && p_count == 1) {
		*r_raw = false;
		const Variant &v = *(p_variants[0]);
		if (v.get_type() == Variant::PACKED_BYTE_ARRAY) {
			*r_raw = true;
			const PackedByteArray pba = v;
			if (p_buffer) {
				memcpy(p_buffer, pba.ptr(), pba.size());
			}
			r_len += pba.size();
		} else {
			encode_and_compress_variant(v, p_buffer, size, p_allow_object_decoding);
			r_len += size;
		}
		return OK;
	}

	// Regular encoding.
	for (int i = 0; i < p_count; i++) {
		const Variant &v = *(p_variants[i]);
		encode_and_compress_variant(v, p_buffer ? p_buffer + r_len : nullptr, size, p_allow_object_decoding);
		r_len += size;
	}
	return OK;
}

Error MultiplayerAPI::decode_and_decompress_variants(Vector<Variant> &r_variants, const uint8_t *p_buffer, int p_len, int &r_len, bool p_raw, bool p_allow_object_decoding) {
	r_len = 0;
	int argc = r_variants.size();
	if (argc == 0 && p_raw) {
		return OK;
	}
	ERR_FAIL_COND_V(p_raw && argc != 1, ERR_INVALID_DATA);
	if (p_raw) {
		r_len = p_len;
		PackedByteArray pba;
		pba.resize(p_len);
		memcpy(pba.ptrw(), p_buffer, p_len);
		r_variants.write[0] = pba;
		return OK;
	}

	Vector<Variant> args;
	Vector<const Variant *> argp;
	args.resize(argc);

	for (int i = 0; i < argc; i++) {
		ERR_FAIL_COND_V_MSG(r_len >= p_len, ERR_INVALID_DATA, "Invalid packet received. Size too small.");

		int vlen;
		Error err = MultiplayerAPI::decode_and_decompress_variant(r_variants.write[i], &p_buffer[r_len], p_len - r_len, &vlen, p_allow_object_decoding);
		ERR_FAIL_COND_V_MSG(err != OK, err, "Invalid packet received. Unable to decode state variable.");
		r_len += vlen;
	}
	return OK;
}

void MultiplayerAPI::_add_peer(int p_id) {
	connected_peers.insert(p_id);
	cache->on_peer_change(p_id, true);
	replicator->on_peer_change(p_id, true);
	emit_signal(SNAME("peer_connected"), p_id);
}

void MultiplayerAPI::_del_peer(int p_id) {
	replicator->on_peer_change(p_id, false);
	cache->on_peer_change(p_id, false);
	connected_peers.erase(p_id);
	emit_signal(SNAME("peer_disconnected"), p_id);
}

void MultiplayerAPI::_connected_to_server() {
	emit_signal(SNAME("connected_to_server"));
}

void MultiplayerAPI::_connection_failed() {
	emit_signal(SNAME("connection_failed"));
}

void MultiplayerAPI::_server_disconnected() {
	replicator->on_reset();
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
	return cache->is_cache_confirmed(p_path, p_peer);
}

bool MultiplayerAPI::send_object_cache(Object *p_obj, NodePath p_path, int p_peer_id, int &r_id) {
	return cache->send_object_cache(p_obj, p_path, p_peer_id, r_id);
}

Object *MultiplayerAPI::get_cached_object(int p_from, uint32_t p_cache_id) {
	return cache->get_cached_object(p_from, p_cache_id);
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
	for (const int &E : connected_peers) {
		ret.push_back(E);
	}

	return ret;
}

void MultiplayerAPI::set_allow_object_decoding(bool p_enable) {
	allow_object_decoding = p_enable;
}

bool MultiplayerAPI::is_object_decoding_allowed() const {
	return allow_object_decoding;
}

String MultiplayerAPI::get_rpc_md5(const Object *p_obj) const {
	return rpc->get_rpc_md5(p_obj);
}

void MultiplayerAPI::rpcp(Object *p_obj, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount) {
	rpc->rpcp(p_obj, p_peer_id, p_method, p_arg, p_argcount);
}

Error MultiplayerAPI::spawn(Object *p_object, Variant p_config) {
	return replicator->on_spawn(p_object, p_config);
}

Error MultiplayerAPI::despawn(Object *p_object, Variant p_config) {
	return replicator->on_despawn(p_object, p_config);
}

Error MultiplayerAPI::replication_start(Object *p_object, Variant p_config) {
	return replicator->on_replication_start(p_object, p_config);
}

Error MultiplayerAPI::replication_stop(Object *p_object, Variant p_config) {
	return replicator->on_replication_stop(p_object, p_config);
}

void MultiplayerAPI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &MultiplayerAPI::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &MultiplayerAPI::get_root_path);
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

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_object_decoding"), "set_allow_object_decoding", "is_object_decoding_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_connections"), "set_refuse_new_connections", "is_refusing_new_connections");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "multiplayer_peer", PROPERTY_HINT_RESOURCE_TYPE, "MultiplayerPeer", PROPERTY_USAGE_NONE), "set_multiplayer_peer", "get_multiplayer_peer");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");
	ADD_PROPERTY_DEFAULT("refuse_new_connections", false);

	ADD_SIGNAL(MethodInfo("peer_connected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_disconnected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_packet", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "packet")));
	ADD_SIGNAL(MethodInfo("connected_to_server"));
	ADD_SIGNAL(MethodInfo("connection_failed"));
	ADD_SIGNAL(MethodInfo("server_disconnected"));
}

MultiplayerAPI::MultiplayerAPI() {
	if (create_default_replication_interface) {
		replicator = Ref<MultiplayerReplicationInterface>(create_default_replication_interface(this));
	} else {
		replicator.instantiate();
	}
	if (create_default_rpc_interface) {
		rpc = Ref<MultiplayerRPCInterface>(create_default_rpc_interface(this));
	} else {
		rpc.instantiate();
	}
	if (create_default_cache_interface) {
		cache = Ref<MultiplayerCacheInterface>(create_default_cache_interface(this));
	} else {
		cache.instantiate();
	}
}

MultiplayerAPI::~MultiplayerAPI() {
	clear();
}
