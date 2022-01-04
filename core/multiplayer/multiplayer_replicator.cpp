/*************************************************************************/
/*  multiplayer_replicator.cpp                                           */
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

#include "core/multiplayer/multiplayer_replicator.h"

#include "core/io/marshalls.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"

#define MAKE_ROOM(m_amount)             \
	if (packet_cache.size() < m_amount) \
		packet_cache.resize(m_amount);

Error MultiplayerReplicator::_sync_all_default(const ResourceUID::ID &p_scene_id, int p_peer) {
	ERR_FAIL_COND_V(!replications.has(p_scene_id), ERR_INVALID_PARAMETER);
	SceneConfig &cfg = replications[p_scene_id];
	int full_size = 0;
	bool same_size = true;
	int last_size = 0;
	bool all_raw = true;
	struct EncodeInfo {
		int size = 0;
		bool raw = false;
		List<Variant> state;
	};
	Map<ObjectID, struct EncodeInfo> state;
	if (tracked_objects.has(p_scene_id)) {
		for (const ObjectID &obj_id : tracked_objects[p_scene_id]) {
			Object *obj = ObjectDB::get_instance(obj_id);
			if (obj) {
				struct EncodeInfo info;
				Error err = _get_state(cfg.sync_properties, obj, info.state);
				ERR_CONTINUE(err);
				err = _encode_state(info.state, nullptr, info.size, &info.raw);
				ERR_CONTINUE(err);
				state[obj_id] = info;
				full_size += info.size;
				if (last_size && info.size != last_size) {
					same_size = false;
				}
				all_raw = all_raw && info.raw;
				last_size = info.size;
			}
		}
	}
	// Default implementation do not send empty updates.
	if (!full_size) {
		return OK;
	}
#ifdef DEBUG_ENABLED
	if (full_size > 4096 && cfg.sync_interval) {
		WARN_PRINT_ONCE(vformat("The timed state update for scene %d is big (%d bytes) consider optimizing it", p_scene_id));
	}
#endif
	if (same_size) {
		// This is fast and small. Should we allow more than 256 objects per type?
		// This costs us 1 byte.
		MAKE_ROOM(SYNC_CMD_OFFSET + 1 + 2 + 2 + full_size);
	} else {
		MAKE_ROOM(SYNC_CMD_OFFSET + 1 + 2 + state.size() * 2 + full_size);
	}
	int ofs = 0;
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = MultiplayerAPI::NETWORK_COMMAND_SYNC | (same_size ? BYTE_OR_ZERO_FLAG : 0);
	ofs = 1;
	ofs += encode_uint64(p_scene_id, &ptr[ofs]);
	ptr[ofs] = cfg.sync_recv++;
	ofs += 1;
	ofs += encode_uint16(state.size(), &ptr[ofs]);
	if (same_size) {
		ofs += encode_uint16(last_size + (all_raw ? 1 << 15 : 0), &ptr[ofs]);
	}
	for (const ObjectID &obj_id : tracked_objects[p_scene_id]) {
		if (!state.has(obj_id)) {
			continue;
		}
		struct EncodeInfo &info = state[obj_id];
		Object *obj = ObjectDB::get_instance(obj_id);
		ERR_CONTINUE(!obj);
		int size = 0;
		if (!same_size) {
			// We need to encode the size of every object.
			ofs += encode_uint16(info.size + (info.raw ? 1 << 15 : 0), &ptr[ofs]);
		}
		Error err = _encode_state(info.state, &ptr[ofs], size, &info.raw);
		ERR_CONTINUE(err);
		ofs += size;
	}
	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	peer->set_target_peer(p_peer);
	peer->set_transfer_channel(0);
	peer->set_transfer_mode(Multiplayer::TRANSFER_MODE_UNRELIABLE);
	return peer->put_packet(ptr, ofs);
}

void MultiplayerReplicator::_process_default_sync(const ResourceUID::ID &p_id, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(p_packet_len < SYNC_CMD_OFFSET + 5, "Invalid spawn packet received");
	ERR_FAIL_COND_MSG(!replications.has(p_id), "Invalid spawn ID received " + itos(p_id));
	SceneConfig &cfg = replications[p_id];
	ERR_FAIL_COND_MSG(cfg.mode != REPLICATION_MODE_SERVER || multiplayer->is_server(), "The default implementation only allows sync packets from the server");
	const bool same_size = p_packet[0] & BYTE_OR_ZERO_FLAG;
	int ofs = SYNC_CMD_OFFSET;
	int time = p_packet[ofs];
	// Skip old update.
	if (time < cfg.sync_recv && cfg.sync_recv - time < 127) {
		return;
	}
	cfg.sync_recv = time;
	ofs += 1;
	int count = decode_uint16(&p_packet[ofs]);
	ofs += 2;
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!tracked_objects.has(p_id) || tracked_objects[p_id].size() != count);
#else
	if (!tracked_objects.has(p_id) || tracked_objects[p_id].size() != count) {
		return;
	}
#endif
	int data_size = 0;
	bool raw = false;
	if (same_size) {
		// This is fast and optimized.
		data_size = decode_uint16(&p_packet[ofs]);
		raw = (data_size & (1 << 15)) != 0;
		data_size = data_size & ~(1 << 15);
		ofs += 2;
		ERR_FAIL_COND(p_packet_len - ofs < data_size * count);
	}
	for (const ObjectID &obj_id : tracked_objects[p_id]) {
		Object *obj = ObjectDB::get_instance(obj_id);
		ERR_CONTINUE(!obj);
		if (!same_size) {
			// This is slow and wasteful.
			data_size = decode_uint16(&p_packet[ofs]);
			raw = (data_size & (1 << 15)) != 0;
			data_size = data_size & ~(1 << 15);
			ofs += 2;
			ERR_FAIL_COND(p_packet_len - ofs < data_size);
		}
		int size = 0;
		Error err = _decode_state(cfg.sync_properties, obj, &p_packet[ofs], data_size, size, raw);
		ofs += data_size;
		ERR_CONTINUE(err);
		ERR_CONTINUE(size != data_size);
	}
}

Error MultiplayerReplicator::_send_default_spawn_despawn(int p_peer_id, const ResourceUID::ID &p_scene_id, Object *p_obj, const NodePath &p_path, bool p_spawn) {
	ERR_FAIL_COND_V(p_spawn && !p_obj, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!replications.has(p_scene_id), ERR_INVALID_PARAMETER);
	Error err;
	// Prepare state
	List<Variant> state_variants;
	int state_len = 0;
	const SceneConfig &cfg = replications[p_scene_id];
	if (p_spawn) {
		if ((err = _get_state(cfg.properties, p_obj, state_variants)) != OK) {
			return err;
		}
	}

	bool is_raw = false;
	if (state_variants.size() == 1 && state_variants[0].get_type() == Variant::PACKED_BYTE_ARRAY) {
		is_raw = true;
		const PackedByteArray pba = state_variants[0];
		state_len = pba.size();
	} else if (state_variants.size()) {
		err = _encode_state(state_variants, nullptr, state_len);
		ERR_FAIL_COND_V(err, err);
	} else {
		is_raw = true;
	}

	int ofs = 0;

	// Prepare simplified path
	const Node *root_node = multiplayer->get_root_node();
	ERR_FAIL_COND_V(!root_node, ERR_UNCONFIGURED);
	NodePath rel_path = (root_node->get_path()).rel_path_to(p_path);
	const Vector<StringName> names = rel_path.get_names();
	ERR_FAIL_COND_V(names.size() < 2, ERR_INVALID_PARAMETER);

	NodePath parent = NodePath(names.slice(0, names.size() - 1), false);
	ERR_FAIL_COND_V_MSG(!root_node->has_node(parent), ERR_INVALID_PARAMETER, "Path not found: " + parent);

	int path_id = 0;
	multiplayer->send_confirm_path(root_node->get_node(parent), parent, p_peer_id, path_id);

	// Encode name and parent ID.
	CharString cname = String(names[names.size() - 1]).utf8();
	int nlen = encode_cstring(cname.get_data(), nullptr);
	MAKE_ROOM(SPAWN_CMD_OFFSET + 4 + 4 + nlen + state_len);
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = (p_spawn ? MultiplayerAPI::NETWORK_COMMAND_SPAWN : MultiplayerAPI::NETWORK_COMMAND_DESPAWN) | (is_raw ? BYTE_OR_ZERO_FLAG : 0);
	ofs = 1;
	ofs += encode_uint64(p_scene_id, &ptr[ofs]);
	ofs += encode_uint32(path_id, &ptr[ofs]);
	ofs += encode_uint32(nlen, &ptr[ofs]);
	ofs += encode_cstring(cname.get_data(), &ptr[ofs]);

	// Encode state.
	if (!is_raw) {
		_encode_state(state_variants, &ptr[ofs], state_len);
	} else if (state_len) {
		PackedByteArray pba = state_variants[0];
		memcpy(&ptr[ofs], pba.ptr(), state_len);
	}

	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	peer->set_target_peer(p_peer_id);
	peer->set_transfer_channel(0);
	peer->set_transfer_mode(Multiplayer::TRANSFER_MODE_RELIABLE);
	return peer->put_packet(ptr, ofs + state_len);
}

void MultiplayerReplicator::_process_default_spawn_despawn(int p_from, const ResourceUID::ID &p_scene_id, const uint8_t *p_packet, int p_packet_len, bool p_spawn) {
	ERR_FAIL_COND_MSG(p_packet_len < SPAWN_CMD_OFFSET + 9, "Invalid spawn packet received");
	int ofs = SPAWN_CMD_OFFSET;
	uint32_t node_target = decode_uint32(&p_packet[ofs]);
	Node *parent = multiplayer->get_cached_node(p_from, node_target);
	ofs += 4;
	ERR_FAIL_COND_MSG(parent == nullptr, "Invalid packet received. Requested node was not found.");

	uint32_t name_len = decode_uint32(&p_packet[ofs]);
	ofs += 4;
	ERR_FAIL_COND_MSG(name_len > uint32_t(p_packet_len - ofs), vformat("Invalid spawn packet size: %d, wants: %d", p_packet_len, ofs + name_len));
	ERR_FAIL_COND_MSG(name_len < 1, "Zero spawn name size.");

	const String name = String::utf8((const char *)&p_packet[ofs], name_len);
	// We need to make sure no trickery happens here (e.g. despawning a subpath), but we want to allow autogenerated ("@") node names.
	ERR_FAIL_COND_MSG(name.validate_node_name() != name.replace("@", ""), vformat("Invalid node name received: '%s'", name));
	ofs += name_len;

	const SceneConfig &cfg = replications[p_scene_id];
	if (cfg.mode == REPLICATION_MODE_SERVER && p_from == 1) {
		String scene_path = ResourceUID::get_singleton()->get_id_path(p_scene_id);
		if (p_spawn) {
			const bool is_raw = ((p_packet[0] & BYTE_OR_ZERO_FLAG) >> BYTE_OR_ZERO_SHIFT) == 1;

			ERR_FAIL_COND_MSG(parent->has_node(name), vformat("Unable to spawn node. Node already exists: %s/%s", parent->get_path(), name));
			RES res = ResourceLoader::load(scene_path);
			ERR_FAIL_COND_MSG(!res.is_valid(), "Unable to load scene to spawn at path: " + scene_path);
			PackedScene *scene = Object::cast_to<PackedScene>(res.ptr());
			ERR_FAIL_COND(!scene);
			Node *node = scene->instantiate();
			ERR_FAIL_COND(!node);
			replicated_nodes[node->get_instance_id()] = p_scene_id;
			_track(p_scene_id, node);
			int size;
			_decode_state(cfg.properties, node, &p_packet[ofs], p_packet_len - ofs, size, is_raw);
			parent->_add_child_nocheck(node, name);
			emit_signal(SNAME("spawned"), p_scene_id, node);
		} else {
			ERR_FAIL_COND_MSG(!parent->has_node(name), vformat("Path not found: %s/%s", parent->get_path(), name));
			Node *node = parent->get_node(name);
			ERR_FAIL_COND_MSG(!replicated_nodes.has(node->get_instance_id()), vformat("Trying to despawn a Node that was not replicated: %s/%s", parent->get_path(), name));
			emit_signal(SNAME("despawned"), p_scene_id, node);
			_untrack(p_scene_id, node);
			replicated_nodes.erase(node->get_instance_id());
			node->queue_delete();
		}
	} else {
		PackedByteArray data;
		if (p_packet_len > ofs) {
			data.resize(p_packet_len - ofs);
			memcpy(data.ptrw(), &p_packet[ofs], data.size());
		}
		if (p_spawn) {
			emit_signal(SNAME("spawn_requested"), p_from, p_scene_id, parent, name, data);
		} else {
			emit_signal(SNAME("despawn_requested"), p_from, p_scene_id, parent, name, data);
		}
	}
}

void MultiplayerReplicator::process_spawn_despawn(int p_from, const uint8_t *p_packet, int p_packet_len, bool p_spawn) {
	ERR_FAIL_COND_MSG(p_packet_len < SPAWN_CMD_OFFSET, "Invalid spawn packet received");
	ResourceUID::ID id = decode_uint64(&p_packet[1]);
	ERR_FAIL_COND_MSG(!replications.has(id), "Invalid spawn ID received " + itos(id));

	const SceneConfig &cfg = replications[id];
	if (cfg.on_spawn_despawn_receive.is_valid()) {
		int ofs = SPAWN_CMD_OFFSET;
		bool is_raw = ((p_packet[0] & BYTE_OR_ZERO_FLAG) >> BYTE_OR_ZERO_SHIFT) == 1;
		Variant data;
		int left = p_packet_len - ofs;
		if (is_raw && left) {
			PackedByteArray pba;
			pba.resize(left);
			memcpy(pba.ptrw(), &p_packet[ofs], pba.size());
			data = pba;
		} else if (left) {
			ERR_FAIL_COND(decode_variant(data, &p_packet[ofs], left) != OK);
		}

		Variant args[4];
		args[0] = p_from;
		args[1] = id;
		args[2] = data;
		args[3] = p_spawn;
		const Variant *argp[] = { &args[0], &args[1], &args[2], &args[3] };
		Callable::CallError ce;
		Variant ret;
		cfg.on_spawn_despawn_receive.call(argp, 4, ret, ce);
		ERR_FAIL_COND_MSG(ce.error != Callable::CallError::CALL_OK, "Custom receive function failed");
	} else {
		_process_default_spawn_despawn(p_from, id, p_packet, p_packet_len, p_spawn);
	}
}

void MultiplayerReplicator::process_sync(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(p_packet_len < SPAWN_CMD_OFFSET, "Invalid spawn packet received");
	ResourceUID::ID id = decode_uint64(&p_packet[1]);
	ERR_FAIL_COND_MSG(!replications.has(id), "Invalid spawn ID received " + itos(id));
	const SceneConfig &cfg = replications[id];
	if (cfg.on_sync_receive.is_valid()) {
		Array objs;
		if (tracked_objects.has(id)) {
			objs.resize(tracked_objects[id].size());
			int idx = 0;
			for (const ObjectID &obj_id : tracked_objects[id]) {
				objs[idx++] = ObjectDB::get_instance(obj_id);
			}
		}
		PackedByteArray pba;
		pba.resize(p_packet_len - SYNC_CMD_OFFSET);
		if (pba.size()) {
			memcpy(pba.ptrw(), p_packet + SYNC_CMD_OFFSET, p_packet_len - SYNC_CMD_OFFSET);
		}
		Variant args[4] = { p_from, id, objs, pba };
		Variant *argp[4] = { args, &args[1], &args[2], &args[3] };
		Callable::CallError ce;
		Variant ret;
		cfg.on_sync_receive.call((const Variant **)argp, 4, ret, ce);
		ERR_FAIL_COND_MSG(ce.error != Callable::CallError::CALL_OK, "Custom sync function failed");
	} else {
		ERR_FAIL_COND_MSG(p_from != 1, "Default sync implementation only allow syncing from server to client");
		_process_default_sync(id, p_packet, p_packet_len);
	}
}

Error MultiplayerReplicator::_get_state(const List<StringName> &p_properties, const Object *p_obj, List<Variant> &r_variant) {
	ERR_FAIL_COND_V_MSG(!p_obj, ERR_INVALID_PARAMETER, "Cannot encode null object");
	for (const StringName &prop : p_properties) {
		bool valid = false;
		const Variant v = p_obj->get(prop, &valid);
		ERR_FAIL_COND_V_MSG(!valid, ERR_INVALID_DATA, vformat("Property '%s' not found.", prop));
		r_variant.push_back(v);
	}
	return OK;
}

Error MultiplayerReplicator::_encode_state(const List<Variant> &p_variants, uint8_t *p_buffer, int &r_len, bool *r_raw) {
	r_len = 0;
	int size = 0;

	// Try raw encoding optimization.
	if (r_raw && p_variants.size() == 1) {
		*r_raw = false;
		const Variant v = p_variants[0];
		if (v.get_type() == Variant::PACKED_BYTE_ARRAY) {
			*r_raw = true;
			const PackedByteArray pba = v;
			if (p_buffer) {
				memcpy(p_buffer, pba.ptr(), pba.size());
			}
			r_len += pba.size();
		} else {
			multiplayer->encode_and_compress_variant(v, p_buffer, size);
			r_len += size;
		}
		return OK;
	}

	// Regular encoding.
	for (const Variant &v : p_variants) {
		multiplayer->encode_and_compress_variant(v, p_buffer ? p_buffer + r_len : nullptr, size);
		r_len += size;
	}
	return OK;
}

Error MultiplayerReplicator::_decode_state(const List<StringName> &p_properties, Object *p_obj, const uint8_t *p_buffer, int p_len, int &r_len, bool p_raw) {
	r_len = 0;
	int argc = p_properties.size();
	if (argc == 0 && p_raw) {
		ERR_FAIL_COND_V_MSG(p_len != 0, ERR_INVALID_DATA, "Buffer has trailing bytes.");
		return OK;
	}
	ERR_FAIL_COND_V(p_raw && argc != 1, ERR_INVALID_DATA);
	if (p_raw) {
		r_len = p_len;
		PackedByteArray pba;
		pba.resize(p_len);
		memcpy(pba.ptrw(), p_buffer, p_len);
		p_obj->set(p_properties[0], pba);
		return OK;
	}

	Vector<Variant> args;
	Vector<const Variant *> argp;
	args.resize(argc);

	for (int i = 0; i < argc; i++) {
		ERR_FAIL_COND_V_MSG(r_len >= p_len, ERR_INVALID_DATA, "Invalid packet received. Size too small.");

		int vlen;
		Error err = multiplayer->decode_and_decompress_variant(args.write[i], &p_buffer[r_len], p_len - r_len, &vlen);
		ERR_FAIL_COND_V_MSG(err != OK, err, "Invalid packet received. Unable to decode state variable.");
		r_len += vlen;
	}
	ERR_FAIL_COND_V_MSG(p_len - r_len != 0, ERR_INVALID_DATA, "Buffer has trailing bytes.");

	int i = 0;
	for (const StringName &prop : p_properties) {
		p_obj->set(prop, args[i]);
		i += 1;
	}
	return OK;
}

Error MultiplayerReplicator::spawn_config(const ResourceUID::ID &p_id, ReplicationMode p_mode, const TypedArray<StringName> &p_props, const Callable &p_on_send, const Callable &p_on_recv) {
	ERR_FAIL_COND_V(p_mode < REPLICATION_MODE_NONE || p_mode > REPLICATION_MODE_CUSTOM, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!ResourceUID::get_singleton()->has_id(p_id), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(p_on_send.is_valid() != p_on_recv.is_valid(), ERR_INVALID_PARAMETER, "Send and receive custom callables must be both valid or both empty");
#ifdef TOOLS_ENABLED
	if (!p_on_send.is_valid()) {
		// We allow non scene spawning with custom callables.
		String path = ResourceUID::get_singleton()->get_id_path(p_id);
		RES res = ResourceLoader::load(path);
		ERR_FAIL_COND_V(!res->is_class("PackedScene"), ERR_INVALID_PARAMETER);
	}
#endif
	if (p_mode == REPLICATION_MODE_NONE) {
		if (replications.has(p_id)) {
			replications.erase(p_id);
		}
	} else {
		SceneConfig cfg;
		cfg.mode = p_mode;
		for (int i = 0; i < p_props.size(); i++) {
			cfg.properties.push_back(p_props[i]);
		}
		cfg.on_spawn_despawn_send = p_on_send;
		cfg.on_spawn_despawn_receive = p_on_recv;
		replications[p_id] = cfg;
	}
	return OK;
}

Error MultiplayerReplicator::sync_config(const ResourceUID::ID &p_id, uint64_t p_interval, const TypedArray<StringName> &p_props, const Callable &p_on_send, const Callable &p_on_recv) {
	ERR_FAIL_COND_V(!ResourceUID::get_singleton()->has_id(p_id), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(p_on_send.is_valid() != p_on_recv.is_valid(), ERR_INVALID_PARAMETER, "Send and receive custom callables must be both valid or both empty");
	ERR_FAIL_COND_V(!replications.has(p_id), ERR_UNCONFIGURED);
	SceneConfig &cfg = replications[p_id];
	ERR_FAIL_COND_V_MSG(p_interval && cfg.mode != REPLICATION_MODE_SERVER && !p_on_send.is_valid(), ERR_INVALID_PARAMETER, "Timed updates in custom mode are only allowed if custom callbacks are also specified");
	for (int i = 0; i < p_props.size(); i++) {
		cfg.sync_properties.push_back(p_props[i]);
	}
	cfg.on_sync_send = p_on_send;
	cfg.on_sync_receive = p_on_recv;
	cfg.sync_interval = p_interval * 1000;
	return OK;
}

Error MultiplayerReplicator::_send_spawn_despawn(int p_peer_id, const ResourceUID::ID &p_scene_id, const Variant &p_data, bool p_spawn) {
	int data_size = 0;
	int is_raw = false;
	if (p_data.get_type() == Variant::PACKED_BYTE_ARRAY) {
		const PackedByteArray pba = p_data;
		is_raw = true;
		data_size = p_data.operator PackedByteArray().size();
	} else if (p_data.get_type() == Variant::NIL) {
		is_raw = true;
	} else {
		Error err = encode_variant(p_data, nullptr, data_size);
		ERR_FAIL_COND_V(err, err);
	}
	MAKE_ROOM(SPAWN_CMD_OFFSET + data_size);
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = (p_spawn ? MultiplayerAPI::NETWORK_COMMAND_SPAWN : MultiplayerAPI::NETWORK_COMMAND_DESPAWN) + ((is_raw ? 1 : 0) << BYTE_OR_ZERO_SHIFT);
	encode_uint64(p_scene_id, &ptr[1]);
	if (p_data.get_type() == Variant::PACKED_BYTE_ARRAY) {
		const PackedByteArray pba = p_data;
		memcpy(&ptr[SPAWN_CMD_OFFSET], pba.ptr(), pba.size());
	} else if (data_size) {
		encode_variant(p_data, &ptr[SPAWN_CMD_OFFSET], data_size);
	}
	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	peer->set_target_peer(p_peer_id);
	peer->set_transfer_channel(0);
	peer->set_transfer_mode(Multiplayer::TRANSFER_MODE_RELIABLE);
	return peer->put_packet(ptr, SPAWN_CMD_OFFSET + data_size);
}

Error MultiplayerReplicator::send_despawn(int p_peer_id, const ResourceUID::ID &p_scene_id, const Variant &p_data, const NodePath &p_path) {
	ERR_FAIL_COND_V(!multiplayer->has_multiplayer_peer(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V_MSG(!replications.has(p_scene_id), ERR_INVALID_PARAMETER, vformat("Spawnable not found: %d", p_scene_id));
	const SceneConfig &cfg = replications[p_scene_id];
	if (cfg.on_spawn_despawn_send.is_valid()) {
		return _send_spawn_despawn(p_peer_id, p_scene_id, p_data, true);
	} else {
		ERR_FAIL_COND_V_MSG(cfg.mode == REPLICATION_MODE_SERVER && multiplayer->is_server(), ERR_UNAVAILABLE, "Manual despawn is restricted in default server mode implementation. Use custom mode if you desire control over server spawn requests.");
		NodePath path = p_path;
		Object *obj = p_data.get_type() == Variant::OBJECT ? p_data.get_validated_object() : nullptr;
		if (path.is_empty() && obj) {
			Node *node = Object::cast_to<Node>(obj);
			if (node && node->is_inside_tree()) {
				path = node->get_path();
			}
		}
		ERR_FAIL_COND_V_MSG(path.is_empty(), ERR_INVALID_PARAMETER, "Despawn default implementation requires a despawn path, or the data to be a node inside the SceneTree");
		return _send_default_spawn_despawn(p_peer_id, p_scene_id, obj, path, false);
	}
}

Error MultiplayerReplicator::send_spawn(int p_peer_id, const ResourceUID::ID &p_scene_id, const Variant &p_data, const NodePath &p_path) {
	ERR_FAIL_COND_V(!multiplayer->has_multiplayer_peer(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V_MSG(!replications.has(p_scene_id), ERR_INVALID_PARAMETER, vformat("Spawnable not found: %d", p_scene_id));
	const SceneConfig &cfg = replications[p_scene_id];
	if (cfg.on_spawn_despawn_send.is_valid()) {
		return _send_spawn_despawn(p_peer_id, p_scene_id, p_data, false);
	} else {
		ERR_FAIL_COND_V_MSG(cfg.mode == REPLICATION_MODE_SERVER && multiplayer->is_server(), ERR_UNAVAILABLE, "Manual spawn is restricted in default server mode implementation. Use custom mode if you desire control over server spawn requests.");
		NodePath path = p_path;
		Object *obj = p_data.get_type() == Variant::OBJECT ? p_data.get_validated_object() : nullptr;
		ERR_FAIL_COND_V_MSG(!obj, ERR_INVALID_PARAMETER, "Spawn default implementation requires the data to be an object.");
		if (path.is_empty()) {
			Node *node = Object::cast_to<Node>(obj);
			if (node && node->is_inside_tree()) {
				path = node->get_path();
			}
		}
		ERR_FAIL_COND_V_MSG(path.is_empty(), ERR_INVALID_PARAMETER, "Spawn default implementation requires a spawn path, or the data to be a node inside the SceneTree");
		return _send_default_spawn_despawn(p_peer_id, p_scene_id, obj, path, true);
	}
}

Error MultiplayerReplicator::_spawn_despawn(ResourceUID::ID p_scene_id, Object *p_obj, int p_peer, bool p_spawn) {
	ERR_FAIL_COND_V_MSG(!replications.has(p_scene_id), ERR_INVALID_PARAMETER, vformat("Spawnable not found: %d", p_scene_id));

	const SceneConfig &cfg = replications[p_scene_id];
	if (cfg.on_spawn_despawn_send.is_valid()) {
		Variant args[4];
		args[0] = p_peer;
		args[1] = p_scene_id;
		args[2] = p_obj;
		args[3] = p_spawn;
		const Variant *argp[] = { &args[0], &args[1], &args[2], &args[3] };
		Callable::CallError ce;
		Variant ret;
		cfg.on_spawn_despawn_send.call(argp, 4, ret, ce);
		ERR_FAIL_COND_V_MSG(ce.error != Callable::CallError::CALL_OK, FAILED, "Custom send function failed");
		return OK;
	} else {
		Node *node = Object::cast_to<Node>(p_obj);
		ERR_FAIL_COND_V_MSG(!p_obj, ERR_INVALID_PARAMETER, "Only nodes can be replicated by the default implementation");
		return _send_default_spawn_despawn(p_peer, p_scene_id, node, node->get_path(), p_spawn);
	}
}

Error MultiplayerReplicator::spawn(ResourceUID::ID p_scene_id, Object *p_obj, int p_peer) {
	return _spawn_despawn(p_scene_id, p_obj, p_peer, true);
}

Error MultiplayerReplicator::despawn(ResourceUID::ID p_scene_id, Object *p_obj, int p_peer) {
	return _spawn_despawn(p_scene_id, p_obj, p_peer, false);
}

PackedByteArray MultiplayerReplicator::encode_state(const ResourceUID::ID &p_scene_id, const Object *p_obj, bool p_initial) {
	PackedByteArray state;
	ERR_FAIL_COND_V_MSG(!replications.has(p_scene_id), state, vformat("Spawnable not found: %d", p_scene_id));
	const SceneConfig &cfg = replications[p_scene_id];
	int len = 0;
	List<Variant> state_vars;
	const List<StringName> props = p_initial ? cfg.properties : cfg.sync_properties;
	Error err = _get_state(props, p_obj, state_vars);
	ERR_FAIL_COND_V_MSG(err != OK, state, "Unable to retrieve object state.");
	err = _encode_state(state_vars, nullptr, len);
	ERR_FAIL_COND_V_MSG(err != OK, state, "Unable to encode object state.");
	state.resize(len);
	_encode_state(state_vars, state.ptrw(), len);
	return state;
}

Error MultiplayerReplicator::decode_state(const ResourceUID::ID &p_scene_id, Object *p_obj, const PackedByteArray p_data, bool p_initial) {
	ERR_FAIL_COND_V_MSG(!replications.has(p_scene_id), ERR_INVALID_PARAMETER, vformat("Spawnable not found: %d", p_scene_id));
	const SceneConfig &cfg = replications[p_scene_id];
	const List<StringName> props = p_initial ? cfg.properties : cfg.sync_properties;
	int size;
	return _decode_state(props, p_obj, p_data.ptr(), p_data.size(), size);
}

void MultiplayerReplicator::scene_enter_exit_notify(const String &p_scene, Node *p_node, bool p_enter) {
	if (!multiplayer->has_multiplayer_peer()) {
		return;
	}
	Node *root_node = multiplayer->get_root_node();
	ERR_FAIL_COND(!p_node || !p_node->get_parent() || !root_node);
	NodePath path = (root_node->get_path()).rel_path_to(p_node->get_parent()->get_path());
	if (path.is_empty()) {
		return;
	}
	ResourceUID::ID id = ResourceLoader::get_resource_uid(p_scene);
	if (!replications.has(id)) {
		return;
	}
	const SceneConfig &cfg = replications[id];
	if (p_enter) {
		if (cfg.mode == REPLICATION_MODE_SERVER && multiplayer->is_server()) {
			replicated_nodes[p_node->get_instance_id()] = id;
			_track(id, p_node);
			spawn(id, p_node, 0);
		}
		emit_signal(SNAME("replicated_instance_added"), id, p_node);
	} else {
		if (cfg.mode == REPLICATION_MODE_SERVER && multiplayer->is_server() && replicated_nodes.has(p_node->get_instance_id())) {
			replicated_nodes.erase(p_node->get_instance_id());
			_untrack(id, p_node);
			despawn(id, p_node, 0);
		}
		emit_signal(SNAME("replicated_instance_removed"), id, p_node);
	}
}

void MultiplayerReplicator::spawn_all(int p_peer) {
	for (const KeyValue<ObjectID, ResourceUID::ID> &E : replicated_nodes) {
		// Only server mode adds to replicated_nodes, no need to check it.
		Object *obj = ObjectDB::get_instance(E.key);
		ERR_CONTINUE(!obj);
		Node *node = Object::cast_to<Node>(obj);
		ERR_CONTINUE(!node);
		spawn(E.value, node, p_peer);
	}
}

void MultiplayerReplicator::poll() {
	for (KeyValue<ResourceUID::ID, SceneConfig> &E : replications) {
		if (!E.value.sync_interval) {
			continue;
		}
		if (E.value.mode == REPLICATION_MODE_SERVER && !multiplayer->is_server()) {
			continue;
		}
		uint64_t time = OS::get_singleton()->get_ticks_usec();
		if (E.value.sync_last + E.value.sync_interval <= time) {
			sync_all(E.key, 0);
			E.value.sync_last = time;
		}
		// Handle wrapping.
		if (E.value.sync_last > time) {
			E.value.sync_last = time;
		}
	}
}

void MultiplayerReplicator::track(const ResourceUID::ID &p_scene_id, Object *p_obj) {
	ERR_FAIL_COND(!replications.has(p_scene_id));
	const SceneConfig &cfg = replications[p_scene_id];
	ERR_FAIL_COND_MSG(cfg.mode == REPLICATION_MODE_SERVER, "Manual object tracking is not allowed in server mode.");
	_track(p_scene_id, p_obj);
}

void MultiplayerReplicator::_track(const ResourceUID::ID &p_scene_id, Object *p_obj) {
	ERR_FAIL_COND(!p_obj);
	ERR_FAIL_COND(!replications.has(p_scene_id));
	if (!tracked_objects.has(p_scene_id)) {
		tracked_objects[p_scene_id] = List<ObjectID>();
	}
	tracked_objects[p_scene_id].push_back(p_obj->get_instance_id());
}

void MultiplayerReplicator::untrack(const ResourceUID::ID &p_scene_id, Object *p_obj) {
	ERR_FAIL_COND(!replications.has(p_scene_id));
	const SceneConfig &cfg = replications[p_scene_id];
	ERR_FAIL_COND_MSG(cfg.mode == REPLICATION_MODE_SERVER, "Manual object tracking is not allowed in server mode.");
	_untrack(p_scene_id, p_obj);
}

void MultiplayerReplicator::_untrack(const ResourceUID::ID &p_scene_id, Object *p_obj) {
	ERR_FAIL_COND(!p_obj);
	ERR_FAIL_COND(!replications.has(p_scene_id));
	if (tracked_objects.has(p_scene_id)) {
		tracked_objects[p_scene_id].erase(p_obj->get_instance_id());
	}
}

Error MultiplayerReplicator::sync_all(const ResourceUID::ID &p_scene_id, int p_peer) {
	ERR_FAIL_COND_V(!replications.has(p_scene_id), ERR_INVALID_PARAMETER);
	if (!tracked_objects.has(p_scene_id)) {
		return OK;
	}
	const SceneConfig &cfg = replications[p_scene_id];
	if (cfg.on_sync_send.is_valid()) {
		Array objs;
		if (tracked_objects.has(p_scene_id)) {
			objs.resize(tracked_objects[p_scene_id].size());
			int idx = 0;
			for (const ObjectID &obj_id : tracked_objects[p_scene_id]) {
				objs[idx++] = ObjectDB::get_instance(obj_id);
			}
		}
		Variant args[3] = { p_scene_id, objs, p_peer };
		Variant *argp[3] = { args, &args[1], &args[2] };
		Callable::CallError ce;
		Variant ret;
		cfg.on_sync_send.call((const Variant **)argp, 3, ret, ce);
		ERR_FAIL_COND_V_MSG(ce.error != Callable::CallError::CALL_OK, FAILED, "Custom sync function failed");
		return OK;
	} else if (cfg.sync_properties.size()) {
		return _sync_all_default(p_scene_id, p_peer);
	}
	return OK;
}

Error MultiplayerReplicator::send_sync(int p_peer_id, const ResourceUID::ID &p_scene_id, PackedByteArray p_data, Multiplayer::TransferMode p_transfer_mode, int p_channel) {
	ERR_FAIL_COND_V(!multiplayer->has_multiplayer_peer(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!replications.has(p_scene_id), ERR_INVALID_PARAMETER);
	const SceneConfig &cfg = replications[p_scene_id];
	ERR_FAIL_COND_V_MSG(!cfg.on_sync_send.is_valid(), ERR_UNCONFIGURED, "Sending raw sync messages is only available with custom functions");
	MAKE_ROOM(SYNC_CMD_OFFSET + p_data.size());
	uint8_t *ptr = packet_cache.ptrw();
	ptr[0] = MultiplayerAPI::NETWORK_COMMAND_SYNC;
	encode_uint64(p_scene_id, &ptr[1]);
	if (p_data.size()) {
		memcpy(&ptr[SYNC_CMD_OFFSET], p_data.ptr(), p_data.size());
	}
	Ref<MultiplayerPeer> peer = multiplayer->get_multiplayer_peer();
	peer->set_target_peer(p_peer_id);
	peer->set_transfer_channel(p_channel);
	peer->set_transfer_mode(p_transfer_mode);
	return peer->put_packet(ptr, SYNC_CMD_OFFSET + p_data.size());
}

void MultiplayerReplicator::clear() {
	tracked_objects.clear();
	replicated_nodes.clear();
}

void MultiplayerReplicator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("spawn_config", "scene_id", "spawn_mode", "properties", "custom_send", "custom_receive"), &MultiplayerReplicator::spawn_config, DEFVAL(TypedArray<StringName>()), DEFVAL(Callable()), DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("sync_config", "scene_id", "interval", "properties", "custom_send", "custom_receive"), &MultiplayerReplicator::sync_config, DEFVAL(TypedArray<StringName>()), DEFVAL(Callable()), DEFVAL(Callable()));
	ClassDB::bind_method(D_METHOD("despawn", "scene_id", "object", "peer_id"), &MultiplayerReplicator::despawn, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("spawn", "scene_id", "object", "peer_id"), &MultiplayerReplicator::spawn, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("send_despawn", "peer_id", "scene_id", "data", "path"), &MultiplayerReplicator::send_despawn, DEFVAL(Variant()), DEFVAL(NodePath()));
	ClassDB::bind_method(D_METHOD("send_spawn", "peer_id", "scene_id", "data", "path"), &MultiplayerReplicator::send_spawn, DEFVAL(Variant()), DEFVAL(NodePath()));
	ClassDB::bind_method(D_METHOD("send_sync", "peer_id", "scene_id", "data", "transfer_mode", "channel"), &MultiplayerReplicator::send_sync, DEFVAL(Multiplayer::TRANSFER_MODE_RELIABLE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("sync_all", "scene_id", "peer_id"), &MultiplayerReplicator::sync_all, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("track", "scene_id", "object"), &MultiplayerReplicator::track);
	ClassDB::bind_method(D_METHOD("untrack", "scene_id", "object"), &MultiplayerReplicator::untrack);
	ClassDB::bind_method(D_METHOD("encode_state", "scene_id", "object", "initial"), &MultiplayerReplicator::encode_state, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("decode_state", "scene_id", "object", "data", "initial"), &MultiplayerReplicator::decode_state, DEFVAL(true));

	ADD_SIGNAL(MethodInfo("despawned", PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("spawned", PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("despawn_requested", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "parent", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data")));
	ADD_SIGNAL(MethodInfo("spawn_requested", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "parent", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data")));
	ADD_SIGNAL(MethodInfo("replicated_instance_added", PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("replicated_instance_removed", PropertyInfo(Variant::INT, "scene_id"), PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));

	BIND_ENUM_CONSTANT(REPLICATION_MODE_NONE);
	BIND_ENUM_CONSTANT(REPLICATION_MODE_SERVER);
	BIND_ENUM_CONSTANT(REPLICATION_MODE_CUSTOM);
}
