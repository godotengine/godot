/**************************************************************************/
/*  scene_saveload.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "scene_saveload.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"

#include <stdint.h>

#ifdef DEBUG_ENABLED
#include "core/os/os.h"
#endif

#ifdef DEBUG_ENABLED
_FORCE_INLINE_ void SceneSaveload::_profile_bandwidth(const String &p_what, int p_value) {
	if (EngineDebugger::is_profiling("saveload:bandwidth")) {
		Array values;
		values.push_back(p_what);
		values.push_back(OS::get_singleton()->get_ticks_msec());
		values.push_back(p_value);
		EngineDebugger::profiler_add_frame_data("saveload:bandwidth", values);
	}
}
#endif

void SceneSaveload::clear() {
	packet_cache.clear();
	saveloader->on_reset();
	cache->clear();
//	relay_buffer->clear();
}

void SceneSaveload::set_root_path(const NodePath &p_path) {
	ERR_FAIL_COND_MSG(!p_path.is_absolute() && !p_path.is_empty(), "SceneSaveload root path must be absolute.");
	root_path = p_path;
}

NodePath SceneSaveload::get_root_path() const {
	return root_path;
}

void SceneSaveload::_process_packet(int p_from, const uint8_t *p_packet, int p_packet_len) {
	ERR_FAIL_COND_MSG(root_path.is_empty(), "Saveload root was not initialized. If you are using custom saveload, remember to set the root path via SceneSaveload.set_root_path before using it.");
	ERR_FAIL_COND_MSG(p_packet_len < 1, "Invalid packet received. Size too small.");

	// Extract the `packet_type` from the LSB three bits:
	uint8_t packet_type = p_packet[0] & CMD_MASK;

	switch (packet_type) {
		case SAVELOAD_COMMAND_SIMPLIFY_PATH: {
			cache->process_simplify_path(p_from, p_packet, p_packet_len);
		} break;

		case SAVELOAD_COMMAND_CONFIRM_PATH: {
			cache->process_confirm_path(p_from, p_packet, p_packet_len);
		} break;

		case SAVELOAD_COMMAND_RAW: {
			_process_raw(p_from, p_packet, p_packet_len);
		} break;
		case SAVELOAD_COMMAND_SPAWN: {
			saveloader->on_spawn_receive(p_from, p_packet, p_packet_len);
		} break;
		case SAVELOAD_COMMAND_DESPAWN: {
			saveloader->on_despawn_receive(p_from, p_packet, p_packet_len);
		} break;
		case SAVELOAD_COMMAND_SYNC: {
			saveloader->on_sync_receive(p_from, p_packet, p_packet_len);
		} break;
		default: {
			ERR_FAIL_MSG("Invalid network command from " + itos(p_from));
		} break;
	}
}

//#ifdef DEBUG_ENABLED
//_FORCE_INLINE_ Error SceneSaveload::_send(const uint8_t *p_packet, int p_packet_len) {
//	_profile_bandwidth("out", p_packet_len);
//	return multiplayer_peer->put_packet(p_packet, p_packet_len);
//}
//#endif

//Error SceneSaveload::send_command(int p_to, const uint8_t *p_packet, int p_packet_len) {
//	if (server_relay && get_unique_id() != 1 && p_to != 1 && multiplayer_peer->is_server_relay_supported()) {
//		// Send relay packet.
//		relay_buffer->seek(0);
//		relay_buffer->put_u8(NETWORK_COMMAND_SYS);
//		relay_buffer->put_u8(SYS_COMMAND_RELAY);
//		relay_buffer->put_32(p_to); // Set the destination.
//		relay_buffer->put_data(p_packet, p_packet_len);
//		multiplayer_peer->set_target_peer(1);
//		const Vector<uint8_t> data = relay_buffer->get_data_array();
//		return _send(data.ptr(), relay_buffer->get_position());
//	}
//	if (p_to > 0) {
//		ERR_FAIL_COND_V(!connected_peers.has(p_to), ERR_BUG);
//		multiplayer_peer->set_target_peer(p_to);
//		return _send(p_packet, p_packet_len);
//	} else {
//		for (const int &pid : connected_peers) {
//			if (p_to && pid == -p_to) {
//				continue;
//			}
//			multiplayer_peer->set_target_peer(pid);
//			_send(p_packet, p_packet_len);
//		}
//		return OK;
//	}
//}

//void SceneSaveload::_process_sys(int p_from, const uint8_t *p_packet, int p_packet_len, MultiplayerPeer::TransferMode p_mode, int p_channel) {
//	ERR_FAIL_COND_MSG(p_packet_len < SYS_CMD_SIZE, "Invalid packet received. Size too small.");
//	uint8_t sys_cmd_type = p_packet[1];
//	int32_t peer = int32_t(decode_uint32(&p_packet[2]));
//	switch (sys_cmd_type) {
//		case SYS_COMMAND_ADD_PEER: {
//			ERR_FAIL_COND(!server_relay || !multiplayer_peer->is_server_relay_supported() || get_unique_id() == 1 || p_from != 1);
//			_admit_peer(peer); // Relayed peers are automatically accepted.
//		} break;
//		case SYS_COMMAND_DEL_PEER: {
//			ERR_FAIL_COND(!server_relay || !multiplayer_peer->is_server_relay_supported() || get_unique_id() == 1 || p_from != 1);
//			_del_peer(peer);
//		} break;
//		case SYS_COMMAND_RELAY: {
//			ERR_FAIL_COND(!server_relay || !multiplayer_peer->is_server_relay_supported());
//			ERR_FAIL_COND(p_packet_len < SYS_CMD_SIZE + 1);
//			const uint8_t *packet = p_packet + SYS_CMD_SIZE;
//			int len = p_packet_len - SYS_CMD_SIZE;
//			bool should_process = false;
//			if (get_unique_id() == 1) { // I am the server.
//				// Direct messages to server should not go through relay.
//				ERR_FAIL_COND(peer > 0 && !connected_peers.has(peer));
//				// Send relay packet.
//				relay_buffer->seek(0);
//				relay_buffer->put_u8(NETWORK_COMMAND_SYS);
//				relay_buffer->put_u8(SYS_COMMAND_RELAY);
//				relay_buffer->put_32(p_from); // Set the source.
//				relay_buffer->put_data(packet, len);
//				const Vector<uint8_t> data = relay_buffer->get_data_array();
//				multiplayer_peer->set_transfer_mode(p_mode);
//				multiplayer_peer->set_transfer_channel(p_channel);
//				if (peer > 0) {
//					multiplayer_peer->set_target_peer(peer);
//					_send(data.ptr(), relay_buffer->get_position());
//				} else {
//					for (const int &P : connected_peers) {
//						// Not to sender, nor excluded.
//						if (P == p_from || (peer < 0 && P != -peer)) {
//							continue;
//						}
//						multiplayer_peer->set_target_peer(P);
//						_send(data.ptr(), relay_buffer->get_position());
//					}
//				}
//				if (peer == 0 || peer == -1) {
//					should_process = true;
//					peer = p_from; // Process as the source.
//				}
//			} else {
//				ERR_FAIL_COND(p_from != 1); // Bug.
//				should_process = true;
//			}
//			if (should_process) {
//				remote_sender_id = peer;
//				_process_packet(peer, packet, len);
//				remote_sender_id = 0;
//			}
//		} break;
//		default: {
//			ERR_FAIL();
//		}
//	}
//}

//void SceneSaveload::_add_peer(int p_id) {
//	if (auth_callback.is_valid()) {
//		pending_peers[p_id] = PendingPeer();
//		pending_peers[p_id].time = OS::get_singleton()->get_ticks_msec();
//		emit_signal(SNAME("peer_authenticating"), p_id);
//		return;
//	} else {
//		_admit_peer(p_id);
//	}
//}

//void SceneSaveload::_admit_peer(int p_id) {
//	if (server_relay && get_unique_id() == 1 && multiplayer_peer->is_server_relay_supported()) {
//		// Notify others of connection, and send connected peers to newly connected one.
//		uint8_t buf[SYS_CMD_SIZE];
//		buf[0] = NETWORK_COMMAND_SYS;
//		buf[1] = SYS_COMMAND_ADD_PEER;
//		multiplayer_peer->set_transfer_channel(0);
//		multiplayer_peer->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
//		for (const int &P : connected_peers) {
//			// Send new peer to already connected.
//			encode_uint32(p_id, &buf[2]);
//			multiplayer_peer->set_target_peer(P);
//			_send(buf, sizeof(buf));
//			// Send already connected to new peer.
//			encode_uint32(P, &buf[2]);
//			multiplayer_peer->set_target_peer(p_id);
//			_send(buf, sizeof(buf));
//		}
//	}
//
//	connected_peers.insert(p_id);
//	cache->on_peer_change(p_id, true);
//	replicator->on_peer_change(p_id, true);
//	if (p_id == 1) {
//		emit_signal(SNAME("connected_to_server"));
//	}
//	emit_signal(SNAME("peer_connected"), p_id);
//}

//void SceneSaveload::_del_peer(int p_id) {
//	if (pending_peers.has(p_id)) {
//		pending_peers.erase(p_id);
//		emit_signal(SNAME("peer_authentication_failed"), p_id);
//		return;
//	} else if (!connected_peers.has(p_id)) {
//		return;
//	}
//
//	if (server_relay && get_unique_id() == 1 && multiplayer_peer->is_server_relay_supported()) {
//		// Notify others of disconnection.
//		uint8_t buf[SYS_CMD_SIZE];
//		buf[0] = NETWORK_COMMAND_SYS;
//		buf[1] = SYS_COMMAND_DEL_PEER;
//		multiplayer_peer->set_transfer_channel(0);
//		multiplayer_peer->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
//		encode_uint32(p_id, &buf[2]);
//		for (const int &P : connected_peers) {
//			if (P == p_id) {
//				continue;
//			}
//			multiplayer_peer->set_target_peer(P);
//			_send(buf, sizeof(buf));
//		}
//	}
//
//	replicator->on_peer_change(p_id, false);
//	cache->on_peer_change(p_id, false);
//	connected_peers.erase(p_id);
//	emit_signal(SNAME("peer_disconnected"), p_id);
//}

//void SceneSaveload::disconnect_peer(int p_id) {
//	ERR_FAIL_COND(multiplayer_peer.is_null() || multiplayer_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED);
//	if (pending_peers.has(p_id)) {
//		pending_peers.erase(p_id);
//	} else if (connected_peers.has(p_id)) {
//		connected_peers.has(p_id);
//	}
//	multiplayer_peer->disconnect_peer(p_id);
//}

//Error SceneSaveload::send_bytes(Vector<uint8_t> p_data, int p_to, MultiplayerPeer::TransferMode p_mode, int p_channel) {
//	ERR_FAIL_COND_V_MSG(p_data.size() < 1, ERR_INVALID_DATA, "Trying to send an empty raw packet.");
//	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), ERR_UNCONFIGURED, "Trying to send a raw packet while no multiplayer peer is active.");
//	ERR_FAIL_COND_V_MSG(multiplayer_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED, ERR_UNCONFIGURED, "Trying to send a raw packet via a multiplayer peer which is not connected.");
//
//	if (packet_cache.size() < p_data.size() + 1) {
//		packet_cache.resize(p_data.size() + 1);
//	}
//
//	const uint8_t *r = p_data.ptr();
//	packet_cache.write[0] = NETWORK_COMMAND_RAW;
//	memcpy(&packet_cache.write[1], &r[0], p_data.size());
//
//	multiplayer_peer->set_transfer_channel(p_channel);
//	multiplayer_peer->set_transfer_mode(p_mode);
//	return send_command(p_to, packet_cache.ptr(), p_data.size() + 1);
//}

//Error SceneSaveload::send_auth(int p_to, Vector<uint8_t> p_data) {
//	ERR_FAIL_COND_V(multiplayer_peer.is_null() || multiplayer_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED, ERR_UNCONFIGURED);
//	ERR_FAIL_COND_V(!pending_peers.has(p_to), ERR_INVALID_PARAMETER);
//	ERR_FAIL_COND_V(p_data.size() < 1, ERR_INVALID_PARAMETER);
//	ERR_FAIL_COND_V_MSG(pending_peers[p_to].local, ERR_FILE_CANT_WRITE, "The authentication session was previously marked as completed, no more authentication data can be sent.");
//	ERR_FAIL_COND_V_MSG(pending_peers[p_to].remote, ERR_FILE_CANT_WRITE, "The remote peer notified that the authentication session was completed, no more authentication data can be sent.");
//
//	if (packet_cache.size() < p_data.size() + 2) {
//		packet_cache.resize(p_data.size() + 2);
//	}
//
//	packet_cache.write[0] = NETWORK_COMMAND_SYS;
//	packet_cache.write[1] = SYS_COMMAND_AUTH;
//	memcpy(&packet_cache.write[2], p_data.ptr(), p_data.size());
//
//	multiplayer_peer->set_target_peer(p_to);
//	multiplayer_peer->set_transfer_channel(0);
//	multiplayer_peer->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
//	return _send(packet_cache.ptr(), p_data.size() + 2);
//}

//Error SceneSaveload::complete_auth(int p_peer) {
//	ERR_FAIL_COND_V(multiplayer_peer.is_null() || multiplayer_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED, ERR_UNCONFIGURED);
//	ERR_FAIL_COND_V(!pending_peers.has(p_peer), ERR_INVALID_PARAMETER);
//	ERR_FAIL_COND_V_MSG(pending_peers[p_peer].local, ERR_FILE_CANT_WRITE, "The authentication session was already marked as completed.");
//	pending_peers[p_peer].local = true;
//	// Notify the remote peer that the authentication has completed.
//	uint8_t buf[2] = { NETWORK_COMMAND_SYS, SYS_COMMAND_AUTH };
//	Error err = _send(buf, 2);
//	// The remote peer already reported the authentication as completed, so admit the peer.
//	// May generate new packets, so it must happen after sending confirmation.
//	if (pending_peers[p_peer].remote) {
//		pending_peers.erase(p_peer);
//		_admit_peer(p_peer);
//	}
//	return err;
//}

//void SceneSaveload::set_auth_callback(Callable p_callback) {
//	auth_callback = p_callback;
//}

//Callable SceneSaveload::get_auth_callback() const {
//	return auth_callback;
//}

//void SceneSaveload::set_auth_timeout(double p_timeout) {
//	ERR_FAIL_COND_MSG(p_timeout < 0, "Timeout must be greater or equal to 0 (where 0 means no timeout)");
//	auth_timeout = uint64_t(p_timeout * 1000);
//}

//double SceneSaveload::get_auth_timeout() const {
//	return double(auth_timeout) / 1000.0;
//}

void SceneSaveload::_process_raw(int p_from, const uint8_t *p_packet, int p_packet_len) {
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

TypedArray<SaveloadSynchronizer> SceneSaveload::get_sync_nodes() {
	return saveloader->get_sync_nodes();
}

Variant SceneSaveload::get_state(const Object *p_object, const StringName section) {
	return Variant();
}
Error SceneSaveload::set_state(const Variant p_value, const Object *p_object, const StringName section) {
	return ERR_UNAVAILABLE;
}

Vector<uint8_t> SceneSaveload::encode(Object *p_object, const StringName section) {
	return saveloader->encode(p_object, section);
}
Error SceneSaveload::decode(Vector<uint8_t> p_bytes, Object *p_object, const StringName section) {
	return saveloader->decode(p_bytes, p_object, section);
}
Error SceneSaveload::save(const String p_path, Object *p_object, const StringName section) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	if (err != OK) {
		return err;
	}
	Vector<uint8_t> bytes = encode(p_object, section);
	file->store_buffer(bytes);
	file->close();
	return err;
}
Error SceneSaveload::load(const String p_path, Object *p_object, const StringName section) {
	Error err;
	Vector<uint8_t> bytes = FileAccess::get_file_as_bytes(p_path, &err);
	if (err != OK) {
		return err;
	}
	return decode(bytes, p_object, section);
}

int SceneSaveload::get_unique_id() {
//	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), 0, "No multiplayer peer is assigned. Unable to get unique ID.");
//	return multiplayer_peer->get_unique_id();
	return 1;
}

//void SceneSaveload::set_refuse_new_connections(bool p_refuse) {
//	ERR_FAIL_COND_MSG(!multiplayer_peer.is_valid(), "No multiplayer peer is assigned. Unable to set 'refuse_new_connections'.");
//	multiplayer_peer->set_refuse_new_connections(p_refuse);
//}

//bool SceneSaveload::is_refusing_new_connections() const {
//	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), false, "No multiplayer peer is assigned. Unable to get 'refuse_new_connections'.");
//	return multiplayer_peer->is_refusing_new_connections();
//}

//Vector<int> SceneSaveload::get_peer_ids() {
//	ERR_FAIL_COND_V_MSG(!multiplayer_peer.is_valid(), Vector<int>(), "No multiplayer peer is assigned. Assume no peers are connected.");
//
//	Vector<int> ret;
//	for (const int &E : connected_peers) {
//		ret.push_back(E);
//	}
//
//	return ret;
//}

//Vector<int> SceneSaveload::get_authenticating_peer_ids() {
//	Vector<int> out;
//	out.resize(pending_peers.size());
//	int idx = 0;
//	for (const KeyValue<int, PendingPeer> &E : pending_peers) {
//		out.write[idx++] = E.key;
//	}
//	return out;
//}

void SceneSaveload::set_allow_object_decoding(bool p_enable) {
	allow_object_decoding = p_enable;
}

bool SceneSaveload::is_object_decoding_allowed() const {
	return allow_object_decoding;
}

Error SceneSaveload::object_configuration_add(Object *p_obj, Variant p_config) {
	if (p_obj == nullptr && p_config.get_type() == Variant::NODE_PATH) {
		set_root_path(p_config);
		return OK;
	}
	SaveloadSpawner *spawner = Object::cast_to<SaveloadSpawner>(p_config.get_validated_object());
	SaveloadSynchronizer *sync = Object::cast_to<SaveloadSynchronizer>(p_config.get_validated_object());
	if (spawner) {
		return saveloader->on_spawn(p_obj, p_config);
	} else if (sync) {
		return saveloader->on_saveload_start(p_obj, p_config);
	}
	return ERR_INVALID_PARAMETER;
}

Error SceneSaveload::object_configuration_remove(Object *p_obj, Variant p_config) {
	if (p_obj == nullptr && p_config.get_type() == Variant::NODE_PATH) {
		ERR_FAIL_COND_V(root_path != p_config.operator NodePath(), ERR_INVALID_PARAMETER);
		set_root_path(NodePath());
		return OK;
	}
	SaveloadSpawner *spawner = Object::cast_to<SaveloadSpawner>(p_config.get_validated_object());
	SaveloadSynchronizer *sync = Object::cast_to<SaveloadSynchronizer>(p_config.get_validated_object());
	if (spawner) {
		return saveloader->on_despawn(p_obj, p_config);
	}
	if (sync) {
		return saveloader->on_saveload_stop(p_obj, p_config);
	}
	return ERR_INVALID_PARAMETER;
}

void SceneSaveload::set_max_sync_packet_size(int p_size) {
	saveloader->set_max_sync_packet_size(p_size);
}

int SceneSaveload::get_max_sync_packet_size() const {
	return saveloader->get_max_sync_packet_size();
}

void SceneSaveload::set_max_delta_packet_size(int p_size) {
	saveloader->set_max_delta_packet_size(p_size);
}

int SceneSaveload::get_max_delta_packet_size() const {
	return saveloader->get_max_delta_packet_size();
}

void SceneSaveload::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &SceneSaveload::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &SceneSaveload::get_root_path);
	ClassDB::bind_method(D_METHOD("clear"), &SceneSaveload::clear);

//	ClassDB::bind_method(D_METHOD("disconnect_peer", "id"), &SceneSaveload::disconnect_peer);
//
//	ClassDB::bind_method(D_METHOD("get_authenticating_peers"), &SceneSaveload::get_authenticating_peer_ids);
//	ClassDB::bind_method(D_METHOD("send_auth", "id", "data"), &SceneSaveload::send_auth);
//	ClassDB::bind_method(D_METHOD("complete_auth", "id"), &SceneSaveload::complete_auth);
//
//	ClassDB::bind_method(D_METHOD("set_auth_callback", "callback"), &SceneSaveload::set_auth_callback);
//	ClassDB::bind_method(D_METHOD("get_auth_callback"), &SceneSaveload::get_auth_callback);
//	ClassDB::bind_method(D_METHOD("set_auth_timeout", "timeout"), &SceneSaveload::set_auth_timeout);
//	ClassDB::bind_method(D_METHOD("get_auth_timeout"), &SceneSaveload::get_auth_timeout);

//	ClassDB::bind_method(D_METHOD("set_refuse_new_connections", "refuse"), &SceneSaveload::set_refuse_new_connections);
//	ClassDB::bind_method(D_METHOD("is_refusing_new_connections"), &SceneSaveload::is_refusing_new_connections);
//	ClassDB::bind_method(D_METHOD("set_allow_object_decoding", "enable"), &SceneSaveload::set_allow_object_decoding);
//	ClassDB::bind_method(D_METHOD("is_object_decoding_allowed"), &SceneSaveload::is_object_decoding_allowed);
//	ClassDB::bind_method(D_METHOD("set_server_relay_enabled", "enabled"), &SceneSaveload::set_server_relay_enabled);
//	ClassDB::bind_method(D_METHOD("is_server_relay_enabled"), &SceneSaveload::is_server_relay_enabled);
//	ClassDB::bind_method(D_METHOD("send_bytes", "bytes", "id", "mode", "channel"), &SceneSaveload::send_bytes, DEFVAL(MultiplayerPeer::TARGET_PEER_BROADCAST), DEFVAL(MultiplayerPeer::TRANSFER_MODE_RELIABLE), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("get_sync_nodes"), &SceneSaveload::get_sync_nodes);

	ClassDB::bind_method(D_METHOD("get_max_sync_packet_size"), &SceneSaveload::get_max_sync_packet_size);
	ClassDB::bind_method(D_METHOD("set_max_sync_packet_size", "size"), &SceneSaveload::set_max_sync_packet_size);
	ClassDB::bind_method(D_METHOD("get_max_delta_packet_size"), &SceneSaveload::get_max_delta_packet_size);
	ClassDB::bind_method(D_METHOD("set_max_delta_packet_size", "size"), &SceneSaveload::set_max_delta_packet_size);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");
//	ADD_PROPERTY(PropertyInfo(Variant::CALLABLE, "auth_callback"), "set_auth_callback", "get_auth_callback");
//	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "auth_timeout", PROPERTY_HINT_RANGE, "0,30,0.1,or_greater,suffix:s"), "set_auth_timeout", "get_auth_timeout");
//	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_object_decoding"), "set_allow_object_decoding", "is_object_decoding_allowed");
//	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "refuse_new_connections"), "set_refuse_new_connections", "is_refusing_new_connections");
//	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "server_relay"), "set_server_relay_enabled", "is_server_relay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_sync_packet_size"), "set_max_sync_packet_size", "get_max_sync_packet_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_delta_packet_size"), "set_max_delta_packet_size", "get_max_delta_packet_size");

	ADD_PROPERTY_DEFAULT("refuse_new_connections", false);

	ADD_SIGNAL(MethodInfo("peer_authenticating", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_authentication_failed", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("peer_packet", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::PACKED_BYTE_ARRAY, "packet")));
}

SceneSaveload::SceneSaveload() {
//	relay_buffer.instantiate();
	saveloader = Ref<SceneSaveloadInterface>(memnew(SceneSaveloadInterface(this)));
	cache = Ref<SceneCacheInterface>(memnew(SceneCacheInterface(this)));
}

SceneSaveload::~SceneSaveload() {
	clear();
}
