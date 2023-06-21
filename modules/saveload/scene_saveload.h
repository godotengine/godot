/**************************************************************************/
/*  scene_saveload.h                                                      */
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

#ifndef SCENE_SAVELOAD_H
#define SCENE_SAVELOAD_H

#include "scene/main/saveload_api.h"

#include "scene_cache_interface.h"
#include "scene_saveload_interface.h"

class SceneSaveload : public SaveloadAPI {
	GDCLASS(SceneSaveload, SaveloadAPI);

public:
	enum SaveloadCommands {
		SAVELOAD_COMMAND_REMOTE_CALL = 0,
		SAVELOAD_COMMAND_SIMPLIFY_PATH,
		SAVELOAD_COMMAND_CONFIRM_PATH,
		SAVELOAD_COMMAND_RAW,
		SAVELOAD_COMMAND_SPAWN,
		SAVELOAD_COMMAND_DESPAWN,
		SAVELOAD_COMMAND_SYNC,
		SAVELOAD_COMMAND_SYS,
};
//
//	enum SysCommands {
//		SYS_COMMAND_AUTH,
//		SYS_COMMAND_ADD_PEER,
//		SYS_COMMAND_DEL_PEER,
//		SYS_COMMAND_RELAY,
//	};

	enum {
		SYS_CMD_SIZE = 6, // Command + sys command + peer_id (+ optional payload).
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
	struct PendingPeer {
		bool local = false;
		bool remote = false;
		uint64_t time = 0;
	};

	Ref<StreamPeerBuffer> stream_peer_buffer;
//	MultiplayerPeer::ConnectionStatus last_connection_status = MultiplayerPeer::CONNECTION_DISCONNECTED;
	Callable auth_callback;
	uint64_t auth_timeout = 3000;
	HashSet<int> connected_peers;
	int remote_sender_id = 0;
	int remote_sender_override = 0;

	Vector<uint8_t> packet_cache;

	NodePath root_path;
	bool allow_object_decoding = false;
	bool server_relay = true;
//	Ref<StreamPeerBuffer> relay_buffer;

	Ref<SceneCacheInterface> cache;
	Ref<SceneSaveloadInterface> saveloader;

#ifdef DEBUG_ENABLED
	_FORCE_INLINE_ void _profile_bandwidth(const String &p_what, int p_value);
	_FORCE_INLINE_ Error _send(const uint8_t *p_packet, int p_packet_len); // Also profiles.
#endif

protected:
	static void _bind_methods();

	void _process_packet(int p_from, const uint8_t *p_packet, int p_packet_len);
	void _process_raw(int p_from, const uint8_t *p_packet, int p_packet_len);

public:
	TypedArray<SaveloadSynchronizer> get_sync_nodes();

	virtual Variant get_state(const Object *p_object, const StringName section = "") override;
	virtual Error set_state(const Variant p_value, const Object *p_object, const StringName section = "") override;

	virtual PackedByteArray encode(Object *p_object, const StringName section = "") override;
	virtual Error decode(PackedByteArray p_bytes, Object *p_object, const StringName section = "") override;

	virtual Error save(const String p_path, Object *p_object, const StringName section = "");
	virtual Error load(const String p_path, Object *p_object, const StringName section = "");

	virtual int get_unique_id() override;

	virtual Error object_configuration_add(Object *p_obj, Variant p_config) override;
	virtual Error object_configuration_remove(Object *p_obj, Variant p_config) override;

	void clear();

	// Usually from object_configuration_add/remove
	void set_root_path(const NodePath &p_path);
	NodePath get_root_path() const;

	Error send_command(int p_to, const uint8_t *p_packet, int p_packet_len); // Used internally to relay packets when needed.

	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;

	void set_max_sync_packet_size(int p_size);
	int get_max_sync_packet_size() const;

	void set_max_delta_packet_size(int p_size);
	int get_max_delta_packet_size() const;

	Ref<SceneCacheInterface> get_path_cache() { return cache; }
	Ref<SceneSaveloadInterface> get_saveloader() { return saveloader; }

	SceneSaveload();
	~SceneSaveload();
};

#endif // SCENE_SAVELOAD_H
