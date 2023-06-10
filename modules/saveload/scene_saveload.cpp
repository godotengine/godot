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
	replicator->on_reset();
	cache->clear();
	relay_buffer->clear();
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
		case NETWORK_COMMAND_SIMPLIFY_PATH: {
			cache->process_simplify_path(p_from, p_packet, p_packet_len);
		} break;

		case NETWORK_COMMAND_CONFIRM_PATH: {
			cache->process_confirm_path(p_from, p_packet, p_packet_len);
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
		default: {
			ERR_FAIL_MSG("Invalid network command from " + itos(p_from));
		} break;
	}
}

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
		return replicator->on_spawn(p_obj, p_config);
	} else if (sync) {
		return replicator->on_replication_start(p_obj, p_config);
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
		return replicator->on_despawn(p_obj, p_config);
	}
	if (sync) {
		return replicator->on_replication_stop(p_obj, p_config);
	}
	return ERR_INVALID_PARAMETER;
}

void SceneSaveload::set_max_sync_packet_size(int p_size) {
	replicator->set_max_sync_packet_size(p_size);
}

int SceneSaveload::get_max_sync_packet_size() const {
	return replicator->get_max_sync_packet_size();
}

void SceneSaveload::set_max_delta_packet_size(int p_size) {
	replicator->set_max_delta_packet_size(p_size);
}

int SceneSaveload::get_max_delta_packet_size() const {
	return replicator->get_max_delta_packet_size();
}

void SceneSaveload::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &SceneSaveload::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &SceneSaveload::get_root_path);
	ClassDB::bind_method(D_METHOD("clear"), &SceneSaveload::clear);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_object_decoding"), "set_allow_object_decoding", "is_object_decoding_allowed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_sync_packet_size"), "set_max_sync_packet_size", "get_max_sync_packet_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_delta_packet_size"), "set_max_delta_packet_size", "get_max_delta_packet_size");
}

SceneSaveload::SceneSaveload() {
	relay_buffer.instantiate();
	replicator = Ref<SceneSaveloadInterface>(memnew(SceneSaveloadInterface(this)));
	cache = Ref<SceneCacheInterface>(memnew(SceneCacheInterface(this)));
}

SceneSaveload::~SceneSaveload() {
	clear();
}
