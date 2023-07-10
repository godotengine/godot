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

}

void SceneSaveload::set_root_path(const NodePath &p_path) {
	ERR_FAIL_COND_MSG(!p_path.is_absolute() && !p_path.is_empty(), "SceneSaveload root path must be absolute.");
	root_path = p_path;
}

NodePath SceneSaveload::get_root_path() const {
	return root_path;
}

TypedArray<SaveloadSpawner> SceneSaveload::get_spawn_nodes() {
	return saveloader->get_spawn_nodes();
}

TypedArray<SaveloadSynchronizer> SceneSaveload::get_sync_nodes() {
	return saveloader->get_sync_nodes();
}

Dictionary SceneSaveload::get_dict() {
	SceneSaveloadInterface::SaveloadState saveload_state = saveloader->get_saveload_state();
	return saveload_state.to_dict();
}

Variant SceneSaveload::get_state(Object *p_object, const StringName section) {
	Dictionary state;
	state[StringName("spawners")] = saveloader->get_spawn_dict();
	state[StringName("synchers")] = saveloader->get_sync_dict();
	return state;
}

Error SceneSaveload::set_state(Variant p_value, Object *p_object, const StringName section) {
	return ERR_UNAVAILABLE;
}

PackedByteArray SceneSaveload::encode(Object *p_object, const StringName section) {
	return saveloader->encode(p_object, section);
}
Error SceneSaveload::decode(PackedByteArray p_bytes, Object *p_object, const StringName section) {
	return saveloader->decode(p_bytes, p_object, section);
}
Error SceneSaveload::save(const String p_path, Object *p_object, const StringName section) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	if (err != OK) {
		return err;
	}
	//PackedByteArray bytes = encode(p_object, section);
	//file->store_buffer(bytes);
	Dictionary dict = saveloader->get_saveload_state().to_dict();
	file->store_var(dict, false);
	file->close();
	return err;
}
Error SceneSaveload::load(const String p_path, Object *p_object, const StringName section) {
	Error err;
	//PackedByteArray bytes = FileAccess::get_file_as_bytes(p_path, &err);
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err != OK) {
		return err;
	}
	Dictionary dict = file->get_var(false);

	SceneSaveloadInterface::SaveloadState saveload_state = SceneSaveloadInterface::SaveloadState(dict);
	saveloader->load_saveload_state(saveload_state);
	return OK;
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
	Node *node = Object::cast_to<Node>(p_obj);
	ERR_FAIL_COND_V(!node, ERR_INVALID_PARAMETER);
	SaveloadSpawner *spawner = Object::cast_to<SaveloadSpawner>(p_config.get_validated_object());
	SaveloadSynchronizer *sync = Object::cast_to<SaveloadSynchronizer>(p_config.get_validated_object());
	if (spawner) {
		saveloader->configure_spawn(node, *spawner);
		return OK;
	} else if (sync) {
		saveloader->configure_sync(node, *sync);
		return OK;
	}
	return ERR_INVALID_PARAMETER;
}

Error SceneSaveload::object_configuration_remove(Object *p_obj, Variant p_config) {
	if (p_obj == nullptr && p_config.get_type() == Variant::NODE_PATH) { //I don't think root path actually does anything
		ERR_FAIL_COND_V(root_path != p_config.operator NodePath(), ERR_INVALID_PARAMETER);
		set_root_path(NodePath());
		return OK;
	}
	SaveloadSpawner *spawner = Object::cast_to<SaveloadSpawner>(p_config.get_validated_object());
	SaveloadSynchronizer *sync = Object::cast_to<SaveloadSynchronizer>(p_config.get_validated_object());
	if (spawner) {
		saveloader->deconfigure_spawn(*spawner);
		return OK;
	}
	if (sync) {
		saveloader->deconfigure_sync(*sync);
		return OK;
	}
	return ERR_INVALID_PARAMETER;
}

void SceneSaveload::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_path", "path"), &SceneSaveload::set_root_path);
	ClassDB::bind_method(D_METHOD("get_root_path"), &SceneSaveload::get_root_path);
	ClassDB::bind_method(D_METHOD("clear"), &SceneSaveload::clear);

	ClassDB::bind_method(D_METHOD("get_sync_nodes"), &SceneSaveload::get_sync_nodes);
	ClassDB::bind_method(D_METHOD("get_dict"), &SceneSaveload::get_dict);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_path"), "set_root_path", "get_root_path");
}

SceneSaveload::SceneSaveload() {
	saveloader = Ref<SceneSaveloadInterface>(memnew(SceneSaveloadInterface(this)));
}

SceneSaveload::~SceneSaveload() {
	clear();
}
