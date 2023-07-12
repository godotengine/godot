/**************************************************************************/
/*  scene_saveload_interface.cpp                                          */
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

#include "scene_saveload_interface.h"

#include "scene_saveload.h"

#include "core/debugger/engine_debugger.h"
#include "scene/scene_string_names.h"

#ifdef DEBUG_ENABLED
_FORCE_INLINE_ void SceneSaveloadInterface::_profile_node_data(const String &p_what, ObjectID p_id, int p_size) {
	if (EngineDebugger::is_profiling("saveload:saveload")) {
		Array values;
		values.push_back(p_what);
		values.push_back(p_id);
		values.push_back(p_size);
		EngineDebugger::profiler_add_frame_data("saveload:saveload", values);
	}
}
#endif

Error SceneSaveloadInterface::track(Object *p_object) {
	SaveloadSpawner *spawner = Object::cast_to<SaveloadSpawner>(p_object);
	SaveloadSynchronizer *syncher = Object::cast_to<SaveloadSynchronizer>(p_object);
	if (spawner) {
		track_spawner(*spawner);
		return OK;
	} else if (syncher) {
		track_syncher(*syncher);
		return OK;
	} else {
		return ERR_INVALID_DATA;
	}
}

Error SceneSaveloadInterface::untrack(Object *p_object) {
	SaveloadSpawner *spawner = Object::cast_to<SaveloadSpawner>(p_object);
	SaveloadSynchronizer *syncher = Object::cast_to<SaveloadSynchronizer>(p_object);
	if (spawner) {
		untrack_spawner(*spawner);
		return OK;
	} else if (syncher) {
		untrack_syncher(*syncher);
		return OK;
	} else {
		return ERR_INVALID_DATA;
	}
}

void SceneSaveloadInterface::track_spawner(const SaveloadSpawner &p_spawner) {
	const ObjectID spawner_id = p_spawner.get_instance_id();
	if (!spawners.has(spawner_id)) {
		spawners.insert(spawner_id);
	}
}

void SceneSaveloadInterface::untrack_spawner(const SaveloadSpawner &p_spawner) {
	spawners.erase(p_spawner.get_instance_id());
}

void SceneSaveloadInterface::track_syncher(const SaveloadSynchronizer &p_syncher) {
	const ObjectID syncher_id = p_syncher.get_instance_id();
	if (!synchers.has(syncher_id)) {
		synchers.insert(syncher_id);
	}
}

void SceneSaveloadInterface::untrack_syncher(const SaveloadSynchronizer &p_syncher) {
	synchers.erase(p_syncher.get_instance_id());
}

TypedArray<SaveloadSpawner> SceneSaveloadInterface::get_spawners() const {
	TypedArray<SaveloadSynchronizer> spawner_array;
	for (const ObjectID &oid : spawners) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid SaveloadSpawner", oid));
		spawner_array.append(spawner);
	}
	return spawner_array;
}

TypedArray<SaveloadSynchronizer> SceneSaveloadInterface::get_synchers() const {
	TypedArray<SaveloadSynchronizer> sync_array;
	for (const ObjectID &oid : synchers) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid SaveloadSynchronizer", oid));
		sync_array.append(sync);
	}
	return sync_array;
}

Dictionary SceneSaveloadInterface::get_spawn_dict() const {
	Dictionary spawn_dict;
	for (const ObjectID &oid : spawners) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid SaveloadSpawner", oid));
		spawn_dict[spawner->get_path()] = spawner->get_spawner_state().to_array();
	}
	return spawn_dict;
}

Dictionary SceneSaveloadInterface::get_sync_dict() const {
	Dictionary sync_dict;
	for (const ObjectID &oid : synchers) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid SaveloadSynchronizer"));
		sync_dict[sync->get_path()] = sync->get_syncher_state().to_dict();
	}
	return sync_dict;
}

SceneSaveloadInterface::SaveloadState SceneSaveloadInterface::get_saveload_state() const {
	SaveloadState saveload_state;
	for (const ObjectID &oid : spawners) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid SaveloadSpawner", oid));
		SaveloadSpawner::SpawnerState spawn_state = spawner->get_spawner_state();
		saveload_state.spawner_states.insert(spawner->get_path(), spawn_state);
	}
	for (const ObjectID &oid : synchers) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid SaveloadSynchronizer", oid));
		SaveloadSynchronizer::SyncherState sync_state = sync->get_syncher_state();
		saveload_state.syncher_states.insert(sync->get_path(), sync_state);
	}
	return saveload_state;
}

Error SceneSaveloadInterface::load_saveload_state(const SaveloadState &p_saveload_state) {
	Node *root = SceneTree::get_singleton()->get_current_scene(); //TODO: Is this the right root?
	for (const KeyValue<const NodePath, SaveloadSpawner::SpawnerState> &spawner_state : p_saveload_state.spawner_states) {
		SaveloadSpawner *spawner_node = Object::cast_to<SaveloadSpawner>(root->get_node(spawner_state.key));
		ERR_CONTINUE_MSG(!spawner_node, vformat("could not find SaveloadSpawner at path %s", spawner_state.key));
		spawner_node->load_spawn_state(spawner_state.value);
	}
	for (const KeyValue<const NodePath, SaveloadSynchronizer::SyncherState> &sync_state : p_saveload_state.syncher_states) {
		SaveloadSynchronizer *sync_node = Object::cast_to<SaveloadSynchronizer>(root->get_node(sync_state.key));
		ERR_CONTINUE_MSG(!sync_node, vformat("could not find SaveloadSynchronizer at path %s", sync_state.key));
		sync_node->synchronize(sync_state.value);
	}
	return OK; //TODO: return some errors
}

Dictionary SceneSaveloadInterface::SaveloadState::to_dict() const {
	Dictionary dict;
	Dictionary spawn_dict;
	for (const KeyValue<const NodePath, SaveloadSpawner::SpawnerState> &spawn_state : SaveloadState::spawner_states) {
		spawn_dict[spawn_state.key] = spawn_state.value.to_array();
	}
	Dictionary sync_dict;
	for (const KeyValue<const NodePath, SaveloadSynchronizer::SyncherState> &sync_state : SaveloadState::syncher_states) {
		sync_dict[sync_state.key] = sync_state.value.to_dict();
	}
	dict[StringName("spawn_states")] = spawn_dict;
	dict[StringName("sync_states")] = sync_dict;
	return dict;
}

SceneSaveloadInterface::SaveloadState::SaveloadState(const Dictionary &p_saveload_dict) {
	Dictionary spawn_states_dict = p_saveload_dict[StringName("spawn_states")];
	Dictionary sync_states_dict = p_saveload_dict[StringName("sync_states")];
	List<Variant> spawn_keys;
	spawn_states_dict.get_key_list(&spawn_keys);
	for (const NodePath spawn_key : spawn_keys) {
		TypedArray<Dictionary> spawn_state_as_array = spawn_states_dict[spawn_key];
		spawner_states.insert(spawn_key, SaveloadSpawner::SpawnerState(spawn_state_as_array));
	}
	List<Variant> sync_keys;
	sync_states_dict.get_key_list(&sync_keys);
	for (const NodePath sync_key : sync_keys) {
		Dictionary sync_state_as_dict = sync_states_dict[sync_key];
		syncher_states.insert(sync_key, SaveloadSynchronizer::SyncherState(sync_state_as_dict));
	}
}