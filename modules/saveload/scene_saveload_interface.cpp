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

void SceneSaveloadInterface::flush_spawn_queue() {
	// Prevent endless stalling in case of unforeseen spawn errors.
	if (spawn_queue.size()) {
		ERR_PRINT("An error happened during last spawn, this usually means the 'ready' signal was not emitted by the spawned node.");
		for (const ObjectID &oid : spawn_queue) {
			Node *node = get_id_as<Node>(oid);
			ERR_CONTINUE(!node);
			if (node->is_connected(SceneStringNames::get_singleton()->ready, callable_mp(this, &SceneSaveloadInterface::_node_ready))) {
				node->disconnect(SceneStringNames::get_singleton()->ready, callable_mp(this, &SceneSaveloadInterface::_node_ready));
			}
		}
		spawn_queue.clear();
	}
}

void SceneSaveloadInterface::configure_spawn(Node *p_node, const SaveloadSpawner &p_spawner) {
	const ObjectID node_id = p_node->get_instance_id();
	const ObjectID spawner_id = p_spawner.get_instance_id();
	if (!spawn_nodes.has(spawner_id)) {
		spawn_nodes.insert(spawner_id);
	}
	p_node->connect(SceneStringNames::get_singleton()->ready, callable_mp(this, &SceneSaveloadInterface::_node_ready).bind(node_id), Node::CONNECT_ONE_SHOT);
}

void SceneSaveloadInterface::deconfigure_spawn(const SaveloadSpawner &p_spawner) {
	spawn_nodes.erase(p_spawner.get_instance_id());
}

void SceneSaveloadInterface::configure_sync(Node *p_node, const SaveloadSynchronizer &p_syncher) {
	const NodePath root_path = p_syncher.get_root_path();
	Node *root_node = p_syncher.get_root_node();
	ERR_FAIL_COND_MSG(!root_node, vformat("Could not find Synchronizer root node at path %s", root_path));
	ERR_FAIL_COND_MSG((root_node->get_path() != p_node->get_path()) && !(root_node->is_ancestor_of(p_node)), vformat("Synchronizer root at %s is not an ancestor of node at %s", root_node->get_path(), p_node->get_path()));
	sync_nodes.insert(p_syncher.get_instance_id());
}

void SceneSaveloadInterface::deconfigure_sync(const SaveloadSynchronizer &p_syncher) {
	sync_nodes.erase(p_syncher.get_instance_id());
}

void SceneSaveloadInterface::_node_ready(const ObjectID &p_oid) {
	spawn_queue.clear();
}

TypedArray<SaveloadSpawner> SceneSaveloadInterface::get_spawn_nodes() const {
	TypedArray<SaveloadSynchronizer> spawners;
	for (const ObjectID &oid : spawn_nodes) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid SaveloadSpawner", oid));
		spawners.append(spawner);
	}
	return spawners;
}

TypedArray<SaveloadSynchronizer> SceneSaveloadInterface::get_sync_nodes() const {
	TypedArray<SaveloadSynchronizer> syncs;
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid SaveloadSynchronizer", oid));
		syncs.append(sync);
	}
	return syncs;
}

Dictionary SceneSaveloadInterface::get_spawn_dict() const {
	Dictionary spawn_dict;
	for (const ObjectID &oid : spawn_nodes) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid SaveloadSpawner", oid));
		spawn_dict[spawner->get_path()] = spawner->get_spawner_state().to_array();
	}
	return spawn_dict;
}

Dictionary SceneSaveloadInterface::get_sync_dict() const {
	Dictionary sync_dict;
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid SaveloadSynchronizer"));
		sync_dict[sync->get_path()] = sync->get_sync_state().to_dict();
	}
	return sync_dict;
}

SceneSaveloadInterface::SaveloadState SceneSaveloadInterface::get_saveload_state() {
	SaveloadState saveload_state;
	for (const ObjectID &oid : spawn_nodes) {
		SaveloadSpawner *spawner = get_id_as<SaveloadSpawner>(oid);
		ERR_CONTINUE_MSG(!spawner, vformat("%s is not a valid SaveloadSpawner", oid));
		SaveloadSpawner::SpawnerState spawn_state = spawner->get_spawner_state();
		saveload_state.spawner_states.insert(spawner->get_path(), spawn_state);
	}
	for (const ObjectID &oid : sync_nodes) {
		SaveloadSynchronizer *sync = get_id_as<SaveloadSynchronizer>(oid);
		ERR_CONTINUE_MSG(!sync, vformat("%s is not a valid SaveloadSynchronizer", oid));
		SaveloadSynchronizer::SyncState sync_state = sync->get_sync_state();
		saveload_state.sync_states.insert(sync->get_path(), sync_state);
	}
	return saveload_state;
}

void SceneSaveloadInterface::load_saveload_state(const SaveloadState &p_saveload_state) {
	print_line("SceneSaveloadInterface::load_saveload_state saveload_state: ");
	print_line(p_saveload_state.to_dict());
	Node *root = SceneTree::get_singleton()->get_current_scene(); //TODO: Is this the right root?
	for (const KeyValue<const NodePath, SaveloadSpawner::SpawnerState> &spawner_state : p_saveload_state.spawner_states) {
		SaveloadSpawner *spawner_node = Object::cast_to<SaveloadSpawner>(root->get_node(spawner_state.key));
		ERR_CONTINUE_MSG(!spawner_node, vformat("could not find SaveloadSpawner at path %s", spawner_state.key));
		print_line("SceneSaveloadInterface::load_saveload_state spawner_state: ");
		print_line(spawner_state.value.to_array());
		spawner_node->load_spawn_state(spawner_state.value);
	}
	for (const KeyValue<const NodePath, SaveloadSynchronizer::SyncState> &sync_state : p_saveload_state.sync_states) {
		SaveloadSynchronizer *sync_node = Object::cast_to<SaveloadSynchronizer>(root->get_node(sync_state.key));
		ERR_CONTINUE_MSG(!sync_node, vformat("could not find SaveloadSynchronizer at path %s", sync_state.key));
		sync_node->synchronize(sync_state.value);
	}
}

Dictionary SceneSaveloadInterface::SaveloadState::to_dict() const {
	Dictionary dict;
	Dictionary spawn_dict;
	for (const KeyValue<const NodePath, SaveloadSpawner::SpawnerState> &spawn_state : SaveloadState::spawner_states) {
		spawn_dict[spawn_state.key] = spawn_state.value.to_array();
	}
	Dictionary sync_dict;
	for (const KeyValue<const NodePath, SaveloadSynchronizer::SyncState> &sync_state : SaveloadState::sync_states) {
		sync_dict[sync_state.key] = sync_state.value.to_dict();
	}
	dict[StringName("spawn_states")] = spawn_dict;
	dict[StringName("sync_states")] = sync_dict;
	return dict;
}

SceneSaveloadInterface::SaveloadState::SaveloadState(const Dictionary &p_saveload_dict) {
	print_line("SceneSveloadInterface::SaveloadState::SaveloadState dictionary in:");
	print_line(p_saveload_dict);
	Dictionary spawn_states_dict = p_saveload_dict[StringName("spawn_states")];
	print_line("SceneSveloadInterface::SaveloadState::SaveloadState spawn_states_dict:");
	print_line(spawn_states_dict);
	Dictionary sync_states_dict = p_saveload_dict[StringName("sync_states")];
	List<Variant> spawn_keys;
	spawn_states_dict.get_key_list(&spawn_keys);
	for (const NodePath spawn_key : spawn_keys) {
		print_line("SceneSveloadInterface::SaveloadState::SaveloadState spawn_key:");
		print_line(spawn_key);
		TypedArray<Dictionary> spawn_state_as_array = spawn_states_dict[spawn_key];
		print_line("SceneSveloadInterface::SaveloadState::SaveloadState spawn_state_as_array:");
		print_line(spawn_state_as_array);
		spawner_states.insert(spawn_key, SaveloadSpawner::SpawnerState(spawn_state_as_array));
	}
	List<Variant> sync_keys;
	sync_states_dict.get_key_list(&sync_keys);
	for (const NodePath sync_key : sync_keys) {
		Dictionary sync_state_as_dict = sync_states_dict[sync_key];
		sync_states.insert(sync_key, SaveloadSynchronizer::SyncState(sync_state_as_dict));
	}
}

PackedByteArray SceneSaveloadInterface::encode(Object *p_obj, const StringName section) {
	//TODO: implement
	flush_spawn_queue();
	return packet_cache;
}
Error SceneSaveloadInterface::decode(PackedByteArray p_bytes, Object *p_obj, const StringName section) {
	return ERR_UNAVAILABLE; //TODO: not sure what to do here
}
