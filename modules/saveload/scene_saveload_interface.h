/**************************************************************************/
/*  scene_saveload_interface.h                                            */
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

#ifndef SCENE_SAVELOAD_INTERFACE_H
#define SCENE_SAVELOAD_INTERFACE_H

#include "core/object/ref_counted.h"

#include "saveload_spawner.h"
#include "saveload_synchronizer.h"

class SceneSaveload;

class SceneSaveloadInterface : public RefCounted {
	GDCLASS(SceneSaveloadInterface, RefCounted);

public:
	struct SaveloadState {
		HashMap<const NodePath, SaveloadSpawner::SpawnerState> spawner_states;
		HashMap<const NodePath, SaveloadSynchronizer::SyncherState> syncher_states;

		Dictionary to_dict() const;

		SaveloadState() {}
		SaveloadState(const Dictionary &saveload_dict);
	};

	HashSet<ObjectID> spawners;
	HashSet<ObjectID> synchers;

private:
	// Replicator config.
	SceneSaveload *saveload = nullptr;

	void _node_ready(const ObjectID &p_oid);

	template <class T>
	static T *get_id_as(const ObjectID &p_id) {
		return p_id.is_valid() ? Object::cast_to<T>(ObjectDB::get_instance(p_id)) : nullptr;
	}

#ifdef DEBUG_ENABLED
	_FORCE_INLINE_ void _profile_node_data(const String &p_what, ObjectID p_id, int p_size);
#endif

public:
	Error track(Object *p_object);
	Error untrack(Object *p_object);
	void track_spawner(const SaveloadSpawner &p_spawner);
	void untrack_spawner(const SaveloadSpawner &p_spawner);
	void track_syncher(const SaveloadSynchronizer &p_syncher);
	void untrack_syncher(const SaveloadSynchronizer &p_syncher);

	TypedArray<SaveloadSpawner> get_spawners() const;
	TypedArray<SaveloadSynchronizer> get_synchers() const;
	Dictionary get_spawn_dict() const;
	Dictionary get_sync_dict() const;

	SaveloadState get_saveload_state() const;
	Error load_saveload_state(const SaveloadState &p_saveload_state);

	SceneSaveloadInterface(SceneSaveload *p_saveload) {
		saveload = p_saveload;
	}
};

#endif // SCENE_SAVELOAD_INTERFACE_H
