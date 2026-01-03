/**************************************************************************/
/*  scene_cache.cpp                                                       */
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

#include "scene_cache.h"

#include "godot_lsp.h"

#include "core/io/resource_loader.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include "editor/file_system/editor_file_system.h"
#include "scene/resources/packed_scene.h"

void SceneCache::_get_owner_paths(EditorFileSystemDirectory *p_dir, const String &p_script_path, LocalVector<String> &r_owner_paths) {
	if (!p_dir) {
		return;
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_get_owner_paths(p_dir->get_subdir(i), p_script_path, r_owner_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_deps(i).has(p_script_path)) {
			r_owner_paths.push_back(p_dir->get_file_path(i));
		}
	}
}

void SceneCache::_finalize_scene_load() {
	ERR_FAIL_COND(current_loaded_owner.is_empty() || script_path_queue.is_empty());

	Ref<PackedScene> scene_res = ResourceLoader::load_threaded_get(current_loaded_owner);

	if (scene_res.is_valid()) {
		cache[script_path_queue[0]] = scene_res->instantiate();
	} else {
		cache[script_path_queue[0]] = nullptr;
	}

	LOG_LSP("Scene cached for script:", script_path_queue[0]);
	LOG_LSP("pending_script_queue length:", script_path_queue.size() - 1);

	script_path_queue.remove_at(0);
	current_loaded_owner = String();
}

void SceneCache::poll() {
	if (current_loaded_owner.is_empty()) {
		// No load ongoing, start the next one.

		if (EditorFileSystem::get_singleton()->is_scanning() || script_path_queue.is_empty()) {
			return;
		}

		LocalVector<String> owners;
		_get_owner_paths(EditorFileSystem::get_singleton()->get_filesystem(), script_path_queue[0], owners);
		for (const String &owner : owners) {
			if (ResourceLoader::load_threaded_request(owner) == Error::OK) {
				current_loaded_owner = owner;
				LOG_LSP("Scene load started for:", current_loaded_owner);
				break;
			}
		}

		if (current_loaded_owner.is_empty()) {
			cache[script_path_queue[0]] = nullptr;
			LOG_LSP("No scene found for script:", script_path_queue[0]);
			script_path_queue.remove_at(0);
			LOG_LSP("pending_script_queue length:", script_path_queue.size());
		}
	} else {
		ERR_FAIL_COND(script_path_queue.is_empty());

		// There is an ongoing load. Check the status.

		ResourceLoader::ThreadLoadStatus status = ResourceLoader::load_threaded_get_status(current_loaded_owner);

		if (status == ResourceLoader::THREAD_LOAD_IN_PROGRESS) {
			return;
		}

		if (status == ResourceLoader::THREAD_LOAD_LOADED) {
			_finalize_scene_load();
		} else {
			LOG_LSP("Scene load failure for:", current_loaded_owner);
			cache[script_path_queue[0]] = nullptr;

			script_path_queue.remove_at(0);
			current_loaded_owner = String();
		}
	}
}

Node *SceneCache::get(const String &p_script_path) {
	if (!script_path_queue.is_empty() && script_path_queue[0] == p_script_path && !current_loaded_owner.is_empty()) {
		_finalize_scene_load();
	} else {
		script_path_queue.erase(p_script_path);
	}

	if (Node **entry = cache.getptr(p_script_path)) {
		return *entry;
	}

	// Fallback to blocking load. This could happen if the open request was only recently sent.
	// TODO: This could also happen when multiple clients are connected.

	LocalVector<String> owners;
	_get_owner_paths(EditorFileSystem::get_singleton()->get_filesystem(), p_script_path, owners);
	for (const String &owner : owners) {
		Ref<PackedScene> scene = ResourceLoader::load(owner);
		if (scene.is_valid()) {
			Node *instance = scene->instantiate();
			cache[p_script_path] = instance;
			return instance;
		}
	}

	cache[p_script_path] = nullptr;
	return nullptr;
}

void SceneCache::request_load(const String &p_script_path) {
	if (!cache.has(p_script_path) && !script_path_queue.has(p_script_path)) {
		script_path_queue.push_back(p_script_path);
		LOG_LSP("Scene load requested for:", p_script_path);
		LOG_LSP("pending_script_queue length:", script_path_queue.size());
	}
}

void SceneCache::unload(const String &p_script_path) {
	if (!script_path_queue.is_empty() && script_path_queue[0] == p_script_path && !current_loaded_owner.is_empty()) {
		_ALLOW_DISCARD_ ResourceLoader::load_threaded_get(current_loaded_owner);

		script_path_queue.remove_at(0);
		current_loaded_owner = String();
	} else {
		script_path_queue.erase(p_script_path);
	}

	if (!cache.has(p_script_path)) {
		return;
	}
	memdelete_notnull(cache[p_script_path]);
	cache.erase(p_script_path);
	LOG_LSP("Cache cleared for path:", p_script_path);
}

void SceneCache::clear() {
	if (!current_loaded_owner.is_empty()) {
		_ALLOW_DISCARD_ ResourceLoader::load_threaded_get(current_loaded_owner);
		current_loaded_owner = String();
	}
	script_path_queue.clear();
	for (const KeyValue<String, Node *> &E : cache) {
		memdelete_notnull(E.value);
	}
	cache.clear();
	LOG_LSP("Cache cleared.");
}
