/*************************************************************************/
/*  editor_paths.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_paths.h"
#include "core/os/dir_access.h"
#include "core/os/os.h"

EditorPaths *EditorPaths::singleton = nullptr;

bool EditorPaths::are_paths_valid() const {
	return paths_valid;
}

String EditorPaths::get_data_dir() const {
	return data_dir;
}
String EditorPaths::get_config_dir() const {
	return config_dir;
}
String EditorPaths::get_cache_dir() const {
	return cache_dir;
}
bool EditorPaths::is_self_contained() const {
	return self_contained;
}
String EditorPaths::get_self_contained_file() const {
	return self_contained_file;
}

void EditorPaths::create(bool p_for_project_manager) {
	ERR_FAIL_COND(singleton != nullptr);
	memnew(EditorPaths(p_for_project_manager));
}
void EditorPaths::free() {
	ERR_FAIL_COND(singleton == nullptr);
	memdelete(singleton);
}

void EditorPaths::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_data_dir"), &EditorPaths::get_data_dir);
	ClassDB::bind_method(D_METHOD("get_config_dir"), &EditorPaths::get_config_dir);
	ClassDB::bind_method(D_METHOD("get_cache_dir"), &EditorPaths::get_cache_dir);
	ClassDB::bind_method(D_METHOD("is_self_contained"), &EditorPaths::is_self_contained);
	ClassDB::bind_method(D_METHOD("get_self_contained_file"), &EditorPaths::get_self_contained_file);
}

EditorPaths::EditorPaths(bool p_for_project_mamanger) {
	singleton = this;

	String exe_path = OS::get_singleton()->get_executable_path().get_base_dir();
	{
		DirAccessRef d = DirAccess::create_for_path(exe_path);

		if (d->file_exists(exe_path + "/._sc_")) {
			self_contained = true;
			self_contained_file = exe_path + "/._sc_";
		} else if (d->file_exists(exe_path + "/_sc_")) {
			self_contained = true;
			self_contained_file = exe_path + "/_sc_";
		}
	}

	String data_path;
	String config_path;
	String cache_path;

	if (self_contained) {
		// editor is self contained, all in same folder
		data_path = exe_path;
		data_dir = data_path.plus_file("editor_data");
		config_path = exe_path;
		config_dir = data_dir;
		cache_path = exe_path;
		cache_dir = data_dir.plus_file("cache");
	} else {
		// Typically XDG_DATA_HOME or %APPDATA%
		data_path = OS::get_singleton()->get_data_path();
		data_dir = data_path.plus_file(OS::get_singleton()->get_godot_dir_name());
		// Can be different from data_path e.g. on Linux or macOS
		config_path = OS::get_singleton()->get_config_path();
		config_dir = config_path.plus_file(OS::get_singleton()->get_godot_dir_name());
		// Can be different from above paths, otherwise a subfolder of data_dir
		cache_path = OS::get_singleton()->get_cache_path();
		if (cache_path == data_path) {
			cache_dir = data_dir.plus_file("cache");
		} else {
			cache_dir = cache_path.plus_file(OS::get_singleton()->get_godot_dir_name());
		}
	}

	paths_valid = (data_path != "" && config_path != "" && cache_path != "");

	if (paths_valid) {
		DirAccessRef dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (dir->change_dir(data_dir) != OK) {
			dir->make_dir_recursive(data_dir);
			if (dir->change_dir(data_dir) != OK) {
				ERR_PRINT("Cannot create data directory!");
				paths_valid = false;
			}
		}

		// Validate/create cache dir

		if (dir->change_dir(EditorPaths::get_singleton()->get_cache_dir()) != OK) {
			dir->make_dir_recursive(cache_dir);
			if (dir->change_dir(cache_dir) != OK) {
				ERR_PRINT("Cannot create cache directory!");
			}
		}

		if (p_for_project_mamanger) {
			Engine::get_singleton()->set_shader_cache_path(get_data_dir());
		} else {
			DirAccessRef dir2 = DirAccess::open("res://");
			if (dir2->change_dir(".godot") != OK) { //ensure the .godot subdir exists
				if (dir2->make_dir(".godot") != OK) {
					ERR_PRINT("Cannot create res://.godot directory!");
				}
			}

			Engine::get_singleton()->set_shader_cache_path("res://.godot");
		}
	}
}
