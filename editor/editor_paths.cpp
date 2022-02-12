/*************************************************************************/
/*  editor_paths.cpp                                                     */
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

#include "editor_paths.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "main/main.h" // For `is_project_manager`.

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

String EditorPaths::get_project_data_dir() const {
	return project_data_dir;
}

bool EditorPaths::is_self_contained() const {
	return self_contained;
}

String EditorPaths::get_self_contained_file() const {
	return self_contained_file;
}

void EditorPaths::create() {
	ERR_FAIL_COND(singleton != nullptr);
	memnew(EditorPaths());
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

EditorPaths::EditorPaths() {
	singleton = this;

	project_data_dir = ProjectSettings::get_singleton()->get_project_data_path();

	// Self-contained mode if a `._sc_` or `_sc_` file is present in executable dir.
	String exe_path = OS::get_singleton()->get_executable_path().get_base_dir();

	// On macOS, look outside .app bundle, since .app bundle is read-only.
	if (OS::get_singleton()->has_feature("macos") && exe_path.ends_with("MacOS") && exe_path.plus_file("..").simplify_path().ends_with("Contents")) {
		exe_path = exe_path.plus_file("../../..").simplify_path();
	}
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
		// Typically XDG_DATA_HOME or %APPDATA%.
		data_path = OS::get_singleton()->get_data_path();
		data_dir = data_path.plus_file(OS::get_singleton()->get_godot_dir_name());
		// Can be different from data_path e.g. on Linux or macOS.
		config_path = OS::get_singleton()->get_config_path();
		config_dir = config_path.plus_file(OS::get_singleton()->get_godot_dir_name());
		// Can be different from above paths, otherwise a subfolder of data_dir.
		cache_path = OS::get_singleton()->get_cache_path();
		if (cache_path == data_path) {
			cache_dir = data_dir.plus_file("cache");
		} else {
			cache_dir = cache_path.plus_file(OS::get_singleton()->get_godot_dir_name());
		}
	}

	paths_valid = (!data_path.is_empty() && !config_path.is_empty() && !cache_path.is_empty());
	ERR_FAIL_COND_MSG(!paths_valid, "Editor data, config, or cache paths are invalid.");

	// Validate or create each dir and its relevant subdirectories.

	DirAccessRef dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	// Data dir.
	{
		if (dir->change_dir(data_dir) != OK) {
			dir->make_dir_recursive(data_dir);
			if (dir->change_dir(data_dir) != OK) {
				ERR_PRINT("Could not create editor data directory: " + data_dir);
				paths_valid = false;
			}
		}

		if (!dir->dir_exists("templates")) {
			dir->make_dir("templates");
		}
	}

	// Config dir.
	{
		if (dir->change_dir(config_dir) != OK) {
			dir->make_dir_recursive(config_dir);
			if (dir->change_dir(config_dir) != OK) {
				ERR_PRINT("Could not create editor config directory: " + config_dir);
				paths_valid = false;
			}
		}

		if (!dir->dir_exists("text_editor_themes")) {
			dir->make_dir("text_editor_themes");
		}
		if (!dir->dir_exists("script_templates")) {
			dir->make_dir("script_templates");
		}
		if (!dir->dir_exists("feature_profiles")) {
			dir->make_dir("feature_profiles");
		}
	}

	// Cache dir.
	{
		if (dir->change_dir(cache_dir) != OK) {
			dir->make_dir_recursive(cache_dir);
			if (dir->change_dir(cache_dir) != OK) {
				ERR_PRINT("Could not create editor cache directory: " + cache_dir);
				paths_valid = false;
			}
		}
	}

	// Validate or create project-specific editor data dir,
	// including shader cache subdir.

	if (Main::is_project_manager() || Main::is_cmdline_tool()) {
		// Nothing to create, use shared editor data dir for shader cache.
		Engine::get_singleton()->set_shader_cache_path(data_dir);
	} else {
		DirAccessRef dir_res = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (dir_res->change_dir(project_data_dir) != OK) {
			dir_res->make_dir_recursive(project_data_dir);
			if (dir_res->change_dir(project_data_dir) != OK) {
				ERR_PRINT("Could not create project data directory (" + project_data_dir + ") in: " + dir_res->get_current_dir());
				paths_valid = false;
			}
		}

		// Check that the project data directory '.gdignore' file exists
		String project_data_gdignore_file_path = project_data_dir.plus_file(".gdignore");
		if (!FileAccess::exists(project_data_gdignore_file_path)) {
			// Add an empty .gdignore file to avoid scan.
			FileAccessRef f = FileAccess::open(project_data_gdignore_file_path, FileAccess::WRITE);
			if (f) {
				f->store_line("");
				f->close();
			} else {
				ERR_PRINT("Failed to create file " + project_data_gdignore_file_path);
			}
		}

		Engine::get_singleton()->set_shader_cache_path(project_data_dir);

		// Editor metadata dir.
		if (!dir_res->dir_exists("editor")) {
			dir_res->make_dir("editor");
		}
		// Imported assets dir.
		String imported_files_path = ProjectSettings::get_singleton()->get_imported_files_path();
		if (!dir_res->dir_exists(imported_files_path)) {
			dir_res->make_dir(imported_files_path);
		}
	}
}
