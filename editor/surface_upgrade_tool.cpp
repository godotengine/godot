/**************************************************************************/
/*  surface_upgrade_tool.cpp                                              */
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

#include "surface_upgrade_tool.h"

#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "scene/scene_string_names.h"
#include "servers/rendering_server.h"

SurfaceUpgradeTool *SurfaceUpgradeTool::singleton = nullptr;

void SurfaceUpgradeTool::_add_files(EditorFileSystemDirectory *p_dir, Vector<String> &r_reimport_paths, Vector<String> &r_resave_paths) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_add_files(p_dir->get_subdir(i), r_reimport_paths, r_resave_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "Mesh" ||
				p_dir->get_file_type(i) == "ArrayMesh" ||
				p_dir->get_file_type(i) == "PackedScene") {
			if (FileAccess::exists(p_dir->get_file_path(i) + ".import")) {
				r_reimport_paths.append(p_dir->get_file_path(i) + ".import");
			} else {
				r_resave_paths.append(p_dir->get_file_path(i));
			}
		}
	}
}

void SurfaceUpgradeTool::_try_show_popup() {
	if (singleton->show_requested || singleton->popped_up) {
		return;
	}
	singleton->show_requested = true;

	RS::get_singleton()->set_warn_on_surface_upgrade(false);

	if (EditorFileSystem::get_singleton()->is_importing()) {
		EditorFileSystem::get_singleton()->connect("resources_reimported", callable_mp(singleton, &SurfaceUpgradeTool::_show_popup), CONNECT_ONE_SHOT);
	} else if (EditorNode::get_singleton()->is_inside_tree()) {
		singleton->_show_popup();
	}

	// EditorNode may not be ready yet. It will call this tool when it is.
}

void SurfaceUpgradeTool::_show_popup() {
	MutexLock lock(mutex);
	if (!show_requested || popped_up) {
		return;
	}
	show_requested = false;
	popped_up = true;

	bool accepted = EditorNode::immediate_confirmation_dialog(TTR("This project uses meshes with an outdated mesh format from previous Godot versions. The engine needs to update the format in order to use those meshes.\n\nPress 'Restart & Upgrade' to run the surface upgrade tool which will update and re-save all meshes and scenes. This update will restart the editor and may take several minutes. Upgrading will make the meshes incompatible with previous versions of Godot.\n\nPress 'Upgrade Only' to continue opening the scene as normal. The engine will update each mesh in memory, but the update will not be saved. Choosing this option will lead to slower load times every time this project is loaded."), TTR("Restart & Upgrade"), TTR("Upgrade Only"), 500);
	if (accepted) {
		EditorSettings::get_singleton()->set_project_metadata("surface_upgrade_tool", "run_on_restart", true);

		Vector<String> reimport_paths;
		Vector<String> resave_paths;
		_add_files(EditorFileSystem::get_singleton()->get_filesystem(), reimport_paths, resave_paths);

		EditorSettings::get_singleton()->set_project_metadata("surface_upgrade_tool", "reimport_paths", reimport_paths);
		EditorSettings::get_singleton()->set_project_metadata("surface_upgrade_tool", "resave_paths", resave_paths);

		// Delay to avoid deadlocks, since this dialog can be triggered by loading a scene.
		MessageQueue::get_singleton()->push_callable(callable_mp(EditorNode::get_singleton(), &EditorNode::restart_editor));
	} else {
		RS::get_singleton()->set_warn_on_surface_upgrade(true);
	}
}

// Ensure that the warnings and popups are skipped.
void SurfaceUpgradeTool::begin_upgrade() {
	EditorSettings::get_singleton()->set_project_metadata("surface_upgrade_tool", "run_on_restart", false);
	RS::get_singleton()->set_surface_upgrade_callback(nullptr);
	RS::get_singleton()->set_warn_on_surface_upgrade(false);
	popped_up = true;
}

void SurfaceUpgradeTool::finish_upgrade() {
	EditorNode::get_singleton()->trigger_menu_option(EditorNode::FILE_CLOSE_ALL, true);

	// Update all meshes here.
	Vector<String> resave_paths = EditorSettings::get_singleton()->get_project_metadata("surface_upgrade_tool", "resave_paths", Vector<String>());
	EditorProgress ep("surface_upgrade_resave", TTR("Upgrading All Meshes in Project"), resave_paths.size());

	for (const String &file_path : resave_paths) {
		Ref<Resource> res = ResourceLoader::load(file_path);
		ep.step(TTR("Attempting to re-save ") + file_path);
		if (res.is_valid()) {
			// Ignore things that fail to load.
			ResourceSaver::save(res);
		}
	}
	EditorSettings::get_singleton()->set_project_metadata("surface_upgrade_tool", "resave_paths", Vector<String>());

	// Remove the imported scenes/meshes from .import so they will be reimported automatically after this.
	Vector<String> reimport_paths = EditorSettings::get_singleton()->get_project_metadata("surface_upgrade_tool", "reimport_paths", Vector<String>());
	for (const String &file_path : reimport_paths) {
		Ref<ConfigFile> config;
		config.instantiate();
		Error err = config->load(file_path);
		if (err != OK) {
			ERR_PRINT("Could not open " + file_path + " for upgrade.");
			continue;
		}

		String remap_path = config->get_value("remap", "path", "");
		if (remap_path.is_empty()) {
			continue;
		}

		String path = OS::get_singleton()->get_resource_dir() + remap_path.replace_first("res://", "/");
		print_verbose("Moving to trash: " + path);
		err = OS::get_singleton()->move_to_trash(path);
		if (err != OK) {
			EditorNode::get_singleton()->add_io_error(TTR("Cannot remove:") + "\n" + remap_path + "\n");
		}
	}
	EditorSettings::get_singleton()->set_project_metadata("surface_upgrade_tool", "reimport_paths", Vector<String>());

	emit_signal(SNAME("upgrade_finished"));
}

void SurfaceUpgradeTool::_bind_methods() {
	ADD_SIGNAL(MethodInfo("upgrade_finished"));
}

SurfaceUpgradeTool::SurfaceUpgradeTool() {
	singleton = this;
	RS::get_singleton()->set_surface_upgrade_callback(_try_show_popup);
}

SurfaceUpgradeTool::~SurfaceUpgradeTool() {}
