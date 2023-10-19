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
#include "servers/rendering_server.h"

void SurfaceUpgradeTool::_add_files(EditorFileSystemDirectory *p_dir, HashSet<String> &r_paths, PackedStringArray &r_files) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_add_files(p_dir->get_subdir(i), r_paths, r_files);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "Mesh" ||
				p_dir->get_file_type(i) == "ArrayMesh" ||
				p_dir->get_file_type(i) == "PackedScene") {
			if (FileAccess::exists(p_dir->get_file_path(i) + ".import")) {
				r_files.push_back(p_dir->get_file_path(i));
			} else {
				r_paths.insert(p_dir->get_file_path(i));
			}
		}
	}
}

void SurfaceUpgradeTool::upgrade_all_meshes() {
	// Update all meshes here.
	HashSet<String> paths;
	PackedStringArray files_to_import;
	_add_files(EditorFileSystem::get_singleton()->get_filesystem(), paths, files_to_import);

	EditorProgress ep("Re-saving all scenes and meshes", TTR("Upgrading All Meshes in Project"), paths.size());

	ep.step(TTR("Re-importing meshes"), 0);
	EditorFileSystem::get_singleton()->reimport_files(files_to_import);

	uint32_t step = 1;
	for (const String &file : paths) {
		Ref<Resource> res = ResourceLoader::load(file);
		ep.step(TTR("Attempting to re-save ") + file, step++);
		if (res.is_valid()) {
			// Ignore things that fail to load.
			ResourceSaver::save(res);
		}
	}
}

void SurfaceUpgradeTool::_show_popup() {
	RS::get_singleton()->set_surface_upgrade_callback(nullptr);
	bool accepted = EditorNode::immediate_confirmation_dialog(TTR("This project uses meshes with an outdated mesh format from previous Godot versions. The engine needs to update the format in order to use those meshes.\n\nPress 'Upgrade & Re-save' to have the engine scan the project folder and automatically update and re-save all meshes and scenes. This update may take a few minutes. Upgrading will make the meshes incompatible with previous versions of Godot.\n\nPress 'Upgrade Only' to continue opening the scene as normal. The engine will update each mesh in memory, but the update will not be saved. Choosing this option will lead to slower load times every time this project is loaded."), TTR("Upgrade & Re-save"), TTR("Upgrade Only"), 500);
	if (accepted) {
		RS::get_singleton()->set_warn_on_surface_upgrade(false);
		upgrade_all_meshes();
	}
}

SurfaceUpgradeTool::SurfaceUpgradeTool() {
	RS::get_singleton()->set_surface_upgrade_callback(_show_popup);
}

SurfaceUpgradeTool::~SurfaceUpgradeTool() {}
