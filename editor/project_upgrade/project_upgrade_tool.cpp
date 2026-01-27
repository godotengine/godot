/**************************************************************************/
/*  project_upgrade_tool.cpp                                              */
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

#include "project_upgrade_tool.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/scene/editor_scene_tabs.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/dialogs.h"

void ProjectUpgradeTool::_add_files(EditorFileSystemDirectory *p_dir, Vector<String> &r_reimport_paths, Vector<String> &r_resave_scenes, Vector<String> &r_resave_resources) {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	for (int i = 0; i < p_dir->get_file_count(); i++) {
		const String path = p_dir->get_file_path(i);
		const String ext = path.get_extension();
		if (ext == "tscn" || ext == "scn") {
			r_resave_scenes.append(path);
		} else if (ext == "tres" || ext == "res") {
			r_resave_resources.append(path);
		} else if (da->file_exists(path + ".import")) {
			r_reimport_paths.append(path);
		}
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_add_files(p_dir->get_subdir(i), r_reimport_paths, r_resave_scenes, r_resave_resources);
	}
}

void ProjectUpgradeTool::_bind_methods() {
	ADD_SIGNAL(MethodInfo("upgrade_finished"));
}

void ProjectUpgradeTool::popup_dialog() {
	if (!upgrade_dialog) {
		upgrade_dialog = memnew(ConfirmationDialog);
		upgrade_dialog->set_autowrap(true);
		upgrade_dialog->set_text(TTRC("Different engine version may have minor differences in various Resources, like additional fields or removed properties. When upgrading the project to a new version, such changes can cause diffs when saving scenes or resources, or reimporting.\n\nThis tool ensures that such changes are performed all at once. It will:\n- Regenerate UID cache\n- Load and re-save every text/binary Resource\n- Reimport every importable Resource\n\nFull upgrade will take considerable amount of time, but afterwards saving/reimporting any scene/resource should not cause unintended changes."));
		upgrade_dialog->get_ok_button()->set_text(TTRC("Restart & Upgrade"));
		upgrade_dialog->get_label()->set_custom_minimum_size(Size2(750 * EDSCALE, 0));
		EditorNode::get_singleton()->get_gui_base()->add_child(upgrade_dialog);
		upgrade_dialog->connect(SceneStringName(confirmed), callable_mp(this, &ProjectUpgradeTool::prepare_upgrade));
	}
	upgrade_dialog->popup_centered();
}

void ProjectUpgradeTool::prepare_upgrade() {
	EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RUN_ON_RESTART, true);

#ifndef DISABLE_DEPRECATED
	ProjectSettings::get_singleton()->set_setting("animation/compatibility/default_parent_skeleton_in_mesh_instance_3d", false);
	ProjectSettings::get_singleton()->save();
#endif

	Vector<String> reimport_paths;
	Vector<String> resave_scenes;
	Vector<String> resave_resources;
	_add_files(EditorFileSystem::get_singleton()->get_filesystem(), reimport_paths, resave_scenes, resave_resources);

	EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_REIMPORT_PATHS, reimport_paths);
	EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RESAVE_SCENES, resave_scenes);
	EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RESAVE_RESOURCES, resave_resources);

	// Delay to avoid deadlocks, since this dialog can be triggered by loading a scene.
	callable_mp(EditorNode::get_singleton(), &EditorNode::restart_editor).call_deferred(false);
}

void ProjectUpgradeTool::begin_upgrade() {
	EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RUN_ON_RESTART, false);
	DirAccess::remove_absolute("res://.godot/uid_cache.bin");
}

void ProjectUpgradeTool::finish_upgrade() {
	EditorNode::get_singleton()->trigger_menu_option(EditorNode::SCENE_CLOSE_ALL, true);

	Vector<String> paths = EditorSettings::get_singleton()->get_project_metadata(META_PROJECT_UPGRADE_TOOL, META_REIMPORT_PATHS, Vector<String>());
	EditorFileSystem::get_singleton()->reimport_files(paths);
	EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_REIMPORT_PATHS, Variant());

#ifndef DISABLE_DEPRECATED
	MeshInstance3D::upgrading_skeleton_compat = true;
#endif

	{
		paths = EditorSettings::get_singleton()->get_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RESAVE_SCENES, Vector<String>());
		EditorProgress ep("uid_upgrade_resave", TTR("Updating Project Scenes"), paths.size());

		int step = 0;
		for (const String &file_path : paths) {
			ep.step(TTR("Re-saving scene:") + " " + file_path, step++);
			EditorNode::get_singleton()->load_scene(file_path);
			EditorNode::get_singleton()->trigger_menu_option(EditorNode::SCENE_SAVE_SCENE, true);
			EditorNode::get_singleton()->trigger_menu_option(EditorNode::SCENE_CLOSE, true);
		}
		EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RESAVE_SCENES, Variant());
	}

#ifndef DISABLE_DEPRECATED
	MeshInstance3D::upgrading_skeleton_compat = false;
#endif

	{
		paths = EditorSettings::get_singleton()->get_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RESAVE_RESOURCES, Vector<String>());
		EditorProgress ep("uid_upgrade_resave", TTR("Updating Project Resources"), paths.size());

		int step = 0;
		for (const String &file_path : paths) {
			ep.step(TTR("Re-saving resource:") + " " + file_path, step++);
			Ref<Resource> res = ResourceLoader::load(file_path, "", ResourceFormatLoader::CACHE_MODE_REPLACE);
			if (res.is_valid()) {
				ResourceSaver::save(res);
			}
		}
		EditorSettings::get_singleton()->set_project_metadata(META_PROJECT_UPGRADE_TOOL, META_RESAVE_RESOURCES, Variant());
	}

	emit_signal(UPGRADE_FINISHED);
}
