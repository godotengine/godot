/**************************************************************************/
/*  uid_upgrade_tool.cpp                                                  */
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

#include "uid_upgrade_tool.h"

#include "editor/editor_file_system.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/gui/editor_scene_tabs.h"
#include "editor/gui/editor_toaster.h"
#include "editor/themes/editor_scale.h"
#include "servers/rendering_server.h"

void UIDUpgradeTool::_add_files(EditorFileSystemDirectory *p_dir, Vector<String> &r_resave_paths) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_add_files(p_dir->get_subdir(i), r_resave_paths);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "PackedScene" || p_dir->get_file_type(i) == "Resource") {
			r_resave_paths.append(p_dir->get_file_path(i));
		}
	}
}

void UIDUpgradeTool::prepare_upgrade() {
	EditorSettings::get_singleton()->set_project_metadata(META_UID_UPGRADE_TOOL, META_RUN_ON_RESTART, true);
	Vector<String> resave_paths;
	_add_files(EditorFileSystem::get_singleton()->get_filesystem(), resave_paths);

	EditorSettings::get_singleton()->set_project_metadata(META_UID_UPGRADE_TOOL, META_RESAVE_PATHS, resave_paths);

	// Delay to avoid deadlocks, since this dialog can be triggered by loading a scene.
	callable_mp(EditorNode::get_singleton(), &EditorNode::restart_editor).call_deferred(false);
}

void UIDUpgradeTool::begin_upgrade() {
	EditorSettings::get_singleton()->set_project_metadata(META_UID_UPGRADE_TOOL, META_RUN_ON_RESTART, false);
}

void UIDUpgradeTool::finish_upgrade() {
	EditorNode::get_singleton()->trigger_menu_option(EditorSceneTabs::SCENE_CLOSE_ALL, true);

	Vector<String> resave_paths = EditorSettings::get_singleton()->get_project_metadata(META_UID_UPGRADE_TOOL, META_RESAVE_PATHS, Variant());
	EditorProgress ep("uid_upgrade_resave", TTR("Updating Script UIDs"), resave_paths.size());

	int step = 0;
	for (const String &file_path : resave_paths) {
		Ref<Resource> res = ResourceLoader::load(file_path, "", ResourceFormatLoader::CACHE_MODE_REPLACE);
		ep.step(TTR("Attempting to re-save ") + file_path, step++);
		if (res.is_valid()) {
			ResourceSaver::save(res);
		}
	}
	EditorSettings::get_singleton()->set_project_metadata(META_UID_UPGRADE_TOOL, META_RESAVE_PATHS, Vector<String>());

	emit_signal(SNAME(UPGRADE_FINISHED));
}

void UIDUpgradeTool::_bind_methods() {
	ADD_SIGNAL(MethodInfo(UPGRADE_FINISHED));
}

UIDUpgradeTool::UIDUpgradeTool() {
	singleton = this;
}

UIDUpgradeTool::~UIDUpgradeTool() {
	singleton = nullptr;
}

void UIDUpgradeDialog::popup_on_demand() {
	const String confirmation_message = TTR("As of Godot 4.4, scenes and resources use UIDs to reference scripts and shaders.\n\nNormally, this update is applied to a single scene or resource once you save it in Godot 4.4 for the first time. If you have a lot of scenes and/or resources, doing this manually may be time-consuming. This tool will update all of the project's scenes and resources at once.\n\nClick \"Restart & Upgrade\" to restart the editor and update all of the scenes and resources in this project. Depending on the project size, it may take several minutes.\n\nNote: Please make sure you have your project backed up before running the tool, to avoid the possibility of data loss. Additionally, make sure to commit all .uid files into version control (and do not add them to ignore-lists like .gitignore).");
	set_text(confirmation_message);
	get_ok_button()->set_text("Restart & Upgrade");
	popup_centered(Size2(750 * EDSCALE, 0));
}

void UIDUpgradeDialog::_on_custom_action(const String &p_action) {
	if (p_action == UID_UPGRADE_LEARN_MORE) {
		OS::get_singleton()->shell_open("https://godotengine.org/article/uid-changes-coming-to-godot-4-4/");
	}
}

void UIDUpgradeDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			connect(SceneStringName(confirmed), callable_mp(UIDUpgradeTool::get_singleton(), &UIDUpgradeTool::prepare_upgrade));
			connect(SNAME("custom_action"), callable_mp(this, &UIDUpgradeDialog::_on_custom_action));
			break;
	}
}

UIDUpgradeDialog::UIDUpgradeDialog() {
	set_autowrap(true);
	get_label()->set_custom_minimum_size(Size2(750 * EDSCALE, 0));
	learn_more_button = add_button(TTR("Learn More"), true, UID_UPGRADE_LEARN_MORE);
}
