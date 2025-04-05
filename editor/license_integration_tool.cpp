/**************************************************************************/
/*  license_integration_tool.cpp                                          */
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

#include "license_integration_tool.h"
#include "core/config/project_settings.h"
#include "core/input/input_map.h"
#include "core/io/dir_access.h"
#include "editor/addons/godot_license_notices_dialog_gds/godot_license_notices_dialog_gds.gen.h"
#include "editor/editor_autoload_settings.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/event_listener_line_edit.h"
#include "editor/project_settings_editor.h"
#include "editor/themes/editor_scale.h"

LicenseIntegrationDialog::LicenseIntegrationDialog() {
	set_autowrap(true);
	get_label()->set_custom_minimum_size(Size2(750 * EDSCALE, 0));
	learn_more_button = add_button(TTR("Learn More"), true, ACTION_LEARN_MORE);
}

void LicenseIntegrationDialog::_on_custom_action(const String &p_action) {
	if (p_action == ACTION_LEARN_MORE) {
		OS::get_singleton()->shell_open("https://docs.godotengine.org/en/stable/about/complying_with_licenses.html");
	}
}

void LicenseIntegrationDialog::_acknowledged() {
	EditorSettings::get_singleton()->set_project_metadata(META_LICENSE_INTEGRATION_TOOL, META_ACKNOWLEDGED, true);
}

void LicenseIntegrationDialog::_integrate_license() {
	String base_path = TARGET_DIR;

	files_succeeded = false;
	if (!DirAccess::dir_exists_absolute(TARGET_DIR)) {
		for (int i = 0; i < EDITOR_ADDONS_GODOT_LICENSE_NOTICES_DIALOG_GDS_FILE_COUNT; i++) {
			const char *rel_path = EDITOR_ADDONS_GODOT_LICENSE_NOTICES_DIALOG_GDS_FILE_PATHS[i];
			String path = base_path.path_join(rel_path);

			String dir = path.get_base_dir();
			if (!DirAccess::dir_exists_absolute(dir)) {
				DirAccess::make_dir_recursive_absolute(dir);
			}
			Error err;
			Ref<FileAccess> f = FileAccess::open(path, FileAccess::ModeFlags::WRITE, &err);
			ERR_FAIL_COND_MSG(f.is_null(), "Failed to write file: " + path + " Because of Error " + itos(err));

			const int length = EDITOR_ADDONS_GODOT_LICENSE_NOTICES_DIALOG_GDS_FILE_LENGTHS[i];
			const unsigned char *content = EDITOR_ADDONS_GODOT_LICENSE_NOTICES_DIALOG_GDS_FILE_CONTENTS[i];

			f->store_buffer(content, length);
		}
		files_succeeded = true;
	}

	EditorFileSystem::get_singleton()->connect("sources_changed", callable_mp(this, &LicenseIntegrationDialog::_on_sources_changed), CONNECT_ONE_SHOT);
	EditorFileSystem::get_singleton()->scan();
}
void LicenseIntegrationDialog::_on_sources_changed(bool p_changed) {
	_integrate_license_setup();
}

void LicenseIntegrationDialog::_integrate_license_setup() {
	String base_path = TARGET_DIR;

	bool autoload_succeeded = false;
	if (!ProjectSettings::get_singleton()->has_autoload(AUTOLOAD_NAME)) {
		ProjectSettings::AutoloadInfo autoload = ProjectSettings::AutoloadInfo();
		String name = "autoload/";
		name += AUTOLOAD_NAME;
		String path = "*";
		path += base_path.path_join(AUTOLOAD_FILE);
		ProjectSettings::get_singleton()->set_setting(name, path);
		ProjectSettingsEditor::get_singleton()->get_autoload_settings()->update_autoload();
		autoload_succeeded = true;
	}

	String name = "input/";
	name += KEYBIND_NAME;
	Ref<InputEvent> keybind_event = InputEventKey::create_reference(KEYBIND_KEY);
	bool keybind_succeeded = false;
	if (!ProjectSettings::get_singleton()->has_setting(name)) {
		Dictionary action;
		Array events;
		events.push_back(keybind_event);
		action["events"] = events;
		action["deadzone"] = InputMap::DEFAULT_DEADZONE;
		ProjectSettings::get_singleton()->set_setting(name, action);
		ProjectSettingsEditor::get_singleton()->update_action_map_editor();
		keybind_succeeded = true;
	}

	ProjectSettings::get_singleton()->save();

	String text;
	if (files_succeeded && autoload_succeeded && keybind_succeeded) {
		text = TTR("The configuration of the license notices was successful.\n\n");
	} else {
		text = TTR("The configuration of the license notices failed.\n\n");
	}

	int success_count = (files_succeeded ? 1 : 0) + (autoload_succeeded ? 1 : 0) + (keybind_succeeded ? 1 : 0);
	int fail_count = (!files_succeeded ? 1 : 0) + (!autoload_succeeded ? 1 : 0) + (!keybind_succeeded ? 1 : 0);

	if (success_count > 0) {
		text += TTRN("This has been added to your project:\n", "These have been added to your project:\n", success_count);
		if (files_succeeded) {
			text += vformat(TTR("    - An add-on at \"%s\".\n"), base_path);
		}
		if (autoload_succeeded) {
			text += vformat(TTR("    - An autoload named \"%s\".\n"), AUTOLOAD_NAME);
		}
		String shortcut = EventListenerLineEdit::get_event_text(keybind_event, false);
		if (keybind_succeeded) {
			text += vformat(TTR("    - An input map action \"%s\" with keyboard shortcut \"%s\".\n"), KEYBIND_NAME, shortcut);
		}
		if (success_count == 3) {
			text += "\n";
			text += vformat(TTR("The license notices will show up when the player presses \"%s\".\nIf this project is also for platforms where keyboard is not available, please add a button in your project to display license notices."), shortcut);
		}
	}

	if (fail_count > 0) {
		if (success_count > 0) {
			text += "\n";
		}
		text += TTRN("Step that has failed:\n", "Steps that have failed:\n", fail_count);
		if (!files_succeeded) {
			text += vformat(TTR("    - Add-on cannot be added because directory \"%s\" already exists.\n"), TARGET_DIR);
		}
		if (!autoload_succeeded) {
			text += vformat(TTR("    - Autoload \"%s\" cannot be set-up because it already exists.\n"), AUTOLOAD_NAME);
		}
		if (!keybind_succeeded) {
			text += vformat(TTR("    - Input map action \"%s\" cannot be added because it already exists.\n"), KEYBIND_NAME);
		}
		text += "\n";
		text += TTR("You can find this tool in Project > Tools sub-menu and re-run at any time.");
	}

	if (!result_dialog) {
		result_dialog = memnew(AcceptDialog);
		result_dialog->set_autowrap(true);
		result_dialog->get_label()->set_custom_minimum_size(Size2(700 * EDSCALE, 0));
		EditorNode::get_singleton()->get_gui_base()->add_child(result_dialog);
	}
	result_dialog->set_text(text);
	result_dialog->popup_centered(Size2(700 * EDSCALE, 0));
	_acknowledged();
}

void LicenseIntegrationDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			connect(SceneStringName(confirmed), callable_mp(this, &LicenseIntegrationDialog::_integrate_license));
			connect(SNAME("canceled"), callable_mp(this, &LicenseIntegrationDialog::_acknowledged));
			connect(SNAME("custom_action"), callable_mp(this, &LicenseIntegrationDialog::_on_custom_action));
			break;
	}
}

bool LicenseIntegrationDialog::should_prompt() {
	if (DirAccess::dir_exists_absolute(TARGET_DIR)) {
		return false;
	}
	bool acknowledged = EditorSettings::get_singleton()->get_project_metadata(META_LICENSE_INTEGRATION_TOOL, META_ACKNOWLEDGED, false);
	return !acknowledged;
}

void LicenseIntegrationDialog::popup_on_demand() {
	const String confirmation_message = TTR("Godot Engine relies on a number of third-party free and open source libraries, all compatible with the terms of its MIT license. You are required to bundle a copy of the license text with your game.\n\nThis license integration tool will add a license notice dialog add-on. The dialog will show up when a key combination is pressed. You should also add a button to show this dialog when the keyboard is not available for players.\n\nYou can access this tool later in the tools sub-menu.");
	set_text(confirmation_message);

	get_ok_button()->set_text("Auto-configure");
	get_cancel_button()->set_text("Don't Auto-configure");
	popup_centered(Size2(750 * EDSCALE, 0));
}
