/**************************************************************************/
/*  project_dialog.cpp                                                    */
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

#include "project_dialog.h"

#include "core/io/dir_access.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "core/version.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/project_manager/project_operations.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "editor/version_control/editor_vcs_interface.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/link_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"

void ProjectDialog::_set_message(const String &p_msg, MessageType p_type, InputType p_input_type) {
	msg->set_text(p_msg);

	if (p_type == MESSAGE_ERROR) {
		invalid_state_flags.set_flag(InvalidStateFlag::INVALID_STATE_FLAG_PATH_INPUT);
	} else {
		invalid_state_flags.clear_flag(InvalidStateFlag::INVALID_STATE_FLAG_PATH_INPUT);
	}

	Ref<Texture2D> new_icon;
	switch (p_type) {
		case MESSAGE_ERROR: {
			msg->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
			new_icon = get_editor_theme_icon(SNAME("StatusError"));
		} break;
		case MESSAGE_WARNING: {
			msg->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
			new_icon = get_editor_theme_icon(SNAME("StatusWarning"));
		} break;
		case MESSAGE_SUCCESS: {
			msg->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("success_color"), EditorStringName(Editor)));
			new_icon = get_editor_theme_icon(SNAME("StatusSuccess"));
		} break;
	}

	if (p_input_type == ORIGIN_PATH) {
		origin_status_rect->set_texture(new_icon);
	} else if (p_input_type == DESTINATION_PATH) {
		destination_status_rect->set_texture(new_icon);
	}

	_update_ok_button();
}

void ProjectDialog::_update_ok_button() {
	get_ok_button()->set_disabled(!invalid_state_flags.is_empty());
}

void ProjectDialog::_validate_path() {
	_set_message("", MESSAGE_SUCCESS, ORIGIN_PATH);
	_set_message("", MESSAGE_SUCCESS, DESTINATION_PATH);

	if (project_name->get_text().strip_edges().is_empty()) {
		_set_message(TTRC("It would be a good idea to name your project."), MESSAGE_ERROR, DESTINATION_PATH);
		return;
	}
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	String chosen_origin_path;
	String chosen_destination_path;

	chosen_origin_path = origin_path->get_text().simplify_path();
	const bool validate_origin = origin_path_container->is_visible() && origin_path->is_editable();

	chosen_destination_path = destination_path->get_text().simplify_path();
	const bool validate_destination = destination_path_container->is_visible() && destination_path->is_editable();

	if (validate_origin && !chosen_origin_path.is_empty()) {
		if (!_pre_validate_path(chosen_origin_path, ORIGIN_PATH)) {
			return;
		}

		Error err;
		if (is_origin_zip) {
			err = ProjectOperations::validate_origin_zip(chosen_origin_path, project_root);

			if (err == ERR_CANT_OPEN) {
				_set_message(TTRC("Invalid \".zip\" project file; it is not in ZIP format."), MESSAGE_ERROR, ORIGIN_PATH);
				return;
			} else if (err == ERR_FILE_NOT_FOUND) {
				_set_message(TTRC("Invalid \".zip\" project file; it doesn't contain a \"project.godot\" file."), MESSAGE_ERROR, ORIGIN_PATH);
				return;
			} else if (err != OK) {
				return;
			}
		} else {
			err = ProjectOperations::validate_origin_folder(chosen_origin_path);
			if (err != OK) {
				_set_message(TTRC("Please choose a \"project.godot\", a directory with one, or a \".zip\" file."), MESSAGE_ERROR, ORIGIN_PATH);
				return;
			}
			_set_message(TTRC("Valid project found at path."), MESSAGE_SUCCESS, ORIGIN_PATH);
		}
	}

	bool is_working_on_origin = (chosen_destination_path == chosen_origin_path);

	if (validate_destination) {
		if (!_pre_validate_path(chosen_destination_path, DESTINATION_PATH)) {
			return;
		}

		String working_dir = d->get_current_dir();
		String executable_dir = OS::get_singleton()->get_executable_path().get_base_dir();

		if (chosen_destination_path == working_dir || chosen_destination_path == executable_dir) {
			_set_message(TTRC("Creating a project at the engine's working directory or executable directory is not allowed, as it would prevent the project manager from starting."), MESSAGE_ERROR, DESTINATION_PATH);
			return;
		}

// TODO: The following 5 lines could be simplified if OS.get_user_home_dir() or SYSTEM_DIR_HOME is implemented. See: https://github.com/godotengine/godot-proposals/issues/4851.
#ifdef WINDOWS_ENABLED
		String home_dir = OS::get_singleton()->get_environment("USERPROFILE");
#else
		String home_dir = OS::get_singleton()->get_environment("HOME");
#endif
		String documents_dir = OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DOCUMENTS);
		if (chosen_destination_path == home_dir || chosen_destination_path == documents_dir) {
			_set_message(TTRC("You cannot save a project at the selected path. Please create a subfolder or choose a new path."), MESSAGE_ERROR, DESTINATION_PATH);
			return;
		}

		if (!is_working_on_origin) {
			is_folder_empty = true;
			bool is_create_dir_pressed = create_dir->is_pressed();
			bool created_dir_exists = true;

			Error err;
			err = ProjectOperations::validate_destination(chosen_destination_path, is_folder_empty, is_create_dir_pressed, &created_dir_exists);

			if (err == ERR_FILE_NOT_FOUND) {
				_set_message(is_create_dir_pressed ? TTRC("The parent directory of the path specified doesn't exist.") : TTRC("The path specified doesn't exist."), MESSAGE_ERROR, DESTINATION_PATH);
				return;
			}

			if (!is_folder_empty) {
				_set_message(TTRC("The selected path is not empty. Choosing an empty folder is highly recommended."), MESSAGE_WARNING, DESTINATION_PATH);
			} else if (!created_dir_exists) {
				_set_message(TTRC("The project folder will be automatically created."), MESSAGE_SUCCESS, DESTINATION_PATH);
			} else {
				_set_message(TTRC("The project folder already exists and is empty."), MESSAGE_SUCCESS, DESTINATION_PATH);
			}
		}
	}

	if (!chosen_origin_path.is_empty() && !is_working_on_origin) {
		if (ProjectOperations::is_destination_inside_origin(chosen_destination_path, chosen_origin_path)) {
			_set_message(TTRC("Cannot duplicate a project into itself."), MESSAGE_ERROR, DESTINATION_PATH);
			return;
		}
	}
}

bool ProjectDialog::_pre_validate_path(const String &p_path, const InputType &p_type) {
	if (p_path.is_relative_path()) {
		_set_message(TTRC("The path specified is invalid."), MESSAGE_ERROR, p_type);
		return false;
	}

	if (p_path.get_file() != OS::get_singleton()->get_safe_dir_name(p_path.get_file())) {
		_set_message(TTRC("The directory name specified contains invalid characters or trailing whitespace."), MESSAGE_ERROR, p_type);
		return false;
	}

	return true;
}

void ProjectDialog::_reset_auto_dir() {
	auto_dir = "";
	last_custom_destination_dir = "";
	_update_destination_auto_dir();
	if (create_dir->is_pressed()) {
		// Append `auto_dir` to destination path.
		_create_dir_toggled(true);
	}
}

void ProjectDialog::_update_destination_auto_dir() {
	String new_auto_dir;
	if (name_container->is_visible()) {
		new_auto_dir = project_name->get_text();
	} else {
		new_auto_dir = origin_path->get_text().get_file().get_basename();
	}
	int naming_convention = (int)EDITOR_GET("project_manager/directory_naming_convention");
	switch (naming_convention) {
		case 0: // No Convention
			break;
		case 1: // kebab-case
			new_auto_dir = new_auto_dir.to_kebab_case();
			break;
		case 2: // snake_case
			new_auto_dir = new_auto_dir.to_snake_case();
			break;
		case 3: // camelCase
			new_auto_dir = new_auto_dir.to_camel_case();
			break;
		case 4: // PascalCase
			new_auto_dir = new_auto_dir.to_pascal_case();
			break;
		case 5: // Title Case
			new_auto_dir = new_auto_dir.capitalize();
			break;
		default:
			ERR_FAIL_MSG("Invalid directory naming convention.");
			break;
	}
	new_auto_dir = OS::get_singleton()->get_safe_dir_name(new_auto_dir);

	if (create_dir->is_pressed()) {
		String current_destination_path = destination_path->get_text();

		if (current_destination_path.get_file() == auto_dir) {
			// Update target dir name to new project name / ZIP name.
			current_destination_path = current_destination_path.get_base_dir().path_join(new_auto_dir);
		}

		destination_path->set_text(current_destination_path);
	}

	auto_dir = new_auto_dir;
}

void ProjectDialog::_create_dir_toggled(bool p_pressed) {
	String current_destination_path = destination_path->get_text();

	if (create_dir->is_pressed()) {
		// (Re-)append destination dir name.
		if (last_custom_destination_dir.is_empty()) {
			current_destination_path = current_destination_path.path_join(auto_dir);
		} else {
			current_destination_path = current_destination_path.path_join(last_custom_destination_dir);
		}
	} else {
		// Strip any trailing slash.
		current_destination_path = current_destination_path.rstrip("/\\");
		// Save and remove destination dir name.
		if (current_destination_path.get_file() == auto_dir) {
			last_custom_destination_dir = "";
		} else {
			last_custom_destination_dir = current_destination_path.get_file();
		}
		current_destination_path = current_destination_path.get_base_dir();
	}

	destination_path->set_text(current_destination_path);
	_validate_path();
}

void ProjectDialog::_project_name_changed() {
	if (destination_path_container->is_visible() && destination_path->is_editable()) {
		_update_destination_auto_dir();
	}

	_validate_path();
}

void ProjectDialog::_origin_path_changed() {
	bool was_origin_zip = is_origin_zip;
	const String path = origin_path->get_text();
	String new_path;
	if (path.get_file().strip_edges() == "project.godot") {
		new_path = path.get_base_dir();
	} else {
		new_path = ProjectOperations::resolve_origin_path(path, is_origin_zip);
	}
	if (new_path != path) {
		origin_path->set_text(new_path);
	}

	if (mode == MODE_IMPORT && was_origin_zip != is_origin_zip) {
		if (is_origin_zip) {
			destination_path_container->show();
			destination_path->set_text(default_destination_dir);
			_reset_auto_dir();
			_set_dialog_focus(FOCUS_TARGET_DESTINATION);

		} else {
			destination_path_container->hide();
			destination_path->set_text(origin_path->get_text());
			get_ok_button()->grab_focus();
		}
	}

	_validate_path();
}

void ProjectDialog::_destination_path_changed() {
	_validate_path();
}

void ProjectDialog::_browse_origin_path() {
	String path = origin_path->get_text();
	if (path.is_relative_path()) {
		path = EDITOR_GET("filesystem/directories/default_project_path");
	}

	fdialog_origin->set_current_dir(path);

	fdialog_origin->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_ANY);

	fdialog_origin->clear_filters();
	fdialog_origin->add_filter("project.godot", vformat("%s %s", GODOT_VERSION_NAME, TTR("Project")));

	if (mode == MODE_IMPORT) {
		fdialog_origin->add_filter("*.zip", TTR("ZIP File"));
	}

	hide();

	fdialog_origin->popup_file_dialog();
}

void ProjectDialog::_browse_destination_path() {
	String path = destination_path->get_text();
	if (path.is_relative_path() || !DirAccess::dir_exists_absolute(path)) {
		path = EDITOR_GET("filesystem/directories/default_project_path");
	}
	if (create_dir->is_pressed()) {
		// Select parent directory of destination path.
		fdialog_destination->set_current_dir(path.get_base_dir());
	} else {
		// Select destination path.
		fdialog_destination->set_current_dir(path);
	}

	fdialog_destination->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	fdialog_destination->popup_file_dialog();
}

void ProjectDialog::_origin_path_selected(const String &p_path) {
	show_dialog(false);

	origin_path->set_text(p_path);
	_origin_path_changed();
}

void ProjectDialog::_destination_path_selected(const String &p_path) {
	if (create_dir->is_pressed()) {
		// Replace parent directory, but keep target dir name.
		destination_path->set_text(p_path.path_join(destination_path->get_text().get_file()));
	} else {
		destination_path->set_text(p_path);
	}

	_destination_path_changed();
}

void ProjectDialog::_reset_name() {
	project_name->set_text(TTRC("New Game Project"));
}

void ProjectDialog::_renderer_selected() {
	if (!renderer_button_group->get_pressed_button()) {
		return;
	}

	String renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));

	bool rd_error = false;

	if (renderer_type == "forward_plus") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop platforms only.") +
				String::utf8("\n•  ") + TTR("Advanced 3D graphics available.") +
				String::utf8("\n•  ") + TTR("Can scale to large complex scenes.") +
				String::utf8("\n•  ") + TTR("Uses RenderingDevice backend.") +
				String::utf8("\n•  ") + TTR("Slower rendering of simple scenes."));
		rd_error = !rendering_device_supported;
	} else if (renderer_type == "mobile") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop + mobile platforms.") +
				String::utf8("\n•  ") + TTR("Less advanced 3D graphics.") +
				String::utf8("\n•  ") + TTR("Less scalable for complex scenes.") +
				String::utf8("\n•  ") + TTR("Uses RenderingDevice backend.") +
				String::utf8("\n•  ") + TTR("Fast rendering of simple scenes."));
		rd_error = !rendering_device_supported;
	} else if (renderer_type == "gl_compatibility") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop, mobile + web platforms.") +
				String::utf8("\n•  ") + TTR("Least advanced 3D graphics.") +
				String::utf8("\n•  ") + TTR("Intended for low-end/older devices.") +
				String::utf8("\n•  ") + TTR("Uses OpenGL 3 backend (OpenGL 3.3/ES 3.0/WebGL2).") +
				String::utf8("\n•  ") + TTR("Fastest rendering of simple scenes."));
	} else {
		WARN_PRINT("Unknown renderer type. Please report this as a bug on GitHub.");
	}

	rd_not_supported->set_visible(rd_error);
	if (rd_error) {
		// Needs to be set here since theme colors aren't available at startup.
		rd_not_supported->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		invalid_state_flags.set_flag(InvalidStateFlag::INVALID_STATE_FLAG_RENDERER_SELECT);
	} else {
		invalid_state_flags.clear_flag(InvalidStateFlag::INVALID_STATE_FLAG_RENDERER_SELECT);
	}

	_update_ok_button();
}

void ProjectDialog::_nonempty_confirmation_ok_pressed() {
	is_folder_empty = true;
	ok_pressed();
}

void ProjectDialog::ok_pressed() {
	// Before we create a project, check that the target folder is empty.
	// If not, we need to ask the user if they're sure they want to do this.
	if (!is_folder_empty) {
		if (!nonempty_confirmation) {
			nonempty_confirmation = memnew(ConfirmationDialog);
			nonempty_confirmation->set_flag(Window::FLAG_RESIZE_DISABLED, true);
			nonempty_confirmation->set_title(TTRC("Warning: This folder is not empty"));
			nonempty_confirmation->set_text(TTRC("You are about to create a Godot project in a non-empty folder.\nThe entire contents of this folder will be imported as project resources!\n\nAre you sure you wish to continue?"));
			nonempty_confirmation->get_ok_button()->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_nonempty_confirmation_ok_pressed));
			add_child(nonempty_confirmation);
		}
		nonempty_confirmation->popup_centered();
		return;
	}

	const String project_origin_path = origin_path->get_text();
	const String project_destination_path = destination_path->get_text();

	Error err;

	if (create_dir->is_pressed()) {
		err = ProjectOperations::create_project_dir(project_destination_path);
		if (err != OK) {
			_set_message(TTRC("Couldn't create project directory, check permissions."), MESSAGE_ERROR, DESTINATION_PATH);
			return;
		}
	}

	if (project_origin_path.is_empty()) {
		HashMap<String, Variant> chosen_settings;

		String renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));
		chosen_settings["rendering/renderer/rendering_method"] = renderer_type;

		EditorSettings::get_singleton()->set("project_manager/default_renderer", renderer_type);
		EditorSettings::get_singleton()->save();

		chosen_settings["application/config/name"] = project_name->get_text().strip_edges();

		ProjectOperations::create_project(project_destination_path, chosen_settings, (EditorVCSInterface::VCSMetadata)vcs_metadata_selection->get_selected());
	} else {
		HashMap<String, Variant> changed_settings;

		if (is_origin_zip) {
			Vector<String> failed_files;
			ProjectOperations::install_zip_project(project_origin_path, project_destination_path, project_root, failed_files);

			if (failed_files.size()) {
				String err_msg = TTR("The following files failed extraction from package:") + "\n\n";
				for (int i = 0; i < failed_files.size(); i++) {
					if (i > 15) {
						err_msg += "\nAnd " + itos(failed_files.size() - i) + " more files.";
						break;
					}
					err_msg += failed_files[i] + "\n";
				}

				dialog_error->set_text(err_msg);
				dialog_error->popup_centered();
				return;
			}
		} else if (project_destination_path != project_origin_path) {
			err = ProjectOperations::duplicate_project(project_origin_path, project_destination_path);
			if (err != OK) {
				dialog_error->set_text(vformat(TTR("Couldn't duplicate project (error %d)."), err));
				dialog_error->popup_centered();
				return;
			}
		}

		if (name_container->is_visible()) {
			changed_settings["application/config/name"] = project_name->get_text().strip_edges();
		}

		if (!changed_settings.is_empty()) {
			// Load project.godot as ConfigFile to set the new name.
			Ref<ConfigFile> cfg;
			cfg.instantiate();
			String project_godot = project_destination_path.path_join("project.godot");

			err = cfg->load(project_godot);
			if (err != OK) {
				dialog_error->set_text(vformat(TTR("Couldn't load project at '%s' (error %d). It may be missing or corrupted."), project_godot, err));
				dialog_error->popup_centered();
				return;
			}

			err = ProjectOperations::edit_existing_project_settings(changed_settings, project_godot, cfg);
			if (err != OK) {
				dialog_error->set_text(vformat(TTR("Couldn't save project at '%s' (error %d)."), project_godot, err));
				dialog_error->popup_centered();
				return;
			}
		}
	}

	hide();
	if (mode == MODE_NEW || mode == MODE_IMPORT || mode == MODE_INSTALL) {
#ifdef ANDROID_ENABLED
		ProjectOperations::create_nomedia_file(project_destination_path);
#endif
		emit_signal(SNAME("project_created"), project_destination_path, mode == MODE_NEW || edit_check_box->is_pressed());
	} else if (mode == MODE_DUPLICATE) {
		emit_signal(SNAME("project_duplicated"), project_origin_path, project_destination_path, edit_check_box->is_visible() && edit_check_box->is_pressed());
	} else if (mode == MODE_RENAME) {
		emit_signal(SNAME("projects_updated"));
	}
}

void ProjectDialog::set_duplicate_can_edit(bool p_duplicate_can_edit) {
	duplicate_can_edit = p_duplicate_can_edit;
}

void ProjectDialog::set_mode(Mode p_mode) {
	mode = p_mode;
}

void ProjectDialog::set_zip_title(const String &p_title) {
	zip_title = p_title;
}

void ProjectDialog::set_zip_project_root(const String &p_path) {
	project_root = p_path;
	is_origin_zip = true;
}

void ProjectDialog::set_project_name(const String &p_name) {
	project_name->set_text(p_name);
}

void ProjectDialog::set_origin_path(const String &p_path, const bool &p_working_on_origin) {
	origin_path->set_text(p_path);
	if (p_working_on_origin) {
		destination_path->set_text(p_path);
	}
}

void ProjectDialog::ask_for_path_and_show() {
	_reset_name();
	_browse_origin_path();
}

void ProjectDialog::show_dialog(bool p_reset_name, bool p_is_confirmed) {
	_update_ok_button();

	if (mode == MODE_IMPORT && !p_is_confirmed) {
		return;
	}

	String fav_dir = EDITOR_GET("filesystem/directories/default_project_path");
	fav_dir = fav_dir.simplify_path();
	if (!fav_dir.is_empty()) {
		default_destination_dir = fav_dir;
	} else {
		Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		default_destination_dir = d->get_current_dir();
	}

	name_container->hide();
	origin_path_container->hide();
	destination_path_container->hide();
	renderer_container->hide();
	default_files_container->hide();
	edit_check_box->hide();

	bool reset_auto_dir = true;

	switch (mode) {
		case MODE_NEW: {
			set_title(TTRC("Create New Project"));
			set_ok_button_text(TTRC("Create"));

			if (!rendering_device_checked) {
				rendering_device_supported = DisplayServer::is_rendering_device_supported();

				if (!rendering_device_supported) {
					List<BaseButton *> buttons;
					renderer_button_group->get_buttons(&buttons);
					for (BaseButton *button : buttons) {
						if (button->get_meta(SNAME("rendering_method")) == "gl_compatibility") {
							button->set_pressed(true);
							break;
						}
					}
				}
				_renderer_selected();
				rendering_device_checked = true;
			}

			_config_name_container(p_reset_name);
			origin_path->set_text("");
			_config_destination_container(TTRC("Project Path:"), default_destination_dir);
			renderer_container->show();
			default_files_container->show();

			_set_dialog_focus(FOCUS_TARGET_NAME);

		} break;
		case MODE_IMPORT: {
			set_title(TTRC("Import Existing Project"));
			set_ok_button_text(TTRC("Import"));

			_config_origin_container(TTRC("Project Path:"));
			//Destination visibility for import determined in _origin_path_changed
			_config_destination_container(TTRC("Project Installation Path:"));
			destination_path_container->hide();
			edit_check_box->show();

		} break;
		case MODE_INSTALL: {
			set_title(TTR("Install Project:") + " " + zip_title);
			set_ok_button_text(TTRC("Install"));

			_config_name_container(p_reset_name);
			_config_destination_container(TTRC("Project Path:"), default_destination_dir);
			edit_check_box->show();

		} break;
		case MODE_RENAME: {
			// Name and path are set in `ProjectManager::_rename_project`.
			set_title(TTRC("Rename Project"));
			set_ok_button_text(TTRC("Rename"));

			_config_name_container(false);
			_config_origin_container(TTRC("Project Path:"), origin_path->get_text(), false);

			reset_auto_dir = false;

			_set_dialog_focus(FOCUS_TARGET_NAME);
		} break;
		case MODE_DUPLICATE: {
			set_title(TTRC("Duplicate Project"));
			set_ok_button_text(TTRC("Duplicate"));

			fdialog_origin->set_current_dir(origin_path->get_text().get_base_dir());

			_config_name_container(p_reset_name);
			_config_destination_container(TTRC("Project Path:"), default_destination_dir);
			edit_check_box->set_visible(duplicate_can_edit);

			_set_dialog_focus(FOCUS_TARGET_NAME);
		} break;
	}

	if (reset_auto_dir) {
		_reset_auto_dir();
	}

	_validate_path();

	popup_centered(Size2(500, 0) * EDSCALE);
}

void ProjectDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (rendering_device_checked) {
				_renderer_selected();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			create_dir->set_button_icon(get_editor_theme_icon(SNAME("FolderCreate")));
			origin_browse->set_button_icon(get_editor_theme_icon(SNAME("FolderBrowse")));
			destination_browse->set_button_icon(get_editor_theme_icon(SNAME("FolderBrowse")));
		} break;
		case NOTIFICATION_READY: {
			fdialog_origin = memnew(EditorFileDialog);
			fdialog_origin->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			fdialog_origin->connect("dir_selected", callable_mp(this, &ProjectDialog::_origin_path_selected));
			fdialog_origin->connect("file_selected", callable_mp(this, &ProjectDialog::_origin_path_selected));
			fdialog_origin->connect("canceled", callable_mp(this, &ProjectDialog::show_dialog).bind(false, false), CONNECT_DEFERRED);
			callable_mp((Node *)this, &Node::add_sibling).call_deferred(fdialog_origin, false);
		} break;
	}
}

void ProjectDialog::_config_name_container(const bool p_reset_name) {
	if (p_reset_name) {
		_reset_name();
	}
	name_container->show();
}

void ProjectDialog::_config_origin_container(const String &p_path_title, const String &p_default_path, const bool p_editable) {
	origin_path_label->set_text(p_path_title);
	origin_path->set_accessibility_name(p_path_title);

	origin_path->set_text(p_default_path);

	origin_path->set_editable(p_editable);
	origin_status_rect->set_visible(p_editable);
	origin_browse->set_visible(p_editable);

	origin_path_container->show();
}

void ProjectDialog::_config_destination_container(const String &path_title, const String &p_default_path, const bool p_editable) {
	destination_path_label->set_text(path_title);
	destination_path->set_accessibility_name(path_title);

	destination_path->set_text(p_default_path);

	destination_path->set_editable(p_editable);
	create_dir->set_visible(p_editable);
	destination_status_rect->set_visible(p_editable);
	destination_browse->set_visible(p_editable);

	destination_path_container->show();
}

void ProjectDialog::_set_dialog_focus(FocusTarget p_focus_target) {
	switch (p_focus_target) {
		case FOCUS_TARGET_NAME: {
			callable_mp((Control *)project_name, &Control::grab_focus).call_deferred(false);
			callable_mp(project_name, &LineEdit::select_all).call_deferred();
		} break;
		case FOCUS_TARGET_ORIGIN: {
			callable_mp((Control *)origin_path, &Control::grab_focus).call_deferred(false);
		} break;
		case FOCUS_TARGET_DESTINATION: {
			callable_mp((Control *)destination_path, &Control::grab_focus).call_deferred(false);
		} break;
	}
}

void ProjectDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("project_created"));
	ADD_SIGNAL(MethodInfo("project_duplicated"));
	ADD_SIGNAL(MethodInfo("projects_updated"));
}

ProjectDialog::ProjectDialog() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	name_container = memnew(VBoxContainer);
	vb->add_child(name_container);

	Label *l = memnew(Label);
	l->set_text(TTRC("Project Name:"));
	name_container->add_child(l);

	project_name = memnew(LineEdit);
	project_name->set_virtual_keyboard_show_on_focus(false);
	project_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	project_name->set_accessibility_name(TTRC("Project Name:"));
	name_container->add_child(project_name);

	origin_path_container = memnew(VBoxContainer);
	vb->add_child(origin_path_container);

	origin_path_label = memnew(Label);
	origin_path_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	origin_path_container->add_child(origin_path_label);

	HBoxContainer *ophb = memnew(HBoxContainer);
	origin_path_container->add_child(ophb);

	origin_path = memnew(LineEdit);
	origin_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	origin_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	ophb->add_child(origin_path);

	destination_path_container = memnew(VBoxContainer);
	vb->add_child(destination_path_container);

	HBoxContainer *dphb_label = memnew(HBoxContainer);
	destination_path_container->add_child(dphb_label);

	destination_path_label = memnew(Label);
	destination_path_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dphb_label->add_child(destination_path_label);

	create_dir = memnew(CheckButton);
	create_dir->set_text(TTRC("Create Folder"));
	create_dir->set_pressed(true);
	dphb_label->add_child(create_dir);
	create_dir->connect(SceneStringName(toggled), callable_mp(this, &ProjectDialog::_create_dir_toggled));

	HBoxContainer *dphb = memnew(HBoxContainer);
	destination_path_container->add_child(dphb);

	destination_path = memnew(LineEdit);
	destination_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	destination_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	dphb->add_child(destination_path);

	//origin status icon
	origin_status_rect = memnew(TextureRect);
	origin_status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	ophb->add_child(origin_status_rect);

	origin_browse = memnew(Button);
	origin_browse->set_text(TTRC("Browse"));
	origin_browse->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_browse_origin_path));
	ophb->add_child(origin_browse);

	// destination status icon
	destination_status_rect = memnew(TextureRect);
	destination_status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	dphb->add_child(destination_status_rect);

	destination_browse = memnew(Button);
	destination_browse->set_text(TTRC("Browse"));
	destination_browse->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_browse_destination_path));
	dphb->add_child(destination_browse);

	msg = memnew(Label);
	msg->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	msg->set_accessibility_live(AccessibilityServerEnums::LIVE_POLITE);
	msg->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	msg->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	msg->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	vb->add_child(msg);

	// Renderer selection.
	renderer_container = memnew(VBoxContainer);
	vb->add_child(renderer_container);
	l = memnew(Label);
	l->set_text(TTRC("Renderer:"));
	renderer_container->add_child(l);
	HBoxContainer *rshc = memnew(HBoxContainer);
	renderer_container->add_child(rshc);
	renderer_button_group.instantiate();

	// Left hand side, used for checkboxes to select renderer.
	Container *rvb = memnew(VBoxContainer);
	rshc->add_child(rvb);

	String default_renderer_type = "forward_plus";
	if (EditorSettings::get_singleton()->has_setting("project_manager/default_renderer")) {
		default_renderer_type = EditorSettings::get_singleton()->get_setting("project_manager/default_renderer");
	}

	Button *rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTRC("Forward+"));
	rs_button->set_accessibility_name(TTRC("Renderer:"));
#if !defined(RD_ENABLED) || !defined(FORWARD_RD_ENABLED)
	rs_button->set_disabled(true);
	rs_button->set_tooltip_text(TTRC("Either RenderingDevice or the Forward+ rendering method was disabled at compile time."));
#endif
	rs_button->set_meta(SNAME("rendering_method"), "forward_plus");
	rs_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	if (default_renderer_type == "forward_plus") {
		rs_button->set_pressed(true);
	}
	rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTRC("Mobile"));
	rs_button->set_accessibility_name(TTRC("Renderer:"));
#if !defined(RD_ENABLED) || !defined(MOBILE_RD_ENABLED)
	rs_button->set_disabled(true);
	rs_button->set_tooltip_text(TTRC("Either RenderingDevice or the Mobile rendering method was disabled at compile time."));
#endif
	rs_button->set_meta(SNAME("rendering_method"), "mobile");
	rs_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	if (default_renderer_type == "mobile") {
		rs_button->set_pressed(true);
	}
	rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTRC("Compatibility"));
	rs_button->set_accessibility_name(TTRC("Renderer:"));
#if !defined(GLES3_ENABLED)
	rs_button->set_disabled(true);
#endif
	rs_button->set_meta(SNAME("rendering_method"), "gl_compatibility");
	rs_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	LinkButton *ri_link = memnew(LinkButton);
	ri_link->set_text(TTRC("More information"));
	ri_link->set_uri(GODOT_VERSION_DOCS_URL "/tutorials/rendering/renderers.html");
	ri_link->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	rvb->add_child(ri_link);
#if defined(GLES3_ENABLED)
	if (default_renderer_type == "gl_compatibility") {
		rs_button->set_pressed(true);
	}
#endif
	rshc->add_child(memnew(VSeparator));

	// Right hand side, used for text explaining each choice.
	rvb = memnew(VBoxContainer);
	rvb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	rshc->add_child(rvb);
	renderer_info = memnew(Label);
	renderer_info->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	renderer_info->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	renderer_info->set_modulate(Color(1, 1, 1, 0.7));
	rvb->add_child(renderer_info);

	rd_not_supported = memnew(Label);
	rd_not_supported->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	rd_not_supported->set_text(vformat(TTR("RenderingDevice-based methods not available on this GPU:\n%s\nPlease use the Compatibility renderer."), RenderingServer::get_singleton()->get_video_adapter_name()));
	rd_not_supported->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	rd_not_supported->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	rd_not_supported->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	rd_not_supported->set_visible(false);
	renderer_container->add_child(rd_not_supported);

	l = memnew(Label);
	l->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	l->set_text(TTRC("The renderer can be changed later, but scenes may need to be adjusted."));
	// Add some extra spacing to separate it from the list above and the buttons below.
	l->set_custom_minimum_size(Size2(0, 40) * EDSCALE);
	l->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	l->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	l->set_modulate(Color(1, 1, 1, 0.7));
	renderer_container->add_child(l);

	default_files_container = memnew(HBoxContainer);
	vb->add_child(default_files_container);
	l = memnew(Label);
	l->set_text(TTRC("Version Control Metadata:"));
	default_files_container->add_child(l);
	vcs_metadata_selection = memnew(OptionButton);
	vcs_metadata_selection->set_custom_minimum_size(Size2(100, 20));
	vcs_metadata_selection->add_item(TTRC("None"), (int)EditorVCSInterface::VCSMetadata::NONE);
	vcs_metadata_selection->add_item(TTRC("Git"), (int)EditorVCSInterface::VCSMetadata::GIT);
	vcs_metadata_selection->select((int)EditorVCSInterface::VCSMetadata::GIT);
	vcs_metadata_selection->set_accessibility_name(TTRC("Version Control Metadata:"));
	default_files_container->add_child(vcs_metadata_selection);

	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	default_files_container->add_child(spacer);

	fdialog_destination = memnew(EditorFileDialog);
	fdialog_destination->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	add_child(fdialog_destination);

	Control *spacer2 = memnew(Control);
	spacer2->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(spacer2);

	edit_check_box = memnew(CheckBox);
	edit_check_box->set_text(TTRC("Edit Now"));
	edit_check_box->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	edit_check_box->set_pressed(true);
	vb->add_child(edit_check_box);

	project_name->connect(SceneStringName(text_changed), callable_mp(this, &ProjectDialog::_project_name_changed).unbind(1));
	project_name->connect(SceneStringName(text_submitted), callable_mp(this, &ProjectDialog::ok_pressed).unbind(1));

	origin_path->connect(SceneStringName(text_changed), callable_mp(this, &ProjectDialog::_origin_path_changed).unbind(1));
	origin_path->connect(SceneStringName(text_submitted), callable_mp(this, &ProjectDialog::ok_pressed).unbind(1));

	destination_path->connect(SceneStringName(text_changed), callable_mp(this, &ProjectDialog::_destination_path_changed).unbind(1));
	destination_path->connect(SceneStringName(text_submitted), callable_mp(this, &ProjectDialog::ok_pressed).unbind(1));

	fdialog_destination->connect("dir_selected", callable_mp(this, &ProjectDialog::_destination_path_selected));
	fdialog_destination->connect("file_selected", callable_mp(this, &ProjectDialog::_destination_path_selected));

	set_hide_on_ok(false);

	dialog_error = memnew(AcceptDialog);
	add_child(dialog_error);
}
