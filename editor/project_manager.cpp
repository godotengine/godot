/**************************************************************************/
/*  project_manager.cpp                                                   */
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

#include "project_manager.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/io/stream_peer_tls.h"
#include "core/io/zip_io.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/translation.h"
#include "core/version.h"
#include "editor/editor_paths.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_themes.h"
#include "editor/editor_vcs_interface.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "main/main.h"
#include "scene/gui/center_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/window.h"
#include "scene/resources/image_texture.h"
#include "servers/display_server.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"

constexpr int GODOT4_CONFIG_VERSION = 5;

/// Project Dialog.

void ProjectDialog::_set_message(const String &p_msg, MessageType p_type, InputType input_type) {
	msg->set_text(p_msg);
	Ref<Texture2D> current_path_icon = status_rect->get_texture();
	Ref<Texture2D> current_install_icon = install_status_rect->get_texture();
	Ref<Texture2D> new_icon;

	switch (p_type) {
		case MESSAGE_ERROR: {
			msg->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
			msg->set_modulate(Color(1, 1, 1, 1));
			new_icon = get_editor_theme_icon(SNAME("StatusError"));

		} break;
		case MESSAGE_WARNING: {
			msg->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
			msg->set_modulate(Color(1, 1, 1, 1));
			new_icon = get_editor_theme_icon(SNAME("StatusWarning"));

		} break;
		case MESSAGE_SUCCESS: {
			msg->remove_theme_color_override("font_color");
			msg->set_modulate(Color(1, 1, 1, 0));
			new_icon = get_editor_theme_icon(SNAME("StatusSuccess"));

		} break;
	}

	if (current_path_icon != new_icon && input_type == PROJECT_PATH) {
		status_rect->set_texture(new_icon);
	} else if (current_install_icon != new_icon && input_type == INSTALL_PATH) {
		install_status_rect->set_texture(new_icon);
	}
}

String ProjectDialog::_test_path() {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String valid_path, valid_install_path;
	if (d->change_dir(project_path->get_text()) == OK) {
		valid_path = project_path->get_text();
	} else if (d->change_dir(project_path->get_text().strip_edges()) == OK) {
		valid_path = project_path->get_text().strip_edges();
	} else if (project_path->get_text().ends_with(".zip")) {
		if (d->file_exists(project_path->get_text())) {
			valid_path = project_path->get_text();
		}
	} else if (project_path->get_text().strip_edges().ends_with(".zip")) {
		if (d->file_exists(project_path->get_text().strip_edges())) {
			valid_path = project_path->get_text().strip_edges();
		}
	}

	if (valid_path.is_empty()) {
		_set_message(TTR("The path specified doesn't exist."), MESSAGE_ERROR);
		get_ok_button()->set_disabled(true);
		return "";
	}

	if (mode == MODE_IMPORT && valid_path.ends_with(".zip")) {
		if (d->change_dir(install_path->get_text()) == OK) {
			valid_install_path = install_path->get_text();
		} else if (d->change_dir(install_path->get_text().strip_edges()) == OK) {
			valid_install_path = install_path->get_text().strip_edges();
		}

		if (valid_install_path.is_empty()) {
			_set_message(TTR("The path specified doesn't exist."), MESSAGE_ERROR, INSTALL_PATH);
			get_ok_button()->set_disabled(true);
			return "";
		}
	}

	if (mode == MODE_IMPORT || mode == MODE_RENAME) {
		if (!valid_path.is_empty() && !d->file_exists("project.godot")) {
			if (valid_path.ends_with(".zip")) {
				Ref<FileAccess> io_fa;
				zlib_filefunc_def io = zipio_create_io(&io_fa);

				unzFile pkg = unzOpen2(valid_path.utf8().get_data(), &io);
				if (!pkg) {
					_set_message(TTR("Error opening package file (it's not in ZIP format)."), MESSAGE_ERROR);
					get_ok_button()->set_disabled(true);
					unzClose(pkg);
					return "";
				}

				int ret = unzGoToFirstFile(pkg);
				while (ret == UNZ_OK) {
					unz_file_info info;
					char fname[16384];
					ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
					if (ret != UNZ_OK) {
						break;
					}

					if (String::utf8(fname).ends_with("project.godot")) {
						break;
					}

					ret = unzGoToNextFile(pkg);
				}

				if (ret == UNZ_END_OF_LIST_OF_FILE) {
					_set_message(TTR("Invalid \".zip\" project file; it doesn't contain a \"project.godot\" file."), MESSAGE_ERROR);
					get_ok_button()->set_disabled(true);
					unzClose(pkg);
					return "";
				}

				unzClose(pkg);

				// check if the specified install folder is empty, even though this is not an error, it is good to check here
				d->list_dir_begin();
				is_folder_empty = true;
				String n = d->get_next();
				while (!n.is_empty()) {
					if (!n.begins_with(".")) {
						// Allow `.`, `..` (reserved current/parent folder names)
						// and hidden files/folders to be present.
						// For instance, this lets users initialize a Git repository
						// and still be able to create a project in the directory afterwards.
						is_folder_empty = false;
						break;
					}
					n = d->get_next();
				}
				d->list_dir_end();

				if (!is_folder_empty) {
					_set_message(TTR("Please choose an empty folder."), MESSAGE_WARNING, INSTALL_PATH);
					get_ok_button()->set_disabled(true);
					return "";
				}

			} else {
				_set_message(TTR("Please choose a \"project.godot\", a directory with it, or a \".zip\" file."), MESSAGE_ERROR);
				install_path_container->hide();
				get_ok_button()->set_disabled(true);
				return "";
			}

		} else if (valid_path.ends_with("zip")) {
			_set_message(TTR("This directory already contains a Godot project."), MESSAGE_ERROR, INSTALL_PATH);
			get_ok_button()->set_disabled(true);
			return "";
		}

	} else {
		// Check if the specified folder is empty, even though this is not an error, it is good to check here.
		d->list_dir_begin();
		is_folder_empty = true;
		String n = d->get_next();
		while (!n.is_empty()) {
			if (!n.begins_with(".")) {
				// Allow `.`, `..` (reserved current/parent folder names)
				// and hidden files/folders to be present.
				// For instance, this lets users initialize a Git repository
				// and still be able to create a project in the directory afterwards.
				is_folder_empty = false;
				break;
			}
			n = d->get_next();
		}
		d->list_dir_end();

		if (!is_folder_empty) {
			if (valid_path == OS::get_singleton()->get_environment("HOME") || valid_path == OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DOCUMENTS) || valid_path == OS::get_singleton()->get_executable_path().get_base_dir()) {
				_set_message(TTR("You cannot save a project in the selected path. Please make a new folder or choose a new path."), MESSAGE_ERROR);
				get_ok_button()->set_disabled(true);
				return "";
			}

			_set_message(TTR("The selected path is not empty. Choosing an empty folder is highly recommended."), MESSAGE_WARNING);
			get_ok_button()->set_disabled(false);
			return valid_path;
		}
	}

	_set_message("");
	_set_message("", MESSAGE_SUCCESS, INSTALL_PATH);
	get_ok_button()->set_disabled(false);
	return valid_path;
}

void ProjectDialog::_path_text_changed(const String &p_path) {
	String sp = _test_path();
	if (!sp.is_empty()) {
		// If the project name is empty or default, infer the project name from the selected folder name
		if (project_name->get_text().strip_edges().is_empty() || project_name->get_text().strip_edges() == TTR("New Game Project")) {
			sp = sp.replace("\\", "/");
			int lidx = sp.rfind("/");

			if (lidx != -1) {
				sp = sp.substr(lidx + 1, sp.length()).capitalize();
			}
			if (sp.is_empty() && mode == MODE_IMPORT) {
				sp = TTR("Imported Project");
			}

			project_name->set_text(sp);
			_text_changed(sp);
		}
	}

	if (!created_folder_path.is_empty() && created_folder_path != p_path) {
		_remove_created_folder();
	}
}

void ProjectDialog::_file_selected(const String &p_path) {
	// If not already shown.
	show_dialog();

	String p = p_path;
	if (mode == MODE_IMPORT) {
		if (p.ends_with("project.godot")) {
			p = p.get_base_dir();
			install_path_container->hide();
			get_ok_button()->set_disabled(false);
		} else if (p.ends_with(".zip")) {
			install_path->set_text(p.get_base_dir());
			install_path_container->show();
			get_ok_button()->set_disabled(false);
		} else {
			_set_message(TTR("Please choose a \"project.godot\" or \".zip\" file."), MESSAGE_ERROR);
			get_ok_button()->set_disabled(true);
			return;
		}
	}

	String sp = p.simplify_path();
	project_path->set_text(sp);
	_path_text_changed(sp);
	if (p.ends_with(".zip")) {
		install_path->call_deferred(SNAME("grab_focus"));
	} else {
		get_ok_button()->call_deferred(SNAME("grab_focus"));
	}
}

void ProjectDialog::_path_selected(const String &p_path) {
	// If not already shown.
	show_dialog();

	String sp = p_path.simplify_path();
	project_path->set_text(sp);
	_path_text_changed(sp);
	get_ok_button()->call_deferred(SNAME("grab_focus"));
}

void ProjectDialog::_install_path_selected(const String &p_path) {
	String sp = p_path.simplify_path();
	install_path->set_text(sp);
	_path_text_changed(sp);
	get_ok_button()->call_deferred(SNAME("grab_focus"));
}

void ProjectDialog::_browse_path() {
	fdialog->set_current_dir(project_path->get_text());

	if (mode == MODE_IMPORT) {
		fdialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_ANY);
		fdialog->clear_filters();
		fdialog->add_filter("project.godot", vformat("%s %s", VERSION_NAME, TTR("Project")));
		fdialog->add_filter("*.zip", TTR("ZIP File"));
	} else {
		fdialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	}
	fdialog->popup_file_dialog();
}

void ProjectDialog::_browse_install_path() {
	fdialog_install->set_current_dir(install_path->get_text());
	fdialog_install->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	fdialog_install->popup_file_dialog();
}

void ProjectDialog::_create_folder() {
	const String project_name_no_edges = project_name->get_text().strip_edges();
	if (project_name_no_edges.is_empty() || !created_folder_path.is_empty() || project_name_no_edges.ends_with(".")) {
		_set_message(TTR("Invalid project name."), MESSAGE_WARNING);
		return;
	}

	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (d->change_dir(project_path->get_text()) == OK) {
		if (!d->dir_exists(project_name_no_edges)) {
			if (d->make_dir(project_name_no_edges) == OK) {
				d->change_dir(project_name_no_edges);
				String dir_str = d->get_current_dir();
				project_path->set_text(dir_str);
				_path_text_changed(dir_str);
				created_folder_path = d->get_current_dir();
				create_dir->set_disabled(true);
			} else {
				dialog_error->set_text(TTR("Couldn't create folder."));
				dialog_error->popup_centered();
			}
		} else {
			dialog_error->set_text(TTR("There is already a folder in this path with the specified name."));
			dialog_error->popup_centered();
		}
	}
}

void ProjectDialog::_text_changed(const String &p_text) {
	if (mode != MODE_NEW) {
		return;
	}

	_test_path();

	if (p_text.strip_edges().is_empty()) {
		_set_message(TTR("It would be a good idea to name your project."), MESSAGE_ERROR);
	}
}

void ProjectDialog::_nonempty_confirmation_ok_pressed() {
	is_folder_empty = true;
	ok_pressed();
}

void ProjectDialog::_renderer_selected() {
	ERR_FAIL_NULL(renderer_button_group->get_pressed_button());

	String renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));

	if (renderer_type == "forward_plus") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop platforms only.") +
				String::utf8("\n•  ") + TTR("Advanced 3D graphics available.") +
				String::utf8("\n•  ") + TTR("Can scale to large complex scenes.") +
				String::utf8("\n•  ") + TTR("Uses RenderingDevice backend.") +
				String::utf8("\n•  ") + TTR("Slower rendering of simple scenes."));
	} else if (renderer_type == "mobile") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop + mobile platforms.") +
				String::utf8("\n•  ") + TTR("Less advanced 3D graphics.") +
				String::utf8("\n•  ") + TTR("Less scalable for complex scenes.") +
				String::utf8("\n•  ") + TTR("Uses RenderingDevice backend.") +
				String::utf8("\n•  ") + TTR("Fast rendering of simple scenes."));
	} else if (renderer_type == "gl_compatibility") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop, mobile + web platforms.") +
				String::utf8("\n•  ") + TTR("Least advanced 3D graphics (currently work-in-progress).") +
				String::utf8("\n•  ") + TTR("Intended for low-end/older devices.") +
				String::utf8("\n•  ") + TTR("Uses OpenGL 3 backend (OpenGL 3.3/ES 3.0/WebGL2).") +
				String::utf8("\n•  ") + TTR("Fastest rendering of simple scenes."));
	} else {
		WARN_PRINT("Unknown renderer type. Please report this as a bug on GitHub.");
	}
}

void ProjectDialog::_remove_created_folder() {
	if (!created_folder_path.is_empty()) {
		Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		d->remove(created_folder_path);

		create_dir->set_disabled(false);
		created_folder_path = "";
	}
}

void ProjectDialog::ok_pressed() {
	String dir = project_path->get_text();

	if (mode == MODE_RENAME) {
		String dir2 = _test_path();
		if (dir2.is_empty()) {
			_set_message(TTR("Invalid project path (changed anything?)."), MESSAGE_ERROR);
			return;
		}

		// Load project.godot as ConfigFile to set the new name.
		ConfigFile cfg;
		String project_godot = dir2.path_join("project.godot");
		Error err = cfg.load(project_godot);
		if (err != OK) {
			_set_message(vformat(TTR("Couldn't load project at '%s' (error %d). It may be missing or corrupted."), project_godot, err), MESSAGE_ERROR);
		} else {
			cfg.set_value("application", "config/name", project_name->get_text().strip_edges());
			err = cfg.save(project_godot);
			if (err != OK) {
				_set_message(vformat(TTR("Couldn't save project at '%s' (error %d)."), project_godot, err), MESSAGE_ERROR);
			}
		}

		hide();
		emit_signal(SNAME("projects_updated"));

	} else {
		if (mode == MODE_IMPORT) {
			if (project_path->get_text().ends_with(".zip")) {
				mode = MODE_INSTALL;
				ok_pressed();

				return;
			}

		} else {
			if (mode == MODE_NEW) {
				// Before we create a project, check that the target folder is empty.
				// If not, we need to ask the user if they're sure they want to do this.
				if (!is_folder_empty) {
					ConfirmationDialog *cd = memnew(ConfirmationDialog);
					cd->set_title(TTR("Warning: This folder is not empty"));
					cd->set_text(TTR("You are about to create a Godot project in a non-empty folder.\nThe entire contents of this folder will be imported as project resources!\n\nAre you sure you wish to continue?"));
					cd->get_ok_button()->connect("pressed", callable_mp(this, &ProjectDialog::_nonempty_confirmation_ok_pressed));
					get_parent()->add_child(cd);
					cd->popup_centered();
					cd->grab_focus();
					return;
				}
				PackedStringArray project_features = ProjectSettings::get_required_features();
				ProjectSettings::CustomMap initial_settings;

				// Be sure to change this code if/when renderers are changed.
				// Default values are "forward_plus" for the main setting, "mobile" for the mobile override,
				// and "gl_compatibility" for the web override.
				String renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));
				initial_settings["rendering/renderer/rendering_method"] = renderer_type;

				EditorSettings::get_singleton()->set("project_manager/default_renderer", renderer_type);
				EditorSettings::get_singleton()->save();

				if (renderer_type == "forward_plus") {
					project_features.push_back("Forward Plus");
				} else if (renderer_type == "mobile") {
					project_features.push_back("Mobile");
				} else if (renderer_type == "gl_compatibility") {
					project_features.push_back("GL Compatibility");
					// Also change the default rendering method for the mobile override.
					initial_settings["rendering/renderer/rendering_method.mobile"] = "gl_compatibility";
				} else {
					WARN_PRINT("Unknown renderer type. Please report this as a bug on GitHub.");
				}

				project_features.sort();
				initial_settings["application/config/features"] = project_features;
				initial_settings["application/config/name"] = project_name->get_text().strip_edges();
				initial_settings["application/config/icon"] = "res://icon.svg";

				if (ProjectSettings::get_singleton()->save_custom(dir.path_join("project.godot"), initial_settings, Vector<String>(), false) != OK) {
					_set_message(TTR("Couldn't create project.godot in project path."), MESSAGE_ERROR);
				} else {
					// Store default project icon in SVG format.
					Error err;
					Ref<FileAccess> fa_icon = FileAccess::open(dir.path_join("icon.svg"), FileAccess::WRITE, &err);
					fa_icon->store_string(get_default_project_icon());

					if (err != OK) {
						_set_message(TTR("Couldn't create icon.svg in project path."), MESSAGE_ERROR);
					}

					EditorVCSInterface::create_vcs_metadata_files(EditorVCSInterface::VCSMetadata(vcs_metadata_selection->get_selected()), dir);
				}
			} else if (mode == MODE_INSTALL) {
				if (project_path->get_text().ends_with(".zip")) {
					dir = install_path->get_text();
					zip_path = project_path->get_text();
				}

				Ref<FileAccess> io_fa;
				zlib_filefunc_def io = zipio_create_io(&io_fa);

				unzFile pkg = unzOpen2(zip_path.utf8().get_data(), &io);
				if (!pkg) {
					dialog_error->set_text(TTR("Error opening package file, not in ZIP format."));
					dialog_error->popup_centered();
					return;
				}

				// Find the zip_root
				String zip_root;
				int ret = unzGoToFirstFile(pkg);
				while (ret == UNZ_OK) {
					unz_file_info info;
					char fname[16384];
					unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

					String name = String::utf8(fname);
					if (name.ends_with("project.godot")) {
						zip_root = name.substr(0, name.rfind("project.godot"));
						break;
					}

					ret = unzGoToNextFile(pkg);
				}

				ret = unzGoToFirstFile(pkg);

				Vector<String> failed_files;

				while (ret == UNZ_OK) {
					//get filename
					unz_file_info info;
					char fname[16384];
					ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
					if (ret != UNZ_OK) {
						break;
					}

					String path = String::utf8(fname);

					if (path.is_empty() || path == zip_root || !zip_root.is_subsequence_of(path)) {
						//
					} else if (path.ends_with("/")) { // a dir
						path = path.substr(0, path.length() - 1);
						String rel_path = path.substr(zip_root.length());

						Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
						da->make_dir(dir.path_join(rel_path));
					} else {
						Vector<uint8_t> uncomp_data;
						uncomp_data.resize(info.uncompressed_size);
						String rel_path = path.substr(zip_root.length());

						//read
						unzOpenCurrentFile(pkg);
						ret = unzReadCurrentFile(pkg, uncomp_data.ptrw(), uncomp_data.size());
						ERR_BREAK_MSG(ret < 0, vformat("An error occurred while attempting to read from file: %s. This file will not be used.", rel_path));
						unzCloseCurrentFile(pkg);

						Ref<FileAccess> f = FileAccess::open(dir.path_join(rel_path), FileAccess::WRITE);
						if (f.is_valid()) {
							f->store_buffer(uncomp_data.ptr(), uncomp_data.size());
						} else {
							failed_files.push_back(rel_path);
						}
					}

					ret = unzGoToNextFile(pkg);
				}

				unzClose(pkg);

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

				} else if (!project_path->get_text().ends_with(".zip")) {
					dialog_error->set_text(TTR("Package installed successfully!"));
					dialog_error->popup_centered();
				}
			}
		}

		dir = dir.replace("\\", "/");
		if (dir.ends_with("/")) {
			dir = dir.substr(0, dir.length() - 1);
		}

		hide();
		emit_signal(SNAME("project_created"), dir);
	}
}

void ProjectDialog::cancel_pressed() {
	_remove_created_folder();

	project_path->clear();
	_path_text_changed("");
	project_name->clear();
	_text_changed("");

	if (status_rect->get_texture() == get_editor_theme_icon(SNAME("StatusError"))) {
		msg->show();
	}

	if (install_status_rect->get_texture() == get_editor_theme_icon(SNAME("StatusError"))) {
		msg->show();
	}
}

void ProjectDialog::set_zip_path(const String &p_path) {
	zip_path = p_path;
}

void ProjectDialog::set_zip_title(const String &p_title) {
	zip_title = p_title;
}

void ProjectDialog::set_mode(Mode p_mode) {
	mode = p_mode;
}

void ProjectDialog::set_project_path(const String &p_path) {
	project_path->set_text(p_path);
}

void ProjectDialog::ask_for_path_and_show() {
	// Workaround: for the file selection dialog content to be rendered we need to show its parent dialog.
	show_dialog();
	_set_message("");

	_browse_path();
}

void ProjectDialog::show_dialog() {
	if (mode == MODE_RENAME) {
		project_path->set_editable(false);
		browse->hide();
		install_browse->hide();

		set_title(TTR("Rename Project"));
		set_ok_button_text(TTR("Rename"));
		name_container->show();
		status_rect->hide();
		msg->hide();
		install_path_container->hide();
		install_status_rect->hide();
		renderer_container->hide();
		default_files_container->hide();
		get_ok_button()->set_disabled(false);

		// Fetch current name from project.godot to prefill the text input.
		ConfigFile cfg;
		String project_godot = project_path->get_text().path_join("project.godot");
		Error err = cfg.load(project_godot);
		if (err != OK) {
			_set_message(vformat(TTR("Couldn't load project at '%s' (error %d). It may be missing or corrupted."), project_godot, err), MESSAGE_ERROR);
			status_rect->show();
			msg->show();
			get_ok_button()->set_disabled(true);
		} else {
			String cur_name = cfg.get_value("application", "config/name", "");
			project_name->set_text(cur_name);
			_text_changed(cur_name);
		}

		project_name->call_deferred(SNAME("grab_focus"));

		create_dir->hide();

	} else {
		fav_dir = EDITOR_GET("filesystem/directories/default_project_path");
		if (!fav_dir.is_empty()) {
			project_path->set_text(fav_dir);
			fdialog->set_current_dir(fav_dir);
		} else {
			Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			project_path->set_text(d->get_current_dir());
			fdialog->set_current_dir(d->get_current_dir());
		}

		if (project_name->get_text().is_empty()) {
			String proj = TTR("New Game Project");
			project_name->set_text(proj);
			_text_changed(proj);
		}

		project_path->set_editable(true);
		browse->set_disabled(false);
		browse->show();
		install_browse->set_disabled(false);
		install_browse->show();
		create_dir->show();
		status_rect->show();
		install_status_rect->show();
		msg->show();

		if (mode == MODE_IMPORT) {
			set_title(TTR("Import Existing Project"));
			set_ok_button_text(TTR("Import & Edit"));
			name_container->hide();
			install_path_container->hide();
			renderer_container->hide();
			default_files_container->hide();
			project_path->grab_focus();

		} else if (mode == MODE_NEW) {
			set_title(TTR("Create New Project"));
			set_ok_button_text(TTR("Create & Edit"));
			name_container->show();
			install_path_container->hide();
			renderer_container->show();
			default_files_container->show();
			project_name->call_deferred(SNAME("grab_focus"));
			project_name->call_deferred(SNAME("select_all"));

		} else if (mode == MODE_INSTALL) {
			set_title(TTR("Install Project:") + " " + zip_title);
			set_ok_button_text(TTR("Install & Edit"));
			project_name->set_text(zip_title);
			name_container->show();
			install_path_container->hide();
			renderer_container->hide();
			default_files_container->hide();
			project_path->grab_focus();
		}

		_test_path();
	}

	popup_centered(Size2(500, 0) * EDSCALE);
}

void ProjectDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_remove_created_folder();
		} break;
	}
}

void ProjectDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("project_created"));
	ADD_SIGNAL(MethodInfo("projects_updated"));
}

ProjectDialog::ProjectDialog() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	name_container = memnew(VBoxContainer);
	vb->add_child(name_container);

	Label *l = memnew(Label);
	l->set_text(TTR("Project Name:"));
	name_container->add_child(l);

	HBoxContainer *pnhb = memnew(HBoxContainer);
	name_container->add_child(pnhb);

	project_name = memnew(LineEdit);
	project_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	pnhb->add_child(project_name);

	create_dir = memnew(Button);
	pnhb->add_child(create_dir);
	create_dir->set_text(TTR("Create Folder"));
	create_dir->connect("pressed", callable_mp(this, &ProjectDialog::_create_folder));

	path_container = memnew(VBoxContainer);
	vb->add_child(path_container);

	l = memnew(Label);
	l->set_text(TTR("Project Path:"));
	path_container->add_child(l);

	HBoxContainer *pphb = memnew(HBoxContainer);
	path_container->add_child(pphb);

	project_path = memnew(LineEdit);
	project_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	project_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	pphb->add_child(project_path);

	install_path_container = memnew(VBoxContainer);
	vb->add_child(install_path_container);

	l = memnew(Label);
	l->set_text(TTR("Project Installation Path:"));
	install_path_container->add_child(l);

	HBoxContainer *iphb = memnew(HBoxContainer);
	install_path_container->add_child(iphb);

	install_path = memnew(LineEdit);
	install_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	install_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	iphb->add_child(install_path);

	// status icon
	status_rect = memnew(TextureRect);
	status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	pphb->add_child(status_rect);

	browse = memnew(Button);
	browse->set_text(TTR("Browse"));
	browse->connect("pressed", callable_mp(this, &ProjectDialog::_browse_path));
	pphb->add_child(browse);

	// install status icon
	install_status_rect = memnew(TextureRect);
	install_status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	iphb->add_child(install_status_rect);

	install_browse = memnew(Button);
	install_browse->set_text(TTR("Browse"));
	install_browse->connect("pressed", callable_mp(this, &ProjectDialog::_browse_install_path));
	iphb->add_child(install_browse);

	msg = memnew(Label);
	msg->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	msg->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	vb->add_child(msg);

	// Renderer selection.
	renderer_container = memnew(VBoxContainer);
	vb->add_child(renderer_container);
	l = memnew(Label);
	l->set_text(TTR("Renderer:"));
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
	rs_button->set_text(TTR("Forward+"));
#if defined(WEB_ENABLED)
	rs_button->set_disabled(true);
#endif
	rs_button->set_meta(SNAME("rendering_method"), "forward_plus");
	rs_button->connect("pressed", callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	if (default_renderer_type == "forward_plus") {
		rs_button->set_pressed(true);
	}
	rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTR("Mobile"));
#if defined(WEB_ENABLED)
	rs_button->set_disabled(true);
#endif
	rs_button->set_meta(SNAME("rendering_method"), "mobile");
	rs_button->connect("pressed", callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	if (default_renderer_type == "mobile") {
		rs_button->set_pressed(true);
	}
	rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTR("Compatibility"));
#if !defined(GLES3_ENABLED)
	rs_button->set_disabled(true);
#endif
	rs_button->set_meta(SNAME("rendering_method"), "gl_compatibility");
	rs_button->connect("pressed", callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
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
	renderer_info->set_modulate(Color(1, 1, 1, 0.7));
	rvb->add_child(renderer_info);
	_renderer_selected();

	l = memnew(Label);
	l->set_text(TTR("The renderer can be changed later, but scenes may need to be adjusted."));
	// Add some extra spacing to separate it from the list above and the buttons below.
	l->set_custom_minimum_size(Size2(0, 40) * EDSCALE);
	l->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	l->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	l->set_modulate(Color(1, 1, 1, 0.7));
	renderer_container->add_child(l);

	default_files_container = memnew(HBoxContainer);
	vb->add_child(default_files_container);
	l = memnew(Label);
	l->set_text(TTR("Version Control Metadata:"));
	default_files_container->add_child(l);
	vcs_metadata_selection = memnew(OptionButton);
	vcs_metadata_selection->set_custom_minimum_size(Size2(100, 20));
	vcs_metadata_selection->add_item(TTR("None"), (int)EditorVCSInterface::VCSMetadata::NONE);
	vcs_metadata_selection->add_item(TTR("Git"), (int)EditorVCSInterface::VCSMetadata::GIT);
	vcs_metadata_selection->select((int)EditorVCSInterface::VCSMetadata::GIT);
	default_files_container->add_child(vcs_metadata_selection);
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	default_files_container->add_child(spacer);

	fdialog = memnew(EditorFileDialog);
	fdialog->set_previews_enabled(false); //Crucial, otherwise the engine crashes.
	fdialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	fdialog_install = memnew(EditorFileDialog);
	fdialog_install->set_previews_enabled(false); //Crucial, otherwise the engine crashes.
	fdialog_install->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	add_child(fdialog);
	add_child(fdialog_install);

	project_name->connect("text_changed", callable_mp(this, &ProjectDialog::_text_changed));
	project_path->connect("text_changed", callable_mp(this, &ProjectDialog::_path_text_changed));
	install_path->connect("text_changed", callable_mp(this, &ProjectDialog::_path_text_changed));
	fdialog->connect("dir_selected", callable_mp(this, &ProjectDialog::_path_selected));
	fdialog->connect("file_selected", callable_mp(this, &ProjectDialog::_file_selected));
	fdialog_install->connect("dir_selected", callable_mp(this, &ProjectDialog::_install_path_selected));
	fdialog_install->connect("file_selected", callable_mp(this, &ProjectDialog::_install_path_selected));

	set_hide_on_ok(false);

	dialog_error = memnew(AcceptDialog);
	add_child(dialog_error);
}

/// Project List and friends.

void ProjectListItemControl::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (icon_needs_reload) {
				// The project icon may not be loaded by the time the control is displayed,
				// so use a loading placeholder.
				project_icon->set_texture(get_editor_theme_icon(SNAME("ProjectIconLoading")));
			}

			project_title->begin_bulk_theme_override();
			project_title->add_theme_font_override("font", get_theme_font(SNAME("title"), EditorStringName(EditorFonts)));
			project_title->add_theme_font_size_override("font_size", get_theme_font_size(SNAME("title_size"), EditorStringName(EditorFonts)));
			project_title->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Tree")));
			project_title->end_bulk_theme_override();

			project_path->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Tree")));
			project_unsupported_features->set_texture(get_editor_theme_icon(SNAME("NodeWarning")));

			favorite_button->set_texture_normal(get_editor_theme_icon(SNAME("Favorites")));
			if (project_is_missing) {
				explore_button->set_icon(get_editor_theme_icon(SNAME("FileBroken")));
			} else {
				explore_button->set_icon(get_editor_theme_icon(SNAME("Load")));
			}
		} break;

		case NOTIFICATION_MOUSE_ENTER: {
			is_hovering = true;
			queue_redraw();
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			is_hovering = false;
			queue_redraw();
		} break;

		case NOTIFICATION_DRAW: {
			if (is_selected) {
				draw_style_box(get_theme_stylebox(SNAME("selected"), SNAME("Tree")), Rect2(Point2(), get_size()));
			}
			if (is_hovering) {
				draw_style_box(get_theme_stylebox(SNAME("hover"), SNAME("Tree")), Rect2(Point2(), get_size()));
			}

			draw_line(Point2(0, get_size().y + 1), Point2(get_size().x, get_size().y + 1), get_theme_color(SNAME("guide_color"), SNAME("Tree")));
		} break;
	}
}

void ProjectListItemControl::set_project_title(const String &p_title) {
	project_title->set_text(p_title);
}

void ProjectListItemControl::set_project_path(const String &p_path) {
	project_path->set_text(p_path);
}

void ProjectListItemControl::set_tags(const PackedStringArray &p_tags, ProjectList *p_parent_list) {
	for (const String &tag : p_tags) {
		ProjectTag *tag_control = memnew(ProjectTag(tag));
		tag_container->add_child(tag_control);
		tag_control->connect_button_to(callable_mp(p_parent_list, &ProjectList::add_search_tag).bind(tag));
	}
}

void ProjectListItemControl::set_project_icon(const Ref<Texture2D> &p_icon) {
	icon_needs_reload = false;

	// The default project icon is 128×128 to look crisp on hiDPI displays,
	// but we want the actual displayed size to be 64×64 on loDPI displays.
	project_icon->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	project_icon->set_custom_minimum_size(Size2(64, 64) * EDSCALE);
	project_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);

	project_icon->set_texture(p_icon);
}

bool _project_feature_looks_like_version(const String &p_feature) {
	return p_feature.contains(".") && p_feature.substr(0, 3).is_numeric();
}

void ProjectListItemControl::set_unsupported_features(PackedStringArray p_features) {
	if (p_features.size() > 0) {
		String tooltip_text = "";
		for (int i = 0; i < p_features.size(); i++) {
			if (_project_feature_looks_like_version(p_features[i])) {
				tooltip_text += TTR("This project was last edited in a different Godot version: ") + p_features[i] + "\n";
				p_features.remove_at(i);
				i--;
			}
		}
		if (p_features.size() > 0) {
			String unsupported_features_str = String(", ").join(p_features);
			tooltip_text += TTR("This project uses features unsupported by the current build:") + "\n" + unsupported_features_str;
		}
		project_unsupported_features->set_tooltip_text(tooltip_text);
		project_unsupported_features->show();
	} else {
		project_unsupported_features->hide();
	}
}

bool ProjectListItemControl::should_load_project_icon() const {
	return icon_needs_reload;
}

void ProjectListItemControl::set_selected(bool p_selected) {
	is_selected = p_selected;
	queue_redraw();
}

void ProjectListItemControl::set_is_favorite(bool p_favorite) {
	favorite_button->set_modulate(p_favorite ? Color(1, 1, 1, 1) : Color(1, 1, 1, 0.2));
}

void ProjectListItemControl::set_is_missing(bool p_missing) {
	if (project_is_missing == p_missing) {
		return;
	}
	project_is_missing = p_missing;

	if (project_is_missing) {
		project_icon->set_modulate(Color(1, 1, 1, 0.5));

		explore_button->set_icon(get_editor_theme_icon(SNAME("FileBroken")));
		explore_button->set_tooltip_text(TTR("Error: Project is missing on the filesystem."));
	} else {
		project_icon->set_modulate(Color(1, 1, 1, 1.0));

		explore_button->set_icon(get_editor_theme_icon(SNAME("Load")));
#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
		explore_button->set_tooltip_text(TTR("Show in File Manager"));
#else
		// Opening the system file manager is not supported on the Android and web editors.
		explore_button->hide();
#endif
	}
}

void ProjectListItemControl::set_is_grayed(bool p_grayed) {
	if (p_grayed) {
		main_vbox->set_modulate(Color(1, 1, 1, 0.5));
		// Don't make the icon less prominent if the parent is already grayed out.
		explore_button->set_modulate(Color(1, 1, 1, 1.0));
	} else {
		main_vbox->set_modulate(Color(1, 1, 1, 1.0));
		explore_button->set_modulate(Color(1, 1, 1, 0.5));
	}
}

void ProjectListItemControl::_favorite_button_pressed() {
	emit_signal(SNAME("favorite_pressed"));
}

void ProjectListItemControl::_explore_button_pressed() {
	emit_signal(SNAME("explore_pressed"));
}

void ProjectListItemControl::_bind_methods() {
	ADD_SIGNAL(MethodInfo("favorite_pressed"));
	ADD_SIGNAL(MethodInfo("explore_pressed"));
}

ProjectListItemControl::ProjectListItemControl() {
	set_focus_mode(FocusMode::FOCUS_ALL);

	VBoxContainer *favorite_box = memnew(VBoxContainer);
	favorite_box->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	add_child(favorite_box);

	favorite_button = memnew(TextureButton);
	favorite_button->set_name("FavoriteButton");
	// This makes the project's "hover" style display correctly when hovering the favorite icon.
	favorite_button->set_mouse_filter(MOUSE_FILTER_PASS);
	favorite_box->add_child(favorite_button);
	favorite_button->connect("pressed", callable_mp(this, &ProjectListItemControl::_favorite_button_pressed));

	project_icon = memnew(TextureRect);
	project_icon->set_name("ProjectIcon");
	project_icon->set_v_size_flags(SIZE_SHRINK_CENTER);
	add_child(project_icon);

	main_vbox = memnew(VBoxContainer);
	main_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(main_vbox);

	Control *ec = memnew(Control);
	ec->set_custom_minimum_size(Size2(0, 1));
	ec->set_mouse_filter(MOUSE_FILTER_PASS);
	main_vbox->add_child(ec);

	// Top half, title, tags and unsupported features labels.
	{
		HBoxContainer *title_hb = memnew(HBoxContainer);
		main_vbox->add_child(title_hb);

		project_title = memnew(Label);
		project_title->set_auto_translate(false);
		project_title->set_name("ProjectName");
		project_title->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		project_title->set_clip_text(true);
		title_hb->add_child(project_title);

		tag_container = memnew(HBoxContainer);
		title_hb->add_child(tag_container);

		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(10, 10));
		title_hb->add_child(spacer);
	}

	// Bottom half, containing the path and view folder button.
	{
		HBoxContainer *path_hb = memnew(HBoxContainer);
		path_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		main_vbox->add_child(path_hb);

		explore_button = memnew(Button);
		explore_button->set_name("ExploreButton");
		explore_button->set_flat(true);
		path_hb->add_child(explore_button);
		explore_button->connect("pressed", callable_mp(this, &ProjectListItemControl::_explore_button_pressed));

		project_path = memnew(Label);
		project_path->set_name("ProjectPath");
		project_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
		project_path->set_clip_text(true);
		project_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		project_path->set_modulate(Color(1, 1, 1, 0.5));
		path_hb->add_child(project_path);

		project_unsupported_features = memnew(TextureRect);
		project_unsupported_features->set_name("ProjectUnsupportedFeatures");
		project_unsupported_features->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		path_hb->add_child(project_unsupported_features);
		project_unsupported_features->hide();

		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(10, 10));
		path_hb->add_child(spacer);
	}
}

struct ProjectListComparator {
	ProjectList::FilterOption order_option = ProjectList::FilterOption::EDIT_DATE;

	// operator<
	_FORCE_INLINE_ bool operator()(const ProjectList::Item &a, const ProjectList::Item &b) const {
		if (a.favorite && !b.favorite) {
			return true;
		}
		if (b.favorite && !a.favorite) {
			return false;
		}
		switch (order_option) {
			case ProjectList::PATH:
				return a.path < b.path;
			case ProjectList::EDIT_DATE:
				return a.last_edited > b.last_edited;
			case ProjectList::TAGS:
				return a.tag_sort_string < b.tag_sort_string;
			default:
				return a.project_name < b.project_name;
		}
	}
};

const char *ProjectList::SIGNAL_LIST_CHANGED = "list_changed";
const char *ProjectList::SIGNAL_SELECTION_CHANGED = "selection_changed";
const char *ProjectList::SIGNAL_PROJECT_ASK_OPEN = "project_ask_open";

void ProjectList::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			// Load icons as a coroutine to speed up launch when you have hundreds of projects
			if (_icon_load_index < _projects.size()) {
				Item &item = _projects.write[_icon_load_index];
				if (item.control->should_load_project_icon()) {
					_load_project_icon(_icon_load_index);
				}
				_icon_load_index++;

			} else {
				set_process(false);
			}
		} break;
	}
}

void ProjectList::_update_icons_async() {
	_icon_load_index = 0;
	set_process(true);
}

void ProjectList::_load_project_icon(int p_index) {
	Item &item = _projects.write[p_index];

	Ref<Texture2D> default_icon = get_editor_theme_icon(SNAME("DefaultProjectIcon"));
	Ref<Texture2D> icon;
	if (!item.icon.is_empty()) {
		Ref<Image> img;
		img.instantiate();
		Error err = img->load(item.icon.replace_first("res://", item.path + "/"));
		if (err == OK) {
			img->resize(default_icon->get_width(), default_icon->get_height(), Image::INTERPOLATE_LANCZOS);
			icon = ImageTexture::create_from_image(img);
		}
	}
	if (icon.is_null()) {
		icon = default_icon;
	}

	item.control->set_project_icon(icon);
}

// Load project data from p_property_key and return it in a ProjectList::Item.
// p_favorite is passed directly into the Item.
ProjectList::Item ProjectList::load_project_data(const String &p_path, bool p_favorite) {
	String conf = p_path.path_join("project.godot");
	bool grayed = false;
	bool missing = false;

	Ref<ConfigFile> cf = memnew(ConfigFile);
	Error cf_err = cf->load(conf);

	int config_version = 0;
	String project_name = TTR("Unnamed Project");
	if (cf_err == OK) {
		String cf_project_name = cf->get_value("application", "config/name", "");
		if (!cf_project_name.is_empty()) {
			project_name = cf_project_name.xml_unescape();
		}
		config_version = (int)cf->get_value("", "config_version", 0);
	}

	if (config_version > ProjectSettings::CONFIG_VERSION) {
		// Comes from an incompatible (more recent) Godot version, gray it out.
		grayed = true;
	}

	const String description = cf->get_value("application", "config/description", "");
	const PackedStringArray tags = cf->get_value("application", "config/tags", PackedStringArray());
	const String icon = cf->get_value("application", "config/icon", "");
	const String main_scene = cf->get_value("application", "run/main_scene", "");

	PackedStringArray project_features = cf->get_value("application", "config/features", PackedStringArray());
	PackedStringArray unsupported_features = ProjectSettings::get_unsupported_features(project_features);

	uint64_t last_edited = 0;
	if (cf_err == OK) {
		// The modification date marks the date the project was last edited.
		// This is because the `project.godot` file will always be modified
		// when editing a project (but not when running it).
		last_edited = FileAccess::get_modified_time(conf);

		String fscache = p_path.path_join(".fscache");
		if (FileAccess::exists(fscache)) {
			uint64_t cache_modified = FileAccess::get_modified_time(fscache);
			if (cache_modified > last_edited) {
				last_edited = cache_modified;
			}
		}
	} else {
		grayed = true;
		missing = true;
		print_line("Project is missing: " + conf);
	}

	for (const String &tag : tags) {
		ProjectManager::get_singleton()->add_new_tag(tag);
	}

	return Item(project_name, description, tags, p_path, icon, main_scene, unsupported_features, last_edited, p_favorite, grayed, missing, config_version);
}

void ProjectList::_migrate_config() {
	// Proposal #1637 moved the project list from editor settings to a separate config file
	// If the new config file doesn't exist, populate it from EditorSettings
	if (FileAccess::exists(_config_path)) {
		return;
	}

	List<PropertyInfo> properties;
	EditorSettings::get_singleton()->get_property_list(&properties);

	for (const PropertyInfo &E : properties) {
		// This is actually something like "projects/C:::Documents::Godot::Projects::MyGame"
		String property_key = E.name;
		if (!property_key.begins_with("projects/")) {
			continue;
		}

		String path = EDITOR_GET(property_key);
		print_line("Migrating legacy project '" + path + "'.");

		String favoriteKey = "favorite_projects/" + property_key.get_slice("/", 1);
		bool favorite = EditorSettings::get_singleton()->has_setting(favoriteKey);
		add_project(path, favorite);
		if (favorite) {
			EditorSettings::get_singleton()->erase(favoriteKey);
		}
		EditorSettings::get_singleton()->erase(property_key);
	}

	save_config();
}

void ProjectList::update_project_list() {
	// This is a full, hard reload of the list. Don't call this unless really required, it's expensive.
	// If you have 150 projects, it may read through 150 files on your disk at once + load 150 icons.
	// FIXME: Does it really have to be a full, hard reload? Runtime updates should be made much cheaper.

	// Clear whole list
	for (int i = 0; i < _projects.size(); ++i) {
		Item &project = _projects.write[i];
		CRASH_COND(project.control == nullptr);
		memdelete(project.control); // Why not queue_free()?
	}
	_projects.clear();
	_last_clicked = "";
	_selected_project_paths.clear();

	List<String> sections;
	_config.load(_config_path);
	_config.get_sections(&sections);

	for (const String &path : sections) {
		bool favorite = _config.get_value(path, "favorite", false);
		_projects.push_back(load_project_data(path, favorite));
	}

	// Create controls
	for (int i = 0; i < _projects.size(); ++i) {
		_create_project_item_control(i);
	}

	sort_projects();
	_update_icons_async();
	update_dock_menu();

	set_v_scroll(0);
	emit_signal(SNAME(SIGNAL_LIST_CHANGED));
}

void ProjectList::update_dock_menu() {
	if (!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU)) {
		return;
	}
	DisplayServer::get_singleton()->global_menu_clear("_dock");

	int favs_added = 0;
	int total_added = 0;
	for (int i = 0; i < _projects.size(); ++i) {
		if (!_projects[i].grayed && !_projects[i].missing) {
			if (_projects[i].favorite) {
				favs_added++;
			} else {
				if (favs_added != 0) {
					DisplayServer::get_singleton()->global_menu_add_separator("_dock");
				}
				favs_added = 0;
			}
			DisplayServer::get_singleton()->global_menu_add_item("_dock", _projects[i].project_name + " ( " + _projects[i].path + " )", callable_mp(this, &ProjectList::_global_menu_open_project), Callable(), i);
			total_added++;
		}
	}
	if (total_added != 0) {
		DisplayServer::get_singleton()->global_menu_add_separator("_dock");
	}
	DisplayServer::get_singleton()->global_menu_add_item("_dock", TTR("New Window"), callable_mp(this, &ProjectList::_global_menu_new_window));
}

void ProjectList::_global_menu_new_window(const Variant &p_tag) {
	List<String> args;
	args.push_back("-p");
	OS::get_singleton()->create_instance(args);
}

void ProjectList::_global_menu_open_project(const Variant &p_tag) {
	int idx = (int)p_tag;

	if (idx >= 0 && idx < _projects.size()) {
		String conf = _projects[idx].path.path_join("project.godot");
		List<String> args;
		args.push_back(conf);
		OS::get_singleton()->create_instance(args);
	}
}

void ProjectList::_create_project_item_control(int p_index) {
	// Will be added last in the list, so make sure indexes match
	ERR_FAIL_COND(p_index != _scroll_children->get_child_count());

	Item &item = _projects.write[p_index];
	ERR_FAIL_COND(item.control != nullptr); // Already created

	ProjectListItemControl *hb = memnew(ProjectListItemControl);
	hb->add_theme_constant_override("separation", 10 * EDSCALE);

	hb->set_project_title(!item.missing ? item.project_name : TTR("Missing Project"));
	hb->set_project_path(item.path);
	hb->set_tooltip_text(item.description);
	hb->set_tags(item.tags, this);
	hb->set_unsupported_features(item.unsupported_features.duplicate());

	hb->set_is_favorite(item.favorite);
	hb->set_is_missing(item.missing);
	hb->set_is_grayed(item.grayed);

	hb->connect("gui_input", callable_mp(this, &ProjectList::_panel_input).bind(hb));
	hb->connect("favorite_pressed", callable_mp(this, &ProjectList::_favorite_pressed).bind(hb));

#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
	hb->connect("explore_pressed", callable_mp(this, &ProjectList::_show_project).bind(item.path));
#endif

	_scroll_children->add_child(hb);
	item.control = hb;
}

void ProjectList::set_search_term(String p_search_term) {
	_search_term = p_search_term;
}

void ProjectList::set_order_option(int p_option) {
	FilterOption selected = (FilterOption)p_option;
	EditorSettings::get_singleton()->set("project_manager/sorting_order", p_option);
	EditorSettings::get_singleton()->save();
	_order_option = selected;

	sort_projects();
}

void ProjectList::sort_projects() {
	SortArray<Item, ProjectListComparator> sorter;
	sorter.compare.order_option = _order_option;
	sorter.sort(_projects.ptrw(), _projects.size());

	String search_term;
	PackedStringArray tags;

	if (!_search_term.is_empty()) {
		PackedStringArray search_parts = _search_term.split(" ");
		if (search_parts.size() > 1 || search_parts[0].begins_with("tag:")) {
			PackedStringArray remaining;
			for (const String &part : search_parts) {
				if (part.begins_with("tag:")) {
					tags.push_back(part.get_slice(":", 1));
				} else {
					remaining.append(part);
				}
			}
			search_term = String(" ").join(remaining); // Search term without tags.
		} else {
			search_term = _search_term;
		}
	}

	for (int i = 0; i < _projects.size(); ++i) {
		Item &item = _projects.write[i];

		bool item_visible = true;
		if (!_search_term.is_empty()) {
			String search_path;
			if (search_term.contains("/")) {
				// Search path will match the whole path
				search_path = item.path;
			} else {
				// Search path will only match the last path component to make searching more strict
				search_path = item.path.get_file();
			}

			bool missing_tags = false;
			for (const String &tag : tags) {
				if (!item.tags.has(tag)) {
					missing_tags = true;
					break;
				}
			}

			// When searching, display projects whose name or path contain the search term and whose tags match the searched tags.
			item_visible = !missing_tags && (search_term.is_empty() || item.project_name.findn(search_term) != -1 || search_path.findn(search_term) != -1);
		}

		item.control->set_visible(item_visible);
	}

	for (int i = 0; i < _projects.size(); ++i) {
		Item &item = _projects.write[i];
		item.control->get_parent()->move_child(item.control, i);
	}

	// Rewind the coroutine because order of projects changed
	_update_icons_async();
	update_dock_menu();
}

const HashSet<String> &ProjectList::get_selected_project_keys() const {
	// Faster if that's all you need
	return _selected_project_paths;
}

Vector<ProjectList::Item> ProjectList::get_selected_projects() const {
	Vector<Item> items;
	if (_selected_project_paths.size() == 0) {
		return items;
	}
	items.resize(_selected_project_paths.size());
	int j = 0;
	for (int i = 0; i < _projects.size(); ++i) {
		const Item &item = _projects[i];
		if (_selected_project_paths.has(item.path)) {
			items.write[j++] = item;
		}
	}
	ERR_FAIL_COND_V(j != items.size(), items);
	return items;
}

void ProjectList::ensure_project_visible(int p_index) {
	const Item &item = _projects[p_index];
	ensure_control_visible(item.control);
}

int ProjectList::get_single_selected_index() const {
	if (_selected_project_paths.size() == 0) {
		// Default selection
		return 0;
	}
	String key;
	if (_selected_project_paths.size() == 1) {
		// Only one selected
		key = *_selected_project_paths.begin();
	} else {
		// Multiple selected, consider the last clicked one as "main"
		key = _last_clicked;
	}
	for (int i = 0; i < _projects.size(); ++i) {
		if (_projects[i].path == key) {
			return i;
		}
	}
	return 0;
}

void ProjectList::_remove_project(int p_index, bool p_update_config) {
	const Item item = _projects[p_index]; // Take a copy

	_selected_project_paths.erase(item.path);

	if (_last_clicked == item.path) {
		_last_clicked = "";
	}

	memdelete(item.control);
	_projects.remove_at(p_index);

	if (p_update_config) {
		_config.erase_section(item.path);
		// Not actually saving the file, in case you are doing more changes to settings
	}

	update_dock_menu();
}

bool ProjectList::is_any_project_missing() const {
	for (int i = 0; i < _projects.size(); ++i) {
		if (_projects[i].missing) {
			return true;
		}
	}
	return false;
}

void ProjectList::erase_missing_projects() {
	if (_projects.is_empty()) {
		return;
	}

	int deleted_count = 0;
	int remaining_count = 0;

	for (int i = 0; i < _projects.size(); ++i) {
		const Item &item = _projects[i];

		if (item.missing) {
			_remove_project(i, true);
			--i;
			++deleted_count;

		} else {
			++remaining_count;
		}
	}

	print_line("Removed " + itos(deleted_count) + " projects from the list, remaining " + itos(remaining_count) + " projects");
	save_config();
}

void ProjectList::_scan_folder_recursive(const String &p_path, List<String> *r_projects) {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error error = da->change_dir(p_path);
	ERR_FAIL_COND_MSG(error != OK, vformat("Failed to open the path \"%s\" for scanning (code %d).", p_path, error));

	da->list_dir_begin();
	String n = da->get_next();
	while (!n.is_empty()) {
		if (da->current_is_dir() && n[0] != '.') {
			_scan_folder_recursive(da->get_current_dir().path_join(n), r_projects);
		} else if (n == "project.godot") {
			r_projects->push_back(da->get_current_dir());
		}
		n = da->get_next();
	}
	da->list_dir_end();
}

void ProjectList::find_projects(const String &p_path) {
	PackedStringArray paths = { p_path };
	find_projects_multiple(paths);
}

void ProjectList::find_projects_multiple(const PackedStringArray &p_paths) {
	List<String> projects;

	for (int i = 0; i < p_paths.size(); i++) {
		const String &base_path = p_paths.get(i);
		print_verbose(vformat("Scanning for projects in \"%s\".", base_path));

		_scan_folder_recursive(base_path, &projects);
		print_verbose(vformat("Found %d project(s).", projects.size()));
	}

	for (const String &E : projects) {
		add_project(E, false);
	}

	save_config();
	update_project_list();
}

int ProjectList::refresh_project(const String &dir_path) {
	// Reloads information about a specific project.
	// If it wasn't loaded and should be in the list, it is added (i.e new project).
	// If it isn't in the list anymore, it is removed.
	// If it is in the list but doesn't exist anymore, it is marked as missing.

	bool should_be_in_list = _config.has_section(dir_path);
	bool is_favourite = _config.get_value(dir_path, "favorite", false);

	bool was_selected = _selected_project_paths.has(dir_path);

	// Remove item in any case
	for (int i = 0; i < _projects.size(); ++i) {
		const Item &existing_item = _projects[i];
		if (existing_item.path == dir_path) {
			_remove_project(i, false);
			break;
		}
	}

	int index = -1;
	if (should_be_in_list) {
		// Recreate it with updated info

		Item item = load_project_data(dir_path, is_favourite);

		_projects.push_back(item);
		_create_project_item_control(_projects.size() - 1);

		sort_projects();

		for (int i = 0; i < _projects.size(); ++i) {
			if (_projects[i].path == dir_path) {
				if (was_selected) {
					select_project(i);
					ensure_project_visible(i);
				}
				_load_project_icon(i);

				index = i;
				break;
			}
		}
	}

	return index;
}

void ProjectList::add_project(const String &dir_path, bool favorite) {
	if (!_config.has_section(dir_path)) {
		_config.set_value(dir_path, "favorite", favorite);
	}
}

void ProjectList::save_config() {
	_config.save(_config_path);
}

void ProjectList::set_project_version(const String &p_project_path, int p_version) {
	for (ProjectList::Item &E : _projects) {
		if (E.path == p_project_path) {
			E.version = p_version;
			break;
		}
	}
}

int ProjectList::get_project_count() const {
	return _projects.size();
}

void ProjectList::_clear_project_selection() {
	Vector<Item> previous_selected_items = get_selected_projects();
	_selected_project_paths.clear();

	for (int i = 0; i < previous_selected_items.size(); ++i) {
		previous_selected_items[i].control->set_selected(false);
	}
}

void ProjectList::_toggle_project(int p_index) {
	// This methods adds to the selection or removes from the
	// selection.
	Item &item = _projects.write[p_index];

	if (_selected_project_paths.has(item.path)) {
		_deselect_project_nocheck(p_index);
	} else {
		_select_project_nocheck(p_index);
	}
}

void ProjectList::_select_project_nocheck(int p_index) {
	Item &item = _projects.write[p_index];
	_selected_project_paths.insert(item.path);
	item.control->set_selected(true);
}

void ProjectList::_deselect_project_nocheck(int p_index) {
	Item &item = _projects.write[p_index];
	_selected_project_paths.erase(item.path);
	item.control->set_selected(false);
}

void ProjectList::select_project(int p_index) {
	// This method keeps only one project selected.
	_clear_project_selection();
	_select_project_nocheck(p_index);
}

void ProjectList::select_first_visible_project() {
	_clear_project_selection();

	for (int i = 0; i < _projects.size(); i++) {
		if (_projects[i].control->is_visible()) {
			_select_project_nocheck(i);
			break;
		}
	}
}

inline void _sort_project_range(int &a, int &b) {
	if (a > b) {
		int temp = a;
		a = b;
		b = temp;
	}
}

void ProjectList::_select_project_range(int p_begin, int p_end) {
	_clear_project_selection();

	_sort_project_range(p_begin, p_end);
	for (int i = p_begin; i <= p_end; ++i) {
		_select_project_nocheck(i);
	}
}

void ProjectList::erase_selected_projects(bool p_delete_project_contents) {
	if (_selected_project_paths.size() == 0) {
		return;
	}

	for (int i = 0; i < _projects.size(); ++i) {
		Item &item = _projects.write[i];
		if (_selected_project_paths.has(item.path) && item.control->is_visible()) {
			_config.erase_section(item.path);

			// Comment out for now until we have a better warning system to
			// ensure users delete their project only.
			//if (p_delete_project_contents) {
			//	OS::get_singleton()->move_to_trash(item.path);
			//}

			memdelete(item.control);
			_projects.remove_at(i);
			--i;
		}
	}

	save_config();
	_selected_project_paths.clear();
	_last_clicked = "";

	update_dock_menu();
}

// Input for each item in the list.
void ProjectList::_panel_input(const Ref<InputEvent> &p_ev, Node *p_hb) {
	Ref<InputEventMouseButton> mb = p_ev;
	int clicked_index = p_hb->get_index();
	const Item &clicked_project = _projects[clicked_index];

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_shift_pressed() && _selected_project_paths.size() > 0 && !_last_clicked.is_empty() && clicked_project.path != _last_clicked) {
			int anchor_index = -1;
			for (int i = 0; i < _projects.size(); ++i) {
				const Item &p = _projects[i];
				if (p.path == _last_clicked) {
					anchor_index = p.control->get_index();
					break;
				}
			}
			CRASH_COND(anchor_index == -1);
			_select_project_range(anchor_index, clicked_index);

		} else if (mb->is_command_or_control_pressed()) {
			_toggle_project(clicked_index);

		} else {
			_last_clicked = clicked_project.path;
			select_project(clicked_index);
		}

		emit_signal(SNAME(SIGNAL_SELECTION_CHANGED));

		// Do not allow opening a project more than once using a single project manager instance.
		// Opening the same project in several editor instances at once can lead to various issues.
		if (!mb->is_command_or_control_pressed() && mb->is_double_click() && !project_opening_initiated) {
			emit_signal(SNAME(SIGNAL_PROJECT_ASK_OPEN));
		}
	}
}

void ProjectList::_favorite_pressed(Node *p_hb) {
	ProjectListItemControl *control = Object::cast_to<ProjectListItemControl>(p_hb);

	int index = control->get_index();
	Item item = _projects.write[index]; // Take copy

	item.favorite = !item.favorite;

	_config.set_value(item.path, "favorite", item.favorite);
	save_config();

	_projects.write[index] = item;

	control->set_is_favorite(item.favorite);

	sort_projects();

	if (item.favorite) {
		for (int i = 0; i < _projects.size(); ++i) {
			if (_projects[i].path == item.path) {
				ensure_project_visible(i);
				break;
			}
		}
	}

	update_dock_menu();
}

void ProjectList::_show_project(const String &p_path) {
	OS::get_singleton()->shell_show_in_file_manager(p_path, true);
}

void ProjectList::_bind_methods() {
	ADD_SIGNAL(MethodInfo(SIGNAL_LIST_CHANGED));
	ADD_SIGNAL(MethodInfo(SIGNAL_SELECTION_CHANGED));
	ADD_SIGNAL(MethodInfo(SIGNAL_PROJECT_ASK_OPEN));
}

ProjectList::ProjectList() {
	_scroll_children = memnew(VBoxContainer);
	_scroll_children->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(_scroll_children);

	_config_path = EditorPaths::get_singleton()->get_data_dir().path_join("projects.cfg");
	_migrate_config();
}

/// Project Manager.

ProjectManager *ProjectManager::singleton = nullptr;

void ProjectManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			settings_hb->set_anchors_and_offsets_preset(Control::PRESET_TOP_RIGHT);
			queue_redraw();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			Engine::get_singleton()->set_editor_hint(false);
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			background_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("Background"), EditorStringName(EditorStyles)));
			loading_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), EditorStringName(EditorFonts)));
			search_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("search_panel"), SNAME("ProjectManager")));

			// Top bar.
			search_box->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			language_btn->set_icon(get_editor_theme_icon(SNAME("Environment")));

			// Sidebar.
			create_btn->set_icon(get_editor_theme_icon(SNAME("Add")));
			import_btn->set_icon(get_editor_theme_icon(SNAME("Load")));
			scan_btn->set_icon(get_editor_theme_icon(SNAME("Search")));
			open_btn->set_icon(get_editor_theme_icon(SNAME("Edit")));
			run_btn->set_icon(get_editor_theme_icon(SNAME("Play")));
			rename_btn->set_icon(get_editor_theme_icon(SNAME("Rename")));
			manage_tags_btn->set_icon(get_editor_theme_icon("Script"));
			erase_btn->set_icon(get_editor_theme_icon(SNAME("Remove")));
			erase_missing_btn->set_icon(get_editor_theme_icon(SNAME("Clear")));
			create_tag_btn->set_icon(get_editor_theme_icon("Add"));

			tag_error->add_theme_color_override("font_color", get_theme_color("error_color", EditorStringName(Editor)));
			tag_edit_error->add_theme_color_override("font_color", get_theme_color("error_color", EditorStringName(Editor)));

			create_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			import_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			scan_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			open_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			run_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			rename_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			manage_tags_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			erase_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));
			erase_missing_btn->add_theme_constant_override("h_separation", get_theme_constant(SNAME("sidebar_button_icon_separation"), SNAME("ProjectManager")));

			// Asset library popup.
			if (asset_library) {
				// Removes extra border margins.
				asset_library->add_theme_style_override("panel", memnew(StyleBoxEmpty));
			}
		} break;

		case NOTIFICATION_RESIZED: {
			if (open_templates && open_templates->is_visible()) {
				open_templates->popup_centered();
			}
			if (asset_library) {
				real_t size = get_size().x / EDSCALE;
				// Adjust names of tabs to fit the new size.
				if (size < 650) {
					local_projects_vb->set_name(TTR("Local"));
					asset_library->set_name(TTR("Asset Library"));
				} else {
					local_projects_vb->set_name(TTR("Local Projects"));
					asset_library->set_name(TTR("Asset Library Projects"));
				}
			}
		} break;

		case NOTIFICATION_READY: {
			int default_sorting = (int)EDITOR_GET("project_manager/sorting_order");
			filter_option->select(default_sorting);
			_project_list->set_order_option(default_sorting);

#ifndef ANDROID_ENABLED
			if (_project_list->get_project_count() >= 1) {
				// Focus on the search box immediately to allow the user
				// to search without having to reach for their mouse
				search_box->grab_focus();
			}
#endif

			// Suggest browsing asset library to get templates/demos.
			if (asset_library && open_templates && _project_list->get_project_count() == 0) {
				open_templates->popup_centered();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_shortcut_input(is_visible_in_tree());
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_dim_window();
		} break;

		case NOTIFICATION_WM_ABOUT: {
			_show_about();
		} break;
	}
}

Ref<Texture2D> ProjectManager::_file_dialog_get_icon(const String &p_path) {
	if (p_path.get_extension().to_lower() == "godot") {
		return singleton->icon_type_cache["GodotMonochrome"];
	}

	return singleton->icon_type_cache["Object"];
}

Ref<Texture2D> ProjectManager::_file_dialog_get_thumbnail(const String &p_path) {
	if (p_path.get_extension().to_lower() == "godot") {
		return singleton->icon_type_cache["GodotFile"];
	}

	return Ref<Texture2D>();
}

void ProjectManager::_build_icon_type_cache(Ref<Theme> p_theme) {
	if (p_theme.is_null()) {
		return;
	}
	List<StringName> tl;
	p_theme->get_icon_list(EditorStringName(EditorIcons), &tl);
	for (List<StringName>::Element *E = tl.front(); E; E = E->next()) {
		icon_type_cache[E->get()] = p_theme->get_icon(E->get(), EditorStringName(EditorIcons));
	}
}

void ProjectManager::_update_size_limits() {
	const Size2 minimum_size = Size2(680, 450) * EDSCALE;
	const Size2 default_size = Size2(1024, 600) * EDSCALE;

	// Define a minimum window size to prevent UI elements from overlapping or being cut off.
	Window *w = Object::cast_to<Window>(SceneTree::get_singleton()->get_root());
	if (w) {
		// Calling Window methods this early doesn't sync properties with DS.
		w->set_min_size(minimum_size);
		DisplayServer::get_singleton()->window_set_min_size(minimum_size);
		w->set_size(default_size);
		DisplayServer::get_singleton()->window_set_size(default_size);
	}

	Rect2i screen_rect = DisplayServer::get_singleton()->screen_get_usable_rect(DisplayServer::get_singleton()->window_get_current_screen());
	if (screen_rect.size != Vector2i()) {
		// Center the window on the screen.
		Vector2i window_position;
		window_position.x = screen_rect.position.x + (screen_rect.size.x - default_size.x) / 2;
		window_position.y = screen_rect.position.y + (screen_rect.size.y - default_size.y) / 2;
		DisplayServer::get_singleton()->window_set_position(window_position);

		// Limit popup menus to prevent unusably long lists.
		// We try to set it to half the screen resolution, but no smaller than the minimum window size.
		Size2 half_screen_rect = (screen_rect.size * EDSCALE) / 2;
		Size2 maximum_popup_size = MAX(half_screen_rect, minimum_size);
		language_btn->get_popup()->set_max_size(maximum_popup_size);
	}
}

void ProjectManager::_dim_window() {
	// This method must be called before calling `get_tree()->quit()`.
	// Otherwise, its effect won't be visible

	// Dim the project manager window while it's quitting to make it clearer that it's busy.
	// No transition is applied, as the effect needs to be visible immediately
	float c = 0.5f;
	Color dim_color = Color(c, c, c);
	set_modulate(dim_color);
}

void ProjectManager::_update_project_buttons() {
	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	bool empty_selection = selected_projects.is_empty();

	bool is_missing_project_selected = false;
	for (int i = 0; i < selected_projects.size(); ++i) {
		if (selected_projects[i].missing) {
			is_missing_project_selected = true;
			break;
		}
	}

	erase_btn->set_disabled(empty_selection);
	open_btn->set_disabled(empty_selection || is_missing_project_selected);
	rename_btn->set_disabled(empty_selection || is_missing_project_selected);
	manage_tags_btn->set_disabled(empty_selection || is_missing_project_selected || selected_projects.size() > 1);
	run_btn->set_disabled(empty_selection || is_missing_project_selected);

	erase_missing_btn->set_disabled(!_project_list->is_any_project_missing());
}

void ProjectManager::shortcut_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventKey> k = p_ev;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}

		// Pressing Command + Q quits the Project Manager
		// This is handled by the platform implementation on macOS,
		// so only define the shortcut on other platforms
#ifndef MACOS_ENABLED
		if (k->get_keycode_with_modifiers() == (KeyModifierMask::META | Key::Q)) {
			_dim_window();
			get_tree()->quit();
		}
#endif

		if (tabs->get_current_tab() != 0) {
			return;
		}

		bool keycode_handled = true;

		switch (k->get_keycode()) {
			case Key::ENTER: {
				_open_selected_projects_ask();
			} break;
			case Key::HOME: {
				if (_project_list->get_project_count() > 0) {
					_project_list->select_project(0);
					_update_project_buttons();
				}

			} break;
			case Key::END: {
				if (_project_list->get_project_count() > 0) {
					_project_list->select_project(_project_list->get_project_count() - 1);
					_update_project_buttons();
				}

			} break;
			case Key::UP: {
				if (k->is_shift_pressed()) {
					break;
				}

				int index = _project_list->get_single_selected_index();
				if (index > 0) {
					_project_list->select_project(index - 1);
					_project_list->ensure_project_visible(index - 1);
					_update_project_buttons();
				}

				break;
			}
			case Key::DOWN: {
				if (k->is_shift_pressed()) {
					break;
				}

				int index = _project_list->get_single_selected_index();
				if (index + 1 < _project_list->get_project_count()) {
					_project_list->select_project(index + 1);
					_project_list->ensure_project_visible(index + 1);
					_update_project_buttons();
				}

			} break;
			case Key::F: {
				if (k->is_command_or_control_pressed()) {
					this->search_box->grab_focus();
				} else {
					keycode_handled = false;
				}
			} break;
			default: {
				keycode_handled = false;
			} break;
		}

		if (keycode_handled) {
			accept_event();
		}
	}
}

void ProjectManager::_on_projects_updated() {
	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	int index = 0;
	for (int i = 0; i < selected_projects.size(); ++i) {
		index = _project_list->refresh_project(selected_projects[i].path);
	}
	if (index != -1) {
		_project_list->ensure_project_visible(index);
	}

	_project_list->update_dock_menu();
}

void ProjectManager::_on_project_created(const String &dir) {
	_project_list->add_project(dir, false);
	_project_list->save_config();
	search_box->clear();
	int i = _project_list->refresh_project(dir);
	_project_list->select_project(i);
	_project_list->ensure_project_visible(i);
	_open_selected_projects_ask();

	_project_list->update_dock_menu();
}

void ProjectManager::_confirm_update_settings() {
	_open_selected_projects();
}

void ProjectManager::_open_selected_projects() {
	// Show loading text to tell the user that the project manager is busy loading.
	// This is especially important for the Web project manager.
	loading_label->show();

	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	for (const String &path : selected_list) {
		String conf = path.path_join("project.godot");

		if (!FileAccess::exists(conf)) {
			dialog_error->set_text(vformat(TTR("Can't open project at '%s'."), path));
			dialog_error->popup_centered();
			return;
		}

		print_line("Editing project: " + path);

		List<String> args;

		for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_TOOL)) {
			args.push_back(a);
		}

		args.push_back("--path");
		args.push_back(path);

		args.push_back("--editor");

		Error err = OS::get_singleton()->create_instance(args);
		ERR_FAIL_COND(err);
	}

	_project_list->project_opening_initiated = true;

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_open_selected_projects_ask() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	const Size2i popup_min_size = Size2i(600.0 * EDSCALE, 0);

	if (selected_list.size() > 1) {
		multi_open_ask->set_text(vformat(TTR("You requested to open %d projects in parallel. Do you confirm?\nNote that usual checks for engine version compatibility will be bypassed."), selected_list.size()));
		multi_open_ask->popup_centered(popup_min_size);
		return;
	}

	ProjectList::Item project = _project_list->get_selected_projects()[0];
	if (project.missing) {
		return;
	}

	// Update the project settings or don't open.
	const int config_version = project.version;
	PackedStringArray unsupported_features = project.unsupported_features;

	Label *ask_update_label = ask_update_settings->get_label();
	ask_update_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT); // Reset in case of previous center align.
	full_convert_button->hide();

	ask_update_settings->get_ok_button()->set_text("OK");

	// Check if the config_version property was empty or 0.
	if (config_version == 0) {
		ask_update_settings->set_text(vformat(TTR("The selected project \"%s\" does not specify its supported Godot version in its configuration file (\"project.godot\").\n\nProject path: %s\n\nIf you proceed with opening it, it will be converted to Godot's current configuration file format.\n\nWarning: You won't be able to open the project with previous versions of the engine anymore."), project.project_name, project.path));
		ask_update_settings->popup_centered(popup_min_size);
		return;
	}
	// Check if we need to convert project settings from an earlier engine version.
	if (config_version < ProjectSettings::CONFIG_VERSION) {
		if (config_version == GODOT4_CONFIG_VERSION - 1 && ProjectSettings::CONFIG_VERSION == GODOT4_CONFIG_VERSION) { // Conversion from Godot 3 to 4.
			full_convert_button->show();
			ask_update_settings->set_text(vformat(TTR("The selected project \"%s\" was generated by Godot 3.x, and needs to be converted for Godot 4.x.\n\nProject path: %s\n\nYou have three options:\n- Convert only the configuration file (\"project.godot\"). Use this to open the project without attempting to convert its scenes, resources and scripts.\n- Convert the entire project including its scenes, resources and scripts (recommended if you are upgrading).\n- Do nothing and go back.\n\nWarning: If you select a conversion option, you won't be able to open the project with previous versions of the engine anymore."), project.project_name, project.path));
			ask_update_settings->get_ok_button()->set_text(TTR("Convert project.godot Only"));
		} else {
			ask_update_settings->set_text(vformat(TTR("The selected project \"%s\" was generated by an older engine version, and needs to be converted for this version.\n\nProject path: %s\n\nDo you want to convert it?\n\nWarning: You won't be able to open the project with previous versions of the engine anymore."), project.project_name, project.path));
			ask_update_settings->get_ok_button()->set_text(TTR("Convert project.godot"));
		}
		ask_update_settings->popup_centered(popup_min_size);
		ask_update_settings->get_cancel_button()->grab_focus(); // To prevent accidents.
		return;
	}
	// Check if the file was generated by a newer, incompatible engine version.
	if (config_version > ProjectSettings::CONFIG_VERSION) {
		dialog_error->set_text(vformat(TTR("Can't open project \"%s\" at the following path:\n\n%s\n\nThe project settings were created by a newer engine version, whose settings are not compatible with this version."), project.project_name, project.path));
		dialog_error->popup_centered(popup_min_size);
		return;
	}
	// Check if the project is using features not supported by this build of Godot.
	if (!unsupported_features.is_empty()) {
		String warning_message = "";
		for (int i = 0; i < unsupported_features.size(); i++) {
			String feature = unsupported_features[i];
			if (feature == "Double Precision") {
				warning_message += TTR("Warning: This project uses double precision floats, but this version of\nGodot uses single precision floats. Opening this project may cause data loss.\n\n");
				unsupported_features.remove_at(i);
				i--;
			} else if (feature == "C#") {
				warning_message += TTR("Warning: This project uses C#, but this build of Godot does not have\nthe Mono module. If you proceed you will not be able to use any C# scripts.\n\n");
				unsupported_features.remove_at(i);
				i--;
			} else if (_project_feature_looks_like_version(feature)) {
				warning_message += vformat(TTR("Warning: This project was last edited in Godot %s. Opening will change it to Godot %s.\n\n"), Variant(feature), Variant(VERSION_BRANCH));
				unsupported_features.remove_at(i);
				i--;
			}
		}
		if (!unsupported_features.is_empty()) {
			String unsupported_features_str = String(", ").join(unsupported_features);
			warning_message += vformat(TTR("Warning: This project uses the following features not supported by this build of Godot:\n\n%s\n\n"), unsupported_features_str);
		}
		warning_message += TTR("Open anyway? Project will be modified.");
		ask_update_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		ask_update_settings->set_text(warning_message);
		ask_update_settings->popup_centered(popup_min_size);
		return;
	}

	// Open if the project is up-to-date.
	_open_selected_projects();
}

void ProjectManager::_full_convert_button_pressed() {
	ask_update_settings->hide();
	ask_full_convert_dialog->popup_centered(Size2i(600.0 * EDSCALE, 0));
	ask_full_convert_dialog->get_cancel_button()->grab_focus();
}

void ProjectManager::_perform_full_project_conversion() {
	Vector<ProjectList::Item> selected_list = _project_list->get_selected_projects();
	if (selected_list.is_empty()) {
		return;
	}

	const String &path = selected_list[0].path;

	print_line("Converting project: " + path);
	List<String> args;
	args.push_back("--path");
	args.push_back(path);
	args.push_back("--convert-3to4");
	args.push_back("--rendering-driver");
	args.push_back(Main::get_rendering_driver_name());

	Error err = OS::get_singleton()->create_instance(args);
	ERR_FAIL_COND(err);

	_project_list->set_project_version(path, GODOT4_CONFIG_VERSION);
}

void ProjectManager::_run_project_confirm() {
	Vector<ProjectList::Item> selected_list = _project_list->get_selected_projects();

	for (int i = 0; i < selected_list.size(); ++i) {
		const String &selected_main = selected_list[i].main_scene;
		if (selected_main.is_empty()) {
			run_error_diag->set_text(TTR("Can't run project: no main scene defined.\nPlease edit the project and set the main scene in the Project Settings under the \"Application\" category."));
			run_error_diag->popup_centered();
			continue;
		}

		const String &path = selected_list[i].path;

		// `.substr(6)` on `ProjectSettings::get_singleton()->get_imported_files_path()` strips away the leading "res://".
		if (!DirAccess::exists(path.path_join(ProjectSettings::get_singleton()->get_imported_files_path().substr(6)))) {
			run_error_diag->set_text(TTR("Can't run project: Assets need to be imported.\nPlease edit the project to trigger the initial import."));
			run_error_diag->popup_centered();
			continue;
		}

		print_line("Running project: " + path);

		List<String> args;

		for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_PROJECT)) {
			args.push_back(a);
		}

		args.push_back("--path");
		args.push_back(path);

		Error err = OS::get_singleton()->create_instance(args);
		ERR_FAIL_COND(err);
	}
}

void ProjectManager::_run_project() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	if (selected_list.size() > 1) {
		multi_run_ask->set_text(vformat(TTR("Are you sure to run %d projects at once?"), selected_list.size()));
		multi_run_ask->popup_centered();
	} else {
		_run_project_confirm();
	}
}

void ProjectManager::_scan_projects() {
	scan_dir->popup_file_dialog();
}

void ProjectManager::_new_project() {
	npdialog->set_mode(ProjectDialog::MODE_NEW);
	npdialog->show_dialog();
}

void ProjectManager::_import_project() {
	npdialog->set_mode(ProjectDialog::MODE_IMPORT);
	npdialog->ask_for_path_and_show();
}

void ProjectManager::_rename_project() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0) {
		return;
	}

	for (const String &E : selected_list) {
		npdialog->set_project_path(E);
		npdialog->set_mode(ProjectDialog::MODE_RENAME);
		npdialog->show_dialog();
	}
}

void ProjectManager::_manage_project_tags() {
	for (int i = 0; i < project_tags->get_child_count(); i++) {
		project_tags->get_child(i)->queue_free();
	}

	const ProjectList::Item item = _project_list->get_selected_projects()[0];
	current_project_tags = item.tags;
	for (const String &tag : current_project_tags) {
		ProjectTag *tag_control = memnew(ProjectTag(tag, true));
		project_tags->add_child(tag_control);
		tag_control->connect_button_to(callable_mp(this, &ProjectManager::_delete_project_tag).bind(tag));
	}

	tag_edit_error->hide();
	tag_manage_dialog->popup_centered(Vector2i(500, 0) * EDSCALE);
}

void ProjectManager::_add_project_tag(const String &p_tag) {
	if (current_project_tags.has(p_tag)) {
		return;
	}
	current_project_tags.append(p_tag);

	ProjectTag *tag_control = memnew(ProjectTag(p_tag, true));
	project_tags->add_child(tag_control);
	tag_control->connect_button_to(callable_mp(this, &ProjectManager::_delete_project_tag).bind(p_tag));
}

void ProjectManager::_delete_project_tag(const String &p_tag) {
	current_project_tags.erase(p_tag);
	for (int i = 0; i < project_tags->get_child_count(); i++) {
		ProjectTag *tag_control = Object::cast_to<ProjectTag>(project_tags->get_child(i));
		if (tag_control && tag_control->get_tag() == p_tag) {
			memdelete(tag_control);
			break;
		}
	}
}

void ProjectManager::_apply_project_tags() {
	PackedStringArray tags;
	for (int i = 0; i < project_tags->get_child_count(); i++) {
		ProjectTag *tag_control = Object::cast_to<ProjectTag>(project_tags->get_child(i));
		if (tag_control) {
			tags.append(tag_control->get_tag());
		}
	}

	ConfigFile cfg;
	const String project_godot = _project_list->get_selected_projects()[0].path.path_join("project.godot");
	Error err = cfg.load(project_godot);
	if (err != OK) {
		tag_edit_error->set_text(vformat(TTR("Couldn't load project at '%s' (error %d). It may be missing or corrupted."), project_godot, err));
		tag_edit_error->show();
		callable_mp((Window *)tag_manage_dialog, &Window::show).call_deferred(); // Make sure the dialog does not disappear.
		return;
	} else {
		tags.sort();
		cfg.set_value("application", "config/tags", tags);
		err = cfg.save(project_godot);
		if (err != OK) {
			tag_edit_error->set_text(vformat(TTR("Couldn't save project at '%s' (error %d)."), project_godot, err));
			tag_edit_error->show();
			callable_mp((Window *)tag_manage_dialog, &Window::show).call_deferred();
			return;
		}
	}

	_on_projects_updated();
}

void ProjectManager::_set_new_tag_name(const String p_name) {
	create_tag_dialog->get_ok_button()->set_disabled(true);
	if (p_name.is_empty()) {
		tag_error->set_text(TTR("Tag name can't be empty."));
		return;
	}

	if (p_name.contains(" ")) {
		tag_error->set_text(TTR("Tag name can't contain spaces."));
		return;
	}

	for (const String &c : forbidden_tag_characters) {
		if (p_name.contains(c)) {
			tag_error->set_text(vformat(TTR("These characters are not allowed in tags: %s."), String(" ").join(forbidden_tag_characters)));
			return;
		}
	}

	if (p_name.to_lower() != p_name) {
		tag_error->set_text(TTR("Tag name must be lowercase."));
		return;
	}

	tag_error->set_text("");
	create_tag_dialog->get_ok_button()->set_disabled(false);
}

void ProjectManager::_create_new_tag() {
	if (!tag_error->get_text().is_empty()) {
		return;
	}
	create_tag_dialog->hide(); // When using text_submitted, need to hide manually.
	add_new_tag(new_tag_name->get_text());
	_add_project_tag(new_tag_name->get_text());
}

void ProjectManager::_erase_project_confirm() {
	_project_list->erase_selected_projects(false);
	_update_project_buttons();
}

void ProjectManager::_erase_missing_projects_confirm() {
	_project_list->erase_missing_projects();
	_update_project_buttons();
}

void ProjectManager::_erase_project() {
	const HashSet<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0) {
		return;
	}

	String confirm_message;
	if (selected_list.size() >= 2) {
		confirm_message = vformat(TTR("Remove %d projects from the list?"), selected_list.size());
	} else {
		confirm_message = TTR("Remove this project from the list?");
	}

	erase_ask_label->set_text(confirm_message);
	//delete_project_contents->set_pressed(false);
	erase_ask->popup_centered();
}

void ProjectManager::_erase_missing_projects() {
	erase_missing_ask->set_text(TTR("Remove all missing projects from the list?\nThe project folders' contents won't be modified."));
	erase_missing_ask->popup_centered();
}

void ProjectManager::_show_about() {
	about->popup_centered(Size2(780, 500) * EDSCALE);
}

void ProjectManager::_language_selected(int p_id) {
	String lang = language_btn->get_item_metadata(p_id);
	EditorSettings::get_singleton()->set("interface/editor/editor_language", lang);

	language_restart_ask->set_text(TTR("Language changed.\nThe interface will update after restarting the editor or project manager."));
	language_restart_ask->popup_centered();
}

void ProjectManager::_restart_confirm() {
	List<String> args = OS::get_singleton()->get_cmdline_args();
	Error err = OS::get_singleton()->create_instance(args);
	ERR_FAIL_COND(err);

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_install_project(const String &p_zip_path, const String &p_title) {
	npdialog->set_mode(ProjectDialog::MODE_INSTALL);
	npdialog->set_zip_path(p_zip_path);
	npdialog->set_zip_title(p_title);
	npdialog->show_dialog();
}

void ProjectManager::_files_dropped(PackedStringArray p_files) {
	// TODO: Support installing multiple ZIPs at the same time?
	if (p_files.size() == 1 && p_files[0].ends_with(".zip")) {
		const String &file = p_files[0];
		_install_project(file, file.get_file().get_basename().capitalize());
		return;
	}

	HashSet<String> folders_set;
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	for (int i = 0; i < p_files.size(); i++) {
		const String &file = p_files[i];
		folders_set.insert(da->dir_exists(file) ? file : file.get_base_dir());
	}
	ERR_FAIL_COND(folders_set.size() == 0); // This can't really happen, we consume every dropped file path above.

	PackedStringArray folders;
	for (const String &E : folders_set) {
		folders.push_back(E);
	}
	_project_list->find_projects_multiple(folders);
}

void ProjectManager::_on_order_option_changed(int p_idx) {
	if (is_inside_tree()) {
		_project_list->set_order_option(p_idx);
	}
}

void ProjectManager::_on_tab_changed(int p_tab) {
#ifndef ANDROID_ENABLED
	if (p_tab == 0) { // Projects
		// Automatically grab focus when the user moves from the Templates tab
		// back to the Projects tab.
		search_box->grab_focus();
	}

	// The Templates tab's search field is focused on display in the asset
	// library editor plugin code.
#endif
}

void ProjectManager::_on_search_term_changed(const String &p_term) {
	_project_list->set_search_term(p_term);
	_project_list->sort_projects();

	// Select the first visible project in the list.
	// This makes it possible to open a project without ever touching the mouse,
	// as the search field is automatically focused on startup.
	_project_list->select_first_visible_project();
	_update_project_buttons();
}

void ProjectManager::_on_search_term_submitted(const String &p_text) {
	if (tabs->get_current_tab() != 0) {
		return;
	}

	_open_selected_projects_ask();
}

void ProjectManager::_open_asset_library() {
	asset_library->disable_community_support();
	tabs->set_current_tab(1);
}

void ProjectManager::_version_button_pressed() {
	DisplayServer::get_singleton()->clipboard_set(version_btn->get_text());
}

LineEdit *ProjectManager::get_search_box() {
	return search_box;
}

void ProjectManager::add_new_tag(const String &p_tag) {
	if (!tag_set.has(p_tag)) {
		tag_set.insert(p_tag);
		ProjectTag *tag_control = memnew(ProjectTag(p_tag));
		all_tags->add_child(tag_control);
		all_tags->move_child(tag_control, -2);
		tag_control->connect_button_to(callable_mp(this, &ProjectManager::_add_project_tag).bind(p_tag));
	}
}

void ProjectList::add_search_tag(const String &p_tag) {
	const String tag_string = "tag:" + p_tag;

	int exists = _search_term.find(tag_string);
	if (exists > -1) {
		_search_term = _search_term.erase(exists, tag_string.length() + 1);
	} else if (_search_term.is_empty() || _search_term.ends_with(" ")) {
		_search_term += tag_string;
	} else {
		_search_term += " " + tag_string;
	}
	ProjectManager::get_singleton()->get_search_box()->set_text(_search_term);

	sort_projects();
}

ProjectManager::ProjectManager() {
	singleton = this;

	// load settings
	if (!EditorSettings::get_singleton()) {
		EditorSettings::create();
	}

	// Turn off some servers we aren't going to be using in the Project Manager.
	NavigationServer3D::get_singleton()->set_active(false);
	PhysicsServer3D::get_singleton()->set_active(false);
	PhysicsServer2D::get_singleton()->set_active(false);

	EditorSettings::get_singleton()->set_optimize_save(false); //just write settings as they came

	{
		int display_scale = EDITOR_GET("interface/editor/display_scale");

		switch (display_scale) {
			case 0:
				// Try applying a suitable display scale automatically.
				EditorScale::set_scale(EditorSettings::get_singleton()->get_auto_display_scale());
				break;
			case 1:
				EditorScale::set_scale(0.75);
				break;
			case 2:
				EditorScale::set_scale(1.0);
				break;
			case 3:
				EditorScale::set_scale(1.25);
				break;
			case 4:
				EditorScale::set_scale(1.5);
				break;
			case 5:
				EditorScale::set_scale(1.75);
				break;
			case 6:
				EditorScale::set_scale(2.0);
				break;
			default:
				EditorScale::set_scale(EDITOR_GET("interface/editor/custom_display_scale"));
				break;
		}
		EditorFileDialog::get_icon_func = &ProjectManager::_file_dialog_get_icon;
		EditorFileDialog::get_thumbnail_func = &ProjectManager::_file_dialog_get_thumbnail;
	}

	// TRANSLATORS: This refers to the application where users manage their Godot projects.
	DisplayServer::get_singleton()->window_set_title(VERSION_NAME + String(" - ") + TTR("Project Manager", "Application"));

	EditorFileDialog::set_default_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));
	EditorFileDialog::set_default_display_mode((EditorFileDialog::DisplayMode)EDITOR_GET("filesystem/file_dialog/display_mode").operator int());

	int swap_cancel_ok = EDITOR_GET("interface/editor/accept_dialog_cancel_ok_buttons");
	if (swap_cancel_ok != 0) { // 0 is auto, set in register_scene based on DisplayServer.
		// Swap on means OK first.
		AcceptDialog::set_swap_cancel_ok(swap_cancel_ok == 2);
	}

	EditorColorMap::create();
	EditorTheme::initialize();
	Ref<Theme> theme = create_custom_theme();
	DisplayServer::set_early_window_clear_color_override(true, theme->get_color(SNAME("background"), EditorStringName(Editor)));

	set_theme(theme);
	set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	background_panel = memnew(Panel);
	add_child(background_panel);
	background_panel->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	VBoxContainer *vb = memnew(VBoxContainer);
	background_panel->add_child(vb);
	vb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 8 * EDSCALE);

	Control *center_box = memnew(Control);
	center_box->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(center_box);

	tabs = memnew(TabContainer);
	tabs->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	center_box->add_child(tabs);
	tabs->connect("tab_changed", callable_mp(this, &ProjectManager::_on_tab_changed));

	local_projects_vb = memnew(VBoxContainer);
	local_projects_vb->set_name(TTR("Local Projects"));
	tabs->add_child(local_projects_vb);

	{
		// A bar at top with buttons and options.
		HBoxContainer *hb = memnew(HBoxContainer);
		hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		local_projects_vb->add_child(hb);

		create_btn = memnew(Button);
		create_btn->set_text(TTR("New"));
		create_btn->set_shortcut(ED_SHORTCUT("project_manager/new_project", TTR("New Project"), KeyModifierMask::CMD_OR_CTRL | Key::N));
		create_btn->connect("pressed", callable_mp(this, &ProjectManager::_new_project));
		hb->add_child(create_btn);

		import_btn = memnew(Button);
		import_btn->set_text(TTR("Import"));
		import_btn->set_shortcut(ED_SHORTCUT("project_manager/import_project", TTR("Import Project"), KeyModifierMask::CMD_OR_CTRL | Key::I));
		import_btn->connect("pressed", callable_mp(this, &ProjectManager::_import_project));
		hb->add_child(import_btn);

		scan_btn = memnew(Button);
		scan_btn->set_text(TTR("Scan"));
		scan_btn->set_shortcut(ED_SHORTCUT("project_manager/scan_projects", TTR("Scan Projects"), KeyModifierMask::CMD_OR_CTRL | Key::S));
		scan_btn->connect("pressed", callable_mp(this, &ProjectManager::_scan_projects));
		hb->add_child(scan_btn);

		loading_label = memnew(Label(TTR("Loading, please wait...")));
		loading_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hb->add_child(loading_label);
		// The loading label is shown later.
		loading_label->hide();

		search_box = memnew(LineEdit);
		search_box->set_placeholder(TTR("Filter Projects"));
		search_box->set_tooltip_text(TTR("This field filters projects by name and last path component.\nTo filter projects by name and full path, the query must contain at least one `/` character."));
		search_box->set_clear_button_enabled(true);
		search_box->connect("text_changed", callable_mp(this, &ProjectManager::_on_search_term_changed));
		search_box->connect("text_submitted", callable_mp(this, &ProjectManager::_on_search_term_submitted));
		search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hb->add_child(search_box);

		Label *sort_label = memnew(Label);
		sort_label->set_text(TTR("Sort:"));
		hb->add_child(sort_label);

		filter_option = memnew(OptionButton);
		filter_option->set_clip_text(true);
		filter_option->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		filter_option->set_stretch_ratio(0.3);
		filter_option->connect("item_selected", callable_mp(this, &ProjectManager::_on_order_option_changed));
		hb->add_child(filter_option);

		Vector<String> sort_filter_titles;
		sort_filter_titles.push_back(TTR("Last Edited"));
		sort_filter_titles.push_back(TTR("Name"));
		sort_filter_titles.push_back(TTR("Path"));
		sort_filter_titles.push_back(TTR("Tags"));

		for (int i = 0; i < sort_filter_titles.size(); i++) {
			filter_option->add_item(sort_filter_titles[i]);
		}
	}

	{
		// A container for the project list and for the side bar with buttons.
		HBoxContainer *search_tree_hb = memnew(HBoxContainer);
		local_projects_vb->add_child(search_tree_hb);
		search_tree_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

		search_panel = memnew(PanelContainer);
		search_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_tree_hb->add_child(search_panel);

		_project_list = memnew(ProjectList);
		_project_list->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		search_panel->add_child(_project_list);
		_project_list->connect(ProjectList::SIGNAL_LIST_CHANGED, callable_mp(this, &ProjectManager::_update_project_buttons));
		_project_list->connect(ProjectList::SIGNAL_SELECTION_CHANGED, callable_mp(this, &ProjectManager::_update_project_buttons));
		_project_list->connect(ProjectList::SIGNAL_PROJECT_ASK_OPEN, callable_mp(this, &ProjectManager::_open_selected_projects_ask));

		// The side bar with the edit, run, rename, etc. buttons.
		VBoxContainer *tree_vb = memnew(VBoxContainer);
		tree_vb->set_custom_minimum_size(Size2(120, 120));
		search_tree_hb->add_child(tree_vb);

		tree_vb->add_child(memnew(HSeparator));

		open_btn = memnew(Button);
		open_btn->set_text(TTR("Edit"));
		open_btn->set_shortcut(ED_SHORTCUT("project_manager/edit_project", TTR("Edit Project"), KeyModifierMask::CMD_OR_CTRL | Key::E));
		open_btn->connect("pressed", callable_mp(this, &ProjectManager::_open_selected_projects_ask));
		tree_vb->add_child(open_btn);

		run_btn = memnew(Button);
		run_btn->set_text(TTR("Run"));
		run_btn->set_shortcut(ED_SHORTCUT("project_manager/run_project", TTR("Run Project"), KeyModifierMask::CMD_OR_CTRL | Key::R));
		run_btn->connect("pressed", callable_mp(this, &ProjectManager::_run_project));
		tree_vb->add_child(run_btn);

		rename_btn = memnew(Button);
		rename_btn->set_text(TTR("Rename"));
		// The F2 shortcut isn't overridden with Enter on macOS as Enter is already used to edit a project.
		rename_btn->set_shortcut(ED_SHORTCUT("project_manager/rename_project", TTR("Rename Project"), Key::F2));
		rename_btn->connect("pressed", callable_mp(this, &ProjectManager::_rename_project));
		tree_vb->add_child(rename_btn);

		manage_tags_btn = memnew(Button);
		manage_tags_btn->set_text(TTR("Manage Tags"));
		tree_vb->add_child(manage_tags_btn);

		erase_btn = memnew(Button);
		erase_btn->set_text(TTR("Remove"));
		erase_btn->set_shortcut(ED_SHORTCUT("project_manager/remove_project", TTR("Remove Project"), Key::KEY_DELETE));
		erase_btn->connect("pressed", callable_mp(this, &ProjectManager::_erase_project));
		tree_vb->add_child(erase_btn);

		erase_missing_btn = memnew(Button);
		erase_missing_btn->set_text(TTR("Remove Missing"));
		erase_missing_btn->connect("pressed", callable_mp(this, &ProjectManager::_erase_missing_projects));
		tree_vb->add_child(erase_missing_btn);

		tree_vb->add_spacer();

		about_btn = memnew(Button);
		about_btn->set_text(TTR("About"));
		about_btn->connect("pressed", callable_mp(this, &ProjectManager::_show_about));
		tree_vb->add_child(about_btn);
	}

	{
		// Version info and language options
		settings_hb = memnew(HBoxContainer);
		settings_hb->set_alignment(BoxContainer::ALIGNMENT_END);
		settings_hb->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);
		settings_hb->set_anchors_and_offsets_preset(Control::PRESET_TOP_RIGHT);

		// A VBoxContainer that contains a dummy Control node to adjust the LinkButton's vertical position.
		VBoxContainer *spacer_vb = memnew(VBoxContainer);
		settings_hb->add_child(spacer_vb);

		Control *v_spacer = memnew(Control);
		spacer_vb->add_child(v_spacer);

		version_btn = memnew(LinkButton);
		String hash = String(VERSION_HASH);
		if (hash.length() != 0) {
			hash = " " + vformat("[%s]", hash.left(9));
		}
		version_btn->set_text("v" VERSION_FULL_BUILD + hash);
		// Fade the version label to be less prominent, but still readable.
		version_btn->set_self_modulate(Color(1, 1, 1, 0.6));
		version_btn->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
		version_btn->set_tooltip_text(TTR("Click to copy."));
		version_btn->connect("pressed", callable_mp(this, &ProjectManager::_version_button_pressed));
		spacer_vb->add_child(version_btn);

		// Add a small horizontal spacer between the version and language buttons
		// to distinguish them.
		Control *h_spacer = memnew(Control);
		settings_hb->add_child(h_spacer);

		language_btn = memnew(OptionButton);
		language_btn->set_focus_mode(Control::FOCUS_NONE);
		language_btn->set_fit_to_longest_item(false);
		language_btn->set_flat(true);
		language_btn->connect("item_selected", callable_mp(this, &ProjectManager::_language_selected));
#ifdef ANDROID_ENABLED
		// The language selection dropdown doesn't work on Android (as the setting isn't saved), see GH-60353.
		// Also, the dropdown it spawns is very tall and can't be scrolled without a hardware mouse.
		// Hiding the language selection dropdown also leaves more space for the version label to display.
		language_btn->hide();
#endif

		Vector<String> editor_languages;
		List<PropertyInfo> editor_settings_properties;
		EditorSettings::get_singleton()->get_property_list(&editor_settings_properties);
		for (const PropertyInfo &pi : editor_settings_properties) {
			if (pi.name == "interface/editor/editor_language") {
				editor_languages = pi.hint_string.split(",");
				break;
			}
		}

		String current_lang = EDITOR_GET("interface/editor/editor_language");
		language_btn->set_text(current_lang);

		for (int i = 0; i < editor_languages.size(); i++) {
			String lang = editor_languages[i];
			String lang_name = TranslationServer::get_singleton()->get_locale_name(lang);
			language_btn->add_item(vformat("[%s] %s", lang, lang_name), i);
			language_btn->set_item_metadata(i, lang);
			if (current_lang == lang) {
				language_btn->select(i);
			}
		}

		settings_hb->add_child(language_btn);
		center_box->add_child(settings_hb);
	}

	if (AssetLibraryEditorPlugin::is_available()) {
		asset_library = memnew(EditorAssetLibrary(true));
		asset_library->set_name(TTR("Asset Library Projects"));
		tabs->add_child(asset_library);
		asset_library->connect("install_asset", callable_mp(this, &ProjectManager::_install_project));
	} else {
		print_verbose("Asset Library not available (due to using Web editor, or SSL support disabled).");
	}

	{
		// Dialogs
		language_restart_ask = memnew(ConfirmationDialog);
		language_restart_ask->set_ok_button_text(TTR("Restart Now"));
		language_restart_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_restart_confirm));
		language_restart_ask->set_cancel_button_text(TTR("Continue"));
		add_child(language_restart_ask);

		scan_dir = memnew(EditorFileDialog);
		scan_dir->set_previews_enabled(false);
		scan_dir->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		scan_dir->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
		scan_dir->set_title(TTR("Select a Folder to Scan")); // must be after mode or it's overridden
		scan_dir->set_current_dir(EDITOR_GET("filesystem/directories/default_project_path"));
		add_child(scan_dir);
		scan_dir->connect("dir_selected", callable_mp(_project_list, &ProjectList::find_projects));

		erase_missing_ask = memnew(ConfirmationDialog);
		erase_missing_ask->set_ok_button_text(TTR("Remove All"));
		erase_missing_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_erase_missing_projects_confirm));
		add_child(erase_missing_ask);

		erase_ask = memnew(ConfirmationDialog);
		erase_ask->set_ok_button_text(TTR("Remove"));
		erase_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_erase_project_confirm));
		add_child(erase_ask);

		VBoxContainer *erase_ask_vb = memnew(VBoxContainer);
		erase_ask->add_child(erase_ask_vb);

		erase_ask_label = memnew(Label);
		erase_ask_vb->add_child(erase_ask_label);

		// Comment out for now until we have a better warning system to
		// ensure users delete their project only.
		//delete_project_contents = memnew(CheckBox);
		//delete_project_contents->set_text(TTR("Also delete project contents (no undo!)"));
		//erase_ask_vb->add_child(delete_project_contents);

		multi_open_ask = memnew(ConfirmationDialog);
		multi_open_ask->set_ok_button_text(TTR("Edit"));
		multi_open_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_open_selected_projects));
		add_child(multi_open_ask);

		multi_run_ask = memnew(ConfirmationDialog);
		multi_run_ask->set_ok_button_text(TTR("Run"));
		multi_run_ask->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_run_project_confirm));
		add_child(multi_run_ask);

		ask_update_settings = memnew(ConfirmationDialog);
		ask_update_settings->set_autowrap(true);
		ask_update_settings->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_confirm_update_settings));
		full_convert_button = ask_update_settings->add_button(TTR("Convert Full Project"), !GLOBAL_GET("gui/common/swap_cancel_ok"));
		full_convert_button->connect("pressed", callable_mp(this, &ProjectManager::_full_convert_button_pressed));
		add_child(ask_update_settings);

		ask_full_convert_dialog = memnew(ConfirmationDialog);
		ask_full_convert_dialog->set_autowrap(true);
		ask_full_convert_dialog->set_text(TTR("This option will perform full project conversion, updating scenes, resources and scripts from Godot 3 to work in Godot 4.\n\nNote that this is a best-effort conversion, i.e. it makes upgrading the project easier, but it will not open out-of-the-box and will still require manual adjustments.\n\nIMPORTANT: Make sure to backup your project before converting, as this operation makes it impossible to open it in older versions of Godot."));
		ask_full_convert_dialog->connect("confirmed", callable_mp(this, &ProjectManager::_perform_full_project_conversion));
		add_child(ask_full_convert_dialog);

		npdialog = memnew(ProjectDialog);
		npdialog->connect("projects_updated", callable_mp(this, &ProjectManager::_on_projects_updated));
		npdialog->connect("project_created", callable_mp(this, &ProjectManager::_on_project_created));
		add_child(npdialog);

		run_error_diag = memnew(AcceptDialog);
		run_error_diag->set_title(TTR("Can't run project"));
		add_child(run_error_diag);

		dialog_error = memnew(AcceptDialog);
		add_child(dialog_error);

		if (asset_library) {
			open_templates = memnew(ConfirmationDialog);
			open_templates->set_text(TTR("You currently don't have any projects.\nWould you like to explore official example projects in the Asset Library?"));
			open_templates->set_ok_button_text(TTR("Open Asset Library"));
			open_templates->connect("confirmed", callable_mp(this, &ProjectManager::_open_asset_library));
			add_child(open_templates);
		}

		about = memnew(EditorAbout);
		add_child(about);

		_build_icon_type_cache(get_theme());
	}

	{
		// Tag management.
		tag_manage_dialog = memnew(ConfirmationDialog);
		add_child(tag_manage_dialog);
		tag_manage_dialog->set_title(TTR("Manage Project Tags"));
		tag_manage_dialog->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_apply_project_tags));
		manage_tags_btn->connect("pressed", callable_mp(this, &ProjectManager::_manage_project_tags));

		VBoxContainer *tag_vb = memnew(VBoxContainer);
		tag_manage_dialog->add_child(tag_vb);

		Label *label = memnew(Label(TTR("Project Tags")));
		tag_vb->add_child(label);
		label->set_theme_type_variation("HeaderMedium");
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		label = memnew(Label(TTR("Click tag to remove it from the project.")));
		tag_vb->add_child(label);
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		project_tags = memnew(HFlowContainer);
		tag_vb->add_child(project_tags);
		project_tags->set_custom_minimum_size(Vector2(0, 100) * EDSCALE);

		tag_vb->add_child(memnew(HSeparator));

		label = memnew(Label(TTR("All Tags")));
		tag_vb->add_child(label);
		label->set_theme_type_variation("HeaderMedium");
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		label = memnew(Label(TTR("Click tag to add it to the project.")));
		tag_vb->add_child(label);
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		all_tags = memnew(HFlowContainer);
		tag_vb->add_child(all_tags);
		all_tags->set_custom_minimum_size(Vector2(0, 100) * EDSCALE);

		tag_edit_error = memnew(Label);
		tag_vb->add_child(tag_edit_error);
		tag_edit_error->set_autowrap_mode(TextServer::AUTOWRAP_WORD);

		create_tag_dialog = memnew(ConfirmationDialog);
		tag_manage_dialog->add_child(create_tag_dialog);
		create_tag_dialog->set_title(TTR("Create New Tag"));
		create_tag_dialog->get_ok_button()->connect("pressed", callable_mp(this, &ProjectManager::_create_new_tag));

		tag_vb = memnew(VBoxContainer);
		create_tag_dialog->add_child(tag_vb);

		Label *info = memnew(Label(TTR("Tags are capitalized automatically when displayed.")));
		tag_vb->add_child(info);

		new_tag_name = memnew(LineEdit);
		tag_vb->add_child(new_tag_name);
		new_tag_name->connect("text_changed", callable_mp(this, &ProjectManager::_set_new_tag_name));
		new_tag_name->connect("text_submitted", callable_mp(this, &ProjectManager::_create_new_tag).unbind(1));
		create_tag_dialog->connect("about_to_popup", callable_mp(new_tag_name, &LineEdit::clear));
		create_tag_dialog->connect("about_to_popup", callable_mp((Control *)new_tag_name, &Control::grab_focus), CONNECT_DEFERRED);

		tag_error = memnew(Label);
		tag_vb->add_child(tag_error);

		create_tag_btn = memnew(Button);
		all_tags->add_child(create_tag_btn);
		create_tag_btn->connect("pressed", callable_mp((Window *)create_tag_dialog, &Window::popup_centered).bind(Vector2i(500, 0) * EDSCALE));
	}

	// Initialize project list.
	{
		Ref<DirAccess> dir_access = DirAccess::create(DirAccess::AccessType::ACCESS_FILESYSTEM);

		String default_project_path = EDITOR_GET("filesystem/directories/default_project_path");
		if (!default_project_path.is_empty() && !dir_access->dir_exists(default_project_path)) {
			Error error = dir_access->make_dir_recursive(default_project_path);
			if (error != OK) {
				ERR_PRINT("Could not create default project directory at: " + default_project_path);
			}
		}

		bool scanned_for_projects = false; // Scanning will update the list automatically.

		String autoscan_path = EDITOR_GET("filesystem/directories/autoscan_project_path");
		if (!autoscan_path.is_empty()) {
			if (dir_access->dir_exists(autoscan_path)) {
				_project_list->find_projects(autoscan_path);
				scanned_for_projects = true;
			} else {
				Error error = dir_access->make_dir_recursive(autoscan_path);
				if (error != OK) {
					ERR_PRINT("Could not create project autoscan directory at: " + autoscan_path);
				}
			}
		}

		if (!scanned_for_projects) {
			_project_list->update_project_list();
		}
	}

	SceneTree::get_singleton()->get_root()->connect("files_dropped", callable_mp(this, &ProjectManager::_files_dropped));

	OS::get_singleton()->set_low_processor_usage_mode(true);

	_update_size_limits();
}

ProjectManager::~ProjectManager() {
	singleton = nullptr;
	if (EditorSettings::get_singleton()) {
		EditorSettings::destroy();
	}

	EditorColorMap::finish();
	EditorTheme::finalize();
}

void ProjectTag::_notification(int p_what) {
	if (display_close && p_what == NOTIFICATION_THEME_CHANGED) {
		button->set_icon(get_theme_icon(SNAME("close"), SNAME("TabBar")));
	}
}

ProjectTag::ProjectTag(const String &p_text, bool p_display_close) {
	add_theme_constant_override(SNAME("separation"), 0);
	set_v_size_flags(SIZE_SHRINK_CENTER);
	tag_string = p_text;
	display_close = p_display_close;

	Color tag_color = Color(1, 0, 0);
	tag_color.set_ok_hsl_s(0.8);
	tag_color.set_ok_hsl_h(float(p_text.hash() * 10001 % UINT32_MAX) / float(UINT32_MAX));
	set_self_modulate(tag_color);

	ColorRect *cr = memnew(ColorRect);
	add_child(cr);
	cr->set_custom_minimum_size(Vector2(4, 0) * EDSCALE);
	cr->set_color(tag_color);

	button = memnew(Button);
	add_child(button);
	button->set_auto_translate(false);
	button->set_text(p_text.capitalize());
	button->set_focus_mode(FOCUS_NONE);
	button->set_icon_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
	button->set_theme_type_variation(SNAME("ProjectTag"));
}

void ProjectTag::connect_button_to(const Callable &p_callable) {
	button->connect(SNAME("pressed"), p_callable, CONNECT_DEFERRED);
}

const String ProjectTag::get_tag() const {
	return tag_string;
}
