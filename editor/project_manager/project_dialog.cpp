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

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/zip_io.h"
#include "core/version.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_vcs_interface.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/themes/editor_icons.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/check_box.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"

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

static bool is_zip_file(Ref<DirAccess> p_d, const String &p_path) {
	return p_path.ends_with(".zip") && p_d->file_exists(p_path);
}

String ProjectDialog::_test_path() {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	const String base_path = project_path->get_text();
	String valid_path, valid_install_path;
	bool is_zip = false;
	if (d->change_dir(base_path) == OK) {
		valid_path = base_path;
	} else if (is_zip_file(d, base_path)) {
		valid_path = base_path;
		is_zip = true;
	} else if (d->change_dir(base_path.strip_edges()) == OK) {
		valid_path = base_path.strip_edges();
	} else if (is_zip_file(d, base_path.strip_edges())) {
		valid_path = base_path.strip_edges();
		is_zip = true;
	}

	if (valid_path.is_empty()) {
		_set_message(TTR("The path specified doesn't exist."), MESSAGE_ERROR);
		get_ok_button()->set_disabled(true);
		return "";
	}

	if (mode == MODE_IMPORT && is_zip) {
		if (d->change_dir(install_path->get_text()) == OK) {
			valid_install_path = install_path->get_text();
		} else if (d->change_dir(install_path->get_text().strip_edges()) == OK) {
			valid_install_path = install_path->get_text().strip_edges();
		}

		if (valid_install_path.is_empty()) {
			_set_message(TTR("The install path specified doesn't exist."), MESSAGE_ERROR, INSTALL_PATH);
			get_ok_button()->set_disabled(true);
			return "";
		}
	}

	if (mode == MODE_IMPORT || mode == MODE_RENAME) {
		if (!d->file_exists("project.godot")) {
			if (is_zip) {
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
					_set_message(TTR("Please choose an empty install folder."), MESSAGE_WARNING, INSTALL_PATH);
					get_ok_button()->set_disabled(true);
					return "";
				}

			} else {
				_set_message(TTR("Please choose a \"project.godot\", a directory with it, or a \".zip\" file."), MESSAGE_ERROR);
				install_path_container->hide();
				get_ok_button()->set_disabled(true);
				return "";
			}

		} else if (is_zip) {
			_set_message(TTR("The install directory already contains a Godot project."), MESSAGE_ERROR, INSTALL_PATH);
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

void ProjectDialog::_update_path(const String &p_path) {
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

void ProjectDialog::_path_text_changed(const String &p_path) {
	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (mode == MODE_IMPORT && is_zip_file(d, p_path)) {
		install_path->set_text(p_path.get_base_dir());
		install_path_container->show();
	} else if (mode == MODE_IMPORT && is_zip_file(d, p_path.strip_edges())) {
		install_path->set_text(p_path.strip_edges().get_base_dir());
		install_path_container->show();
	} else {
		install_path_container->hide();
	}

	_update_path(p_path.simplify_path());
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
	_update_path(sp);
	if (p.ends_with(".zip")) {
		callable_mp((Control *)install_path, &Control::grab_focus).call_deferred();
	} else {
		callable_mp((Control *)get_ok_button(), &Control::grab_focus).call_deferred();
	}
}

void ProjectDialog::_path_selected(const String &p_path) {
	// If not already shown.
	show_dialog();

	String sp = p_path.simplify_path();
	project_path->set_text(sp);
	_update_path(sp);
	callable_mp((Control *)get_ok_button(), &Control::grab_focus).call_deferred();
}

void ProjectDialog::_install_path_selected(const String &p_path) {
	String sp = p_path.simplify_path();
	install_path->set_text(sp);
	_update_path(sp);
	callable_mp((Control *)get_ok_button(), &Control::grab_focus).call_deferred();
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
				_update_path(dir_str);
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
	_update_path("");
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

		callable_mp((Control *)project_name, &Control::grab_focus).call_deferred();

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
			callable_mp((Control *)project_name, &Control::grab_focus).call_deferred();
			callable_mp(project_name, &LineEdit::select_all).call_deferred();

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
		case NOTIFICATION_THEME_CHANGED: {
			create_dir->set_icon(get_editor_theme_icon(SNAME("FolderCreate")));
		} break;

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
	install_path->connect("text_changed", callable_mp(this, &ProjectDialog::_update_path));
	fdialog->connect("dir_selected", callable_mp(this, &ProjectDialog::_path_selected));
	fdialog->connect("file_selected", callable_mp(this, &ProjectDialog::_file_selected));
	fdialog_install->connect("dir_selected", callable_mp(this, &ProjectDialog::_install_path_selected));
	fdialog_install->connect("file_selected", callable_mp(this, &ProjectDialog::_install_path_selected));

	set_hide_on_ok(false);

	dialog_error = memnew(AcceptDialog);
	add_child(dialog_error);
}
