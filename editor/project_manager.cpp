/*************************************************************************/
/*  project_manager.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "project_manager.h"

#include "core/io/config_file.h"
#include "core/io/resource_saver.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/translation.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "editor_themes.h"
#include "scene/gui/center_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"

// Used to test for GLES3 support.
#ifndef SERVER_ENABLED
#include "drivers/gles3/rasterizer_gles3.h"
#endif

static inline String get_project_key_from_path(const String &dir) {
	return dir.replace("/", "::");
}

class ProjectDialog : public ConfirmationDialog {

	GDCLASS(ProjectDialog, ConfirmationDialog);

public:
	enum Mode {
		MODE_NEW,
		MODE_IMPORT,
		MODE_INSTALL,
		MODE_RENAME
	};

private:
	enum MessageType {
		MESSAGE_ERROR,
		MESSAGE_WARNING,
		MESSAGE_SUCCESS
	};

	enum InputType {
		PROJECT_PATH,
		INSTALL_PATH
	};

	Mode mode;
	Button *browse;
	Button *install_browse;
	Button *create_dir;
	Container *name_container;
	Container *path_container;
	Container *install_path_container;
	Container *rasterizer_container;
	Ref<ButtonGroup> rasterizer_button_group;
	Label *msg;
	LineEdit *project_path;
	LineEdit *project_name;
	LineEdit *install_path;
	TextureRect *status_rect;
	TextureRect *install_status_rect;
	FileDialog *fdialog;
	FileDialog *fdialog_install;
	String zip_path;
	String zip_title;
	AcceptDialog *dialog_error;
	String fav_dir;

	String created_folder_path;

	void set_message(const String &p_msg, MessageType p_type = MESSAGE_SUCCESS, InputType input_type = PROJECT_PATH) {

		msg->set_text(p_msg);
		Ref<Texture> current_path_icon = status_rect->get_texture();
		Ref<Texture> current_install_icon = install_status_rect->get_texture();
		Ref<Texture> new_icon;

		switch (p_type) {

			case MESSAGE_ERROR: {

				msg->add_color_override("font_color", get_color("error_color", "Editor"));
				msg->set_modulate(Color(1, 1, 1, 1));
				new_icon = get_icon("StatusError", "EditorIcons");

			} break;
			case MESSAGE_WARNING: {

				msg->add_color_override("font_color", get_color("warning_color", "Editor"));
				msg->set_modulate(Color(1, 1, 1, 1));
				new_icon = get_icon("StatusWarning", "EditorIcons");

			} break;
			case MESSAGE_SUCCESS: {

				msg->set_modulate(Color(1, 1, 1, 0));
				new_icon = get_icon("StatusSuccess", "EditorIcons");

			} break;
		}

		if (current_path_icon != new_icon && input_type == PROJECT_PATH) {
			status_rect->set_texture(new_icon);
		} else if (current_install_icon != new_icon && input_type == INSTALL_PATH) {
			install_status_rect->set_texture(new_icon);
		}

		set_size(Size2(500, 0) * EDSCALE);
	}

	String _test_path() {

		DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
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

		if (valid_path == "") {
			set_message(TTR("The path specified doesn't exist."), MESSAGE_ERROR);
			memdelete(d);
			get_ok()->set_disabled(true);
			return "";
		}

		if (mode == MODE_IMPORT && valid_path.ends_with(".zip")) {
			if (d->change_dir(install_path->get_text()) == OK) {
				valid_install_path = install_path->get_text();
			} else if (d->change_dir(install_path->get_text().strip_edges()) == OK) {
				valid_install_path = install_path->get_text().strip_edges();
			}

			if (valid_install_path == "") {
				set_message(TTR("The path specified doesn't exist."), MESSAGE_ERROR, INSTALL_PATH);
				memdelete(d);
				get_ok()->set_disabled(true);
				return "";
			}
		}

		if (mode == MODE_IMPORT || mode == MODE_RENAME) {

			if (valid_path != "" && !d->file_exists("project.godot")) {

				if (valid_path.ends_with(".zip")) {
					FileAccess *src_f = NULL;
					zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

					unzFile pkg = unzOpen2(valid_path.utf8().get_data(), &io);
					if (!pkg) {

						set_message(TTR("Error opening package file (it's not in ZIP format)."), MESSAGE_ERROR);
						memdelete(d);
						get_ok()->set_disabled(true);
						unzClose(pkg);
						return "";
					}

					int ret = unzGoToFirstFile(pkg);
					while (ret == UNZ_OK) {
						unz_file_info info;
						char fname[16384];
						ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

						if (String(fname).ends_with("project.godot")) {
							break;
						}

						ret = unzGoToNextFile(pkg);
					}

					if (ret == UNZ_END_OF_LIST_OF_FILE) {
						set_message(TTR("Invalid \".zip\" project file; it doesn't contain a \"project.godot\" file."), MESSAGE_ERROR);
						memdelete(d);
						get_ok()->set_disabled(true);
						unzClose(pkg);
						return "";
					}

					unzClose(pkg);

					// check if the specified install folder is empty, even though this is not an error, it is good to check here
					d->list_dir_begin();
					bool is_empty = true;
					String n = d->get_next();
					while (n != String()) {
						if (!n.begins_with(".")) {
							// Allow `.`, `..` (reserved current/parent folder names)
							// and hidden files/folders to be present.
							// For instance, this lets users initialize a Git repository
							// and still be able to create a project in the directory afterwards.
							is_empty = false;
							break;
						}
						n = d->get_next();
					}
					d->list_dir_end();

					if (!is_empty) {

						set_message(TTR("Please choose an empty folder."), MESSAGE_WARNING, INSTALL_PATH);
						memdelete(d);
						get_ok()->set_disabled(true);
						return "";
					}

				} else {
					set_message(TTR("Please choose a \"project.godot\" or \".zip\" file."), MESSAGE_ERROR);
					memdelete(d);
					install_path_container->hide();
					get_ok()->set_disabled(true);
					return "";
				}

			} else if (valid_path.ends_with("zip")) {

				set_message(TTR("This directory already contains a Godot project."), MESSAGE_ERROR, INSTALL_PATH);
				memdelete(d);
				get_ok()->set_disabled(true);
				return "";
			}

		} else {

			// check if the specified folder is empty, even though this is not an error, it is good to check here
			d->list_dir_begin();
			bool is_empty = true;
			String n = d->get_next();
			while (n != String()) {
				if (!n.begins_with(".")) {
					// Allow `.`, `..` (reserved current/parent folder names)
					// and hidden files/folders to be present.
					// For instance, this lets users initialize a Git repository
					// and still be able to create a project in the directory afterwards.
					is_empty = false;
					break;
				}
				n = d->get_next();
			}
			d->list_dir_end();

			if (!is_empty) {

				set_message(TTR("Please choose an empty folder."), MESSAGE_ERROR);
				memdelete(d);
				get_ok()->set_disabled(true);
				return "";
			}
		}

		set_message("");
		set_message("", MESSAGE_SUCCESS, INSTALL_PATH);
		memdelete(d);
		get_ok()->set_disabled(false);
		return valid_path;
	}

	void _path_text_changed(const String &p_path) {

		String sp = _test_path();
		if (sp != "") {

			// If the project name is empty or default, infer the project name from the selected folder name
			if (project_name->get_text() == "" || project_name->get_text() == TTR("New Game Project")) {
				sp = sp.replace("\\", "/");
				int lidx = sp.find_last("/");

				if (lidx != -1) {
					sp = sp.substr(lidx + 1, sp.length()).capitalize();
				}
				if (sp == "" && mode == MODE_IMPORT)
					sp = TTR("Imported Project");

				project_name->set_text(sp);
				_text_changed(sp);
			}
		}

		if (created_folder_path != "" && created_folder_path != p_path) {
			_remove_created_folder();
		}
	}

	void _file_selected(const String &p_path) {

		String p = p_path;
		if (mode == MODE_IMPORT) {
			if (p.ends_with("project.godot")) {
				p = p.get_base_dir();
				install_path_container->hide();
				get_ok()->set_disabled(false);
			} else if (p.ends_with(".zip")) {
				install_path->set_text(p.get_base_dir());
				install_path_container->show();
				get_ok()->set_disabled(false);
			} else {
				set_message(TTR("Please choose a \"project.godot\" or \".zip\" file."), MESSAGE_ERROR);
				get_ok()->set_disabled(true);
				return;
			}
		}
		String sp = p.simplify_path();
		project_path->set_text(sp);
		_path_text_changed(sp);
		if (p.ends_with(".zip")) {
			install_path->call_deferred("grab_focus");
		} else {
			get_ok()->call_deferred("grab_focus");
		}
	}

	void _path_selected(const String &p_path) {

		String sp = p_path.simplify_path();
		project_path->set_text(sp);
		_path_text_changed(sp);
		get_ok()->call_deferred("grab_focus");
	}

	void _install_path_selected(const String &p_path) {

		String sp = p_path.simplify_path();
		install_path->set_text(sp);
		_path_text_changed(sp);
		get_ok()->call_deferred("grab_focus");
	}

	void _browse_path() {

		fdialog->set_current_dir(project_path->get_text());

		if (mode == MODE_IMPORT) {

			fdialog->set_mode(FileDialog::MODE_OPEN_FILE);
			fdialog->clear_filters();
			fdialog->add_filter(vformat("project.godot ; %s %s", VERSION_NAME, TTR("Project")));
			fdialog->add_filter("*.zip ; " + TTR("ZIP File"));
		} else {
			fdialog->set_mode(FileDialog::MODE_OPEN_DIR);
		}
		fdialog->popup_centered_ratio();
	}

	void _browse_install_path() {
		fdialog_install->set_current_dir(install_path->get_text());
		fdialog_install->set_mode(FileDialog::MODE_OPEN_DIR);
		fdialog_install->popup_centered_ratio();
	}

	void _create_folder() {

		if (project_name->get_text() == "" || created_folder_path != "" || project_name->get_text().ends_with(".") || project_name->get_text().ends_with(" ")) {
			set_message(TTR("Invalid Project Name."), MESSAGE_WARNING);
			return;
		}

		DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (d->change_dir(project_path->get_text()) == OK) {

			if (!d->dir_exists(project_name->get_text())) {

				if (d->make_dir(project_name->get_text()) == OK) {

					d->change_dir(project_name->get_text());
					String dir_str = d->get_current_dir();
					project_path->set_text(dir_str);
					_path_text_changed(dir_str);
					created_folder_path = d->get_current_dir();
					create_dir->set_disabled(true);
				} else {

					dialog_error->set_text(TTR("Couldn't create folder."));
					dialog_error->popup_centered_minsize();
				}
			} else {

				dialog_error->set_text(TTR("There is already a folder in this path with the specified name."));
				dialog_error->popup_centered_minsize();
			}
		}

		memdelete(d);
	}

	void _text_changed(const String &p_text) {

		if (mode != MODE_NEW)
			return;

		_test_path();

		if (p_text == "")
			set_message(TTR("It would be a good idea to name your project."), MESSAGE_ERROR);
	}

	void ok_pressed() {

		String dir = project_path->get_text();

		if (mode == MODE_RENAME) {

			String dir2 = _test_path();
			if (dir2 == "") {
				set_message(TTR("Invalid project path (changed anything?)."), MESSAGE_ERROR);
				return;
			}

			ProjectSettings *current = memnew(ProjectSettings);

			int err = current->setup(dir2, "");
			if (err != OK) {
				set_message(vformat(TTR("Couldn't load project.godot in project path (error %d). It may be missing or corrupted."), err), MESSAGE_ERROR);
			} else {
				ProjectSettings::CustomMap edited_settings;
				edited_settings["application/config/name"] = project_name->get_text();

				if (current->save_custom(dir2.plus_file("project.godot"), edited_settings, Vector<String>(), true) != OK) {
					set_message(TTR("Couldn't edit project.godot in project path."), MESSAGE_ERROR);
				}
			}

			hide();
			emit_signal("projects_updated");

		} else {

			if (mode == MODE_IMPORT) {

				if (project_path->get_text().ends_with(".zip")) {

					mode = MODE_INSTALL;
					ok_pressed();

					return;
				}

			} else {
				if (mode == MODE_NEW) {

					ProjectSettings::CustomMap initial_settings;
					if (rasterizer_button_group->get_pressed_button()->get_meta("driver_name") == "GLES3") {
						initial_settings["rendering/quality/driver/driver_name"] = "GLES3";
					} else {
						initial_settings["rendering/quality/driver/driver_name"] = "GLES2";
						initial_settings["rendering/vram_compression/import_etc2"] = false;
						initial_settings["rendering/vram_compression/import_etc"] = true;
					}
					initial_settings["application/config/name"] = project_name->get_text();
					initial_settings["application/config/icon"] = "res://icon.png";
					initial_settings["rendering/environment/default_environment"] = "res://default_env.tres";

					if (ProjectSettings::get_singleton()->save_custom(dir.plus_file("project.godot"), initial_settings, Vector<String>(), false) != OK) {
						set_message(TTR("Couldn't create project.godot in project path."), MESSAGE_ERROR);
					} else {
						ResourceSaver::save(dir.plus_file("icon.png"), get_icon("DefaultProjectIcon", "EditorIcons"));

						FileAccess *f = FileAccess::open(dir.plus_file("default_env.tres"), FileAccess::WRITE);
						if (!f) {
							set_message(TTR("Couldn't create project.godot in project path."), MESSAGE_ERROR);
						} else {
							f->store_line("[gd_resource type=\"Environment\" load_steps=2 format=2]");
							f->store_line("[sub_resource type=\"ProceduralSky\" id=1]");
							f->store_line("[resource]");
							f->store_line("background_mode = 2");
							f->store_line("background_sky = SubResource( 1 )");
							memdelete(f);
						}
					}

				} else if (mode == MODE_INSTALL) {

					if (project_path->get_text().ends_with(".zip")) {
						dir = install_path->get_text();
						zip_path = project_path->get_text();
					}

					FileAccess *src_f = NULL;
					zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

					unzFile pkg = unzOpen2(zip_path.utf8().get_data(), &io);
					if (!pkg) {

						dialog_error->set_text(TTR("Error opening package file, not in ZIP format."));
						dialog_error->popup_centered_minsize();
						return;
					}

					int ret = unzGoToFirstFile(pkg);

					Vector<String> failed_files;

					int idx = 0;
					while (ret == UNZ_OK) {

						//get filename
						unz_file_info info;
						char fname[16384];
						ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

						String path = fname;

						int depth = 1; //stuff from github comes with tag
						bool skip = false;
						while (depth > 0) {
							int pp = path.find("/");
							if (pp == -1) {
								skip = true;
								break;
							}
							path = path.substr(pp + 1, path.length());
							depth--;
						}

						if (skip || path == String()) {
							//
						} else if (path.ends_with("/")) { // a dir

							path = path.substr(0, path.length() - 1);

							DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
							da->make_dir(dir.plus_file(path));
							memdelete(da);

						} else {

							Vector<uint8_t> data;
							data.resize(info.uncompressed_size);

							//read
							unzOpenCurrentFile(pkg);
							unzReadCurrentFile(pkg, data.ptrw(), data.size());
							unzCloseCurrentFile(pkg);

							FileAccess *f = FileAccess::open(dir.plus_file(path), FileAccess::WRITE);

							if (f) {
								f->store_buffer(data.ptr(), data.size());
								memdelete(f);
							} else {
								failed_files.push_back(path);
							}
						}

						idx++;
						ret = unzGoToNextFile(pkg);
					}

					unzClose(pkg);

					if (failed_files.size()) {
						String msg = TTR("The following files failed extraction from package:") + "\n\n";
						for (int i = 0; i < failed_files.size(); i++) {

							if (i > 15) {
								msg += "\nAnd " + itos(failed_files.size() - i) + " more files.";
								break;
							}
							msg += failed_files[i] + "\n";
						}

						dialog_error->set_text(msg);
						dialog_error->popup_centered_minsize();

					} else if (!project_path->get_text().ends_with(".zip")) {
						dialog_error->set_text(TTR("Package installed successfully!"));
						dialog_error->popup_centered_minsize();
					}
				}
			}

			dir = dir.replace("\\", "/");
			if (dir.ends_with("/"))
				dir = dir.substr(0, dir.length() - 1);
			String proj = get_project_key_from_path(dir);
			EditorSettings::get_singleton()->set("projects/" + proj, dir);
			EditorSettings::get_singleton()->save();

			hide();
			emit_signal("project_created", dir);
		}
	}

	void _remove_created_folder() {

		if (created_folder_path != "") {
			DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			d->remove(created_folder_path);
			memdelete(d);

			create_dir->set_disabled(false);
			created_folder_path = "";
		}
	}

	void cancel_pressed() {

		_remove_created_folder();

		project_path->clear();
		_path_text_changed("");
		project_name->clear();
		_text_changed("");

		if (status_rect->get_texture() == get_icon("StatusError", "EditorIcons"))
			msg->show();

		if (install_status_rect->get_texture() == get_icon("StatusError", "EditorIcons"))
			msg->show();
	}

	void _notification(int p_what) {

		if (p_what == MainLoop::NOTIFICATION_WM_QUIT_REQUEST)
			_remove_created_folder();
	}

protected:
	static void _bind_methods() {

		ClassDB::bind_method("_browse_path", &ProjectDialog::_browse_path);
		ClassDB::bind_method("_create_folder", &ProjectDialog::_create_folder);
		ClassDB::bind_method("_text_changed", &ProjectDialog::_text_changed);
		ClassDB::bind_method("_path_text_changed", &ProjectDialog::_path_text_changed);
		ClassDB::bind_method("_path_selected", &ProjectDialog::_path_selected);
		ClassDB::bind_method("_file_selected", &ProjectDialog::_file_selected);
		ClassDB::bind_method("_install_path_selected", &ProjectDialog::_install_path_selected);
		ClassDB::bind_method("_browse_install_path", &ProjectDialog::_browse_install_path);
		ADD_SIGNAL(MethodInfo("project_created"));
		ADD_SIGNAL(MethodInfo("projects_updated"));
	}

public:
	void set_zip_path(const String &p_path) {
		zip_path = p_path;
	}
	void set_zip_title(const String &p_title) {
		zip_title = p_title;
	}

	void set_mode(Mode p_mode) {

		mode = p_mode;
	}

	void set_project_path(const String &p_path) {
		project_path->set_text(p_path);
	}

	void show_dialog() {

		if (mode == MODE_RENAME) {

			project_path->set_editable(false);
			browse->hide();
			install_browse->hide();

			set_title(TTR("Rename Project"));
			get_ok()->set_text(TTR("Rename"));
			name_container->show();
			status_rect->hide();
			msg->hide();
			install_path_container->hide();
			install_status_rect->hide();
			rasterizer_container->hide();
			get_ok()->set_disabled(false);

			ProjectSettings *current = memnew(ProjectSettings);

			int err = current->setup(project_path->get_text(), "");
			if (err != OK) {
				set_message(vformat(TTR("Couldn't load project.godot in project path (error %d). It may be missing or corrupted."), err), MESSAGE_ERROR);
				status_rect->show();
				msg->show();
				get_ok()->set_disabled(true);
			} else if (current->has_setting("application/config/name")) {
				String proj = current->get("application/config/name");
				project_name->set_text(proj);
				_text_changed(proj);
			}

			project_name->call_deferred("grab_focus");

			create_dir->hide();

		} else {

			fav_dir = EditorSettings::get_singleton()->get("filesystem/directories/default_project_path");
			if (fav_dir != "") {
				project_path->set_text(fav_dir);
				fdialog->set_current_dir(fav_dir);
			} else {
				DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
				project_path->set_text(d->get_current_dir());
				fdialog->set_current_dir(d->get_current_dir());
				memdelete(d);
			}
			String proj = TTR("New Game Project");
			project_name->set_text(proj);
			_text_changed(proj);

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
				get_ok()->set_text(TTR("Import & Edit"));
				name_container->hide();
				install_path_container->hide();
				rasterizer_container->hide();
				project_path->grab_focus();

			} else if (mode == MODE_NEW) {

				set_title(TTR("Create New Project"));
				get_ok()->set_text(TTR("Create & Edit"));
				name_container->show();
				install_path_container->hide();
				rasterizer_container->show();
				project_name->call_deferred("grab_focus");
				project_name->call_deferred("select_all");

			} else if (mode == MODE_INSTALL) {

				set_title(TTR("Install Project:") + " " + zip_title);
				get_ok()->set_text(TTR("Install & Edit"));
				project_name->set_text(zip_title);
				name_container->show();
				install_path_container->hide();
				rasterizer_container->hide();
				project_path->grab_focus();
			}

			_test_path();
		}

		// Reset the dialog to its initial size. Otherwise, the dialog window would be too large
		// when opening a small dialog after closing a large dialog.
		set_size(get_minimum_size());
		popup_centered_minsize(Size2(500, 0) * EDSCALE);
	}

	ProjectDialog() {

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
		project_name->set_h_size_flags(SIZE_EXPAND_FILL);
		pnhb->add_child(project_name);

		create_dir = memnew(Button);
		pnhb->add_child(create_dir);
		create_dir->set_text(TTR("Create Folder"));
		create_dir->connect("pressed", this, "_create_folder");

		path_container = memnew(VBoxContainer);
		vb->add_child(path_container);

		l = memnew(Label);
		l->set_text(TTR("Project Path:"));
		path_container->add_child(l);

		HBoxContainer *pphb = memnew(HBoxContainer);
		path_container->add_child(pphb);

		project_path = memnew(LineEdit);
		project_path->set_h_size_flags(SIZE_EXPAND_FILL);
		pphb->add_child(project_path);

		install_path_container = memnew(VBoxContainer);
		vb->add_child(install_path_container);

		l = memnew(Label);
		l->set_text(TTR("Project Installation Path:"));
		install_path_container->add_child(l);

		HBoxContainer *iphb = memnew(HBoxContainer);
		install_path_container->add_child(iphb);

		install_path = memnew(LineEdit);
		install_path->set_h_size_flags(SIZE_EXPAND_FILL);
		iphb->add_child(install_path);

		// status icon
		status_rect = memnew(TextureRect);
		status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		pphb->add_child(status_rect);

		browse = memnew(Button);
		browse->set_text(TTR("Browse"));
		browse->connect("pressed", this, "_browse_path");
		pphb->add_child(browse);

		// install status icon
		install_status_rect = memnew(TextureRect);
		install_status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		iphb->add_child(install_status_rect);

		install_browse = memnew(Button);
		install_browse->set_text(TTR("Browse"));
		install_browse->connect("pressed", this, "_browse_install_path");
		iphb->add_child(install_browse);

		msg = memnew(Label);
		msg->set_align(Label::ALIGN_CENTER);
		vb->add_child(msg);

		// rasterizer selection
		rasterizer_container = memnew(VBoxContainer);
		vb->add_child(rasterizer_container);
		l = memnew(Label);
		l->set_text(TTR("Renderer:"));
		rasterizer_container->add_child(l);
		Container *rshb = memnew(HBoxContainer);
		rasterizer_container->add_child(rshb);
		rasterizer_button_group.instance();

		// Enable GLES3 by default as it's the default value for the project setting.
#ifndef SERVER_ENABLED
		bool gles3_viable = RasterizerGLES3::is_viable() == OK;
#else
		// Whatever, project manager isn't even used in headless builds.
		bool gles3_viable = false;
#endif

		Container *rvb = memnew(VBoxContainer);
		rvb->set_h_size_flags(SIZE_EXPAND_FILL);
		rshb->add_child(rvb);
		Button *rs_button = memnew(CheckBox);
		rs_button->set_button_group(rasterizer_button_group);
		rs_button->set_text(TTR("OpenGL ES 3.0"));
		rs_button->set_meta("driver_name", "GLES3");
		rs_button->set_pressed(gles3_viable);
		// If GLES3 can't be used, don't let users shoot themselves in the foot.
		rs_button->set_disabled(!gles3_viable);
		rvb->add_child(rs_button);
		l = memnew(Label);
		l->set_text(TTR("Not supported by your GPU drivers."));
		rvb->add_child(l);
		l = memnew(Label);
		l->set_text(TTR("Higher visual quality\nAll features available\nIncompatible with older hardware\nNot recommended for web games"));
		rvb->add_child(l);

		rshb->add_child(memnew(VSeparator));

		rvb = memnew(VBoxContainer);
		rvb->set_h_size_flags(SIZE_EXPAND_FILL);
		rshb->add_child(rvb);
		rs_button = memnew(CheckBox);
		rs_button->set_button_group(rasterizer_button_group);
		rs_button->set_text(TTR("OpenGL ES 2.0"));
		rs_button->set_meta("driver_name", "GLES2");
		rs_button->set_pressed(!gles3_viable);
		rvb->add_child(rs_button);
		l = memnew(Label);
		l->set_text(TTR("Lower visual quality\nSome features not available\nWorks on most hardware\nRecommended for web games"));
		rvb->add_child(l);

		l = memnew(Label);
		l->set_text(TTR("Renderer can be changed later, but scenes may need to be adjusted."));
		l->set_align(Label::ALIGN_CENTER);
		rasterizer_container->add_child(l);

		fdialog = memnew(FileDialog);
		fdialog->set_access(FileDialog::ACCESS_FILESYSTEM);
		fdialog_install = memnew(FileDialog);
		fdialog_install->set_access(FileDialog::ACCESS_FILESYSTEM);
		add_child(fdialog);
		add_child(fdialog_install);
		project_name->connect("text_changed", this, "_text_changed");
		project_path->connect("text_changed", this, "_path_text_changed");
		install_path->connect("text_changed", this, "_path_text_changed");
		fdialog->connect("dir_selected", this, "_path_selected");
		fdialog->connect("file_selected", this, "_file_selected");
		fdialog_install->connect("dir_selected", this, "_install_path_selected");
		fdialog_install->connect("file_selected", this, "_install_path_selected");

		set_hide_on_ok(false);
		mode = MODE_NEW;

		dialog_error = memnew(AcceptDialog);
		add_child(dialog_error);
	}
};

class ProjectListItemControl : public HBoxContainer {
	GDCLASS(ProjectListItemControl, HBoxContainer)
public:
	TextureButton *favorite_button;
	TextureRect *icon;
	bool icon_needs_reload;
	bool hover;

	ProjectListItemControl() {
		favorite_button = NULL;
		icon = NULL;
		icon_needs_reload = true;
		hover = false;

		set_focus_mode(FocusMode::FOCUS_ALL);
	}

	void set_is_favorite(bool fav) {
		favorite_button->set_modulate(fav ? Color(1, 1, 1, 1) : Color(1, 1, 1, 0.2));
	}

	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_MOUSE_ENTER: {
				hover = true;
				update();
			} break;
			case NOTIFICATION_MOUSE_EXIT: {
				hover = false;
				update();
			} break;
			case NOTIFICATION_DRAW: {
				if (hover) {
					draw_style_box(get_stylebox("hover", "Tree"), Rect2(Point2(), get_size() - Size2(10, 0) * EDSCALE));
				}
			} break;
		}
	}
};

class ProjectList : public ScrollContainer {
	GDCLASS(ProjectList, ScrollContainer)
public:
	static const char *SIGNAL_SELECTION_CHANGED;
	static const char *SIGNAL_PROJECT_ASK_OPEN;

	enum MenuOptions {
		GLOBAL_NEW_WINDOW,
		GLOBAL_OPEN_PROJECT
	};

	// Can often be passed by copy
	struct Item {
		String project_key;
		String project_name;
		String description;
		String path;
		String icon;
		String main_scene;
		uint64_t last_modified;
		bool favorite;
		bool grayed;
		bool missing;
		int version;

		ProjectListItemControl *control;

		Item() {}

		Item(const String &p_project,
				const String &p_name,
				const String &p_description,
				const String &p_path,
				const String &p_icon,
				const String &p_main_scene,
				uint64_t p_last_modified,
				bool p_favorite,
				bool p_grayed,
				bool p_missing,
				int p_version) {

			project_key = p_project;
			project_name = p_name;
			description = p_description;
			path = p_path;
			icon = p_icon;
			main_scene = p_main_scene;
			last_modified = p_last_modified;
			favorite = p_favorite;
			grayed = p_grayed;
			missing = p_missing;
			version = p_version;
			control = NULL;
		}

		_FORCE_INLINE_ bool operator==(const Item &l) const {
			return project_key == l.project_key;
		}
	};

	ProjectList();
	~ProjectList();

	void update_dock_menu();
	void load_projects();
	void set_search_term(String p_search_term);
	void set_order_option(ProjectListFilter::FilterOption p_option);
	void sort_projects();
	int get_project_count() const;
	void select_project(int p_index);
	void erase_selected_projects();
	Vector<Item> get_selected_projects() const;
	const Set<String> &get_selected_project_keys() const;
	void ensure_project_visible(int p_index);
	int get_single_selected_index() const;
	bool is_any_project_missing() const;
	void erase_missing_projects();
	int refresh_project(const String &dir_path);

private:
	static void _bind_methods();
	void _notification(int p_what);

	void _panel_draw(Node *p_hb);
	void _panel_input(const Ref<InputEvent> &p_ev, Node *p_hb);
	void _favorite_pressed(Node *p_hb);
	void _show_project(const String &p_path);

	void select_range(int p_begin, int p_end);
	void toggle_select(int p_index);
	void create_project_item_control(int p_index);
	void remove_project(int p_index, bool p_update_settings);
	void update_icons_async();
	void load_project_icon(int p_index);

	static void load_project_data(const String &p_property_key, Item &p_item, bool p_favorite);

	String _search_term;
	ProjectListFilter::FilterOption _order_option;
	Set<String> _selected_project_keys;
	String _last_clicked; // Project key
	VBoxContainer *_scroll_children;
	int _icon_load_index;

	Vector<Item> _projects;
};

struct ProjectListComparator {
	ProjectListFilter::FilterOption order_option;

	// operator<
	_FORCE_INLINE_ bool operator()(const ProjectList::Item &a, const ProjectList::Item &b) const {
		if (a.favorite && !b.favorite) {
			return true;
		}
		if (b.favorite && !a.favorite) {
			return false;
		}
		switch (order_option) {
			case ProjectListFilter::FILTER_PATH:
				return a.project_key < b.project_key;
			case ProjectListFilter::FILTER_MODIFIED:
				return a.last_modified > b.last_modified;
			default:
				return a.project_name < b.project_name;
		}
	}
};

ProjectList::ProjectList() {
	_order_option = ProjectListFilter::FILTER_MODIFIED;

	_scroll_children = memnew(VBoxContainer);
	_scroll_children->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(_scroll_children);

	_icon_load_index = 0;
}

ProjectList::~ProjectList() {
}

void ProjectList::update_icons_async() {
	_icon_load_index = 0;
	set_process(true);
}

void ProjectList::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {

		// Load icons as a coroutine to speed up launch when you have hundreds of projects
		if (_icon_load_index < _projects.size()) {
			Item &item = _projects.write[_icon_load_index];
			if (item.control->icon_needs_reload) {
				load_project_icon(_icon_load_index);
			}
			_icon_load_index++;

		} else {
			set_process(false);
		}
	}
}

void ProjectList::load_project_icon(int p_index) {
	Item &item = _projects.write[p_index];

	Ref<Texture> default_icon = get_icon("DefaultProjectIcon", "EditorIcons");
	Ref<Texture> icon;
	if (item.icon != "") {
		Ref<Image> img;
		img.instance();
		Error err = img->load(item.icon.replace_first("res://", item.path + "/"));
		if (err == OK) {

			img->resize(default_icon->get_width(), default_icon->get_height(), Image::INTERPOLATE_LANCZOS);
			Ref<ImageTexture> it = memnew(ImageTexture);
			it->create_from_image(img);
			icon = it;
		}
	}
	if (icon.is_null()) {
		icon = default_icon;
	}

	item.control->icon->set_texture(icon);
	item.control->icon_needs_reload = false;
}

void ProjectList::load_project_data(const String &p_property_key, Item &p_item, bool p_favorite) {

	String path = EditorSettings::get_singleton()->get(p_property_key);
	String conf = path.plus_file("project.godot");
	bool grayed = false;
	bool missing = false;

	Ref<ConfigFile> cf = memnew(ConfigFile);
	Error cf_err = cf->load(conf);

	int config_version = 0;
	String project_name = TTR("Unnamed Project");
	if (cf_err == OK) {
		String cf_project_name = static_cast<String>(cf->get_value("application", "config/name", ""));
		if (cf_project_name != "")
			project_name = cf_project_name.xml_unescape();
		config_version = (int)cf->get_value("", "config_version", 0);
	}

	if (config_version > ProjectSettings::CONFIG_VERSION) {
		// Comes from an incompatible (more recent) Godot version, grey it out
		grayed = true;
	}

	String description = cf->get_value("application", "config/description", "");
	String icon = cf->get_value("application", "config/icon", "");
	String main_scene = cf->get_value("application", "run/main_scene", "");

	uint64_t last_modified = 0;
	if (FileAccess::exists(conf)) {
		last_modified = FileAccess::get_modified_time(conf);

		String fscache = path.plus_file(".fscache");
		if (FileAccess::exists(fscache)) {
			uint64_t cache_modified = FileAccess::get_modified_time(fscache);
			if (cache_modified > last_modified)
				last_modified = cache_modified;
		}
	} else {
		grayed = true;
		missing = true;
		print_line("Project is missing: " + conf);
	}

	String project_key = p_property_key.get_slice("/", 1);

	p_item = Item(project_key, project_name, description, path, icon, main_scene, last_modified, p_favorite, grayed, missing, config_version);
}

void ProjectList::load_projects() {
	// This is a full, hard reload of the list. Don't call this unless really required, it's expensive.
	// If you have 150 projects, it may read through 150 files on your disk at once + load 150 icons.

	// Clear whole list
	for (int i = 0; i < _projects.size(); ++i) {
		Item &project = _projects.write[i];
		CRASH_COND(project.control == NULL);
		memdelete(project.control); // Why not queue_free()?
	}
	_projects.clear();
	_last_clicked = "";
	_selected_project_keys.clear();

	// Load data
	// TODO Would be nice to change how projects and favourites are stored... it complicates things a bit.
	// Use a dictionary associating project path to metadata (like is_favorite).

	List<PropertyInfo> properties;
	EditorSettings::get_singleton()->get_property_list(&properties);

	Set<String> favorites;
	// Find favourites...
	for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
		String property_key = E->get().name;
		if (property_key.begins_with("favorite_projects/")) {
			favorites.insert(property_key);
		}
	}

	for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
		// This is actually something like "projects/C:::Documents::Godot::Projects::MyGame"
		String property_key = E->get().name;
		if (!property_key.begins_with("projects/"))
			continue;

		String project_key = property_key.get_slice("/", 1);
		bool favorite = favorites.has("favorite_projects/" + project_key);

		Item item;
		load_project_data(property_key, item, favorite);

		_projects.push_back(item);
	}

	// Create controls
	for (int i = 0; i < _projects.size(); ++i) {
		create_project_item_control(i);
	}

	sort_projects();

	set_v_scroll(0);

	update_icons_async();

	update_dock_menu();
}

void ProjectList::update_dock_menu() {
	OS::get_singleton()->global_menu_clear("_dock");

	int favs_added = 0;
	int total_added = 0;
	for (int i = 0; i < _projects.size(); ++i) {
		if (!_projects[i].grayed && !_projects[i].missing) {
			if (_projects[i].favorite) {
				favs_added++;
			} else {
				if (favs_added != 0) {
					OS::get_singleton()->global_menu_add_separator("_dock");
				}
				favs_added = 0;
			}
			OS::get_singleton()->global_menu_add_item("_dock", _projects[i].project_name + " ( " + _projects[i].path + " )", GLOBAL_OPEN_PROJECT, Variant(_projects[i].path.plus_file("project.godot")));
			total_added++;
		}
	}
	if (total_added != 0) {
		OS::get_singleton()->global_menu_add_separator("_dock");
	}
	OS::get_singleton()->global_menu_add_item("_dock", TTR("New Window"), GLOBAL_NEW_WINDOW, Variant());
}

void ProjectList::create_project_item_control(int p_index) {

	// Will be added last in the list, so make sure indexes match
	ERR_FAIL_COND(p_index != _scroll_children->get_child_count());

	Item &item = _projects.write[p_index];
	ERR_FAIL_COND(item.control != NULL); // Already created

	Ref<Texture> favorite_icon = get_icon("Favorites", "EditorIcons");
	Color font_color = get_color("font_color", "Tree");

	ProjectListItemControl *hb = memnew(ProjectListItemControl);
	hb->connect("draw", this, "_panel_draw", varray(hb));
	hb->connect("gui_input", this, "_panel_input", varray(hb));
	hb->add_constant_override("separation", 10 * EDSCALE);
	hb->set_tooltip(item.description);

	VBoxContainer *favorite_box = memnew(VBoxContainer);
	favorite_box->set_name("FavoriteBox");
	TextureButton *favorite = memnew(TextureButton);
	favorite->set_name("FavoriteButton");
	favorite->set_normal_texture(favorite_icon);
	// This makes the project's "hover" style display correctly when hovering the favorite icon
	favorite->set_mouse_filter(MOUSE_FILTER_PASS);
	favorite->connect("pressed", this, "_favorite_pressed", varray(hb));
	favorite_box->add_child(favorite);
	favorite_box->set_alignment(BoxContainer::ALIGN_CENTER);
	hb->add_child(favorite_box);
	hb->favorite_button = favorite;
	hb->set_is_favorite(item.favorite);

	TextureRect *tf = memnew(TextureRect);
	// The project icon may not be loaded by the time the control is displayed,
	// so use a loading placeholder.
	tf->set_texture(get_icon("ProjectIconLoading", "EditorIcons"));
	tf->set_v_size_flags(SIZE_SHRINK_CENTER);
	if (item.missing) {
		tf->set_modulate(Color(1, 1, 1, 0.5));
	}
	hb->add_child(tf);
	hb->icon = tf;

	VBoxContainer *vb = memnew(VBoxContainer);
	if (item.grayed)
		vb->set_modulate(Color(1, 1, 1, 0.5));
	vb->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(vb);
	Control *ec = memnew(Control);
	ec->set_custom_minimum_size(Size2(0, 1));
	ec->set_mouse_filter(MOUSE_FILTER_PASS);
	vb->add_child(ec);
	Label *title = memnew(Label(!item.missing ? item.project_name : TTR("Missing Project")));
	title->add_font_override("font", get_font("title", "EditorFonts"));
	title->add_color_override("font_color", font_color);
	title->set_clip_text(true);
	vb->add_child(title);

	HBoxContainer *path_hb = memnew(HBoxContainer);
	path_hb->set_h_size_flags(SIZE_EXPAND_FILL);
	vb->add_child(path_hb);

	Button *show = memnew(Button);
	// Display a folder icon if the project directory can be opened, or a "broken file" icon if it can't
	show->set_icon(get_icon(!item.missing ? "Load" : "FileBroken", "EditorIcons"));
	show->set_flat(true);
	if (!item.grayed) {
		// Don't make the icon less prominent if the parent is already grayed out
		show->set_modulate(Color(1, 1, 1, 0.5));
	}
	path_hb->add_child(show);

	if (!item.missing) {
		show->connect("pressed", this, "_show_project", varray(item.path));
		show->set_tooltip(TTR("Show in File Manager"));
	} else {
		show->set_tooltip(TTR("Error: Project is missing on the filesystem."));
	}

	Label *fpath = memnew(Label(item.path));
	path_hb->add_child(fpath);
	fpath->set_h_size_flags(SIZE_EXPAND_FILL);
	fpath->set_modulate(Color(1, 1, 1, 0.5));
	fpath->add_color_override("font_color", font_color);
	fpath->set_clip_text(true);

	_scroll_children->add_child(hb);
	item.control = hb;
}

void ProjectList::set_search_term(String p_search_term) {
	_search_term = p_search_term;
}

void ProjectList::set_order_option(ProjectListFilter::FilterOption p_option) {
	if (_order_option != p_option) {
		_order_option = p_option;
		EditorSettings::get_singleton()->set("project_manager/sorting_order", (int)_order_option);
		EditorSettings::get_singleton()->save();
	}
}

void ProjectList::sort_projects() {

	SortArray<Item, ProjectListComparator> sorter;
	sorter.compare.order_option = _order_option;
	sorter.sort(_projects.ptrw(), _projects.size());

	for (int i = 0; i < _projects.size(); ++i) {
		Item &item = _projects.write[i];

		bool visible = true;
		if (_search_term != "") {

			String search_path;
			if (_search_term.find("/") != -1) {
				// Search path will match the whole path
				search_path = item.path;
			} else {
				// Search path will only match the last path component to make searching more strict
				search_path = item.path.get_file();
			}

			// When searching, display projects whose name or path contain the search term
			visible = item.project_name.findn(_search_term) != -1 || search_path.findn(_search_term) != -1;
		}

		item.control->set_visible(visible);
	}

	for (int i = 0; i < _projects.size(); ++i) {
		Item &item = _projects.write[i];
		item.control->get_parent()->move_child(item.control, i);
	}

	// Rewind the coroutine because order of projects changed
	update_icons_async();

	update_dock_menu();
}

const Set<String> &ProjectList::get_selected_project_keys() const {
	// Faster if that's all you need
	return _selected_project_keys;
}

Vector<ProjectList::Item> ProjectList::get_selected_projects() const {
	Vector<Item> items;
	if (_selected_project_keys.size() == 0) {
		return items;
	}
	items.resize(_selected_project_keys.size());
	int j = 0;
	for (int i = 0; i < _projects.size(); ++i) {
		const Item &item = _projects[i];
		if (_selected_project_keys.has(item.project_key)) {
			items.write[j++] = item;
		}
	}
	ERR_FAIL_COND_V(j != items.size(), items);
	return items;
}

void ProjectList::ensure_project_visible(int p_index) {
	const Item &item = _projects[p_index];

	int item_top = item.control->get_position().y;
	int item_bottom = item.control->get_position().y + item.control->get_size().y;

	if (item_top < get_v_scroll()) {
		set_v_scroll(item_top);

	} else if (item_bottom > get_v_scroll() + get_size().y) {
		set_v_scroll(item_bottom - get_size().y);
	}
}

int ProjectList::get_single_selected_index() const {
	if (_selected_project_keys.size() == 0) {
		// Default selection
		return 0;
	}
	String key;
	if (_selected_project_keys.size() == 1) {
		// Only one selected
		key = _selected_project_keys.front()->get();
	} else {
		// Multiple selected, consider the last clicked one as "main"
		key = _last_clicked;
	}
	for (int i = 0; i < _projects.size(); ++i) {
		if (_projects[i].project_key == key) {
			return i;
		}
	}
	return 0;
}

void ProjectList::remove_project(int p_index, bool p_update_settings) {
	const Item item = _projects[p_index]; // Take a copy

	_selected_project_keys.erase(item.project_key);

	if (_last_clicked == item.project_key) {
		_last_clicked = "";
	}

	memdelete(item.control);
	_projects.remove(p_index);

	if (p_update_settings) {
		EditorSettings::get_singleton()->erase("projects/" + item.project_key);
		EditorSettings::get_singleton()->erase("favorite_projects/" + item.project_key);
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

	if (_projects.empty()) {
		return;
	}

	int deleted_count = 0;
	int remaining_count = 0;

	for (int i = 0; i < _projects.size(); ++i) {
		const Item &item = _projects[i];

		if (item.missing) {
			remove_project(i, true);
			--i;
			++deleted_count;

		} else {
			++remaining_count;
		}
	}

	print_line("Removed " + itos(deleted_count) + " projects from the list, remaining " + itos(remaining_count) + " projects");

	EditorSettings::get_singleton()->save();
}

int ProjectList::refresh_project(const String &dir_path) {
	// Reads editor settings and reloads information about a specific project.
	// If it wasn't loaded and should be in the list, it is added (i.e new project).
	// If it isn't in the list anymore, it is removed.
	// If it is in the list but doesn't exist anymore, it is marked as missing.

	String project_key = get_project_key_from_path(dir_path);

	// Read project manager settings
	bool is_favourite = false;
	bool should_be_in_list = false;
	String property_key = "projects/" + project_key;
	{
		List<PropertyInfo> properties;
		EditorSettings::get_singleton()->get_property_list(&properties);
		String favorite_property_key = "favorite_projects/" + project_key;

		bool found = false;
		for (List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			String prop = E->get().name;
			if (!found && prop == property_key) {
				found = true;
			} else if (!is_favourite && prop == favorite_property_key) {
				is_favourite = true;
			}
		}

		should_be_in_list = found;
	}

	bool was_selected = _selected_project_keys.has(project_key);

	// Remove item in any case
	for (int i = 0; i < _projects.size(); ++i) {
		const Item &existing_item = _projects[i];
		if (existing_item.path == dir_path) {
			remove_project(i, false);
			break;
		}
	}

	int index = -1;
	if (should_be_in_list) {
		// Recreate it with updated info

		Item item;
		load_project_data(property_key, item, is_favourite);

		_projects.push_back(item);
		create_project_item_control(_projects.size() - 1);

		sort_projects();

		for (int i = 0; i < _projects.size(); ++i) {
			if (_projects[i].project_key == project_key) {
				if (was_selected) {
					select_project(i);
					ensure_project_visible(i);
				}
				load_project_icon(i);

				index = i;
				break;
			}
		}
	}

	return index;
}

int ProjectList::get_project_count() const {
	return _projects.size();
}

void ProjectList::select_project(int p_index) {

	Vector<Item> previous_selected_items = get_selected_projects();
	_selected_project_keys.clear();

	for (int i = 0; i < previous_selected_items.size(); ++i) {
		previous_selected_items[i].control->update();
	}

	toggle_select(p_index);
}

inline void sort(int &a, int &b) {
	if (a > b) {
		int temp = a;
		a = b;
		b = temp;
	}
}

void ProjectList::select_range(int p_begin, int p_end) {
	sort(p_begin, p_end);
	select_project(p_begin);
	for (int i = p_begin + 1; i <= p_end; ++i) {
		toggle_select(i);
	}
}

void ProjectList::toggle_select(int p_index) {
	Item &item = _projects.write[p_index];
	if (_selected_project_keys.has(item.project_key)) {
		_selected_project_keys.erase(item.project_key);
	} else {
		_selected_project_keys.insert(item.project_key);
	}
	item.control->update();
}

void ProjectList::erase_selected_projects() {

	if (_selected_project_keys.size() == 0) {
		return;
	}

	for (int i = 0; i < _projects.size(); ++i) {
		Item &item = _projects.write[i];
		if (_selected_project_keys.has(item.project_key) && item.control->is_visible()) {

			EditorSettings::get_singleton()->erase("projects/" + item.project_key);
			EditorSettings::get_singleton()->erase("favorite_projects/" + item.project_key);

			memdelete(item.control);
			_projects.remove(i);
			--i;
		}
	}

	EditorSettings::get_singleton()->save();

	_selected_project_keys.clear();
	_last_clicked = "";

	update_dock_menu();
}

// Draws selected project highlight
void ProjectList::_panel_draw(Node *p_hb) {
	Control *hb = Object::cast_to<Control>(p_hb);

	hb->draw_line(Point2(0, hb->get_size().y + 1), Point2(hb->get_size().x - 10, hb->get_size().y + 1), get_color("guide_color", "Tree"));

	String key = _projects[p_hb->get_index()].project_key;

	if (_selected_project_keys.has(key)) {
		hb->draw_style_box(get_stylebox("selected", "Tree"), Rect2(Point2(), hb->get_size() - Size2(10, 0) * EDSCALE));
	}
}

// Input for each item in the list
void ProjectList::_panel_input(const Ref<InputEvent> &p_ev, Node *p_hb) {

	Ref<InputEventMouseButton> mb = p_ev;
	int clicked_index = p_hb->get_index();
	const Item &clicked_project = _projects[clicked_index];

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

		if (mb->get_shift() && _selected_project_keys.size() > 0 && _last_clicked != "" && clicked_project.project_key != _last_clicked) {

			int anchor_index = -1;
			for (int i = 0; i < _projects.size(); ++i) {
				const Item &p = _projects[i];
				if (p.project_key == _last_clicked) {
					anchor_index = p.control->get_index();
					break;
				}
			}
			CRASH_COND(anchor_index == -1);
			select_range(anchor_index, clicked_index);

		} else if (mb->get_control()) {
			toggle_select(clicked_index);

		} else {
			_last_clicked = clicked_project.project_key;
			select_project(clicked_index);
		}

		emit_signal(SIGNAL_SELECTION_CHANGED);

		if (!mb->get_control() && mb->is_doubleclick()) {
			emit_signal(SIGNAL_PROJECT_ASK_OPEN);
		}
	}
}

void ProjectList::_favorite_pressed(Node *p_hb) {

	ProjectListItemControl *control = Object::cast_to<ProjectListItemControl>(p_hb);

	int index = control->get_index();
	Item item = _projects.write[index]; // Take copy

	item.favorite = !item.favorite;

	if (item.favorite) {
		EditorSettings::get_singleton()->set("favorite_projects/" + item.project_key, item.path);
	} else {
		EditorSettings::get_singleton()->erase("favorite_projects/" + item.project_key);
	}
	EditorSettings::get_singleton()->save();

	_projects.write[index] = item;

	control->set_is_favorite(item.favorite);

	sort_projects();

	if (item.favorite) {
		for (int i = 0; i < _projects.size(); ++i) {
			if (_projects[i].project_key == item.project_key) {
				ensure_project_visible(i);
				break;
			}
		}
	}

	update_dock_menu();
}

void ProjectList::_show_project(const String &p_path) {

	OS::get_singleton()->shell_open(String("file://") + p_path);
}

const char *ProjectList::SIGNAL_SELECTION_CHANGED = "selection_changed";
const char *ProjectList::SIGNAL_PROJECT_ASK_OPEN = "project_ask_open";

void ProjectList::_bind_methods() {

	ClassDB::bind_method("_panel_draw", &ProjectList::_panel_draw);
	ClassDB::bind_method("_panel_input", &ProjectList::_panel_input);
	ClassDB::bind_method("_favorite_pressed", &ProjectList::_favorite_pressed);
	ClassDB::bind_method("_show_project", &ProjectList::_show_project);

	ADD_SIGNAL(MethodInfo(SIGNAL_SELECTION_CHANGED));
	ADD_SIGNAL(MethodInfo(SIGNAL_PROJECT_ASK_OPEN));
}

void ProjectManager::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {

			Engine::get_singleton()->set_editor_hint(false);
		} break;
		case NOTIFICATION_RESIZED: {

			if (open_templates->is_visible()) {
				open_templates->popup_centered_minsize();
			}
		} break;
		case NOTIFICATION_READY: {

			if (_project_list->get_project_count() == 0 && StreamPeerSSL::is_available())
				open_templates->popup_centered_minsize();

			if (_project_list->get_project_count() >= 1) {
				// Focus on the search box immediately to allow the user
				// to search without having to reach for their mouse
				project_filter->search_box->grab_focus();
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			set_process_unhandled_input(is_visible_in_tree());
		} break;
		case NOTIFICATION_WM_QUIT_REQUEST: {

			_dim_window();
		} break;
	}
}

void ProjectManager::_dim_window() {

	// This method must be called before calling `get_tree()->quit()`.
	// Otherwise, its effect won't be visible

	// Dim the project manager window while it's quitting to make it clearer that it's busy.
	// No transition is applied, as the effect needs to be visible immediately
	float c = 0.5f;
	Color dim_color = Color(c, c, c);
	gui_base->set_modulate(dim_color);
}

void ProjectManager::_update_project_buttons() {

	Vector<ProjectList::Item> selected_projects = _project_list->get_selected_projects();
	bool empty_selection = selected_projects.empty();

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
	run_btn->set_disabled(empty_selection || is_missing_project_selected);

	erase_missing_btn->set_disabled(!_project_list->is_any_project_missing());
}

void ProjectManager::_unhandled_input(const Ref<InputEvent> &p_ev) {

	Ref<InputEventKey> k = p_ev;

	if (k.is_valid()) {

		if (!k->is_pressed()) {
			return;
		}

		// Pressing Command + Q quits the Project Manager
		// This is handled by the platform implementation on macOS,
		// so only define the shortcut on other platforms
#ifndef OSX_ENABLED
		if (k->get_scancode_with_modifiers() == (KEY_MASK_CMD | KEY_Q)) {
			_dim_window();
			get_tree()->quit();
		}
#endif

		if (tabs->get_current_tab() != 0)
			return;

		bool scancode_handled = true;

		switch (k->get_scancode()) {

			case KEY_ENTER: {

				_open_selected_projects_ask();
			} break;
			case KEY_DELETE: {

				_erase_project();
			} break;
			case KEY_HOME: {

				if (_project_list->get_project_count() > 0) {
					_project_list->select_project(0);
					_update_project_buttons();
				}

			} break;
			case KEY_END: {

				if (_project_list->get_project_count() > 0) {
					_project_list->select_project(_project_list->get_project_count() - 1);
					_update_project_buttons();
				}

			} break;
			case KEY_UP: {

				if (k->get_shift())
					break;

				int index = _project_list->get_single_selected_index();
				if (index > 0) {
					_project_list->select_project(index - 1);
					_project_list->ensure_project_visible(index - 1);
					_update_project_buttons();
				}

				break;
			}
			case KEY_DOWN: {

				if (k->get_shift())
					break;

				int index = _project_list->get_single_selected_index();
				if (index + 1 < _project_list->get_project_count()) {
					_project_list->select_project(index + 1);
					_project_list->ensure_project_visible(index + 1);
					_update_project_buttons();
				}

			} break;
			case KEY_F: {
				if (k->get_command())
					this->project_filter->search_box->grab_focus();
				else
					scancode_handled = false;
			} break;
			default: {
				scancode_handled = false;
			} break;
		}

		if (scancode_handled) {
			accept_event();
		}
	}
}

void ProjectManager::_load_recent_projects() {

	_project_list->set_order_option(project_order_filter->get_filter_option());
	_project_list->set_search_term(project_filter->get_search_term());
	_project_list->load_projects();

	_update_project_buttons();

	tabs->set_current_tab(0);
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
	project_filter->clear();
	int i = _project_list->refresh_project(dir);
	_project_list->select_project(i);
	_project_list->ensure_project_visible(i);
	_open_selected_projects_ask();

	_project_list->update_dock_menu();
}

void ProjectManager::_confirm_update_settings() {
	_open_selected_projects();
}

void ProjectManager::_global_menu_action(const Variant &p_id, const Variant &p_meta) {

	int id = (int)p_id;
	if (id == ProjectList::GLOBAL_NEW_WINDOW) {
		List<String> args;
		args.push_back("-p");
		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		OS::get_singleton()->execute(exec, args, false, &pid);
	} else if (id == ProjectList::GLOBAL_OPEN_PROJECT) {
		String conf = (String)p_meta;

		if (conf != String()) {
			List<String> args;
			args.push_back(conf);
			String exec = OS::get_singleton()->get_executable_path();

			OS::ProcessID pid = 0;
			OS::get_singleton()->execute(exec, args, false, &pid);
		}
	}
}

void ProjectManager::_open_selected_projects() {

	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	for (const Set<String>::Element *E = selected_list.front(); E; E = E->next()) {
		const String &selected = E->get();
		String path = EditorSettings::get_singleton()->get("projects/" + selected);
		String conf = path.plus_file("project.godot");

		if (!FileAccess::exists(conf)) {
			dialog_error->set_text(vformat(TTR("Can't open project at '%s'."), path));
			dialog_error->popup_centered_minsize();
			return;
		}

		print_line("Editing project: " + path + " (" + selected + ")");

		List<String> args;

		args.push_back("--path");
		args.push_back(path);

		args.push_back("--editor");

		if (OS::get_singleton()->is_stdout_debug_enabled()) {
			args.push_back("--debug");
		}

		if (OS::get_singleton()->is_stdout_verbose()) {
			args.push_back("--verbose");
		}

		if (OS::get_singleton()->is_disable_crash_handler()) {
			args.push_back("--disable-crash-handler");
		}

		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		Error err = OS::get_singleton()->execute(exec, args, false, &pid);
		ERR_FAIL_COND(err);
	}

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_open_selected_projects_ask() {

	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	if (selected_list.size() > 1) {
		multi_open_ask->set_text(TTR("Are you sure to open more than one project?"));
		multi_open_ask->popup_centered_minsize();
		return;
	}

	ProjectList::Item project = _project_list->get_selected_projects()[0];
	if (project.missing) {
		return;
	}

	// Update the project settings or don't open
	String conf = project.path.plus_file("project.godot");
	int config_version = project.version;

	// Check if the config_version property was empty or 0
	if (config_version == 0) {
		ask_update_settings->set_text(vformat(TTR("The following project settings file does not specify the version of Godot through which it was created.\n\n%s\n\nIf you proceed with opening it, it will be converted to Godot's current configuration file format.\nWarning: You won't be able to open the project with previous versions of the engine anymore."), conf));
		ask_update_settings->popup_centered_minsize();
		return;
	}
	// Check if we need to convert project settings from an earlier engine version
	if (config_version < ProjectSettings::CONFIG_VERSION) {
		ask_update_settings->set_text(vformat(TTR("The following project settings file was generated by an older engine version, and needs to be converted for this version:\n\n%s\n\nDo you want to convert it?\nWarning: You won't be able to open the project with previous versions of the engine anymore."), conf));
		ask_update_settings->popup_centered_minsize();
		return;
	}
	// Check if the file was generated by a newer, incompatible engine version
	if (config_version > ProjectSettings::CONFIG_VERSION) {
		dialog_error->set_text(vformat(TTR("Can't open project at '%s'.") + "\n" + TTR("The project settings were created by a newer engine version, whose settings are not compatible with this version."), project.path));
		dialog_error->popup_centered_minsize();
		return;
	}

	// Open if the project is up-to-date
	_open_selected_projects();
}

void ProjectManager::_run_project_confirm() {

	Vector<ProjectList::Item> selected_list = _project_list->get_selected_projects();

	for (int i = 0; i < selected_list.size(); ++i) {

		const String &selected_main = selected_list[i].main_scene;
		if (selected_main == "") {
			run_error_diag->set_text(TTR("Can't run project: no main scene defined.\nPlease edit the project and set the main scene in the Project Settings under the \"Application\" category."));
			run_error_diag->popup_centered();
			continue;
		}

		const String &selected = selected_list[i].project_key;
		String path = EditorSettings::get_singleton()->get("projects/" + selected);

		if (!DirAccess::exists(path + "/.import")) {
			run_error_diag->set_text(TTR("Can't run project: Assets need to be imported.\nPlease edit the project to trigger the initial import."));
			run_error_diag->popup_centered();
			continue;
		}

		print_line("Running project: " + path + " (" + selected + ")");

		List<String> args;

		args.push_back("--path");
		args.push_back(path);

		if (OS::get_singleton()->is_disable_crash_handler()) {
			args.push_back("--disable-crash-handler");
		}

		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		Error err = OS::get_singleton()->execute(exec, args, false, &pid);
		ERR_FAIL_COND(err);
	}
}

// When you press the "Run" button
void ProjectManager::_run_project() {

	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() < 1) {
		return;
	}

	if (selected_list.size() > 1) {
		multi_run_ask->set_text(vformat(TTR("Are you sure to run %d projects at once?"), selected_list.size()));
		multi_run_ask->popup_centered_minsize();
	} else {
		_run_project_confirm();
	}
}

void ProjectManager::_scan_dir(const String &path, List<String> *r_projects) {
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->change_dir(path);
	da->list_dir_begin();
	String n = da->get_next();
	while (n != String()) {
		if (da->current_is_dir() && !n.begins_with(".")) {
			_scan_dir(da->get_current_dir().plus_file(n), r_projects);
		} else if (n == "project.godot") {
			r_projects->push_back(da->get_current_dir());
		}
		n = da->get_next();
	}
	da->list_dir_end();
	memdelete(da);
}

void ProjectManager::_scan_begin(const String &p_base) {

	print_line("Scanning projects at: " + p_base);
	List<String> projects;
	_scan_dir(p_base, &projects);
	print_line("Found " + itos(projects.size()) + " projects.");

	for (List<String>::Element *E = projects.front(); E; E = E->next()) {
		String proj = get_project_key_from_path(E->get());
		EditorSettings::get_singleton()->set("projects/" + proj, E->get());
	}
	EditorSettings::get_singleton()->save();
	_load_recent_projects();
}

void ProjectManager::_scan_projects() {

	scan_dir->popup_centered_ratio();
}

void ProjectManager::_new_project() {

	npdialog->set_mode(ProjectDialog::MODE_NEW);
	npdialog->show_dialog();
}

void ProjectManager::_import_project() {

	npdialog->set_mode(ProjectDialog::MODE_IMPORT);
	npdialog->show_dialog();
}

void ProjectManager::_rename_project() {

	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0) {
		return;
	}

	for (Set<String>::Element *E = selected_list.front(); E; E = E->next()) {
		const String &selected = E->get();
		String path = EditorSettings::get_singleton()->get("projects/" + selected);
		npdialog->set_project_path(path);
		npdialog->set_mode(ProjectDialog::MODE_RENAME);
		npdialog->show_dialog();
	}
}

void ProjectManager::_erase_project_confirm() {
	_project_list->erase_selected_projects();
	_update_project_buttons();
}

void ProjectManager::_erase_missing_projects_confirm() {
	_project_list->erase_missing_projects();
	_update_project_buttons();
}

void ProjectManager::_erase_project() {

	const Set<String> &selected_list = _project_list->get_selected_project_keys();

	if (selected_list.size() == 0)
		return;

	String confirm_message;
	if (selected_list.size() >= 2) {
		confirm_message = vformat(TTR("Remove %d projects from the list?\nThe project folders' contents won't be modified."), selected_list.size());
	} else {
		confirm_message = TTR("Remove this project from the list?\nThe project folder's contents won't be modified.");
	}

	erase_ask->set_text(confirm_message);
	erase_ask->popup_centered_minsize();
}

void ProjectManager::_erase_missing_projects() {

	erase_missing_ask->set_text(TTR("Remove all missing projects from the list?\nThe project folders' contents won't be modified."));
	erase_missing_ask->popup_centered_minsize();
}

void ProjectManager::_language_selected(int p_id) {

	String lang = language_btn->get_item_metadata(p_id);
	EditorSettings::get_singleton()->set("interface/editor/editor_language", lang);
	language_btn->set_text(lang);
	language_btn->set_icon(get_icon("Environment", "EditorIcons"));

	language_restart_ask->set_text(TTR("Language changed.\nThe interface will update after restarting the editor or project manager."));
	language_restart_ask->popup_centered();
}

void ProjectManager::_restart_confirm() {

	List<String> args = OS::get_singleton()->get_cmdline_args();
	String exec = OS::get_singleton()->get_executable_path();
	OS::ProcessID pid = 0;
	Error err = OS::get_singleton()->execute(exec, args, false, &pid);
	ERR_FAIL_COND(err);

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_exit_dialog() {

	_dim_window();
	get_tree()->quit();
}

void ProjectManager::_install_project(const String &p_zip_path, const String &p_title) {

	npdialog->set_mode(ProjectDialog::MODE_INSTALL);
	npdialog->set_zip_path(p_zip_path);
	npdialog->set_zip_title(p_title);
	npdialog->show_dialog();
}

void ProjectManager::_files_dropped(PoolStringArray p_files, int p_screen) {
	Set<String> folders_set;
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	for (int i = 0; i < p_files.size(); i++) {
		String file = p_files[i];
		folders_set.insert(da->dir_exists(file) ? file : file.get_base_dir());
	}
	memdelete(da);
	if (folders_set.size() > 0) {
		PoolStringArray folders;
		for (Set<String>::Element *E = folders_set.front(); E; E = E->next()) {
			folders.append(E->get());
		}

		bool confirm = true;
		if (folders.size() == 1) {
			DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			if (dir->change_dir(folders[0]) == OK) {
				dir->list_dir_begin();
				String file = dir->get_next();
				while (confirm && file != String()) {
					if (!dir->current_is_dir() && file.ends_with("project.godot")) {
						confirm = false;
					}
					file = dir->get_next();
				}
				dir->list_dir_end();
			}
			memdelete(dir);
		}
		if (confirm) {
			multi_scan_ask->get_ok()->disconnect("pressed", this, "_scan_multiple_folders");
			multi_scan_ask->get_ok()->connect("pressed", this, "_scan_multiple_folders", varray(folders));
			multi_scan_ask->set_text(
					vformat(TTR("Are you sure to scan %s folders for existing Godot projects?\nThis could take a while."), folders.size()));
			multi_scan_ask->popup_centered_minsize();
		} else {
			_scan_multiple_folders(folders);
		}
	}
}

void ProjectManager::_scan_multiple_folders(PoolStringArray p_files) {
	for (int i = 0; i < p_files.size(); i++) {
		_scan_begin(p_files.get(i));
	}
}

void ProjectManager::_on_order_option_changed() {
	_project_list->set_order_option(project_order_filter->get_filter_option());
	_project_list->sort_projects();
}

void ProjectManager::_on_filter_option_changed() {
	_project_list->set_search_term(project_filter->get_search_term());
	_project_list->sort_projects();
}

void ProjectManager::_bind_methods() {

	ClassDB::bind_method("_open_selected_projects_ask", &ProjectManager::_open_selected_projects_ask);
	ClassDB::bind_method("_open_selected_projects", &ProjectManager::_open_selected_projects);
	ClassDB::bind_method(D_METHOD("_global_menu_action"), &ProjectManager::_global_menu_action, DEFVAL(Variant()));
	ClassDB::bind_method("_run_project", &ProjectManager::_run_project);
	ClassDB::bind_method("_run_project_confirm", &ProjectManager::_run_project_confirm);
	ClassDB::bind_method("_scan_projects", &ProjectManager::_scan_projects);
	ClassDB::bind_method("_scan_begin", &ProjectManager::_scan_begin);
	ClassDB::bind_method("_import_project", &ProjectManager::_import_project);
	ClassDB::bind_method("_new_project", &ProjectManager::_new_project);
	ClassDB::bind_method("_rename_project", &ProjectManager::_rename_project);
	ClassDB::bind_method("_erase_project", &ProjectManager::_erase_project);
	ClassDB::bind_method("_erase_missing_projects", &ProjectManager::_erase_missing_projects);
	ClassDB::bind_method("_erase_project_confirm", &ProjectManager::_erase_project_confirm);
	ClassDB::bind_method("_erase_missing_projects_confirm", &ProjectManager::_erase_missing_projects_confirm);
	ClassDB::bind_method("_language_selected", &ProjectManager::_language_selected);
	ClassDB::bind_method("_restart_confirm", &ProjectManager::_restart_confirm);
	ClassDB::bind_method("_exit_dialog", &ProjectManager::_exit_dialog);
	ClassDB::bind_method("_on_order_option_changed", &ProjectManager::_on_order_option_changed);
	ClassDB::bind_method("_on_filter_option_changed", &ProjectManager::_on_filter_option_changed);
	ClassDB::bind_method("_on_projects_updated", &ProjectManager::_on_projects_updated);
	ClassDB::bind_method("_on_project_created", &ProjectManager::_on_project_created);
	ClassDB::bind_method("_unhandled_input", &ProjectManager::_unhandled_input);
	ClassDB::bind_method("_install_project", &ProjectManager::_install_project);
	ClassDB::bind_method("_files_dropped", &ProjectManager::_files_dropped);
	ClassDB::bind_method("_open_asset_library", &ProjectManager::_open_asset_library);
	ClassDB::bind_method("_confirm_update_settings", &ProjectManager::_confirm_update_settings);
	ClassDB::bind_method("_update_project_buttons", &ProjectManager::_update_project_buttons);
	ClassDB::bind_method(D_METHOD("_scan_multiple_folders", "files"), &ProjectManager::_scan_multiple_folders);
}

void ProjectManager::_open_asset_library() {
	asset_library->disable_community_support();
	tabs->set_current_tab(1);
}

ProjectManager::ProjectManager() {

	// load settings
	if (!EditorSettings::get_singleton())
		EditorSettings::create();

	EditorSettings::get_singleton()->set_optimize_save(false); //just write settings as they came

	{
		int display_scale = EditorSettings::get_singleton()->get("interface/editor/display_scale");
		float custom_display_scale = EditorSettings::get_singleton()->get("interface/editor/custom_display_scale");

		switch (display_scale) {
			case 0: {
				// Try applying a suitable display scale automatically.
#ifdef OSX_ENABLED
				editor_set_scale(OS::get_singleton()->get_screen_max_scale());
#else
				const int screen = OS::get_singleton()->get_current_screen();
				float scale;
				if (OS::get_singleton()->get_screen_dpi(screen) >= 192 && OS::get_singleton()->get_screen_size(screen).y >= 1400) {
					// hiDPI display.
					scale = 2.0;
				} else if (OS::get_singleton()->get_screen_size(screen).y <= 800) {
					// Small loDPI display. Use a smaller display scale so that editor elements fit more easily.
					// Icons won't look great, but this is better than having editor elements overflow from its window.
					scale = 0.75;
				} else {
					scale = 1.0;
				}

				editor_set_scale(scale);
#endif
			} break;

			case 1:
				editor_set_scale(0.75);
				break;
			case 2:
				editor_set_scale(1.0);
				break;
			case 3:
				editor_set_scale(1.25);
				break;
			case 4:
				editor_set_scale(1.5);
				break;
			case 5:
				editor_set_scale(1.75);
				break;
			case 6:
				editor_set_scale(2.0);
				break;
			default:
				editor_set_scale(custom_display_scale);
				break;
		}

		// Define a minimum window size to prevent UI elements from overlapping or being cut off
		OS::get_singleton()->set_min_window_size(Size2(750, 420) * EDSCALE);

		// TODO: Resize windows on hiDPI displays on Windows and Linux and remove the line below
		OS::get_singleton()->set_window_size(OS::get_singleton()->get_window_size() * MAX(1, EDSCALE));
	}

	FileDialog::set_default_show_hidden_files(EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files"));

	set_anchors_and_margins_preset(Control::PRESET_WIDE);
	set_theme(create_custom_theme());

	gui_base = memnew(Control);
	add_child(gui_base);
	gui_base->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	Panel *panel = memnew(Panel);
	gui_base->add_child(panel);
	panel->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	panel->add_style_override("panel", gui_base->get_stylebox("Background", "EditorStyles"));

	VBoxContainer *vb = memnew(VBoxContainer);
	panel->add_child(vb);
	vb->set_anchors_and_margins_preset(Control::PRESET_WIDE, Control::PRESET_MODE_MINSIZE, 8 * EDSCALE);

	String cp;
	cp += 0xA9;
	// TRANSLATORS: This refers to the application where users manage their Godot projects.
	OS::get_singleton()->set_window_title(VERSION_NAME + String(" - ") + TTR("Project Manager") + " - " + cp + " 2007-2020 Juan Linietsky, Ariel Manzur & Godot Contributors");

	Control *center_box = memnew(Control);
	center_box->set_v_size_flags(SIZE_EXPAND_FILL);
	vb->add_child(center_box);

	tabs = memnew(TabContainer);
	center_box->add_child(tabs);
	tabs->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	tabs->set_tab_align(TabContainer::ALIGN_LEFT);

	HBoxContainer *tree_hb = memnew(HBoxContainer);
	projects_hb = tree_hb;

	projects_hb->set_name(TTR("Projects"));

	tabs->add_child(tree_hb);

	VBoxContainer *search_tree_vb = memnew(VBoxContainer);
	tree_hb->add_child(search_tree_vb);
	search_tree_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *sort_filters = memnew(HBoxContainer);
	Label *sort_label = memnew(Label);
	sort_label->set_text(TTR("Sort:"));
	sort_filters->add_child(sort_label);
	Vector<String> sort_filter_titles;
	sort_filter_titles.push_back(TTR("Name"));
	sort_filter_titles.push_back(TTR("Path"));
	sort_filter_titles.push_back(TTR("Last Modified"));
	project_order_filter = memnew(ProjectListFilter);
	project_order_filter->add_filter_option();
	project_order_filter->_setup_filters(sort_filter_titles);
	project_order_filter->set_filter_size(150);
	sort_filters->add_child(project_order_filter);
	project_order_filter->connect("filter_changed", this, "_on_order_option_changed");
	project_order_filter->set_custom_minimum_size(Size2(180, 10) * EDSCALE);

	int projects_sorting_order = (int)EditorSettings::get_singleton()->get("project_manager/sorting_order");
	project_order_filter->set_filter_option((ProjectListFilter::FilterOption)projects_sorting_order);

	sort_filters->add_spacer(true);

	project_filter = memnew(ProjectListFilter);
	project_filter->add_search_box();
	project_filter->connect("filter_changed", this, "_on_filter_option_changed");
	project_filter->set_custom_minimum_size(Size2(280, 10) * EDSCALE);
	sort_filters->add_child(project_filter);

	search_tree_vb->add_child(sort_filters);

	PanelContainer *pc = memnew(PanelContainer);
	pc->add_style_override("panel", gui_base->get_stylebox("bg", "Tree"));
	search_tree_vb->add_child(pc);
	pc->set_v_size_flags(SIZE_EXPAND_FILL);

	_project_list = memnew(ProjectList);
	_project_list->connect(ProjectList::SIGNAL_SELECTION_CHANGED, this, "_update_project_buttons");
	_project_list->connect(ProjectList::SIGNAL_PROJECT_ASK_OPEN, this, "_open_selected_projects_ask");
	pc->add_child(_project_list);
	_project_list->set_enable_h_scroll(false);

	VBoxContainer *tree_vb = memnew(VBoxContainer);
	tree_vb->set_custom_minimum_size(Size2(120, 120));
	tree_hb->add_child(tree_vb);

	Button *open = memnew(Button);
	open->set_text(TTR("Edit"));
	tree_vb->add_child(open);
	open->connect("pressed", this, "_open_selected_projects_ask");
	open_btn = open;

	Button *run = memnew(Button);
	run->set_text(TTR("Run"));
	tree_vb->add_child(run);
	run->connect("pressed", this, "_run_project");
	run_btn = run;

	tree_vb->add_child(memnew(HSeparator));

	Button *scan = memnew(Button);
	scan->set_text(TTR("Scan"));
	tree_vb->add_child(scan);
	scan->connect("pressed", this, "_scan_projects");

	tree_vb->add_child(memnew(HSeparator));

	scan_dir = memnew(FileDialog);
	scan_dir->set_access(FileDialog::ACCESS_FILESYSTEM);
	scan_dir->set_mode(FileDialog::MODE_OPEN_DIR);
	scan_dir->set_title(TTR("Select a Folder to Scan")); // must be after mode or it's overridden
	scan_dir->set_current_dir(EditorSettings::get_singleton()->get("filesystem/directories/default_project_path"));
	gui_base->add_child(scan_dir);
	scan_dir->connect("dir_selected", this, "_scan_begin");

	Button *create = memnew(Button);
	create->set_text(TTR("New Project"));
	tree_vb->add_child(create);
	create->connect("pressed", this, "_new_project");

	Button *import = memnew(Button);
	import->set_text(TTR("Import"));
	tree_vb->add_child(import);
	import->connect("pressed", this, "_import_project");

	Button *rename = memnew(Button);
	rename->set_text(TTR("Rename"));
	tree_vb->add_child(rename);
	rename->connect("pressed", this, "_rename_project");
	rename_btn = rename;

	Button *erase = memnew(Button);
	erase->set_text(TTR("Remove"));
	tree_vb->add_child(erase);
	erase->connect("pressed", this, "_erase_project");
	erase_btn = erase;

	Button *erase_missing = memnew(Button);
	erase_missing->set_text(TTR("Remove Missing"));
	tree_vb->add_child(erase_missing);
	erase_missing->connect("pressed", this, "_erase_missing_projects");
	erase_missing_btn = erase_missing;

	tree_vb->add_spacer();

	if (StreamPeerSSL::is_available()) {
		asset_library = memnew(EditorAssetLibrary(true));
		asset_library->set_name(TTR("Templates"));
		tabs->add_child(asset_library);
		asset_library->connect("install_asset", this, "_install_project");
	} else {
		WARN_PRINT("Asset Library not available, as it requires SSL to work.");
	}

	HBoxContainer *settings_hb = memnew(HBoxContainer);
	settings_hb->set_alignment(BoxContainer::ALIGN_END);
	settings_hb->set_h_grow_direction(Control::GROW_DIRECTION_BEGIN);

	Label *version_label = memnew(Label);
	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = "." + hash.left(9);
	}
	version_label->set_text("v" VERSION_FULL_BUILD "" + hash);
	// Fade out the version label to be less prominent, but still readable
	version_label->set_self_modulate(Color(1, 1, 1, 0.6));
	version_label->set_align(Label::ALIGN_CENTER);
	settings_hb->add_child(version_label);

	language_btn = memnew(OptionButton);
	language_btn->set_flat(true);
	language_btn->set_focus_mode(Control::FOCUS_NONE);

	Vector<String> editor_languages;
	List<PropertyInfo> editor_settings_properties;
	EditorSettings::get_singleton()->get_property_list(&editor_settings_properties);
	for (List<PropertyInfo>::Element *E = editor_settings_properties.front(); E; E = E->next()) {
		PropertyInfo &pi = E->get();
		if (pi.name == "interface/editor/editor_language") {
			editor_languages = pi.hint_string.split(",");
		}
	}
	String current_lang = EditorSettings::get_singleton()->get("interface/editor/editor_language");
	for (int i = 0; i < editor_languages.size(); i++) {
		String lang = editor_languages[i];
		String lang_name = TranslationServer::get_singleton()->get_locale_name(lang);
		language_btn->add_item(lang_name + " [" + lang + "]", i);
		language_btn->set_item_metadata(i, lang);
		if (current_lang == lang) {
			language_btn->select(i);
			language_btn->set_text(lang);
		}
	}
	language_btn->set_icon(get_icon("Environment", "EditorIcons"));

	settings_hb->add_child(language_btn);
	language_btn->connect("item_selected", this, "_language_selected");

	center_box->add_child(settings_hb);
	settings_hb->set_anchors_and_margins_preset(Control::PRESET_TOP_RIGHT);

	//////////////////////////////////////////////////////////////

	language_restart_ask = memnew(ConfirmationDialog);
	language_restart_ask->get_ok()->set_text(TTR("Restart Now"));
	language_restart_ask->get_ok()->connect("pressed", this, "_restart_confirm");
	language_restart_ask->get_cancel()->set_text(TTR("Continue"));
	gui_base->add_child(language_restart_ask);

	erase_missing_ask = memnew(ConfirmationDialog);
	erase_missing_ask->get_ok()->set_text(TTR("Remove All"));
	erase_missing_ask->get_ok()->connect("pressed", this, "_erase_missing_projects_confirm");
	gui_base->add_child(erase_missing_ask);

	erase_ask = memnew(ConfirmationDialog);
	erase_ask->get_ok()->set_text(TTR("Remove"));
	erase_ask->get_ok()->connect("pressed", this, "_erase_project_confirm");
	gui_base->add_child(erase_ask);

	multi_open_ask = memnew(ConfirmationDialog);
	multi_open_ask->get_ok()->set_text(TTR("Edit"));
	multi_open_ask->get_ok()->connect("pressed", this, "_open_selected_projects");
	gui_base->add_child(multi_open_ask);

	multi_run_ask = memnew(ConfirmationDialog);
	multi_run_ask->get_ok()->set_text(TTR("Run"));
	multi_run_ask->get_ok()->connect("pressed", this, "_run_project_confirm");
	gui_base->add_child(multi_run_ask);

	multi_scan_ask = memnew(ConfirmationDialog);
	multi_scan_ask->get_ok()->set_text(TTR("Scan"));
	gui_base->add_child(multi_scan_ask);

	ask_update_settings = memnew(ConfirmationDialog);
	ask_update_settings->get_ok()->connect("pressed", this, "_confirm_update_settings");
	gui_base->add_child(ask_update_settings);

	OS::get_singleton()->set_low_processor_usage_mode(true);

	npdialog = memnew(ProjectDialog);
	gui_base->add_child(npdialog);

	npdialog->connect("projects_updated", this, "_on_projects_updated");
	npdialog->connect("project_created", this, "_on_project_created");

	_load_recent_projects();

	if (EditorSettings::get_singleton()->get("filesystem/directories/autoscan_project_path")) {
		_scan_begin(EditorSettings::get_singleton()->get("filesystem/directories/autoscan_project_path"));
	}

	SceneTree::get_singleton()->connect("files_dropped", this, "_files_dropped");
	SceneTree::get_singleton()->connect("global_menu_action", this, "_global_menu_action");

	run_error_diag = memnew(AcceptDialog);
	gui_base->add_child(run_error_diag);
	run_error_diag->set_title(TTR("Can't run project"));

	dialog_error = memnew(AcceptDialog);
	gui_base->add_child(dialog_error);

	open_templates = memnew(ConfirmationDialog);
	open_templates->set_text(TTR("You currently don't have any projects.\nWould you like to explore official example projects in the Asset Library?"));
	open_templates->get_ok()->set_text(TTR("Open Asset Library"));
	open_templates->connect("confirmed", this, "_open_asset_library");
	add_child(open_templates);
}

ProjectManager::~ProjectManager() {

	if (EditorSettings::get_singleton())
		EditorSettings::destroy();
}

void ProjectListFilter::_setup_filters(Vector<String> options) {

	filter_option->clear();
	for (int i = 0; i < options.size(); i++)
		filter_option->add_item(options[i]);
}

void ProjectListFilter::_search_text_changed(const String &p_newtext) {
	emit_signal("filter_changed");
}

String ProjectListFilter::get_search_term() {
	return search_box->get_text().strip_edges();
}

ProjectListFilter::FilterOption ProjectListFilter::get_filter_option() {
	return _current_filter;
}

void ProjectListFilter::set_filter_option(FilterOption option) {
	filter_option->select((int)option);
	_filter_option_selected(0);
}

void ProjectListFilter::_filter_option_selected(int p_idx) {
	FilterOption selected = (FilterOption)(filter_option->get_selected());
	if (_current_filter != selected) {
		_current_filter = selected;
		emit_signal("filter_changed");
	}
}

void ProjectListFilter::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE && has_search_box) {
		search_box->set_right_icon(get_icon("Search", "EditorIcons"));
		search_box->set_clear_button_enabled(true);
	}
}

void ProjectListFilter::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_search_text_changed"), &ProjectListFilter::_search_text_changed);
	ClassDB::bind_method(D_METHOD("_filter_option_selected"), &ProjectListFilter::_filter_option_selected);

	ADD_SIGNAL(MethodInfo("filter_changed"));
}

void ProjectListFilter::add_filter_option() {
	filter_option = memnew(OptionButton);
	filter_option->set_clip_text(true);
	filter_option->connect("item_selected", this, "_filter_option_selected");
	add_child(filter_option);
}

void ProjectListFilter::add_search_box() {
	search_box = memnew(LineEdit);
	search_box->set_placeholder(TTR("Search"));
	search_box->set_tooltip(
			TTR("The search box filters projects by name and last path component.\nTo filter projects by name and full path, the query must contain at least one `/` character."));
	search_box->connect("text_changed", this, "_search_text_changed");
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(search_box);

	has_search_box = true;
}

void ProjectListFilter::set_filter_size(int h_size) {
	filter_option->set_custom_minimum_size(Size2(h_size * EDSCALE, 10 * EDSCALE));
}

ProjectListFilter::ProjectListFilter() {

	_current_filter = FILTER_NAME;
	has_search_box = false;
}

void ProjectListFilter::clear() {
	if (has_search_box) {
		search_box->clear();
	}
}
