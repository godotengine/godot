/**************************************************************************/
/*  editor_dir_dialog.cpp                                                 */
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

#include "editor_dir_dialog.h"

#include "core/object/callable_mp.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/directory_create_dialog.h"
#include "editor/gui/editor_validation_panel.h"
#include "editor/themes/editor_theme_manager.h"
#include "scene/gui/box_container.h"
#include "scene/gui/foldable_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"
#include "servers/display/display_server.h"

void EditorDirDialog::_on_dir_picker_folding_changed(bool p_is_folded) {
	if (p_is_folded) {
		set_size(Size2i(get_size().x, 0));
	}
}

void EditorDirDialog::_update_dir(const Color &p_default_folder_color, const Dictionary &p_assigned_folder_colors, const HashMap<String, Color> &p_folder_colors, bool p_is_dark_icon_and_font, TreeItem *p_item, EditorFileSystemDirectory *p_dir, const String &p_select_path) {
	updating = true;

	const String path = p_dir->get_path();

	p_item->set_metadata(0, path);
	p_item->set_icon(0, tree->get_editor_theme_icon(SNAME("Folder")));

	if (!p_item->get_parent()) {
		p_item->set_text(0, "res://");
		p_item->set_icon_modulate(0, p_default_folder_color);
	} else {
		if (!opened_paths.has(path) && (p_select_path.is_empty() || !p_select_path.begins_with(path))) {
			p_item->set_collapsed(true);
		}

		p_item->set_text(0, p_dir->get_name());

		if (p_assigned_folder_colors.has(path)) {
			const Color &folder_color = p_folder_colors[p_assigned_folder_colors[path]];
			p_item->set_icon_modulate(0, p_is_dark_icon_and_font ? folder_color : folder_color * FileSystemDock::ITEM_COLOR_SCALE);
			p_item->set_custom_bg_color(0, Color(folder_color, p_is_dark_icon_and_font ? FileSystemDock::ITEM_ALPHA_MIN : FileSystemDock::ITEM_ALPHA_MAX));
		} else {
			TreeItem *parent_item = p_item->get_parent();
			Color parent_bg_color = parent_item->get_custom_bg_color(0);
			if (parent_bg_color != Color()) {
				p_item->set_custom_bg_color(0, p_assigned_folder_colors.has(parent_item->get_metadata(0)) ? parent_bg_color.darkened(FileSystemDock::ITEM_BG_DARK_SCALE) : parent_bg_color);
				p_item->set_icon_modulate(0, parent_item->get_icon_modulate(0));
			} else {
				p_item->set_icon_modulate(0, p_default_folder_color);
			}
		}
	}

	if (path == new_dir_path || !p_item->get_parent()) {
		p_item->select(0);
	}

	updating = false;
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		TreeItem *ti = tree->create_item(p_item);
		_update_dir(p_default_folder_color, p_assigned_folder_colors, p_folder_colors, p_is_dark_icon_and_font, ti, p_dir->get_subdir(i));
	}
}

void EditorDirDialog::_validate_name() {
	const bool should_validate = name_hb->is_visible();
	validation_panel->set_visible(should_validate);
	bool has_error = false;
	if (should_validate) {
		String path = name->get_text();
		path = path.strip_edges();
		if (name_is_folder) {
			path = path.trim_suffix("/");
		}

		const String error = _validate_path(path);

		has_error = !error.is_empty();

		if (has_error) {
			validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, error, EditorValidationPanel::MSG_ERROR);
		} else {
			if (path.contains_char('/')) {
				if (name_is_folder) {
					validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, TTRC("Using slashes in folder names will create subfolders recursively."), EditorValidationPanel::MSG_OK);
				} else {
					validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, TTRC("Using slashes in path will create the file in subfolder, creating new subfolders if necessary."), EditorValidationPanel::MSG_OK);
				}
			} else {
				validation_panel->set_message(EditorValidationPanel::MSG_ID_DEFAULT, name_is_folder ? TTRC("Folder name is valid.") : TTRC("File name is valid."), EditorValidationPanel::MSG_OK);
			}
		}
	}

	get_ok_button()->set_disabled(has_error);
	copy->set_disabled(has_error);
}

String EditorDirDialog::_validate_path(const String &p_path) {
	if (p_path.is_empty()) {
		return TTR("Name cannot be empty.");
	}
	if (!name_is_folder && p_path.ends_with("/")) {
		return TTR("File name can't end with /.");
	}

	const PackedStringArray splits = p_path.split("/");
	for (int i = 0; i < splits.size(); i++) {
		const String &part = splits[i];
		bool is_file = (!name_is_folder) && i == splits.size() - 1;

		if (part.is_empty()) {
			if (is_file) {
				return TTR("File name cannot be empty.");
			} else {
				return TTR("Folder name cannot be empty.");
			}
		}
		if (part.contains_char('\\') || part.contains_char(':') || part.contains_char('*') ||
				part.contains_char('|') || part.contains_char('>') || part.ends_with(".") || part.ends_with(" ")) {
			if (is_file) {
				return TTR("File name contains invalid characters.");
			} else {
				return TTR("Folder name contains invalid characters.");
			}
		}
		if (part[0] == '.') {
			if (is_file) {
				return TTR("File name begins with a dot.");
			} else {
				return TTR("Folder name begins with a dot.");
			}
		}
	}

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	da->change_dir(base_directory_path);
	if (da->file_exists(p_path)) {
		return TTR("File with that name already exists.");
	}
	if (da->dir_exists(p_path)) {
		return TTR("Folder with that name already exists.");
	}

	return String();
}

void EditorDirDialog::config(const Vector<String> &p_paths, bool p_from_duplicate) {
	ERR_FAIL_COND(p_paths.is_empty());

	if (p_paths.size() == 1) {
		String path = p_paths[0];
		name_is_folder = path.ends_with("/");
		if (name_is_folder) {
			path = path.substr(0, path.length() - 1);
		}
		name->set_text(path.get_file());
		name_hb->show();
		// TRANSLATORS: %s is the file name that will be moved or duplicated.
		String new_title = TTR("Move/Duplicate: %s");
		if (p_from_duplicate) {
			new_title = name_is_folder ? TTR("Duplicating folder:") : TTR("Duplicating file:");
			new_title += " %s";
		}
		set_title(vformat(new_title, path.get_file()));
	} else {
		// TRANSLATORS: %d is the number of files that will be moved or duplicated.
		set_title(vformat(TTRN("Move/Duplicate %d Item", "Move/Duplicate %d Items", p_paths.size()), p_paths.size()));
		name_hb->hide();
	}
	base_directory_path = p_paths[0].get_base_dir();
	selected_path->set_title(TTR("Destination: ") + base_directory_path);
	selected_path->set_folded(p_from_duplicate);
	get_ok_button()->set_visible(!p_from_duplicate);
	copy->set_text(p_from_duplicate ? TTRC("Duplicate") : TTRC("Copy"));
	set_size(Size2i(get_size().x, 0));
}

void EditorDirDialog::reload(const String &p_path) {
	if (!is_visible()) {
		must_reload = true;
		return;
	}

	tree->clear();
	TreeItem *root = tree->create_item();
	_update_dir(tree->get_theme_color(SNAME("folder_icon_color"), SNAME("FileDialog")), FileSystemDock::get_singleton()->get_assigned_folder_colors(), FileSystemDock::get_singleton()->get_folder_colors(), EditorThemeManager::is_dark_icon_and_font(), root, EditorFileSystem::get_singleton()->get_filesystem(), p_path);
	_item_collapsed(root);
	new_dir_path.clear();
	must_reload = false;
}

void EditorDirDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			FileSystemDock::get_singleton()->connect("folder_color_changed", callable_mp(this, &EditorDirDialog::reload).bind(""));
			EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload).bind(""));
			reload();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			EditorFileSystem::get_singleton()->disconnect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload));
			FileSystemDock::get_singleton()->disconnect("folder_color_changed", callable_mp(this, &EditorDirDialog::reload));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				if (must_reload) {
					reload();
				}

				tree->deselect_all();

				TreeItem *item = tree->get_root();
				const PackedStringArray parts = base_directory_path.trim_prefix("res://").split("/", false);
				if (parts.is_empty()) {
					item->select(0);
					tree->ensure_cursor_is_visible();
					break;
				}

				int i = 0;
				while (i < parts.size()) {
					const String &folder = parts[i];
					for (TreeItem *child = item->get_first_child(); child; child = child->get_next()) {
						if (child->get_text(0) == folder) {
							i++;
							if (i == parts.size()) {
								TreeItem *parent_item = child->get_parent();
								while (parent_item) {
									parent_item->set_collapsed(false);
									parent_item = parent_item->get_parent();
								}
								child->select(0);
								tree->ensure_cursor_is_visible();
							} else {
								item = child;
							}
							break;
						}
					}
				}
			}
		} break;
	}
}

void EditorDirDialog::_item_collapsed(Object *p_item) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);

	if (updating) {
		return;
	}

	if (item->is_collapsed()) {
		opened_paths.erase(item->get_metadata(0));
	} else {
		opened_paths.insert(item->get_metadata(0));
	}
}

void EditorDirDialog::_item_activated() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);
	if (ti->get_child_count() > 0) {
		ti->set_collapsed(!ti->is_collapsed());
	}
}

void EditorDirDialog::_item_selected() {
	if (!updating) {
		if (TreeItem *ti = tree->get_selected()) {
			base_directory_path = ti->get_metadata(0);
			selected_path->set_title(TTR("Destination: ") + base_directory_path);
			_validate_name();
		}
	}
}

void EditorDirDialog::_copy_pressed() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);

	hide();
	String new_name = name->is_visible() ? name->get_text() : String();
	emit_signal(SNAME("copy_pressed"), ti->get_metadata(0), new_name);
}

void EditorDirDialog::ok_pressed() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);

	hide();
	String new_name = name->is_visible() ? name->get_text() : String();
	emit_signal(SNAME("move_pressed"), ti->get_metadata(0), new_name);
}

void EditorDirDialog::_make_dir() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);
	const String &directory = ti->get_metadata(0);
	makedialog->config(directory, callable_mp(this, &EditorDirDialog::_make_dir_confirm).bind(directory), DirectoryCreateDialog::MODE_DIRECTORY, TTR("Create Folder"), "new folder");
	makedialog->popup_centered();
}

void EditorDirDialog::_make_dir_confirm(const String &p_path, const String &p_base_dir) {
	FileSystemDock::get_singleton()->create_directory(p_path, p_base_dir);

	// Multiple level of directories can be created at once.
	String base_dir = p_path.get_base_dir();
	while (true) {
		opened_paths.insert(base_dir + "/");
		if (base_dir == "res://") {
			break;
		}
		base_dir = base_dir.get_base_dir();
	}

	new_dir_path = p_path + "/";
}

void EditorDirDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("copy_pressed", PropertyInfo(Variant::STRING, "dir"), PropertyInfo(Variant::STRING, "rename_or_empty")));
	ADD_SIGNAL(MethodInfo("move_pressed", PropertyInfo(Variant::STRING, "dir"), PropertyInfo(Variant::STRING, "rename_or_empty")));
}

EditorDirDialog::EditorDirDialog() {
	set_hide_on_ok(false);
	set_min_size(Size2i(350, 0));

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	selected_path = memnew(FoldableContainer);
	selected_path->connect(SNAME("folding_changed"), callable_mp(this, &EditorDirDialog::_on_dir_picker_folding_changed));
	selected_path->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(selected_path);

	VBoxContainer *fvb = memnew(VBoxContainer);
	fvb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	selected_path->add_child(fvb);

	tree = memnew(Tree);
	tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_custom_minimum_size(Size2(0, 350));
	fvb->add_child(tree);
	tree->connect("item_activated", callable_mp(this, &EditorDirDialog::_item_activated));
	tree->connect("item_collapsed", callable_mp(this, &EditorDirDialog::_item_collapsed), CONNECT_DEFERRED);
	tree->connect("item_selected", callable_mp(this, &EditorDirDialog::_item_selected));

	makedir = memnew(Button(TTR("Create Folder")));
	makedir->set_h_size_flags(Control::SIZE_SHRINK_END);
	fvb->add_child(makedir);
	makedir->connect(SceneStringName(pressed), callable_mp(this, &EditorDirDialog::_make_dir));

	name_hb = memnew(HBoxContainer);
	vb->add_child(name_hb);

	name_hb->add_child(memnew(Label(TTR("Name:"))));

	name = memnew(LineEdit);
	name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name->connect(SceneStringName(text_changed), callable_mp(this, &EditorDirDialog::_validate_name).unbind(1));
	name_hb->add_child(name);

	validation_panel = memnew(EditorValidationPanel);
	validation_panel->add_line(EditorValidationPanel::MSG_ID_DEFAULT, "");
	validation_panel->set_update_callback(callable_mp(this, &EditorDirDialog::_validate_name));
	validation_panel->set_v_size_flags(Control::SIZE_FILL);
	vb->add_child(validation_panel);

	set_ok_button_text(TTRC("Move"));

	copy = add_button(TTRC("Copy"), !DisplayServer::get_singleton()->get_swap_cancel_ok());
	copy->connect(SceneStringName(pressed), callable_mp(this, &EditorDirDialog::_copy_pressed));

	makedialog = memnew(DirectoryCreateDialog);
	add_child(makedialog);
}
