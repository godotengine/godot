/**************************************************************************/
/*  dependency_editor.cpp                                                 */
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

#include "dependency_editor.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_quick_open_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/tree.h"

static void _setup_search_file_dialog(EditorFileDialog *p_dialog, const String &p_file, const String &p_type) {
	p_dialog->set_title(vformat(TTR("Search Replacement For: %s"), p_file.get_file()));

	// Set directory to closest existing directory.
	p_dialog->set_current_dir(p_file.get_base_dir());

	p_dialog->clear_filters();
	List<String> ext;
	ResourceLoader::get_recognized_extensions_for_type(p_type, &ext);
	for (const String &E : ext) {
		p_dialog->add_filter("*." + E);
	}
}

void DependencyEditor::_searched(const String &p_path) {
	HashMap<String, String> dep_rename;
	dep_rename[replacing] = p_path;

	ResourceLoader::rename_dependencies(editing, dep_rename);

	_update_list();
	_update_file();
}

void DependencyEditor::_load_pressed(Object *p_item, int p_cell, int p_button, MouseButton p_mouse_button) {
	if (p_mouse_button != MouseButton::LEFT) {
		return;
	}
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	replacing = ti->get_text(1);

	_setup_search_file_dialog(search, replacing, ti->get_metadata(0));
	search->popup_file_dialog();
}

void DependencyEditor::_fix_and_find(EditorFileSystemDirectory *efsd, HashMap<String, HashMap<String, String>> &candidates) {
	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_fix_and_find(efsd->get_subdir(i), candidates);
	}

	for (int i = 0; i < efsd->get_file_count(); i++) {
		String file = efsd->get_file(i);
		if (!candidates.has(file)) {
			continue;
		}

		String path = efsd->get_file_path(i);

		for (KeyValue<String, String> &E : candidates[file]) {
			if (E.value.is_empty()) {
				E.value = path;
				continue;
			}

			//must match the best, using subdirs
			String existing = E.value.replace_first("res://", "");
			String current = path.replace_first("res://", "");
			String lost = E.key.replace_first("res://", "");

			Vector<String> existingv = existing.split("/");
			existingv.reverse();
			Vector<String> currentv = current.split("/");
			currentv.reverse();
			Vector<String> lostv = lost.split("/");
			lostv.reverse();

			int existing_score = 0;
			int current_score = 0;

			for (int j = 0; j < lostv.size(); j++) {
				if (j < existingv.size() && lostv[j] == existingv[j]) {
					existing_score++;
				}
				if (j < currentv.size() && lostv[j] == currentv[j]) {
					current_score++;
				}
			}

			if (current_score > existing_score) {
				//if it was the same, could track distance to new path but..

				E.value = path; //replace by more accurate
			}
		}
	}
}

void DependencyEditor::_fix_all() {
	if (!EditorFileSystem::get_singleton()->get_filesystem()) {
		return;
	}

	HashMap<String, HashMap<String, String>> candidates;

	for (const String &E : missing) {
		String base = E.get_file();
		if (!candidates.has(base)) {
			candidates[base] = HashMap<String, String>();
		}

		candidates[base][E] = "";
	}

	_fix_and_find(EditorFileSystem::get_singleton()->get_filesystem(), candidates);

	HashMap<String, String> remaps;

	for (KeyValue<String, HashMap<String, String>> &E : candidates) {
		for (const KeyValue<String, String> &F : E.value) {
			if (!F.value.is_empty()) {
				remaps[F.key] = F.value;
			}
		}
	}

	if (remaps.size()) {
		ResourceLoader::rename_dependencies(editing, remaps);

		_update_list();
		_update_file();
	}
}

void DependencyEditor::_update_file() {
	EditorFileSystem::get_singleton()->update_file(editing);
}

void DependencyEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		warning_label->add_theme_color_override(SceneStringName(font_color), get_theme_color("warning_color", EditorStringName(Editor)));
	}
}

static String _get_resolved_dep_path(const String &p_dep) {
	if (p_dep.get_slice_count("::") < 3) {
		return p_dep.get_slice("::", 0); // No UID, just return the path.
	}

	const String uid_text = p_dep.get_slice("::", 0);
	ResourceUID::ID uid = ResourceUID::get_singleton()->text_to_id(uid_text);

	// Dependency is in UID format, obtain proper path.
	if (uid != ResourceUID::INVALID_ID && ResourceUID::get_singleton()->has_id(uid)) {
		return ResourceUID::get_singleton()->get_id_path(uid);
	}

	// UID fallback path.
	return p_dep.get_slice("::", 2);
}

static String _get_stored_dep_path(const String &p_dep) {
	if (p_dep.get_slice_count("::") > 2) {
		return p_dep.get_slice("::", 2);
	}
	return p_dep.get_slice("::", 0);
}

void DependencyEditor::_update_list() {
	List<String> deps;
	ResourceLoader::get_dependencies(editing, &deps, true);

	tree->clear();
	missing.clear();

	TreeItem *root = tree->create_item();

	Ref<Texture2D> folder = tree->get_theme_icon(SNAME("folder"), SNAME("FileDialog"));

	bool broken = false;

	for (const String &dep : deps) {
		TreeItem *item = tree->create_item(root);

		const String path = _get_resolved_dep_path(dep);
		if (FileAccess::exists(path)) {
			item->set_text(0, path.get_file());
			item->set_text(1, path);
		} else {
			const String &stored_path = _get_stored_dep_path(dep);
			item->set_text(0, stored_path.get_file());
			item->set_text(1, stored_path);
			item->set_custom_color(1, Color(1, 0.4, 0.3));
			missing.push_back(stored_path);
			broken = true;
		}

		const String type = dep.contains("::") ? dep.get_slice("::", 1) : "Resource";
		Ref<Texture2D> icon = EditorNode::get_singleton()->get_class_icon(type);
		item->set_icon(0, icon);
		item->set_metadata(0, type);

		item->add_button(1, folder, 0);
	}

	fixdeps->set_disabled(!broken);
}

void DependencyEditor::edit(const String &p_path) {
	editing = p_path;
	set_title(TTR("Dependencies For:") + " " + p_path.get_file());

	_update_list();

	if (EditorNode::get_singleton()->is_scene_open(p_path)) {
		warning_label->show();
		warning_label->set_text(vformat(TTR("Scene \"%s\" is currently being edited. Changes will only take effect when reloaded."), p_path.get_file()));
	} else if (ResourceCache::has(p_path)) {
		warning_label->show();
		warning_label->set_text(vformat(TTR("Resource \"%s\" is in use. Changes will only take effect when reloaded."), p_path.get_file()));
	} else {
		warning_label->hide();
	}
	popup_centered_ratio(0.4);
}

DependencyEditor::DependencyEditor() {
	VBoxContainer *vb = memnew(VBoxContainer);
	vb->set_name(TTR("Dependencies"));
	add_child(vb);

	tree = memnew(Tree);
	tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tree->set_columns(2);
	tree->set_column_titles_visible(true);
	tree->set_column_title(0, TTR("Resource"));
	tree->set_column_clip_content(0, true);
	tree->set_column_title(1, TTR("Path"));
	tree->set_column_clip_content(1, true);
	tree->set_hide_root(true);
	tree->connect("button_clicked", callable_mp(this, &DependencyEditor::_load_pressed));

	HBoxContainer *hbc = memnew(HBoxContainer);
	Label *label = memnew(Label(TTR("Dependencies:")));
	label->set_theme_type_variation("HeaderSmall");

	hbc->add_child(label);
	hbc->add_spacer();
	fixdeps = memnew(Button(TTR("Fix Broken")));
	hbc->add_child(fixdeps);
	fixdeps->connect(SceneStringName(pressed), callable_mp(this, &DependencyEditor::_fix_all));

	vb->add_child(hbc);

	MarginContainer *mc = memnew(MarginContainer);
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	mc->add_child(tree);
	vb->add_child(mc);

	warning_label = memnew(Label);
	warning_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
	vb->add_child(warning_label);

	set_title(TTR("Dependency Editor"));
	search = memnew(EditorFileDialog);
	search->connect("file_selected", callable_mp(this, &DependencyEditor::_searched));
	search->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(search);
}

/////////////////////////////////////
void DependencyEditorOwners::_list_rmb_clicked(int p_item, const Vector2 &p_pos, MouseButton p_mouse_button_index) {
	if (p_mouse_button_index != MouseButton::RIGHT) {
		return;
	}

	file_options->clear();
	file_options->reset_size();
	if (p_item >= 0) {
		PackedInt32Array selected_items = owners->get_selected_items();
		bool only_scenes_selected = true;

		for (int i = 0; i < selected_items.size(); i++) {
			int item_idx = selected_items[i];
			if (ResourceLoader::get_resource_type(owners->get_item_text(item_idx)) != "PackedScene") {
				only_scenes_selected = false;
				break;
			}
		}

		if (only_scenes_selected) {
			file_options->add_icon_item(get_editor_theme_icon(SNAME("Load")), TTRN("Open Scene", "Open Scenes", selected_items.size()), FILE_MENU_OPEN);
		} else if (selected_items.size() == 1) {
			file_options->add_icon_item(get_editor_theme_icon(SNAME("Load")), TTR("Open"), FILE_MENU_OPEN);
		} else {
			return;
		}
	}

	file_options->set_position(owners->get_screen_position() + p_pos);
	file_options->reset_size();
	file_options->popup();
}

void DependencyEditorOwners::_select_file(int p_idx) {
	String fpath = owners->get_item_text(p_idx);
	EditorNode::get_singleton()->load_scene_or_resource(fpath);

	hide();
	emit_signal(SceneStringName(confirmed));
}

void DependencyEditorOwners::_empty_clicked(const Vector2 &p_pos, MouseButton p_mouse_button_index) {
	if (p_mouse_button_index != MouseButton::LEFT) {
		return;
	}

	owners->deselect_all();
}

void DependencyEditorOwners::_file_option(int p_option) {
	switch (p_option) {
		case FILE_MENU_OPEN: {
			PackedInt32Array selected_items = owners->get_selected_items();
			for (int i = 0; i < selected_items.size(); i++) {
				int item_idx = selected_items[i];
				if (item_idx < 0 || item_idx >= owners->get_item_count()) {
					break;
				}
				_select_file(item_idx);
			}
		} break;
	}
}

void DependencyEditorOwners::_fill_owners(EditorFileSystemDirectory *efsd) {
	if (!efsd) {
		return;
	}

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_fill_owners(efsd->get_subdir(i));
	}

	for (int i = 0; i < efsd->get_file_count(); i++) {
		Vector<String> deps = efsd->get_file_deps(i);
		bool found = false;
		for (int j = 0; j < deps.size(); j++) {
			if (deps[j] == editing) {
				found = true;
				break;
			}
		}
		if (!found) {
			continue;
		}

		Ref<Texture2D> icon = EditorNode::get_singleton()->get_class_icon(efsd->get_file_type(i));

		owners->add_item(efsd->get_file_path(i), icon);
	}
}

void DependencyEditorOwners::show(const String &p_path) {
	editing = p_path;
	owners->clear();
	_fill_owners(EditorFileSystem::get_singleton()->get_filesystem());

	int count = owners->get_item_count();
	if (count > 0) {
		empty->hide();
		owners_count->set_text(vformat(TTR("Owners of: %s (Total: %d)"), p_path.get_file(), count));
		owners_count->show();
		owners_mc->show();
	} else {
		owners_count->hide();
		owners_mc->hide();
		empty->set_text(vformat(TTR("No owners found for: %s"), p_path.get_file()));
		empty->show();
	}

	popup_centered_ratio(0.3);
}

DependencyEditorOwners::DependencyEditorOwners() {
	file_options = memnew(PopupMenu);
	add_child(file_options);
	file_options->connect(SceneStringName(id_pressed), callable_mp(this, &DependencyEditorOwners::_file_option));

	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	owners_count = memnew(Label);
	owners_count->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	owners_count->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	owners_count->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
	vbox->add_child(owners_count);

	empty = memnew(Label);
	empty->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	empty->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	empty->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	empty->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	empty->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
	empty->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	empty->hide();
	vbox->add_child(empty);

	owners_mc = memnew(MarginContainer);
	owners_mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	owners_mc->set_theme_type_variation("NoBorderHorizontalWindow");
	vbox->add_child(owners_mc);

	owners = memnew(ItemList);
	owners->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	owners->set_select_mode(ItemList::SELECT_MULTI);
	owners->set_scroll_hint_mode(ItemList::SCROLL_HINT_MODE_BOTH);
	owners->connect("item_clicked", callable_mp(this, &DependencyEditorOwners::_list_rmb_clicked));
	owners->connect("item_activated", callable_mp(this, &DependencyEditorOwners::_select_file));
	owners->connect("empty_clicked", callable_mp(this, &DependencyEditorOwners::_empty_clicked));
	owners->set_allow_rmb_select(true);
	owners_mc->add_child(owners);

	set_title(TTRC("Owners List"));
}

///////////////////////

void DependencyRemoveDialog::_find_files_in_removed_folder(EditorFileSystemDirectory *efsd, const String &p_folder) {
	if (!efsd) {
		return;
	}

	for (int i = 0; i < efsd->get_subdir_count(); ++i) {
		_find_files_in_removed_folder(efsd->get_subdir(i), p_folder);
	}
	for (int i = 0; i < efsd->get_file_count(); i++) {
		String file = efsd->get_file_path(i);
		ERR_FAIL_COND(all_remove_files.has(file)); //We are deleting a directory which is contained in a directory we are deleting...
		all_remove_files[file] = p_folder; //Point the file to the ancestor directory we are deleting so we know what to parent it under in the tree.
	}
}

void DependencyRemoveDialog::_find_all_removed_dependencies(EditorFileSystemDirectory *efsd, Vector<RemovedDependency> &p_removed) {
	if (!efsd) {
		return;
	}

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		_find_all_removed_dependencies(efsd->get_subdir(i), p_removed);
	}

	for (int i = 0; i < efsd->get_file_count(); i++) {
		const String path = efsd->get_file_path(i);

		//It doesn't matter if a file we are about to delete will have some of its dependencies removed too
		if (all_remove_files.has(path)) {
			continue;
		}

		Vector<String> all_deps = efsd->get_file_deps(i);
		for (int j = 0; j < all_deps.size(); ++j) {
			if (all_remove_files.has(all_deps[j])) {
				RemovedDependency dep;
				dep.file = path;
				dep.file_type = efsd->get_file_type(i);
				dep.dependency = all_deps[j];
				dep.dependency_folder = all_remove_files[all_deps[j]];
				p_removed.push_back(dep);
			}
		}
	}
}

void DependencyRemoveDialog::_find_localization_remaps_of_removed_files(Vector<RemovedDependency> &p_removed) {
	for (KeyValue<String, String> &files : all_remove_files) {
		const String &path = files.key;

		// Look for dependencies in the translation remaps.
		if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
			Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");

			if (remaps.has(path)) {
				RemovedDependency dep;
				dep.file = TTR("Localization remap");
				dep.file_type = "";
				dep.dependency = path;
				dep.dependency_folder = files.value;
				p_removed.push_back(dep);
			}

			for (const KeyValue<Variant, Variant> &remap_kv : remaps) {
				PackedStringArray remapped_files = remap_kv.value;
				for (const String &remapped_file : remapped_files) {
					int splitter_pos = remapped_file.rfind_char(':');
					String res_path = remapped_file.substr(0, splitter_pos);
					if (res_path == path) {
						String locale_name = remapped_file.substr(splitter_pos + 1);

						RemovedDependency dep;
						dep.file = vformat(TTR("Localization remap for path '%s' and locale '%s'."), remap_kv.key, locale_name);
						dep.file_type = "";
						dep.dependency = path;
						dep.dependency_folder = files.value;
						p_removed.push_back(dep);
					}
				}
			}
		}
	}
}

void DependencyRemoveDialog::_build_removed_dependency_tree(const Vector<RemovedDependency> &p_removed) {
	owners->clear();
	owners->create_item(); // root

	HashMap<String, TreeItem *> tree_items;
	for (int i = 0; i < p_removed.size(); i++) {
		RemovedDependency rd = p_removed[i];

		//Ensure that the dependency is already in the tree
		if (!tree_items.has(rd.dependency)) {
			if (rd.dependency_folder.length() > 0) {
				//Ensure the ancestor folder is already in the tree
				if (!tree_items.has(rd.dependency_folder)) {
					TreeItem *folder_item = owners->create_item(owners->get_root());
					folder_item->set_text(0, rd.dependency_folder);
					folder_item->set_icon(0, owners->get_editor_theme_icon(SNAME("Folder")));
					tree_items[rd.dependency_folder] = folder_item;
				}
				TreeItem *dependency_item = owners->create_item(tree_items[rd.dependency_folder]);
				dependency_item->set_text(0, rd.dependency);
				dependency_item->set_icon(0, owners->get_editor_theme_icon(SNAME("Warning")));
				tree_items[rd.dependency] = dependency_item;
			} else {
				TreeItem *dependency_item = owners->create_item(owners->get_root());
				dependency_item->set_text(0, rd.dependency);
				dependency_item->set_icon(0, owners->get_editor_theme_icon(SNAME("Warning")));
				tree_items[rd.dependency] = dependency_item;
			}
		}

		//List this file under this dependency
		Ref<Texture2D> icon = EditorNode::get_singleton()->get_class_icon(rd.file_type);
		TreeItem *file_item = owners->create_item(tree_items[rd.dependency]);
		file_item->set_text(0, rd.file);
		file_item->set_icon(0, icon);
	}
}

void DependencyRemoveDialog::_show_files_to_delete_list() {
	files_to_delete_list->clear();

	for (const String &s : dirs_to_delete) {
		String t = s.trim_prefix("res://");
		files_to_delete_list->add_item(t, Ref<Texture2D>(), false);
	}

	for (const String &s : files_to_delete) {
		String t = s.trim_prefix("res://");
		files_to_delete_list->add_item(t, Ref<Texture2D>(), false);
	}
}

void DependencyRemoveDialog::show(const Vector<String> &p_folders, const Vector<String> &p_files) {
	all_remove_files.clear();
	dirs_to_delete.clear();
	files_to_delete.clear();
	owners->clear();

	for (int i = 0; i < p_folders.size(); ++i) {
		String folder = p_folders[i].ends_with("/") ? p_folders[i] : (p_folders[i] + "/");
		_find_files_in_removed_folder(EditorFileSystem::get_singleton()->get_filesystem_path(folder), folder);
		dirs_to_delete.push_back(folder);
	}
	for (int i = 0; i < p_files.size(); ++i) {
		all_remove_files[p_files[i]] = String();
		files_to_delete.push_back(p_files[i]);
	}

	_show_files_to_delete_list();

	Vector<RemovedDependency> removed_deps;
	_find_all_removed_dependencies(EditorFileSystem::get_singleton()->get_filesystem(), removed_deps);
	_find_localization_remaps_of_removed_files(removed_deps);
	removed_deps.sort();
	if (removed_deps.is_empty()) {
		vb_owners->hide();
		text->set_text(TTR("Remove the selected files from the project? (Cannot be undone.)\nDepending on your filesystem configuration, the files will either be moved to the system trash or deleted permanently."));
		reset_size();
		popup_centered();
	} else {
		_build_removed_dependency_tree(removed_deps);
		vb_owners->show();
		text->set_text(TTR("The files being removed are required by other resources in order for them to work.\nRemove them anyway? (Cannot be undone.)\nDepending on your filesystem configuration, the files will either be moved to the system trash or deleted permanently."));
		popup_centered(Size2(500, 350));
	}

	EditorFileSystem::get_singleton()->scan_changes();
}

void DependencyRemoveDialog::ok_pressed() {
	HashMap<String, StringName> setting_path_map;
	for (const StringName &setting : path_project_settings) {
		const String path = ResourceUID::ensure_path(GLOBAL_GET(setting));
		setting_path_map[path] = setting;
	}

	bool project_settings_modified = false;

	for (const KeyValue<String, String> &E : all_remove_files) {
		String file = E.key;

		if (ResourceCache::has(file)) {
			Ref<Resource> res = ResourceCache::get_ref(file);
			emit_signal(SNAME("resource_removed"), res);
			res->set_path("");
		}

		// If the file we are deleting for e.g. the main scene, default environment,
		// or audio bus layout, we must clear its definition in Project Settings.
		const StringName *setting_name = setting_path_map.getptr(file);
		if (setting_name) {
			ProjectSettings::get_singleton()->set(*setting_name, "");
			project_settings_modified = true;
		}
	}

	for (const String &file : files_to_delete) {
		const String path = OS::get_singleton()->get_resource_dir() + file.replace_first("res://", "/");
		print_verbose("Moving to trash: " + path);
		Error err = OS::get_singleton()->move_to_trash(path);
		if (err != OK) {
			EditorNode::get_singleton()->add_io_error(TTR("Cannot remove:") + "\n" + file + "\n");
		} else {
			emit_signal(SNAME("file_removed"), file);
		}
	}
	if (project_settings_modified) {
		ProjectSettings::get_singleton()->save();
	}

	if (dirs_to_delete.is_empty()) {
		// If we only deleted files we should only need to tell the file system about the files we touched.
		for (int i = 0; i < files_to_delete.size(); ++i) {
			EditorFileSystem::get_singleton()->update_file(files_to_delete[i]);
		}
	} else {
		for (int i = 0; i < dirs_to_delete.size(); ++i) {
			String path = OS::get_singleton()->get_resource_dir() + dirs_to_delete[i].replace_first("res://", "/");
			print_verbose("Moving to trash: " + path);
			Error err = OS::get_singleton()->move_to_trash(path);
			if (err != OK) {
				EditorNode::get_singleton()->add_io_error(TTR("Cannot remove:") + "\n" + dirs_to_delete[i] + "\n");
			} else {
				emit_signal(SNAME("folder_removed"), dirs_to_delete[i]);
			}
		}

		EditorFileSystem::get_singleton()->scan_changes();
	}

	// If some files/dirs would be deleted, favorite dirs need to be updated
	Vector<String> previous_favorites = EditorSettings::get_singleton()->get_favorites();
	Vector<String> new_favorites;

	for (int i = 0; i < previous_favorites.size(); ++i) {
		if (previous_favorites[i].ends_with("/")) {
			if (!dirs_to_delete.has(previous_favorites[i])) {
				new_favorites.push_back(previous_favorites[i]);
			}
		} else {
			if (!files_to_delete.has(previous_favorites[i])) {
				new_favorites.push_back(previous_favorites[i]);
			}
		}
	}

	if (new_favorites.size() < previous_favorites.size()) {
		EditorSettings::get_singleton()->set_favorites(new_favorites);
	}
}

void DependencyRemoveDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("resource_removed", PropertyInfo(Variant::OBJECT, "obj")));
	ADD_SIGNAL(MethodInfo("file_removed", PropertyInfo(Variant::STRING, "file")));
	ADD_SIGNAL(MethodInfo("folder_removed", PropertyInfo(Variant::STRING, "folder")));
}

DependencyRemoveDialog::DependencyRemoveDialog() {
	set_ok_button_text(TTR("Remove"));

	VBoxContainer *vb = memnew(VBoxContainer);
	vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(vb);

	text = memnew(Label);
	text->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	vb->add_child(text);

	Label *files_to_delete_label = memnew(Label);
	files_to_delete_label->set_theme_type_variation("HeaderSmall");
	files_to_delete_label->set_text(TTR("Files to be deleted:"));
	vb->add_child(files_to_delete_label);

	MarginContainer *mc = memnew(MarginContainer);
	mc->set_theme_type_variation("NoBorderHorizontalWindow");
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(mc);

	files_to_delete_list = memnew(ItemList);
	files_to_delete_list->set_scroll_hint_mode(ItemList::SCROLL_HINT_MODE_BOTH);
	files_to_delete_list->set_custom_minimum_size(Size2(0, 94) * EDSCALE);
	files_to_delete_list->set_accessibility_name(TTRC("Files to be deleted:"));
	mc->add_child(files_to_delete_list);

	vb_owners = memnew(VBoxContainer);
	vb_owners->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vb_owners->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(vb_owners);

	Label *owners_label = memnew(Label);
	owners_label->set_theme_type_variation("HeaderSmall");
	owners_label->set_text(TTR("Dependencies of files to be deleted:"));
	vb_owners->add_child(owners_label);

	mc = memnew(MarginContainer);
	mc->set_theme_type_variation("NoBorderHorizontalWindow");
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb_owners->add_child(mc);

	owners = memnew(Tree);
	owners->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	owners->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
	owners->set_hide_root(true);
	owners->set_custom_minimum_size(Size2(0, 94) * EDSCALE);
	owners->set_accessibility_name(TTRC("Dependencies"));
	mc->add_child(owners);
	owners->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	List<PropertyInfo> property_list;
	ProjectSettings::get_singleton()->get_property_list(&property_list);
	for (const PropertyInfo &pi : property_list) {
		if (pi.type == Variant::STRING && pi.hint == PROPERTY_HINT_FILE) {
			path_project_settings.push_back(pi.name);
		}
	}
}

//////////////
enum {
	BUTTON_ID_SEARCH,
	BUTTON_ID_OPEN_DEPS_EDITOR,
};

void DependencyErrorDialog::show(const String &p_for_file, const HashMap<String, HashSet<String>> &p_report) {
	for_file = p_for_file;

	// TRANSLATORS: The placeholder is a filename.
	set_title(vformat(TTR("Error loading: %s"), p_for_file.get_file()));

	HashMap<String, HashSet<String>> missing_to_owners;
	for (const KeyValue<String, HashSet<String>> &E : p_report) {
		for (const String &missing : E.value) {
			missing_to_owners[missing].insert(E.key);
		}
	}

	files->clear();
	TreeItem *root = files->create_item(nullptr);
	Ref<Texture2D> folder_icon = get_theme_icon(SNAME("folder"), SNAME("FileDialog"));

	for (const KeyValue<String, HashSet<String>> &E : missing_to_owners) {
		const String &missing_path = E.key.get_slice("::", 0);
		const String &missing_type = E.key.get_slice("::", 1);

		TreeItem *missing_ti = root->create_child();
		missing_ti->set_text(0, missing_path);
		missing_ti->set_metadata(0, E.key);
		missing_ti->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_DISABLED);
		missing_ti->set_icon(0, EditorNode::get_singleton()->get_class_icon(missing_type));
		missing_ti->set_icon(1, get_editor_theme_icon(icon_name_fail));
		missing_ti->add_button(1, folder_icon, BUTTON_ID_SEARCH, false, TTRC("Search"));
		missing_ti->set_collapsed(true);

		for (const String &owner_path : E.value) {
			TreeItem *owner_ti = missing_ti->create_child();
			// TRANSLATORS: The placeholder is a file path.
			owner_ti->set_text(0, vformat(TTR("Referenced by %s"), owner_path));
			owner_ti->set_metadata(0, owner_path);
			owner_ti->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_DISABLED);
			owner_ti->add_button(1, files->get_editor_theme_icon(SNAME("Edit")), BUTTON_ID_OPEN_DEPS_EDITOR, false, TTRC("Fix Dependencies"));
		}
	}

	set_ok_button_text(TTRC("Open Anyway"));
	popup_centered();
}

void DependencyErrorDialog::ok_pressed() {
	EditorNode::get_singleton()->load_scene_or_resource(for_file, !errors_fixed);
}

void DependencyErrorDialog::_on_files_button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_button) {
	switch (p_id) {
		case BUTTON_ID_SEARCH: {
			const String &meta = p_item->get_metadata(0);
			const String &missing_path = meta.get_slice("::", 0);
			const String &missing_type = meta.get_slice("::", 1);
			if (replacement_file_dialog == nullptr) {
				replacement_file_dialog = memnew(EditorFileDialog);
				replacement_file_dialog->connect("file_selected", callable_mp(this, &DependencyErrorDialog::_on_replacement_file_selected));
				replacement_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
				add_child(replacement_file_dialog);
			}
			replacing_item = p_item;
			_setup_search_file_dialog(replacement_file_dialog, missing_path, missing_type);
			replacement_file_dialog->popup_file_dialog();
		} break;

		case BUTTON_ID_OPEN_DEPS_EDITOR: {
			const String &owner_path = p_item->get_metadata(0);
			if (deps_editor == nullptr) {
				deps_editor = memnew(DependencyEditor);
				deps_editor->connect(SceneStringName(visibility_changed), callable_mp(this, &DependencyErrorDialog::_check_for_resolved));
				add_child(deps_editor);
			}
			deps_editor->edit(owner_path);
		} break;
	}
}

void DependencyErrorDialog::_on_replacement_file_selected(const String &p_path) {
	const String &missing_path = String(replacing_item->get_metadata(0)).get_slice("::", 0);

	for (TreeItem *owner_ti = replacing_item->get_first_child(); owner_ti; owner_ti = owner_ti->get_next()) {
		const String &owner_path = owner_ti->get_metadata(0);
		ResourceLoader::rename_dependencies(owner_path, { { missing_path, p_path } });
	}

	_check_for_resolved();
}

void DependencyErrorDialog::_check_for_resolved() {
	if (deps_editor && deps_editor->is_visible()) {
		return; // Only update when the dialog is closed.
	}

	errors_fixed = true;
	HashMap<String, LocalVector<String>> owner_deps;

	TreeItem *root = files->get_root();
	for (TreeItem *missing_ti = root->get_first_child(); missing_ti; missing_ti = missing_ti->get_next()) {
		bool all_owners_fixed = true;

		for (TreeItem *owner_ti = missing_ti->get_first_child(); owner_ti; owner_ti = owner_ti->get_next()) {
			const String &owner_path = owner_ti->get_metadata(0);

			if (!owner_deps.has(owner_path)) {
				List<String> deps;
				ResourceLoader::get_dependencies(owner_path, &deps);

				LocalVector<String> &stored_paths = owner_deps[owner_path];
				for (const String &dep : deps) {
					if (!errors_fixed && !FileAccess::exists(_get_resolved_dep_path(dep))) {
						errors_fixed = false;
					}
					stored_paths.push_back(_get_stored_dep_path(dep));
				}
			}
			const LocalVector<String> &stored_paths = owner_deps[owner_path];
			const String &missing_path = String(missing_ti->get_metadata(0)).get_slice("::", 0);

			if (stored_paths.has(missing_path)) {
				all_owners_fixed = false;
				break;
			}
		}

		missing_ti->set_icon(1, get_editor_theme_icon(all_owners_fixed ? icon_name_check : icon_name_fail));
	}

	set_ok_button_text(errors_fixed ? TTRC("Open") : TTRC("Open Anyway"));
}

DependencyErrorDialog::DependencyErrorDialog() {
	icon_name_fail = StringName("ImportFail");
	icon_name_check = StringName("ImportCheck");

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	files = memnew(Tree);
	files->set_hide_root(true);
	files->set_select_mode(Tree::SELECT_ROW);
	files->set_columns(2);
	files->set_column_expand(1, false);
	files->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	files->connect("button_clicked", callable_mp(this, &DependencyErrorDialog::_on_files_button_clicked));
	vb->add_margin_child(TTRC("Load failed due to missing dependencies:"), files, true);

	set_min_size(Size2(500, 320) * EDSCALE);
	set_cancel_button_text(TTRC("Close"));

	Label *text = memnew(Label(TTRC("Which action should be taken?")));
	text->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	vb->add_child(text);
}

//////////////////////////////////////////////////////////////////////

void OrphanResourcesDialog::ok_pressed() {
	paths.clear();

	_find_to_delete(files->get_root(), paths);
	if (paths.is_empty()) {
		return;
	}

	delete_confirm->set_text(vformat(TTR("Permanently delete %d item(s)? (No undo!)"), paths.size()));
	delete_confirm->popup_centered();
}

bool OrphanResourcesDialog::_fill_owners(EditorFileSystemDirectory *efsd, HashMap<String, int> &refs, TreeItem *p_parent) {
	if (!efsd) {
		return false;
	}

	bool has_children = false;

	for (int i = 0; i < efsd->get_subdir_count(); i++) {
		TreeItem *dir_item = nullptr;
		if (p_parent) {
			dir_item = files->create_item(p_parent);
			dir_item->set_text(0, efsd->get_subdir(i)->get_name());
			dir_item->set_icon(0, files->get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
		}
		bool children = _fill_owners(efsd->get_subdir(i), refs, dir_item);

		if (p_parent) {
			if (!children) {
				memdelete(dir_item);
			} else {
				has_children = true;
			}
		}
	}

	for (int i = 0; i < efsd->get_file_count(); i++) {
		if (!p_parent) {
			Vector<String> deps = efsd->get_file_deps(i);
			for (int j = 0; j < deps.size(); j++) {
				if (!refs.has(deps[j])) {
					refs[deps[j]] = 1;
				}
			}
		} else {
			String path = efsd->get_file_path(i);
			if (!refs.has(path)) {
				TreeItem *ti = files->create_item(p_parent);
				ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				ti->set_text(0, efsd->get_file(i));
				ti->set_editable(0, true);

				String type = efsd->get_file_type(i);

				Ref<Texture2D> icon = EditorNode::get_singleton()->get_class_icon(type);
				ti->set_icon(0, icon);
				int ds = efsd->get_file_deps(i).size();
				ti->set_text(1, itos(ds));
				if (ds) {
					ti->add_button(1, files->get_editor_theme_icon(SNAME("GuiVisibilityVisible")), -1, false, TTR("Show Dependencies"));
				}
				ti->set_metadata(0, path);
				has_children = true;
			}
		}
	}

	return has_children;
}

void OrphanResourcesDialog::refresh() {
	HashMap<String, int> refs;
	_fill_owners(EditorFileSystem::get_singleton()->get_filesystem(), refs, nullptr);
	files->clear();
	TreeItem *root = files->create_item();
	_fill_owners(EditorFileSystem::get_singleton()->get_filesystem(), refs, root);
}

void OrphanResourcesDialog::show() {
	refresh();
	popup_centered_ratio(0.4);
}

void OrphanResourcesDialog::_find_to_delete(TreeItem *p_item, List<String> &r_paths) {
	while (p_item) {
		if (p_item->get_cell_mode(0) == TreeItem::CELL_MODE_CHECK && p_item->is_checked(0)) {
			r_paths.push_back(p_item->get_metadata(0));
		}

		if (p_item->get_first_child()) {
			_find_to_delete(p_item->get_first_child(), r_paths);
		}

		p_item = p_item->get_next();
	}
}

void OrphanResourcesDialog::_delete_confirm() {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	for (const String &E : paths) {
		da->remove(E);
		EditorFileSystem::get_singleton()->update_file(E);
	}
	refresh();
}

void OrphanResourcesDialog::_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button) {
	if (p_button != MouseButton::LEFT) {
		return;
	}
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);

	String path = ti->get_metadata(0);
	dep_edit->edit(path);
}

OrphanResourcesDialog::OrphanResourcesDialog() {
	set_title(TTR("Orphan Resource Explorer"));
	delete_confirm = memnew(ConfirmationDialog);
	set_ok_button_text(TTR("Delete"));
	add_child(delete_confirm);
	dep_edit = memnew(DependencyEditor);
	add_child(dep_edit);
	delete_confirm->connect(SceneStringName(confirmed), callable_mp(this, &OrphanResourcesDialog::_delete_confirm));
	set_hide_on_ok(false);

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	files = memnew(Tree);
	files->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	files->set_columns(2);
	files->set_column_titles_visible(true);
	files->set_column_custom_minimum_width(1, 100 * EDSCALE);
	files->set_column_expand(0, true);
	files->set_column_clip_content(0, true);
	files->set_column_expand(1, false);
	files->set_column_clip_content(1, true);
	files->set_column_title(0, TTR("Resource"));
	files->set_column_title(1, TTR("Owns"));
	files->set_hide_root(true);
	files->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTTOM);
	files->connect("button_clicked", callable_mp(this, &OrphanResourcesDialog::_button_pressed));

	MarginContainer *mc = vbc->add_margin_child(TTRC("Resources Without Explicit Ownership:"), files, true);
	mc->set_theme_type_variation("NoBorderHorizontalWindow");
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
}
