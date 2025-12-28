/**************************************************************************/
/*  localization_editor.cpp                                               */
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

#include "localization_editor.h"

#include "core/config/project_settings.h"
#include "core/string/translation_server.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/translations/editor_translation_parser.h"
#include "editor/translations/template_generator.h"
#include "scene/gui/control.h"
#include "scene/gui/tab_container.h"

void LocalizationEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			translation_list->connect("button_clicked", callable_mp(this, &LocalizationEditor::_translation_delete));
			template_source_list->connect("button_clicked", callable_mp(this, &LocalizationEditor::_template_source_delete));
			template_add_builtin->set_pressed(GLOBAL_GET("internationalization/locale/translation_add_builtin_strings_to_pot"));

			List<String> tfn;
			ResourceLoader::get_recognized_extensions_for_type("Translation", &tfn);
			tfn.erase("csv"); // CSV is recognized by the resource importer to generate translation files, but it's not a translation file itself.
			for (const String &E : tfn) {
				translation_file_open->add_filter("*." + E);
			}

			List<String> rfn;
			ResourceLoader::get_recognized_extensions_for_type("Resource", &rfn);
			for (const String &E : rfn) {
				translation_res_file_open_dialog->add_filter("*." + E);
				translation_res_option_file_open_dialog->add_filter("*." + E);
			}

			_update_template_source_file_extensions();
			template_generate_dialog->add_filter("*.pot");
			template_generate_dialog->add_filter("*.csv");
		} break;

		case NOTIFICATION_DRAG_END: {
			for (Tree *tree : trees) {
				tree->set_drop_mode_flags(Tree::DROP_MODE_DISABLED);
			}
		} break;
	}
}

void LocalizationEditor::add_translation(const String &p_translation) {
	PackedStringArray translations;
	translations.push_back(p_translation);
	_translation_add(translations);
}

void LocalizationEditor::_translation_add(const PackedStringArray &p_paths) {
	PackedStringArray translations = GLOBAL_GET("internationalization/locale/translations");
	int count = 0;
	for (const String &path : p_paths) {
		if (!translations.has(path)) {
			// Don't add duplicate translation paths.
			translations.push_back(path);
			count += 1;
		}
	}
	if (count == 0) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTRN("Add %d Translation", "Add %d Translations", count), count));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", translations);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", GLOBAL_GET("internationalization/locale/translations"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_file_open() {
	translation_file_open->popup_file_dialog();
}

void LocalizationEditor::_translation_delete(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button) {
	if (p_mouse_button != MouseButton::LEFT) {
		return;
	}

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_NULL(ti);

	int idx = ti->get_metadata(0);

	PackedStringArray translations = GLOBAL_GET("internationalization/locale/translations");

	ERR_FAIL_INDEX(idx, translations.size());

	translations.remove_at(idx);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove Translation"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", translations);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", GLOBAL_GET("internationalization/locale/translations"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_res_file_open() {
	translation_res_file_open_dialog->popup_file_dialog();
}

void LocalizationEditor::_translation_res_add(const PackedStringArray &p_paths) {
	Variant prev;
	Dictionary remaps;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		remaps = GLOBAL_GET("internationalization/locale/translation_remaps");
		prev = remaps;
	}

	int count = 0;
	for (const String &path : p_paths) {
		if (!remaps.has(path)) {
			// Don't overwrite with an empty remap array if an array already exists for the given path.
			remaps[path] = PackedStringArray();
			count += 1;
		}
	}
	if (count == 0) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTRN("Translation Resource Remap: Add %d Path", "Translation Resource Remap: Add %d Paths", count), count));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", prev);
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_res_option_file_open() {
	translation_res_option_file_open_dialog->popup_file_dialog();
}

void LocalizationEditor::_translation_res_option_add(const PackedStringArray &p_paths) {
	ERR_FAIL_COND(!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps"));

	Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_NULL(k);

	String key = k->get_metadata(0);

	ERR_FAIL_COND(!remaps.has(key));
	PackedStringArray r = remaps[key];
	for (int i = 0; i < p_paths.size(); i++) {
		r.push_back(p_paths[i] + ":" + "en");
	}
	remaps[key] = r;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTRN("Translation Resource Remap: Add %d Remap", "Translation Resource Remap: Add %d Remaps", p_paths.size()), p_paths.size()));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", GLOBAL_GET("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_res_select() {
	if (updating_translations) {
		return;
	}
	callable_mp(this, &LocalizationEditor::update_translations).call_deferred();
}

void LocalizationEditor::_translation_res_option_popup(bool p_arrow_clicked) {
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_NULL(ed);

	locale_select->set_locale(ed->get_tooltip_text(1));
	locale_select->popup_locale_dialog();
}

void LocalizationEditor::_translation_res_option_selected(const String &p_locale) {
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_NULL(ed);

	ed->set_text(1, TranslationServer::get_singleton()->get_locale_name(p_locale));
	ed->set_tooltip_text(1, p_locale);

	LocalizationEditor::_translation_res_option_changed();
}

void LocalizationEditor::_translation_res_option_changed() {
	if (updating_translations) {
		return;
	}

	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_NULL(k);
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_NULL(ed);

	String key = k->get_metadata(0);
	int idx = ed->get_metadata(0);
	String path = ed->get_metadata(1);
	String locale = ed->get_tooltip_text(1);

	ERR_FAIL_COND(!remaps.has(key));
	PackedStringArray r = remaps[key];
	r.set(idx, path + ":" + locale);
	remaps[key] = r;

	updating_translations = true;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Change Resource Remap Language"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", GLOBAL_GET("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
	updating_translations = false;
}

void LocalizationEditor::_translation_res_delete(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button) {
	if (updating_translations) {
		return;
	}

	if (p_mouse_button != MouseButton::LEFT) {
		return;
	}

	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");

	TreeItem *k = Object::cast_to<TreeItem>(p_item);

	String key = k->get_metadata(0);
	ERR_FAIL_COND(!remaps.has(key));

	remaps.erase(key);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove Resource Remap"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", GLOBAL_GET("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_res_option_delete(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button) {
	if (updating_translations) {
		return;
	}

	if (p_mouse_button != MouseButton::LEFT) {
		return;
	}

	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_NULL(k);
	TreeItem *ed = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_NULL(ed);

	String key = k->get_metadata(0);
	int idx = ed->get_metadata(0);

	ERR_FAIL_COND(!remaps.has(key));
	PackedStringArray r = remaps[key];
	ERR_FAIL_INDEX(idx, r.size());
	r.remove_at(idx);
	remaps[key] = r;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove Resource Remap Option"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", GLOBAL_GET("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_template_source_add(const PackedStringArray &p_paths) {
	PackedStringArray sources = GLOBAL_GET("internationalization/locale/translations_pot_files");
	int count = 0;
	for (const String &path : p_paths) {
		if (!sources.has(path)) {
			sources.push_back(path);
			count += 1;
		}
	}
	if (count == 0) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTRN("Add %d file for template generation", "Add %d files for template generation", count), count));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", sources);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", GLOBAL_GET("internationalization/locale/translations_pot_files"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_template_source_delete(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button) {
	if (p_mouse_button != MouseButton::LEFT) {
		return;
	}

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_NULL(ti);

	int idx = ti->get_metadata(0);

	PackedStringArray sources = GLOBAL_GET("internationalization/locale/translations_pot_files");

	ERR_FAIL_INDEX(idx, sources.size());

	sources.remove_at(idx);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove file from template generation"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", sources);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", GLOBAL_GET("internationalization/locale/translations_pot_files"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_template_source_file_open() {
	template_source_open_dialog->popup_file_dialog();
}

void LocalizationEditor::_template_generate_open() {
	template_generate_dialog->popup_file_dialog();
}

void LocalizationEditor::_template_add_builtin_toggled() {
	ProjectSettings::get_singleton()->set_setting("internationalization/locale/translation_add_builtin_strings_to_pot", template_add_builtin->is_pressed());
	ProjectSettings::get_singleton()->save();
}

void LocalizationEditor::_template_generate(const String &p_file) {
	EditorSettings::get_singleton()->set_project_metadata("pot_generator", "last_pot_path", p_file);
	TranslationTemplateGenerator::get_singleton()->generate(p_file);
}

void LocalizationEditor::_update_template_source_file_extensions() {
	template_source_open_dialog->clear_filters();
	List<String> translation_parse_file_extensions;
	EditorTranslationParser::get_singleton()->get_recognized_extensions(&translation_parse_file_extensions);
	for (const String &E : translation_parse_file_extensions) {
		template_source_open_dialog->add_filter("*." + E);
	}
}

void LocalizationEditor::connect_filesystem_dock_signals(FileSystemDock *p_fs_dock) {
	p_fs_dock->connect("files_moved", callable_mp(this, &LocalizationEditor::_filesystem_files_moved));
	p_fs_dock->connect("file_removed", callable_mp(this, &LocalizationEditor::_filesystem_file_removed));
}

void LocalizationEditor::_filesystem_files_moved(const String &p_old_file, const String &p_new_file) {
	// Update source files if the moved file is a part of them.
	PackedStringArray sources = GLOBAL_GET("internationalization/locale/translations_pot_files");
	if (sources.has(p_old_file)) {
		sources.erase(p_old_file);
		ProjectSettings::get_singleton()->set_setting("internationalization/locale/translations_pot_files", sources);

		PackedStringArray new_file;
		new_file.push_back(p_new_file);
		_template_source_add(new_file);
	}

	// Update remaps if the moved file is a part of them.
	Dictionary remaps;
	bool remaps_changed = false;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		remaps = GLOBAL_GET("internationalization/locale/translation_remaps");
	}

	// Check for the keys.
	if (remaps.has(p_old_file)) {
		PackedStringArray remapped_files = remaps[p_old_file];
		remaps.erase(p_old_file);
		remaps[p_new_file] = remapped_files;
		remaps_changed = true;
		print_verbose(vformat("Changed remap key \"%s\" to \"%s\" due to a moved file.", p_old_file, p_new_file));
	}

	// Check for the Array elements of the values.
	Array remap_keys = remaps.keys();
	for (const Variant &remap_key : remap_keys) {
		PackedStringArray remapped_files = remaps[remap_key];
		bool remapped_files_updated = false;

		for (int j = 0; j < remapped_files.size(); j++) {
			int splitter_pos = remapped_files[j].rfind_char(':');
			String res_path = remapped_files[j].substr(0, splitter_pos);

			if (res_path == p_old_file) {
				String locale_name = remapped_files[j].substr(splitter_pos + 1);
				// Replace the element at that index.
				remapped_files.insert(j, p_new_file + ":" + locale_name);
				remapped_files.remove_at(j + 1);
				remaps_changed = true;
				remapped_files_updated = true;
				print_verbose(vformat("Changed remap value \"%s\" to \"%s\" of key \"%s\" due to a moved file.", res_path + ":" + locale_name, remapped_files[j], remap_key));
			}
		}

		if (remapped_files_updated) {
			remaps[remap_key] = remapped_files;
		}
	}

	if (remaps_changed) {
		ProjectSettings::get_singleton()->set_setting("internationalization/locale/translation_remaps", remaps);
		update_translations();
		emit_signal("localization_changed");
	}
}

void LocalizationEditor::_filesystem_file_removed(const String &p_file) {
	// Check if the source files are affected.
	PackedStringArray sources = GLOBAL_GET("internationalization/locale/translations_pot_files");
	if (sources.has(p_file)) {
		sources.erase(p_file);
		ProjectSettings::get_singleton()->set_setting("internationalization/locale/translations_pot_files", sources);
	}

	// Check if the remaps are affected.
	Dictionary remaps;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		remaps = GLOBAL_GET("internationalization/locale/translation_remaps");
	}

	bool remaps_changed = remaps.has(p_file);

	if (!remaps_changed) {
		Array remap_keys = remaps.keys();
		for (int i = 0; i < remap_keys.size() && !remaps_changed; i++) {
			PackedStringArray remapped_files = remaps[remap_keys[i]];
			for (int j = 0; j < remapped_files.size() && !remaps_changed; j++) {
				int splitter_pos = remapped_files[j].rfind_char(':');
				String res_path = remapped_files[j].substr(0, splitter_pos);
				remaps_changed = p_file == res_path;
				if (remaps_changed) {
					print_verbose(vformat("Remap value \"%s\" of key \"%s\" has been removed from the file system.", remapped_files[j], remap_keys[i]));
				}
			}
		}
	} else {
		print_verbose(vformat("Remap key \"%s\" has been removed from the file system.", p_file));
	}

	if (remaps_changed) {
		update_translations();
		emit_signal("localization_changed");
	}
}

Variant LocalizationEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	Tree *tree = Object::cast_to<Tree>(p_from);
	ERR_FAIL_COND_V(trees.find(tree) == -1, Variant());

	if (tree->get_button_id_at_position(p_point) != -1) {
		return Variant();
	}

	TreeItem *selected = tree->get_next_selected(nullptr);
	if (!selected) {
		return Variant();
	}
	tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);

	Label *preview = memnew(Label);
	preview->set_text(selected->get_text(0));
	set_drag_preview(preview);

	Dictionary drag_data;
	drag_data["type"] = tree_data_types[tree];
	drag_data["item"] = selected;

	return drag_data;
}

bool LocalizationEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Tree *tree = Object::cast_to<Tree>(p_from);
	ERR_FAIL_COND_V(trees.find(tree) == -1, false);

	Dictionary drop_data = p_data;
	return drop_data.get("type", "") == tree_data_types[tree];
}

void LocalizationEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	Tree *tree = Object::cast_to<Tree>(p_from);
	ERR_FAIL_COND(trees.find(tree) == -1);

	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	TreeItem *item = tree->get_item_at_position(p_point);
	if (!item) {
		return;
	}
	int section = MAX(tree->get_drop_section_at_position(p_point), 0);
	Dictionary drop_data = p_data;

	TreeItem *from = Object::cast_to<TreeItem>(drop_data["item"]);
	if (item == from) {
		return;
	}

	const StringName &setting = tree_settings[tree];
	PackedStringArray setting_value = GLOBAL_GET(setting);
	const PackedStringArray original_setting_value = setting_value;
	const int index_from = from->get_metadata(0);
	const String path = setting_value[index_from];

	int target_index = item->get_metadata(0);
	target_index = MAX(target_index + section, 0);
	if (target_index > index_from) {
		target_index -= 1; // Account for item being removed.
	}

	if (target_index == index_from) {
		return;
	}

	setting_value.remove_at(index_from);
	setting_value.insert(target_index, path);

	EditorUndoRedoManager *ur_man = EditorUndoRedoManager::get_singleton();
	ur_man->create_action(TTR("Rearrange Localization Items"));
	ur_man->add_do_method(ProjectSettings::get_singleton(), "set", setting, setting_value);
	ur_man->add_do_method(ProjectSettings::get_singleton(), "save");
	ur_man->add_do_method(this, "update_translations");
	ur_man->add_undo_method(ProjectSettings::get_singleton(), "set", setting, original_setting_value);
	ur_man->add_undo_method(ProjectSettings::get_singleton(), "save");
	ur_man->add_undo_method(this, "update_translations");
	ur_man->commit_action();
}

void LocalizationEditor::update_translations() {
	if (updating_translations) {
		return;
	}

	updating_translations = true;

	translation_list->clear();
	TreeItem *root = translation_list->create_item(nullptr);
	translation_list->set_hide_root(true);
	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translations")) {
		PackedStringArray translations = GLOBAL_GET("internationalization/locale/translations");
		for (int i = 0; i < translations.size(); i++) {
			TreeItem *t = translation_list->create_item(root);
			t->set_editable(0, false);
			t->set_text(0, translations[i].replace_first("res://", ""));
			t->set_tooltip_text(0, translations[i]);
			t->set_metadata(0, i);
			t->add_button(0, get_editor_theme_icon(SNAME("Remove")), 0, false, TTRC("Remove"));
		}
	}

	// Update translation remaps.
	String remap_selected;
	if (translation_remap->get_selected()) {
		remap_selected = translation_remap->get_selected()->get_metadata(0);
	}

	translation_remap->clear();
	translation_remap_options->clear();
	root = translation_remap->create_item(nullptr);
	TreeItem *root2 = translation_remap_options->create_item(nullptr);
	translation_remap->set_hide_root(true);
	translation_remap_options->set_hide_root(true);
	translation_res_option_add_button->set_disabled(true);

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		Dictionary remaps = GLOBAL_GET("internationalization/locale/translation_remaps");
		Vector<String> keys;
		for (const KeyValue<Variant, Variant> &kv : remaps) {
			keys.push_back(kv.key);
		}
		keys.sort();

		for (int i = 0; i < keys.size(); i++) {
			TreeItem *t = translation_remap->create_item(root);
			t->set_editable(0, false);
			t->set_text(0, keys[i].replace_first("res://", ""));
			t->set_tooltip_text(0, keys[i]);
			t->set_metadata(0, keys[i]);
			t->add_button(0, get_editor_theme_icon(SNAME("Remove")), 0, false, TTRC("Remove"));

			// Display that it has been removed if this is the case.
			if (!FileAccess::exists(keys[i])) {
				t->set_text(0, t->get_text(0) + vformat(" (%s)", TTR("Removed")));
				t->set_tooltip_text(0, vformat(TTR("%s cannot be found."), t->get_tooltip_text(0)));
			}

			if (keys[i] == remap_selected) {
				t->select(0);
				translation_res_option_add_button->set_disabled(false);

				PackedStringArray selected = remaps[keys[i]];
				for (int j = 0; j < selected.size(); j++) {
					const String &s2 = selected[j];
					int qp = s2.rfind_char(':');
					String path = s2.substr(0, qp);
					String locale = s2.substr(qp + 1);

					TreeItem *t2 = translation_remap_options->create_item(root2);
					t2->set_editable(0, false);
					t2->set_text(0, path.replace_first("res://", ""));
					t2->set_tooltip_text(0, path);
					t2->set_metadata(0, j);
					t2->add_button(0, get_editor_theme_icon(SNAME("Remove")), 0, false, TTRC("Remove"));
					t2->set_cell_mode(1, TreeItem::CELL_MODE_CUSTOM);
					t2->set_text(1, TranslationServer::get_singleton()->get_locale_name(locale));
					t2->set_editable(1, true);
					t2->set_metadata(1, path);
					t2->set_tooltip_text(1, locale);

					// Display that it has been removed if this is the case.
					if (!FileAccess::exists(path)) {
						t2->set_text(0, t2->get_text(0) + vformat(" (%s)", TTR("Removed")));
						t2->set_tooltip_text(0, vformat(TTR("%s cannot be found."), t2->get_tooltip_text(0)));
					}
				}
			}
		}
	}

	// Update translation source files.
	template_source_list->clear();
	root = template_source_list->create_item(nullptr);
	template_source_list->set_hide_root(true);
	PackedStringArray sources = GLOBAL_GET("internationalization/locale/translations_pot_files");
	for (int i = 0; i < sources.size(); i++) {
		TreeItem *t = template_source_list->create_item(root);
		t->set_editable(0, false);
		t->set_text(0, sources[i].replace_first("res://", ""));
		t->set_tooltip_text(0, sources[i]);
		t->set_metadata(0, i);
		t->add_button(0, get_editor_theme_icon(SNAME("Remove")), 0, false, TTRC("Remove"));
	}

	// New translation parser plugin might extend possible file extensions in template generation.
	_update_template_source_file_extensions();

	updating_translations = false;
}

void LocalizationEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("update_translations"), &LocalizationEditor::update_translations);

	ADD_SIGNAL(MethodInfo("localization_changed"));
}

LocalizationEditor::LocalizationEditor() {
	localization_changed = "localization_changed";

	TabContainer *translations = memnew(TabContainer);
	translations->set_theme_type_variation("TabContainerInner");
	translations->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(translations);

	{
		VBoxContainer *tvb = memnew(VBoxContainer);
		tvb->set_name(TTRC("Translations"));
		translations->add_child(tvb);

		HBoxContainer *thb = memnew(HBoxContainer);
		Label *l = memnew(Label(TTRC("Translations:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		Button *addtr = memnew(Button(TTRC("Add...")));
		addtr->connect(SceneStringName(pressed), callable_mp(this, &LocalizationEditor::_translation_file_open));
		thb->add_child(addtr);

		MarginContainer *mc = memnew(MarginContainer);
		mc->set_theme_type_variation("NoBorderHorizontalBottom");
		mc->set_v_size_flags(SIZE_EXPAND_FILL);
		tvb->add_child(mc);

		translation_list = memnew(Tree);
		translation_list->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_TOP);
		mc->add_child(translation_list);
		trees.push_back(translation_list);
		tree_data_types[translation_list] = "localization_editor_translation_item";
		tree_settings[translation_list] = "internationalization/locale/translations";

		locale_select = memnew(EditorLocaleDialog);
		locale_select->connect("locale_selected", callable_mp(this, &LocalizationEditor::_translation_res_option_selected));
		add_child(locale_select);

		translation_file_open = memnew(EditorFileDialog);
		translation_file_open->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		translation_file_open->connect("files_selected", callable_mp(this, &LocalizationEditor::_translation_add));
		add_child(translation_file_open);
	}

	{
		VBoxContainer *tvb = memnew(VBoxContainer);
		tvb->set_name(TTRC("Remaps"));
		translations->add_child(tvb);

		HBoxContainer *thb = memnew(HBoxContainer);
		Label *l = memnew(Label(TTRC("Resources:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		Button *addtr = memnew(Button(TTRC("Add...")));
		addtr->connect(SceneStringName(pressed), callable_mp(this, &LocalizationEditor::_translation_res_file_open));
		thb->add_child(addtr);

		MarginContainer *mc = memnew(MarginContainer);
		mc->set_theme_type_variation("NoBorderHorizontal");
		mc->set_v_size_flags(SIZE_EXPAND_FILL);
		tvb->add_child(mc);

		translation_remap = memnew(Tree);
		translation_remap->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
		translation_remap->connect("cell_selected", callable_mp(this, &LocalizationEditor::_translation_res_select));
		translation_remap->connect("button_clicked", callable_mp(this, &LocalizationEditor::_translation_res_delete));
		mc->add_child(translation_remap);

		translation_res_file_open_dialog = memnew(EditorFileDialog);
		translation_res_file_open_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		translation_res_file_open_dialog->connect("files_selected", callable_mp(this, &LocalizationEditor::_translation_res_add));
		add_child(translation_res_file_open_dialog);

		thb = memnew(HBoxContainer);
		l = memnew(Label(TTRC("Remaps by Locale:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		addtr = memnew(Button(TTRC("Add...")));
		addtr->connect(SceneStringName(pressed), callable_mp(this, &LocalizationEditor::_translation_res_option_file_open));
		translation_res_option_add_button = addtr;
		thb->add_child(addtr);

		mc = memnew(MarginContainer);
		mc->set_theme_type_variation("NoBorderHorizontalBottom");
		mc->set_v_size_flags(SIZE_EXPAND_FILL);
		tvb->add_child(mc);

		translation_remap_options = memnew(Tree);
		translation_remap_options->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		translation_remap_options->set_columns(2);
		translation_remap_options->set_column_title(0, TTRC("Path"));
		translation_remap_options->set_column_title(1, TTRC("Locale"));
		translation_remap_options->set_column_titles_visible(true);
		translation_remap_options->set_column_expand(0, true);
		translation_remap_options->set_column_clip_content(0, true);
		translation_remap_options->set_column_expand(1, false);
		translation_remap_options->set_column_clip_content(1, false);
		translation_remap_options->set_column_custom_minimum_width(1, 250);
		translation_remap_options->connect("item_edited", callable_mp(this, &LocalizationEditor::_translation_res_option_changed));
		translation_remap_options->connect("button_clicked", callable_mp(this, &LocalizationEditor::_translation_res_option_delete));
		translation_remap_options->connect("custom_popup_edited", callable_mp(this, &LocalizationEditor::_translation_res_option_popup));
		mc->add_child(translation_remap_options);

		translation_res_option_file_open_dialog = memnew(EditorFileDialog);
		translation_res_option_file_open_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		translation_res_option_file_open_dialog->connect("files_selected", callable_mp(this, &LocalizationEditor::_translation_res_option_add));
		add_child(translation_res_option_file_open_dialog);
	}

	{
		VBoxContainer *tvb = memnew(VBoxContainer);
		tvb->set_name(TTRC("Template Generation"));
		translations->add_child(tvb);

		HBoxContainer *thb = memnew(HBoxContainer);
		Label *l = memnew(Label(TTRC("Files with translation strings:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		Button *addtr = memnew(Button(TTRC("Add...")));
		addtr->connect(SceneStringName(pressed), callable_mp(this, &LocalizationEditor::_template_source_file_open));
		thb->add_child(addtr);

		template_generate_button = memnew(Button(TTRC("Generate")));
		template_generate_button->connect(SceneStringName(pressed), callable_mp(this, &LocalizationEditor::_template_generate_open));
		thb->add_child(template_generate_button);

		MarginContainer *mc = memnew(MarginContainer);
		mc->set_theme_type_variation("NoBorderHorizontal");
		mc->set_v_size_flags(SIZE_EXPAND_FILL);
		tvb->add_child(mc);

		template_source_list = memnew(Tree);
		template_source_list->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
		mc->add_child(template_source_list);
		trees.push_back(template_source_list);
		tree_data_types[template_source_list] = "localization_editor_pot_item";
		tree_settings[template_source_list] = "internationalization/locale/translations_pot_files";

		template_add_builtin = memnew(CheckBox(TTRC("Add Built-in Strings")));
		template_add_builtin->set_tooltip_text(TTRC("Add strings from built-in components such as certain Control nodes."));
		template_add_builtin->connect(SceneStringName(pressed), callable_mp(this, &LocalizationEditor::_template_add_builtin_toggled));
		tvb->add_child(template_add_builtin);

		template_generate_dialog = memnew(EditorFileDialog);
		template_generate_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
		template_generate_dialog->set_current_path(EditorSettings::get_singleton()->get_project_metadata("pot_generator", "last_pot_path", String()));
		template_generate_dialog->connect("file_selected", callable_mp(this, &LocalizationEditor::_template_generate));
		add_child(template_generate_dialog);

		template_source_open_dialog = memnew(EditorFileDialog);
		template_source_open_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		template_source_open_dialog->connect("files_selected", callable_mp(this, &LocalizationEditor::_template_source_add));
		add_child(template_source_open_dialog);
	}

	for (Tree *tree : trees) {
		SET_DRAG_FORWARDING_GCD(tree, LocalizationEditor);
	}
}
