/*************************************************************************/
/*  localization_editor.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "localization_editor.h"

#include "core/string/translation.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_translation_parser.h"
#include "pot_generator.h"
#include "scene/gui/control.h"

void LocalizationEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		translation_list->connect("button_pressed", callable_mp(this, &LocalizationEditor::_translation_delete));
		translation_pot_list->connect("button_pressed", callable_mp(this, &LocalizationEditor::_pot_delete));

		List<String> tfn;
		ResourceLoader::get_recognized_extensions_for_type("Translation", &tfn);
		for (const String &E : tfn) {
			translation_file_open->add_filter("*." + E);
		}

		List<String> rfn;
		ResourceLoader::get_recognized_extensions_for_type("Resource", &rfn);
		for (const String &E : rfn) {
			translation_res_file_open_dialog->add_filter("*." + E);
			translation_res_option_file_open_dialog->add_filter("*." + E);
		}

		_update_pot_file_extensions();
		pot_generate_dialog->add_filter("*.pot");
	}
}

void LocalizationEditor::add_translation(const String &p_translation) {
	PackedStringArray translations;
	translations.push_back(p_translation);
	_translation_add(translations);
}

void LocalizationEditor::_translation_add(const PackedStringArray &p_paths) {
	PackedStringArray translations = ProjectSettings::get_singleton()->get("internationalization/locale/translations");
	for (int i = 0; i < p_paths.size(); i++) {
		if (!translations.has(p_paths[i])) {
			// Don't add duplicate translation paths.
			translations.push_back(p_paths[i]);
		}
	}

	undo_redo->create_action(vformat(TTR("Add %d Translations"), p_paths.size()));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", translations);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", ProjectSettings::get_singleton()->get("internationalization/locale/translations"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_file_open() {
	translation_file_open->popup_file_dialog();
}

void LocalizationEditor::_translation_delete(Object *p_item, int p_column, int p_button) {
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND(!ti);

	int idx = ti->get_metadata(0);

	PackedStringArray translations = ProjectSettings::get_singleton()->get("internationalization/locale/translations");

	ERR_FAIL_INDEX(idx, translations.size());

	translations.remove_at(idx);

	undo_redo->create_action(TTR("Remove Translation"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", translations);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations", ProjectSettings::get_singleton()->get("internationalization/locale/translations"));
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
		remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");
		prev = remaps;
	}

	for (int i = 0; i < p_paths.size(); i++) {
		if (!remaps.has(p_paths[i])) {
			// Don't overwrite with an empty remap array if an array already exists for the given path.
			remaps[p_paths[i]] = PackedStringArray();
		}
	}

	undo_redo->create_action(vformat(TTR("Translation Resource Remap: Add %d Path(s)"), p_paths.size()));
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

	Dictionary remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_COND(!k);

	String key = k->get_metadata(0);

	ERR_FAIL_COND(!remaps.has(key));
	PackedStringArray r = remaps[key];
	for (int i = 0; i < p_paths.size(); i++) {
		r.push_back(p_paths[i] + ":" + "en");
	}
	remaps[key] = r;

	undo_redo->create_action(vformat(TTR("Translation Resource Remap: Add %d Remap(s)"), p_paths.size()));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps"));
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
	call_deferred(SNAME("update_translations"));
}

void LocalizationEditor::_translation_res_option_popup(bool p_arrow_clicked) {
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_COND(!ed);

	locale_select->set_locale(ed->get_tooltip(1));
	locale_select->popup_locale_dialog();
}

void LocalizationEditor::_translation_res_option_selected(const String &p_locale) {
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_COND(!ed);

	ed->set_text(1, TranslationServer::get_singleton()->get_locale_name(p_locale));
	ed->set_tooltip(1, p_locale);

	LocalizationEditor::_translation_res_option_changed();
}

void LocalizationEditor::_translation_res_option_changed() {
	if (updating_translations) {
		return;
	}

	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_COND(!k);
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_COND(!ed);

	String key = k->get_metadata(0);
	int idx = ed->get_metadata(0);
	String path = ed->get_metadata(1);
	String locale = ed->get_tooltip(1);

	ERR_FAIL_COND(!remaps.has(key));
	PackedStringArray r = remaps[key];
	r.set(idx, path + ":" + locale);
	remaps[key] = r;

	updating_translations = true;
	undo_redo->create_action(TTR("Change Resource Remap Language"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
	updating_translations = false;
}

void LocalizationEditor::_translation_res_delete(Object *p_item, int p_column, int p_button) {
	if (updating_translations) {
		return;
	}

	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");

	TreeItem *k = Object::cast_to<TreeItem>(p_item);

	String key = k->get_metadata(0);
	ERR_FAIL_COND(!remaps.has(key));

	remaps.erase(key);

	undo_redo->create_action(TTR("Remove Resource Remap"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_translation_res_option_delete(Object *p_item, int p_column, int p_button) {
	if (updating_translations) {
		return;
	}

	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translation_remaps")) {
		return;
	}

	Dictionary remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_COND(!k);
	TreeItem *ed = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND(!ed);

	String key = k->get_metadata(0);
	int idx = ed->get_metadata(0);

	ERR_FAIL_COND(!remaps.has(key));
	PackedStringArray r = remaps[key];
	ERR_FAIL_INDEX(idx, r.size());
	r.remove_at(idx);
	remaps[key] = r;

	undo_redo->create_action(TTR("Remove Resource Remap Option"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", remaps);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translation_remaps", ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_pot_add(const PackedStringArray &p_paths) {
	PackedStringArray pot_translations = ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files");
	for (int i = 0; i < p_paths.size(); i++) {
		if (!pot_translations.has(p_paths[i])) {
			pot_translations.push_back(p_paths[i]);
		}
	}

	undo_redo->create_action(vformat(TTR("Add %d file(s) for POT generation"), p_paths.size()));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", pot_translations);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_pot_delete(Object *p_item, int p_column, int p_button) {
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND(!ti);

	int idx = ti->get_metadata(0);

	PackedStringArray pot_translations = ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files");

	ERR_FAIL_INDEX(idx, pot_translations.size());

	pot_translations.remove_at(idx);

	undo_redo->create_action(TTR("Remove file from POT generation"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", pot_translations);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/translations_pot_files", ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files"));
	undo_redo->add_do_method(this, "update_translations");
	undo_redo->add_undo_method(this, "update_translations");
	undo_redo->add_do_method(this, "emit_signal", localization_changed);
	undo_redo->add_undo_method(this, "emit_signal", localization_changed);
	undo_redo->commit_action();
}

void LocalizationEditor::_pot_file_open() {
	pot_file_open_dialog->popup_file_dialog();
}

void LocalizationEditor::_pot_generate_open() {
	pot_generate_dialog->popup_file_dialog();
}

void LocalizationEditor::_pot_generate(const String &p_file) {
	POTGenerator::get_singleton()->generate_pot(p_file);
}

void LocalizationEditor::_update_pot_file_extensions() {
	pot_file_open_dialog->clear_filters();
	List<String> translation_parse_file_extensions;
	EditorTranslationParser::get_singleton()->get_recognized_extensions(&translation_parse_file_extensions);
	for (const String &E : translation_parse_file_extensions) {
		pot_file_open_dialog->add_filter("*." + E);
	}
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
		PackedStringArray translations = ProjectSettings::get_singleton()->get("internationalization/locale/translations");
		for (int i = 0; i < translations.size(); i++) {
			TreeItem *t = translation_list->create_item(root);
			t->set_editable(0, false);
			t->set_text(0, translations[i].replace_first("res://", ""));
			t->set_tooltip(0, translations[i]);
			t->set_metadata(0, i);
			t->add_button(0, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), 0, false, TTR("Remove"));
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
		Dictionary remaps = ProjectSettings::get_singleton()->get("internationalization/locale/translation_remaps");
		List<Variant> rk;
		remaps.get_key_list(&rk);
		Vector<String> keys;
		for (const Variant &E : rk) {
			keys.push_back(E);
		}
		keys.sort();

		for (int i = 0; i < keys.size(); i++) {
			TreeItem *t = translation_remap->create_item(root);
			t->set_editable(0, false);
			t->set_text(0, keys[i].replace_first("res://", ""));
			t->set_tooltip(0, keys[i]);
			t->set_metadata(0, keys[i]);
			t->add_button(0, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), 0, false, TTR("Remove"));
			if (keys[i] == remap_selected) {
				t->select(0);
				translation_res_option_add_button->set_disabled(false);

				PackedStringArray selected = remaps[keys[i]];
				for (int j = 0; j < selected.size(); j++) {
					String s2 = selected[j];
					int qp = s2.rfind(":");
					String path = s2.substr(0, qp);
					String locale = s2.substr(qp + 1, s2.length());

					TreeItem *t2 = translation_remap_options->create_item(root2);
					t2->set_editable(0, false);
					t2->set_text(0, path.replace_first("res://", ""));
					t2->set_tooltip(0, path);
					t2->set_metadata(0, j);
					t2->add_button(0, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), 0, false, TTR("Remove"));
					t2->set_cell_mode(1, TreeItem::CELL_MODE_CUSTOM);
					t2->set_text(1, TranslationServer::get_singleton()->get_locale_name(locale));
					t2->set_editable(1, true);
					t2->set_metadata(1, path);
					t2->set_tooltip(1, locale);
				}
			}
		}
	}

	// Update translation POT files.
	translation_pot_list->clear();
	root = translation_pot_list->create_item(nullptr);
	translation_pot_list->set_hide_root(true);
	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/translations_pot_files")) {
		PackedStringArray pot_translations = ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files");
		for (int i = 0; i < pot_translations.size(); i++) {
			TreeItem *t = translation_pot_list->create_item(root);
			t->set_editable(0, false);
			t->set_text(0, pot_translations[i].replace_first("res://", ""));
			t->set_tooltip(0, pot_translations[i]);
			t->set_metadata(0, i);
			t->add_button(0, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), 0, false, TTR("Remove"));
		}
	}

	// New translation parser plugin might extend possible file extensions in POT generation.
	_update_pot_file_extensions();

	updating_translations = false;
}

void LocalizationEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("update_translations"), &LocalizationEditor::update_translations);

	ADD_SIGNAL(MethodInfo("localization_changed"));
}

LocalizationEditor::LocalizationEditor() {
	undo_redo = EditorNode::get_undo_redo();
	updating_translations = false;
	localization_changed = "localization_changed";

	TabContainer *translations = memnew(TabContainer);
	translations->set_tab_alignment(TabContainer::ALIGNMENT_LEFT);
	translations->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(translations);

	{
		VBoxContainer *tvb = memnew(VBoxContainer);
		tvb->set_name(TTR("Translations"));
		translations->add_child(tvb);

		HBoxContainer *thb = memnew(HBoxContainer);
		Label *l = memnew(Label(TTR("Translations:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		Button *addtr = memnew(Button(TTR("Add...")));
		addtr->connect("pressed", callable_mp(this, &LocalizationEditor::_translation_file_open));
		thb->add_child(addtr);

		VBoxContainer *tmc = memnew(VBoxContainer);
		tmc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		tvb->add_child(tmc);

		translation_list = memnew(Tree);
		translation_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		tmc->add_child(translation_list);

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
		tvb->set_name(TTR("Remaps"));
		translations->add_child(tvb);

		HBoxContainer *thb = memnew(HBoxContainer);
		Label *l = memnew(Label(TTR("Resources:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		Button *addtr = memnew(Button(TTR("Add...")));
		addtr->connect("pressed", callable_mp(this, &LocalizationEditor::_translation_res_file_open));
		thb->add_child(addtr);

		VBoxContainer *tmc = memnew(VBoxContainer);
		tmc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		tvb->add_child(tmc);

		translation_remap = memnew(Tree);
		translation_remap->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		translation_remap->connect("cell_selected", callable_mp(this, &LocalizationEditor::_translation_res_select));
		translation_remap->connect("button_pressed", callable_mp(this, &LocalizationEditor::_translation_res_delete));
		tmc->add_child(translation_remap);

		translation_res_file_open_dialog = memnew(EditorFileDialog);
		translation_res_file_open_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		translation_res_file_open_dialog->connect("files_selected", callable_mp(this, &LocalizationEditor::_translation_res_add));
		add_child(translation_res_file_open_dialog);

		thb = memnew(HBoxContainer);
		l = memnew(Label(TTR("Remaps by Locale:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		addtr = memnew(Button(TTR("Add...")));
		addtr->connect("pressed", callable_mp(this, &LocalizationEditor::_translation_res_option_file_open));
		translation_res_option_add_button = addtr;
		thb->add_child(addtr);

		tmc = memnew(VBoxContainer);
		tmc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		tvb->add_child(tmc);

		translation_remap_options = memnew(Tree);
		translation_remap_options->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		translation_remap_options->set_columns(2);
		translation_remap_options->set_column_title(0, TTR("Path"));
		translation_remap_options->set_column_title(1, TTR("Locale"));
		translation_remap_options->set_column_titles_visible(true);
		translation_remap_options->set_column_expand(0, true);
		translation_remap_options->set_column_clip_content(0, true);
		translation_remap_options->set_column_expand(1, false);
		translation_remap_options->set_column_clip_content(1, false);
		translation_remap_options->set_column_custom_minimum_width(1, 250);
		translation_remap_options->connect("item_edited", callable_mp(this, &LocalizationEditor::_translation_res_option_changed));
		translation_remap_options->connect("button_pressed", callable_mp(this, &LocalizationEditor::_translation_res_option_delete));
		translation_remap_options->connect("custom_popup_edited", callable_mp(this, &LocalizationEditor::_translation_res_option_popup));
		tmc->add_child(translation_remap_options);

		translation_res_option_file_open_dialog = memnew(EditorFileDialog);
		translation_res_option_file_open_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		translation_res_option_file_open_dialog->connect("files_selected", callable_mp(this, &LocalizationEditor::_translation_res_option_add));
		add_child(translation_res_option_file_open_dialog);
	}

	{
		VBoxContainer *tvb = memnew(VBoxContainer);
		tvb->set_name(TTR("POT Generation"));
		translations->add_child(tvb);

		HBoxContainer *thb = memnew(HBoxContainer);
		Label *l = memnew(Label(TTR("Files with translation strings:")));
		l->set_theme_type_variation("HeaderSmall");
		thb->add_child(l);
		thb->add_spacer();
		tvb->add_child(thb);

		Button *addtr = memnew(Button(TTR("Add...")));
		addtr->connect("pressed", callable_mp(this, &LocalizationEditor::_pot_file_open));
		thb->add_child(addtr);

		Button *generate = memnew(Button(TTR("Generate POT")));
		generate->connect("pressed", callable_mp(this, &LocalizationEditor::_pot_generate_open));
		thb->add_child(generate);

		VBoxContainer *tmc = memnew(VBoxContainer);
		tmc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		tvb->add_child(tmc);

		translation_pot_list = memnew(Tree);
		translation_pot_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		tmc->add_child(translation_pot_list);

		pot_generate_dialog = memnew(EditorFileDialog);
		pot_generate_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
		pot_generate_dialog->connect("file_selected", callable_mp(this, &LocalizationEditor::_pot_generate));
		add_child(pot_generate_dialog);

		pot_file_open_dialog = memnew(EditorFileDialog);
		pot_file_open_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		pot_file_open_dialog->connect("files_selected", callable_mp(this, &LocalizationEditor::_pot_add));
		add_child(pot_file_open_dialog);
	}
}
