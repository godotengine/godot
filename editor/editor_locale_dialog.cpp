/**************************************************************************/
/*  editor_locale_dialog.cpp                                              */
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

#include "editor_locale_dialog.h"

#include "core/config/project_settings.h"
#include "editor/editor_scale.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/check_button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tree.h"

void EditorLocaleDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("locale_selected", PropertyInfo(Variant::STRING, "locale")));
}

void EditorLocaleDialog::ok_pressed() {
	if (edit_filters->is_pressed()) {
		return; // Do not update, if in filter edit mode.
	}

	String locale;
	if (lang_code->get_text().is_empty()) {
		return; // Language code is required.
	}
	locale = lang_code->get_text();

	if (!script_code->get_text().is_empty()) {
		locale += "_" + script_code->get_text();
	}
	if (!country_code->get_text().is_empty()) {
		locale += "_" + country_code->get_text();
	}
	if (!variant_code->get_text().is_empty()) {
		locale += "_" + variant_code->get_text();
	}

	emit_signal(SNAME("locale_selected"), TranslationServer::get_singleton()->standardize_locale(locale));
	hide();
}

void EditorLocaleDialog::_item_selected() {
	if (updating_lists) {
		return;
	}

	if (edit_filters->is_pressed()) {
		return; // Do not update, if in filter edit mode.
	}

	TreeItem *l = lang_list->get_selected();
	if (l) {
		lang_code->set_text(l->get_metadata(0).operator String());
	}

	TreeItem *s = script_list->get_selected();
	if (s) {
		script_code->set_text(s->get_metadata(0).operator String());
	}

	TreeItem *c = cnt_list->get_selected();
	if (c) {
		country_code->set_text(c->get_metadata(0).operator String());
	}
}

void EditorLocaleDialog::_toggle_advanced(bool p_checked) {
	if (!p_checked) {
		script_code->set_text("");
		variant_code->set_text("");
	}
	_update_tree();
}

void EditorLocaleDialog::_post_popup() {
	ConfirmationDialog::_post_popup();

	if (!locale_set) {
		lang_code->set_text("");
		script_code->set_text("");
		country_code->set_text("");
		variant_code->set_text("");
	}
	edit_filters->set_pressed(false);
	_update_tree();
}

void EditorLocaleDialog::_filter_lang_option_changed() {
	TreeItem *t = lang_list->get_edited();
	String lang = t->get_metadata(0);
	bool checked = t->is_checked(0);

	Variant prev;
	Array f_lang_all;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/language_filter")) {
		f_lang_all = GLOBAL_GET("internationalization/locale/language_filter");
		prev = f_lang_all;
	}

	int l_idx = f_lang_all.find(lang);

	if (checked) {
		if (l_idx == -1) {
			f_lang_all.append(lang);
		}
	} else {
		if (l_idx != -1) {
			f_lang_all.remove_at(l_idx);
		}
	}

	f_lang_all.sort();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Changed Locale Language Filter"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/language_filter", f_lang_all);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/language_filter", prev);
	undo_redo->commit_action();
}

void EditorLocaleDialog::_filter_script_option_changed() {
	TreeItem *t = script_list->get_edited();
	String scr_code = t->get_metadata(0);
	bool checked = t->is_checked(0);

	Variant prev;
	Array f_script_all;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/script_filter")) {
		f_script_all = GLOBAL_GET("internationalization/locale/script_filter");
		prev = f_script_all;
	}

	int l_idx = f_script_all.find(scr_code);

	if (checked) {
		if (l_idx == -1) {
			f_script_all.append(scr_code);
		}
	} else {
		if (l_idx != -1) {
			f_script_all.remove_at(l_idx);
		}
	}

	f_script_all.sort();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Changed Locale Script Filter"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/script_filter", f_script_all);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/script_filter", prev);
	undo_redo->commit_action();
}

void EditorLocaleDialog::_filter_cnt_option_changed() {
	TreeItem *t = cnt_list->get_edited();
	String cnt = t->get_metadata(0);
	bool checked = t->is_checked(0);

	Variant prev;
	Array f_cnt_all;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/country_filter")) {
		f_cnt_all = GLOBAL_GET("internationalization/locale/country_filter");
		prev = f_cnt_all;
	}

	int l_idx = f_cnt_all.find(cnt);

	if (checked) {
		if (l_idx == -1) {
			f_cnt_all.append(cnt);
		}
	} else {
		if (l_idx != -1) {
			f_cnt_all.remove_at(l_idx);
		}
	}

	f_cnt_all.sort();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Changed Locale Country Filter"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/country_filter", f_cnt_all);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/country_filter", prev);
	undo_redo->commit_action();
}

void EditorLocaleDialog::_filter_mode_changed(int p_mode) {
	int f_mode = filter_mode->get_selected_id();
	Variant prev;

	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/locale_filter_mode")) {
		prev = GLOBAL_GET("internationalization/locale/locale_filter_mode");
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Changed Locale Filter Mode"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "internationalization/locale/locale_filter_mode", f_mode);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "internationalization/locale/locale_filter_mode", prev);
	undo_redo->commit_action();

	_update_tree();
}

void EditorLocaleDialog::_edit_filters(bool p_checked) {
	_update_tree();
}

void EditorLocaleDialog::_update_tree() {
	updating_lists = true;

	int filter = SHOW_ALL_LOCALES;
	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/locale_filter_mode")) {
		filter = GLOBAL_GET("internationalization/locale/locale_filter_mode");
	}
	Array f_lang_all;
	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/language_filter")) {
		f_lang_all = GLOBAL_GET("internationalization/locale/language_filter");
	}
	Array f_cnt_all;
	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/country_filter")) {
		f_cnt_all = GLOBAL_GET("internationalization/locale/country_filter");
	}
	Array f_script_all;
	if (ProjectSettings::get_singleton()->has_setting("internationalization/locale/script_filter")) {
		f_script_all = GLOBAL_GET("internationalization/locale/script_filter");
	}
	bool is_edit_mode = edit_filters->is_pressed();

	filter_mode->select(filter);

	// Hide text advanced edit and disable OK button if in filter edit mode.
	advanced->set_visible(!is_edit_mode);
	hb_locale->set_visible(!is_edit_mode && advanced->is_pressed());
	vb_script_list->set_visible(advanced->is_pressed());
	get_ok_button()->set_disabled(is_edit_mode);

	// Update language list.
	lang_list->clear();
	TreeItem *l_root = lang_list->create_item(nullptr);
	lang_list->set_hide_root(true);

	Vector<String> languages = TranslationServer::get_singleton()->get_all_languages();
	for (const String &E : languages) {
		if (is_edit_mode || (filter == SHOW_ALL_LOCALES) || f_lang_all.has(E) || f_lang_all.is_empty()) {
			const String &lang = TranslationServer::get_singleton()->get_language_name(E);
			TreeItem *t = lang_list->create_item(l_root);
			if (is_edit_mode) {
				t->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				t->set_editable(0, true);
				t->set_checked(0, f_lang_all.has(E));
			} else if (lang_code->get_text() == E) {
				t->select(0);
			}
			t->set_text(0, vformat("%s [%s]", lang, E));
			t->set_metadata(0, E);
		}
	}

	// Update script list.
	script_list->clear();
	TreeItem *s_root = script_list->create_item(nullptr);
	script_list->set_hide_root(true);

	if (!is_edit_mode) {
		TreeItem *t = script_list->create_item(s_root);
		t->set_text(0, TTR("[Default]"));
		t->set_metadata(0, "");
	}

	Vector<String> scripts = TranslationServer::get_singleton()->get_all_scripts();
	for (const String &E : scripts) {
		if (is_edit_mode || (filter == SHOW_ALL_LOCALES) || f_script_all.has(E) || f_script_all.is_empty()) {
			const String &scr_code = TranslationServer::get_singleton()->get_script_name(E);
			TreeItem *t = script_list->create_item(s_root);
			if (is_edit_mode) {
				t->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				t->set_editable(0, true);
				t->set_checked(0, f_script_all.has(E));
			} else if (script_code->get_text() == E) {
				t->select(0);
			}
			t->set_text(0, vformat("%s [%s]", scr_code, E));
			t->set_metadata(0, E);
		}
	}

	// Update country list.
	cnt_list->clear();
	TreeItem *c_root = cnt_list->create_item(nullptr);
	cnt_list->set_hide_root(true);

	if (!is_edit_mode) {
		TreeItem *t = cnt_list->create_item(c_root);
		t->set_text(0, "[Default]");
		t->set_metadata(0, "");
	}

	Vector<String> countries = TranslationServer::get_singleton()->get_all_countries();
	for (const String &E : countries) {
		if (is_edit_mode || (filter == SHOW_ALL_LOCALES) || f_cnt_all.has(E) || f_cnt_all.is_empty()) {
			const String &cnt = TranslationServer::get_singleton()->get_country_name(E);
			TreeItem *t = cnt_list->create_item(c_root);
			if (is_edit_mode) {
				t->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				t->set_editable(0, true);
				t->set_checked(0, f_cnt_all.has(E));
			} else if (country_code->get_text() == E) {
				t->select(0);
			}
			t->set_text(0, vformat("%s [%s]", cnt, E));
			t->set_metadata(0, E);
		}
	}
	updating_lists = false;
}

void EditorLocaleDialog::set_locale(const String &p_locale) {
	const String &locale = TranslationServer::get_singleton()->standardize_locale(p_locale);
	if (locale.is_empty()) {
		locale_set = false;

		lang_code->set_text("");
		script_code->set_text("");
		country_code->set_text("");
		variant_code->set_text("");
	} else {
		locale_set = true;

		Vector<String> locale_elements = p_locale.split("_");
		lang_code->set_text(locale_elements[0]);
		if (locale_elements.size() >= 2) {
			if (locale_elements[1].length() == 4 && is_ascii_upper_case(locale_elements[1][0]) && is_ascii_lower_case(locale_elements[1][1]) && is_ascii_lower_case(locale_elements[1][2]) && is_ascii_lower_case(locale_elements[1][3])) {
				script_code->set_text(locale_elements[1]);
				advanced->set_pressed(true);
			}
			if (locale_elements[1].length() == 2 && is_ascii_upper_case(locale_elements[1][0]) && is_ascii_upper_case(locale_elements[1][1])) {
				country_code->set_text(locale_elements[1]);
			}
		}
		if (locale_elements.size() >= 3) {
			if (locale_elements[2].length() == 2 && is_ascii_upper_case(locale_elements[2][0]) && is_ascii_upper_case(locale_elements[2][1])) {
				country_code->set_text(locale_elements[2]);
			} else {
				variant_code->set_text(locale_elements[2].to_lower());
				advanced->set_pressed(true);
			}
		}
		if (locale_elements.size() >= 4) {
			variant_code->set_text(locale_elements[3].to_lower());
			advanced->set_pressed(true);
		}
	}
}

void EditorLocaleDialog::popup_locale_dialog() {
	popup_centered_clamped(Size2(1050, 700) * EDSCALE, 0.8);
}

EditorLocaleDialog::EditorLocaleDialog() {
	set_title(TTR("Select a Locale"));

	VBoxContainer *vb = memnew(VBoxContainer);
	{
		HBoxContainer *hb_filter = memnew(HBoxContainer);
		{
			filter_mode = memnew(OptionButton);
			filter_mode->add_item(TTR("Show All Locales"), SHOW_ALL_LOCALES);
			filter_mode->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			filter_mode->add_item(TTR("Show Selected Locales Only"), SHOW_ONLY_SELECTED_LOCALES);
			filter_mode->select(0);
			filter_mode->connect("item_selected", callable_mp(this, &EditorLocaleDialog::_filter_mode_changed));
			hb_filter->add_child(filter_mode);
		}
		{
			edit_filters = memnew(CheckButton);
			edit_filters->set_text(TTR("Edit Filters"));
			edit_filters->set_toggle_mode(true);
			edit_filters->set_pressed(false);
			edit_filters->connect("toggled", callable_mp(this, &EditorLocaleDialog::_edit_filters));
			hb_filter->add_child(edit_filters);
		}
		{
			advanced = memnew(CheckButton);
			advanced->set_text(TTR("Advanced"));
			advanced->set_toggle_mode(true);
			advanced->set_pressed(false);
			advanced->connect("toggled", callable_mp(this, &EditorLocaleDialog::_toggle_advanced));
			hb_filter->add_child(advanced);
		}
		vb->add_child(hb_filter);
	}
	{
		HBoxContainer *hb_lists = memnew(HBoxContainer);
		hb_lists->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		{
			VBoxContainer *vb_lang_list = memnew(VBoxContainer);
			vb_lang_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			{
				Label *lang_lbl = memnew(Label);
				lang_lbl->set_text(TTR("Language:"));
				vb_lang_list->add_child(lang_lbl);
			}
			{
				lang_list = memnew(Tree);
				lang_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
				lang_list->connect("cell_selected", callable_mp(this, &EditorLocaleDialog::_item_selected));
				lang_list->set_columns(1);
				lang_list->connect("item_edited", callable_mp(this, &EditorLocaleDialog::_filter_lang_option_changed));
				vb_lang_list->add_child(lang_list);
			}
			hb_lists->add_child(vb_lang_list);
		}
		{
			vb_script_list = memnew(VBoxContainer);
			vb_script_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			{
				Label *script_lbl = memnew(Label);
				// TRANSLATORS: This is the label for a list of writing systems.
				script_lbl->set_text(TTR("Script:", "Locale"));
				vb_script_list->add_child(script_lbl);
			}
			{
				script_list = memnew(Tree);
				script_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
				script_list->connect("cell_selected", callable_mp(this, &EditorLocaleDialog::_item_selected));
				script_list->set_columns(1);
				script_list->connect("item_edited", callable_mp(this, &EditorLocaleDialog::_filter_script_option_changed));
				vb_script_list->add_child(script_list);
			}
			hb_lists->add_child(vb_script_list);
		}
		{
			VBoxContainer *vb_cnt_list = memnew(VBoxContainer);
			vb_cnt_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			{
				Label *cnt_lbl = memnew(Label);
				cnt_lbl->set_text(TTR("Country:"));
				vb_cnt_list->add_child(cnt_lbl);
			}
			{
				cnt_list = memnew(Tree);
				cnt_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
				cnt_list->connect("cell_selected", callable_mp(this, &EditorLocaleDialog::_item_selected));
				cnt_list->set_columns(1);
				cnt_list->connect("item_edited", callable_mp(this, &EditorLocaleDialog::_filter_cnt_option_changed));
				vb_cnt_list->add_child(cnt_list);
			}
			hb_lists->add_child(vb_cnt_list);
		}
		vb->add_child(hb_lists);
	}
	{
		hb_locale = memnew(HBoxContainer);
		hb_locale->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		{
			{
				VBoxContainer *vb_language = memnew(VBoxContainer);
				vb_language->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				{
					Label *language_lbl = memnew(Label);
					language_lbl->set_text(TTR("Language"));
					vb_language->add_child(language_lbl);
				}
				{
					lang_code = memnew(LineEdit);
					lang_code->set_max_length(3);
					lang_code->set_tooltip_text("Language");
					vb_language->add_child(lang_code);
				}
				hb_locale->add_child(vb_language);
			}
			{
				VBoxContainer *vb_script = memnew(VBoxContainer);
				vb_script->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				{
					Label *script_lbl = memnew(Label);
					// TRANSLATORS: This refers to a writing system.
					script_lbl->set_text(TTR("Script", "Locale"));
					vb_script->add_child(script_lbl);
				}
				{
					script_code = memnew(LineEdit);
					script_code->set_max_length(4);
					script_code->set_tooltip_text("Script");
					vb_script->add_child(script_code);
				}
				hb_locale->add_child(vb_script);
			}
			{
				VBoxContainer *vb_country = memnew(VBoxContainer);
				vb_country->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				{
					Label *country_lbl = memnew(Label);
					country_lbl->set_text(TTR("Country"));
					vb_country->add_child(country_lbl);
				}
				{
					country_code = memnew(LineEdit);
					country_code->set_max_length(2);
					country_code->set_tooltip_text("Country");
					vb_country->add_child(country_code);
				}
				hb_locale->add_child(vb_country);
			}
			{
				VBoxContainer *vb_variant = memnew(VBoxContainer);
				vb_variant->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				{
					Label *variant_lbl = memnew(Label);
					variant_lbl->set_text(TTR("Variant"));
					vb_variant->add_child(variant_lbl);
				}
				{
					variant_code = memnew(LineEdit);
					variant_code->set_h_size_flags(Control::SIZE_EXPAND_FILL);
					variant_code->set_placeholder("Variant");
					variant_code->set_tooltip_text("Variant");
					vb_variant->add_child(variant_code);
				}
				hb_locale->add_child(vb_variant);
			}
		}
		vb->add_child(hb_locale);
	}
	add_child(vb);
	_update_tree();

	set_ok_button_text(TTR("Select"));
}
