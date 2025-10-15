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
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/check_button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"

/**************************************************************************/
/* EditorAddCustomLocaleDialog                                            */
/**************************************************************************/

void EditorAddCustomLocaleDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("remove_code", PropertyInfo(Variant::STRING, "code")));
	ADD_SIGNAL(MethodInfo("add_code", PropertyInfo(Variant::STRING, "code"), PropertyInfo(Variant::STRING, "name")));
}

void EditorAddCustomLocaleDialog::ok_pressed() {
	if (edit && !old_code.is_empty() && (old_code != code->get_text() || old_name != name->get_text())) {
		emit_signal(SNAME("remove_code"), old_code, true);
	}
	if (!code->get_text().is_empty() && (old_code != code->get_text() || old_name != name->get_text())) {
		emit_signal(SNAME("add_code"), code->get_text(), name->get_text());
	}
	hide();
}

void EditorAddCustomLocaleDialog::set_data(const String &p_code, const String &p_name, bool p_is_lang, bool p_edit) {
	old_code = p_code;
	old_name = p_name;
	is_lang = p_is_lang;
	edit = p_edit;
	code->set_text(p_code);
	name->set_text(p_name);

	_validate_code(code->get_text());
}

void EditorAddCustomLocaleDialog::_validate_code(const String &p_text) {
	if (is_lang) {
		if (!TranslationServer::get_singleton()->is_language_code_free(p_text)) {
			warn_lbl->set_text(TTR("Invalid custom language code, this code is already in use."));
			get_ok_button()->set_disabled(true);
		} else if (!TranslationServer::is_language_code(p_text)) {
			warn_lbl->set_text(TTR("Invalid custom language code, language code should be two lower case Latin letters (Alpha-2 code) or three lower case Latin letters (Alpha-3 code)."));
			get_ok_button()->set_disabled(true);
		} else {
			warn_lbl->set_text("");
			get_ok_button()->set_disabled(false);
		}
	} else {
		if (!TranslationServer::get_singleton()->is_country_code_free(p_text)) {
			warn_lbl->set_text(TTR("Invalid custom country code, this code is already in use."));
			get_ok_button()->set_disabled(true);
		} else if (!TranslationServer::is_country_code(p_text)) {
			warn_lbl->set_text(TTR("Invalid custom country code, country code should be two upper case Latin letters (Alpha-2 code), three upper case Latin letters (Alpha-3 code), or three digits (UN M49 area code)."));
			get_ok_button()->set_disabled(true);
		} else {
			warn_lbl->set_text("");
			get_ok_button()->set_disabled(false);
		}
	}
}

EditorAddCustomLocaleDialog::EditorAddCustomLocaleDialog() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *hb_main = memnew(HBoxContainer);
	vb->add_child(hb_main);

	Label *code_lbl = memnew(Label);
	code_lbl->set_text(TTR("Code:"));
	hb_main->add_child(code_lbl);

	code = memnew(LineEdit);
	code->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	code->connect(SceneStringName(text_changed), callable_mp(this, &EditorAddCustomLocaleDialog::_validate_code));
	hb_main->add_child(code);

	Label *name_lbl = memnew(Label);
	name_lbl->set_text(TTR("Name:"));
	hb_main->add_child(name_lbl);

	name = memnew(LineEdit);
	name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb_main->add_child(name);

	warn_lbl = memnew(Label);
	warn_lbl->add_theme_color_override(SceneStringName(font_color), EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("warning_color"), EditorStringName(Editor)));
	vb->add_child(warn_lbl);
}

/**************************************************************************/
/* EditorLocaleDialog                                                     */
/**************************************************************************/

void EditorLocaleDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("locale_selected", PropertyInfo(Variant::STRING, "locale")));
}

void EditorLocaleDialog::ok_pressed() {
	emit_signal(SNAME("locale_selected"), locale.operator String());
	hide();
}

void EditorLocaleDialog::_post_popup() {
	ConfirmationDialog::_post_popup();
}

void EditorLocaleDialog::set_locale(const String &p_locale) {
	locale_valid = !p_locale.is_empty();
	locale = TranslationServer::Locale(p_locale, false);

	_lang_search(lang_search->get_text());
	_script_search(script_search->get_text());
	_country_search(country_search->get_text());
	variant_code->set_text(locale.variant);
	_item_selected();
}

void EditorLocaleDialog::popup_locale_dialog() {
	popup_centered_clamped(Size2(1050, 700) * EDSCALE, 0.8);
}

void EditorLocaleDialog::_toggle_advanced(bool p_checked) {
	script_vb->set_visible(p_checked);
	variant_code->set_visible(p_checked);
	variant_lbl->set_visible(p_checked);
}

void EditorLocaleDialog::_update_cache() {
	if (updating_settings) {
		return;
	}
	translation_cache.clear();
}

void EditorLocaleDialog::_lang_search(const String &p_text) {
	updating_lists = true;
	lang_list->clear();

	TreeItem *root = lang_list->create_item(nullptr);
	lang_list->set_hide_root(true);
	bool current_found = false;

	// 1. Add search results.
	TreeItem *search_root = lang_list->create_item(root);
	search_root->set_icon(0, get_editor_theme_icon(SNAME("Search")));
	search_root->set_selectable(0, false);
	search_root->set_text(0, TTR("Search results"));
	const PackedStringArray &search_results = TranslationServer::get_singleton()->find_language(p_text);
	for (const String &code : search_results) {
		const String &name = TranslationServer::get_singleton()->get_language_name(code);
		TreeItem *t = lang_list->create_item(search_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Favorites")), BUTTON_FAVORITE_LANG, false, TTR("Add to favorites"));
		if (locale_valid && code == locale.language) {
			t->select(0);
			current_found = true;
		}
	}
	if (search_root->get_child_count() == 0) {
		search_root->set_visible(false);
	}

	// 2. Add translations.
	TreeItem *trans_root = lang_list->create_item(root);
	trans_root->set_icon(0, get_editor_theme_icon(SNAME("Translation")));
	trans_root->set_selectable(0, false);
	trans_root->set_text(0, TTR("Project translations"));
	const PackedStringArray &translations = GLOBAL_GET("internationalization/locale/translations");
	for (const String &path : translations) {
		if (!translation_cache.has(path)) {
			Ref<Translation> tr = ResourceLoader::load(path);
			if (tr.is_null()) {
				continue;
			}
			translation_cache[path] = tr->get_locale();
		}
		TranslationServer::Locale loc(translation_cache[path], false);
		const String &code = loc.language;
		const String &name = TranslationServer::get_singleton()->get_language_name(code);
		TreeItem *t = lang_list->create_item(trans_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		if (locale_valid && code == locale.language) {
			t->select(0);
			current_found = true;
		}
	}
	if (trans_root->get_child_count() == 0) {
		trans_root->set_visible(false);
	}

	// 3. Add favorites.
	TreeItem *fav_root = lang_list->create_item(root);
	fav_root->set_icon(0, get_editor_theme_icon(SNAME("Favorites")));
	fav_root->set_selectable(0, false);
	fav_root->set_text(0, TTR("Favorites"));
	const PackedStringArray &favorites = EDITOR_GET("interface/editor/favorite_language_codes");
	for (const String &code : favorites) {
		const String &name = TranslationServer::get_singleton()->get_language_name(code);
		TreeItem *t = lang_list->create_item(fav_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Unfavorite")), BUTTON_UNFAVORITE_LANG, false, TTR("Remove from favorites"));
		if (locale_valid && code == locale.language) {
			t->select(0);
			current_found = true;
		}
	}
	if (fav_root->get_child_count() == 0) {
		fav_root->set_visible(false);
	}

	//4. Add project custom codes.
	TreeItem *user_root = lang_list->create_item(root);
	user_root->set_icon(0, get_editor_theme_icon(SNAME("User")));
	user_root->set_selectable(0, false);
	user_root->set_text(0, TTR("Custom"));
	user_root->add_button(0, get_editor_theme_icon(SNAME("Add")), BUTTON_ADD_LANG, false, TTR("Add"));
	const Dictionary &custom = TranslationServer::get_singleton()->get_custom_language_codes();
	for (const Variant *key = custom.next(nullptr); key; key = custom.next(key)) {
		const String &code = *key;
		const String &name = custom[*key];
		TreeItem *t = lang_list->create_item(user_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Edit")), BUTTON_EDIT_LANG, false, TTR("Edit"));
		t->add_button(0, get_editor_theme_icon(SNAME("Remove")), BUTTON_REMOVE_LANG, false, TTR("Remove"));
		if (locale_valid && code == locale.language) {
			t->select(0);
			current_found = true;
		}
	}

	//5. Add current.
	if (!current_found && locale_valid) {
		search_root->set_visible(true);

		const String &name = TranslationServer::get_singleton()->get_language_name(locale.language);
		TreeItem *t = lang_list->create_item(search_root, 0);
		t->set_text(0, vformat("%s [%s]", name, locale.language));
		t->set_metadata(0, locale.language);
		t->select(0);
	}

	updating_lists = false;
}

void EditorLocaleDialog::_script_search(const String &p_text) {
	updating_lists = true;
	script_list->clear();

	TreeItem *root = script_list->create_item(nullptr);
	script_list->set_hide_root(true);
	bool current_found = false;

	// 0. Add "Default/Unspecified".
	TreeItem *default_root = script_list->create_item(root);
	default_root->set_text(0, TTR("Default/Unspecified"));
	default_root->set_icon(0, get_editor_theme_icon(SNAME("Marker")));
	default_root->set_metadata(0, "");
	if (locale.script.is_empty()) {
		default_root->select(0);
		current_found = true;
	}

	// 1. Add search results.
	TreeItem *search_root = script_list->create_item(root);
	search_root->set_icon(0, get_editor_theme_icon(SNAME("Search")));
	search_root->set_selectable(0, false);
	search_root->set_text(0, TTR("Search results"));
	const PackedStringArray &search_results = TranslationServer::get_singleton()->find_script(p_text);
	for (const String &code : search_results) {
		const String &name = TranslationServer::get_singleton()->get_script_name(code);
		TreeItem *t = script_list->create_item(search_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Favorites")), BUTTON_FAVORITE_SCRIPT, false, TTR("Add to favorites"));
		if (locale_valid && code == locale.script) {
			t->select(0);
			current_found = true;
		}
	}
	if (search_root->get_child_count() == 0) {
		search_root->set_visible(false);
	}

	// 2. Add translations.
	TreeItem *trans_root = script_list->create_item(root);
	trans_root->set_icon(0, get_editor_theme_icon(SNAME("Translation")));
	trans_root->set_selectable(0, false);
	trans_root->set_text(0, TTR("Project translations"));
	const PackedStringArray &translations = GLOBAL_GET("internationalization/locale/translations");
	for (const String &path : translations) {
		if (!translation_cache.has(path)) {
			Ref<Translation> tr = ResourceLoader::load(path);
			if (tr.is_null()) {
				continue;
			}
			translation_cache[path] = tr->get_locale();
		}
		TranslationServer::Locale loc(translation_cache[path], false);
		const String &code = loc.script;
		if (!code.is_empty()) {
			const String &name = TranslationServer::get_singleton()->get_script_name(code);
			TreeItem *t = script_list->create_item(trans_root);
			t->set_text(0, vformat("%s [%s]", name, code));
			t->set_metadata(0, code);
			if (locale_valid && code == locale.script) {
				t->select(0);
				current_found = true;
			}
		}
	}
	if (trans_root->get_child_count() == 0) {
		trans_root->set_visible(false);
	}

	// 3. Add favorites.
	TreeItem *fav_root = script_list->create_item(root);
	fav_root->set_icon(0, get_editor_theme_icon(SNAME("Favorites")));
	fav_root->set_selectable(0, false);
	fav_root->set_text(0, TTR("Favorites"));
	const PackedStringArray &favorites = EDITOR_GET("interface/editor/favorite_script_codes");
	for (const String &code : favorites) {
		const String &name = TranslationServer::get_singleton()->get_script_name(code);
		TreeItem *t = script_list->create_item(fav_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Unfavorite")), BUTTON_UNFAVORITE_SCRIPT, false, TTR("Remove from favorites"));
		if (locale_valid && code == locale.script) {
			t->select(0);
			current_found = true;
		}
	}
	if (fav_root->get_child_count() == 0) {
		fav_root->set_visible(false);
	}

	//4. Add current.
	if (!current_found && locale_valid && !locale.script.is_empty()) {
		search_root->set_visible(true);

		const String &name = TranslationServer::get_singleton()->get_script_name(locale.script);
		TreeItem *t = script_list->create_item(search_root, 0);
		t->set_text(0, vformat("%s [%s]", name, locale.script));
		t->set_metadata(0, locale.script);
		t->select(0);
	}

	updating_lists = false;
}

void EditorLocaleDialog::_country_search(const String &p_text) {
	updating_lists = true;
	country_list->clear();

	TreeItem *root = country_list->create_item(nullptr);
	country_list->set_hide_root(true);
	bool current_found = false;

	// 0. Add "Default/Unspecified".
	TreeItem *default_root = country_list->create_item(root);
	default_root->set_text(0, TTR("Default/Unspecified"));
	default_root->set_icon(0, get_editor_theme_icon(SNAME("Marker")));
	default_root->set_metadata(0, "");
	if (locale.country.is_empty()) {
		default_root->select(0);
		current_found = true;
	}

	// 1. Add search results.
	TreeItem *search_root = country_list->create_item(root);
	search_root->set_icon(0, get_editor_theme_icon(SNAME("Search")));
	search_root->set_selectable(0, false);
	search_root->set_text(0, TTR("Search results"));
	const PackedStringArray &search_results = TranslationServer::get_singleton()->find_country(p_text);
	for (const String &code : search_results) {
		const String &name = TranslationServer::get_singleton()->get_country_name(code);
		TreeItem *t = country_list->create_item(search_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Favorites")), BUTTON_FAVORITE_COUNTRY, false, TTR("Add to favorites"));
		if (locale_valid && code == locale.country) {
			t->select(0);
			current_found = true;
		}
	}
	if (search_root->get_child_count() == 0) {
		search_root->set_visible(false);
	}

	// 2. Add translations.
	TreeItem *trans_root = country_list->create_item(root);
	trans_root->set_icon(0, get_editor_theme_icon(SNAME("Translation")));
	trans_root->set_selectable(0, false);
	trans_root->set_text(0, TTR("Project translations"));
	const PackedStringArray &translations = GLOBAL_GET("internationalization/locale/translations");
	for (const String &path : translations) {
		if (!translation_cache.has(path)) {
			Ref<Translation> tr = ResourceLoader::load(path);
			if (tr.is_null()) {
				continue;
			}
			translation_cache[path] = tr->get_locale();
		}
		TranslationServer::Locale loc(translation_cache[path], false);
		const String &code = loc.country;
		if (!code.is_empty()) {
			const String &name = TranslationServer::get_singleton()->get_country_name(code);
			TreeItem *t = country_list->create_item(trans_root);
			t->set_text(0, vformat("%s [%s]", name, code));
			t->set_metadata(0, code);
			if (locale_valid && code == locale.country) {
				t->select(0);
				current_found = true;
			}
		}
	}
	if (trans_root->get_child_count() == 0) {
		trans_root->set_visible(false);
	}

	// 3. Add favorites.
	TreeItem *fav_root = country_list->create_item(root);
	fav_root->set_icon(0, get_editor_theme_icon(SNAME("Favorites")));
	fav_root->set_selectable(0, false);
	fav_root->set_text(0, TTR("Favorites"));
	const PackedStringArray &favorites = EDITOR_GET("interface/editor/favorite_country_codes");
	for (const String &code : favorites) {
		const String &name = TranslationServer::get_singleton()->get_country_name(code);
		TreeItem *t = country_list->create_item(fav_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Unfavorite")), BUTTON_UNFAVORITE_COUNTRY, false, TTR("Remove from favorites"));
		if (locale_valid && code == locale.country) {
			t->select(0);
			current_found = true;
		}
	}
	if (fav_root->get_child_count() == 0) {
		fav_root->set_visible(false);
	}

	//4. Add project custom codes.
	TreeItem *user_root = country_list->create_item(root);
	user_root->set_icon(0, get_editor_theme_icon(SNAME("User")));
	user_root->set_selectable(0, false);
	user_root->set_text(0, TTR("Custom"));
	user_root->add_button(0, get_editor_theme_icon(SNAME("Add")), BUTTON_ADD_COUNTRY, false, TTR("Add"));
	const Dictionary &custom = TranslationServer::get_singleton()->get_custom_country_codes();
	for (const Variant *key = custom.next(nullptr); key; key = custom.next(key)) {
		const String &code = *key;
		const String &name = custom[*key];
		TreeItem *t = country_list->create_item(user_root);
		t->set_text(0, vformat("%s [%s]", name, code));
		t->set_metadata(0, code);
		t->add_button(0, get_editor_theme_icon(SNAME("Edit")), BUTTON_EDIT_COUNTRY, false, TTR("Edit"));
		t->add_button(0, get_editor_theme_icon(SNAME("Remove")), BUTTON_REMOVE_COUNTRY, false, TTR("Remove"));
		if (locale_valid && code == locale.country) {
			t->select(0);
			current_found = true;
		}
	}

	//5. Add current.
	if (!current_found && locale_valid && !locale.country.is_empty()) {
		search_root->set_visible(true);

		const String &name = TranslationServer::get_singleton()->get_country_name(locale.country);
		TreeItem *t = country_list->create_item(search_root, 0);
		t->set_text(0, vformat("%s [%s]", name, locale.country));
		t->set_metadata(0, locale.country);
		t->select(0);
	}

	updating_lists = false;
}

void EditorLocaleDialog::_item_selected() {
	if (updating_lists) {
		return;
	}

	TreeItem *lang_item = lang_list->get_selected();
	if (lang_item) {
		locale.language = lang_item->get_metadata(0);
	}

	TreeItem *script_item = script_list->get_selected();
	if (script_item) {
		locale.script = script_item->get_metadata(0);
	} else {
		locale.script = StringName();
	}

	TreeItem *country_item = country_list->get_selected();
	if (country_item) {
		locale.country = country_item->get_metadata(0);
	} else {
		locale.country = StringName();
	}

	locale.variant = variant_code->get_text();
	locale_display->set_text(TTR("Locale Code") + ": " + locale.operator String());
}

void EditorLocaleDialog::_add_lang(const String &p_code, const String &p_name) {
	Dictionary custom = TranslationServer::get_singleton()->get_custom_language_codes();
	custom[p_code] = p_name;
	TranslationServer::get_singleton()->set_custom_language_codes(custom);

	updating_settings = true;
	ProjectSettings::get_singleton()->set("internationalization/locale/custom_language_codes", custom);
	ProjectSettings::get_singleton()->save();
	updating_settings = false;
	_lang_search(lang_search->get_text());
}

void EditorLocaleDialog::_remove_lang(const String &p_code, bool p_edit) {
	Dictionary custom = TranslationServer::get_singleton()->get_custom_language_codes();
	custom.erase(p_code);
	TranslationServer::get_singleton()->set_custom_language_codes(custom);

	if (!p_edit) {
		updating_settings = true;
		ProjectSettings::get_singleton()->set("internationalization/locale/custom_language_codes", custom);
		ProjectSettings::get_singleton()->save();
		updating_settings = false;
		_lang_search(lang_search->get_text());
	}
}

void EditorLocaleDialog::_add_country(const String &p_code, const String &p_name) {
	Dictionary custom = TranslationServer::get_singleton()->get_custom_country_codes();
	custom[p_code] = p_name;
	TranslationServer::get_singleton()->set_custom_country_codes(custom);

	updating_settings = true;
	ProjectSettings::get_singleton()->set("internationalization/locale/custom_country_codes", custom);
	ProjectSettings::get_singleton()->save();
	updating_settings = false;
	_country_search(country_search->get_text());
}

void EditorLocaleDialog::_remove_country(const String &p_code, bool p_edit) {
	Dictionary custom = TranslationServer::get_singleton()->get_custom_country_codes();
	custom.erase(p_code);
	TranslationServer::get_singleton()->set_custom_country_codes(custom);

	if (!p_edit) {
		updating_settings = true;
		ProjectSettings::get_singleton()->set("internationalization/locale/custom_country_codes", custom);
		ProjectSettings::get_singleton()->save();
		updating_settings = false;
		_country_search(country_search->get_text());
	}
}

void EditorLocaleDialog::_button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_mouse_button_index) {
	ERR_FAIL_COND(!p_item);

	const String &code = p_item->get_metadata(0);
	switch (p_id) {
		case BUTTON_FAVORITE_LANG: {
			PackedStringArray favorites = EDITOR_GET("interface/editor/favorite_language_codes");
			if (!favorites.has(code)) {
				favorites.push_back(code);
				EditorSettings::get_singleton()->set("interface/editor/favorite_language_codes", favorites);

				_lang_search(lang_search->get_text());
			}
		} break;
		case BUTTON_UNFAVORITE_LANG: {
			PackedStringArray favorites = EDITOR_GET("interface/editor/favorite_language_codes");
			if (favorites.has(code)) {
				favorites.erase(code);
				EditorSettings::get_singleton()->set("interface/editor/favorite_language_codes", favorites);

				_lang_search(lang_search->get_text());
			}
		} break;
		case BUTTON_ADD_LANG: {
			add_dialog_lang->set_title(TTR("Add Custom Language Code"));
			add_dialog_lang->set_data(String(), String(), true, false);
			add_dialog_lang->popup_centered();
		} break;
		case BUTTON_EDIT_LANG: {
			add_dialog_lang->set_title(TTR("Edit Custom Language Code"));
			add_dialog_lang->set_data(code, TranslationServer::get_singleton()->get_language_name(code), true, true);
			add_dialog_lang->popup_centered();
		} break;
		case BUTTON_REMOVE_LANG: {
			_remove_lang(code, false);
		} break;
		case BUTTON_FAVORITE_COUNTRY: {
			PackedStringArray favorites = EDITOR_GET("interface/editor/favorite_country_codes");
			if (!favorites.has(code)) {
				favorites.push_back(code);
				EditorSettings::get_singleton()->set("interface/editor/favorite_country_codes", favorites);

				_country_search(country_search->get_text());
			}
		} break;
		case BUTTON_UNFAVORITE_COUNTRY: {
			PackedStringArray favorites = EDITOR_GET("interface/editor/favorite_country_codes");
			if (favorites.has(code)) {
				favorites.erase(code);
				EditorSettings::get_singleton()->set("interface/editor/favorite_country_codes", favorites);

				_country_search(country_search->get_text());
			}
		} break;
		case BUTTON_ADD_COUNTRY: {
			add_dialog_country->set_title(TTR("Add Custom Country Code"));
			add_dialog_country->set_data(String(), String(), false, false);
			add_dialog_country->popup_centered();
		} break;
		case BUTTON_EDIT_COUNTRY: {
			add_dialog_country->set_title(TTR("Edit Custom Country Code"));
			add_dialog_country->set_data(code, TranslationServer::get_singleton()->get_country_name(code), false, true);
			add_dialog_country->popup_centered();
		} break;
		case BUTTON_REMOVE_COUNTRY: {
			_remove_country(code, false);
		} break;
		case BUTTON_FAVORITE_SCRIPT: {
			PackedStringArray favorites = EDITOR_GET("interface/editor/favorite_script_codes");
			if (!favorites.has(code)) {
				favorites.push_back(code);
				EditorSettings::get_singleton()->set("interface/editor/favorite_script_codes", favorites);

				_script_search(script_search->get_text());
			}
		} break;
		case BUTTON_UNFAVORITE_SCRIPT: {
			PackedStringArray favorites = EDITOR_GET("interface/editor/favorite_script_codes");
			if (favorites.has(code)) {
				favorites.erase(code);
				EditorSettings::get_singleton()->set("interface/editor/favorite_script_codes", favorites);

				_script_search(script_search->get_text());
			}
		} break;
		default:
			break;
	}
}

void EditorLocaleDialog::_varinat_selected(const String &p_text) {
	_item_selected();
}

void EditorLocaleDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_update_cache();
			ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &EditorLocaleDialog::_update_cache));
		} break;
		case NOTIFICATION_ENTER_TREE: {
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			lang_search->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			script_search->set_right_icon(get_editor_theme_icon(SNAME("Search")));
			country_search->set_right_icon(get_editor_theme_icon(SNAME("Search")));

			_lang_search(lang_search->get_text());
			_script_search(script_search->get_text());
			_country_search(country_search->get_text());
			_item_selected();
		} break;
		default: {
		} break;
	}
}

EditorLocaleDialog::EditorLocaleDialog() {
	set_title(TTRC("Select a Locale"));

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *hb_main = memnew(HBoxContainer);
	hb_main->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(hb_main);

	VBoxContainer *lang_vb = memnew(VBoxContainer);
	lang_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb_main->add_child(lang_vb);
	Label *lang_lbl = memnew(Label);
	lang_lbl->set_text(TTR("Language"));
	lang_vb->add_child(lang_lbl);
	lang_search = memnew(LineEdit);
	lang_search->set_placeholder(TTR("Language"));
	lang_search->set_accessibility_name(TTR("Language Filter"));
	lang_search->connect(SceneStringName(text_changed), callable_mp(this, &EditorLocaleDialog::_lang_search));
	lang_vb->add_child(lang_search);
	lang_list = memnew(Tree);
	lang_list->set_accessibility_name(TTR("Language List"));
	lang_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	lang_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	lang_list->connect("cell_selected", callable_mp(this, &EditorLocaleDialog::_item_selected));
	lang_list->connect("button_clicked", callable_mp(this, &EditorLocaleDialog::_button_clicked));
	lang_list->set_hide_folding(true);
	lang_list->set_columns(1);
	lang_vb->add_child(lang_list);

	script_vb = memnew(VBoxContainer);
	script_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb_main->add_child(script_vb);
	Label *script_lbl = memnew(Label);
	script_lbl->set_text(TTR("Script"));
	script_vb->add_child(script_lbl);
	script_search = memnew(LineEdit);
	script_search->set_placeholder(TTR("Script"));
	script_search->set_accessibility_name(TTR("Script Filter"));
	script_search->connect(SceneStringName(text_changed), callable_mp(this, &EditorLocaleDialog::_script_search));
	script_vb->add_child(script_search);
	script_list = memnew(Tree);
	script_list->set_accessibility_name(TTR("Script List"));
	script_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	script_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	script_list->connect("cell_selected", callable_mp(this, &EditorLocaleDialog::_item_selected));
	script_list->connect("button_clicked", callable_mp(this, &EditorLocaleDialog::_button_clicked));
	script_list->set_hide_folding(true);
	script_list->set_columns(1);
	script_vb->add_child(script_list);
	script_vb->set_visible(false);

	VBoxContainer *country_vb = memnew(VBoxContainer);
	country_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb_main->add_child(country_vb);
	Label *country_lbl = memnew(Label);
	country_lbl->set_text(TTR("Country"));
	country_vb->add_child(country_lbl);
	country_search = memnew(LineEdit);
	country_search->set_placeholder(TTR("Country"));
	country_search->set_accessibility_name(TTR("Country Filter"));
	country_search->connect(SceneStringName(text_changed), callable_mp(this, &EditorLocaleDialog::_country_search));
	country_vb->add_child(country_search);
	country_list = memnew(Tree);
	country_list->set_accessibility_name(TTR("Country List"));
	country_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	country_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	country_list->connect("cell_selected", callable_mp(this, &EditorLocaleDialog::_item_selected));
	country_list->connect("button_clicked", callable_mp(this, &EditorLocaleDialog::_button_clicked));
	country_list->set_hide_folding(true);
	country_list->set_columns(1);
	country_vb->add_child(country_list);

	variant_lbl = memnew(Label);
	variant_lbl->set_text(TTR("Variant"));
	variant_lbl->set_visible(false);
	country_vb->add_child(variant_lbl);
	variant_code = memnew(LineEdit);
	variant_code->set_accessibility_name(TTR("Variant"));
	variant_code->connect(SceneStringName(text_changed), callable_mp(this, &EditorLocaleDialog::_varinat_selected));
	variant_code->set_visible(false);
	country_vb->add_child(variant_code);

	HBoxContainer *hb_bottom = memnew(HBoxContainer);
	vb->add_child(hb_bottom);

	locale_display = memnew(Label);
	locale_display->set_accessibility_name(TTR("Locale Code"));
	locale_display->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb_bottom->add_child(locale_display);

	advanced = memnew(CheckButton);
	advanced->set_text(TTR("Advanced"));
	advanced->set_toggle_mode(true);
	advanced->set_pressed(false);
	advanced->connect(SceneStringName(toggled), callable_mp(this, &EditorLocaleDialog::_toggle_advanced));
	hb_bottom->add_child(advanced);

	add_dialog_lang = memnew(EditorAddCustomLocaleDialog);
	add_dialog_lang->connect("add_code", callable_mp(this, &EditorLocaleDialog::_add_lang));
	add_dialog_lang->connect("remove_code", callable_mp(this, &EditorLocaleDialog::_remove_lang));
	add_child(add_dialog_lang);

	add_dialog_country = memnew(EditorAddCustomLocaleDialog);
	add_dialog_country->connect("add_code", callable_mp(this, &EditorLocaleDialog::_add_country));
	add_dialog_country->connect("remove_code", callable_mp(this, &EditorLocaleDialog::_remove_country));
	add_child(add_dialog_country);

	set_ok_button_text(TTR("Select"));
}
