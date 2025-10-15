/**************************************************************************/
/*  editor_locale_dialog.h                                                */
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

#pragma once

#include "core/string/translation_server.h"
#include "scene/gui/dialogs.h"

class Button;
class HBoxContainer;
class VBoxContainer;
class LineEdit;
class Tree;
class TreeItem;
class OptionButton;

class EditorAddCustomLocaleDialog : public ConfirmationDialog {
	GDCLASS(EditorAddCustomLocaleDialog, ConfirmationDialog);

	String old_code;
	String old_name;
	LineEdit *code = nullptr;
	LineEdit *name = nullptr;
	Label *warn_lbl = nullptr;
	bool is_lang = false;
	bool edit = false;

protected:
	static void _bind_methods();
	virtual void ok_pressed() override;

	void _validate_code(const String &p_text);

public:
	void set_data(const String &p_code, const String &p_name, bool p_is_lang, bool p_edit);
	EditorAddCustomLocaleDialog();
};

class EditorLocaleDialog : public ConfirmationDialog {
	GDCLASS(EditorLocaleDialog, ConfirmationDialog);

	enum TreeButtonIDs {
		BUTTON_FAVORITE_LANG,
		BUTTON_UNFAVORITE_LANG,
		BUTTON_ADD_LANG,
		BUTTON_EDIT_LANG,
		BUTTON_REMOVE_LANG,
		BUTTON_FAVORITE_COUNTRY,
		BUTTON_UNFAVORITE_COUNTRY,
		BUTTON_ADD_COUNTRY,
		BUTTON_EDIT_COUNTRY,
		BUTTON_REMOVE_COUNTRY,
		BUTTON_FAVORITE_SCRIPT,
		BUTTON_UNFAVORITE_SCRIPT,
	};

	bool updating_lists = false;
	bool updating_settings = false;
	TranslationServer::Locale locale;
	bool locale_valid = false;

	VBoxContainer *script_vb = nullptr;

	LineEdit *lang_search = nullptr;
	Tree *lang_list = nullptr;

	LineEdit *script_search = nullptr;
	Tree *script_list = nullptr;
	Label *variant_lbl = nullptr;
	LineEdit *variant_code = nullptr;

	LineEdit *country_search = nullptr;
	Tree *country_list = nullptr;

	Label *locale_display = nullptr;
	Button *advanced = nullptr;

	EditorAddCustomLocaleDialog *add_dialog_lang = nullptr;
	EditorAddCustomLocaleDialog *add_dialog_country = nullptr;

	HashMap<String, String> translation_cache;

protected:
	static void _bind_methods();
	virtual void _post_popup() override;
	virtual void ok_pressed() override;
	void _notification(int p_what);

	void _update_cache();
	void _toggle_advanced(bool p_checked);

	void _lang_search(const String &p_text);
	void _script_search(const String &p_text);
	void _country_search(const String &p_text);

	void _item_selected();
	void _varinat_selected(const String &p_text);
	void _button_clicked(TreeItem *p_item, int p_column, int p_id, MouseButton p_mouse_button_index);

	void _add_lang(const String &p_code, const String &p_name);
	void _remove_lang(const String &p_code, bool p_edit = false);
	void _add_country(const String &p_code, const String &p_name);
	void _remove_country(const String &p_code, bool p_edit = false);

public:
	EditorLocaleDialog();

	void set_locale(const String &p_locale);
	void popup_locale_dialog();
};
