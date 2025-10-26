/**************************************************************************/
/*  editor_translation_preview_menu.cpp                                   */
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

#include "editor_translation_preview_menu.h"

#include "core/string/translation_server.h"
#include "editor/editor_node.h"

void EditorTranslationPreviewMenu::_prepare() {
	const String current_preview_locale = EditorNode::get_singleton()->get_preview_locale();

	clear();
	reset_size();

	add_radio_check_item(TTRC("None"));
	set_item_metadata(-1, "");
	if (current_preview_locale.is_empty()) {
		set_item_checked(-1, true);
	}

	add_separator();

	const Vector<String> locales = TranslationServer::get_singleton()->get_loaded_locales();
	if (locales.is_empty()) {
		add_item(TTRC("No Translations Configured"));
		set_item_tooltip(-1, TTRC("You can add translations in the Project Settings."));
		set_item_disabled(-1, true);
		return;
	}

	for (const String &locale : locales) {
		const String name = TranslationServer::get_singleton()->get_locale_name(locale);
		add_radio_check_item(name == locale ? name : name + " [" + locale + "]");
		set_item_auto_translate_mode(-1, AUTO_TRANSLATE_MODE_DISABLED);
		set_item_metadata(-1, locale);
		if (locale == current_preview_locale) {
			set_item_checked(-1, true);
		}
	}
}

void EditorTranslationPreviewMenu::_pressed(int p_index) {
	for (int i = 0; i < get_item_count(); i++) {
		set_item_checked(i, i == p_index);
	}
	EditorNode::get_singleton()->set_preview_locale(get_item_metadata(p_index));
}

void EditorTranslationPreviewMenu::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect("about_to_popup", callable_mp(this, &EditorTranslationPreviewMenu::_prepare));
			connect("index_pressed", callable_mp(this, &EditorTranslationPreviewMenu::_pressed));
		} break;
	}
}

EditorTranslationPreviewMenu::EditorTranslationPreviewMenu() {
	set_hide_on_checkable_item_selection(false);
}
