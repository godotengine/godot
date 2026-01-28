/**************************************************************************/
/*  editor_translation_preview_button.cpp                                 */
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

#include "editor_translation_preview_button.h"

#include "core/string/translation_server.h"
#include "editor/editor_node.h"

void EditorTranslationPreviewButton::_update() {
	const String &locale = EditorNode::get_singleton()->get_preview_locale();

	if (locale.is_empty()) {
		hide();
		return;
	}

	const String name = TranslationServer::get_singleton()->get_locale_name(locale);
	set_text(vformat(TTR("Previewing: %s"), name == locale ? locale : name + " [" + locale + "]"));
	show();
}

void EditorTranslationPreviewButton::pressed() {
	EditorNode::get_singleton()->set_preview_locale(String());
}

void EditorTranslationPreviewButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			set_button_icon(get_editor_theme_icon(SNAME("Translation")));
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			_update();
		} break;

		case NOTIFICATION_READY: {
			EditorNode::get_singleton()->connect("preview_locale_changed", callable_mp(this, &EditorTranslationPreviewButton::_update));
		} break;
	}
}

EditorTranslationPreviewButton::EditorTranslationPreviewButton() {
	set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
	set_accessibility_name(TTRC("Disable Translation Preview"));
	set_tooltip_text(TTRC("Previewing translation. Click to disable."));
	set_focus_mode(FOCUS_NONE);
	set_visible(false);
}
