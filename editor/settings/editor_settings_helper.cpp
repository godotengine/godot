/**************************************************************************/
/*  editor_settings_helper.cpp                                            */
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

#include "editor/settings/editor_settings_helper.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/text_edit.h"

static Ref<EditorSettingsHelper> singleton;

void EditorSettingsHelper::create() {
	if (singleton.ptr()) {
		ERR_PRINT("Can't recreate EditorSettingsHelper as it already exists.");
		return;
	}
	singleton.instantiate();
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(singleton.ptr(), &EditorSettingsHelper::_settings_changed));
}

EditorSettingsHelper *EditorSettingsHelper::get_singleton() {
	return singleton.ptr();
}

void EditorSettingsHelper::destroy() {
	if (!singleton.ptr()) {
		return;
	}
	DEV_ASSERT(singleton->text_edits.is_empty());
	DEV_ASSERT(singleton->line_edits.is_empty());
	EditorSettings::get_singleton()->disconnect("settings_changed", callable_mp(singleton.ptr(), &EditorSettingsHelper::_settings_changed));
	singleton = Ref<EditorSettingsHelper>();
}

void EditorSettingsHelper::_settings_changed() {
	const bool middle_mouse_paste_enabled = EDITOR_GET("text_editor/behavior/general/middle_mouse_paste");
	for (List<TextEdit *>::Element *E = text_edits.front(); E; E = E->next()) {
		E->get()->set_middle_mouse_paste_enabled(middle_mouse_paste_enabled);
	}
	for (List<LineEdit *>::Element *E = line_edits.front(); E; E = E->next()) {
		E->get()->set_middle_mouse_paste_enabled(middle_mouse_paste_enabled);
	}

	if (EditorSettings::get_singleton()->check_changed_settings_in_group("text_editor/appearance/caret")) {
		const bool caret_blink = EDITOR_GET("text_editor/appearance/caret/caret_blink");
		const float caret_blink_interval = EDITOR_GET("text_editor/appearance/caret/caret_blink_interval");
		for (List<LineEdit *>::Element *E = line_edits.front(); E; E = E->next()) {
			E->get()->set_caret_blink_enabled(caret_blink);
			E->get()->set_caret_blink_interval(caret_blink_interval);
		}
	}
}

void EditorSettingsHelper::postinitialize_text_edit(TextEdit *p_text_edit) {
	p_text_edit->connect(SceneStringName(tree_entered), callable_mp(this, &EditorSettingsHelper::_text_edit_tree_entered).bind(p_text_edit));
}

void EditorSettingsHelper::predelete_text_edit(TextEdit *p_text_edit) {
	text_edits.erase(p_text_edit);
}

void EditorSettingsHelper::_text_edit_tree_entered(TextEdit *p_text_edit) {
	if (!p_text_edit->is_part_of_edited_scene()) {
		text_edits.push_front(p_text_edit);
		p_text_edit->connect(SceneStringName(tree_exited), callable_mp(this, &EditorSettingsHelper::_text_edit_tree_exited).bind(p_text_edit), CONNECT_ONE_SHOT);
		p_text_edit->set_middle_mouse_paste_enabled(EDITOR_GET("text_editor/behavior/general/middle_mouse_paste"));
	}
}

void EditorSettingsHelper::_text_edit_tree_exited(TextEdit *p_text_edit) {
	text_edits.erase(p_text_edit);
}

void EditorSettingsHelper::postinitialize_line_edit(LineEdit *p_line_edit) {
	p_line_edit->connect(SceneStringName(tree_entered), callable_mp(this, &EditorSettingsHelper::_line_edit_tree_entered).bind(p_line_edit));
}

void EditorSettingsHelper::predelete_line_edit(LineEdit *p_line_edit) {
	line_edits.erase(p_line_edit);
}

void EditorSettingsHelper::_line_edit_tree_entered(LineEdit *p_line_edit) {
	if (!p_line_edit->is_part_of_edited_scene()) {
		line_edits.push_front(p_line_edit);
		p_line_edit->connect(SceneStringName(tree_exited), callable_mp(this, &EditorSettingsHelper::_line_edit_tree_exited).bind(p_line_edit), CONNECT_ONE_SHOT);
		p_line_edit->set_middle_mouse_paste_enabled(EDITOR_GET("text_editor/behavior/general/middle_mouse_paste"));
		p_line_edit->set_caret_blink_enabled(EDITOR_GET("text_editor/appearance/caret/caret_blink"));
		p_line_edit->set_caret_blink_interval(EDITOR_GET("text_editor/appearance/caret/caret_blink_interval"));
	}
}

void EditorSettingsHelper::_line_edit_tree_exited(LineEdit *p_line_edit) {
	line_edits.erase(p_line_edit);
}
