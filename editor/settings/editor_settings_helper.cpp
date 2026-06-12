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
	ERR_FAIL_COND_MSG(singleton.is_valid(), "Can't recreate EditorSettingsHelper as it already exists.");

	singleton.instantiate();
	SceneTree::get_singleton()->connect("node_added", callable_mp(singleton.ptr(), &EditorSettingsHelper::_scene_tree_node_added));
	SceneTree::get_singleton()->connect("node_removed", callable_mp(singleton.ptr(), &EditorSettingsHelper::_scene_tree_node_removed));
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(singleton.ptr(), &EditorSettingsHelper::_settings_changed));
}

void EditorSettingsHelper::destroy() {
	if (singleton.is_null()) {
		return;
	}

	DEV_ASSERT(singleton->text_edits.is_empty());
	DEV_ASSERT(singleton->line_edits.is_empty());

	EditorSettings::get_singleton()->disconnect("settings_changed", callable_mp(singleton.ptr(), &EditorSettingsHelper::_settings_changed));
	SceneTree::get_singleton()->disconnect("node_added", callable_mp(singleton.ptr(), &EditorSettingsHelper::_scene_tree_node_added));
	SceneTree::get_singleton()->disconnect("node_removed", callable_mp(singleton.ptr(), &EditorSettingsHelper::_scene_tree_node_removed));
	singleton = Ref<EditorSettingsHelper>();
}

void EditorSettingsHelper::_settings_changed() {
	const bool middle_mouse_paste_enabled = EDITOR_GET("text_editor/behavior/general/middle_mouse_paste");
	for (TextEdit *E : text_edits) {
		E->set_middle_mouse_paste_enabled(middle_mouse_paste_enabled);
	}
	for (LineEdit *E : line_edits) {
		E->set_middle_mouse_paste_enabled(middle_mouse_paste_enabled);
	}

	if (EditorSettings::get_singleton()->check_changed_settings_in_group("text_editor/appearance/caret")) {
		const bool caret_blink = EDITOR_GET("text_editor/appearance/caret/caret_blink");
		const float caret_blink_interval = EDITOR_GET("text_editor/appearance/caret/caret_blink_interval");
		for (LineEdit *E : line_edits) {
			E->set_caret_blink_enabled(caret_blink);
			E->set_caret_blink_interval(caret_blink_interval);
		}
	}
}

void EditorSettingsHelper::_scene_tree_node_added(Node *p_node) {
	if (p_node->is_part_of_edited_scene()) {
		return;
	}

	TextEdit *text_edit = Object::cast_to<TextEdit>(p_node);
	if (unlikely(text_edit)) {
		text_edits.push_back(text_edit);
		text_edit->set_middle_mouse_paste_enabled(EDITOR_GET("text_editor/behavior/general/middle_mouse_paste"));
		return;
	}

	LineEdit *line_edit = Object::cast_to<LineEdit>(p_node);
	if (unlikely(line_edit)) {
		line_edits.push_back(line_edit);
		line_edit->set_middle_mouse_paste_enabled(EDITOR_GET("text_editor/behavior/general/middle_mouse_paste"));
		line_edit->set_caret_blink_enabled(EDITOR_GET("text_editor/appearance/caret/caret_blink"));
		line_edit->set_caret_blink_interval(EDITOR_GET("text_editor/appearance/caret/caret_blink_interval"));
		return;
	}
}

void EditorSettingsHelper::_scene_tree_node_removed(Node *p_node) {
	if (p_node->is_part_of_edited_scene()) {
		return;
	}

	TextEdit *text_edit = Object::cast_to<TextEdit>(p_node);
	if (unlikely(text_edit)) {
		text_edits.erase(text_edit);
		return;
	}

	LineEdit *line_edit = Object::cast_to<LineEdit>(p_node);
	if (unlikely(line_edit)) {
		line_edits.erase(line_edit);
		return;
	}
}
