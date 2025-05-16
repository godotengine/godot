/**************************************************************************/
/*  editor_quick_movie_maker_config.cpp                                   */
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

#include "editor_quick_movie_maker_config.h"

#include "core/config/project_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/project_settings_editor.h"
#include "editor/themes/editor_scale.h"

void EditorQuickMovieMakerConfig::_close_requested() {
	if (movie_path_was_changed) {
		_update_movie_file_path(filepath_select->get_edit()->get_text());
	}
}

void EditorQuickMovieMakerConfig::_path_edit_focus_exited() {
	if (movie_path_was_changed) {
		_update_movie_file_path(filepath_select->get_edit()->get_text());
	}
}

void EditorQuickMovieMakerConfig::_path_edit_text_submitted(const String &p_new_text) {
	_update_movie_file_path(p_new_text);
}

void EditorQuickMovieMakerConfig::_path_edit_text_changed(const String &p_new_text) {
	movie_path_was_changed = true;
}

void EditorQuickMovieMakerConfig::_update_movie_file_path(const String &p_new_text) {
	// Immediately save the project settings value.
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set movie_file"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), "editor/movie_writer/movie_file", p_new_text);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "editor/movie_writer/movie_file", GLOBAL_GET("editor/movie_writer/movie_file"));
	undo_redo->add_undo_property(filepath_select->get_edit(), "text", GLOBAL_GET("editor/movie_writer/movie_file"));
	undo_redo->commit_action();
}

void EditorQuickMovieMakerConfig::_visibility_changed() {
	if (is_visible()) {
		// Populate the field with the current Movie Maker path.
		Variant current_path_variant = ProjectSettings::get_singleton()->get("editor/movie_writer/movie_file");
		String current_path = current_path_variant;
		filepath_select->get_edit()->set_text(current_path);

		// Set to true when the LineEdit contents are changed;
		// tells Godot to actually change the property string.
		movie_path_was_changed = false;
	}
}

void EditorQuickMovieMakerConfig::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE:
			connect("visibility_changed", callable_mp(this, &EditorQuickMovieMakerConfig::_visibility_changed));
			connect("close_requested", callable_mp(this, &EditorQuickMovieMakerConfig::_close_requested));
			break;
		case NOTIFICATION_THEME_CHANGED:
			set_min_size(Size2i(600.0, 0.0) * EDSCALE);
			break;
	}
}

void EditorQuickMovieMakerConfig::_open_settings_pressed() {
	set_visible(false);
	ProjectSettingsEditor::get_singleton()->popup_project_settings(true);
	ProjectSettingsEditor::get_singleton()->set_general_page("editor/movie_writer");
}

EditorQuickMovieMakerConfig::EditorQuickMovieMakerConfig() {
	set_transient(true);

	parts_container = memnew(VBoxContainer);
	add_child(parts_container);

	path_container = memnew(VBoxContainer);
	parts_container->add_child(path_container);

	path_label = memnew(Label);
	path_container->add_child(path_label);
	path_label->set_text("Movie output path");

	// NOTE: Use EditorPropertyPath as reference.
	filepath_select = memnew(EditorFilepathSelect);
	parts_container->add_child(filepath_select);
	filepath_select->get_edit()->connect("text_submitted", callable_mp(this, &EditorQuickMovieMakerConfig::_path_edit_text_submitted));
	filepath_select->get_edit()->connect("text_changed", callable_mp(this, &EditorQuickMovieMakerConfig::_path_edit_text_changed));

	String extensions_string = ProjectSettings::get_singleton()->get_custom_property_info().get(StringName("editor/movie_writer/movie_file")).hint_string;
	Vector<String> extensions = extensions_string.split(",");
	for (int i = 0; i < extensions.size(); i++) {
		String e = extensions[i].strip_edges();
		if (!e.is_empty()) {
			filepath_select->get_dialog()->add_filter(e);
		}
	}

	open_settings_button = memnew(Button);
	parts_container->add_child(open_settings_button);
	open_settings_button->set_text("Open Project Settings...");
	open_settings_button->connect("pressed", callable_mp(this, &EditorQuickMovieMakerConfig::_open_settings_pressed));
}
