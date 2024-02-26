/**************************************************************************/
/*  editor_global_variables_settings.cpp                                  */
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

#include "editor_global_variables_settings.h"

#include "core/config/project_settings.h"
#include "core/core_constants.h"
#include "core/object/script_language.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"

#include "scene/2d/node_2d.h"

void EditorGlobalVariablesSettings::_global_variable_add() {
	global_variable_add(global_variable_add_name->get_text());
	global_variable_add_name->set_text("");
	add_global_variable->set_disabled(true);
}

void EditorGlobalVariablesSettings::_global_variable_selected() {
	TreeItem *ti = tree->get_selected();

	if (!ti) {
		return;
	}

	selected_global_variable = "global_variables/" + ti->get_text(0);
}

void EditorGlobalVariablesSettings::_global_variable_edited() {
	if (updating_global_variables) {
		return;
	}

	TreeItem *ti = tree->get_edited();
	int column = tree->get_edited_column();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

	if (column == 0) {
		// Edit the name.
		String name = ti->get_text(0);
		String old_name = selected_global_variable.get_slice("/", 1);

		if (name == old_name) {
			return;
		}

		String error;
		if (!EditorGlobalVariablesSettings::global_variable_name_is_valid(name, &error)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(error);
			return;
		}

		if (ProjectSettings::get_singleton()->has_setting("global_variables/" + name)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(vformat(TTR("Singleton '%s' already exists!"), name));
			return;
		}

		updating_global_variables = true;

		name = "global_variables/" + name;
		Variant value = GLOBAL_GET(name);

		undo_redo->create_action(TTR("Rename Singleton"));

		undo_redo->add_do_property(ProjectSettings::get_singleton(), name, value);
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", selected_global_variable);

		undo_redo->add_undo_property(ProjectSettings::get_singleton(), selected_global_variable, value);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", name);

		undo_redo->add_do_method(this, "call_deferred", "update_global_variables");
		undo_redo->add_undo_method(this, "call_deferred", "update_global_variables");

		undo_redo->add_do_method(this, "emit_signal", global_variables_changed);
		undo_redo->add_undo_method(this, "emit_signal", global_variables_changed);

		undo_redo->commit_action();

		selected_global_variable = name;
	} else if (column == 1) {
		// Edit the type.
		String type = ti->get_text(1);
		String old_type = GLOBAL_GET(selected_global_variable);

		if (type == old_type) {
			return;
		}

		updating_global_variables = true;

		undo_redo->create_action(TTR("Rename Singleton"));

		undo_redo->add_do_property(ProjectSettings::get_singleton(), selected_global_variable, type);
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), selected_global_variable, old_type);

		undo_redo->add_do_method(this, "call_deferred", "update_global_variables");
		undo_redo->add_undo_method(this, "call_deferred", "update_global_variables");

		undo_redo->add_do_method(this, "emit_signal", global_variables_changed);
		undo_redo->add_undo_method(this, "emit_signal", global_variables_changed);

		undo_redo->commit_action();
	}

	updating_global_variables = false;
}

void EditorGlobalVariablesSettings::_global_variable_button_pressed(Object *p_item, int p_column, int p_button, MouseButton p_mouse_button) {
	if (p_mouse_button != MouseButton::LEFT) {
		return;
	}
	TreeItem *ti = Object::cast_to<TreeItem>(p_item);

	String name = "global_variables/" + ti->get_text(0);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

	switch (p_button) {
		case BUTTON_DELETE: {
			int order = ProjectSettings::get_singleton()->get_order(name);

			undo_redo->create_action(TTR("Remove Singleton"));

			undo_redo->add_do_property(ProjectSettings::get_singleton(), name, Variant());

			undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, GLOBAL_GET(name));
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", name, order);

			undo_redo->add_do_method(this, "update_global_variables");
			undo_redo->add_undo_method(this, "update_global_variables");

			undo_redo->add_do_method(this, "emit_signal", global_variables_changed);
			undo_redo->add_undo_method(this, "emit_signal", global_variables_changed);

			undo_redo->commit_action();
		} break;
	}
}

void EditorGlobalVariablesSettings::_global_variable_text_submitted(const String p_name) {
}

void EditorGlobalVariablesSettings::_global_variable_text_changed(const String p_name) {
	String error_string;
	bool is_name_valid = global_variable_name_is_valid(p_name, &error_string);
	add_global_variable->set_disabled(!is_name_valid);
	error_message->set_text(error_string);
	error_message->set_visible(!global_variable_add_name->get_text().is_empty() && !is_name_valid);
}

void EditorGlobalVariablesSettings::_global_variable_file_callback(const String &p_path) {
}

void EditorGlobalVariablesSettings::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			add_global_variable->set_icon(get_editor_theme_icon(SNAME("Add")));
		} break;
	}
}

void EditorGlobalVariablesSettings::_bind_methods() {
	ClassDB::bind_method("update_global_variables", &EditorGlobalVariablesSettings::update_global_variables);
	ClassDB::bind_method("global_variable_add", &EditorGlobalVariablesSettings::global_variable_add);
	ClassDB::bind_method("global_variable_remove", &EditorGlobalVariablesSettings::global_variable_remove);

	ADD_SIGNAL(MethodInfo("global_variables_changed"));
}

bool EditorGlobalVariablesSettings::global_variable_name_is_valid(const String &p_name, String *r_error) {
	if (!p_name.is_valid_identifier()) {
		if (r_error) {
			*r_error = TTR("Invalid name.") + " ";
			if (p_name.size() > 0 && p_name.left(1).is_numeric()) {
				*r_error += TTR("Cannot begin with a digit.");
			} else {
				*r_error += TTR("Valid characters:") + " a-z, A-Z, 0-9 or _";
			}
		}

		return false;
	}

	if (ClassDB::class_exists(p_name)) {
		if (r_error) {
			*r_error = TTR("Invalid name.") + " " + TTR("Must not collide with an existing engine class name.");
		}

		return false;
	}

	if (ScriptServer::is_global_class(p_name)) {
		if (r_error) {
			*r_error = TTR("Invalid name.") + "\n" + TTR("Must not collide with an existing global script class name.");
		}

		return false;
	}

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (Variant::get_type_name(Variant::Type(i)) == p_name) {
			if (r_error) {
				*r_error = TTR("Invalid name.") + " " + TTR("Must not collide with an existing built-in type name.");
			}

			return false;
		}
	}

	for (int i = 0; i < CoreConstants::get_global_constant_count(); i++) {
		if (CoreConstants::get_global_constant_name(i) == p_name) {
			if (r_error) {
				*r_error = TTR("Invalid name.") + " " + TTR("Must not collide with an existing global constant name.");
			}

			return false;
		}
	}

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		List<String> keywords;
		ScriptServer::get_language(i)->get_reserved_words(&keywords);
		for (const String &E : keywords) {
			if (E == p_name) {
				if (r_error) {
					*r_error = TTR("Invalid name.") + " " + TTR("Keyword cannot be used as a Singleton name.");
				}

				return false;
			}
		}
	}

	return true;
}

void EditorGlobalVariablesSettings::update_global_variables() {
	if (updating_global_variables) {
		return;
	}

	updating_global_variables = true;

	// Mark all global_variables as "to remove" for now.
	HashSet<String> to_remove;
	for (const GlobalVariableInfo &info : global_variables_cache) {
		to_remove.insert(info.name);
	}
	HashSet<String> to_add;

	// Rebuild global_variable_cache.
	global_variables_cache.clear();
	List<PropertyInfo> props;
	ProjectSettings::get_singleton()->get_property_list(&props);
	for (const PropertyInfo &pi : props) {
		if (!pi.name.begins_with("global_variables/")) {
			continue;
		}

		String name = pi.name.get_slice("/", 1);
		if (name.is_empty()) {
			continue;
		}

		GlobalVariableInfo info;
		info.name = name;
		global_variables_cache.push_back(info);

		if (to_remove.has(name)) {
			to_remove.erase(name);
		} else {
			to_add.insert(name);
		}
	}
	global_variables_cache.sort();

	// Remove unused global constants.
	for (const String &name : to_remove) {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->remove_named_global_variable(name);
		}
	}

	// Re-add new global constants.
	for (const String &name : to_add) {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->add_named_global_variable(name);
		}
	}

	// Rebuild UI tree.
	tree->clear();
	TreeItem *root = tree->create_item();
	for (const GlobalVariableInfo &info : global_variables_cache) {
		TreeItem *item = tree->create_item(root);
		item->set_text(0, info.name);
		item->set_editable(0, true);
		/*
				item->set_cell_mode(1, TreeItem::CELL_MODE_RANGE);
				item->set_text(1, "1,2,3");
				item->set_editable(1, true);

				item->set_text(2, "Object");
				item->set_editable(2, true);
		*/
		item->add_button(1, get_editor_theme_icon(SNAME("Remove")), BUTTON_DELETE);
		item->set_selectable(1, false);
	}

	updating_global_variables = false;
}

bool EditorGlobalVariablesSettings::global_variable_add(const String &p_name) {
	String name = p_name;

	String error;
	if (!EditorGlobalVariablesSettings::global_variable_name_is_valid(name, &error)) {
		EditorNode::get_singleton()->show_warning(TTR("Can't add global variable:") + "\n" + error);
		return false;
	}

	name = "global_variables/" + name;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

	undo_redo->create_action(TTR("Add global variable"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), name, true);

	if (ProjectSettings::get_singleton()->has_setting(name)) {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, GLOBAL_GET(name));
	} else {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, Variant());
	}

	undo_redo->add_do_method(this, "update_global_variables");
	undo_redo->add_undo_method(this, "update_global_variables");

	undo_redo->add_do_method(this, "emit_signal", global_variables_changed);
	undo_redo->add_undo_method(this, "emit_signal", global_variables_changed);

	undo_redo->commit_action();

	return true;
}
void EditorGlobalVariablesSettings::global_variable_remove(const String &p_name) {
}

EditorGlobalVariablesSettings::EditorGlobalVariablesSettings() {
	ProjectSettings::get_singleton()->add_hidden_prefix("global_variables/");

	// Make first cache
	List<PropertyInfo> props;
	ProjectSettings::get_singleton()->get_property_list(&props);
	for (const PropertyInfo &pi : props) {
		if (!pi.name.begins_with("global_variables/")) {
			continue;
		}

		String name = pi.name.get_slice("/", 1);
		if (name.is_empty()) {
			continue;
		}

		GlobalVariableInfo info;
		info.name = name;
		Variant value = GLOBAL_GET(pi.name);

		// Make sure name references work before parsing scripts
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->add_named_global_variable(info.name);
		}

		global_variables_cache.push_back(info);
	}

	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(hbc);

	error_message = memnew(Label);
	error_message->hide();
	error_message->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
	error_message->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("error_color"), EditorStringName(Editor)));
	add_child(error_message);

	Label *l = memnew(Label);
	l->set_text(TTR("Singleton Name:"));
	hbc->add_child(l);

	global_variable_add_name = memnew(LineEdit);
	global_variable_add_name->set_h_size_flags(SIZE_EXPAND_FILL);
	global_variable_add_name->connect("text_submitted", callable_mp(this, &EditorGlobalVariablesSettings::_global_variable_text_submitted));
	global_variable_add_name->connect("text_changed", callable_mp(this, &EditorGlobalVariablesSettings::_global_variable_text_changed));
	hbc->add_child(global_variable_add_name);

	add_global_variable = memnew(Button);
	add_global_variable->set_text(TTR("Add"));
	add_global_variable->connect("pressed", callable_mp(this, &EditorGlobalVariablesSettings::_global_variable_add));
	// The button will be enabled once a valid name is entered (either automatically or manually).
	add_global_variable->set_disabled(true);
	hbc->add_child(add_global_variable);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_select_mode(Tree::SELECT_MULTI);
	tree->set_allow_reselect(true);

	tree->set_columns(2);
	tree->set_column_titles_visible(true);

	tree->set_column_title(0, TTR("Name"));
	tree->set_column_expand(0, true);
	tree->set_column_expand_ratio(0, 2);
	/*
		tree->set_column_title(1, TTR("Type"));
		tree->set_column_expand(1, true);

		tree->set_column_title(2, TTR("Class"));
		tree->set_column_clip_content(2, true);
		tree->set_column_expand(2, true);
	*/
	tree->set_column_expand(1, false);

	tree->connect("cell_selected", callable_mp(this, &EditorGlobalVariablesSettings::_global_variable_selected));
	tree->connect("item_edited", callable_mp(this, &EditorGlobalVariablesSettings::_global_variable_edited));
	tree->connect("button_clicked", callable_mp(this, &EditorGlobalVariablesSettings::_global_variable_button_pressed));
	tree->set_v_size_flags(SIZE_EXPAND_FILL);

	add_child(tree, true);
}

EditorGlobalVariablesSettings::~EditorGlobalVariablesSettings() {
}