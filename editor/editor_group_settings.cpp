/*************************************************************************/
/*  editor_group_settings.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_group_settings.h"

#include "core/global_constants.h"
#include "core/project_settings.h"
#include "editor_node.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

void EditorGroupSettings::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			update_tree();
			break;
	}
}

bool EditorGroupSettings::_group_exists(const String &p_name) {
	return names.find(p_name) != -1;
}

bool EditorGroupSettings::_group_name_is_valid(const String &p_name, String *r_error) {

	if (!p_name.is_valid_identifier()) {
		if (r_error)
			*r_error = TTR("Invalid name.") + "\n" + TTR("Valid characters:") + " a-z, A-Z, 0-9 or _";

		return false;
	}

	if (ClassDB::class_exists(p_name)) {
		if (r_error)
			*r_error = TTR("Invalid name. Must not collide with an existing engine class name.");

		return false;
	}

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (Variant::get_type_name(Variant::Type(i)) == p_name) {
			if (r_error)
				*r_error = TTR("Invalid name. Must not collide with an existing built-in type name.");

			return false;
		}
	}

	for (int i = 0; i < GlobalConstants::get_global_constant_count(); i++) {
		if (GlobalConstants::get_global_constant_name(i) == p_name) {
			if (r_error)
				*r_error = TTR("Invalid name. Must not collide with an existing global constant name.");

			return false;
		}
	}

	return true;
}

void EditorGroupSettings::_item_edited() {

	if (updating_group)
		return;

	TreeItem *ti = tree->get_edited();

	String name = ti->get_text(0);
	int column = tree->get_edited_column();

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	if (column == 0) {
		String old_name = selected_group;

		if (name == old_name)
			return;

		String error;
		if (!_group_name_is_valid(name, &error)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(error);
			return;
		}

		if (names.find(name) >= 0) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(vformat(TTR("Group '%s' already exists!"), name));
			return;
		}

		updating_group = true;

		undo_redo->create_action(TTR("Rename Group"));

		undo_redo->add_do_method(this, "_rename_group", old_name, name);
		undo_redo->add_undo_method(this, "_rename_group", name, old_name);

		undo_redo->add_do_method(this, "_rename_references", old_name, name);
		undo_redo->add_undo_method(this, "_rename_references", name, old_name);

		undo_redo->add_do_method(this, "call_deferred", "update_tree");
		undo_redo->add_undo_method(this, "call_deferred", "update_tree");

		undo_redo->add_do_method(this, "emit_signal", group_changed);
		undo_redo->add_undo_method(this, "emit_signal", group_changed);

		undo_redo->add_do_method(this, "_save_groups");
		undo_redo->add_undo_method(this, "_save_groups");

		undo_redo->commit_action();

		selected_group = name;
	} else if (column == 1) {
		updating_group = true;

		String description = ti->get_text(1);
		String old_description = group_cache[name].description;

		undo_redo->create_action(TTR("Change Group Description"));

		undo_redo->add_do_method(this, "_set_description", name, description);
		undo_redo->add_undo_method(this, "_set_description", name, old_description);

		undo_redo->add_do_method(this, "call_deferred", "update_tree");
		undo_redo->add_undo_method(this, "call_deferred", "update_tree");

		undo_redo->add_do_method(this, "emit_signal", group_changed);
		undo_redo->add_undo_method(this, "emit_signal", group_changed);

		undo_redo->add_do_method(this, "_save_groups");
		undo_redo->add_undo_method(this, "_save_groups");

		undo_redo->commit_action();
	}

	updating_group = false;
}

void EditorGroupSettings::_item_selected() {

	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;

	selected_group = ti->get_text(0);
}

void EditorGroupSettings::_item_activated() {

	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;
}

void EditorGroupSettings::_item_button_pressed(Object *p_item, int p_column, int p_button) {

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);

	selected_group = ti->get_text(0);

	remove_confirmation->popup_centered_minsize();
}

void EditorGroupSettings::_create_button_pressed() {

	String name = create_group_name->get_text();

	String error;
	if (!_group_name_is_valid(name, &error)) {
		EditorNode::get_singleton()->show_warning(error);
		return;
	}

	if (names.find(name) >= 0) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Group '%s' already exists!"), name));
		return;
	}

	String description = create_group_description->get_text();

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

	undo_redo->create_action(TTR("Add Group"));

	undo_redo->add_do_method(this, "_create_group", name, description);
	undo_redo->add_undo_method(this, "_delete_group", name);

	undo_redo->add_do_method(this, "call_deferred", "update_tree");
	undo_redo->add_undo_method(this, "call_deferred", "update_tree");

	undo_redo->add_do_method(this, "emit_signal", group_changed);
	undo_redo->add_undo_method(this, "emit_signal", group_changed);

	undo_redo->add_do_method(this, "_save_groups");
	undo_redo->add_undo_method(this, "_save_groups");

	undo_redo->commit_action();

	create_group_name->set_text("");
	create_group_description->set_text("");
}

void EditorGroupSettings::_remove_references(const String &p_name) {

	Set<String> scenes;
	_get_all_scenes(EditorFileSystem::get_singleton()->get_filesystem(), scenes);
	for (Set<String>::Element *E = scenes.front(); E; E = E->next()) {

		Ref<PackedScene> data = ResourceLoader::load(E->get());
		if (data->get_state()->remove_group_references(StringName(p_name))) {
			ResourceSaver::save(E->get(), data);
		}
	}

	// Update opened scenes
	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (es != NULL) {
		_remove_node_references(es, p_name);
		EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor()->update_tree();
	}
}

void EditorGroupSettings::_rename_references(const String &p_old_name, const String &p_new_name) {

	Set<String> scenes;
	_get_all_scenes(EditorFileSystem::get_singleton()->get_filesystem(), scenes);
	for (Set<String>::Element *E = scenes.front(); E; E = E->next()) {

		Ref<PackedScene> data = ResourceLoader::load(E->get());
		if (data->get_state()->rename_group_references(StringName(p_old_name), StringName(p_new_name))) {
			ResourceSaver::save(E->get(), data);
		}
	}

	// Update opened scenes
	Node *es = EditorNode::get_singleton()->get_edited_scene();
	if (es != NULL) {
		_rename_node_references(es, p_old_name, p_new_name);
		EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor()->update_tree();
	}
}

void EditorGroupSettings::_remove_node_references(Node *p_node, const String &p_name) {

	if (p_node->is_in_group(p_name)) {
		p_node->remove_from_group(p_name);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_remove_node_references(p_node->get_child(i), p_name);
	}
}

void EditorGroupSettings::_rename_node_references(Node *p_node, const String &p_old_name, const String &p_new_name) {

	if (p_node->is_in_group(p_old_name)) {
		p_node->remove_from_group(p_old_name);
		p_node->add_to_group(p_new_name, true);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_rename_node_references(p_node->get_child(i), p_old_name, p_new_name);
	}
}

void EditorGroupSettings::_get_all_scenes(EditorFileSystemDirectory *p_dir, Set<String> &r_list) {

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		if (p_dir->get_file_type(i) == "PackedScene")
			r_list.insert(p_dir->get_file_path(i));
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_get_all_scenes(p_dir->get_subdir(i), r_list);
	}
}

void EditorGroupSettings::_confirm_delete() {

	String name = selected_group;
	String description = group_cache[name].description;

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	undo_redo->create_action(TTR("Remove Group"));

	undo_redo->add_do_method(this, "_delete_group", name);
	undo_redo->add_undo_method(this, "_create_group", name, description);

	undo_redo->add_do_method(this, "_remove_references", name);

	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");

	undo_redo->add_do_method(this, "emit_signal", group_changed);
	undo_redo->add_undo_method(this, "emit_signal", group_changed);

	undo_redo->add_do_method(this, "_save_groups");
	undo_redo->add_undo_method(this, "_save_groups");

	undo_redo->commit_action();
}

void EditorGroupSettings::_create_group(const String &p_name, const String &p_description) {

	GroupInfo gi;

	gi.name = p_name;
	gi.description = p_description;

	group_cache[p_name] = gi;

	names.push_back(p_name);
	names.sort_custom<StringComparator>();
}

void EditorGroupSettings::_delete_group(const String &p_name) {

	names.erase(p_name);
	group_cache.erase(p_name);
}

void EditorGroupSettings::_rename_group(const String &p_old_name, const String &p_new_name) {

	GroupInfo gi = group_cache[p_old_name];
	gi.name = p_new_name;

	group_cache.erase(p_old_name);
	group_cache[p_new_name] = gi;

	names.erase(p_old_name);
	names.push_back(p_new_name);
	names.sort_custom<StringComparator>();
}

void EditorGroupSettings::_set_description(const String &p_name, const String &p_description) {

	GroupInfo gi = group_cache[p_name];
	gi.description = p_description;
}

void EditorGroupSettings::_init_groups() {
	group_cache.clear();
	if (ProjectSettings::get_singleton()->has_setting("_global_groups")) {
		Array groups = ProjectSettings::get_singleton()->get("_global_groups");

		for (int i = 0; i < groups.size(); i++) {
			Dictionary g = groups[i];

			_create_group(g["name"], g["description"]);
		}
	}
}

void EditorGroupSettings::_save_groups() {
	Array g_array;
	for (int i = 0; i < names.size(); i++) {
		GroupInfo gi = group_cache[names[i]];
		Dictionary d;
		d["name"] = gi.name;
		d["description"] = gi.description;
		g_array.push_back(d);
	}

	ProjectSettings::get_singleton()->set("_global_groups", g_array);
	ProjectSettings::get_singleton()->save();
}

void EditorGroupSettings::update_tree() {

	if (updating_group)
		return;

	updating_group = true;

	tree->clear();
	TreeItem *root = tree->create_item();

	for (int i = 0; i < names.size(); i++) {

		String name = names[i];

		GroupInfo gi = group_cache[names[i]];

		TreeItem *item = tree->create_item(root);
		item->set_text(0, gi.name);
		item->set_editable(0, true);

		item->set_text(1, gi.description);
		item->set_editable(1, true);
		item->add_button(2, get_icon("Remove", "EditorIcons"), BUTTON_DELETE);
		item->set_selectable(2, false);
	}

	updating_group = false;
}

void EditorGroupSettings::create_group(const String &p_name, const String &p_description) {

	if (_group_exists(p_name))
		return;

	_create_group(p_name, p_description);
	_save_groups();
	update_tree();
}

void EditorGroupSettings::delete_group(const String &p_name) {

	if (!_group_exists(p_name))
		return;

	_delete_group(p_name);
	_remove_references(p_name);
	_save_groups();
	update_tree();
}

void EditorGroupSettings::get_groups(List<String> *current_groups) {

	for (int i = 0; i < names.size(); i++) {
		current_groups->push_back(names[i]);
	}
}

void EditorGroupSettings::_bind_methods() {

	ClassDB::bind_method("_item_edited", &EditorGroupSettings::_item_edited);
	ClassDB::bind_method("_item_selected", &EditorGroupSettings::_item_selected);
	ClassDB::bind_method("_item_activated", &EditorGroupSettings::_item_activated);
	ClassDB::bind_method("_item_button_pressed", &EditorGroupSettings::_item_button_pressed);

	ClassDB::bind_method("_create_button_pressed", &EditorGroupSettings::_create_button_pressed);

	ClassDB::bind_method("_create_group", &EditorGroupSettings::_create_group);
	ClassDB::bind_method("_delete_group", &EditorGroupSettings::_delete_group);
	ClassDB::bind_method("_rename_group", &EditorGroupSettings::_rename_group);
	ClassDB::bind_method("_set_description", &EditorGroupSettings::_set_description);

	ClassDB::bind_method("_remove_references", &EditorGroupSettings::_remove_references);
	ClassDB::bind_method("_rename_references", &EditorGroupSettings::_rename_references);

	ClassDB::bind_method("_confirm_delete", &EditorGroupSettings::_confirm_delete);

	ClassDB::bind_method("_init_groups", &EditorGroupSettings::_init_groups);
	ClassDB::bind_method("_save_groups", &EditorGroupSettings::_save_groups);

	ClassDB::bind_method("create_group", &EditorGroupSettings::create_group);
	ClassDB::bind_method("delete_group", &EditorGroupSettings::delete_group);
	ClassDB::bind_method("update_tree", &EditorGroupSettings::update_tree);

	ADD_SIGNAL(MethodInfo("group_changed"));
}

EditorGroupSettings::EditorGroupSettings() {

	_init_groups();

	group_changed = "group_changed";

	updating_group = false;
	selected_group = "";

	remove_confirmation = memnew(ConfirmationDialog);
	remove_confirmation->set_title(TTR("Delete confirmation"));
	remove_confirmation->set_text(TTR("Deleting a group will also remove its references. This action is undoable."));
	remove_confirmation->get_ok()->set_text(TTR("Delete"));
	remove_confirmation->connect("confirmed", this, "_confirm_delete");
	add_child(remove_confirmation);

	HBoxContainer *hbc = memnew(HBoxContainer);
	add_child(hbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Group Name:"));
	hbc->add_child(l);

	create_group_name = memnew(LineEdit);
	create_group_name->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(create_group_name);

	l = memnew(Label);
	l->set_text(TTR("Group Description:"));
	hbc->add_child(l);

	create_group_description = memnew(LineEdit);
	create_group_description->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(create_group_description);

	Button *create_group = memnew(Button);
	create_group->set_text(TTR("Create"));
	create_group->connect("pressed", this, "_create_button_pressed");
	hbc->add_child(create_group);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_select_mode(Tree::SELECT_MULTI);
	tree->set_allow_reselect(true);

	tree->set_drag_forwarding(this);

	tree->set_columns(3);
	tree->set_column_titles_visible(true);

	tree->set_column_title(0, TTR("Name"));
	tree->set_column_expand(0, true);
	tree->set_column_min_width(0, 100);

	tree->set_column_title(1, TTR("Description"));
	tree->set_column_expand(1, true);
	tree->set_column_min_width(1, 100);

	tree->set_column_expand(2, false);
	tree->set_column_min_width(2, 1 * 40 * EDSCALE);

	tree->connect("item_edited", this, "_item_edited");
	tree->connect("cell_selected", this, "_item_selected");
	tree->connect("item_activated", this, "_item_activated");
	tree->connect("button_pressed", this, "_item_button_pressed");
	tree->set_v_size_flags(SIZE_EXPAND_FILL);

	add_child(tree, true);
}

EditorGroupSettings::~EditorGroupSettings() {
}
