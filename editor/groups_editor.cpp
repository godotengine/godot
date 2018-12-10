/*************************************************************************/
/*  groups_editor.cpp                                                    */
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

#include "groups_editor.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/resources/packed_scene.h"

void GroupsEditor::_add_to_group(const String &p_group) {

	if (!node)
		return;

	String name = p_group;
	if (node->is_in_group(name))
		return;

	undo_redo->create_action(TTR("Add to Group"));

	undo_redo->add_do_method(node, "add_to_group", name, true);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(node, "remove_from_group", name);
	undo_redo->add_undo_method(this, "update_tree");

	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree

	undo_redo->commit_action();
}

void GroupsEditor::_remove_from_group(const String &p_group) {

	if (!node)
		return;
	String name = p_group;

	undo_redo->create_action(TTR("Remove from Group"));

	undo_redo->add_do_method(node, "remove_from_group", name);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(node, "add_to_group", name, true);
	undo_redo->add_undo_method(this, "update_tree");

	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree

	undo_redo->commit_action();
}

void GroupsEditor::_group_toggled() {
	if (updating_group)
		return;

	TreeItem *ti = tree->get_edited();
	int column = tree->get_edited_column();

	if (column == 0) {
		updating_group = true;

		bool checked = ti->is_checked(0);
		String group_name = ti->get_text(0);

		if (checked) {
			_add_to_group(group_name);
		} else {
			_remove_from_group(group_name);
		}
	}

	updating_group = false;
}

void GroupsEditor::_manage_groups() {
	EditorNode::get_singleton()->get_project_settings()->set_groups_page();
	EditorNode::get_singleton()->get_project_settings()->popup_project_settings();
}

void GroupsEditor::update_tree() {

	if (!node)
		return;

	if (updating_group)
		return;

	updating_group = true;

	tree->clear();

	List<Node::GroupInfo> current_groups;
	node->get_groups(&current_groups);

	List<String> current_groups_names;
	for (List<GroupInfo>::Element *E = current_groups.front(); E; E = E->next()) {

		Node::GroupInfo gi = E->get();
		current_groups_names.push_back(gi.name);
	}

	TreeItem *root = tree->create_item();

	List<String> group_names;
	ProjectSettingsEditor::get_singleton()->get_group_settings()->get_groups(&group_names);

	for (List<String>::Element *E = group_names.front(); E; E = E->next()) {

		String name = E->get();

		TreeItem *item = tree->create_item(root);
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_editable(0, true);
		item->set_checked(0, current_groups_names.find(name) != NULL);
		item->set_text(0, name);
	}

	updating_group = false;
}

void GroupsEditor::set_current(Node *p_node) {

	node = p_node;
	update_tree();
}

void GroupsEditor::_bind_methods() {

	ClassDB::bind_method("_add_to_group", &GroupsEditor::_add_to_group);
	ClassDB::bind_method("_remove_from_group", &GroupsEditor::_remove_from_group);

	ClassDB::bind_method("update_tree", &GroupsEditor::update_tree);
}

GroupsEditor::GroupsEditor() {

	node = NULL;
	updating_group = false;

	Button *group_dialog_button = memnew(Button);
	group_dialog_button->set_text(TTR("Manage Groups"));
	add_child(group_dialog_button);
	group_dialog_button->connect("pressed", this, "_manage_groups");

	add_constant_override("separation", 3 * EDSCALE);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_column_titles_visible(false);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);

	tree->connect("item_edited", this, "_group_toggled");
	add_child(tree);

	ProjectSettingsEditor::get_singleton()->get_group_settings()->connect("group_changed", this, "update_tree");
}

GroupsEditor::~GroupsEditor() {
}
