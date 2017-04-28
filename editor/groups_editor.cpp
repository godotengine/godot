/*************************************************************************/
/*  groups_editor.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_node.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/resources/packed_scene.h"

void GroupsEditor::_add_group(const String &p_group) {

	if (!node)
		return;

	String name = group_name->get_text();
	if (name.strip_edges() == "")
		return;

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

	group_name->clear();
}

void GroupsEditor::_remove_group(Object *p_item, int p_column, int p_id) {

	if (!node)
		return;

	TreeItem *ti = p_item->cast_to<TreeItem>();
	if (!ti)
		return;

	String name = ti->get_text(0);

	undo_redo->create_action(TTR("Remove from Group"));

	undo_redo->add_do_method(node, "remove_from_group", name);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(node, "add_to_group", name, true);
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree

	undo_redo->commit_action();
}

struct _GroupInfoComparator {

	bool operator()(const Node::GroupInfo &p_a, const Node::GroupInfo &p_b) const {
		return p_a.name.operator String() < p_b.name.operator String();
	}
};

void GroupsEditor::update_tree() {

	tree->clear();

	if (!node)
		return;

	List<Node::GroupInfo> groups;
	node->get_groups(&groups);
	groups.sort_custom<_GroupInfoComparator>();

	TreeItem *root = tree->create_item();

	for (List<GroupInfo>::Element *E = groups.front(); E; E = E->next()) {

		Node::GroupInfo gi = E->get();
		if (!gi.persistent)
			continue;

		Node *n = node;
		bool can_be_deleted = true;

		while (n) {

			Ref<SceneState> ss = (n == EditorNode::get_singleton()->get_edited_scene()) ? n->get_scene_inherited_state() : n->get_scene_instance_state();

			if (ss.is_valid()) {

				int path = ss->find_node_by_path(n->get_path_to(node));
				if (path != -1) {
					if (ss->is_node_in_group(path, gi.name)) {
						can_be_deleted = false;
					}
				}
			}

			n = n->get_owner();
		}

		TreeItem *item = tree->create_item(root);
		item->set_text(0, gi.name);
		if (can_be_deleted) {
			item->add_button(0, get_icon("Remove", "EditorIcons"), 0);
		} else {
			item->set_selectable(0, false);
		}
	}
}

void GroupsEditor::set_current(Node *p_node) {

	node = p_node;
	update_tree();
}

void GroupsEditor::_bind_methods() {

	ClassDB::bind_method("_add_group", &GroupsEditor::_add_group);
	ClassDB::bind_method("_remove_group", &GroupsEditor::_remove_group);
	ClassDB::bind_method("update_tree", &GroupsEditor::update_tree);
}

GroupsEditor::GroupsEditor() {

	node = NULL;

	VBoxContainer *vbc = this;

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	group_name = memnew(LineEdit);
	group_name->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(group_name);
	group_name->connect("text_entered", this, "_add_group");

	add = memnew(Button);
	add->set_text(TTR("Add"));
	hbc->add_child(add);
	add->connect("pressed", this, "_add_group", varray(String()));

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(tree);
	tree->connect("button_pressed", this, "_remove_group");
	add_constant_override("separation", 3 * EDSCALE);
}

GroupsEditor::~GroupsEditor() {
}
