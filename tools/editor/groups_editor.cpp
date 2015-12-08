/*************************************************************************/
/*  groups_editor.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "scene/gui/box_container.h"
#include "scene/gui/label.h"

void GroupsEditor::_add_group(const String& p_group) {

	if (!node)
		return;

	String name = group_name->get_text();
	if (name.strip_edges()=="")
		return;

	if (node->is_in_group(name))
		return;

	undo_redo->create_action("Add to Group");

	undo_redo->add_do_method(node,"add_to_group",name,true);
	undo_redo->add_do_method(this,"update_tree");
	undo_redo->add_undo_method(node,"remove_from_group",name,get_text());
	undo_redo->add_undo_method(this,"update_tree");

	undo_redo->commit_action();
}

void GroupsEditor::_remove_group(Object *p_item, int p_column, int p_id) {

	if (!node)
		return;

	TreeItem *ti = p_item->cast_to<TreeItem>();
	if (!ti)
		return;

	String name = ti->get_text(0);

	undo_redo->create_action("Remove from Group");

	undo_redo->add_do_method(node,"remove_from_group",name);
	undo_redo->add_do_method(this,"update_tree");
	undo_redo->add_undo_method(node,"add_to_group",name,true);
	undo_redo->add_undo_method(this,"update_tree");

	undo_redo->commit_action();
}

struct _GroupInfoComparator {

	bool operator()(const Node::GroupInfo& p_a, const Node::GroupInfo& p_b) const {
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

	TreeItem *root=tree->create_item();

	for(List<GroupInfo>::Element *E=groups.front();E;E=E->next()) {

		Node::GroupInfo gi = E->get();
		if (!gi.persistent)
			continue;

		TreeItem *item=tree->create_item(root);
		item->set_text(0, gi.name);
		item->add_button(0, get_icon("Remove", "EditorIcons"), 0);
	}
}

void GroupsEditor::set_current(Node* p_node) {

	node=p_node;
	update_tree();
}

void GroupsEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_add_group",&GroupsEditor::_add_group);
	ObjectTypeDB::bind_method("_remove_group",&GroupsEditor::_remove_group);
	ObjectTypeDB::bind_method("update_tree",&GroupsEditor::update_tree);
}

GroupsEditor::GroupsEditor() {

	node=NULL;

	set_title("Group Editor");

	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_margin_child("Group", hbc);

	group_name = memnew( LineEdit );
	group_name->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(group_name);
	group_name->connect("text_entered",this,"_add_group");

	add = memnew( Button );
	add->set_text("Add");
	hbc->add_child(add);
	add->connect("pressed", this,"_add_group", varray(String()));

	tree = memnew( Tree );
	tree->set_hide_root(true);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_margin_child("Node Group(s)", tree, true);
	tree->connect("button_pressed",this,"_remove_group");

	get_ok()->set_text("Close");
}

GroupsEditor::~GroupsEditor()
{
}



