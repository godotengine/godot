/*************************************************************************/
/*  groups_editor.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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


#include "print_string.h"

void GroupsEditor::_notification(int p_what) {
	
	if (p_what==NOTIFICATION_ENTER_TREE) {
		connect("confirmed", this,"_close");
	}	
}

void GroupsEditor::_close() {
	
	hide();
	
}
void GroupsEditor::_add() {
	
	if (!node)
		return;
		
	undo_redo->create_action("Add To Group");
	undo_redo->add_do_method(node,"add_to_group",group_name->get_text(),true);
	undo_redo->add_undo_method(node,"remove_from_group",group_name->get_text());

	undo_redo->add_do_method(this,"update_tree");
	undo_redo->add_undo_method(this,"update_tree");

	undo_redo->commit_action();
}


void GroupsEditor::_remove() {
	
	if (!tree->get_selected())
		return;
	if (!node)
		return;

	TreeItem *sel = tree->get_selected();
	if (!sel)
		return;
		
	node->remove_from_group( sel->get_text(0) );
	update_tree();
}

void GroupsEditor::update_tree() {

	
	tree->clear();
	
	if (!node)
		return;
		
	List<GroupInfo> groups;
	node->get_groups(&groups);
	
	TreeItem *root=tree->create_item();
	
	for(List<GroupInfo>::Element *E=groups.front();E;E=E->next()) {
	
		if (!E->get().persistent)
			continue;
		TreeItem *item=tree->create_item(root);
		item->set_text(0, E->get().name);	
	
	}

}

void GroupsEditor::set_current(Node* p_node) {
	
	node=p_node;
	update_tree();

}

void GroupsEditor::_bind_methods() {
	
	ObjectTypeDB::bind_method("_add",&GroupsEditor::_add);
	ObjectTypeDB::bind_method("_close",&GroupsEditor::_close);
	ObjectTypeDB::bind_method("_remove",&GroupsEditor::_remove);	
	ObjectTypeDB::bind_method("update_tree",&GroupsEditor::update_tree);
}

GroupsEditor::GroupsEditor() {

	set_title("Group Editor");
	
	Label * label = memnew( Label );
	label->set_pos( Point2( 8,11) );
	label->set_text("Groups:");
	
	add_child(label);	
	
	group_name = memnew(LineEdit);
	group_name->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	group_name->set_begin( Point2( 15,28) );
	group_name->set_end( Point2( 94,48 ) );
	
	add_child(group_name);
	
	tree = memnew( Tree );
	tree->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	tree->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	tree->set_begin( Point2( 15,52) );
	tree->set_end( Point2( 94,42 ) );
	tree->set_hide_root(true);		
	add_child(tree);
	
	add = memnew( Button );
	add->set_anchor( MARGIN_LEFT, ANCHOR_END );
	add->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	add->set_begin( Point2( 90, 28 ) );
	add->set_end( Point2( 15, 48 ) );	
	add->set_text("Add");
	
	add_child(add);
	
	remove = memnew( Button );
	remove->set_anchor( MARGIN_LEFT, ANCHOR_END );
	remove->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	remove->set_begin( Point2( 90, 52 ) );
	remove->set_end( Point2( 15, 72 ) );	
	remove->set_text("Remove");
	
	add_child(remove);

	get_ok()->set_text("Close");
			
	add->connect("pressed", this,"_add");
	remove->connect("pressed", this,"_remove");	

	
	node=NULL;
}


GroupsEditor::~GroupsEditor()
{
}



