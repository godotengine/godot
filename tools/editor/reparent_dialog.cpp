/*************************************************************************/
/*  reparent_dialog.cpp                                                  */
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
#include "reparent_dialog.h"

#include "scene/gui/label.h"
#include "scene/gui/box_container.h"


#include "print_string.h"

void ReparentDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE)	{

		connect("confirmed", this,"_reparent");
	}

	if (p_what==NOTIFICATION_EXIT_TREE)	{

		disconnect("confirmed", this,"_reparent");
	}

	if (p_what==NOTIFICATION_DRAW) {
		
		//RID ci = get_canvas_item();
		//get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	}	
}

void ReparentDialog::_cancel() {
	
	hide();
	
}
void ReparentDialog::_reparent() {
	
	if (tree->get_selected()) {

		emit_signal("reparent",tree->get_selected()->get_path(),node_only->is_pressed());
		hide();
	}
}

void ReparentDialog::set_current(const Set<Node*>& p_selection) {
	
	tree->set_marked(p_selection,false,false);
	//tree->set_selected(p_node->get_parent());
}

void ReparentDialog::_bind_methods() {
	
	ObjectTypeDB::bind_method("_reparent",&ReparentDialog::_reparent); 
	ObjectTypeDB::bind_method("_cancel",&ReparentDialog::_cancel);

	ADD_SIGNAL( MethodInfo("reparent",PropertyInfo(Variant::NODE_PATH,"path"),PropertyInfo(Variant::BOOL,"only_node")));
}


ReparentDialog::ReparentDialog() {
	

	set_title("Reparent Node");
	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);

	tree = memnew( SceneTreeEditor(false) );
	

	vbc->add_margin_child("Reparent Location (Select new Parent):",tree,true);
	
	//Label *label = memnew( Label );
	//label->set_pos( Point2( 15,8) );
	//label->set_text("Reparent Location (Select new Parent):");
	
	node_only = memnew( CheckButton );
	add_child(node_only);
	node_only->hide();

	tree->set_show_enabled_subscene(true);
	//vbc->add_margin_child("Options:",node_only);;
	

	//cancel->connect("pressed", this,"_cancel");

	get_ok()->set_text("Reparent");
}


ReparentDialog::~ReparentDialog()
{
}

