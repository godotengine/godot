/*************************************************************************/
/*  scene_tree_editor.cpp                                                */
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
#include "scene_tree_editor.h"
#include "scene/gui/label.h"
#include "editor_node.h"
#include "print_string.h"
#include "message_queue.h"
#include "scene/main/viewport.h"
#include "tools/editor/plugins/canvas_item_editor_plugin.h"

Node *SceneTreeEditor::get_scene_node() {

	ERR_FAIL_COND_V(!is_inside_tree(),NULL);
	if (get_tree()->get_root()->get_child_count() && get_tree()->get_root()->get_child(0)->cast_to<EditorNode>())
		return get_tree()->get_root()->get_child(0)->cast_to<EditorNode>()->get_edited_scene();
	else
		return get_tree()->get_root();

	return NULL;
}


void SceneTreeEditor::_subscene_option(int p_idx) {

	Object *obj = ObjectDB::get_instance(instance_node);
	if (!obj)
		return;
	Node *node = obj->cast_to<Node>();
	if (!node)
		return;

	switch(p_idx) {

		case SCENE_MENU_SHOW_CHILDREN: {

			if (node->has_meta("__editor_show_subtree")) {
				instance_menu->set_item_checked(0,true);
				node->set_meta("__editor_show_subtree",Variant());
				_update_tree();
			} else {
				node->set_meta("__editor_show_subtree",true);
				_update_tree();
			}

		} break;
		case SCENE_MENU_OPEN: {

			emit_signal("open",node->get_filename());
		} break;

	}

}


void SceneTreeEditor::_cell_button_pressed(Object *p_item,int p_column,int p_id) {

	TreeItem *item=p_item->cast_to<TreeItem>();
	ERR_FAIL_COND(!item);

	NodePath np = item->get_metadata(0);

	Node *n=get_node(np);
	ERR_FAIL_COND(!n);

	if (p_id==BUTTON_SUBSCENE) {
		//open scene request
		Rect2 item_rect = tree->get_item_rect(item,0);
		item_rect.pos.y-=tree->get_scroll().y;
		item_rect.pos+=tree->get_global_pos();
		instance_menu->set_pos(item_rect.pos+Vector2(0,item_rect.size.y));
		instance_menu->set_size(Vector2(item_rect.size.x,0));
		if (n->has_meta("__editor_show_subtree"))
			instance_menu->set_item_checked(0,true);
		else
			instance_menu->set_item_checked(0,false);

		instance_menu->popup();
		instance_node=n->get_instance_ID();
		//emit_signal("open",n->get_filename());
	} else if (p_id==BUTTON_SCRIPT) {
		RefPtr script=n->get_script();
		if (!script.is_null())
			emit_signal("open_script",script);

	} else if (p_id==BUTTON_VISIBILITY) {


		if (n->is_type("Spatial")) {

			Spatial *ci = n->cast_to<Spatial>();
			if (!ci->is_visible() && ci->get_parent_spatial() && !ci->get_parent_spatial()->is_visible()) {
				error->set_text("This item cannot be made visible because the parent is hidden. Unhide the parent first.");
				error->popup_centered_minsize(Size2(400,80));
				return;
			}

			bool v = !bool(n->call("is_hidden"));
			undo_redo->create_action("Toggle Spatial Visible");
			undo_redo->add_do_method(n,"_set_visible_",!v);
			undo_redo->add_undo_method(n,"_set_visible_",v);
			undo_redo->commit_action();
		} else if (n->is_type("CanvasItem")) {

			CanvasItem *ci = n->cast_to<CanvasItem>();
			if (!ci->is_visible() && ci->get_parent_item() && !ci->get_parent_item()->is_visible()) {
				error->set_text("This item cannot be made visible because the parent is hidden. Unhide the parent first.");
				error->popup_centered_minsize(Size2(400,80));
				return;
			}
			bool v = !bool(n->call("is_hidden"));
			undo_redo->create_action("Toggle CanvasItem Visible");
			undo_redo->add_do_method(n,v?"hide":"show");
			undo_redo->add_undo_method(n,v?"show":"hide");
			undo_redo->commit_action();
		}

	} else if (p_id==BUTTON_LOCK) {

		if (n->is_type("CanvasItem")) {
			n->set_meta("_edit_lock_", Variant());
			_update_tree();
			emit_signal("node_changed");
		}

	} else if (p_id==BUTTON_GROUP) {
		if (n->is_type("CanvasItem")) {
			n->set_meta("_edit_group_", Variant());
			_update_tree();
			emit_signal("node_changed");
		}
	}
}

void SceneTreeEditor::_add_nodes(Node *p_node,TreeItem *p_parent) {
	
	if (!p_node)
		return;

	// only owned nodes are editable, since nodes can create their own (manually owned) child nodes,
	// which the editor needs not to know about.

	bool part_of_subscene=false;

	if (!display_foreign && p_node->get_owner()!=get_scene_node() && p_node!=get_scene_node()) {

		if ((show_enabled_subscene || can_open_instance) && p_node->get_owner() && p_node->get_owner()->get_owner()==get_scene_node() && p_node->get_owner()->has_meta("__editor_show_subtree")) {

			part_of_subscene=true;
			//allow
		} else {
			return;
		}
	}

	TreeItem *item = tree->create_item(p_parent);
	item->set_text(0, p_node->get_name() );
	if (can_rename && (p_node->get_owner() == get_scene_node() || p_node==get_scene_node()))
		item->set_editable(0, true);

	item->set_selectable(0,true);
	if (can_rename) {

		bool collapsed = p_node->has_meta("_editor_collapsed") ? (bool)p_node->get_meta("_editor_collapsed") : false;
		if (collapsed)
			item->set_collapsed(true);
	}

	Ref<Texture> icon;
	if (p_node->has_meta("_editor_icon"))
		icon=p_node->get_meta("_editor_icon");
	else
		icon=get_icon( (has_icon(p_node->get_type(),"EditorIcons")?p_node->get_type():String("Object")),"EditorIcons");
	item->set_icon(0, icon );
	item->set_metadata( 0,p_node->get_path() );	
	if (part_of_subscene) {

		//item->set_selectable(0,marked_selectable);
		item->set_custom_color(0,Color(0.8,0.4,0.20));

	} else if (marked.has(p_node)) {
				
		item->set_selectable(0,marked_selectable);
		item->set_custom_color(0,Color(0.8,0.1,0.10));
	} else if (!marked_selectable && !marked_children_selectable) {

		Node *node=p_node;
		while(node) {
			if (marked.has(node)) {
				item->set_selectable(0,false);
				item->set_custom_color(0,Color(0.8,0.1,0.10));
				break;
			}
			node=node->get_parent();
		}
	}

	if (p_node!=get_scene_node() && p_node->get_filename()!="" && can_open_instance) {

		item->add_button(0,get_icon("InstanceOptions","EditorIcons"),BUTTON_SUBSCENE);
		item->set_tooltip(0,"Instance: "+p_node->get_filename()+"\nType: "+p_node->get_type());
	} else {
		item->set_tooltip(0,String(p_node->get_name())+"\nType: "+p_node->get_type());
	}

	if (can_open_instance) {

		if (!p_node->is_connected("script_changed",this,"_node_script_changed"))
			p_node->connect("script_changed",this,"_node_script_changed",varray(p_node));


		if (!p_node->get_script().is_null()) {

			item->add_button(0,get_icon("Script","EditorIcons"),BUTTON_SCRIPT);
		}

		if (p_node->is_type("CanvasItem")) {

			bool is_locked = p_node->has_meta("_edit_lock_");//_edit_group_
			if (is_locked)
				item->add_button(0,get_icon("Lock", "EditorIcons"), BUTTON_LOCK);

			bool is_grouped = p_node->has_meta("_edit_group_");
			if (is_grouped)
				item->add_button(0,get_icon("Group", "EditorIcons"), BUTTON_GROUP);

			bool h = p_node->call("is_hidden");
			if (h)
				item->add_button(0,get_icon("Hidden","EditorIcons"),BUTTON_VISIBILITY);
			else
				item->add_button(0,get_icon("Visible","EditorIcons"),BUTTON_VISIBILITY);

			if (!p_node->is_connected("visibility_changed",this,"_node_visibility_changed"))
				p_node->connect("visibility_changed",this,"_node_visibility_changed",varray(p_node));

		} else if (p_node->is_type("Spatial")) {

			bool h = p_node->call("is_hidden");
			if (h)
				item->add_button(0,get_icon("Hidden","EditorIcons"),BUTTON_VISIBILITY);
			else
				item->add_button(0,get_icon("Visible","EditorIcons"),BUTTON_VISIBILITY);

			if (!p_node->is_connected("visibility_changed",this,"_node_visibility_changed"))
				p_node->connect("visibility_changed",this,"_node_visibility_changed",varray(p_node));

		}

	}

	if (editor_selection) {
		if (editor_selection->is_selected(p_node)) {

			item->select(0);
		}
	}

	if (selected==p_node) {
		if (!editor_selection)
			item->select(0);
		item->set_as_cursor(0);
	}
		
	for (int i=0;i<p_node->get_child_count();i++) {
		
		_add_nodes(p_node->get_child(i),item);
	}
}


void SceneTreeEditor::_node_visibility_changed(Node *p_node) {


	if (p_node!=get_scene_node() && !p_node->get_owner()) {

		return;
	}
	TreeItem* item=p_node?_find(tree->get_root(),p_node->get_path()):NULL;
	if (!item) {

		return;
	}
	int idx=item->get_button_by_id(0,BUTTON_VISIBILITY);
	ERR_FAIL_COND(idx==-1);

	bool visible=false;

	if (p_node->is_type("CanvasItem")) {
		visible = !p_node->call("is_hidden");
	} else if (p_node->is_type("Spatial")) {
		visible = !p_node->call("is_hidden");
	}

	if (!visible)
		item->set_button(0,idx,get_icon("Hidden","EditorIcons"));
	else
		item->set_button(0,idx,get_icon("Visible","EditorIcons"));


}


void SceneTreeEditor::_node_script_changed(Node *p_node) {

	_update_tree();
	/*
	changes the order :|
	TreeItem* item=p_node?_find(tree->get_root(),p_node->get_path()):NULL;
	if (p_node->get_script().is_null()) {

		int idx=item->get_button_by_id(0,2);
		if (idx>=0)
			item->erase_button(0,idx);
	} else {

		int idx=item->get_button_by_id(0,2);
		if (idx<0)
			item->add_button(0,get_icon("Script","EditorIcons"),2);

	}*/

}

void SceneTreeEditor::_node_removed(Node *p_node) {
	
	if (p_node->is_connected("script_changed",this,"_node_script_changed"))
		p_node->disconnect("script_changed",this,"_node_script_changed");

	if (p_node->is_type("Spatial") || p_node->is_type("CanvasItem")) {
		if (p_node->is_connected("visibility_changed",this,"_node_visibility_changed"))
			p_node->disconnect("visibility_changed",this,"_node_visibility_changed");
	}

	if (p_node==selected) {
		selected=NULL;
		emit_signal("node_selected");
	}
		
	
}
void SceneTreeEditor::_update_tree() {


	if (!is_inside_tree()) {
		tree_dirty=false;
		return;
	}

	updating_tree=true;
	tree->clear();
	if (get_scene_node()) {
		_add_nodes( get_scene_node(), NULL );
		last_hash = hash_djb2_one_64(0);
		_compute_hash(get_scene_node(),last_hash);

	}
	updating_tree=false;

	tree_dirty=false;

}

void SceneTreeEditor::_compute_hash(Node *p_node,uint64_t &hash) {

	hash=hash_djb2_one_64(p_node->get_instance_ID(),hash);
	if (p_node->get_parent())
		hash=hash_djb2_one_64(p_node->get_parent()->get_instance_ID(),hash); //so a reparent still produces a different hash


	for(int i=0;i<p_node->get_child_count();i++) {

		_compute_hash(p_node->get_child(i),hash);
	}
}

void SceneTreeEditor::_test_update_tree() {

	pending_test_update=false;

	if (!is_inside_tree())
		return;

	if(tree_dirty)
		return; // don't even bother

	uint64_t hash = hash_djb2_one_64(0);
	if (get_scene_node())
		_compute_hash(get_scene_node(),hash);
	//test hash
	if (hash==last_hash)
		return; // did not change

	MessageQueue::get_singleton()->push_call(this,"_update_tree");
	tree_dirty=true;
}

void SceneTreeEditor::_tree_changed() {

	if (pending_test_update)
		return;
	if (tree_dirty)
		return;

	MessageQueue::get_singleton()->push_call(this,"_test_update_tree");
	pending_test_update=true;

}

void SceneTreeEditor::_selected_changed() {

	
	TreeItem *s = tree->get_selected();
	ERR_FAIL_COND(!s);
	NodePath np = s->get_metadata(0);
	
	Node *n=get_node(np);


	if (n==selected)
		return;


	selected = get_node(np);

	blocked++;
	emit_signal("node_selected");
	blocked--;
	
	
}


void SceneTreeEditor::_cell_multi_selected(Object *p_object,int p_cell,bool p_selected) {

	TreeItem *item = p_object->cast_to<TreeItem>();
	ERR_FAIL_COND(!item);

	NodePath np = item->get_metadata(0);

	Node *n=get_node(np);

	if (!n)
		return;

	if (!editor_selection)
		return;

	if (p_selected) {
		editor_selection->add_node(n);

	} else {
		editor_selection->remove_node(n);

	}

}

void SceneTreeEditor::_notification(int p_what) {
	
	if (p_what==NOTIFICATION_ENTER_TREE) {

		get_tree()->connect("tree_changed",this,"_tree_changed");
		get_tree()->connect("node_removed",this,"_node_removed");
		instance_menu->set_item_icon(2,get_icon("Load","EditorIcons"));
		tree->connect("item_collapsed",this,"_cell_collapsed");

//		get_scene()->connect("tree_changed",this,"_tree_changed",Vector<Variant>(),CONNECT_DEFERRED);
//		get_scene()->connect("node_removed",this,"_node_removed",Vector<Variant>(),CONNECT_DEFERRED);
		_update_tree();
	}
	if (p_what==NOTIFICATION_EXIT_TREE) {

		get_tree()->disconnect("tree_changed",this,"_tree_changed");
		get_tree()->disconnect("node_removed",this,"_node_removed");
		_update_tree();
	}

}


TreeItem* SceneTreeEditor::_find(TreeItem *p_node,const NodePath& p_path) {

	if (!p_node)
		return NULL;
		
	NodePath np=p_node->get_metadata(0);
	if (np==p_path)
		return p_node;
		
	TreeItem *children=p_node->get_children();
	while(children) {
	
		TreeItem *n=_find(children,p_path);
		if (n)
			return n;
		children=children->get_next();
	}
	
	return NULL;
}

void SceneTreeEditor::set_selected(Node *p_node,bool p_emit_selected) {

	ERR_FAIL_COND(blocked>0);

	if (pending_test_update)
		_test_update_tree();
	if (tree_dirty)
		_update_tree();

	if (selected==p_node)
		return;
		
	
	TreeItem* item=p_node?_find(tree->get_root(),p_node->get_path()):NULL;

	if (item) {
		item->select(0);
		item->set_as_cursor(0);
		selected=p_node;	
		tree->ensure_cursor_is_visible();
	} else {
		if (!p_node)
			selected=NULL;
		_update_tree();
		selected=p_node;	
		if (p_emit_selected)
			emit_signal("node_selected");
	}
	
}

void SceneTreeEditor::_rename_node(ObjectID p_node,const String& p_name) {

	Object *o = ObjectDB::get_instance(p_node);
	ERR_FAIL_COND(!o);
	Node *n = o->cast_to<Node>();
	ERR_FAIL_COND(!n);
	TreeItem* item=_find(tree->get_root(),n->get_path());
	ERR_FAIL_COND(!item);

	n->set_name( p_name );
	item->set_metadata(0,n->get_path());
	item->set_text(0,p_name);
	emit_signal("node_renamed");

	if (!tree_dirty) {
		MessageQueue::get_singleton()->push_call(this,"_update_tree");
		tree_dirty=true;
	}


}


void SceneTreeEditor::_renamed() {

	TreeItem *which=tree->get_edited();
	
	ERR_FAIL_COND(!which);
	NodePath np = which->get_metadata(0);
	Node *n=get_node(np);
	ERR_FAIL_COND(!n);

	if (!undo_redo) {
		n->set_name( which->get_text(0) );
		which->set_metadata(0,n->get_path());
		emit_signal("node_renamed");
	} else {
		undo_redo->create_action("Rename Node");
		emit_signal("node_prerename",n,which->get_text(0));
		undo_redo->add_do_method(this,"_rename_node",n->get_instance_ID(),which->get_text(0));
		undo_redo->add_undo_method(this,"_rename_node",n->get_instance_ID(),n->get_name());
		undo_redo->commit_action();
	}
}


Node *SceneTreeEditor::get_selected() {

	return selected;
}

void SceneTreeEditor::set_marked(const Set<Node*>& p_marked,bool p_selectable,bool p_children_selectable) {

	if (tree_dirty)
		_update_tree();
	marked=p_marked;
	marked_selectable=p_selectable;
	marked_children_selectable=p_children_selectable;
	_update_tree();
}

void SceneTreeEditor::set_marked(Node *p_marked,bool p_selectable,bool p_children_selectable) {

	Set<Node*> s;
	if (p_marked)
		s.insert(p_marked);
	set_marked(s,p_selectable,p_children_selectable);


}

void SceneTreeEditor::set_display_foreign_nodes(bool p_display) {

	display_foreign=p_display;
	_update_tree();
}
bool SceneTreeEditor::get_display_foreign_nodes() const {

	return display_foreign;
}

void SceneTreeEditor::set_editor_selection(EditorSelection *p_selection) {

	editor_selection=p_selection;
	tree->set_select_mode(Tree::SELECT_MULTI);
	tree->set_cursor_can_exit_tree(false);
	editor_selection->connect("selection_changed",this,"_selection_changed");
}

void SceneTreeEditor::_update_selection(TreeItem *item) {

	ERR_FAIL_COND(!item);

	NodePath np = item->get_metadata(0);

	Node *n=get_node(np);

	if (!n)
		return;

	if (editor_selection->is_selected(n))
		item->select(0);
	else
		item->deselect(0);

	TreeItem *c=item->get_children();


	while(c) {

		_update_selection(c);
		c=c->get_next();
	}
}

void SceneTreeEditor::_selection_changed() {

	if (!editor_selection)
		return;

	TreeItem *root=tree->get_root();

	if (!root)
		return;
	_update_selection(root);
}

void SceneTreeEditor::_cell_collapsed(Object *p_obj) {

	if (updating_tree)
		return;
	if (!can_rename)
		return;

	TreeItem *ti=p_obj->cast_to<TreeItem>();
	if (!ti)
		return;

	bool collapsed=ti->is_collapsed();

	NodePath np = ti->get_metadata(0);

	Node *n=get_node(np);
	ERR_FAIL_COND(!n);

	if (collapsed)
		n->set_meta("_editor_collapsed",true);
	else
		n->set_meta("_editor_collapsed",Variant());

}



void SceneTreeEditor::_bind_methods() {
	
	ObjectTypeDB::bind_method("_tree_changed",&SceneTreeEditor::_tree_changed);
	ObjectTypeDB::bind_method("_update_tree",&SceneTreeEditor::_update_tree);
	ObjectTypeDB::bind_method("_node_removed",&SceneTreeEditor::_node_removed);	
	ObjectTypeDB::bind_method("_selected_changed",&SceneTreeEditor::_selected_changed);
	ObjectTypeDB::bind_method("_renamed",&SceneTreeEditor::_renamed);
	ObjectTypeDB::bind_method("_rename_node",&SceneTreeEditor::_rename_node);
	ObjectTypeDB::bind_method("_test_update_tree",&SceneTreeEditor::_test_update_tree);
	ObjectTypeDB::bind_method("_cell_multi_selected",&SceneTreeEditor::_cell_multi_selected);
	ObjectTypeDB::bind_method("_selection_changed",&SceneTreeEditor::_selection_changed);
	ObjectTypeDB::bind_method("_cell_button_pressed",&SceneTreeEditor::_cell_button_pressed);
	ObjectTypeDB::bind_method("_cell_collapsed",&SceneTreeEditor::_cell_collapsed);
	ObjectTypeDB::bind_method("_subscene_option",&SceneTreeEditor::_subscene_option);

	ObjectTypeDB::bind_method("_node_script_changed",&SceneTreeEditor::_node_script_changed);
	ObjectTypeDB::bind_method("_node_visibility_changed",&SceneTreeEditor::_node_visibility_changed);

	ADD_SIGNAL( MethodInfo("node_selected") );
	ADD_SIGNAL( MethodInfo("node_renamed") );
	ADD_SIGNAL( MethodInfo("node_prerename") );
	ADD_SIGNAL( MethodInfo("node_changed") );

	ADD_SIGNAL( MethodInfo("open") );
	ADD_SIGNAL( MethodInfo("open_script") );


}


SceneTreeEditor::SceneTreeEditor(bool p_label,bool p_can_rename, bool p_can_open_instance) {
	

	undo_redo=NULL;
	tree_dirty=true;
	selected=NULL;

	marked_selectable=false;
	marked_children_selectable=false;
	can_rename=p_can_rename;
	can_open_instance=p_can_open_instance;
	display_foreign=false;
	editor_selection=NULL;
	
	if (p_label) {
		Label *label = memnew( Label );
		label->set_pos( Point2(10, 0));
		label->set_text("Scene Tree (Nodes):");
		
		add_child(label);
	}
	
	tree = memnew( Tree );
	tree->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	tree->set_anchor( MARGIN_BOTTOM, ANCHOR_END );	
	tree->set_begin( Point2(0,p_label?18:0 ));
	tree->set_end( Point2(0,0 ));
	
	add_child( tree );
		
	tree->connect("cell_selected", this,"_selected_changed");
	tree->connect("item_edited", this,"_renamed");
	tree->connect("multi_selected",this,"_cell_multi_selected");
	tree->connect("button_pressed",this,"_cell_button_pressed");
//	tree->connect("item_edited", this,"_renamed",Vector<Variant>(),true);

	error = memnew( AcceptDialog );
	add_child(error);

	show_enabled_subscene=false;

	last_hash=0;
	pending_test_update=false;
	updating_tree=false;
	blocked=0;

	instance_menu = memnew( PopupMenu );
	instance_menu->add_check_item("Show Children",SCENE_MENU_SHOW_CHILDREN);
	instance_menu->add_separator();
	instance_menu->add_item("Open in Editor",SCENE_MENU_OPEN);
	instance_menu->connect("item_pressed",this,"_subscene_option");
	add_child(instance_menu);

}



SceneTreeEditor::~SceneTreeEditor() {
	
}


/******** DIALOG *********/

void SceneTreeDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {
		connect("confirmed", this,"_select");

	}
	if (p_what==NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	}

	if (p_what==NOTIFICATION_VISIBILITY_CHANGED && is_visible()) {

		tree->update_tree();
	}


}

void SceneTreeDialog::_cancel() {

	hide();



}
void SceneTreeDialog::_select() {

	if (tree->get_selected()) {
	        emit_signal("selected",tree->get_selected()->get_path());
		hide();
	}
}

void SceneTreeDialog::_bind_methods() {

	ObjectTypeDB::bind_method("_select",&SceneTreeDialog::_select);
	ObjectTypeDB::bind_method("_cancel",&SceneTreeDialog::_cancel);
	ADD_SIGNAL( MethodInfo("selected",PropertyInfo(Variant::NODE_PATH,"path")));


}


SceneTreeDialog::SceneTreeDialog() {

	set_title("Select a Node");

	tree = memnew( SceneTreeEditor(false,false) );
	add_child(tree);
	set_child_rect(tree);



}


SceneTreeDialog::~SceneTreeDialog()
{
}

