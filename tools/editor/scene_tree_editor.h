/*************************************************************************/
/*  scene_tree_editor.h                                                  */
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
#ifndef SCENE_TREE_EDITOR_H
#define SCENE_TREE_EDITOR_H

#include "scene/gui/tree.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "undo_redo.h"
#include "editor_data.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class SceneTreeEditor : public Control {
	
	OBJ_TYPE( SceneTreeEditor, Control );

	EditorSelection *editor_selection;

	enum {
		BUTTON_SUBSCENE=0,
		BUTTON_VISIBILITY=1,
		BUTTON_SCRIPT=2,
		BUTTON_LOCK=3,
		BUTTON_GROUP=4,
	};

	enum {
		SCENE_MENU_EDITABLE_CHILDREN,
		SCENE_MENU_USE_PLACEHOLDER,
		SCENE_MENU_OPEN,
		SCENE_MENU_CLEAR_INHERITANCE,
		SCENE_MENU_OPEN_INHERITED,
		SCENE_MENU_CLEAR_INHERITANCE_CONFIRM,
	};

	Tree *tree;
	Node *selected;
	PopupMenu *instance_menu;
	PopupMenu *inheritance_menu;
	ObjectID instance_node;

	AcceptDialog *error;
	ConfirmationDialog *clear_inherit_confirm;

	int blocked;
	
	void _compute_hash(Node *p_node,uint64_t &hash);

	void _add_nodes(Node *p_node,TreeItem *p_parent);
	void _test_update_tree();
	void _update_tree();
	void _tree_changed();
	void _node_removed(Node *p_node);
		
	TreeItem* _find(TreeItem *p_node,const NodePath& p_path);
	void _notification(int p_what);
	void _selected_changed();
	void _rename_node(ObjectID p_node,const String& p_name);

	void _cell_collapsed(Object *p_obj);

	uint64_t last_hash;

	bool can_rename;
	bool can_open_instance;
	bool updating_tree;
	bool show_enabled_subscene;
	
	void _renamed();
	UndoRedo *undo_redo;
	
	Set<Node*> marked;
	bool marked_selectable;
	bool marked_children_selectable;
	bool display_foreign;
	bool tree_dirty;
	bool pending_test_update;
	static void _bind_methods();	

	void _cell_button_pressed(Object *p_item,int p_column,int p_id);
	void _cell_multi_selected(Object *p_object,int p_cel,bool p_selected);
	void _update_selection(TreeItem *item);
	void _node_script_changed(Node *p_node);
	void _node_visibility_changed(Node *p_node);
	void _subscene_option(int p_idx);

	void _selection_changed();
	Node *get_scene_node();
public:


	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo=p_undo_redo; };
	void set_display_foreign_nodes(bool p_display);
	bool get_display_foreign_nodes() const;
	
	void set_marked(const Set<Node*>& p_marked,bool p_selectable=false,bool p_children_selectable=true);
	void set_marked(Node *p_marked,bool p_selectable=false,bool p_children_selectable=true);
	void set_selected(Node *p_node,bool p_emit_selected=true);
	Node *get_selected();
	void set_can_rename(bool p_can_rename) { can_rename=p_can_rename; }
	void set_editor_selection(EditorSelection *p_selection);

	void set_show_enabled_subscene(bool p_show) { show_enabled_subscene=p_show; }

	void update_tree() { _update_tree(); }


	Tree* get_scene_tree() { return tree; }

	SceneTreeEditor(bool p_label=true,bool p_can_rename=false, bool p_can_open_instance=false);
	~SceneTreeEditor();

};


class SceneTreeDialog : public ConfirmationDialog {

	OBJ_TYPE( SceneTreeDialog, ConfirmationDialog );

	SceneTreeEditor *tree;
//	Button *select;
//	Button *cancel;

	void update_tree();
	void _select();
	void _cancel();	


protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	SceneTreeEditor *get_scene_tree() { return tree; }
	SceneTreeDialog();
	~SceneTreeDialog();

};


#endif
