/*************************************************************************/
/*  groups_editor.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GROUPS_EDITOR_H
#define GROUPS_EDITOR_H

#include "core/object/undo_redo.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class GroupDialog : public AcceptDialog {
	GDCLASS(GroupDialog, AcceptDialog);

	ConfirmationDialog *error;

	SceneTree *scene_tree;
	TreeItem *groups_root;

	LineEdit *add_group_text;
	Button *add_group_button;

	Tree *groups;

	Tree *nodes_to_add;
	TreeItem *add_node_root;
	LineEdit *add_filter;

	Tree *nodes_to_remove;
	TreeItem *remove_node_root;
	LineEdit *remove_filter;

	Label *group_empty;

	Button *add_button;
	Button *remove_button;

	String selected_group;

	UndoRedo *undo_redo;

	void _group_selected();

	void _remove_filter_changed(const String &p_filter);
	void _add_filter_changed(const String &p_filter);

	void _add_pressed();
	void _removed_pressed();
	void _add_group_pressed(const String &p_name);
	void _add_group_text_changed(const String &p_new_text);

	void _group_renamed();
	void _rename_group_item(const String &p_old_name, const String &p_new_name);

	void _add_group(String p_name);
	void _modify_group_pressed(Object *p_item, int p_column, int p_id);
	void _delete_group_item(const String &p_name);

	bool _can_edit(Node *p_node, String p_group);

	void _load_groups(Node *p_current);
	void _load_nodes(Node *p_current);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	enum ModifyButton {
		DELETE_GROUP,
		COPY_GROUP,
	};

	void edit();
	void set_undo_redo(UndoRedo *p_undoredo) { undo_redo = p_undoredo; }

	GroupDialog();
};

class GroupsEditor : public VBoxContainer {
	GDCLASS(GroupsEditor, VBoxContainer);

	Node *node;

	GroupDialog *group_dialog;

	LineEdit *group_name;
	Button *add;
	Tree *tree;

	UndoRedo *undo_redo;

	void update_tree();
	void _add_group(const String &p_group = "");
	void _modify_group(Object *p_item, int p_column, int p_id);
	void _group_name_changed(const String &p_new_text);

	void _show_group_dialog();

protected:
	static void _bind_methods();

public:
	enum ModifyButton {
		DELETE_GROUP,
		COPY_GROUP,
	};

	void set_undo_redo(UndoRedo *p_undoredo) { undo_redo = p_undoredo; }
	void set_current(Node *p_node);

	GroupsEditor();
	~GroupsEditor();
};

#endif
