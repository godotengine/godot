/**************************************************************************/
/*  groups_editor.h                                                       */
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

#ifndef GROUPS_EDITOR_H
#define GROUPS_EDITOR_H

#include "scene/gui/dialogs.h"

class Button;
class LineEdit;
class Tree;
class TreeItem;

class GroupDialog : public AcceptDialog {
	GDCLASS(GroupDialog, AcceptDialog);

	AcceptDialog *error = nullptr;

	SceneTree *scene_tree = nullptr;
	TreeItem *groups_root = nullptr;

	LineEdit *add_group_text = nullptr;
	Button *add_group_button = nullptr;

	Tree *groups = nullptr;

	Tree *nodes_to_add = nullptr;
	TreeItem *add_node_root = nullptr;
	LineEdit *add_filter = nullptr;

	Tree *nodes_to_remove = nullptr;
	TreeItem *remove_node_root = nullptr;
	LineEdit *remove_filter = nullptr;

	Label *group_empty = nullptr;

	Button *add_button = nullptr;
	Button *remove_button = nullptr;

	String selected_group;

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
	void _modify_group_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button);
	void _delete_group_item(const String &p_name);

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

	GroupDialog();
};

class GroupsEditor : public VBoxContainer {
	GDCLASS(GroupsEditor, VBoxContainer);

	Node *node = nullptr;
	TreeItem *groups_root = nullptr;

	GroupDialog *group_dialog = nullptr;
	AcceptDialog *error = nullptr;

	LineEdit *group_name = nullptr;
	Button *add = nullptr;
	Tree *tree = nullptr;

	String selected_group;

	void update_tree();
	void _add_group(const String &p_group = "");
	void _modify_group(Object *p_item, int p_column, int p_id, MouseButton p_button);
	void _group_name_changed(const String &p_new_text);

	void _group_selected();
	void _group_renamed();

	void _show_group_dialog();

protected:
	static void _bind_methods();

public:
	enum ModifyButton {
		DELETE_GROUP,
		COPY_GROUP,
	};

	void set_current(Node *p_node);

	GroupsEditor();
	~GroupsEditor();
};

#endif // GROUPS_EDITOR_H
