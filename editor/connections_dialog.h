/*************************************************************************/
/*  connections_dialog.h                                                 */
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
#ifndef CONNECTIONS_DIALOG_H
#define CONNECTIONS_DIALOG_H

#include "editor/property_editor.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"
#include "undo_redo.h"

/**
@author Juan Linietsky <reduzio@gmail.com>
*/

class ConnectDialogBinds;

class ConnectDialog : public ConfirmationDialog {

	GDCLASS(ConnectDialog, ConfirmationDialog);

	ConfirmationDialog *error;
	LineEdit *dst_path;
	LineEdit *dst_method;
	SceneTreeEditor *tree;
	//MenuButton *dst_method_list;
	OptionButton *type_list;
	CheckButton *deferred;
	CheckButton *oneshot;
	CheckButton *make_callback;
	PropertyEditor *bind_editor;
	Node *node;
	ConnectDialogBinds *cdbinds;
	void ok_pressed();
	void _cancel_pressed();
	void _tree_node_selected();
	void _dst_method_list_selected(int p_idx);
	void _add_bind();
	void _remove_bind();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	bool get_make_callback() { return make_callback->is_visible() && make_callback->is_pressed(); }
	NodePath get_dst_path() const;
	StringName get_dst_method() const;
	bool get_deferred() const;
	bool get_oneshot() const;
	Vector<Variant> get_binds() const;
	void set_dst_method(const StringName &p_method);
	void set_dst_node(Node *p_node);

	//Button *get_ok() { return ok; }
	//Button *get_cancel() { return cancel; }
	void edit(Node *p_node);

	ConnectDialog();
	~ConnectDialog();
};

class ConnectionsDock : public VBoxContainer {

	GDCLASS(ConnectionsDock, VBoxContainer);

	Button *connect_button;
	EditorNode *editor;
	Node *node;
	Tree *tree;
	ConfirmationDialog *remove_confirm;
	ConnectDialog *connect_dialog;

	void update_tree();

	void _close();
	void _connect();
	void _something_selected();
	void _something_activated();
	UndoRedo *undo_redo;

protected:
	void _connect_pressed();
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_undoredo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }

	void set_node(Node *p_node);
	String get_selected_type();

	ConnectionsDock(EditorNode *p_editor = NULL);
	~ConnectionsDock();
};

#endif
