/*************************************************************************/
/*  groups_editor.h                                                      */
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
#ifndef GROUPS_EDITOR_H
#define GROUPS_EDITOR_H

#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"
#include "undo_redo.h"

/**
@author Juan Linietsky <reduzio@gmail.com>
*/

class GroupsEditor : public VBoxContainer {

	GDCLASS(GroupsEditor, VBoxContainer);

	Node *node;

	LineEdit *group_name;
	Button *add;
	Tree *tree;

	UndoRedo *undo_redo;

	void update_tree();
	void _add_group(const String &p_group = "");
	void _remove_group(Object *p_item, int p_column, int p_id);
	void _close();

protected:
	static void _bind_methods();

public:
	void set_undo_redo(UndoRedo *p_undoredo) { undo_redo = p_undoredo; }
	void set_current(Node *p_node);

	GroupsEditor();
	~GroupsEditor();
};

#endif
