/*************************************************************************/
/*  editor_sub_scene.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef EDITOR_SUB_SCENE_H
#define EDITOR_SUB_SCENE_H

#include "editor/editor_file_dialog.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"

class EditorSubScene : public ConfirmationDialog {

	GDCLASS(EditorSubScene, ConfirmationDialog);

	List<Node *> selection;
	LineEdit *path;
	Tree *tree;
	Node *scene;
	bool is_root;

	EditorFileDialog *file_dialog;

	void _fill_tree(Node *p_node, TreeItem *p_parent);
	void _selected_changed();
	void _item_multi_selected(Object *p_object, int p_cell, bool p_selected);
	void _remove_selection_child(Node *c);
	void _reown(Node *p_node, List<Node *> *p_to_reown);

	void ok_pressed();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _path_browse();
	void _path_selected(const String &p_path);
	void _path_changed(const String &p_path);

public:
	void move(Node *p_new_parent, Node *p_new_owner);
	void clear();
	EditorSubScene();
};

#endif // EDITOR_SUB_SCENE_H
