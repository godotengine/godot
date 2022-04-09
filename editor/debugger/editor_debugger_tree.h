/*************************************************************************/
/*  editor_debugger_tree.h                                               */
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

#include "scene/gui/tree.h"

#ifndef EDITOR_DEBUGGER_TREE_H
#define EDITOR_DEBUGGER_TREE_H

class SceneDebuggerTree;
class EditorFileDialog;

class EditorDebuggerTree : public Tree {
	GDCLASS(EditorDebuggerTree, Tree);

private:
	enum ItemMenu {
		ITEM_MENU_SAVE_REMOTE_NODE,
		ITEM_MENU_COPY_NODE_PATH,
	};

	ObjectID inspected_object_id;
	int debugger_id = 0;
	bool updating_scene_tree = false;
	Set<ObjectID> unfold_cache;
	PopupMenu *item_menu = nullptr;
	EditorFileDialog *file_dialog = nullptr;
	String last_filter;

	String _get_path(TreeItem *p_item);
	void _scene_tree_folded(Object *p_obj);
	void _scene_tree_selected();
	void _scene_tree_rmb_selected(const Vector2 &p_position);
	void _item_menu_id_pressed(int p_option);
	void _file_selected(const String &p_file);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	String get_selected_path();
	ObjectID get_selected_object();
	int get_current_debugger(); // Would love to have one tree for every debugger.
	void update_scene_tree(const SceneDebuggerTree *p_tree, int p_debugger);
	EditorDebuggerTree();
};
#endif // EDITOR_DEBUGGER_TREE_H
