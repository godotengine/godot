/*************************************************************************/
/*  scenes_dock.h                                                        */
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
#ifndef SCENES_DOCK_H
#define SCENES_DOCK_H

#include "scene/main/timer.h"
#include "scene/gui/control.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "os/dir_access.h"
#include "os/thread.h"

#include "editor_file_system.h"


class EditorNode;

class ScenesDock : public Control {
	OBJ_TYPE( ScenesDock, Control );

	EditorNode *editor;
	Set<String> favorites;

	Button *button_reload;
	Button *button_instance;
	Button *button_favorite;
	Button *button_open;
	Button *button_replace;
	Timer *timer;

	bool updating_tree;
	Tree * tree;
	bool _create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir);

	void _update_tree();
	void _rescan();
	void _favorites_toggled(bool);
	void _favorite_toggled();
	void _instance_pressed();
	void _open_pressed();
	void _replace_pressed();
	void _save_favorites();

protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	String get_selected_path() const;

	ScenesDock(EditorNode *p_editor);
	~ScenesDock();
};

#endif // SCENES_DOCK_H
