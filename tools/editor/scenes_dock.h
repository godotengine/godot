/*************************************************************************/
/*  scenes_dock.h                                                        */
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
#ifndef SCENES_DOCK_H
#define SCENES_DOCK_H

#include "scene/main/timer.h"
#include "scene/gui/control.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/box_container.h"
#include "os/dir_access.h"
#include "os/thread.h"

#include "editor_file_system.h"


class EditorNode;

class ScenesDockFilter;
class ScenesDock : public VBoxContainer {
	OBJ_TYPE( ScenesDock, VBoxContainer );

	EditorNode *editor;
	Set<String> favorites;

	Button *button_reload;
	Button *button_instance;
	Button *button_favorite;
	Button *button_open;
	Timer *timer;

	ScenesDockFilter *tree_filter;

	bool updating_tree;
	Tree * tree;
	bool _create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir);

	void _update_tree();
	void _rescan();
	void _favorites_toggled(bool);
	void _favorite_toggled();
	void _instance_pressed();
	void _open_pressed();
	void _save_favorites();

protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	String get_selected_path() const;

	ScenesDock(EditorNode *p_editor);
	~ScenesDock();
};

class ScenesDockFilter : public HBoxContainer {

	OBJ_TYPE( ScenesDockFilter, HBoxContainer );

private:
	friend class ScenesDock;

	enum Command {
		CMD_CLEAR_FILTER,
	};

	Tree *tree;
	OptionButton *filter_option;
	LineEdit *search_box;
	ToolButton *clear_search_button;

	enum FilterOption {
		FILTER_PATH, // NAME or Folder
		FILTER_NAME,
		FILTER_FOLDER,
	};
	FilterOption _current_filter;
	//Vector<String> filters;

	void _command(int p_command);
	void _search_text_changed(const String& p_newtext);
	void _setup_filters();
	void _file_filter_selected(int p_idx);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	String get_search_term();
	FilterOption get_file_filter();
	ScenesDockFilter();
};

#endif // SCENES_DOCK_H
