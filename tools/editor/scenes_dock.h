/*************************************************************************/
/*  scenes_dock.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "scene/gui/menu_button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/progress_bar.h"

#include "os/dir_access.h"
#include "os/thread.h"

#include "editor_file_system.h"
#include "editor_dir_dialog.h"
#include "dependency_editor.h"

class EditorNode;


class ScenesDock : public VBoxContainer {
	OBJ_TYPE( ScenesDock, VBoxContainer );

	enum FileMenu {
		FILE_DEPENDENCIES,
		FILE_OWNERS,
		FILE_MOVE,
		FILE_REMOVE,
		FILE_REIMPORT,
		FILE_INFO
	};


	VBoxContainer *scanning_vb;
	ProgressBar *scanning_progress;

	EditorNode *editor;
	Set<String> favorites;

	Button *button_reload;
	Button *button_instance;
	Button *button_favorite;
	Button *button_fav_up;
	Button *button_fav_down;
	Button *button_open;
	Button *button_back;
	Button *display_mode;
	Button *button_hist_next;
	Button *button_hist_prev;
	LineEdit *current_path;
	HBoxContainer *path_hb;

	MenuButton *file_options;


	DependencyEditor *deps_editor;
	DependencyEditorOwners *owners_editor;
	DependencyRemoveDialog *remove_dialog;

	EditorDirDialog *move_dialog;
	EditorFileDialog *rename_dialog;

	Vector<String> move_dirs;
	Vector<String> move_files;


	Vector<String> history;
	int history_pos;

	String path;

	bool initialized;

	bool updating_tree;
	Tree * tree; //directories
	ItemList *files;

	bool tree_mode;

	void _go_to_tree();
	void _go_to_dir(const String& p_dir);
	void _select_file(int p_idx);

	bool _create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir);
	void _thumbnail_done(const String& p_path,const Ref<Texture>& p_preview, const Variant& p_udata);
	void _find_inside_move_files(EditorFileSystemDirectory *efsd,Vector<String>& files);
	void _find_remaps(EditorFileSystemDirectory *efsd,Map<String,String> &renames,List<String>& to_remaps);

	void _rename_operation(const String& p_to_path);
	void _move_operation(const String& p_to_path);


	void _file_option(int p_option);
	void _update_files(bool p_keep_selection);
	void _change_file_display();

	void _fs_changed();
	void _fw_history();
	void _bw_history();
	void _push_to_history();

	void _fav_up_pressed();
	void _fav_down_pressed();
	void _dir_selected();
	void _update_tree();
	void _rescan();
	void _set_scannig_mode();

	void _favorites_pressed();	
	void _instance_pressed();
	void _open_pressed();


protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	String get_selected_path() const;
	void open(const String& p_path);

	void fix_dependencies(const String& p_for_file);

	void set_use_thumbnails(bool p_use);

	ScenesDock(EditorNode *p_editor);
	~ScenesDock();
};


#endif // SCENES_DOCK_H
