/*************************************************************************/
/*  filesystem_dock.h                                                    */
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
#ifndef FILESYSTEM_DOCK_H
#define FILESYSTEM_DOCK_H

#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"

#include "os/dir_access.h"
#include "os/thread.h"

#include "dependency_editor.h"
#include "editor_dir_dialog.h"
#include "editor_file_system.h"

class EditorNode;

class FileSystemDock : public VBoxContainer {
	GDCLASS(FileSystemDock, VBoxContainer);

public:
	enum DisplayMode {
		DISPLAY_THUMBNAILS,
		DISPLAY_LIST
	};

private:
	enum FileMenu {
		FILE_OPEN,
		FILE_INSTANCE,
		FILE_DEPENDENCIES,
		FILE_OWNERS,
		FILE_MOVE,
		FILE_REMOVE,
		FILE_REIMPORT,
		FILE_INFO,
		FILE_SHOW_IN_EXPLORER,
		FILE_COPY_PATH
	};

	enum FolderMenu {
		FOLDER_EXPAND_ALL,
		FOLDER_COLLAPSE_ALL
	};

	VBoxContainer *scanning_vb;
	ProgressBar *scanning_progress;
	VSplitContainer *split_box;
	VBoxContainer *file_list_vb;

	EditorNode *editor;
	Set<String> favorites;

	Button *button_reload;
	Button *button_favorite;
	Button *button_back;
	Button *button_display_mode;
	Button *button_hist_next;
	Button *button_hist_prev;
	LineEdit *current_path;
	LineEdit *search_box;
	TextureRect *search_icon;
	HBoxContainer *path_hb;

	bool split_mode;
	DisplayMode display_mode;

	PopupMenu *file_options;
	PopupMenu *folder_options;

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
	Tree *tree; //directories
	ItemList *files;

	void _file_multi_selected(int p_index, bool p_selected);
	void _file_selected();

	void _go_to_tree();
	void _go_to_dir(const String &p_dir);
	void _select_file(int p_idx);

	bool _create_tree(TreeItem *p_parent, EditorFileSystemDirectory *p_dir);
	void _thumbnail_done(const String &p_path, const Ref<Texture> &p_preview, const Variant &p_udata);
	void _find_inside_move_files(EditorFileSystemDirectory *efsd, Vector<String> &files);
	void _find_remaps(EditorFileSystemDirectory *efsd, Map<String, String> &renames, List<String> &to_remaps);

	void _rename_operation(const String &p_to_path);
	void _move_operation(const String &p_to_path);

	void _file_option(int p_option);
	void _folder_option(int p_option);
	void _update_files(bool p_keep_selection);
	void _change_file_display();

	void _fs_changed();
	void _fw_history();
	void _bw_history();
	void _push_to_history();

	void _dir_selected();
	void _update_tree();
	void _rescan();
	void _set_scanning_mode();

	void _favorites_pressed();
	void _open_pressed();
	void _dir_rmb_pressed(const Vector2 &local_mouse_pos);
	void _search_changed(const String &p_text);

	void _files_list_rmb_select(int p_item, const Vector2 &p_pos);

	struct FileInfo {
		String name;
		String path;
		StringName type;
		int import_status; //0 not imported, 1 - ok, 2- must reimport, 3- broken
		Vector<String> sources;

		bool operator<(const FileInfo &fi) const {
			return name < fi.name;
		}
	};

	void _search(EditorFileSystemDirectory *p_path, List<FileInfo> *matches, int p_max_items);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _preview_invalidated(const String &p_path);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	String get_selected_path() const;

	String get_current_path() const;
	void navigate_to_path(const String &p_path);
	void focus_on_filter();

	void fix_dependencies(const String &p_for_file);

	void set_display_mode(int p_mode);

	int get_split_offset() { return split_box->get_split_offset(); }
	void set_split_offset(int p_offset) { split_box->set_split_offset(p_offset); }
	void select_file(const String &p_file);

	FileSystemDock(EditorNode *p_editor);
	~FileSystemDock();
};

#endif // SCENES_DOCK_H
