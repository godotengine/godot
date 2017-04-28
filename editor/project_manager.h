/*************************************************************************/
/*  project_manager.h                                                    */
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
#ifndef PROJECT_MANAGER_H
#define PROJECT_MANAGER_H

#include "editor/asset_library_editor_plugin.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"

class NewProjectDialog;
class ProjectListFilter;

class ProjectManager : public Control {
	GDCLASS(ProjectManager, Control);

	Button *erase_btn;
	Button *open_btn;
	Button *run_btn;

	FileDialog *scan_dir;

	EditorAssetLibrary *asset_library;

	ProjectListFilter *project_filter;

	ConfirmationDialog *erase_ask;
	ConfirmationDialog *multi_open_ask;
	ConfirmationDialog *multi_run_ask;
	ConfirmationDialog *multi_scan_ask;
	NewProjectDialog *npdialog;
	ScrollContainer *scroll;
	VBoxContainer *scroll_childs;
	Map<String, String> selected_list; // name -> main_scene
	String last_clicked;
	bool importing;

	HBoxContainer *projects_hb;

	TabContainer *tabs;

	Control *gui_base;

	void _scan_projects();
	void _run_project();
	void _run_project_confirm();
	void _open_project();
	void _open_project_confirm();
	void _import_project();
	void _new_project();
	void _erase_project();
	void _erase_project_confirm();
	void _update_project_buttons();
	void _exit_dialog();
	void _scan_begin(const String &p_base);

	void _load_recent_projects();
	void _on_project_created(const String &dir);
	void _update_scroll_pos(const String &dir);
	void _scan_dir(DirAccess *da, float pos, float total, List<String> *r_projects);

	void _install_project(const String &p_zip_path, const String &p_title);

	void _panel_draw(Node *p_hb);
	void _panel_input(const InputEvent &p_ev, Node *p_hb);
	void _unhandled_input(const InputEvent &p_ev);
	void _favorite_pressed(Node *p_hb);
	void _files_dropped(PoolStringArray p_files, int p_screen);
	void _scan_multiple_folders(PoolStringArray p_files);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	ProjectManager();
	~ProjectManager();
};

class ProjectListFilter : public HBoxContainer {

	GDCLASS(ProjectListFilter, HBoxContainer);

private:
	friend class ProjectManager;

	enum Command {
		CMD_CLEAR_FILTER,
	};

	OptionButton *filter_option;
	LineEdit *search_box;
	ToolButton *clear_search_button;

	enum FilterOption {
		FILTER_NAME,
		FILTER_PATH,
	};
	FilterOption _current_filter;

	void _command(int p_command);
	void _search_text_changed(const String &p_newtext);
	void _setup_filters();
	void _filter_option_selected(int p_idx);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	String get_search_term();
	FilterOption get_filter_option();
	ProjectListFilter();
};

#endif // PROJECT_MANAGER_H
