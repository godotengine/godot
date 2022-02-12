/*************************************************************************/
/*  project_manager.h                                                    */
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

#ifndef PROJECT_MANAGER_H
#define PROJECT_MANAGER_H

#include "editor/editor_about.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/tree.h"

class ProjectDialog;
class ProjectList;

enum FilterOption {
	NAME,
	PATH,
	EDIT_DATE,
};

class ProjectManager : public Control {
	GDCLASS(ProjectManager, Control);

	TabContainer *tabs;

	ProjectList *_project_list;

	LineEdit *search_box;
	Label *loading_label;
	OptionButton *filter_option;

	Button *run_btn;
	Button *open_btn;
	Button *rename_btn;
	Button *erase_btn;
	Button *erase_missing_btn;
	Button *about_btn;

	EditorAssetLibrary *asset_library;

	FileDialog *scan_dir;
	ConfirmationDialog *language_restart_ask;

	ConfirmationDialog *erase_ask;
	Label *erase_ask_label;
	CheckBox *delete_project_contents;

	ConfirmationDialog *erase_missing_ask;
	ConfirmationDialog *multi_open_ask;
	ConfirmationDialog *multi_run_ask;
	ConfirmationDialog *multi_scan_ask;
	ConfirmationDialog *ask_update_settings;
	ConfirmationDialog *open_templates;
	EditorAbout *about;

	HBoxContainer *settings_hb;

	AcceptDialog *run_error_diag;
	AcceptDialog *dialog_error;
	ProjectDialog *npdialog;

	OptionButton *language_btn;
	LinkButton *version_btn;

	void _open_asset_library();
	void _scan_projects();
	void _run_project();
	void _run_project_confirm();
	void _open_selected_projects();
	void _open_selected_projects_ask();
	void _import_project();
	void _new_project();
	void _rename_project();
	void _erase_project();
	void _erase_missing_projects();
	void _erase_project_confirm();
	void _erase_missing_projects_confirm();
	void _show_about();
	void _update_project_buttons();
	void _language_selected(int p_id);
	void _restart_confirm();
	void _confirm_update_settings();
	void _nonempty_confirmation_ok_pressed();

	void _load_recent_projects();
	void _on_project_created(const String &dir);
	void _on_projects_updated();
	void _scan_multiple_folders(PackedStringArray p_files);
	void _scan_begin(const String &p_base);
	void _scan_dir(const String &path, List<String> *r_projects);

	void _install_project(const String &p_zip_path, const String &p_title);

	void _dim_window();
	virtual void unhandled_key_input(const Ref<InputEvent> &p_ev) override;
	void _files_dropped(PackedStringArray p_files, int p_screen);

	void _version_button_pressed();
	void _on_order_option_changed(int p_idx);
	void _on_tab_changed(int p_tab);
	void _on_search_term_changed(const String &p_term);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	ProjectManager();
	~ProjectManager();
};

#endif // PROJECT_MANAGER_H
