/**************************************************************************/
/*  project_manager.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef PROJECT_MANAGER_H
#define PROJECT_MANAGER_H

#include "editor/editor_about.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/tree.h"

class CheckBox;
class ProjectDialog;
class ProjectList;

enum FilterOption {
	EDIT_DATE,
	NAME,
	PATH,
};

class ProjectManager : public Control {
	GDCLASS(ProjectManager, Control);

	HashMap<String, Ref<Texture2D>> icon_type_cache;
	void _build_icon_type_cache(Ref<Theme> p_theme);

	static ProjectManager *singleton;

	TabContainer *tabs = nullptr;

	ProjectList *_project_list = nullptr;

	LineEdit *search_box = nullptr;
	Label *loading_label = nullptr;
	OptionButton *filter_option = nullptr;

	Button *create_btn = nullptr;
	Button *import_btn = nullptr;
	Button *scan_btn = nullptr;
	Button *open_btn = nullptr;
	Button *run_btn = nullptr;
	Button *rename_btn = nullptr;
	Button *erase_btn = nullptr;
	Button *erase_missing_btn = nullptr;
	Button *about_btn = nullptr;

	HBoxContainer *local_projects_hb = nullptr;
	EditorAssetLibrary *asset_library = nullptr;

	EditorFileDialog *scan_dir = nullptr;
	ConfirmationDialog *language_restart_ask = nullptr;

	ConfirmationDialog *erase_ask = nullptr;
	Label *erase_ask_label = nullptr;
	// Comment out for now until we have a better warning system to
	// ensure users delete their project only.
	//CheckBox *delete_project_contents = nullptr;

	ConfirmationDialog *erase_missing_ask = nullptr;
	ConfirmationDialog *multi_open_ask = nullptr;
	ConfirmationDialog *multi_run_ask = nullptr;
	ConfirmationDialog *multi_scan_ask = nullptr;
	ConfirmationDialog *ask_full_convert_dialog = nullptr;
	ConfirmationDialog *ask_update_settings = nullptr;
	ConfirmationDialog *open_templates = nullptr;
	EditorAbout *about = nullptr;

	HBoxContainer *settings_hb = nullptr;

	AcceptDialog *run_error_diag = nullptr;
	AcceptDialog *dialog_error = nullptr;
	ProjectDialog *npdialog = nullptr;

	Button *full_convert_button = nullptr;
	OptionButton *language_btn = nullptr;
	LinkButton *version_btn = nullptr;

	void _open_asset_library();
	void _scan_projects();
	void _run_project();
	void _run_project_confirm();
	void _open_selected_projects();
	void _open_selected_projects_ask();
	void _full_convert_button_pressed();
	void _perform_full_project_conversion();
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
	void _scan_dir(const String &path);

	void _install_project(const String &p_zip_path, const String &p_title);

	void _dim_window();
	virtual void shortcut_input(const Ref<InputEvent> &p_ev) override;
	void _files_dropped(PackedStringArray p_files);

	void _version_button_pressed();
	void _on_order_option_changed(int p_idx);
	void _on_tab_changed(int p_tab);
	void _on_search_term_changed(const String &p_term);

	static Ref<Texture2D> _file_dialog_get_icon(const String &p_path);
	static Ref<Texture2D> _file_dialog_get_thumbnail(const String &p_path);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ProjectManager *get_singleton() { return singleton; }

	ProjectManager();
	~ProjectManager();
};

#endif // PROJECT_MANAGER_H
