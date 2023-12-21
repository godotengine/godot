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

#include "core/io/config_file.h"
#include "editor/editor_about.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/scroll_container.h"

class CheckBox;
class EditorAssetLibrary;
class EditorFileDialog;
class HFlowContainer;
class PanelContainer;
class ProjectList;

class ProjectDialog : public ConfirmationDialog {
	GDCLASS(ProjectDialog, ConfirmationDialog);

public:
	enum Mode {
		MODE_NEW,
		MODE_IMPORT,
		MODE_INSTALL,
		MODE_RENAME,
	};

private:
	enum MessageType {
		MESSAGE_ERROR,
		MESSAGE_WARNING,
		MESSAGE_SUCCESS,
	};

	enum InputType {
		PROJECT_PATH,
		INSTALL_PATH,
	};

	Mode mode = MODE_NEW;
	bool is_folder_empty = true;

	Button *browse = nullptr;
	Button *install_browse = nullptr;
	Button *create_dir = nullptr;
	Container *name_container = nullptr;
	Container *path_container = nullptr;
	Container *install_path_container = nullptr;

	Container *renderer_container = nullptr;
	Label *renderer_info = nullptr;
	HBoxContainer *default_files_container = nullptr;
	Ref<ButtonGroup> renderer_button_group;

	Label *msg = nullptr;
	LineEdit *project_path = nullptr;
	LineEdit *project_name = nullptr;
	LineEdit *install_path = nullptr;
	TextureRect *status_rect = nullptr;
	TextureRect *install_status_rect = nullptr;

	OptionButton *vcs_metadata_selection = nullptr;

	EditorFileDialog *fdialog = nullptr;
	EditorFileDialog *fdialog_install = nullptr;
	AcceptDialog *dialog_error = nullptr;

	String zip_path;
	String zip_title;
	String fav_dir;

	String created_folder_path;

	void _set_message(const String &p_msg, MessageType p_type = MESSAGE_SUCCESS, InputType input_type = PROJECT_PATH);

	String _test_path();
	void _path_text_changed(const String &p_path);
	void _path_selected(const String &p_path);
	void _file_selected(const String &p_path);
	void _install_path_selected(const String &p_path);

	void _browse_path();
	void _browse_install_path();
	void _create_folder();

	void _text_changed(const String &p_text);
	void _nonempty_confirmation_ok_pressed();
	void _renderer_selected();
	void _remove_created_folder();

	void ok_pressed() override;
	void cancel_pressed() override;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_zip_path(const String &p_path);
	void set_zip_title(const String &p_title);
	void set_mode(Mode p_mode);
	void set_project_path(const String &p_path);

	void ask_for_path_and_show();
	void show_dialog();

	ProjectDialog();
};

class ProjectListItemControl : public HBoxContainer {
	GDCLASS(ProjectListItemControl, HBoxContainer)

	VBoxContainer *main_vbox = nullptr;
	TextureButton *favorite_button = nullptr;
	Button *explore_button = nullptr;

	TextureRect *project_icon = nullptr;
	Label *project_title = nullptr;
	Label *project_path = nullptr;
	TextureRect *project_unsupported_features = nullptr;
	HBoxContainer *tag_container = nullptr;

	bool project_is_missing = false;
	bool icon_needs_reload = true;
	bool is_selected = false;
	bool is_hovering = false;

	void _favorite_button_pressed();
	void _explore_button_pressed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_project_title(const String &p_title);
	void set_project_path(const String &p_path);
	void set_tags(const PackedStringArray &p_tags, ProjectList *p_parent_list);
	void set_project_icon(const Ref<Texture2D> &p_icon);
	void set_unsupported_features(PackedStringArray p_features);

	bool should_load_project_icon() const;
	void set_selected(bool p_selected);

	void set_is_favorite(bool p_favorite);
	void set_is_missing(bool p_missing);
	void set_is_grayed(bool p_grayed);

	ProjectListItemControl();
};

class ProjectList : public ScrollContainer {
	GDCLASS(ProjectList, ScrollContainer)

	friend class ProjectManager;

public:
	enum FilterOption {
		EDIT_DATE,
		NAME,
		PATH,
		TAGS,
	};

	// Can often be passed by copy
	struct Item {
		String project_name;
		String description;
		PackedStringArray tags;
		String tag_sort_string;
		String path;
		String icon;
		String main_scene;
		PackedStringArray unsupported_features;
		uint64_t last_edited = 0;
		bool favorite = false;
		bool grayed = false;
		bool missing = false;
		int version = 0;

		ProjectListItemControl *control = nullptr;

		Item() {}

		Item(const String &p_name,
				const String &p_description,
				const PackedStringArray &p_tags,
				const String &p_path,
				const String &p_icon,
				const String &p_main_scene,
				const PackedStringArray &p_unsupported_features,
				uint64_t p_last_edited,
				bool p_favorite,
				bool p_grayed,
				bool p_missing,
				int p_version) {
			project_name = p_name;
			description = p_description;
			tags = p_tags;
			path = p_path;
			icon = p_icon;
			main_scene = p_main_scene;
			unsupported_features = p_unsupported_features;
			last_edited = p_last_edited;
			favorite = p_favorite;
			grayed = p_grayed;
			missing = p_missing;
			version = p_version;
			control = nullptr;

			PackedStringArray sorted_tags = tags;
			sorted_tags.sort();
			tag_sort_string = String().join(sorted_tags);
		}

		_FORCE_INLINE_ bool operator==(const Item &l) const {
			return path == l.path;
		}
	};

private:
	bool project_opening_initiated = false;

	String _search_term;
	FilterOption _order_option = FilterOption::EDIT_DATE;
	HashSet<String> _selected_project_paths;
	String _last_clicked; // Project key
	VBoxContainer *_scroll_children = nullptr;
	int _icon_load_index = 0;

	Vector<Item> _projects;

	ConfigFile _config;
	String _config_path;

	void _panel_input(const Ref<InputEvent> &p_ev, Node *p_hb);
	void _favorite_pressed(Node *p_hb);
	void _show_project(const String &p_path);

	void _migrate_config();
	void _scan_folder_recursive(const String &p_path, List<String> *r_projects);

	void _clear_project_selection();
	void _toggle_project(int p_index);
	void _select_project_nocheck(int p_index);
	void _deselect_project_nocheck(int p_index);
	void _select_project_range(int p_begin, int p_end);

	void _create_project_item_control(int p_index);
	void _remove_project(int p_index, bool p_update_settings);

	static Item load_project_data(const String &p_property_key, bool p_favorite);
	void _update_icons_async();
	void _load_project_icon(int p_index);

	void _global_menu_new_window(const Variant &p_tag);
	void _global_menu_open_project(const Variant &p_tag);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static const char *SIGNAL_LIST_CHANGED;
	static const char *SIGNAL_SELECTION_CHANGED;
	static const char *SIGNAL_PROJECT_ASK_OPEN;

	void update_project_list();
	int get_project_count() const;

	void find_projects(const String &p_path);
	void find_projects_multiple(const PackedStringArray &p_paths);
	void sort_projects();

	void add_project(const String &dir_path, bool favorite);
	void set_project_version(const String &p_project_path, int version);
	int refresh_project(const String &dir_path);
	void ensure_project_visible(int p_index);

	void select_project(int p_index);
	void select_first_visible_project();
	void erase_selected_projects(bool p_delete_project_contents);
	Vector<Item> get_selected_projects() const;
	const HashSet<String> &get_selected_project_keys() const;
	int get_single_selected_index() const;

	bool is_any_project_missing() const;
	void erase_missing_projects();

	void set_search_term(String p_search_term);
	void add_search_tag(const String &p_tag);
	void set_order_option(int p_option);

	void update_dock_menu();
	void save_config();

	ProjectList();
};

class ProjectManager : public Control {
	GDCLASS(ProjectManager, Control);

	HashMap<String, Ref<Texture2D>> icon_type_cache;
	void _build_icon_type_cache(Ref<Theme> p_theme);

	static ProjectManager *singleton;

	void _update_size_limits();

	Panel *background_panel = nullptr;
	TabContainer *tabs = nullptr;
	ProjectList *_project_list = nullptr;

	LineEdit *search_box = nullptr;
	Label *loading_label = nullptr;
	OptionButton *filter_option = nullptr;
	PanelContainer *search_panel = nullptr;

	Button *create_btn = nullptr;
	Button *import_btn = nullptr;
	Button *scan_btn = nullptr;
	Button *open_btn = nullptr;
	Button *run_btn = nullptr;
	Button *rename_btn = nullptr;
	Button *manage_tags_btn = nullptr;
	Button *erase_btn = nullptr;
	Button *erase_missing_btn = nullptr;
	Button *about_btn = nullptr;

	VBoxContainer *local_projects_vb = nullptr;
	EditorAssetLibrary *asset_library = nullptr;

	Ref<StyleBox> tag_stylebox;

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

	HashSet<String> tag_set;
	PackedStringArray current_project_tags;
	PackedStringArray forbidden_tag_characters{ "/", "\\", "-" };
	ConfirmationDialog *tag_manage_dialog = nullptr;
	HFlowContainer *project_tags = nullptr;
	HFlowContainer *all_tags = nullptr;
	Label *tag_edit_error = nullptr;
	Button *create_tag_btn = nullptr;
	ConfirmationDialog *create_tag_dialog = nullptr;
	LineEdit *new_tag_name = nullptr;
	Label *tag_error = nullptr;

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

	void _manage_project_tags();
	void _add_project_tag(const String &p_tag);
	void _delete_project_tag(const String &p_tag);
	void _apply_project_tags();
	void _set_new_tag_name(const String p_name);
	void _create_new_tag();

	void _on_project_created(const String &dir);
	void _on_projects_updated();

	void _install_project(const String &p_zip_path, const String &p_title);

	void _dim_window();
	virtual void shortcut_input(const Ref<InputEvent> &p_ev) override;
	void _files_dropped(PackedStringArray p_files);

	void _version_button_pressed();
	void _on_order_option_changed(int p_idx);
	void _on_tab_changed(int p_tab);
	void _on_search_term_changed(const String &p_term);
	void _on_search_term_submitted(const String &p_text);

	static Ref<Texture2D> _file_dialog_get_icon(const String &p_path);
	static Ref<Texture2D> _file_dialog_get_thumbnail(const String &p_path);

protected:
	void _notification(int p_what);

public:
	static ProjectManager *get_singleton() { return singleton; }

	LineEdit *get_search_box();
	void add_new_tag(const String &p_tag);

	ProjectManager();
	~ProjectManager();
};

class ProjectTag : public HBoxContainer {
	GDCLASS(ProjectTag, HBoxContainer);

	String tag_string;
	Button *button = nullptr;
	bool display_close = false;

protected:
	void _notification(int p_what);

public:
	ProjectTag(const String &p_text, bool p_display_close = false);

	void connect_button_to(const Callable &p_callable);
	const String get_tag() const;
};

#endif // PROJECT_MANAGER_H
