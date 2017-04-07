/*************************************************************************/
/*  editor_node.h                                                        */
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
#ifndef EDITOR_NODE_H
#define EDITOR_NODE_H

#include "editor/call_dialog.h"
#include "editor/connections_dialog.h"
#include "editor/create_dialog.h"
#include "editor/editor_data.h"
#include "editor/editor_path.h"
#include "editor/editor_run.h"
#include "editor/export_template_manager.h"
#include "editor/filesystem_dock.h"
#include "editor/groups_editor.h"
#include "editor/import_dock.h"
#include "editor/node_dock.h"
#include "editor/property_editor.h"
#include "editor/reparent_dialog.h"
#include "editor/scene_tree_editor.h"
#include "editor/settings_config_dialog.h"
#include "scene/gui/center_container.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_progress.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"

#include "editor/pane_drag.h"

#include "editor/editor_log.h"
#include "editor/editor_run_script.h"
#include "editor/project_export.h"
#include "editor/project_settings.h"
#include "editor/resources_dock.h"
#include "editor/run_settings_dialog.h"
#include "editor/scene_tree_dock.h"
#include "editor/script_create_dialog.h"

#include "editor/editor_name_dialog.h"
#include "editor/editor_plugin.h"
#include "editor/editor_run_native.h"
#include "editor/editor_sub_scene.h"
#include "editor/project_export.h"
#include "editor/quick_open.h"
#include "editor_export.h"
#include "editor_reimport_dialog.h"
#include "scene/gui/tabs.h"

#include "editor_resource_preview.h"
#include "fileserver/editor_file_server.h"
#include "scene/gui/viewport_container.h"

#include "progress_dialog.h"

#include "editor_scale.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

typedef void (*EditorNodeInitCallback)();
typedef void (*EditorPluginInitializeCallback)();
typedef void (*EditorBuildCallback)();

class EditorPluginList;

class EditorNode : public Node {

	GDCLASS(EditorNode, Node);

public:
	enum DockSlot {
		DOCK_SLOT_LEFT_UL,
		DOCK_SLOT_LEFT_BL,
		DOCK_SLOT_LEFT_UR,
		DOCK_SLOT_LEFT_BR,
		DOCK_SLOT_RIGHT_UL,
		DOCK_SLOT_RIGHT_BL,
		DOCK_SLOT_RIGHT_UR,
		DOCK_SLOT_RIGHT_BR,
		DOCK_SLOT_MAX
	};

private:
	enum {
		HISTORY_SIZE = 64
	};

	enum MenuOptions {
		FILE_NEW_SCENE,
		FILE_NEW_INHERITED_SCENE,
		FILE_OPEN_SCENE,
		FILE_SAVE_SCENE,
		FILE_SAVE_AS_SCENE,
		FILE_SAVE_ALL_SCENES,
		FILE_SAVE_BEFORE_RUN,
		FILE_SAVE_AND_RUN,
		FILE_IMPORT_SUBSCENE,
		FILE_EXPORT_PROJECT,
		FILE_EXPORT_MESH_LIBRARY,
		FILE_EXPORT_TILESET,
		FILE_SAVE_OPTIMIZED,
		FILE_OPEN_RECENT,
		FILE_OPEN_OLD_SCENE,
		FILE_QUICK_OPEN_SCENE,
		FILE_QUICK_OPEN_SCRIPT,
		FILE_RUN_SCRIPT,
		FILE_OPEN_PREV,
		FILE_CLOSE,
		FILE_QUIT,
		FILE_EXTERNAL_OPEN_SCENE,
		EDIT_UNDO,
		EDIT_REDO,
		EDIT_REVERT,
		TOOLS_ORPHAN_RESOURCES,
		RESOURCE_NEW,
		RESOURCE_LOAD,
		RESOURCE_SAVE,
		RESOURCE_SAVE_AS,
		RESOURCE_UNREF,
		RESOURCE_COPY,
		RESOURCE_PASTE,
		OBJECT_COPY_PARAMS,
		OBJECT_PASTE_PARAMS,
		OBJECT_UNIQUE_RESOURCES,
		OBJECT_REQUEST_HELP,
		RUN_PLAY,

		RUN_STOP,
		RUN_PLAY_SCENE,
		RUN_PLAY_NATIVE,
		RUN_PLAY_CUSTOM_SCENE,
		RUN_SCENE_SETTINGS,
		RUN_SETTINGS,
		RUN_PROJECT_MANAGER,
		RUN_FILE_SERVER,
		//RUN_DEPLOY_DUMB_CLIENTS,
		RUN_LIVE_DEBUG,
		RUN_DEBUG_COLLISONS,
		RUN_DEBUG_NAVIGATION,
		RUN_DEPLOY_REMOTE_DEBUG,
		RUN_RELOAD_SCRIPTS,
		SETTINGS_UPDATE_ALWAYS,
		SETTINGS_UPDATE_CHANGES,
		SETTINGS_UPDATE_SPINNER_HIDE,
		SETTINGS_EXPORT_PREFERENCES,
		SETTINGS_PREFERENCES,
		SETTINGS_OPTIMIZED_PRESETS,
		SETTINGS_LAYOUT_SAVE,
		SETTINGS_LAYOUT_DELETE,
		SETTINGS_LAYOUT_DEFAULT,
		SETTINGS_MANAGE_EXPORT_TEMPLATES,
		SETTINGS_PICK_MAIN_SCENE,
		SETTINGS_TOGGLE_FULLSCREN,
		SETTINGS_HELP,
		SETTINGS_ABOUT,
		SOURCES_REIMPORT,
		DEPENDENCY_LOAD_CHANGED_IMAGES,
		DEPENDENCY_UPDATE_IMPORTED,
		SCENE_TAB_CLOSE,

		IMPORT_PLUGIN_BASE = 100,

		OBJECT_METHOD_BASE = 500,

		TOOL_MENU_BASE = 1000
	};

	//Node *edited_scene; //scene being edited
	Viewport *scene_root; //root of the scene being edited

	//Ref<ResourceImportMetadata> scene_import_metadata;

	PanelContainer *scene_root_parent;
	Control *gui_base;
	VBoxContainer *main_vbox;

	//split

	HSplitContainer *left_l_hsplit;
	VSplitContainer *left_l_vsplit;
	HSplitContainer *left_r_hsplit;
	VSplitContainer *left_r_vsplit;
	HSplitContainer *main_hsplit;
	HSplitContainer *right_hsplit;
	VSplitContainer *right_l_vsplit;
	VSplitContainer *right_r_vsplit;

	VSplitContainer *center_split;

	//main tabs

	Tabs *scene_tabs;
	int tab_closing;

	bool exiting;

	int old_split_ofs;
	VSplitContainer *top_split;
	HBoxContainer *bottom_hb;
	Control *vp_base;
	PaneDrag *pd;
	//PaneDrag *pd_anim;
	Panel *menu_panel;

	//HSplitContainer *editor_hsplit;
	//VSplitContainer *editor_vsplit;
	CenterContainer *play_cc;
	HBoxContainer *menu_hb;
	Control *viewport;
	MenuButton *file_menu;
	MenuButton *import_menu;
	MenuButton *tool_menu;
	ToolButton *export_button;
	ToolButton *prev_scene;
	MenuButton *object_menu;
	MenuButton *settings_menu;
	ToolButton *play_button;
	MenuButton *native_play_button;
	ToolButton *pause_button;
	ToolButton *stop_button;
	ToolButton *run_settings_button;
	ToolButton *play_scene_button;
	ToolButton *play_custom_scene_button;
	MenuButton *debug_button;
	ToolButton *search_button;
	TextureProgress *audio_vu;
	//MenuButton *fileserver_menu;

	RichTextLabel *load_errors;
	AcceptDialog *load_error_dialog;

	//Control *scene_root_base;
	Ref<Theme> theme;

	PopupMenu *recent_scenes;
	Button *property_back;
	Button *property_forward;
	SceneTreeDock *scene_tree_dock;
	//ResourcesDock *resources_dock;
	PropertyEditor *property_editor;
	NodeDock *node_dock;
	ImportDock *import_dock;
	VBoxContainer *prop_editor_vb;
	FileSystemDock *filesystem_dock;
	EditorRunNative *run_native;

	HBoxContainer *search_bar;
	LineEdit *search_box;

	CreateDialog *create_dialog;

	//CallDialog *call_dialog;
	ConfirmationDialog *confirmation;
	ConfirmationDialog *import_confirmation;
	ConfirmationDialog *open_recent_confirmation;
	ConfirmationDialog *pick_main_scene;
	AcceptDialog *accept;
	AcceptDialog *about;
	AcceptDialog *warning;

	int overridden_default_layout;
	Ref<ConfigFile> default_layout;
	PopupMenu *editor_layouts;
	EditorNameDialog *layout_dialog;

	//OptimizedPresetsDialog *optimized_presets;
	EditorSettingsDialog *settings_config_dialog;
	RunSettingsDialog *run_settings_dialog;
	ProjectSettings *project_settings;
	EditorFileDialog *file;
	ExportTemplateManager *export_template_manager;
	FileDialog *file_templates;
	FileDialog *file_export;
	FileDialog *file_export_lib;
	FileDialog *file_script;
	CheckButton *file_export_lib_merge;
	LineEdit *file_export_password;
	String current_path;
	MenuButton *update_menu;
	ToolButton *sources_button;
	//TabContainer *prop_pallete;
	//TabContainer *top_pallete;
	String defer_load_scene;
	String defer_export;
	String defer_export_platform;
	bool defer_export_debug;
	Node *_last_instanced_scene;
	EditorPath *editor_path;
	ToolButton *resource_new_button;
	ToolButton *resource_load_button;
	MenuButton *resource_save_button;
	MenuButton *editor_history_menu;

	EditorLog *log;
	CenterContainer *tabs_center;
	EditorQuickOpen *quick_open;
	EditorQuickOpen *quick_run;

	HBoxContainer *main_editor_button_vb;
	Vector<ToolButton *> main_editor_buttons;
	Vector<EditorPlugin *> editor_table;

	//	EditorReImportDialog *reimport_dialog;

	ProgressDialog *progress_dialog;
	BackgroundProgress *progress_hb;

	DependencyErrorDialog *dependency_error;
	DependencyEditor *dependency_fixer;
	OrphanResourcesDialog *orphan_resources;
	ConfirmationDialog *open_imported;
	Button *new_inherited_button;
	String open_import_request;

	TabContainer *dock_slot[DOCK_SLOT_MAX];
	Rect2 dock_select_rect[DOCK_SLOT_MAX];
	int dock_select_rect_over;
	PopupPanel *dock_select_popoup;
	Control *dock_select;
	ToolButton *dock_tab_move_left;
	ToolButton *dock_tab_move_right;
	int dock_popup_selected;
	Timer *dock_drag_timer;
	bool docks_visible;
	ToolButton *distraction_free;

	String _tmp_import_path;

	EditorExport *editor_export;

	Object *current;

	bool _playing_edited;
	String run_custom_filename;
	bool reference_resource_mem;
	bool save_external_resources_mem;
	uint64_t saved_version;
	uint64_t last_checked_version;
	bool unsaved_cache;
	String open_navigate;
	bool changing_scene;
	bool waiting_for_first_scan;

	bool waiting_for_sources_changed;

	uint32_t circle_step_msec;
	uint64_t circle_step_frame;
	int circle_step;

	Vector<EditorPlugin *> editor_plugins;
	EditorPlugin *editor_plugin_screen;
	EditorPluginList *editor_plugins_over;

	EditorHistory editor_history;
	EditorData editor_data;
	EditorRun editor_run;
	EditorSelection *editor_selection;
	//	ProjectExport *project_export;
	ProjectExportDialog *project_export;
	EditorResourcePreview *resource_preview;

	EditorFileServer *file_server;

	struct BottomPanelItem {
		String name;
		Control *control;
		ToolButton *button;
	};

	Vector<BottomPanelItem> bottom_panel_items;

	PanelContainer *bottom_panel;
	HBoxContainer *bottom_panel_hb;
	VBoxContainer *bottom_panel_vb;

	void _bottom_panel_switch(bool p_enable, int p_idx);

	String external_file;
	List<String> previous_scenes;
	bool opening_prev;

	void _dialog_action(String p_file);

	void _edit_current();
	void _dialog_display_file_error(String p_file, Error p_error);

	int current_option;
	//void _animation_visibility_toggle();
	void _resource_created();
	void _resource_selected(const RES &p_res, const String &p_property = "");
	void _menu_option(int p_option);
	void _menu_confirm_current();
	void _menu_option_confirm(int p_option, bool p_confirmed);
	void _update_debug_options();

	void _property_editor_forward();
	void _property_editor_back();

	void _select_history(int p_idx);
	void _prepare_history();

	void _fs_changed();
	void _sources_changed(bool p_exist);
	void _imported(Node *p_node);

	void _node_renamed();
	void _editor_select(int p_which);
	void _set_scene_metadata(const String &p_file, int p_idx = -1);
	void _get_scene_metadata(const String &p_file);
	void _update_title();
	void _update_scene_tabs();
	void _close_messages();
	void _show_messages();
	void _vp_resized();

	void _rebuild_import_menu();

	void _save_scene(String p_file, int idx = -1);

	void _instance_request(const Vector<String> &p_files);

	void _property_keyed(const String &p_keyed, const Variant &p_value, bool p_advance);
	void _transform_keyed(Object *sp, const String &p_sub, const Transform &p_key);

	void _hide_top_editors();
	void _display_top_editors(bool p_display);
	void _set_top_editors(Vector<EditorPlugin *> p_editor_plugins_over);
	void _set_editing_top_editors(Object *p_current_object);

	void _quick_opened();
	void _quick_run();

	void _run(bool p_current = false, const String &p_custom = "");

	void _save_optimized();
	void _import_action(const String &p_action);
	void _import(const String &p_file);
	void _add_to_recent_scenes(const String &p_scene);
	void _update_recent_scenes();
	void _open_recent_scene(int p_idx);
	void _dropped_files(const Vector<String> &p_files, int p_screen);
	//void _open_recent_scene_confirm();
	String _recent_scene;

	bool convert_old;

	void _unhandled_input(const InputEvent &p_event);

	static void _load_error_notify(void *p_ud, const String &p_text);

	bool has_main_screen() const { return true; }

	bool _find_editing_changed_scene(Node *p_from);

	String import_reload_fn;

	Set<FileDialog *> file_dialogs;
	Set<EditorFileDialog *> editor_file_dialogs;

	Map<String, Ref<Texture> > icon_type_cache;

	bool _initializing_addons;
	Map<String, EditorPlugin *> plugin_addons;

	static Ref<Texture> _file_dialog_get_icon(const String &p_path);
	static void _file_dialog_register(FileDialog *p_dialog);
	static void _file_dialog_unregister(FileDialog *p_dialog);
	static void _editor_file_dialog_register(EditorFileDialog *p_dialog);
	static void _editor_file_dialog_unregister(EditorFileDialog *p_dialog);

	void _cleanup_scene();
	void _remove_edited_scene();
	void _remove_scene(int index);
	bool _find_and_save_resource(RES p_res, Map<RES, bool> &processed, int32_t flags);
	bool _find_and_save_edited_subresources(Object *obj, Map<RES, bool> &processed, int32_t flags);
	void _save_edited_subresources(Node *scene, Map<RES, bool> &processed, int32_t flags);

	void _find_node_types(Node *p_node, int &count_2d, int &count_3d);
	void _save_scene_with_preview(String p_file);

	Map<String, Set<String> > dependency_errors;

	static void _dependency_error_report(void *ud, const String &p_path, const String &p_dep, const String &p_type) {
		EditorNode *en = (EditorNode *)ud;
		if (!en->dependency_errors.has(p_path))
			en->dependency_errors[p_path] = Set<String>();
		en->dependency_errors[p_path].insert(p_dep + "::" + p_type);
	}

	struct ExportDefer {
		String platform;
		String path;
		bool debug;
		String password;

	} export_defer;

	static EditorNode *singleton;

	static Vector<EditorNodeInitCallback> _init_callbacks;

	bool _find_scene_in_use(Node *p_node, const String &p_path) const;

	void _dock_select_input(const InputEvent &p_input);
	void _dock_move_left();
	void _dock_move_right();
	void _dock_select_draw();
	void _dock_pre_popup(int p_which);
	void _dock_split_dragged(int ofs);
	void _dock_popup_exit();
	void _scene_tab_changed(int p_tab);
	void _scene_tab_closed(int p_tab);
	void _scene_tab_script_edited(int p_tab);

	Dictionary _get_main_scene_state();
	void _set_main_scene_state(Dictionary p_state, Node *p_for_scene);

	int _get_current_main_editor();

	void _save_docks();
	void _load_docks();
	void _save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section);
	void _load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section);
	void _update_dock_slots_visibility();
	void _update_top_menu_visibility();

	void _update_layouts_menu();
	void _layout_menu_option(int p_idx);

	void _toggle_search_bar(bool p_pressed);
	void _clear_search_box();
	void _clear_undo_history();

	void _update_addon_config();

	static void _file_access_close_error_notify(const String &p_str);

	void _toggle_distraction_free_mode();

	enum {
		MAX_INIT_CALLBACKS = 128,
		MAX_BUILD_CALLBACKS = 128
	};

	void _inherit_imported(const String &p_action);
	void _open_imported();

	static int plugin_init_callback_count;
	static EditorPluginInitializeCallback plugin_init_callbacks[MAX_INIT_CALLBACKS];

	void _call_build();
	static int build_callback_count;
	static EditorBuildCallback build_callbacks[MAX_BUILD_CALLBACKS];

	bool _initializing_tool_menu;

	struct ToolMenuItem {
		String name;
		String submenu;
		Variant ud;
		ObjectID handler;
		String callback;
	};

	Vector<ToolMenuItem> tool_menu_items;

	void _tool_menu_insert_item(const ToolMenuItem &p_item);
	void _rebuild_tool_menu() const;

	bool _dimming;
	float _dim_time;
	Timer *_dim_timer;

	void _start_dimming(bool p_dimming);
	void _dim_timeout();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static void add_plugin_init_callback(EditorPluginInitializeCallback p_callback);

	enum EditorTable {
		EDITOR_2D = 0,
		EDITOR_3D,
		EDITOR_SCRIPT
	};

	void set_visible_editor(EditorTable p_table) { _editor_select(p_table); }
	static EditorNode *get_singleton() { return singleton; }

	EditorPlugin *get_editor_plugin_screen() { return editor_plugin_screen; }
	EditorPluginList *get_editor_plugins_over() { return editor_plugins_over; }
	PropertyEditor *get_property_editor() { return property_editor; }
	VBoxContainer *get_property_editor_vb() { return prop_editor_vb; }

	static void add_editor_plugin(EditorPlugin *p_editor);
	static void remove_editor_plugin(EditorPlugin *p_editor);

	void new_inherited_scene() { _menu_option_confirm(FILE_NEW_INHERITED_SCENE, false); }

	void set_docks_visible(bool p_show);
	bool get_docks_visible() const;

	void set_distraction_free_mode(bool p_enter);
	bool get_distraction_free_mode() const;

	void add_control_to_dock(DockSlot p_slot, Control *p_control);
	void remove_control_from_dock(Control *p_control);

	void set_addon_plugin_enabled(const String &p_addon, bool p_enabled);
	bool is_addon_plugin_enabled(const String &p_addon) const;

	void edit_node(Node *p_node);
	void edit_resource(const Ref<Resource> &p_resource);
	void open_resource(const String &p_type = "");

	void save_resource_in_path(const Ref<Resource> &p_resource, const String &p_path);
	void save_resource(const Ref<Resource> &p_resource);
	void save_resource_as(const Ref<Resource> &p_resource, const String &p_at_path = String());

	void merge_from_scene() { _menu_option_confirm(FILE_IMPORT_SUBSCENE, false); }

	static bool has_unsaved_changes() { return singleton->unsaved_cache; }

	static HBoxContainer *get_menu_hb() { return singleton->menu_hb; }

	void push_item(Object *p_object, const String &p_property = "");

	void open_request(const String &p_path);

	bool is_changing_scene() const;

	static EditorLog *get_log() { return singleton->log; }
	Control *get_viewport();

	//void animation_editor_make_visible(bool p_visible);
	//void hide_animation_player_editors();
	//void animation_panel_make_visible(bool p_visible);

	void set_edited_scene(Node *p_scene);

	Node *get_edited_scene() { return editor_data.get_edited_scene_root(); }

	Viewport *get_scene_root() { return scene_root; } //root of the scene being edited

	void fix_dependencies(const String &p_for_file);
	void clear_scene() { _cleanup_scene(); }
	Error load_scene(const String &p_scene, bool p_ignore_broken_deps = false, bool p_set_inherited = false, bool p_clear_errors = true, bool p_force_open_imported = false);
	Error load_resource(const String &p_scene);

	bool is_scene_open(const String &p_path);

	void set_current_version(uint64_t p_version);
	void set_current_scene(int p_idx);

	static EditorData &get_editor_data() { return singleton->editor_data; }
	EditorHistory *get_editor_history() { return &editor_history; }

	static VSplitContainer *get_top_split() { return singleton->top_split; }

	void request_instance_scene(const String &p_path);
	void request_instance_scenes(const Vector<String> &p_files);
	FileSystemDock *get_filesystem_dock();
	ImportDock *get_import_dock();
	SceneTreeDock *get_scene_tree_dock();
	static UndoRedo *get_undo_redo() { return &singleton->editor_data.get_undo_redo(); }

	EditorSelection *get_editor_selection() { return editor_selection; }

	void set_convert_old_scene(bool p_old) { convert_old = p_old; }

	void notify_child_process_exited();

	OS::ProcessID get_child_process_id() const { return editor_run.get_pid(); }
	void stop_child_process();

	Ref<Theme> get_editor_theme() const { return theme; }

	void show_warning(const String &p_text, const String &p_title = "Warning!");

	Error export_platform(const String &p_platform, const String &p_path, bool p_debug, const String &p_password, bool p_quit_after = false);

	static void register_editor_types();
	static void unregister_editor_types();

	Control *get_gui_base() { return gui_base; }
	Control *get_theme_base() { return gui_base->get_parent_control(); }

	static void add_io_error(const String &p_error);

	static void progress_add_task(const String &p_task, const String &p_label, int p_steps);
	static void progress_task_step(const String &p_task, const String &p_state, int p_step = -1, bool p_force_refresh = true);
	static void progress_end_task(const String &p_task);

	static void progress_add_task_bg(const String &p_task, const String &p_label, int p_steps);
	static void progress_task_step_bg(const String &p_task, int p_step = -1);
	static void progress_end_task_bg(const String &p_task);

	void save_scene(String p_file) { _save_scene(p_file); }

	bool is_scene_in_use(const String &p_path);

	void scan_import_changes();

	void save_layout();

	void update_keying();

	void open_export_template_manager();

	void reload_scene(const String &p_path);

	bool is_exiting() const { return exiting; }

	ToolButton *get_pause_button() { return pause_button; }

	ToolButton *add_bottom_panel_item(String p_text, Control *p_item);
	bool are_bottom_panels_hidden() const;
	void make_bottom_panel_item_visible(Control *p_item);
	void raise_bottom_panel_item(Control *p_item);
	void hide_bottom_panel();
	void remove_bottom_panel_item(Control *p_item);

	Variant drag_resource(const Ref<Resource> &p_res, Control *p_from);
	Variant drag_files(const Vector<String> &p_files, Control *p_from);
	Variant drag_files_and_dirs(const Vector<String> &p_files, Control *p_from);

	void add_tool_menu_item(const String &p_name, Object *p_handler, const String &p_callback, const Variant &p_ud = Variant());
	void add_tool_submenu_item(const String &p_name, PopupMenu *p_submenu);
	void remove_tool_menu_item(const String &p_name);

	void dim_editor(bool p_dimming);

	EditorNode();
	~EditorNode();
	void get_singleton(const char *arg1, bool arg2);

	static void add_init_callback(EditorNodeInitCallback p_callback) { _init_callbacks.push_back(p_callback); }
	static void add_build_callback(EditorBuildCallback p_callback);
};

struct EditorProgress {

	String task;
	void step(const String &p_state, int p_step = -1, bool p_force_refresh = true) { EditorNode::progress_task_step(task, p_state, p_step, p_force_refresh); }
	EditorProgress(const String &p_task, const String &p_label, int p_amount) {
		EditorNode::progress_add_task(p_task, p_label, p_amount);
		task = p_task;
	}
	~EditorProgress() { EditorNode::progress_end_task(task); }
};

class EditorPluginList : public Object {
private:
	Vector<EditorPlugin *> plugins_list;

public:
	void set_plugins_list(Vector<EditorPlugin *> p_plugins_list) {
		plugins_list = p_plugins_list;
	}

	Vector<EditorPlugin *> &get_plugins_list() {
		return plugins_list;
	}

	void make_visible(bool p_visible);
	void edit(Object *p_object);
	bool forward_gui_input(const Transform2D &p_canvas_xform, const InputEvent &p_event);
	bool forward_spatial_gui_input(Camera *p_camera, const InputEvent &p_event);
	void forward_draw_over_canvas(const Transform2D &p_canvas_xform, Control *p_canvas);
	void clear();
	bool empty();

	EditorPluginList();
	~EditorPluginList();
};

struct EditorProgressBG {

	String task;
	void step(int p_step = -1) { EditorNode::progress_task_step_bg(task, p_step); }
	EditorProgressBG(const String &p_task, const String &p_label, int p_amount) {
		EditorNode::progress_add_task_bg(p_task, p_label, p_amount);
		task = p_task;
	}
	~EditorProgressBG() { EditorNode::progress_end_task_bg(task); }
};

#endif
