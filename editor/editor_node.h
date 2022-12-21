/*************************************************************************/
/*  editor_node.h                                                        */
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

#ifndef EDITOR_NODE_H
#define EDITOR_NODE_H

#include "core/templates/safe_refcount.h"
#include "editor/editor_data.h"
#include "editor/editor_folding.h"
#include "editor/editor_native_shader_source_visualizer.h"
#include "editor/editor_plugin.h"
#include "editor/editor_run.h"
#include "editor/editor_title_bar.h"
#include "editor/export/editor_export.h"

typedef void (*EditorNodeInitCallback)();
typedef void (*EditorPluginInitializeCallback)();
typedef bool (*EditorBuildCallback)();

class AcceptDialog;
class AudioStreamPreviewGenerator;
class BackgroundProgress;
class CenterContainer;
class CheckBox;
class ColorPicker;
class ConfirmationDialog;
class Control;
class DependencyEditor;
class DependencyErrorDialog;
class DynamicFontImportSettings;
class EditorAbout;
class EditorCommandPalette;
class EditorExport;
class EditorExtensionManager;
class EditorFeatureProfileManager;
class EditorFileDialog;
class EditorFileServer;
class EditorFolding;
class EditorInspector;
class EditorLayoutsDialog;
class EditorLog;
class EditorPluginList;
class EditorQuickOpen;
class EditorResourcePreview;
class EditorResourceConversionPlugin;
class EditorRun;
class EditorRunNative;
class EditorSelectionHistory;
class EditorSettingsDialog;
class EditorToaster;
class EditorUndoRedoManager;
class ExportTemplateManager;
class FBXImporterManager;
class FileDialog;
class FileSystemDock;
class HistoryDock;
class HSplitContainer;
class ImportDock;
class LinkButton;
class MenuBar;
class MenuButton;
class NodeDock;
class OptionButton;
class OrphanResourcesDialog;
class Panel;
class PanelContainer;
class PluginConfigDialog;
class ProgressDialog;
class ProjectExportDialog;
class ProjectSettingsEditor;
class RunSettingsDialog;
class SceneImportSettings;
class AudioStreamImportSettings;
class ScriptCreateDialog;
class SubViewport;
class TabBar;
class TabContainer;
class TextureRect;
class TextureProgressBar;
class Tree;
class VSplitContainer;
class Window;
class EditorBuildProfileManager;

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

	enum EditorTable {
		EDITOR_2D = 0,
		EDITOR_3D,
		EDITOR_SCRIPT,
		EDITOR_ASSETLIB
	};

	enum SceneNameCasing {
		SCENE_NAME_CASING_AUTO,
		SCENE_NAME_CASING_PASCAL_CASE,
		SCENE_NAME_CASING_SNAKE_CASE
	};

	struct ExecuteThreadArgs {
		String path;
		List<String> args;
		String output;
		Thread execute_output_thread;
		Mutex execute_output_mutex;
		int exitcode = 0;
		SafeFlag done;
	};

private:
	enum MenuOptions {
		FILE_NEW_SCENE,
		FILE_NEW_INHERITED_SCENE,
		FILE_OPEN_SCENE,
		FILE_SAVE_SCENE,
		FILE_SAVE_AS_SCENE,
		FILE_SAVE_ALL_SCENES,
		FILE_SAVE_AND_RUN,
		FILE_SAVE_AND_RUN_MAIN_SCENE,
		FILE_SHOW_IN_FILESYSTEM,
		FILE_EXPORT_PROJECT,
		FILE_EXPORT_MESH_LIBRARY,
		FILE_INSTALL_ANDROID_SOURCE,
		FILE_EXPLORE_ANDROID_BUILD_TEMPLATES,
		FILE_SAVE_OPTIMIZED,
		FILE_OPEN_RECENT,
		FILE_OPEN_OLD_SCENE,
		FILE_QUICK_OPEN,
		FILE_QUICK_OPEN_SCENE,
		FILE_QUICK_OPEN_SCRIPT,
		FILE_OPEN_PREV,
		FILE_CLOSE,
		FILE_CLOSE_OTHERS,
		FILE_CLOSE_RIGHT,
		FILE_CLOSE_ALL,
		FILE_CLOSE_ALL_AND_QUIT,
		FILE_CLOSE_ALL_AND_RUN_PROJECT_MANAGER,
		FILE_CLOSE_ALL_AND_RELOAD_CURRENT_PROJECT,
		FILE_QUIT,
		FILE_EXTERNAL_OPEN_SCENE,
		EDIT_UNDO,
		EDIT_REDO,
		EDIT_RELOAD_SAVED_SCENE,
		TOOLS_ORPHAN_RESOURCES,
		TOOLS_BUILD_PROFILE_MANAGER,
		TOOLS_CUSTOM,
		RESOURCE_SAVE,
		RESOURCE_SAVE_AS,
		RUN_PLAY,

		RUN_STOP,
		RUN_PLAY_SCENE,
		RUN_PLAY_CUSTOM_SCENE,
		RUN_SETTINGS,
		RUN_USER_DATA_FOLDER,
		RELOAD_CURRENT_PROJECT,
		RUN_PROJECT_MANAGER,
		RUN_VCS_METADATA,
		RUN_VCS_SETTINGS,
		SETTINGS_UPDATE_CONTINUOUSLY,
		SETTINGS_UPDATE_WHEN_CHANGED,
		SETTINGS_UPDATE_ALWAYS,
		SETTINGS_UPDATE_CHANGES,
		SETTINGS_UPDATE_SPINNER_HIDE,
		SETTINGS_PREFERENCES,
		SETTINGS_LAYOUT_SAVE,
		SETTINGS_LAYOUT_DELETE,
		SETTINGS_LAYOUT_DEFAULT,
		SETTINGS_EDITOR_DATA_FOLDER,
		SETTINGS_EDITOR_CONFIG_FOLDER,
		SETTINGS_MANAGE_EXPORT_TEMPLATES,
		SETTINGS_MANAGE_FBX_IMPORTER,
		SETTINGS_MANAGE_FEATURE_PROFILES,
		SETTINGS_INSTALL_ANDROID_BUILD_TEMPLATE,
		SETTINGS_PICK_MAIN_SCENE,
		SETTINGS_TOGGLE_FULLSCREEN,
		SETTINGS_HELP,
		SCENE_TAB_CLOSE,

		EDITOR_SCREENSHOT,
		EDITOR_OPEN_SCREENSHOT,

		HELP_SEARCH,
		HELP_COMMAND_PALETTE,
		HELP_DOCS,
		HELP_QA,
		HELP_REPORT_A_BUG,
		HELP_SUGGEST_A_FEATURE,
		HELP_SEND_DOCS_FEEDBACK,
		HELP_COMMUNITY,
		HELP_ABOUT,
		HELP_SUPPORT_GODOT_DEVELOPMENT,

		SET_RENDERER_NAME_SAVE_AND_RESTART,

		GLOBAL_NEW_WINDOW,
		GLOBAL_SCENE,

		IMPORT_PLUGIN_BASE = 100,

		TOOL_MENU_BASE = 1000
	};

	enum {
		MAX_INIT_CALLBACKS = 128,
		MAX_BUILD_CALLBACKS = 128
	};

	struct BottomPanelItem {
		String name;
		Control *control = nullptr;
		Button *button = nullptr;
	};

	struct ExportDefer {
		String preset;
		String path;
		bool debug = false;
		bool pack_only = false;
	} export_defer;

	static EditorNode *singleton;

	EditorData editor_data;
	EditorFolding editor_folding;
	EditorRun editor_run;
	EditorSelectionHistory editor_history;

	EditorCommandPalette *command_palette = nullptr;
	EditorExport *editor_export = nullptr;
	EditorInterface *editor_interface = nullptr;
	EditorLog *log = nullptr;
	EditorNativeShaderSourceVisualizer *native_shader_source_visualizer = nullptr;
	EditorPlugin *editor_plugin_screen = nullptr;
	EditorPluginList *editor_plugins_force_input_forwarding = nullptr;
	EditorPluginList *editor_plugins_force_over = nullptr;
	EditorPluginList *editor_plugins_over = nullptr;
	EditorQuickOpen *quick_open = nullptr;
	EditorQuickOpen *quick_run = nullptr;
	EditorResourcePreview *resource_preview = nullptr;
	EditorRunNative *run_native = nullptr;
	EditorSelection *editor_selection = nullptr;
	EditorSettingsDialog *editor_settings_dialog = nullptr;
	HistoryDock *history_dock = nullptr;

	ProjectExportDialog *project_export = nullptr;
	ProjectSettingsEditor *project_settings_editor = nullptr;

	FBXImporterManager *fbx_importer_manager = nullptr;

	Vector<EditorPlugin *> editor_plugins;
	bool _initializing_plugins = false;
	HashMap<String, EditorPlugin *> addon_name_to_plugin;

	PanelContainer *scene_root_parent = nullptr;
	Control *theme_base = nullptr;
	Control *gui_base = nullptr;
	VBoxContainer *main_vbox = nullptr;
	OptionButton *renderer = nullptr;

	ConfirmationDialog *video_restart_dialog = nullptr;

	int renderer_current = 0;
	String renderer_request;

	// Split containers.
	HSplitContainer *left_l_hsplit = nullptr;
	VSplitContainer *left_l_vsplit = nullptr;
	HSplitContainer *left_r_hsplit = nullptr;
	VSplitContainer *left_r_vsplit = nullptr;
	HSplitContainer *main_hsplit = nullptr;
	HSplitContainer *right_hsplit = nullptr;
	VSplitContainer *right_l_vsplit = nullptr;
	VSplitContainer *right_r_vsplit = nullptr;
	VSplitContainer *center_split = nullptr;
	// To access those easily by index.
	Vector<VSplitContainer *> vsplits;
	Vector<HSplitContainer *> hsplits;

	// Main tabs.
	TabBar *scene_tabs = nullptr;
	PopupMenu *scene_tabs_context_menu = nullptr;
	Panel *tab_preview_panel = nullptr;
	TextureRect *tab_preview = nullptr;
	int tab_closing_idx = 0;

	bool exiting = false;
	bool dimmed = false;

	int old_split_ofs = 0;
	VSplitContainer *top_split = nullptr;
	HBoxContainer *bottom_hb = nullptr;
	Control *vp_base = nullptr;

	Label *project_title = nullptr;
	Control *left_menu_spacer = nullptr;
	Control *right_menu_spacer = nullptr;
	EditorTitleBar *menu_hb = nullptr;
	VBoxContainer *main_screen_vbox = nullptr;
	MenuBar *main_menu = nullptr;
	PopupMenu *file_menu = nullptr;
	PopupMenu *project_menu = nullptr;
	PopupMenu *debug_menu = nullptr;
	PopupMenu *settings_menu = nullptr;
	PopupMenu *help_menu = nullptr;
	PopupMenu *tool_menu = nullptr;
	PopupMenu *export_as_menu = nullptr;
	Button *export_button = nullptr;
	Button *prev_scene = nullptr;
	Button *search_button = nullptr;
	TextureProgressBar *audio_vu = nullptr;

	PanelContainer *launch_pad = nullptr;
	Button *play_button = nullptr;
	Button *pause_button = nullptr;
	Button *stop_button = nullptr;
	Button *play_scene_button = nullptr;
	Button *play_custom_scene_button = nullptr;
	PanelContainer *write_movie_panel = nullptr;
	Button *write_movie_button = nullptr;

	Timer *screenshot_timer = nullptr;

	PluginConfigDialog *plugin_config_dialog = nullptr;

	RichTextLabel *load_errors = nullptr;
	AcceptDialog *load_error_dialog = nullptr;

	RichTextLabel *execute_outputs = nullptr;
	AcceptDialog *execute_output_dialog = nullptr;

	Ref<Theme> theme;

	PopupMenu *recent_scenes = nullptr;
	String _recent_scene;
	List<String> previous_scenes;
	String defer_load_scene;
	Node *_last_instantiated_scene = nullptr;

	ConfirmationDialog *confirmation = nullptr;
	ConfirmationDialog *save_confirmation = nullptr;
	ConfirmationDialog *import_confirmation = nullptr;
	ConfirmationDialog *pick_main_scene = nullptr;
	Button *select_current_scene_button = nullptr;
	AcceptDialog *accept = nullptr;
	AcceptDialog *save_accept = nullptr;
	EditorAbout *about = nullptr;
	AcceptDialog *warning = nullptr;

	int overridden_default_layout = -1;
	Ref<ConfigFile> default_layout;
	PopupMenu *editor_layouts = nullptr;
	EditorLayoutsDialog *layout_dialog = nullptr;

	ConfirmationDialog *custom_build_manage_templates = nullptr;
	ConfirmationDialog *install_android_build_template = nullptr;
	ConfirmationDialog *remove_android_build_template = nullptr;

	PopupMenu *vcs_actions_menu = nullptr;
	EditorFileDialog *file = nullptr;
	ExportTemplateManager *export_template_manager = nullptr;
	EditorFeatureProfileManager *feature_profile_manager = nullptr;
	EditorBuildProfileManager *build_profile_manager = nullptr;
	EditorFileDialog *file_templates = nullptr;
	EditorFileDialog *file_export_lib = nullptr;
	EditorFileDialog *file_script = nullptr;
	EditorFileDialog *file_android_build_source = nullptr;
	CheckBox *file_export_lib_merge = nullptr;
	CheckBox *file_export_lib_apply_xforms = nullptr;
	String current_path;
	MenuButton *update_spinner = nullptr;

	HBoxContainer *main_editor_button_hb = nullptr;
	Vector<Button *> main_editor_buttons;
	Vector<EditorPlugin *> editor_table;

	AudioStreamPreviewGenerator *audio_preview_gen = nullptr;
	ProgressDialog *progress_dialog = nullptr;
	BackgroundProgress *progress_hb = nullptr;

	DependencyErrorDialog *dependency_error = nullptr;
	HashMap<String, HashSet<String>> dependency_errors;
	DependencyEditor *dependency_fixer = nullptr;
	OrphanResourcesDialog *orphan_resources = nullptr;
	ConfirmationDialog *open_imported = nullptr;
	Button *new_inherited_button = nullptr;
	String open_import_request;

	Vector<Control *> floating_docks;

	Button *dock_float = nullptr;
	Button *dock_tab_move_left = nullptr;
	Button *dock_tab_move_right = nullptr;
	Control *dock_select = nullptr;
	PopupPanel *dock_select_popup = nullptr;
	Rect2 dock_select_rect[DOCK_SLOT_MAX];
	TabContainer *dock_slot[DOCK_SLOT_MAX];
	Timer *dock_drag_timer = nullptr;
	bool docks_visible = true;
	int dock_popup_selected_idx = -1;
	int dock_select_rect_over_idx = -1;

	PanelContainer *tabbar_panel = nullptr;
	HBoxContainer *tabbar_container = nullptr;
	Button *distraction_free = nullptr;
	Button *scene_tab_add = nullptr;
	Control *scene_tab_add_ph = nullptr;

	Vector<BottomPanelItem> bottom_panel_items;
	PanelContainer *bottom_panel = nullptr;
	HBoxContainer *bottom_panel_hb = nullptr;
	HBoxContainer *bottom_panel_hb_editors = nullptr;
	VBoxContainer *bottom_panel_vb = nullptr;
	EditorToaster *editor_toaster = nullptr;
	LinkButton *version_btn = nullptr;
	Button *bottom_panel_raise = nullptr;

	Tree *disk_changed_list = nullptr;
	ConfirmationDialog *disk_changed = nullptr;

	bool scene_distraction_free = false;
	bool script_distraction_free = false;

	bool _playing_edited = false;
	bool changing_scene = false;
	bool cmdline_export_mode = false;
	bool convert_old = false;
	bool immediate_dialog_confirmed = false;
	bool opening_prev = false;
	bool restoring_scenes = false;
	bool settings_changed = true; // Make it update settings on first frame.
	bool unsaved_cache = true;
	bool waiting_for_first_scan = true;

	int current_menu_option = 0;

	SubViewport *scene_root = nullptr; // Root of the scene being edited.
	Object *current = nullptr;

	Ref<Resource> saving_resource;

	uint64_t update_spinner_step_msec = 0;
	uint64_t update_spinner_step_frame = 0;
	int update_spinner_step = 0;

	String _tmp_import_path;
	String external_file;
	String open_navigate;

	String run_custom_filename;
	String run_current_filename;

	DynamicFontImportSettings *fontdata_import_settings = nullptr;
	SceneImportSettings *scene_import_settings = nullptr;
	AudioStreamImportSettings *audio_stream_import_settings = nullptr;

	String import_reload_fn;

	HashSet<String> textfile_extensions;
	HashSet<FileDialog *> file_dialogs;
	HashSet<EditorFileDialog *> editor_file_dialogs;

	Vector<Ref<EditorResourceConversionPlugin>> resource_conversion_plugins;
	PrintHandlerList print_handler;

	HashMap<String, Ref<Texture2D>> icon_type_cache;
	HashMap<Ref<Script>, Ref<Texture>> script_icon_cache;

	static EditorBuildCallback build_callbacks[MAX_BUILD_CALLBACKS];
	static EditorPluginInitializeCallback plugin_init_callbacks[MAX_INIT_CALLBACKS];
	static int build_callback_count;
	static int plugin_init_callback_count;
	static Vector<EditorNodeInitCallback> _init_callbacks;

	static void _dependency_error_report(void *ud, const String &p_path, const String &p_dep, const String &p_type) {
		EditorNode *en = static_cast<EditorNode *>(ud);
		if (!en->dependency_errors.has(p_path)) {
			en->dependency_errors[p_path] = HashSet<String>();
		}
		en->dependency_errors[p_path].insert(p_dep + "::" + p_type);
	}

	static Ref<Texture2D> _file_dialog_get_icon(const String &p_path);
	static void _file_dialog_register(FileDialog *p_dialog);
	static void _file_dialog_unregister(FileDialog *p_dialog);
	static void _editor_file_dialog_register(EditorFileDialog *p_dialog);
	static void _editor_file_dialog_unregister(EditorFileDialog *p_dialog);

	static void _load_error_notify(void *p_ud, const String &p_text);
	static void _file_access_close_error_notify(const String &p_str);

	static void _print_handler(void *p_this, const String &p_string, bool p_error, bool p_rich);
	static void _resource_saved(Ref<Resource> p_resource, const String &p_path);
	static void _resource_loaded(Ref<Resource> p_resource, const String &p_path);

	void _build_icon_type_cache();

	void _dialog_action(String p_file);

	void _edit_current(bool p_skip_foreign = false);
	void _dialog_display_save_error(String p_file, Error p_error);
	void _dialog_display_load_error(String p_file, Error p_error);

	void _menu_option(int p_option);
	void _menu_confirm_current();
	void _menu_option_confirm(int p_option, bool p_confirmed);

	void _android_build_source_selected(const String &p_file);

	void _request_screenshot();
	void _screenshot(bool p_use_utc = false);
	void _save_screenshot(NodePath p_path);

	void _tool_menu_option(int p_idx);
	void _export_as_menu_option(int p_idx);
	void _update_file_menu_opened();
	void _update_file_menu_closed();

	void _remove_plugin_from_enabled(const String &p_name);

	void _fs_changed();
	void _resources_reimported(const Vector<String> &p_resources);
	void _sources_changed(bool p_exist);

	void _node_renamed();
	void _editor_select_next();
	void _editor_select_prev();
	void _set_scene_metadata(const String &p_file, int p_idx = -1);
	void _get_scene_metadata(const String &p_file);
	void _update_title();
	void _update_scene_tabs();
	void _version_control_menu_option(int p_idx);
	void _close_messages();
	void _show_messages();
	void _vp_resized();
	void _titlebar_resized();
	void _version_button_pressed();

	int _save_external_resources();

	bool _validate_scene_recursive(const String &p_filename, Node *p_node);
	void _save_scene(String p_file, int idx = -1);
	void _save_all_scenes();
	int _next_unsaved_scene(bool p_valid_filename, int p_start = 0);
	void _discard_changes(const String &p_str = String());

	void _inherit_request(String p_file);
	void _instantiate_request(const Vector<String> &p_files);

	void _display_top_editors(bool p_display);
	void _set_top_editors(Vector<EditorPlugin *> p_editor_plugins_over);
	void _set_editing_top_editors(Object *p_current_object);

	void _quick_opened();
	void _quick_run();
	void _open_command_palette();

	void _write_movie_toggled(bool p_enabled);

	void _run(bool p_current = false, const String &p_custom = "");
	void _run_native(const Ref<EditorExportPreset> &p_preset);
	void _reset_play_buttons();

	void _add_to_recent_scenes(const String &p_scene);
	void _update_recent_scenes();
	void _open_recent_scene(int p_idx);
	void _global_menu_scene(const Variant &p_tag);
	void _global_menu_new_window(const Variant &p_tag);
	void _dropped_files(const Vector<String> &p_files);
	void _add_dropped_files_recursive(const Vector<String> &p_files, String to_path);

	void _update_from_settings();

	void _renderer_selected(int);
	void _update_renderer_color();

	void _exit_editor(int p_exit_code);

	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

	bool has_main_screen() const { return true; }

	void _remove_edited_scene(bool p_change_tab = true);
	void _remove_scene(int index, bool p_change_tab = true);
	bool _find_and_save_resource(Ref<Resource> p_res, HashMap<Ref<Resource>, bool> &processed, int32_t flags);
	bool _find_and_save_edited_subresources(Object *obj, HashMap<Ref<Resource>, bool> &processed, int32_t flags);
	void _save_edited_subresources(Node *scene, HashMap<Ref<Resource>, bool> &processed, int32_t flags);
	void _mark_unsaved_scenes();

	void _find_node_types(Node *p_node, int &count_2d, int &count_3d);
	void _save_scene_with_preview(String p_file, int p_idx = -1);

	bool _find_scene_in_use(Node *p_node, const String &p_path) const;

	void _update_dock_containers();

	void _dock_select_input(const Ref<InputEvent> &p_input);
	void _dock_move_left();
	void _dock_move_right();
	void _dock_select_draw();
	void _dock_pre_popup(int p_which);
	void _dock_split_dragged(int ofs);
	void _dock_popup_exit();
	void _dock_floating_close_request(Control *p_control);
	void _dock_make_float();
	void _scene_tab_changed(int p_tab);
	void _scene_tab_closed(int p_tab, int option = SCENE_TAB_CLOSE);
	void _scene_tab_hovered(int p_tab);
	void _scene_tab_exit();
	void _scene_tab_input(const Ref<InputEvent> &p_input);
	void _reposition_active_tab(int idx_to);
	void _thumbnail_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_udata);
	void _scene_tab_script_edited(int p_tab);

	Dictionary _get_main_scene_state();
	void _set_main_scene_state(Dictionary p_state, Node *p_for_scene);

	int _get_current_main_editor();

	void _save_docks();
	void _load_docks();
	void _save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section);
	void _load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section);
	void _update_dock_slots_visibility(bool p_keep_selected_tabs = false);
	void _dock_tab_changed(int p_tab);

	void _save_open_scenes_to_config(Ref<ConfigFile> p_layout, const String &p_section);
	void _load_open_scenes_from_config(Ref<ConfigFile> p_layout, const String &p_section);

	void _update_layouts_menu();
	void _layout_menu_option(int p_id);

	void _update_addon_config();

	void _toggle_distraction_free_mode();

	void _inherit_imported(const String &p_action);
	void _open_imported();

	void _save_default_environment();
	void _update_update_spinner();

	void _resources_changed(const Vector<String> &p_resources);
	void _scan_external_changes();
	void _reload_modified_scenes();
	void _reload_project_settings();
	void _resave_scenes(String p_str);

	void _feature_profile_changed();
	bool _is_class_editor_disabled_by_feature_profile(const StringName &p_class);
	Ref<ImageTexture> _load_custom_class_icon(const String &p_path) const;

	void _pick_main_scene_custom_action(const String &p_custom_action_name);

	void _immediate_dialog_confirmed();
	void _select_default_main_screen_plugin();

	void _bottom_panel_switch(bool p_enable, int p_idx);
	void _bottom_panel_raise_toggled(bool);

	void _begin_first_scan();
	bool use_startup_benchmark = false;
	String startup_benchmark_file;

protected:
	friend class FileSystemDock;

	static void _bind_methods();

	void _notification(int p_what);

	int get_current_tab();
	void set_current_tab(int p_tab);

public:
	// Public for use with callable_mp.
	void _on_plugin_ready(Object *p_script, const String &p_activate_name);

	void editor_select(int p_which);
	void set_visible_editor(EditorTable p_table) { editor_select(p_table); }

	bool call_build();

	static EditorNode *get_singleton() { return singleton; }

	static EditorLog *get_log() { return singleton->log; }
	static EditorData &get_editor_data() { return singleton->editor_data; }
	static EditorFolding &get_editor_folding() { return singleton->editor_folding; }
	static Ref<EditorUndoRedoManager> &get_undo_redo();

	static HBoxContainer *get_menu_hb() { return singleton->menu_hb; }
	static VSplitContainer *get_top_split() { return singleton->top_split; }

	static String adjust_scene_name_casing(const String &root_name);

	static bool has_unsaved_changes() { return singleton->unsaved_cache; }
	static void disambiguate_filenames(const Vector<String> p_full_paths, Vector<String> &r_filenames);
	static void add_io_error(const String &p_error);

	static void progress_add_task(const String &p_task, const String &p_label, int p_steps, bool p_can_cancel = false);
	static bool progress_task_step(const String &p_task, const String &p_state, int p_step = -1, bool p_force_refresh = true);
	static void progress_end_task(const String &p_task);

	static void progress_add_task_bg(const String &p_task, const String &p_label, int p_steps);
	static void progress_task_step_bg(const String &p_task, int p_step = -1);
	static void progress_end_task_bg(const String &p_task);

	static void add_editor_plugin(EditorPlugin *p_editor, bool p_config_changed = false);
	static void remove_editor_plugin(EditorPlugin *p_editor, bool p_config_changed = false);

	static void add_plugin_init_callback(EditorPluginInitializeCallback p_callback);
	static void add_init_callback(EditorNodeInitCallback p_callback) { _init_callbacks.push_back(p_callback); }
	static void add_build_callback(EditorBuildCallback p_callback);

	static bool immediate_confirmation_dialog(const String &p_text, const String &p_ok_text = TTR("Ok"), const String &p_cancel_text = TTR("Cancel"));

	static void cleanup();

	EditorPlugin *get_editor_plugin_screen() { return editor_plugin_screen; }
	EditorPluginList *get_editor_plugins_force_input_forwarding() { return editor_plugins_force_input_forwarding; }
	EditorPluginList *get_editor_plugins_force_over() { return editor_plugins_force_over; }
	EditorPluginList *get_editor_plugins_over() { return editor_plugins_over; }
	EditorSelection *get_editor_selection() { return editor_selection; }
	EditorSelectionHistory *get_editor_selection_history() { return &editor_history; }

	ProjectSettingsEditor *get_project_settings() { return project_settings_editor; }

	void new_inherited_scene() { _menu_option_confirm(FILE_NEW_INHERITED_SCENE, false); }

	void set_docks_visible(bool p_show);
	bool get_docks_visible() const;

	void set_distraction_free_mode(bool p_enter);
	bool is_distraction_free_mode_enabled() const;

	void add_control_to_dock(DockSlot p_slot, Control *p_control);
	void remove_control_from_dock(Control *p_control);

	void set_addon_plugin_enabled(const String &p_addon, bool p_enabled, bool p_config_changed = false);
	bool is_addon_plugin_enabled(const String &p_addon) const;

	void set_movie_maker_enabled(bool p_enabled);
	bool is_movie_maker_enabled() const;

	void edit_node(Node *p_node);
	void edit_resource(const Ref<Resource> &p_resource);

	void save_resource_in_path(const Ref<Resource> &p_resource, const String &p_path);
	void save_resource(const Ref<Resource> &p_resource);
	void save_resource_as(const Ref<Resource> &p_resource, const String &p_at_path = String());

	void show_about() { _menu_option_confirm(HELP_ABOUT, false); }

	void push_item(Object *p_object, const String &p_property = "", bool p_inspector_only = false);
	void edit_item(Object *p_object);
	void edit_item_resource(Ref<Resource> p_resource);
	bool item_has_editor(Object *p_object);
	void hide_top_editors();

	void select_editor_by_name(const String &p_name);

	void open_request(const String &p_path);
	void edit_foreign_resource(Ref<Resource> p_resource);

	bool is_resource_read_only(Ref<Resource> p_resource, bool p_foreign_resources_are_writable = false);

	bool is_changing_scene() const;

	VBoxContainer *get_main_screen_control();
	SubViewport *get_scene_root() { return scene_root; } // Root of the scene being edited.

	void set_edited_scene(Node *p_scene);
	Node *get_edited_scene() { return editor_data.get_edited_scene_root(); }

	void fix_dependencies(const String &p_for_file);
	int new_scene();
	Error load_scene(const String &p_scene, bool p_ignore_broken_deps = false, bool p_set_inherited = false, bool p_clear_errors = true, bool p_force_open_imported = false, bool p_silent_change_tab = false);
	Error load_resource(const String &p_resource, bool p_ignore_broken_deps = false);

	bool is_scene_open(const String &p_path);

	void set_current_scene(int p_idx);

	void setup_color_picker(ColorPicker *picker);

	void request_instantiate_scene(const String &p_path);
	void request_instantiate_scenes(const Vector<String> &p_files);

	void set_convert_old_scene(bool p_old) { convert_old = p_old; }

	void notify_all_debug_sessions_exited();

	OS::ProcessID has_child_process(OS::ProcessID p_pid) const { return editor_run.has_child_process(p_pid); }
	void stop_child_process(OS::ProcessID p_pid);

	Ref<Theme> get_editor_theme() const { return theme; }
	Ref<Script> get_object_custom_type_base(const Object *p_object) const;
	StringName get_object_custom_type_name(const Object *p_object) const;
	Ref<Texture2D> get_object_icon(const Object *p_object, const String &p_fallback = "Object");
	Ref<Texture2D> get_class_icon(const String &p_class, const String &p_fallback = "Object") const;

	bool is_object_of_custom_type(const Object *p_object, const StringName &p_class);

	void show_accept(const String &p_text, const String &p_title);
	void show_save_accept(const String &p_text, const String &p_title);
	void show_warning(const String &p_text, const String &p_title = TTR("Warning!"));

	void _copy_warning(const String &p_str);

	void set_use_startup_benchmark(bool p_use_startup_benchmark, const String &p_startup_benchmark_file);
	Error export_preset(const String &p_preset, const String &p_path, bool p_debug, bool p_pack_only);

	Control *get_gui_base() { return gui_base; }
	Control *get_theme_base() { return gui_base->get_parent_control(); }

	void save_scene_to_path(String p_file, bool p_with_preview = true) {
		if (p_with_preview) {
			_save_scene_with_preview(p_file);
		} else {
			_save_scene(p_file);
		}
	}

	bool is_scene_in_use(const String &p_path);

	void save_layout();

	void open_export_template_manager();

	void reload_scene(const String &p_path);

	bool is_exiting() const { return exiting; }

	Button *get_pause_button() { return pause_button; }

	Button *add_bottom_panel_item(String p_text, Control *p_item);
	void make_bottom_panel_item_visible(Control *p_item);
	void raise_bottom_panel_item(Control *p_item);
	void hide_bottom_panel();
	void remove_bottom_panel_item(Control *p_item);

	Variant drag_resource(const Ref<Resource> &p_res, Control *p_from);
	Variant drag_files_and_dirs(const Vector<String> &p_paths, Control *p_from);

	void add_tool_menu_item(const String &p_name, const Callable &p_callback);
	void add_tool_submenu_item(const String &p_name, PopupMenu *p_submenu);
	void remove_tool_menu_item(const String &p_name);

	PopupMenu *get_export_as_menu();

	void save_all_scenes();
	void save_scene_list(Vector<String> p_scene_filenames);
	void restart_editor();

	void notify_settings_changed();

	void dim_editor(bool p_dimming);
	bool is_editor_dimmed() const;

	void edit_current() { _edit_current(); };

	bool has_scenes_in_session();

	int execute_and_show_output(const String &p_title, const String &p_path, const List<String> &p_arguments, bool p_close_on_ok = true, bool p_close_on_errors = false);

	EditorNode();
	~EditorNode();

	void add_resource_conversion_plugin(const Ref<EditorResourceConversionPlugin> &p_plugin);
	void remove_resource_conversion_plugin(const Ref<EditorResourceConversionPlugin> &p_plugin);
	Vector<Ref<EditorResourceConversionPlugin>> find_resource_conversion_plugin(const Ref<Resource> &p_for_resource);

	bool ensure_main_scene(bool p_from_native);

	Error run_play_native(int p_idx, int p_platform);
	void run_play();
	void run_play_current();
	void run_play_custom(const String &p_custom);
	void run_stop();
	bool is_run_playing() const;
	String get_run_playing_scene() const;
};

struct EditorProgress {
	String task;
	bool step(const String &p_state, int p_step = -1, bool p_force_refresh = true) { return EditorNode::progress_task_step(task, p_state, p_step, p_force_refresh); }
	EditorProgress(const String &p_task, const String &p_label, int p_amount, bool p_can_cancel = false) {
		EditorNode::progress_add_task(p_task, p_label, p_amount, p_can_cancel);
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
	bool forward_gui_input(const Ref<InputEvent> &p_event);
	void forward_canvas_draw_over_viewport(Control *p_overlay);
	void forward_canvas_force_draw_over_viewport(Control *p_overlay);
	EditorPlugin::AfterGUIInput forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event, bool serve_when_force_input_enabled);
	void forward_3d_draw_over_viewport(Control *p_overlay);
	void forward_3d_force_draw_over_viewport(Control *p_overlay);
	void add_plugin(EditorPlugin *p_plugin);
	void remove_plugin(EditorPlugin *p_plugin);
	void clear();
	bool is_empty();

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

#endif // EDITOR_NODE_H
