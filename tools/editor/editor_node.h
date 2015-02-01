/*************************************************************************/
/*  editor_node.h                                                        */
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
#ifndef EDITOR_NODE_H
#define EDITOR_NODE_H

#include "scene/gui/control.h"
#include "scene/gui/panel.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tree.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/separator.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/split_container.h"
#include "scene/gui/center_container.h"
#include "scene/gui/texture_progress.h"
#include "tools/editor/scenes_dock.h"
#include "tools/editor/scene_tree_editor.h"
#include "tools/editor/property_editor.h"
#include "tools/editor/create_dialog.h"
#include "tools/editor/call_dialog.h"
#include "tools/editor/reparent_dialog.h"
#include "tools/editor/connections_dialog.h"
#include "tools/editor/settings_config_dialog.h"
#include "tools/editor/groups_editor.h"
#include "tools/editor/editor_data.h"
#include "tools/editor/editor_path.h"
#include "tools/editor/editor_run.h"

#include "tools/editor/pane_drag.h"
#include "tools/editor/animation_editor.h"
#include "tools/editor/script_create_dialog.h"
#include "tools/editor/run_settings_dialog.h"
#include "tools/editor/project_settings.h"
#include "tools/editor/project_export.h"
#include "tools/editor/editor_log.h"
#include "tools/editor/scene_tree_dock.h"
#include "tools/editor/resources_dock.h"
#include "tools/editor/optimized_save_dialog.h"
#include "tools/editor/editor_run_script.h"

#include "tools/editor/editor_run_native.h"
#include "scene/gui/tabs.h"
#include "tools/editor/quick_open.h"
#include "tools/editor/project_export.h"
#include "tools/editor/editor_sub_scene.h"
#include "editor_import_export.h"
#include "editor_reimport_dialog.h"
#include "import_settings.h"
#include "tools/editor/editor_plugin.h"

#include "fileserver/editor_file_server.h"




#include "progress_dialog.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/





typedef void (*EditorNodeInitCallback)();

class EditorNode : public Node {

	OBJ_TYPE( EditorNode, Node );
	
	enum {
		
		HISTORY_SIZE=64	
	};
	enum MenuOptions {
	
		FILE_NEW_SCENE,
		FILE_OPEN_SCENE,
		FILE_SAVE_SCENE,
		FILE_SAVE_AS_SCENE,
		FILE_SAVE_BEFORE_RUN,
		FILE_SAVE_AND_RUN,
		FILE_IMPORT_SUBSCENE,
		FILE_EXPORT_PROJECT,
		FILE_EXPORT_MESH_LIBRARY,
		FILE_EXPORT_TILESET,
		FILE_SAVE_OPTIMIZED,
		FILE_SAVE_SUBSCENE,
		FILE_DUMP_STRINGS,
		FILE_OPEN_RECENT,
		FILE_OPEN_OLD_SCENE,
		FILE_QUICK_OPEN_SCENE,
		FILE_QUICK_OPEN_SCRIPT,
		FILE_RUN_SCRIPT,
		FILE_OPEN_PREV,
		FILE_QUIT,
		FILE_EXTERNAL_OPEN_SCENE,
		EDIT_UNDO,
		EDIT_REDO,
		RESOURCE_NEW,
		RESOURCE_LOAD,
		RESOURCE_SAVE,
		RESOURCE_SAVE_AS,
		RESOURCE_UNREF,
		RESOURCE_COPY,
		OBJECT_COPY_PARAMS,
		OBJECT_PASTE_PARAMS,
		OBJECT_UNIQUE_RESOURCES,
		OBJECT_CALL_METHOD,
		OBJECT_REQUEST_HELP,
		RUN_PLAY,
		RUN_PAUSE,
		RUN_STOP,
		RUN_PLAY_SCENE,
		RUN_PLAY_CUSTOM_SCENE,
		RUN_SCENE_SETTINGS,
		RUN_SETTINGS,
		RUN_PROJECT_MANAGER,
		RUN_FILE_SERVER,
		RUN_DEPLOY_DUMB_CLIENTS,
		SETTINGS_UPDATE_ALWAYS,
		SETTINGS_UPDATE_CHANGES,
		SETTINGS_IMPORT,
		SETTINGS_EXPORT_PREFERENCES,
		SETTINGS_PREFERENCES,
		SETTINGS_OPTIMIZED_PRESETS,
		SETTINGS_SHOW_ANIMATION,
		SETTINGS_LOAD_EXPORT_TEMPLATES,
		SETTINGS_HELP,
		SETTINGS_ABOUT,
		SOURCES_REIMPORT,
		DEPENDENCY_LOAD_CHANGED_IMAGES,
		DEPENDENCY_UPDATE_IMPORTED,

		IMPORT_PLUGIN_BASE=100,

		OBJECT_METHOD_BASE=500
	};




	Node *edited_scene; //scene being edited
	Viewport *scene_root; //root of the scene being edited

	Ref<ResourceImportMetadata> scene_import_metadata;

	Control* scene_root_parent;
	Control *gui_base;
	VBoxContainer *main_vbox;
	HSplitContainer *main_split;
	VSplitContainer *left_split;
	int old_split_ofs;
	VSplitContainer *top_split;
	HBoxContainer *bottom_hb;
	Control *vp_base;
	PaneDrag *pd;
	//PaneDrag *pd_anim;
	TextureButton *anim_close;
	Panel *menu_panel;

	HSplitContainer *editor_hsplit;
	VSplitContainer *editor_vsplit;
	HBoxContainer *menu_hb;
	Control *viewport;
	MenuButton *file_menu;
	MenuButton *import_menu;
	ToolButton *export_button;
	ToolButton *prev_scene;
	MenuButton *object_menu;
	MenuButton *settings_menu;
	ToolButton *play_button;
	MenuButton *native_play_button;
	ToolButton *pause_button;
	ToolButton *stop_button;
	ToolButton *run_settings_button;
	ToolButton *animation_menu;
	ToolButton *play_scene_button;
	ToolButton *play_custom_scene_button;
	TextureProgress *audio_vu;
	MenuButton *fileserver_menu;

	TextEdit *load_errors;
	AcceptDialog *load_error_dialog;

	Control *scene_root_base;
	Ref<Theme> theme;

	PopupMenu *recent_scenes;
	Button *property_back;
	Button *property_forward;
	SceneTreeDock *scene_tree_dock;
	ResourcesDock *resources_dock;
	PropertyEditor *property_editor;
	ScenesDock *scenes_dock;
	EditorRunNative *run_native;

	CallDialog *call_dialog;
	ConfirmationDialog *confirmation;
	ConfirmationDialog *import_confirmation;
	ConfirmationDialog *open_recent_confirmation;
	AcceptDialog *accept;
	AcceptDialog *about;

	//OptimizedPresetsDialog *optimized_presets;
	EditorSettingsDialog *settings_config_dialog;
	RunSettingsDialog *run_settings_dialog;
	ProjectSettings *project_settings;
	FileDialog *file;
	FileDialog *file_templates;
	FileDialog *file_export;
	FileDialog *file_export_lib;
	FileDialog *file_script;
	CheckButton *file_export_check;
	CheckButton *file_export_lib_merge;
	LineEdit *file_export_password;
	String current_path;
	MenuButton *update_menu;
	ToolButton *sources_button;
	TabContainer *prop_pallete;
	TabContainer *top_pallete;
	String defer_load_scene;
	String defer_translatable;
	String defer_optimize;
	String defer_optimize_preset;
	String defer_export;
	String defer_export_platform;
	bool defer_export_debug;
	Node *_last_instanced_scene;
	PanelContainer *animation_panel;
	HBoxContainer *animation_panel_hb;
	VBoxContainer *animation_vb;
	EditorPath *editor_path;
	AnimationKeyEditor *animation_editor;
	EditorLog *log;
	CenterContainer *tabs_center;
	EditorQuickOpen *quick_open;
	EditorQuickOpen *quick_run;
	Tabs *main_editor_tabs;
	Vector<EditorPlugin*> editor_table;

	EditorReImportDialog *reimport_dialog;
	ImportSettingsDialog *import_settings;

	ProgressDialog *progress_dialog;
	BackgroundProgress *progress_hb;

	String _tmp_import_path;

	EditorImportExport *editor_import_export;

	Object *current;

	bool _playing_edited;
	bool reference_resource_mem;
	bool save_external_resources_mem;
	uint64_t saved_version;
	bool unsaved_cache;
	String open_navigate;

	uint32_t circle_step_msec;
	uint64_t circle_step_frame;
	int circle_step;

	Vector<EditorPlugin*> editor_plugins;
	EditorPlugin *editor_plugin_screen;
	EditorPlugin *editor_plugin_over;

	EditorHistory editor_history;
	EditorData editor_data;
	EditorRun editor_run;
	EditorSelection *editor_selection;
	ProjectExport *project_export;
	ProjectExportDialog *project_export_settings;

	EditorFileServer *file_server;

	String external_file;
	List<String> previous_scenes;
	bool opening_prev;
	
	void _dialog_action(String p_file);


	void _edit_current();
	void _dialog_display_file_error(String p_file,Error p_error);
	
	int current_option;
	//void _animation_visibility_toggle();
	void _resource_selected(const RES& p_res,const String& p_property="");
	void _menu_option(int p_option);
	void _menu_confirm_current();
	void _menu_option_confirm(int p_option,bool p_confirmed);

	void _property_editor_forward();
	void _property_editor_back();

	
	void _fs_changed();
	void _sources_changed(bool p_exist);
	void _imported(Node *p_node);

	void _node_renamed();
	void _editor_select(int p_which);
	void _set_scene_metadata();
	void _get_scene_metadata();
	void _update_title();
	void _close_messages();
	void _show_messages();
	void _vp_resized();

	void _save_scene(String p_file);


	void _instance_request(const String& p_path);

	void _property_keyed(const String& p_keyed,const Variant& p_value);
	void _transform_keyed(Object *sp,const String& p_sub,const Transform& p_key);

	void _update_keying();
	void _hide_top_editors();
	void _quick_opened(const String& p_resource);
	void _quick_run(const String& p_resource);

	void _run(bool p_current=false, const String &p_custom="");

	void _save_optimized();
	void _import_action(const String& p_action);
	void _import(const String &p_file);
	void _add_to_recent_scenes(const String& p_scene);
	void _update_recent_scenes();
	void _open_recent_scene(int p_idx);
	void _open_recent_scene_confirm();
	String _recent_scene;

	bool convert_old;

	void _unhandled_input(const InputEvent& p_event);

	static void _load_error_notify(void* p_ud,const String& p_text);

	bool has_main_screen() const { return true; }
	void _fetch_translatable_strings(const Object *p_object,Set<StringName>& strings);

	bool _find_editing_changed_scene(Node *p_from);

	String import_reload_fn;

	Set<FileDialog*> file_dialogs;
	Map<String,Ref<Texture> > icon_type_cache;

	static Ref<Texture> _file_dialog_get_icon(const String& p_path);
	static void _file_dialog_register(FileDialog *p_dialog);
	static void _file_dialog_unregister(FileDialog *p_dialog);


	void _cleanup_scene();

	bool _find_and_save_edited_subresources(Object *obj,Set<RES>& processed,int32_t flags);
	void _save_edited_subresources(Node* scene,Set<RES>& processed,int32_t flags);


	struct ExportDefer {
		String platform;
		String path;
		bool debug;
		String password;

	} export_defer;

	static EditorNode *singleton;

	static Vector<EditorNodeInitCallback> _init_callbacks;

	bool _find_scene_in_use(Node* p_node,const String& p_path) const;


protected:
	void _notification(int p_what);
	static void _bind_methods();		
public:

	static EditorNode* get_singleton() { return singleton; }


	EditorPlugin *get_editor_plugin_screen() { return editor_plugin_screen; }
	EditorPlugin *get_editor_plugin_over() { return editor_plugin_over; }
	PropertyEditor *get_property_editor() { return property_editor; }

	static void add_editor_plugin(EditorPlugin *p_editor);
	static void remove_editor_plugin(EditorPlugin *p_editor);


	void edit_node(Node *p_node);
	void edit_resource(const Ref<Resource>& p_resource);
	void open_resource(const String& p_type="");
	void save_resource(const Ref<Resource>& p_resource);
	void save_resource_as(const Ref<Resource>& p_resource);

	static bool has_unsaved_changes() { return singleton->unsaved_cache; }

	static HBoxContainer *get_menu_hb() { return singleton->menu_hb; }

	void push_item(Object *p_object,const String& p_property="");

	void open_request(const String& p_path);

	void set_edited_scene(Node *p_scene);


	static EditorLog *get_log() { return singleton->log; }
	Control* get_viewport();
	AnimationKeyEditor *get_animation_editor() const { return animation_editor; }
	Control *get_animation_panel() { return animation_vb; }
	HBoxContainer *get_animation_panel_hb() { return animation_panel_hb; }

	void animation_editor_make_visible(bool p_visible);
	void hide_animation_player_editors();
	void animation_panel_make_visible(bool p_visible);

	Node *get_edited_scene() { return edited_scene; }

	Viewport *get_scene_root() { return scene_root; } //root of the scene being edited
	Error save_optimized_copy(const String& p_scene,const String& p_preset);

	void clear_scene() { _cleanup_scene(); }
	Error load_scene(const String& p_scene);
	Error load_resource(const String& p_scene);

	static EditorData& get_editor_data() { return singleton->editor_data; }

	static VSplitContainer *get_top_split() { return singleton->top_split; }

	Node* request_instance_scene(const String &p_path);
	ScenesDock *get_scenes_dock();
	static UndoRedo* get_undo_redo() { return &singleton->editor_data.get_undo_redo(); }

	EditorSelection *get_editor_selection() { return editor_selection; }

	Error save_translatable_strings(const String& p_to_file);

	void set_convert_old_scene(bool p_old) { convert_old=p_old; }

	void notify_child_process_exited();

	void stop_child_process();

	Ref<Theme> get_editor_theme() const { return theme; }


	Error export_platform(const String& p_platform, const String& p_path, bool p_debug,const String& p_password,bool p_quit_after=false);

	static void register_editor_types();
	static void unregister_editor_types();

	Control *get_gui_base() { return gui_base; }

	static void add_io_error(const String& p_error);

	static void progress_add_task(const String& p_task,const String& p_label, int p_steps);
	static void progress_task_step(const String& p_task,const String& p_state, int p_step=-1);
	static void progress_end_task(const String& p_task);

	static void progress_add_task_bg(const String& p_task,const String& p_label, int p_steps);
	static void progress_task_step_bg(const String& p_task,int p_step=-1);
	static void progress_end_task_bg(const String& p_task);

	void save_scene(String p_file) { _save_scene(p_file); }

	bool is_scene_in_use(const String& p_path);

	void scan_import_changes();
	EditorNode();	
	~EditorNode();
	void get_singleton(const char* arg1, bool arg2);

	static void add_init_callback(EditorNodeInitCallback p_callback) { _init_callbacks.push_back(p_callback); }

};


struct EditorProgress {

	String task;
	void step(const String& p_state, int p_step=-1) { EditorNode::progress_task_step(task,p_state,p_step); }
	EditorProgress(const String& p_task,const String& p_label,int p_amount) { EditorNode::progress_add_task(p_task,p_label,p_amount); task=p_task; }
	~EditorProgress() { EditorNode::progress_end_task(task); }
};

struct EditorProgressBG {

	String task;
	void step(int p_step=-1) { EditorNode::progress_task_step_bg(task,p_step); }
	EditorProgressBG(const String& p_task,const String& p_label,int p_amount) { EditorNode::progress_add_task_bg(p_task,p_label,p_amount); task=p_task; }
	~EditorProgressBG() { EditorNode::progress_end_task_bg(task); }
};

#endif
