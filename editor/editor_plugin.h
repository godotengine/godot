/*************************************************************************/
/*  editor_plugin.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_PLUGIN_H
#define EDITOR_PLUGIN_H

#include "core/io/config_file.h"
#include "core/object/undo_redo.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_inspector.h"
#include "editor/editor_translation_parser.h"
#include "editor/import/editor_import_plugin.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/script_create_dialog.h"
#include "scene/3d/camera_3d.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"
class EditorNode;
class Node3D;
class Camera3D;
class EditorCommandPalette;
class EditorSelection;
class EditorExport;
class EditorSettings;
class EditorImportPlugin;
class EditorExportPlugin;
class EditorNode3DGizmoPlugin;
class EditorResourcePreview;
class EditorFileSystem;
class EditorToolAddons;
class EditorPaths;
class FileSystemDock;
class ScriptEditor;

class EditorInterface : public Node {
	GDCLASS(EditorInterface, Node);

protected:
	static void _bind_methods();
	static EditorInterface *singleton;

	Array _make_mesh_previews(const Array &p_meshes, int p_preview_size);

public:
	static EditorInterface *get_singleton() { return singleton; }

	Control *get_editor_main_control();
	void edit_resource(const Ref<Resource> &p_resource);
	void edit_node(Node *p_node);
	void open_scene_from_path(const String &scene_path);
	void reload_scene_from_path(const String &scene_path);

	void play_main_scene();
	void play_current_scene();
	void play_custom_scene(const String &scene_path);
	void stop_playing_scene();
	bool is_playing_scene() const;
	String get_playing_scene() const;

	Node *get_edited_scene_root();
	Array get_open_scenes() const;
	ScriptEditor *get_script_editor();

	EditorCommandPalette *get_command_palette() const;

	void select_file(const String &p_file);
	String get_selected_path() const;
	String get_current_path() const;

	void inspect_object(Object *p_obj, const String &p_for_property = String(), bool p_inspector_only = false);

	EditorSelection *get_selection();
	//EditorImportExport *get_import_export();
	Ref<EditorSettings> get_editor_settings();
	EditorPaths *get_editor_paths();
	EditorResourcePreview *get_resource_previewer();
	EditorFileSystem *get_resource_file_system();

	FileSystemDock *get_file_system_dock();

	Control *get_base_control();
	float get_editor_scale() const;

	void set_plugin_enabled(const String &p_plugin, bool p_enabled);
	bool is_plugin_enabled(const String &p_plugin) const;

	EditorInspector *get_inspector() const;

	Error save_scene();
	void save_scene_as(const String &p_scene, bool p_with_preview = true);

	Vector<Ref<Texture2D>> make_mesh_previews(const Vector<Ref<Mesh>> &p_meshes, Vector<Transform3D> *p_transforms, int p_preview_size);

	void set_main_screen_editor(const String &p_name);
	void set_distraction_free_mode(bool p_enter);
	bool is_distraction_free_mode_enabled() const;

	EditorInterface();
};

class EditorPlugin : public Node {
	GDCLASS(EditorPlugin, Node);
	friend class EditorData;
	UndoRedo *undo_redo = nullptr;

	UndoRedo *_get_undo_redo() { return undo_redo; }

	bool input_event_forwarding_always_enabled = false;
	bool force_draw_over_forwarding_enabled = false;

	String last_main_screen_name;

	void _editor_project_settings_changed();

protected:
	void _notification(int p_what);

	static void _bind_methods();
	UndoRedo &get_undo_redo() { return *undo_redo; }

	void add_custom_type(const String &p_type, const String &p_base, const Ref<Script> &p_script, const Ref<Texture2D> &p_icon);
	void remove_custom_type(const String &p_type);

	GDVIRTUAL1R(bool, _forward_canvas_gui_input, Ref<InputEvent>)
	GDVIRTUAL1(_forward_canvas_draw_over_viewport, Control *)
	GDVIRTUAL1(_forward_canvas_force_draw_over_viewport, Control *)
	GDVIRTUAL2R(int, _forward_3d_gui_input, Camera3D *, Ref<InputEvent>)
	GDVIRTUAL1(_forward_3d_draw_over_viewport, Control *)
	GDVIRTUAL1(_forward_3d_force_draw_over_viewport, Control *)
	GDVIRTUAL0RC(String, _get_plugin_name)
	GDVIRTUAL0RC(Ref<Texture2D>, _get_plugin_icon)
	GDVIRTUAL0RC(bool, _has_main_screen)
	GDVIRTUAL1(_make_visible, bool)
	GDVIRTUAL1(_edit, Variant)
	GDVIRTUAL1RC(bool, _handles, Variant)
	GDVIRTUAL0RC(Dictionary, _get_state)
	GDVIRTUAL1(_set_state, Dictionary)
	GDVIRTUAL0(_clear)
	GDVIRTUAL0(_save_external_data)
	GDVIRTUAL0(_apply_changes)
	GDVIRTUAL0RC(Vector<String>, _get_breakpoints)
	GDVIRTUAL1(_set_window_layout, Ref<ConfigFile>)
	GDVIRTUAL1(_get_window_layout, Ref<ConfigFile>)
	GDVIRTUAL0R(bool, _build)
	GDVIRTUAL0(_enable_plugin)
	GDVIRTUAL0(_disable_plugin)

public:
	enum CustomControlContainer {
		CONTAINER_TOOLBAR,
		CONTAINER_SPATIAL_EDITOR_MENU,
		CONTAINER_SPATIAL_EDITOR_SIDE_LEFT,
		CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT,
		CONTAINER_SPATIAL_EDITOR_BOTTOM,
		CONTAINER_CANVAS_EDITOR_MENU,
		CONTAINER_CANVAS_EDITOR_SIDE_LEFT,
		CONTAINER_CANVAS_EDITOR_SIDE_RIGHT,
		CONTAINER_CANVAS_EDITOR_BOTTOM,
		CONTAINER_PROPERTY_EDITOR_BOTTOM,
		CONTAINER_PROJECT_SETTING_TAB_LEFT,
		CONTAINER_PROJECT_SETTING_TAB_RIGHT,
	};

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

	enum AfterGUIInput {
		AFTER_GUI_INPUT_PASS,
		AFTER_GUI_INPUT_STOP,
		AFTER_GUI_INPUT_DESELECT
	};

	//TODO: send a resource for editing to the editor node?

	void add_control_to_container(CustomControlContainer p_location, Control *p_control);
	void remove_control_from_container(CustomControlContainer p_location, Control *p_control);
	Button *add_control_to_bottom_panel(Control *p_control, const String &p_title);
	void add_control_to_dock(DockSlot p_slot, Control *p_control);
	void remove_control_from_docks(Control *p_control);
	void remove_control_from_bottom_panel(Control *p_control);

	void add_tool_menu_item(const String &p_name, const Callable &p_callable);
	void add_tool_submenu_item(const String &p_name, Object *p_submenu);
	void remove_tool_menu_item(const String &p_name);

	void set_input_event_forwarding_always_enabled();
	bool is_input_event_forwarding_always_enabled() { return input_event_forwarding_always_enabled; }

	void set_force_draw_over_forwarding_enabled();
	bool is_force_draw_over_forwarding_enabled() { return force_draw_over_forwarding_enabled; }

	void notify_main_screen_changed(const String &screen_name);
	void notify_scene_changed(const Node *scn_root);
	void notify_scene_closed(const String &scene_filepath);
	void notify_resource_saved(const Ref<Resource> &p_resource);

	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay);
	virtual void forward_canvas_force_draw_over_viewport(Control *p_overlay);

	virtual EditorPlugin::AfterGUIInput forward_spatial_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event);
	virtual void forward_spatial_draw_over_viewport(Control *p_overlay);
	virtual void forward_spatial_force_draw_over_viewport(Control *p_overlay);

	virtual String get_name() const;
	virtual const Ref<Texture2D> get_icon() const;
	virtual bool has_main_screen() const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify() {} //notify that it was raised by the user, not the editor
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual Dictionary get_state() const; //save editor state so it can't be reloaded when reloading scene
	virtual void set_state(const Dictionary &p_state); //restore editor state (likely was saved with the scene)
	virtual void clear(); // clear any temporary data in the editor, reset it (likely new scene or load another scene)
	virtual void save_external_data(); // if editor references external resources/scenes, save them
	virtual void apply_changes(); // if changes are pending in editor, apply them
	virtual void get_breakpoints(List<String> *p_breakpoints);
	virtual bool get_remove_list(List<Node *> *p_list);
	virtual void set_window_layout(Ref<ConfigFile> p_layout);
	virtual void get_window_layout(Ref<ConfigFile> p_layout);
	virtual void edited_scene_changed() {} // if changes are pending in editor, apply them
	virtual bool build(); // builds with external tools. Returns true if safe to continue running scene.

	EditorInterface *get_editor_interface();
	ScriptCreateDialog *get_script_create_dialog();

	void add_undo_redo_inspector_hook_callback(Callable p_callable);
	void remove_undo_redo_inspector_hook_callback(Callable p_callable);

	int update_overlays() const;

	void queue_save_layout();

	void make_bottom_panel_item_visible(Control *p_item);
	void hide_bottom_panel();

	virtual void restore_global_state();
	virtual void save_global_state();

	void add_translation_parser_plugin(const Ref<EditorTranslationParserPlugin> &p_parser);
	void remove_translation_parser_plugin(const Ref<EditorTranslationParserPlugin> &p_parser);

	void add_import_plugin(const Ref<EditorImportPlugin> &p_importer);
	void remove_import_plugin(const Ref<EditorImportPlugin> &p_importer);

	void add_export_plugin(const Ref<EditorExportPlugin> &p_exporter);
	void remove_export_plugin(const Ref<EditorExportPlugin> &p_exporter);

	void add_spatial_gizmo_plugin(const Ref<EditorNode3DGizmoPlugin> &p_gizmo_plugin);
	void remove_spatial_gizmo_plugin(const Ref<EditorNode3DGizmoPlugin> &p_gizmo_plugin);

	void add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	void remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);

	void add_scene_format_importer_plugin(const Ref<EditorSceneFormatImporter> &p_importer);
	void remove_scene_format_importer_plugin(const Ref<EditorSceneFormatImporter> &p_importer);

	void add_scene_post_import_plugin(const Ref<EditorScenePostImportPlugin> &p_importer);
	void remove_scene_post_import_plugin(const Ref<EditorScenePostImportPlugin> &p_importer);

	void add_autoload_singleton(const String &p_name, const String &p_path);
	void remove_autoload_singleton(const String &p_name);

	void add_debugger_plugin(const Ref<Script> &p_script);
	void remove_debugger_plugin(const Ref<Script> &p_script);

	void enable_plugin();
	void disable_plugin();

	EditorPlugin() {}
	virtual ~EditorPlugin() {}
};

VARIANT_ENUM_CAST(EditorPlugin::CustomControlContainer);
VARIANT_ENUM_CAST(EditorPlugin::DockSlot);

typedef EditorPlugin *(*EditorPluginCreateFunc)(EditorNode *);

class EditorPlugins {
	enum {
		MAX_CREATE_FUNCS = 64
	};

	static EditorPluginCreateFunc creation_funcs[MAX_CREATE_FUNCS];
	static int creation_func_count;

	template <class T>
	static EditorPlugin *creator(EditorNode *p_node) {
		return memnew(T(p_node));
	}

public:
	static int get_plugin_count() { return creation_func_count; }
	static EditorPlugin *create(int p_idx, EditorNode *p_editor) {
		ERR_FAIL_INDEX_V(p_idx, creation_func_count, nullptr);
		return creation_funcs[p_idx](p_editor);
	}

	template <class T>
	static void add_by_type() {
		add_create_func(creator<T>);
	}

	static void add_create_func(EditorPluginCreateFunc p_func) {
		ERR_FAIL_COND(creation_func_count >= MAX_CREATE_FUNCS);
		creation_funcs[creation_func_count++] = p_func;
	}
};

#endif
