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
#include "editor/editor_export.h"
#include "editor/editor_inspector.h"
#include "editor/editor_singletons.h"
#include "editor/editor_translation_parser.h"
#include "editor/import/editor_import_plugin.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/script_create_dialog.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"

class EditorNode;
class EditorInterface;
class EditorDocks;
class EditorNode3DGizmoPlugin;

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

	//TODO: send a resource for editing to the editor node?

	void add_control_to_container(CustomControlContainer p_location, Control *p_control);
	void remove_control_from_container(CustomControlContainer p_location, Control *p_control);

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

	virtual bool forward_spatial_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event);
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
	EditorDocks *get_editor_docks();
	EditorBottomPanels *get_editor_bottom_panels();
	EditorWorkspaces *get_editor_workspaces();
	EditorPluginInterfaces *get_editor_plugin_interfaces();
	EditorTopBars *get_editor_top_bars();

	ScriptCreateDialog *get_script_create_dialog();

	int update_overlays() const;

	void queue_save_layout();

	virtual void restore_global_state();
	virtual void save_global_state();

	void add_autoload_singleton(const String &p_name, const String &p_path);
	void remove_autoload_singleton(const String &p_name);

	void enable_plugin();
	void disable_plugin();

	EditorPlugin() {}
	virtual ~EditorPlugin() {}
};

VARIANT_ENUM_CAST(EditorPlugin::CustomControlContainer);

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
