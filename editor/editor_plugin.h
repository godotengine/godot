/*************************************************************************/
/*  editor_plugin.h                                                      */
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
#ifndef EDITOR_PLUGIN_H
#define EDITOR_PLUGIN_H

#include "io/config_file.h"
#include "scene/gui/tool_button.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"
#include "undo_redo.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class EditorNode;
class Spatial;
class Camera;
class EditorSelection;
class EditorExport;
class EditorSettings;
class SpatialEditorGizmo;
class EditorImportPlugin;
class EditorExportPlugin;
class EditorResourcePreview;
class EditorFileSystem;

class EditorPlugin : public Node {

	GDCLASS(EditorPlugin, Node);
	friend class EditorData;
	UndoRedo *undo_redo;

	UndoRedo *_get_undo_redo() { return undo_redo; }

protected:
	static void _bind_methods();
	UndoRedo &get_undo_redo() { return *undo_redo; }

	void add_custom_type(const String &p_type, const String &p_base, const Ref<Script> &p_script, const Ref<Texture> &p_icon);
	void remove_custom_type(const String &p_type);

public:
	enum CustomControlContainer {
		CONTAINER_TOOLBAR,
		CONTAINER_SPATIAL_EDITOR_MENU,
		CONTAINER_SPATIAL_EDITOR_SIDE,
		CONTAINER_SPATIAL_EDITOR_BOTTOM,
		CONTAINER_CANVAS_EDITOR_MENU,
		CONTAINER_CANVAS_EDITOR_SIDE,
		CONTAINER_CANVAS_EDITOR_BOTTOM,
		CONTAINER_PROPERTY_EDITOR_BOTTOM
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

	//TODO: send a resoucre for editing to the editor node?

	void add_control_to_container(CustomControlContainer p_location, Control *p_control);
	ToolButton *add_control_to_bottom_panel(Control *p_control, const String &p_title);
	void add_control_to_dock(DockSlot p_slot, Control *p_control);
	void remove_control_from_docks(Control *p_control);
	void remove_control_from_bottom_panel(Control *p_control);
	Control *get_editor_viewport();
	void edit_resource(const Ref<Resource> &p_resource);

	void add_tool_menu_item(const String &p_name, Object *p_handler, const String &p_callback, const Variant &p_ud = Variant());
	void add_tool_submenu_item(const String &p_name, Object *p_submenu);
	void remove_tool_menu_item(const String &p_name);

	virtual Ref<SpatialEditorGizmo> create_spatial_gizmo(Spatial *p_spatial);
	virtual bool forward_canvas_gui_input(const Transform2D &p_canvas_xform, const InputEvent &p_event);
	virtual void forward_draw_over_canvas(const Transform2D &p_canvas_xform, Control *p_canvas);
	virtual bool forward_spatial_gui_input(Camera *p_camera, const InputEvent &p_event);
	virtual String get_name() const;
	virtual bool has_main_screen() const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify() {} //notify that it was raised by the user, not the editor
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_node) const;
	virtual Dictionary get_state() const; //save editor state so it can't be reloaded when reloading scene
	virtual void set_state(const Dictionary &p_state); //restore editor state (likely was saved with the scene)
	virtual void clear(); // clear any temporary data in te editor, reset it (likely new scene or load another scene)
	virtual void save_external_data(); // if editor references external resources/scenes, save them
	virtual void apply_changes(); // if changes are pending in editor, apply them
	virtual void get_breakpoints(List<String> *p_breakpoints);
	virtual bool get_remove_list(List<Node *> *p_list);
	virtual void set_window_layout(Ref<ConfigFile> p_layout);
	virtual void get_window_layout(Ref<ConfigFile> p_layout);
	virtual void edited_scene_changed() {} // if changes are pending in editor, apply them

	void update_canvas();

	virtual void inspect_object(Object *p_obj, const String &p_for_property = String());

	void queue_save_layout() const;

	Control *get_base_control();

	void make_bottom_panel_item_visible(Control *p_item);
	void hide_bottom_panel();

	EditorSelection *get_selection();
	//EditorImportExport *get_import_export();
	EditorSettings *get_editor_settings();
	EditorResourcePreview *get_resource_previewer();
	EditorFileSystem *get_resource_file_system();

	virtual void restore_global_state();
	virtual void save_global_state();

	EditorPlugin();
	virtual ~EditorPlugin();
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
		ERR_FAIL_INDEX_V(p_idx, creation_func_count, NULL);
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
