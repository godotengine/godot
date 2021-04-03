/*************************************************************************/
/*  editor_plugin.cpp                                                    */
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

#include "editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_singletons.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/project_settings_editor.h"

void EditorPlugin::add_custom_type(const String &p_type, const String &p_base, const Ref<Script> &p_script, const Ref<Texture2D> &p_icon) {
	EditorNode::get_editor_data().add_custom_type(p_type, p_base, p_script, p_icon);
}

void EditorPlugin::remove_custom_type(const String &p_type) {
	EditorNode::get_editor_data().remove_custom_type(p_type);
}

void EditorPlugin::add_autoload_singleton(const String &p_name, const String &p_path) {
	EditorNode::get_singleton()->get_project_settings()->get_autoload_settings()->autoload_add(p_name, p_path);
}

void EditorPlugin::remove_autoload_singleton(const String &p_name) {
	EditorNode::get_singleton()->get_project_settings()->get_autoload_settings()->autoload_remove(p_name);
}

void EditorPlugin::add_control_to_container(CustomControlContainer p_location, Control *p_control) {
	ERR_FAIL_NULL(p_control);

	switch (p_location) {
		case CONTAINER_TOOLBAR: {
			EditorNode::get_menu_hb()->add_child(p_control);
		} break;

		case CONTAINER_SPATIAL_EDITOR_MENU: {
			Node3DEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE_LEFT: {
			Node3DEditor::get_singleton()->get_palette_split()->add_child(p_control);
			Node3DEditor::get_singleton()->get_palette_split()->move_child(p_control, 0);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT: {
			Node3DEditor::get_singleton()->get_palette_split()->add_child(p_control);
			Node3DEditor::get_singleton()->get_palette_split()->move_child(p_control, 1);

		} break;
		case CONTAINER_SPATIAL_EDITOR_BOTTOM: {
			Node3DEditor::get_singleton()->get_shader_split()->add_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_MENU: {
			CanvasItemEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE_LEFT: {
			CanvasItemEditor::get_singleton()->get_palette_split()->add_child(p_control);
			CanvasItemEditor::get_singleton()->get_palette_split()->move_child(p_control, 0);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE_RIGHT: {
			CanvasItemEditor::get_singleton()->get_palette_split()->add_child(p_control);
			CanvasItemEditor::get_singleton()->get_palette_split()->move_child(p_control, 1);

		} break;
		case CONTAINER_CANVAS_EDITOR_BOTTOM: {
			CanvasItemEditor::get_singleton()->get_bottom_split()->add_child(p_control);

		} break;
		case CONTAINER_PROPERTY_EDITOR_BOTTOM: {
			EditorNode::get_singleton()->get_inspector_dock_addon_area()->add_child(p_control);

		} break;
		case CONTAINER_PROJECT_SETTING_TAB_LEFT: {
			ProjectSettingsEditor::get_singleton()->get_tabs()->add_child(p_control);
			ProjectSettingsEditor::get_singleton()->get_tabs()->move_child(p_control, 0);

		} break;
		case CONTAINER_PROJECT_SETTING_TAB_RIGHT: {
			ProjectSettingsEditor::get_singleton()->get_tabs()->add_child(p_control);
			ProjectSettingsEditor::get_singleton()->get_tabs()->move_child(p_control, 1);

		} break;
	}
}

void EditorPlugin::remove_control_from_container(CustomControlContainer p_location, Control *p_control) {
	ERR_FAIL_NULL(p_control);

	switch (p_location) {
		case CONTAINER_TOOLBAR: {
			EditorNode::get_menu_hb()->remove_child(p_control);
		} break;

		case CONTAINER_SPATIAL_EDITOR_MENU: {
			Node3DEditor::get_singleton()->remove_control_from_menu_panel(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE_LEFT:
		case CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT: {
			Node3DEditor::get_singleton()->get_palette_split()->remove_child(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_BOTTOM: {
			Node3DEditor::get_singleton()->get_shader_split()->remove_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_MENU: {
			CanvasItemEditor::get_singleton()->remove_control_from_menu_panel(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE_LEFT:
		case CONTAINER_CANVAS_EDITOR_SIDE_RIGHT: {
			CanvasItemEditor::get_singleton()->get_palette_split()->remove_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_BOTTOM: {
			CanvasItemEditor::get_singleton()->get_bottom_split()->remove_child(p_control);

		} break;
		case CONTAINER_PROPERTY_EDITOR_BOTTOM: {
			EditorNode::get_singleton()->get_inspector_dock_addon_area()->remove_child(p_control);

		} break;
		case CONTAINER_PROJECT_SETTING_TAB_LEFT:
		case CONTAINER_PROJECT_SETTING_TAB_RIGHT: {
			ProjectSettingsEditor::get_singleton()->get_tabs()->remove_child(p_control);

		} break;
	}
}

void EditorPlugin::add_tool_menu_item(const String &p_name, const Callable &p_callable) {
	EditorNode::get_singleton()->add_tool_menu_item(p_name, p_callable);
}

void EditorPlugin::add_tool_submenu_item(const String &p_name, Object *p_submenu) {
	ERR_FAIL_NULL(p_submenu);
	PopupMenu *submenu = Object::cast_to<PopupMenu>(p_submenu);
	ERR_FAIL_NULL(submenu);
	EditorNode::get_singleton()->add_tool_submenu_item(p_name, submenu);
}

void EditorPlugin::remove_tool_menu_item(const String &p_name) {
	EditorNode::get_singleton()->remove_tool_menu_item(p_name);
}

void EditorPlugin::set_input_event_forwarding_always_enabled() {
	input_event_forwarding_always_enabled = true;
	EditorPluginList *always_input_forwarding_list = EditorNode::get_singleton()->get_editor_plugins_force_input_forwarding();
	always_input_forwarding_list->add_plugin(this);
}

void EditorPlugin::set_force_draw_over_forwarding_enabled() {
	force_draw_over_forwarding_enabled = true;
	EditorPluginList *always_draw_over_forwarding_list = EditorNode::get_singleton()->get_editor_plugins_force_over();
	always_draw_over_forwarding_list->add_plugin(this);
}

void EditorPlugin::notify_scene_changed(const Node *scn_root) {
	emit_signal("scene_changed", scn_root);
}

void EditorPlugin::notify_main_screen_changed(const String &screen_name) {
	if (screen_name == last_main_screen_name) {
		return;
	}

	emit_signal("main_screen_changed", screen_name);
	last_main_screen_name = screen_name;
}

void EditorPlugin::notify_scene_closed(const String &scene_filepath) {
	emit_signal("scene_closed", scene_filepath);
}

void EditorPlugin::notify_resource_saved(const Ref<Resource> &p_resource) {
	emit_signal("resource_saved", p_resource);
}

bool EditorPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (get_script_instance() && get_script_instance()->has_method("forward_canvas_gui_input")) {
		return get_script_instance()->call("forward_canvas_gui_input", p_event);
	}
	return false;
}

void EditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (get_script_instance() && get_script_instance()->has_method("forward_canvas_draw_over_viewport")) {
		get_script_instance()->call("forward_canvas_draw_over_viewport", p_overlay);
	}
}

void EditorPlugin::forward_canvas_force_draw_over_viewport(Control *p_overlay) {
	if (get_script_instance() && get_script_instance()->has_method("forward_canvas_force_draw_over_viewport")) {
		get_script_instance()->call("forward_canvas_force_draw_over_viewport", p_overlay);
	}
}

// Updates the overlays of the 2D viewport or, if in 3D mode, of every 3D viewport.
int EditorPlugin::update_overlays() const {
	if (Node3DEditor::get_singleton()->is_visible()) {
		int count = 0;
		for (uint32_t i = 0; i < Node3DEditor::VIEWPORTS_COUNT; i++) {
			Node3DEditorViewport *vp = Node3DEditor::get_singleton()->get_editor_viewport(i);
			if (vp->is_visible()) {
				vp->update_surface();
				count++;
			}
		}
		return count;
	} else {
		// This will update the normal viewport itself as well
		CanvasItemEditor::get_singleton()->get_viewport_control()->update();
		return 1;
	}
}

bool EditorPlugin::forward_spatial_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (get_script_instance() && get_script_instance()->has_method("forward_spatial_gui_input")) {
		return get_script_instance()->call("forward_spatial_gui_input", p_camera, p_event);
	}

	return false;
}

void EditorPlugin::forward_spatial_draw_over_viewport(Control *p_overlay) {
	if (get_script_instance() && get_script_instance()->has_method("forward_spatial_draw_over_viewport")) {
		get_script_instance()->call("forward_spatial_draw_over_viewport", p_overlay);
	}
}

void EditorPlugin::forward_spatial_force_draw_over_viewport(Control *p_overlay) {
	if (get_script_instance() && get_script_instance()->has_method("forward_spatial_force_draw_over_viewport")) {
		get_script_instance()->call("forward_spatial_force_draw_over_viewport", p_overlay);
	}
}

String EditorPlugin::get_name() const {
	if (get_script_instance() && get_script_instance()->has_method("get_plugin_name")) {
		return get_script_instance()->call("get_plugin_name");
	}

	return String();
}

const Ref<Texture2D> EditorPlugin::get_icon() const {
	if (get_script_instance() && get_script_instance()->has_method("get_plugin_icon")) {
		return get_script_instance()->call("get_plugin_icon");
	}

	return Ref<Texture2D>();
}

bool EditorPlugin::has_main_screen() const {
	if (get_script_instance() && get_script_instance()->has_method("has_main_screen")) {
		return get_script_instance()->call("has_main_screen");
	}

	return false;
}

void EditorPlugin::make_visible(bool p_visible) {
	if (get_script_instance() && get_script_instance()->has_method("make_visible")) {
		get_script_instance()->call("make_visible", p_visible);
	}
}

void EditorPlugin::edit(Object *p_object) {
	if (get_script_instance() && get_script_instance()->has_method("edit")) {
		if (p_object->is_class("Resource")) {
			get_script_instance()->call("edit", Ref<Resource>(Object::cast_to<Resource>(p_object)));
		} else {
			get_script_instance()->call("edit", p_object);
		}
	}
}

bool EditorPlugin::handles(Object *p_object) const {
	if (get_script_instance() && get_script_instance()->has_method("handles")) {
		return get_script_instance()->call("handles", p_object);
	}

	return false;
}

Dictionary EditorPlugin::get_state() const {
	if (get_script_instance() && get_script_instance()->has_method("get_state")) {
		return get_script_instance()->call("get_state");
	}

	return Dictionary();
}

void EditorPlugin::set_state(const Dictionary &p_state) {
	if (get_script_instance() && get_script_instance()->has_method("set_state")) {
		get_script_instance()->call("set_state", p_state);
	}
}

void EditorPlugin::clear() {
	if (get_script_instance() && get_script_instance()->has_method("clear")) {
		get_script_instance()->call("clear");
	}
}

// if editor references external resources/scenes, save them
void EditorPlugin::save_external_data() {
	if (get_script_instance() && get_script_instance()->has_method("save_external_data")) {
		get_script_instance()->call("save_external_data");
	}
}

// if changes are pending in editor, apply them
void EditorPlugin::apply_changes() {
	if (get_script_instance() && get_script_instance()->has_method("apply_changes")) {
		get_script_instance()->call("apply_changes");
	}
}

void EditorPlugin::get_breakpoints(List<String> *p_breakpoints) {
	if (get_script_instance() && get_script_instance()->has_method("get_breakpoints")) {
		PackedStringArray arr = get_script_instance()->call("get_breakpoints");
		for (int i = 0; i < arr.size(); i++) {
			p_breakpoints->push_back(arr[i]);
		}
	}
}

bool EditorPlugin::get_remove_list(List<Node *> *p_list) {
	return false;
}

void EditorPlugin::restore_global_state() {}
void EditorPlugin::save_global_state() {}

int find(const PackedStringArray &a, const String &v) {
	const String *r = a.ptr();
	for (int j = 0; j < a.size(); ++j) {
		if (r[j] == v) {
			return j;
		}
	}
	return -1;
}

void EditorPlugin::enable_plugin() {
	// Called when the plugin gets enabled in project settings, after it's added to the tree.
	// You can implement it to register autoloads.
	if (get_script_instance() && get_script_instance()->has_method("enable_plugin")) {
		get_script_instance()->call("enable_plugin");
	}
}

void EditorPlugin::disable_plugin() {
	// Last function called when the plugin gets disabled in project settings.
	// Implement it to cleanup things from the project, such as unregister autoloads.

	if (get_script_instance() && get_script_instance()->has_method("disable_plugin")) {
		get_script_instance()->call("disable_plugin");
	}
}

void EditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {
	if (get_script_instance() && get_script_instance()->has_method("set_window_layout")) {
		get_script_instance()->call("set_window_layout", p_layout);
	}
}

void EditorPlugin::get_window_layout(Ref<ConfigFile> p_layout) {
	if (get_script_instance() && get_script_instance()->has_method("get_window_layout")) {
		get_script_instance()->call("get_window_layout", p_layout);
	}
}

bool EditorPlugin::build() {
	if (get_script_instance() && get_script_instance()->has_method("build")) {
		return get_script_instance()->call("build");
	}

	return true;
}

void EditorPlugin::queue_save_layout() {
	EditorNode::get_singleton()->save_layout();
}

EditorInterface *EditorPlugin::get_editor_interface() {
	return EditorInterface::get_singleton();
}

EditorDocks *EditorPlugin::get_editor_docks() {
	return EditorDocks::get_singleton();
}

EditorBottomPanels *EditorPlugin::get_editor_bottom_panels() {
	return EditorBottomPanels::get_singleton();
}

ScriptCreateDialog *EditorPlugin::get_script_create_dialog() {
	return EditorNode::get_singleton()->get_script_create_dialog();
}

void EditorPlugin::_editor_project_settings_changed() {
	emit_signal("project_settings_changed");
}
void EditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		EditorNode::get_singleton()->connect("project_settings_changed", callable_mp(this, &EditorPlugin::_editor_project_settings_changed));
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {
		EditorNode::get_singleton()->disconnect("project_settings_changed", callable_mp(this, &EditorPlugin::_editor_project_settings_changed));
	}
}

void EditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_control_to_container", "container", "control"), &EditorPlugin::add_control_to_container);
	ClassDB::bind_method(D_METHOD("remove_control_from_container", "container", "control"), &EditorPlugin::remove_control_from_container);
	ClassDB::bind_method(D_METHOD("add_tool_menu_item", "name", "callable"), &EditorPlugin::add_tool_menu_item);
	ClassDB::bind_method(D_METHOD("add_tool_submenu_item", "name", "submenu"), &EditorPlugin::add_tool_submenu_item);
	ClassDB::bind_method(D_METHOD("remove_tool_menu_item", "name"), &EditorPlugin::remove_tool_menu_item);
	ClassDB::bind_method(D_METHOD("add_custom_type", "type", "base", "script", "icon"), &EditorPlugin::add_custom_type);
	ClassDB::bind_method(D_METHOD("remove_custom_type", "type"), &EditorPlugin::remove_custom_type);

	ClassDB::bind_method(D_METHOD("add_autoload_singleton", "name", "path"), &EditorPlugin::add_autoload_singleton);
	ClassDB::bind_method(D_METHOD("remove_autoload_singleton", "name"), &EditorPlugin::remove_autoload_singleton);

	ClassDB::bind_method(D_METHOD("update_overlays"), &EditorPlugin::update_overlays);

	ClassDB::bind_method(D_METHOD("get_undo_redo"), &EditorPlugin::_get_undo_redo);
	ClassDB::bind_method(D_METHOD("queue_save_layout"), &EditorPlugin::queue_save_layout);
	ClassDB::bind_method(D_METHOD("set_input_event_forwarding_always_enabled"), &EditorPlugin::set_input_event_forwarding_always_enabled);
	ClassDB::bind_method(D_METHOD("set_force_draw_over_forwarding_enabled"), &EditorPlugin::set_force_draw_over_forwarding_enabled);

	ClassDB::bind_method(D_METHOD("get_editor_interface"), &EditorPlugin::get_editor_interface);
	ClassDB::bind_method(D_METHOD("get_editor_docks"), &EditorPlugin::get_editor_docks);
	ClassDB::bind_method(D_METHOD("get_editor_bottom_panels"), &EditorPlugin::get_editor_bottom_panels);
	ClassDB::bind_method(D_METHOD("get_script_create_dialog"), &EditorPlugin::get_script_create_dialog);

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "forward_canvas_gui_input", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("forward_canvas_draw_over_viewport", PropertyInfo(Variant::OBJECT, "overlay", PROPERTY_HINT_RESOURCE_TYPE, "Control")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("forward_canvas_force_draw_over_viewport", PropertyInfo(Variant::OBJECT, "overlay", PROPERTY_HINT_RESOURCE_TYPE, "Control")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "forward_spatial_gui_input", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera3D"), PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("forward_spatial_draw_over_viewport", PropertyInfo(Variant::OBJECT, "overlay", PROPERTY_HINT_RESOURCE_TYPE, "Control")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("forward_spatial_force_draw_over_viewport", PropertyInfo(Variant::OBJECT, "overlay", PROPERTY_HINT_RESOURCE_TYPE, "Control")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::STRING, "get_plugin_name"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "get_plugin_icon"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "has_main_screen"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("make_visible", PropertyInfo(Variant::BOOL, "visible")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("edit", PropertyInfo(Variant::OBJECT, "object")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "handles", PropertyInfo(Variant::OBJECT, "object")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::DICTIONARY, "get_state"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("set_state", PropertyInfo(Variant::DICTIONARY, "state")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("clear"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("save_external_data"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("apply_changes"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::PACKED_STRING_ARRAY, "get_breakpoints"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("set_window_layout", PropertyInfo(Variant::OBJECT, "layout", PROPERTY_HINT_RESOURCE_TYPE, "ConfigFile")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("get_window_layout", PropertyInfo(Variant::OBJECT, "layout", PROPERTY_HINT_RESOURCE_TYPE, "ConfigFile")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "build"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("enable_plugin"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("disable_plugin"));

	ADD_SIGNAL(MethodInfo("scene_changed", PropertyInfo(Variant::OBJECT, "scene_root", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("scene_closed", PropertyInfo(Variant::STRING, "filepath")));
	ADD_SIGNAL(MethodInfo("main_screen_changed", PropertyInfo(Variant::STRING, "screen_name")));
	ADD_SIGNAL(MethodInfo("resource_saved", PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource")));
	ADD_SIGNAL(MethodInfo("project_settings_changed"));

	BIND_ENUM_CONSTANT(CONTAINER_TOOLBAR);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_MENU);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_SIDE_LEFT);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_MENU);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_SIDE_LEFT);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_SIDE_RIGHT);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_PROPERTY_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_PROJECT_SETTING_TAB_LEFT);
	BIND_ENUM_CONSTANT(CONTAINER_PROJECT_SETTING_TAB_RIGHT);
}

EditorPluginCreateFunc EditorPlugins::creation_funcs[MAX_CREATE_FUNCS];

int EditorPlugins::creation_func_count = 0;
