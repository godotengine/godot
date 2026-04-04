/**************************************************************************/
/*  editor_plugin.hpp                                                     */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/editor_context_menu_plugin.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/shortcut.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/classes/editor_plugin_registration.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Button;
class Callable;
class Camera3D;
class ConfigFile;
class Control;
class EditorDebuggerPlugin;
class EditorDock;
class EditorExportPlatform;
class EditorExportPlugin;
class EditorImportPlugin;
class EditorInspectorPlugin;
class EditorInterface;
class EditorNode3DGizmoPlugin;
class EditorResourceConversionPlugin;
class EditorSceneFormatImporter;
class EditorScenePostImportPlugin;
class EditorTranslationParserPlugin;
class EditorUndoRedoManager;
class InputEvent;
class Object;
class PopupMenu;
class Script;
class ScriptCreateDialog;
class Texture2D;

class EditorPlugin : public Node {
	GDEXTENSION_CLASS(EditorPlugin, Node)

public:
	enum CustomControlContainer {
		CONTAINER_TOOLBAR = 0,
		CONTAINER_SPATIAL_EDITOR_MENU = 1,
		CONTAINER_SPATIAL_EDITOR_SIDE_LEFT = 2,
		CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT = 3,
		CONTAINER_SPATIAL_EDITOR_BOTTOM = 4,
		CONTAINER_CANVAS_EDITOR_MENU = 5,
		CONTAINER_CANVAS_EDITOR_SIDE_LEFT = 6,
		CONTAINER_CANVAS_EDITOR_SIDE_RIGHT = 7,
		CONTAINER_CANVAS_EDITOR_BOTTOM = 8,
		CONTAINER_INSPECTOR_BOTTOM = 9,
		CONTAINER_PROJECT_SETTING_TAB_LEFT = 10,
		CONTAINER_PROJECT_SETTING_TAB_RIGHT = 11,
	};

	enum DockSlot {
		DOCK_SLOT_NONE = -1,
		DOCK_SLOT_LEFT_UL = 0,
		DOCK_SLOT_LEFT_BL = 1,
		DOCK_SLOT_LEFT_UR = 2,
		DOCK_SLOT_LEFT_BR = 3,
		DOCK_SLOT_RIGHT_UL = 4,
		DOCK_SLOT_RIGHT_BL = 5,
		DOCK_SLOT_RIGHT_UR = 6,
		DOCK_SLOT_RIGHT_BR = 7,
		DOCK_SLOT_BOTTOM = 8,
		DOCK_SLOT_MAX = 9,
	};

	enum AfterGUIInput {
		AFTER_GUI_INPUT_PASS = 0,
		AFTER_GUI_INPUT_STOP = 1,
		AFTER_GUI_INPUT_CUSTOM = 2,
	};

	void add_dock(EditorDock *p_dock);
	void remove_dock(EditorDock *p_dock);
	void add_control_to_container(EditorPlugin::CustomControlContainer p_container, Control *p_control);
	void remove_control_from_container(EditorPlugin::CustomControlContainer p_container, Control *p_control);
	void add_tool_menu_item(const String &p_name, const Callable &p_callable);
	void add_tool_submenu_item(const String &p_name, PopupMenu *p_submenu);
	void remove_tool_menu_item(const String &p_name);
	PopupMenu *get_export_as_menu();
	void add_custom_type(const String &p_type, const String &p_base, const Ref<Script> &p_script, const Ref<Texture2D> &p_icon);
	void remove_custom_type(const String &p_type);
	void add_control_to_dock(EditorPlugin::DockSlot p_slot, Control *p_control, const Ref<Shortcut> &p_shortcut = nullptr);
	void remove_control_from_docks(Control *p_control);
	void set_dock_tab_icon(Control *p_control, const Ref<Texture2D> &p_icon);
	Button *add_control_to_bottom_panel(Control *p_control, const String &p_title, const Ref<Shortcut> &p_shortcut = nullptr);
	void remove_control_from_bottom_panel(Control *p_control);
	void add_autoload_singleton(const String &p_name, const String &p_path);
	void remove_autoload_singleton(const String &p_name);
	int32_t update_overlays() const;
	void make_bottom_panel_item_visible(Control *p_item);
	void hide_bottom_panel();
	EditorUndoRedoManager *get_undo_redo();
	void add_undo_redo_inspector_hook_callback(const Callable &p_callable);
	void remove_undo_redo_inspector_hook_callback(const Callable &p_callable);
	void queue_save_layout();
	void add_translation_parser_plugin(const Ref<EditorTranslationParserPlugin> &p_parser);
	void remove_translation_parser_plugin(const Ref<EditorTranslationParserPlugin> &p_parser);
	void add_import_plugin(const Ref<EditorImportPlugin> &p_importer, bool p_first_priority = false);
	void remove_import_plugin(const Ref<EditorImportPlugin> &p_importer);
	void add_scene_format_importer_plugin(const Ref<EditorSceneFormatImporter> &p_scene_format_importer, bool p_first_priority = false);
	void remove_scene_format_importer_plugin(const Ref<EditorSceneFormatImporter> &p_scene_format_importer);
	void add_scene_post_import_plugin(const Ref<EditorScenePostImportPlugin> &p_scene_import_plugin, bool p_first_priority = false);
	void remove_scene_post_import_plugin(const Ref<EditorScenePostImportPlugin> &p_scene_import_plugin);
	void add_export_plugin(const Ref<EditorExportPlugin> &p_plugin);
	void remove_export_plugin(const Ref<EditorExportPlugin> &p_plugin);
	void add_export_platform(const Ref<EditorExportPlatform> &p_platform);
	void remove_export_platform(const Ref<EditorExportPlatform> &p_platform);
	void add_node_3d_gizmo_plugin(const Ref<EditorNode3DGizmoPlugin> &p_plugin);
	void remove_node_3d_gizmo_plugin(const Ref<EditorNode3DGizmoPlugin> &p_plugin);
	void add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	void remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin);
	void add_resource_conversion_plugin(const Ref<EditorResourceConversionPlugin> &p_plugin);
	void remove_resource_conversion_plugin(const Ref<EditorResourceConversionPlugin> &p_plugin);
	void set_input_event_forwarding_always_enabled();
	void set_force_draw_over_forwarding_enabled();
	void add_context_menu_plugin(EditorContextMenuPlugin::ContextMenuSlot p_slot, const Ref<EditorContextMenuPlugin> &p_plugin);
	void remove_context_menu_plugin(const Ref<EditorContextMenuPlugin> &p_plugin);
	EditorInterface *get_editor_interface();
	ScriptCreateDialog *get_script_create_dialog();
	void add_debugger_plugin(const Ref<EditorDebuggerPlugin> &p_script);
	void remove_debugger_plugin(const Ref<EditorDebuggerPlugin> &p_script);
	String get_plugin_version() const;
	virtual bool _forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	virtual void _forward_canvas_draw_over_viewport(Control *p_viewport_control);
	virtual void _forward_canvas_force_draw_over_viewport(Control *p_viewport_control);
	virtual int32_t _forward_3d_gui_input(Camera3D *p_viewport_camera, const Ref<InputEvent> &p_event);
	virtual void _forward_3d_draw_over_viewport(Control *p_viewport_control);
	virtual void _forward_3d_force_draw_over_viewport(Control *p_viewport_control);
	virtual String _get_plugin_name() const;
	virtual Ref<Texture2D> _get_plugin_icon() const;
	virtual bool _has_main_screen() const;
	virtual void _make_visible(bool p_visible);
	virtual void _edit(Object *p_object);
	virtual bool _handles(Object *p_object) const;
	virtual Dictionary _get_state() const;
	virtual void _set_state(const Dictionary &p_state);
	virtual void _clear();
	virtual String _get_unsaved_status(const String &p_for_scene) const;
	virtual void _save_external_data();
	virtual void _apply_changes();
	virtual PackedStringArray _get_breakpoints() const;
	virtual void _set_window_layout(const Ref<ConfigFile> &p_configuration);
	virtual void _get_window_layout(const Ref<ConfigFile> &p_configuration);
	virtual bool _build();
	virtual PackedStringArray _run_scene(const String &p_scene, const PackedStringArray &p_args) const;
	virtual void _enable_plugin();
	virtual void _disable_plugin();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_forward_canvas_gui_input), decltype(&T::_forward_canvas_gui_input)>) {
			BIND_VIRTUAL_METHOD(T, _forward_canvas_gui_input, 1062211774);
		}
		if constexpr (!std::is_same_v<decltype(&B::_forward_canvas_draw_over_viewport), decltype(&T::_forward_canvas_draw_over_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _forward_canvas_draw_over_viewport, 1496901182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_forward_canvas_force_draw_over_viewport), decltype(&T::_forward_canvas_force_draw_over_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _forward_canvas_force_draw_over_viewport, 1496901182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_forward_3d_gui_input), decltype(&T::_forward_3d_gui_input)>) {
			BIND_VIRTUAL_METHOD(T, _forward_3d_gui_input, 1018736637);
		}
		if constexpr (!std::is_same_v<decltype(&B::_forward_3d_draw_over_viewport), decltype(&T::_forward_3d_draw_over_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _forward_3d_draw_over_viewport, 1496901182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_forward_3d_force_draw_over_viewport), decltype(&T::_forward_3d_force_draw_over_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _forward_3d_force_draw_over_viewport, 1496901182);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_plugin_name), decltype(&T::_get_plugin_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_plugin_name, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_plugin_icon), decltype(&T::_get_plugin_icon)>) {
			BIND_VIRTUAL_METHOD(T, _get_plugin_icon, 3635182373);
		}
		if constexpr (!std::is_same_v<decltype(&B::_has_main_screen), decltype(&T::_has_main_screen)>) {
			BIND_VIRTUAL_METHOD(T, _has_main_screen, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_make_visible), decltype(&T::_make_visible)>) {
			BIND_VIRTUAL_METHOD(T, _make_visible, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_edit), decltype(&T::_edit)>) {
			BIND_VIRTUAL_METHOD(T, _edit, 3975164845);
		}
		if constexpr (!std::is_same_v<decltype(&B::_handles), decltype(&T::_handles)>) {
			BIND_VIRTUAL_METHOD(T, _handles, 397768994);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_state), decltype(&T::_get_state)>) {
			BIND_VIRTUAL_METHOD(T, _get_state, 3102165223);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_state), decltype(&T::_set_state)>) {
			BIND_VIRTUAL_METHOD(T, _set_state, 4155329257);
		}
		if constexpr (!std::is_same_v<decltype(&B::_clear), decltype(&T::_clear)>) {
			BIND_VIRTUAL_METHOD(T, _clear, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_unsaved_status), decltype(&T::_get_unsaved_status)>) {
			BIND_VIRTUAL_METHOD(T, _get_unsaved_status, 3135753539);
		}
		if constexpr (!std::is_same_v<decltype(&B::_save_external_data), decltype(&T::_save_external_data)>) {
			BIND_VIRTUAL_METHOD(T, _save_external_data, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_apply_changes), decltype(&T::_apply_changes)>) {
			BIND_VIRTUAL_METHOD(T, _apply_changes, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_breakpoints), decltype(&T::_get_breakpoints)>) {
			BIND_VIRTUAL_METHOD(T, _get_breakpoints, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_window_layout), decltype(&T::_set_window_layout)>) {
			BIND_VIRTUAL_METHOD(T, _set_window_layout, 853519107);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_window_layout), decltype(&T::_get_window_layout)>) {
			BIND_VIRTUAL_METHOD(T, _get_window_layout, 853519107);
		}
		if constexpr (!std::is_same_v<decltype(&B::_build), decltype(&T::_build)>) {
			BIND_VIRTUAL_METHOD(T, _build, 2240911060);
		}
		if constexpr (!std::is_same_v<decltype(&B::_run_scene), decltype(&T::_run_scene)>) {
			BIND_VIRTUAL_METHOD(T, _run_scene, 3911848509);
		}
		if constexpr (!std::is_same_v<decltype(&B::_enable_plugin), decltype(&T::_enable_plugin)>) {
			BIND_VIRTUAL_METHOD(T, _enable_plugin, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_disable_plugin), decltype(&T::_disable_plugin)>) {
			BIND_VIRTUAL_METHOD(T, _disable_plugin, 3218959716);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorPlugin::CustomControlContainer);
VARIANT_ENUM_CAST(EditorPlugin::DockSlot);
VARIANT_ENUM_CAST(EditorPlugin::AfterGUIInput);

