/**************************************************************************/
/*  scene_debugger.h                                                      */
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

#pragma once

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "scene/debugger/runtime_node_select.h"
#include "scene/main/window.h"

class Array;
#ifdef TOOLS_ENABLED
class Button;
class MenuButton;
#endif // TOOLS_ENABLED
class InputEvent;
class Node;
class Shortcut;

#ifdef TOOLS_ENABLED
class SceneDebuggerToolbar : public Window {
	GDCLASS(SceneDebuggerToolbar, Window);

	enum CameraOverride {
		OVERRIDE_NONE,
		OVERRIDE_INGAME,
		OVERRIDE_EDITORS,
	};

	enum {
		CAMERA_RESET_2D,
		CAMERA_RESET_3D,
		CAMERA_MODE_INGAME,
		CAMERA_MODE_EDITORS,
		SELECTION_AVOID_LOCKED,
		SELECTION_PREFER_GROUP,
		WINDOW_RUN_GAME_EMBEDDED,
		WINDOW_MAKE_FLOATING_ON_PLAY,
		WINDOW_SIZE_MODE_FIXED,
		WINDOW_SIZE_MODE_KEEP_ASPECT,
		WINDOW_SIZE_MODE_STRETCH,
		WINDOW_SEPARATOR_DYNAMIC_RANGE,
		WINDOW_REQUEST_HDR_OUTPUT,
		WINDOW_HDR_OUTPUT_ERROR,
	};

	CameraOverride camera_override_mode = OVERRIDE_INGAME;

	bool selection_avoid_locked = false;
	bool selection_prefer_group = false;

	bool paused = false;
	bool debug_mute_audio = false;

	Button *suspend_button = nullptr;
	Button *next_frame_button = nullptr;

	Button *node_type_button[RuntimeNodeSelect::NODE_TYPE_MAX];
	Button *select_mode_button[RuntimeNodeSelect::SELECT_MODE_MAX];

	Button *hide_selection = nullptr;
	MenuButton *selection_options_menu = nullptr;

	Button *debug_mute_audio_button = nullptr;

	Button *camera_override_button = nullptr;
	MenuButton *camera_override_menu = nullptr;

	int const DEFAULT_TIME_SCALE_INDEX = 5;
	Array time_scale_range = { 0.0625f, 0.125f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 4.0f, 8.0f, 16.0f };
	Array time_scale_label = { "1/16", "1/8", "1/4", "1/2", "3/4", "1.0", "1.25", "1.5", "1.75", "2.0", "4.0", "8.0", "16.0" };
	int time_scale_index = DEFAULT_TIME_SCALE_INDEX;

	MenuButton *speed_state_button = nullptr;
	Button *reset_speed_button = nullptr;

	void _drag_handle_gui_input(const Ref<InputEvent> &p_event);

	void _update_debugger_buttons();

	void _hide_selection_toggled(bool p_pressed);
	void _debug_mute_audio_button_pressed();

	void _camera_override_button_toggled(bool p_pressed);
	void _camera_override_menu_id_pressed(int p_id);

	void _toggle_suspend_button();
	void _suspend_button_toggled(bool p_pressed);

	void _node_type_pressed(int p_option);
	void _select_mode_pressed(int p_option);
	void _selection_options_menu_id_pressed(int p_id);

	void _reset_time_scales();
	void _speed_state_menu_pressed(int p_id);
	void _update_speed_buttons();
	void _update_speed_state_color();
	void _update_speed_state_size();

protected:
	void _notification(int p_what);

public:
	SceneDebuggerToolbar();
};
#endif // TOOLS_ENABLED

class SceneDebugger {
	friend class SceneDebuggerToolbar;

private:
	inline static SceneDebugger *singleton = nullptr;

	SceneDebugger();

public:
	static void initialize();
	static void deinitialize();

	~SceneDebugger();

#ifdef DEBUG_ENABLED
private:
	static void _handle_input(const Ref<InputEvent> &p_event, const Ref<Shortcut> &p_shortcut);
	static void _handle_embed_input(const Ref<InputEvent> &p_event, const Dictionary &p_settings);
	static void _on_window_size_changed();
	static void _on_output_max_linear_value_changed(float output_max_linear_value);

	static void _save_node(ObjectID id, const String &p_path);
	static void _set_node_owner_recursive(Node *p_node, Node *p_owner);
	static void _set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value, const String &p_field = "");
	static void _send_object_ids(const Vector<ObjectID> &p_ids, bool p_update_selection);
	static void _next_frame();

	/// Message handler function for parse_message.
	typedef Error (*ParseMessageFunc)(const Array &p_args);
	static HashMap<String, ParseMessageFunc> message_handlers;
	static void _init_message_handlers();

	static Error _msg_setup_scene(const Array &p_args);
	static Error _msg_setup_game_view(const Array &p_args);
	static Error _msg_setup_embedded_shortcuts(const Array &p_args);
	static Error _msg_request_scene_tree(const Array &p_args);
	static Error _msg_save_node(const Array &p_args);
	static Error _msg_inspect_objects(const Array &p_args);
#ifndef DISABLE_DEPRECATED
	static Error _msg_inspect_object(const Array &p_args);
#endif // DISABLE_DEPRECATED
	static Error _msg_clear_selection(const Array &p_args);
	static Error _msg_suspend_changed(const Array &p_args);
	static Error _msg_next_frame(const Array &p_args);
	static Error _msg_speed_changed(const Array &p_args);
	static Error _msg_debug_mute_audio(const Array &p_args);
	static Error _msg_window_request_size(const Array &p_args);
	static Error _msg_hdr_output_request_state(const Array &p_args);
	static Error _msg_hdr_output_toggle_requested(const Array &p_args);
	static Error _msg_override_cameras(const Array &p_args);
	static Error _msg_set_object_property(const Array &p_args);
	static Error _msg_set_object_property_field(const Array &p_args);
	static Error _msg_reload_cached_files(const Array &p_args);
	static Error _msg_live_set_root(const Array &p_args);
	static Error _msg_live_node_path(const Array &p_args);
	static Error _msg_live_res_path(const Array &p_args);
	static Error _msg_live_node_prop_res(const Array &p_args);
	static Error _msg_live_node_prop(const Array &p_args);
	static Error _msg_live_res_prop_res(const Array &p_args);
	static Error _msg_live_res_prop(const Array &p_args);
	static Error _msg_live_node_call(const Array &p_args);
	static Error _msg_live_res_call(const Array &p_args);
	static Error _msg_live_create_node(const Array &p_args);
	static Error _msg_live_instantiate_node(const Array &p_args);
	static Error _msg_live_remove_node(const Array &p_args);
	static Error _msg_live_remove_and_keep_node(const Array &p_args);
	static Error _msg_live_restore_node(const Array &p_args);
	static Error _msg_live_duplicate_node(const Array &p_args);
	static Error _msg_live_reparent_node(const Array &p_args);
	static Error _msg_runtime_node_select_setup(const Array &p_args);
	static Error _msg_runtime_node_select_set_type(const Array &p_args);
	static Error _msg_runtime_node_select_set_mode(const Array &p_args);
	static Error _msg_runtime_node_select_set_visible(const Array &p_args);
	static Error _msg_runtime_node_select_set_avoid_locked(const Array &p_args);
	static Error _msg_runtime_node_select_set_prefer_group(const Array &p_args);
	static Error _msg_rq_screenshot(const Array &p_args);
	static Error _msg_report_window_focused(const Array &p_args);

	static Error _msg_runtime_node_select_reset_camera_2d(const Array &p_args);
	static Error _msg_transform_camera_2d(const Array &p_args);
#ifndef _3D_DISABLED
	static Error _msg_runtime_node_select_reset_camera_3d(const Array &p_args);
	static Error _msg_transform_camera_3d(const Array &p_args);
#endif // _3D_DISABLED

#ifdef TOOLS_ENABLED
	static inline SceneDebuggerToolbar *toolbar = nullptr;
	static void create_debug_menu(float p_ed_scale);
#endif // TOOLS_ENABLED

public:
	static Error parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured);
	static void add_to_cache(const String &p_filename, Node *p_node);
	static void remove_from_cache(const String &p_filename, Node *p_node);
	static void reload_cached_files(const PackedStringArray &p_files);
#endif
};

#ifdef DEBUG_ENABLED
class LiveEditor {
private:
	friend class SceneDebugger;
	HashMap<int, NodePath> live_edit_node_path_cache;
	HashMap<int, String> live_edit_resource_cache;

	NodePath live_edit_root;
	String live_edit_scene;

	HashMap<String, HashSet<Node *>> live_scene_edit_cache;
	HashMap<Node *, HashMap<ObjectID, Node *>> live_edit_remove_list;

	void _send_tree();

	void _node_path_func(const NodePath &p_path, int p_id);
	void _res_path_func(const String &p_path, int p_id);

	void _node_set_func(int p_id, const StringName &p_prop, const Variant &p_value);
	void _node_set_res_func(int p_id, const StringName &p_prop, const String &p_value);
	void _node_call_func(int p_id, const StringName &p_method, const Variant **p_args, int p_argcount);
	void _res_set_func(int p_id, const StringName &p_prop, const Variant &p_value);
	void _res_set_res_func(int p_id, const StringName &p_prop, const String &p_value);
	void _res_call_func(int p_id, const StringName &p_method, const Variant **p_args, int p_argcount);
	void _root_func(const NodePath &p_scene_path, const String &p_scene_from);

	void _create_node_func(const NodePath &p_parent, const String &p_type, const String &p_name);
	void _instance_node_func(const NodePath &p_parent, const String &p_path, const String &p_name);
	void _remove_node_func(const NodePath &p_at);
	void _remove_and_keep_node_func(const NodePath &p_at, ObjectID p_keep_id);
	void _restore_node_func(ObjectID p_id, const NodePath &p_at, int p_at_pos);
	void _duplicate_node_func(const NodePath &p_at, const String &p_new_name);
	void _reparent_node_func(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);

	LiveEditor() {
		singleton = this;
		live_edit_root = NodePath("/root");
	}

	inline static LiveEditor *singleton = nullptr;

public:
	static LiveEditor *get_singleton();
};
#endif // DEBUG_ENABLED
