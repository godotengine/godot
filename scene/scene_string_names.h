/**************************************************************************/
/*  scene_string_names.h                                                  */
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

#include "core/string/node_path.h"
#include "core/string/string_name.h"

class SceneStringNames {
	inline static SceneStringNames *singleton = nullptr;

public:
	static void create() { singleton = memnew(SceneStringNames); }
	static void free() {
		memdelete(singleton);
		singleton = nullptr;
	}

	_FORCE_INLINE_ static SceneStringNames *get_singleton() { return singleton; }

	const StringName resized = "resized";
	const StringName draw = "draw";
	const StringName hidden = "hidden";
	const StringName visibility_changed = "visibility_changed";

	const StringName input_event = "input_event";
	const StringName gui_input = "gui_input";
	const StringName window_input = "window_input";
	const StringName nonclient_window_input = "nonclient_window_input";

	const StringName tree_entered = "tree_entered";
	const StringName tree_exiting = "tree_exiting";
	const StringName tree_exited = "tree_exited";
	const StringName ready = "ready";
	const StringName _ready = "_ready";

	const StringName item_rect_changed = "item_rect_changed";
	const StringName size_flags_changed = "size_flags_changed";
	const StringName minimum_size_changed = "minimum_size_changed";
	const StringName sleeping_state_changed = "sleeping_state_changed";
	const StringName node_configuration_warning_changed = "node_configuration_warning_changed";
	const StringName update = "update";
	const StringName updated = "updated";

	const StringName line_separation = "line_separation";
	const StringName paragraph_separation = "paragraph_separation";
	const StringName font = "font";
	const StringName font_size = "font_size";
	const StringName font_color = "font_color";

	const StringName mouse_entered = "mouse_entered";
	const StringName mouse_exited = "mouse_exited";
	const StringName mouse_shape_entered = "mouse_shape_entered";
	const StringName mouse_shape_exited = "mouse_shape_exited";
	const StringName focus_entered = "focus_entered";
	const StringName focus_exited = "focus_exited";

	const StringName pre_sort_children = "pre_sort_children";
	const StringName sort_children = "sort_children";

	const StringName finished = "finished";
	const StringName animation_finished = "animation_finished";
	const StringName animation_changed = "animation_changed";
	const StringName animation_started = "animation_started";
	const StringName RESET = "RESET";

	const StringName pose_updated = "pose_updated";
	const StringName skeleton_updated = "skeleton_updated";
	const StringName bone_enabled_changed = "bone_enabled_changed";
	const StringName show_rest_only_changed = "show_rest_only_changed";

	const StringName body_shape_entered = "body_shape_entered";
	const StringName body_entered = "body_entered";
	const StringName body_shape_exited = "body_shape_exited";
	const StringName body_exited = "body_exited";

	const StringName area_shape_entered = "area_shape_entered";
	const StringName area_shape_exited = "area_shape_exited";

	const StringName screen_entered = "screen_entered";
	const StringName screen_exited = "screen_exited";

	const StringName _canvas_item_editor_group = "_canvas_item_editor_group";
	const StringName _spatial_editor_group = "_spatial_editor_group";
	const StringName _request_gizmo = "_request_gizmo";

	const StringName offset = "offset";
	const StringName rotation_mode = "rotation_mode";
	const StringName rotate = "rotate";
	const StringName h_offset = "h_offset";
	const StringName v_offset = "v_offset";

	const StringName area_entered = "area_entered";
	const StringName area_exited = "area_exited";

	const StringName frame_changed = "frame_changed";
	const StringName texture_changed = "texture_changed";

	const StringName autoplay = "autoplay";
	const StringName blend_times = "blend_times";
	const StringName speed = "speed";

	const NodePath path_pp = NodePath("..");

	const StringName default_ = "default"; // default would conflict with C++ keyword.
	const StringName output = "output";

	const StringName Master = "Master"; // Audio bus name.

	const StringName theme_changed = "theme_changed";
	const StringName shader = "shader";
	const StringName shader_overrides_group = "_shader_overrides_group_";
	const StringName shader_overrides_group_active = "_shader_overrides_group_active_";

	const StringName _custom_type_script = "_custom_type_script";

	const StringName pressed = "pressed";
	const StringName id_pressed = "id_pressed";
	const StringName toggled = "toggled";
	const StringName hover = "hover";

	const StringName panel = "panel";
	const StringName item_selected = "item_selected";
	const StringName confirmed = "confirmed";

	const StringName text_changed = "text_changed";
	const StringName text_submitted = "text_submitted";
	const StringName value_changed = "value_changed";

	const StringName Start = "Start";
	const StringName End = "End";
	const StringName state_started = "state_started";
	const StringName state_finished = "state_finished";

	const StringName FlatButton = "FlatButton";
};

#define SceneStringName(m_name) SceneStringNames::get_singleton()->m_name
