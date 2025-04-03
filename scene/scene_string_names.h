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

	const StringName resized = StringName("resized");
	const StringName draw = StringName("draw");
	const StringName hidden = StringName("hidden");
	const StringName visibility_changed = StringName("visibility_changed");

	const StringName input_event = StringName("input_event");
	const StringName gui_input = StringName("gui_input");
	const StringName window_input = StringName("window_input");

	const StringName tree_entered = StringName("tree_entered");
	const StringName tree_exiting = StringName("tree_exiting");
	const StringName tree_exited = StringName("tree_exited");
	const StringName ready = StringName("ready");
	const StringName _ready = StringName("_ready");

	const StringName item_rect_changed = StringName("item_rect_changed");
	const StringName size_flags_changed = StringName("size_flags_changed");
	const StringName minimum_size_changed = StringName("minimum_size_changed");
	const StringName sleeping_state_changed = StringName("sleeping_state_changed");
	const StringName node_configuration_warning_changed = StringName("node_configuration_warning_changed");
	const StringName update = StringName("update");
	const StringName updated = StringName("updated");

	const StringName line_separation = StringName("line_separation");
	const StringName font = StringName("font");
	const StringName font_size = StringName("font_size");
	const StringName font_color = StringName("font_color");

	const StringName mouse_entered = StringName("mouse_entered");
	const StringName mouse_exited = StringName("mouse_exited");
	const StringName mouse_shape_entered = StringName("mouse_shape_entered");
	const StringName mouse_shape_exited = StringName("mouse_shape_exited");
	const StringName focus_entered = StringName("focus_entered");
	const StringName focus_exited = StringName("focus_exited");

	const StringName pre_sort_children = StringName("pre_sort_children");
	const StringName sort_children = StringName("sort_children");

	const StringName finished = StringName("finished");
	const StringName animation_finished = StringName("animation_finished");
	const StringName animation_changed = StringName("animation_changed");
	const StringName animation_started = StringName("animation_started");
	const StringName RESET = StringName("RESET");

	const StringName pose_updated = StringName("pose_updated");
	const StringName skeleton_updated = StringName("skeleton_updated");
	const StringName bone_enabled_changed = StringName("bone_enabled_changed");
	const StringName show_rest_only_changed = StringName("show_rest_only_changed");

	const StringName body_shape_entered = StringName("body_shape_entered");
	const StringName body_entered = StringName("body_entered");
	const StringName body_shape_exited = StringName("body_shape_exited");
	const StringName body_exited = StringName("body_exited");

	const StringName area_shape_entered = StringName("area_shape_entered");
	const StringName area_shape_exited = StringName("area_shape_exited");

	const StringName screen_entered = StringName("screen_entered");
	const StringName screen_exited = StringName("screen_exited");

	const StringName _spatial_editor_group = StringName("_spatial_editor_group");
	const StringName _request_gizmo = StringName("_request_gizmo");

	const StringName offset = StringName("offset");
	const StringName rotation_mode = StringName("rotation_mode");
	const StringName rotate = StringName("rotate");
	const StringName h_offset = StringName("h_offset");
	const StringName v_offset = StringName("v_offset");

	const StringName area_entered = StringName("area_entered");
	const StringName area_exited = StringName("area_exited");

	const StringName frame_changed = StringName("frame_changed");
	const StringName texture_changed = StringName("texture_changed");

	const StringName autoplay = StringName("autoplay");
	const StringName blend_times = StringName("blend_times");
	const StringName speed = StringName("speed");

	const NodePath path_pp = NodePath("..");

	const StringName default_ = StringName("default"); // default would conflict with C++ keyword.
	const StringName output = StringName("output");

	const StringName Master = StringName("Master"); // Audio bus name.

	const StringName theme_changed = StringName("theme_changed");
	const StringName shader = StringName("shader");
	const StringName shader_overrides_group = StringName("_shader_overrides_group_");
	const StringName shader_overrides_group_active = StringName("_shader_overrides_group_active_");

	const StringName _custom_type_script = StringName("_custom_type_script");

	const StringName pressed = StringName("pressed");
	const StringName id_pressed = StringName("id_pressed");
	const StringName toggled = StringName("toggled");
	const StringName hover = StringName("hover");

	const StringName panel = StringName("panel");
	const StringName item_selected = StringName("item_selected");
	const StringName confirmed = StringName("confirmed");

	const StringName text_changed = StringName("text_changed");
	const StringName text_submitted = StringName("text_submitted");
	const StringName value_changed = StringName("value_changed");

	const StringName Start = StringName("Start");
	const StringName End = StringName("End");

	const StringName FlatButton = StringName("FlatButton");
};

#define SceneStringName(m_name) SceneStringNames::get_singleton()->m_name
