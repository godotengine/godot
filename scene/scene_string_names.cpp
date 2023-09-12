/**************************************************************************/
/*  scene_string_names.cpp                                                */
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

#include "scene_string_names.h"

SceneStringNames *SceneStringNames::singleton = nullptr;

SceneStringNames::SceneStringNames() {
	_estimate_cost = StaticCString::create("_estimate_cost");
	_compute_cost = StaticCString::create("_compute_cost");

	resized = StaticCString::create("resized");
	dot = StaticCString::create(".");
	doubledot = StaticCString::create("..");
	draw = StaticCString::create("draw");
	_draw = StaticCString::create("_draw");
	hide = StaticCString::create("hide");
	visibility_changed = StaticCString::create("visibility_changed");
	input_event = StaticCString::create("input_event");
	shader = StaticCString::create("shader");
	shader_unshaded = StaticCString::create("shader/unshaded");
	shading_mode = StaticCString::create("shader/shading_mode");
	tree_entered = StaticCString::create("tree_entered");
	tree_exiting = StaticCString::create("tree_exiting");
	tree_exited = StaticCString::create("tree_exited");
	ready = StaticCString::create("ready");
	child_entered_tree = StaticCString::create("child_entered_tree");
	child_exiting_tree = StaticCString::create("child_exiting_tree");
	item_rect_changed = StaticCString::create("item_rect_changed");
	size_flags_changed = StaticCString::create("size_flags_changed");
	minimum_size_changed = StaticCString::create("minimum_size_changed");
	sleeping_state_changed = StaticCString::create("sleeping_state_changed");

	finished = StaticCString::create("finished");
	loop_finished = StaticCString::create("loop_finished");
	step_finished = StaticCString::create("step_finished");
	animation_finished = StaticCString::create("animation_finished");
	animation_changed = StaticCString::create("animation_changed");
	animation_started = StaticCString::create("animation_started");

	mouse_entered = StaticCString::create("mouse_entered");
	mouse_exited = StaticCString::create("mouse_exited");

	focus_entered = StaticCString::create("focus_entered");
	focus_exited = StaticCString::create("focus_exited");

	sort_children = StaticCString::create("sort_children");

	body_shape_entered = StaticCString::create("body_shape_entered");
	body_entered = StaticCString::create("body_entered");
	body_shape_exited = StaticCString::create("body_shape_exited");
	body_exited = StaticCString::create("body_exited");

	area_shape_entered = StaticCString::create("area_shape_entered");
	area_shape_exited = StaticCString::create("area_shape_exited");

	_body_inout = StaticCString::create("_body_inout");
	_area_inout = StaticCString::create("_area_inout");

	idle = StaticCString::create("idle");
	iteration = StaticCString::create("iteration");
	update = StaticCString::create("update");
	updated = StaticCString::create("updated");

	_physics_process = StaticCString::create("_physics_process");
	_process = StaticCString::create("_process");

	_enter_tree = StaticCString::create("_enter_tree");
	_exit_tree = StaticCString::create("_exit_tree");
	_enter_world = StaticCString::create("_enter_world");
	_exit_world = StaticCString::create("_exit_world");
	_ready = StaticCString::create("_ready");

	_update_scroll = StaticCString::create("_update_scroll");
	_update_xform = StaticCString::create("_update_xform");

	_clips_input = StaticCString::create("_clips_input");

	_proxgroup_add = StaticCString::create("_proxgroup_add");
	_proxgroup_remove = StaticCString::create("_proxgroup_remove");

	grouped = StaticCString::create("grouped");
	ungrouped = StaticCString::create("ungrouped");

	screen_entered = StaticCString::create("screen_entered");
	screen_exited = StaticCString::create("screen_exited");

	viewport_entered = StaticCString::create("viewport_entered");
	viewport_exited = StaticCString::create("viewport_exited");

	camera_entered = StaticCString::create("camera_entered");
	camera_exited = StaticCString::create("camera_exited");

	_body_enter_tree = StaticCString::create("_body_enter_tree");
	_body_exit_tree = StaticCString::create("_body_exit_tree");

	_area_enter_tree = StaticCString::create("_area_enter_tree");
	_area_exit_tree = StaticCString::create("_area_exit_tree");

	_input = StaticCString::create("_input");
	_input_event = StaticCString::create("_input_event");

	gui_input = StaticCString::create("gui_input");
	_gui_input = StaticCString::create("_gui_input");

	_unhandled_input = StaticCString::create("_unhandled_input");
	_unhandled_key_input = StaticCString::create("_unhandled_key_input");

	changed = StaticCString::create("changed");
	_shader_changed = StaticCString::create("_shader_changed");

	_spatial_editor_group = StaticCString::create("_spatial_editor_group");
	_request_gizmo = StaticCString::create("_request_gizmo");

	offset = StaticCString::create("offset");
	unit_offset = StaticCString::create("unit_offset");
	rotation_mode = StaticCString::create("rotation_mode");
	rotate = StaticCString::create("rotate");
	h_offset = StaticCString::create("h_offset");
	v_offset = StaticCString::create("v_offset");

	transform_pos = StaticCString::create("position");
	transform_rot = StaticCString::create("rotation_degrees");
	transform_scale = StaticCString::create("scale");

	_update_remote = StaticCString::create("_update_remote");
	_update_pairs = StaticCString::create("_update_pairs");

	_get_minimum_size = StaticCString::create("_get_minimum_size");

	area_entered = StaticCString::create("area_entered");
	area_exited = StaticCString::create("area_exited");

	has_point = StaticCString::create("has_point");

	line_separation = StaticCString::create("line_separation");

	get_drag_data = StaticCString::create("get_drag_data");
	drop_data = StaticCString::create("drop_data");
	can_drop_data = StaticCString::create("can_drop_data");

	_im_update = StaticCString::create("_im_update");
	_queue_update = StaticCString::create("_queue_update");

	baked_light_changed = StaticCString::create("baked_light_changed");
	_baked_light_changed = StaticCString::create("_baked_light_changed");

	_mouse_enter = StaticCString::create("_mouse_enter");
	_mouse_exit = StaticCString::create("_mouse_exit");

	_pressed = StaticCString::create("_pressed");
	_toggled = StaticCString::create("_toggled");

	frame_changed = StaticCString::create("frame_changed");

	playback_speed = StaticCString::create("playback/speed");
	playback_active = StaticCString::create("playback/active");
	autoplay = StaticCString::create("autoplay");
	blend_times = StaticCString::create("blend_times");
	speed = StaticCString::create("speed");

	node_configuration_warning_changed = StaticCString::create("node_configuration_warning_changed");

	output = StaticCString::create("output");

	path_pp = NodePath("..");

	_default = StaticCString::create("default");

	_mesh_changed = StaticCString::create("_mesh_changed");

	parameters_base_path = "parameters/";

	tracks_changed = "tracks_changed";
}
