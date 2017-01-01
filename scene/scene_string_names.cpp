/*************************************************************************/
/*  scene_string_names.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "scene_string_names.h"

SceneStringNames* SceneStringNames::singleton=NULL;

SceneStringNames::SceneStringNames() {

	resized=StaticCString::create("resized");
	dot=StaticCString::create(".");
	doubledot=StaticCString::create("..");
	draw=StaticCString::create("draw");
	_draw=StaticCString::create("_draw");
	hide=StaticCString::create("hide");
	visibility_changed=StaticCString::create("visibility_changed");
	input_event=StaticCString::create("input_event");
	shader_shader=StaticCString::create("shader/shader");
	shader_unshaded=StaticCString::create("shader/unshaded");
	shading_mode=StaticCString::create("shader/shading_mode");
	enter_tree=StaticCString::create("enter_tree");
	exit_tree=StaticCString::create("exit_tree");
	item_rect_changed=StaticCString::create("item_rect_changed");
	size_flags_changed=StaticCString::create("size_flags_changed");
	minimum_size_changed=StaticCString::create("minimum_size_changed");
	sleeping_state_changed=StaticCString::create("sleeping_state_changed");

	finished=StaticCString::create("finished");
	animation_changed=StaticCString::create("animation_changed");
	animation_started=StaticCString::create("animation_started");

	mouse_enter=StaticCString::create("mouse_enter");
	mouse_exit=StaticCString::create("mouse_exit");

	focus_enter=StaticCString::create("focus_enter");
	focus_exit=StaticCString::create("focus_exit");

	sort_children = StaticCString::create("sort_children");

	body_enter_shape = StaticCString::create("body_enter_shape");
	body_enter = StaticCString::create("body_enter");
	body_exit_shape = StaticCString::create("body_exit_shape");
	body_exit = StaticCString::create("body_exit");

	area_enter_shape = StaticCString::create("area_enter_shape");
	area_exit_shape = StaticCString::create("area_exit_shape");

	_body_inout = StaticCString::create("_body_inout");
	_area_inout = StaticCString::create("_area_inout");

	idle=StaticCString::create("idle");
	iteration=StaticCString::create("iteration");
	update=StaticCString::create("update");
	updated=StaticCString::create("updated");

	_get_gizmo_geometry=StaticCString::create("_get_gizmo_geometry");
	_can_gizmo_scale=StaticCString::create("_can_gizmo_scale");

	_fixed_process=StaticCString::create("_fixed_process");
	_process=StaticCString::create("_process");

	_enter_tree=StaticCString::create("_enter_tree");
	_exit_tree=StaticCString::create("_exit_tree");
	_enter_world=StaticCString::create("_enter_world");
	_exit_world=StaticCString::create("_exit_world");
	_ready=StaticCString::create("_ready");

	_update_scroll=StaticCString::create("_update_scroll");
	_update_xform=StaticCString::create("_update_xform");

	_proxgroup_add=StaticCString::create("_proxgroup_add");
	_proxgroup_remove=StaticCString::create("_proxgroup_remove");

	grouped=StaticCString::create("grouped");
	ungrouped=StaticCString::create("ungrouped");

	enter_screen=StaticCString::create("enter_screen");
	exit_screen=StaticCString::create("exit_screen");

	enter_viewport=StaticCString::create("enter_viewport");
	exit_viewport=StaticCString::create("exit_viewport");

	enter_camera=StaticCString::create("enter_camera");
	exit_camera=StaticCString::create("exit_camera");

	_body_enter_tree = StaticCString::create("_body_enter_tree");
	_body_exit_tree = StaticCString::create("_body_exit_tree");

	_area_enter_tree = StaticCString::create("_area_enter_tree");
	_area_exit_tree = StaticCString::create("_area_exit_tree");

	_input_event=StaticCString::create("_input_event");

	changed=StaticCString::create("changed");
	_shader_changed=StaticCString::create("_shader_changed");

	_spatial_editor_group=StaticCString::create("_spatial_editor_group");
	_request_gizmo=StaticCString::create("_request_gizmo");

	offset=StaticCString::create("offset");
	unit_offset=StaticCString::create("unit_offset");
	rotation_mode=StaticCString::create("rotation_mode");
	rotate=StaticCString::create("rotate");
	h_offset=StaticCString::create("h_offset");
	v_offset=StaticCString::create("v_offset");

	transform_pos=StaticCString::create("transform/pos");
	transform_rot=StaticCString::create("transform/rot");
	transform_scale=StaticCString::create("transform/scale");

	_update_remote=StaticCString::create("_update_remote");
	_update_pairs=StaticCString::create("_update_pairs");

	get_minimum_size=StaticCString::create("get_minimum_size");

	area_enter=StaticCString::create("area_enter");
	area_exit=StaticCString::create("area_exit");

	has_point = StaticCString::create("has_point");

	line_separation = StaticCString::create("line_separation");

	play_play = StaticCString::create("play/play");

	get_drag_data = StaticCString::create("get_drag_data");
	drop_data = StaticCString::create("drop_data");
	can_drop_data = StaticCString::create("can_drop_data");

	_im_update = StaticCString::create("_im_update");
	_queue_update = StaticCString::create("_queue_update");

	baked_light_changed = StaticCString::create("baked_light_changed");
	_baked_light_changed = StaticCString::create("_baked_light_changed");

	_mouse_enter=StaticCString::create("_mouse_enter");
	_mouse_exit=StaticCString::create("_mouse_exit");

	_pressed=StaticCString::create("_pressed");
	_toggled=StaticCString::create("_toggled");

	frame_changed=StaticCString::create("frame_changed");

	playback_speed=StaticCString::create("playback/speed");
	playback_active=StaticCString::create("playback/active");
	autoplay=StaticCString::create("autoplay");
	blend_times=StaticCString::create("blend_times");
	speed=StaticCString::create("speed");

	node_configuration_warning_changed = StaticCString::create("node_configuration_warning_changed");

	path_pp=NodePath("..");

	_default=StaticCString::create("default");

	for(int i=0;i<MAX_MATERIALS;i++) {

		mesh_materials[i]="material/"+itos(i);
	}

	_mesh_changed=StaticCString::create("_mesh_changed");
}
