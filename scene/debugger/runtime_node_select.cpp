/**************************************************************************/
/*  runtime_node_select.cpp                                               */
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

#ifdef DEBUG_ENABLED

#include "runtime_node_select.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/engine_debugger.h"
#include "core/input/input.h"
#include "core/input/input_event.h"
#include "core/math/geometry_3d.h"
#include "core/object/callable_mp.h"
#include "scene/2d/camera_2d.h"
#include "scene/debugger/scene_debugger_object.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/view_panner.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"
#include "scene/theme/theme_db.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"

#ifndef PHYSICS_2D_DISABLED
#include "scene/2d/physics/collision_shape_2d.h"
#endif // PHYSICS_2D_DISABLED

#ifndef _3D_DISABLED
#include "scene/3d/camera_3d.h"
#ifndef PHYSICS_3D_DISABLED
#include "scene/3d/physics/collision_object_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "servers/physics_3d/direct_states/physics_direct_space_state_3d.h"
#endif // PHYSICS_3D_DISABLED
#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/3d/convex_polygon_shape_3d.h"
#include "scene/resources/surface_tool.h"
#endif // _3D_DISABLED

RuntimeNodeSelect *RuntimeNodeSelect::get_singleton() {
	return singleton;
}

RuntimeNodeSelect::~RuntimeNodeSelect() {
	if (selection_list && !selection_list->is_visible()) {
		memdelete(selection_list);
	}

	if (draw_canvas.is_valid()) {
		RS::get_singleton()->free_rid(sel_drag_ci);
		RS::get_singleton()->free_rid(srect_ci);
		RS::get_singleton()->free_rid(draw_canvas);
	}
}

void RuntimeNodeSelect::_setup(const Dictionary &p_settings) {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(root->is_connected(SceneStringName(window_input), callable_mp(this, &RuntimeNodeSelect::_root_window_input)));

	root->connect(SceneStringName(window_input), callable_mp(this, &RuntimeNodeSelect::_root_window_input));
	root->connect("size_changed", callable_mp(this, &RuntimeNodeSelect::_queue_selection_update), CONNECT_DEFERRED);

	max_selection = p_settings.get("debugger/max_node_selection", 1);
	scale = GLOBAL_GET("display/window/stretch/scale");

	accent_color = p_settings.get("accent_color", Color());
	axis_x_color = p_settings.get("axis_x_color", Color());
	axis_y_color = p_settings.get("axis_y_color", Color());

	Ref<Theme> theme = ThemeDB::get_singleton()->get_default_theme();
	pivot_icon = theme->get_icon(SNAME("pivot"), SNAME("Debug"));
	resize_icon = theme->get_icon(SNAME("resize"), SNAME("Debug"));
	anchor_icon = theme->get_icon(SNAME("anchor"), SNAME("Debug"));

	// Panner Setup

	panner.instantiate();
	panner->set_callbacks(callable_mp(this, &RuntimeNodeSelect::_pan_callback), callable_mp(this, &RuntimeNodeSelect::_zoom_callback));

	ViewPanner::ControlScheme panning_scheme = (ViewPanner::ControlScheme)p_settings.get("editors/panning/2d_editor_panning_scheme", 0).operator int();
	bool simple_panning = p_settings.get("editors/panning/simple_panning", false);
	int pan_speed = p_settings.get("editors/panning/2d_editor_pan_speed", 20);
	Array keys = p_settings.get("canvas_item_editor/pan_view", Array()).operator Array();
	panner->setup(panning_scheme, DebuggerMarshalls::deserialize_key_shortcut(keys), simple_panning);
	panner->setup_warped_panning(root, p_settings.get("editors/panning/warped_mouse_panning", true));
	panner->set_scroll_speed(pan_speed);

	// CanvasItemManipulator Setup

	ci_manipulator.instantiate();
	ci_manipulator->set_show_transformation_gizmos(true);
	ci_manipulator->set_grab_distance(p_settings.get("editors/polygon_editor/point_grab_radius", 1));
	ci_manipulator->set_scale(scale);
	ci_manipulator->set_viewport(root);
	ci_manipulator->set_callbacks(
			callable_mp(this, &RuntimeNodeSelect::_find_ci_start_callback),
			callable_mp(this, &RuntimeNodeSelect::_point_selected_ci_callback),
			callable_mp(this, &RuntimeNodeSelect::_get_selection_ci_callback),
			callable_mp(this, &RuntimeNodeSelect::_local_transform_callback),
			callable_mp((Viewport *)root, &Viewport::get_screen_transform),
			callable_mp(this, &RuntimeNodeSelect::_local_mouse_pos_callback),
			callable_mp(this, &RuntimeNodeSelect::_plugin_input_callback));
	ci_manipulator->connect("box_selected", callable_mp(this, &RuntimeNodeSelect::_box_selected_ci));
	ci_manipulator->connect("box_selection_updated", callable_mp(this, &RuntimeNodeSelect::_set_selection_area));
	ci_manipulator->connect("clear_selection_requested", callable_mp(this, &RuntimeNodeSelect::_clear_selection).unbind(1).bind(true));
	ci_manipulator->connect("selection_menu_requested", callable_mp(this, &RuntimeNodeSelect::_open_selection_list));
	ci_manipulator->connect("scene_double_clicked", callable_mp(this, &RuntimeNodeSelect::_scene_double_clicked));
	ci_manipulator->connect("update_canvas_requested", callable_mp(this, &RuntimeNodeSelect::_queue_selection_update));
	ci_manipulator->connect("save_canvas_state_requested", callable_mp(this, &RuntimeNodeSelect::_save_canvas_state_requested));
	ci_manipulator->connect("restore_canvas_state_requested", callable_mp(this, &RuntimeNodeSelect::_restore_canvas_state_requested));
	ci_manipulator->connect("commit_canvas_state_requested", callable_mp(this, &RuntimeNodeSelect::_commit_canvas_state_requested));
	ci_manipulator->connect("cursor_shape_changed", callable_mp(this, &RuntimeNodeSelect::_update_cursor_shape));

#define SET_CI_SHORTCUT(p_name, p_setting) \
	{ \
		Ref<Shortcut> shortcut = DebuggerMarshalls::deserialize_key_shortcut(p_settings.get(p_setting, Array()).operator Array()); \
		if (shortcut.is_valid()) { \
			ci_manipulator->set_shortcut(p_name, shortcut); \
		} \
	}

	SET_CI_SHORTCUT(CanvasItemManipulator::SHORTCUT_CANCEL_TRANSFORM, "canvas_item_editor/cancel_transform");

#undef SET_CI_SHORTCUT

	/// 2D Selection Rectangle Generation

	sel_2d_scale = MAX(1, Math::ceil(2.0 / scale));

	selection_area_fill = p_settings.get("box_selection_fill_color", Color());
	selection_area_outline = p_settings.get("box_selection_stroke_color", Color());

	draw_canvas = RS::get_singleton()->canvas_create();
	sel_drag_ci = RS::get_singleton()->canvas_item_create();

	srect_color = p_settings.get("editors/2d/selection_rectangle_color", Color());
	srect_locked_color = p_settings.get("editors/2d/locked_selection_rectangle_color", Color());

	srect_ci = RS::get_singleton()->canvas_item_create();
	RS::get_singleton()->viewport_attach_canvas(root->get_viewport_rid(), draw_canvas);
	RS::get_singleton()->canvas_item_set_parent(sel_drag_ci, draw_canvas);
	RS::get_singleton()->canvas_item_set_parent(srect_ci, draw_canvas);

#ifndef _3D_DISABLED
	camera_fov = p_settings.get("editors/3d/default_fov", 70);
	camera_znear = p_settings.get("editors/3d/default_z_near", 0.05);
	camera_zfar = p_settings.get("editors/3d/default_z_far", 4'000);

	int freelook_mod_idx = p_settings.get("editors/3d/freelook/freelook_activation_modifier", 0);
	switch (freelook_mod_idx) {
		case 1: {
			freelook_modifier = Key::SHIFT;
		} break;
		case 2: {
			freelook_modifier = Key::ALT;
		} break;
		case 3: {
			freelook_modifier = Key::META;
		} break;
		case 4: {
			freelook_modifier = Key::CTRL;
		} break;
	}

	// View3DController Setup

	view_3d_controller.instantiate();

	view_3d_controller->set_freelook_scheme((View3DController::FreelookScheme)p_settings.get("editors/3d/freelook/freelook_navigation_scheme", View3DController::FREELOOK_DEFAULT).operator int());
	view_3d_controller->set_freelook_base_speed(p_settings.get("editors/3d/freelook/freelook_base_speed", 5));
	view_3d_controller->set_freelook_sensitivity(p_settings.get("editors/3d/freelook/freelook_sensitivity", 0.25));
	view_3d_controller->set_freelook_inertia(p_settings.get("editors/3d/freelook/freelook_inertia", 0));
	view_3d_controller->set_freelook_speed_zoom_link(p_settings.get("editors/3d/freelook/freelook_speed_zoom_link", false));
	view_3d_controller->set_freelook_invert_y_axis(p_settings.get("editors/3d/freelook/freelook_invert_y_axis", false));

	view_3d_controller->set_translation_sensitivity(p_settings.get("editors/3d/navigation_feel/translation_sensitivity", 1));
	view_3d_controller->set_translation_inertia(p_settings.get("editors/3d/navigation_feel/translation_inertia", 0));

	view_3d_controller->set_pan_mouse_button(p_settings.get("editors/3d/navigation/pan_mouse_button", View3DController::NAV_MOUSE_BUTTON_MIDDLE));

	view_3d_controller->set_orbit_mouse_button(p_settings.get("editors/3d/navigation/orbit_mouse_button", View3DController::NAV_MOUSE_BUTTON_MIDDLE));
	view_3d_controller->set_orbit_sensitivity(p_settings.get("editors/3d/navigation_feel/orbit_sensitivity", 0.004));
	view_3d_controller->set_orbit_inertia(p_settings.get("editors/3d/navigation_feel/orbit_inertia", 0));

	view_3d_controller->set_zoom_style(p_settings.get("editors/3d/navigation/zoom_style", View3DController::ZOOM_VERTICAL));
	view_3d_controller->set_zoom_inertia(p_settings.get("editors/3d/navigation_feel/zoom_inertia", 0));
	view_3d_controller->set_zoom_mouse_button(p_settings.get("editors/3d/navigation/zoom_mouse_button", View3DController::NAV_MOUSE_BUTTON_MIDDLE));

	view_3d_controller->set_angle_snap_threshold(p_settings.get("editors/3d/navigation_feel/angle_snap_threshold", 10));

	view_3d_controller->set_emulate_3_button_mouse(p_settings.get("editors/3d/navigation/emulate_3_button_mouse", false));
	view_3d_controller->set_emulate_numpad(p_settings.get("editors/3d/navigation/emulate_numpad", true));

	view_3d_controller->set_z_near(camera_znear);
	view_3d_controller->set_z_far(camera_zfar);

	view_3d_controller->set_invert_x_axis(p_settings.get("editors/3d/navigation/invert_x_axis", false));
	view_3d_controller->set_invert_y_axis(p_settings.get("editors/3d/navigation/invert_y_axis", false));

	view_3d_controller->set_warped_mouse_panning(p_settings.get("editors/3d/navigation/warped_mouse_panning", true));

	view_3d_controller->connect("fov_scaled", callable_mp(this, &RuntimeNodeSelect::_fov_scaled));
	view_3d_controller->connect("cursor_interpolated", callable_mp(this, &RuntimeNodeSelect::_cursor_interpolated));

	freelook_toggle = DebuggerMarshalls::deserialize_key_shortcut(p_settings.get("spatial_editor/freelook_toggle", Array()).operator Array());
	if (freelook_toggle.is_valid()) {
		for (Ref<InputEventKey> k : freelook_toggle->get_events()) {
			if (k->get_physical_keycode() == Key::NONE) {
				k->set_keycode(view_3d_controller->emulate_numpad_key(k->get_keycode()));
			} else {
				k->set_physical_keycode(view_3d_controller->emulate_numpad_key(k->get_physical_keycode()));
			}
		}
	}

#define SET_VIEW3D_SHORTCUT(p_name, p_setting) \
	{ \
		Ref<Shortcut> shortcut = DebuggerMarshalls::deserialize_key_shortcut(p_settings.get(p_setting, Array()).operator Array()); \
		if (shortcut.is_valid()) { \
			view_3d_controller->set_shortcut(p_name, shortcut); \
		} \
	}

	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FOV_DECREASE, "spatial_editor/decrease_fov");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FOV_INCREASE, "spatial_editor/increase_fov");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FOV_RESET, "spatial_editor/reset_fov");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_PAN_MOD_1, "spatial_editor/viewport_pan_modifier_1");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_PAN_MOD_2, "spatial_editor/viewport_pan_modifier_2");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_ORBIT_MOD_1, "spatial_editor/viewport_orbit_modifier_1");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_ORBIT_MOD_2, "spatial_editor/viewport_orbit_modifier_2");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_ORBIT_SNAP_MOD_1, "spatial_editor/viewport_orbit_snap_modifier_1");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_ORBIT_SNAP_MOD_2, "spatial_editor/viewport_orbit_snap_modifier_2");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_ZOOM_MOD_1, "spatial_editor/viewport_zoom_modifier_1");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_ZOOM_MOD_2, "spatial_editor/viewport_zoom_modifier_2");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_FORWARD, "spatial_editor/freelook_forward");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_BACKWARDS, "spatial_editor/freelook_backwards");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_LEFT, "spatial_editor/freelook_left");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_RIGHT, "spatial_editor/freelook_right");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_UP, "spatial_editor/freelook_up");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_DOWN, "spatial_editor/freelook_down");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_SPEED_MOD, "spatial_editor/freelook_speed_modifier");
	SET_VIEW3D_SHORTCUT(View3DController::SHORTCUT_FREELOOK_SLOW_MOD, "spatial_editor/freelook_slow_modifier");

#undef SET_VIEW3D_SHORTCUT

	/// 3D Selection Box Generation
	// Copied from the Node3DEditor implementation.

	sbox_color = p_settings.get("editors/3d/selection_box_color", Color());

	// Use two AABBs to create the illusion of a slightly thicker line.
	AABB aabb(Vector3(), Vector3(1, 1, 1));

	// Create a x-ray (visible through solid surfaces) and standard version of the selection box.
	// Both will be drawn at the same position, but with different opacity.
	// This lets the user see where the selection is while still having a sense of depth.
	Ref<SurfaceTool> st = memnew(SurfaceTool);
	Ref<SurfaceTool> st_xray = memnew(SurfaceTool);

	st->begin(Mesh::PRIMITIVE_LINES);
	st_xray->begin(Mesh::PRIMITIVE_LINES);
	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);

		st->add_vertex(a);
		st->add_vertex(b);
		st_xray->add_vertex(a);
		st_xray->add_vertex(b);
	}

	Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	mat->set_albedo(sbox_color);
	mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st->set_material(mat);
	sbox_mesh = st->commit();

	Ref<StandardMaterial3D> mat_xray = memnew(StandardMaterial3D);
	mat_xray->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	mat_xray->set_albedo(sbox_color * Color(1, 1, 1, 0.15));
	mat_xray->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st_xray->set_material(mat_xray);
	sbox_mesh_xray = st_xray->commit();
#endif // _3D_DISABLED

	SceneTree::get_singleton()->connect("process_frame", callable_mp(this, &RuntimeNodeSelect::_process_frame));
#ifndef _3D_DISABLED
	SceneTree::get_singleton()->connect("physics_frame", callable_mp(this, &RuntimeNodeSelect::_physics_frame));
#endif

	// This function will be called before the root enters the tree at first when the Game view is passing its settings to
	// the debugger, so queue the update for after it enters.
	root->connect(SceneStringName(tree_entered), callable_mp(this, &RuntimeNodeSelect::_update_input_state), Object::CONNECT_ONE_SHOT);
}

void RuntimeNodeSelect::_set_node_type(NodeType p_type) {
	node_select_type = p_type;
	_update_input_state();
}

void RuntimeNodeSelect::_set_camera_override_enabled(bool p_enabled) {
	camera_override = p_enabled;

	if (camera_first_override) {
		_reset_camera_2d();
#ifndef _3D_DISABLED
		_reset_camera_3d();
#endif // _3D_DISABLED

		camera_first_override = false;
	} else if (p_enabled) {
		_update_view_2d();

#ifndef _3D_DISABLED
		Window *root = SceneTree::get_singleton()->get_root();
		ERR_FAIL_COND(!root->is_camera_3d_override_enabled());
		Camera3D *override_camera = root->get_override_camera_3d();
		override_camera->set_transform(view_3d_controller->to_camera_transform());
		override_camera->set_perspective(camera_fov * view_3d_controller->cursor.fov_scale, camera_znear, camera_zfar);
#endif // _3D_DISABLED
	}
}

void RuntimeNodeSelect::_set_ci_tool(CanvasItemManipulator::Tool p_tool) {
	ci_manipulator->set_tool(p_tool);
	panner->set_force_drag(p_tool == CanvasItemManipulator::TOOL_PAN);
	_queue_selection_update();
}

void RuntimeNodeSelect::_set_ci_local_space(bool p_enabled) {
	ci_manipulator->set_local_space_enabled(p_enabled);
	_queue_selection_update();
}

void RuntimeNodeSelect::_set_ci_smart_snap(bool p_enabled) {
	ci_manipulator->set_smart_snap_enabled(p_enabled);
}

void RuntimeNodeSelect::_set_n3d_tool(SelectMode p_tool) {
	node_select_mode = p_tool;
}

Variant RuntimeNodeSelect::_find_ci_start_callback() const {
	return Object::cast_to<Node>(SceneTree::get_singleton()->get_root());
}

bool RuntimeNodeSelect::_point_selected_ci_callback(const Variant &p_node, bool p_append) {
	Node *node = Object::cast_to<Node>(p_node);
	if (p_append || !selected_ci_nodes.has(node->get_instance_id())) {
		_send_ids({ node }, p_append, true);
	}
	return selected_ci_nodes.has(node->get_instance_id());
}

bool RuntimeNodeSelect::_get_selection_ci_callback(Array r_selection) {
	for (const KeyValue<ObjectID, Dictionary> &kv : selected_ci_nodes) {
		Object *obj = ObjectDB::get_instance(kv.key);
		if (obj) {
			r_selection.append(obj);
		}
	}
	return false;
}

Point2 RuntimeNodeSelect::_local_mouse_pos_callback() const {
	Window *root = SceneTree::get_singleton()->get_root();
	return root->get_screen_transform().affine_inverse().xform(root->get_mouse_position());
}

void RuntimeNodeSelect::_root_window_input(const Ref<InputEvent> &p_event) {
	Window *root = SceneTree::get_singleton()->get_root();
	if (node_select_type == NODE_TYPE_NONE || (selection_list && selection_list->is_visible())) {
		// Workaround for platforms that don't allow subwindows.
		if (selection_list && selection_list->is_visible() && selection_list->is_embedded()) {
			root->set_disable_input_override(false);
			selection_list->push_input(p_event);
			callable_mp(root->get_viewport(), &Viewport::set_disable_input_override).call_deferred(true);
		}

		return;
	}

	if (SceneTree::get_singleton()->is_suspended() && ci_manipulator->is_showing_transformation_gizmos()) {
		Ref<InputEventKey> k = p_event;
		if (k.is_valid() && !k->is_echo() && (k->get_keycode() == Key::CTRL || k->get_keycode() == Key::ALT || k->get_keycode() == Key::SHIFT)) {
			_queue_selection_update();
		}
	}

	_update_cursor_shape();

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid() && b->is_pressed()) {
		list_shortcut_pressed = node_select_mode == SELECT_MODE_SINGLE && b->get_button_index() == MouseButton::RIGHT && b->is_alt_pressed();
	}

	bool is_dragging_camera = false;
	if (camera_override) {
		if (node_select_type == NODE_TYPE_2D) {
			is_dragging_camera = panner->gui_input(p_event, Rect2(Vector2(), root->get_visible_rect().get_size()));

			if (is_panning != panner->is_panning()) {
				is_panning = panner->is_panning();
				_update_cursor_shape();
			}
#ifndef _3D_DISABLED
		} else if (!list_shortcut_pressed && node_select_type == NODE_TYPE_3D && selection_drag_state == SELECTION_DRAG_NONE) {
			if (_handle_3d_input(p_event)) {
				return;
			}
#endif // _3D_DISABLED
		}
	}

	if (selection_drag_state == SELECTION_DRAG_MOVE) {
		Ref<InputEventMouseMotion> m = p_event;
		if (m.is_valid()) {
			_update_selection_drag(root->get_screen_transform().affine_inverse().xform(m->get_position()));
			return;
		} else if (b.is_valid()) {
			// Account for actions like zooming.
			_update_selection_drag(root->get_screen_transform().affine_inverse().xform(b->get_position()));
		}
	}

	if (b.is_valid()) {
		if (node_select_type == NODE_TYPE_3D) {
			if (selection_drag_state == SELECTION_DRAG_MOVE && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT) {
				selection_drag_state = SELECTION_DRAG_END;
				selection_drag_area = selection_drag_area.abs();
				_update_selection_drag();

				// Trigger a selection in the position on release.
				if (multi_shortcut_pressed) {
					selection_position = root->get_screen_transform().affine_inverse().xform(b->get_position());
				}
			}
		}

		if (!is_dragging_camera && b->is_pressed()) {
			if (b->get_button_index() != MouseButton::NONE) {
				multi_shortcut_pressed = b->is_shift_pressed();
			}

			if (node_select_type == NODE_TYPE_3D) {
				if (list_shortcut_pressed || b->get_button_index() == MouseButton::LEFT) {
					selection_position = root->get_screen_transform().affine_inverse().xform(b->get_position());
				}
			}
		}
	}

	if (node_select_type == NODE_TYPE_2D) {
		Input *input = Input::get_singleton();
		bool was_input_disabled = input->is_input_disabled();
		if (was_input_disabled) {
			input->set_disable_input(false);
		}

		if (!ci_manipulator->gui_input(p_event)) {
			// Remind the user to enable the camera override when panning.
			if (!camera_override && ci_manipulator->get_tool() == CanvasItemManipulator::TOOL_PAN && b.is_valid() && b->is_pressed()) {
				const String msg = RTR("Camera override must be enabled to allow panning.");
				EngineDebugger::get_singleton()->send_message("game_view:show_toaster", { msg, 0 });
			}
		}

		if (was_input_disabled) {
			input->set_disable_input(true);
		}
	}
}

void RuntimeNodeSelect::_items_popup_index_pressed(int p_index, PopupMenu *p_popup) {
	Object *obj = p_popup->get_item_metadata(p_index).get_validated_object();
	if (obj) {
		Vector<Node *> node;
		node.append(Object::cast_to<Node>(obj));
		_send_ids(node, selection_list_appending, true);
	}
}

void RuntimeNodeSelect::_update_input_state() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	// This function can be called at the very beginning, when the root hasn't entered the tree yet.
	// So check first to avoid a crash.
	if (!scene_tree->get_root()->is_inside_tree()) {
		return;
	}

	bool disable_input = scene_tree->is_suspended() || node_select_type != RuntimeNodeSelect::NODE_TYPE_NONE;
	Input::get_singleton()->set_disable_input(disable_input);
	Input::get_singleton()->set_mouse_mode_override_enabled(disable_input);
	scene_tree->get_root()->set_disable_input_override(disable_input);
}

void RuntimeNodeSelect::_update_cursor_shape() {
	if (!DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_CURSOR_SHAPE)) {
		return;
	}

	// Choose the correct default cursor.
	DisplayServerEnums::CursorShape c = DisplayServerEnums::CURSOR_ARROW;
	if (panner->is_panning()) {
		c = DisplayServerEnums::CursorShape::CURSOR_DRAG;
	} else {
		c = (DisplayServerEnums::CursorShape)ci_manipulator->get_cursor_shape();
	}

	// Since this is only used when inputs are disabled, just change the shape directly.
	DisplayServer::get_singleton()->cursor_set_shape(c);
}

void RuntimeNodeSelect::_show_limit_warning() {
	const String msg = vformat(RTR("Some remote nodes were not selected, as the configured maximum selection is %d. This can be changed at \"debugger/max_node_selection\" in the Editor Settings."), max_selection);
	EngineDebugger::get_singleton()->send_message("game_view:show_toaster", { msg, 1 });
}

void RuntimeNodeSelect::_scene_double_clicked(const String &p_path) {
	EngineDebugger::get_singleton()->send_message("game_view:open_scene", { p_path });
}

void RuntimeNodeSelect::_process_frame() {
#ifndef _3D_DISABLED
	// Calculate the process time manually, as the time scale can be frozen.
	const double process_time = (1.0 / Engine::get_singleton()->get_frames_per_second());

	if (view_3d_controller->is_freelook_enabled()) {
		Input *input = Input::get_singleton();
		bool was_input_disabled = input->is_input_disabled();
		if (was_input_disabled) {
			input->set_disable_input(false);
		}

		view_3d_controller->update_freelook(process_time);

		if (was_input_disabled) {
			input->set_disable_input(true);
		}
	}

	view_3d_controller->update_camera(process_time);
#endif // _3D_DISABLED

	if (selection_update_queued || !SceneTree::get_singleton()->is_suspended()) {
		selection_update_queued = false;
		if (has_selection && selection_visible) {
			_update_selection();
		}
	}
}

#ifndef _3D_DISABLED
void RuntimeNodeSelect::_physics_frame() {
	if (node_select_type != NODE_TYPE_3D) {
		return;
	}

	if (selection_drag_state != SELECTION_DRAG_END && (selection_drag_state == SELECTION_DRAG_MOVE || !selection_position.is_finite())) {
		return;
	}

	Window *root = SceneTree::get_singleton()->get_root();
	bool selection_drag_valid = selection_drag_state == SELECTION_DRAG_END && selection_drag_area.get_area() > SELECTION_MIN_AREA;
	Vector<DebuggerHelpers::SelectResult> items;

	if (selection_drag_valid) {
		_find_3d_items_at_rect(selection_drag_area, items);
	} else if (selection_position.is_finite()) {
		_find_3d_items_at_pos(selection_position, items);
	}

	if ((prefer_group_selection || avoid_locked_nodes) && !list_shortcut_pressed && node_select_mode == SELECT_MODE_SINGLE) {
		for (int i = 0; i < items.size(); i++) {
			Node *node = items[i].item;
			Node *final_node = node;
			real_t order = items[i].order;

			// Replace the node by the group if grouped.
			if (prefer_group_selection) {
				while (node && node != root) {
					if (node->has_meta("_edit_group_")) {
						final_node = node;

						if (Object::cast_to<Node3D>(final_node)) {
							Node3D *node3d_tmp = Object::cast_to<Node3D>(final_node);
							Camera3D *camera = root->get_camera_3d();
							Vector3 pos = camera->project_ray_origin(selection_position);
							order = -pos.distance_to(node3d_tmp->get_global_transform().origin);
						}
					}
					node = node->get_parent();
				}
			}

			// Filter out locked nodes.
			if (avoid_locked_nodes && ci_manipulator->is_node_locked(final_node)) {
				items.remove_at(i);
				i--;
				continue;
			}

			items.write[i].item = final_node;
			items.write[i].order = order;
		}
	}

	// Remove possible duplicates.
	for (int i = 0; i < items.size(); i++) {
		Node *item = items[i].item;
		for (int j = 0; j < i; j++) {
			if (items[j].item == item) {
				items.remove_at(i);
				i--;

				break;
			}
		}
	}

	items.sort();

	switch (selection_drag_state) {
		case SELECTION_DRAG_END: {
			selection_position = Point2(Math::INF, Math::INF);
			selection_drag_state = SELECTION_DRAG_NONE;

			if (selection_drag_area.get_area() > SELECTION_MIN_AREA) {
				if (!items.is_empty()) {
					Vector<Node *> nodes;
					for (const DebuggerHelpers::SelectResult item : items) {
						nodes.append(item.item);
					}
					_send_ids(nodes, multi_shortcut_pressed);
				}

				_update_selection_drag();
				return;
			}

			_update_selection_drag();
		} break;

		case SELECTION_DRAG_NONE: {
			if (list_shortcut_pressed || node_select_mode == SELECT_MODE_LIST) {
				break;
			}

			if (multi_shortcut_pressed) {
				// Allow forcing box selection when an item was clicked.
				selection_drag_state = SELECTION_DRAG_MOVE;
			} else if (items.is_empty()) {
				if (!selected_ci_nodes.is_empty() || !selected_3d_nodes.is_empty()) {
					_clear_selection();
				}

				selection_drag_state = SELECTION_DRAG_MOVE;
			} else {
				break;
			}

			[[fallthrough]];
		}

		case SELECTION_DRAG_MOVE: {
			selection_drag_area.position = selection_position;

			// Stop selection on click, so it can happen on release if the selection area doesn't pass the threshold.
			if (multi_shortcut_pressed) {
				return;
			}
		}
	}

	if (items.is_empty()) {
		selection_position = Point2(Math::INF, Math::INF);
		return;
	}

	if (items.size() == 1 || (!list_shortcut_pressed && node_select_mode == SELECT_MODE_SINGLE)) {
		selection_position = Point2(Math::INF, Math::INF);

		Vector<Node *> node;
		node.append(items[0].item);
		_send_ids(node, multi_shortcut_pressed, true);

		return;
	}

	if (!selection_list && (list_shortcut_pressed || node_select_mode == SELECT_MODE_LIST)) {
		Array nodes;
		for (const DebuggerHelpers::SelectResult item : items) {
			nodes.append(item.item);
		}
		_open_selection_list(nodes, selection_position, multi_shortcut_pressed);
	}

	selection_position = Point2(Math::INF, Math::INF);
}
#endif // _3D_DISABLED

void RuntimeNodeSelect::_send_ids(const Vector<Node *> &p_picked_nodes, bool p_append, bool p_invert) {
	ERR_FAIL_COND(p_picked_nodes.is_empty());

	Vector<Node *> picked_nodes = p_picked_nodes;
	Array message;

	if (!p_append) {
		if (picked_nodes.size() > max_selection) {
			picked_nodes.resize(max_selection);
			_show_limit_warning();
		}

		Array fuck;
		for (const Node *node : picked_nodes) {
			SceneDebuggerObject obj(node->get_instance_id());
			Array arr;
			fuck.append(node);
			obj.serialize(arr);
			message.append(arr);
		}

		EngineDebugger::get_singleton()->send_message("remote_objects_selected", message);
		_set_selected_nodes(picked_nodes);

		return;
	}

	LocalVector<ObjectID> ids;
	LocalVector<Node *> nodes;

	for (const KeyValue<ObjectID, Dictionary> &kv : selected_ci_nodes) {
		ids.push_back(kv.key);
		nodes.push_back(ObjectDB::get_instance<Node>(kv.key));
	}
#ifndef _3D_DISABLED
	for (const KeyValue<ObjectID, Ref<SelectionBox>> &kv : selected_3d_nodes) {
		ids.push_back(kv.key);
		nodes.push_back(ObjectDB::get_instance<Node>(kv.key));
	}
#endif // _3D_DISABLED

	for (Node *node : picked_nodes) {
		ObjectID id = node->get_instance_id();
		if (ids.has(id)) {
			if (p_append && p_invert) {
				ids.erase(id);
				nodes.erase(node);
			}
		} else {
			ids.push_back(id);
			nodes.push_back(node);
		}
	}

	if ((int)ids.size() > max_selection) {
		ids.resize(max_selection);
		_show_limit_warning();
	}

	if (ids.is_empty()) {
		EngineDebugger::get_singleton()->send_message("remote_nothing_selected", message);
	} else {
		for (const ObjectID id : ids) {
			SceneDebuggerObject obj(id);
			Array arr;
			obj.serialize(arr);
			message.append(arr);
		}

		EngineDebugger::get_singleton()->send_message("remote_objects_selected", message);
	}

	_set_selected_nodes(Vector<Node *>(nodes));
}

void RuntimeNodeSelect::_set_selected_nodes(const Vector<Node *> &p_nodes) {
	if (p_nodes.is_empty()) {
		_clear_selection();
		return;
	}

	bool changed = false;
	HashMap<ObjectID, Dictionary> nodes_ci;
#ifndef _3D_DISABLED
	HashMap<ObjectID, Ref<SelectionBox>> nodes_3d;
#endif // _3D_DISABLED

	for (Node *node : p_nodes) {
		ObjectID id = node->get_instance_id();
		if (Object::cast_to<CanvasItem>(node)) {
			if (!changed || !selected_ci_nodes.has(id)) {
				changed = true;
			}

			nodes_ci.insert(id, Dictionary());
		} else {
#ifndef _3D_DISABLED
			Node3D *node_3d = Object::cast_to<Node3D>(node);
			if (!node_3d || !node_3d->is_inside_world()) {
				continue;
			}

			if (!changed || !selected_3d_nodes.has(id)) {
				changed = true;
			}

			if (selected_3d_nodes.has(id)) {
				// Assign an already available visual instance.
				nodes_3d[id] = selected_3d_nodes[id];
				continue;
			}

			if (sbox_mesh.is_null() || sbox_mesh_xray.is_null()) {
				continue;
			}

			Ref<SelectionBox> sb;
			sb.instantiate();
			nodes_3d[id] = sb;

			RID scenario = node_3d->get_world_3d()->get_scenario();

			sb->instance = RS::get_singleton()->instance_create2(sbox_mesh->get_rid(), scenario);
			sb->instance_ofs = RS::get_singleton()->instance_create2(sbox_mesh->get_rid(), scenario);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance, RSE::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance_ofs, RSE::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_ofs, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_ofs, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

			sb->instance_xray = RS::get_singleton()->instance_create2(sbox_mesh_xray->get_rid(), scenario);
			sb->instance_xray_ofs = RS::get_singleton()->instance_create2(sbox_mesh_xray->get_rid(), scenario);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance_xray, RSE::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance_xray_ofs, RSE::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray_ofs, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray_ofs, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
#endif // _3D_DISABLED
		}
	}

#ifdef _3D_DISABLED
	if (!changed && nodes_ci.size() == selected_ci_nodes.size()) {
		return;
	}
#else
	if (!changed && nodes_ci.size() == selected_ci_nodes.size() && nodes_3d.size() == selected_3d_nodes.size()) {
		return;
	}
#endif // _3D_DISABLED

	_clear_selection(false);

	selected_ci_nodes = nodes_ci;
	has_selection = !nodes_ci.is_empty();

#ifndef _3D_DISABLED
	if (!nodes_3d.is_empty()) {
		selected_3d_nodes = nodes_3d;
		has_selection = true;
	}
#endif // _3D_DISABLED

	_queue_selection_update();
}

void RuntimeNodeSelect::_queue_selection_update() {
	if (has_selection && selection_visible) {
		if (SceneTree::get_singleton()->is_suspended()) {
			_update_selection();
		} else {
			selection_update_queued = true;
		}
	}
}

void RuntimeNodeSelect::_update_selection() {
	RS::get_singleton()->canvas_item_clear(srect_ci);
	RS::get_singleton()->canvas_item_set_visible(srect_ci, selection_visible);

	CanvasItemManipulator::Tool tool = ci_manipulator->get_tool();
	bool transform_tool = tool == CanvasItemManipulator::TOOL_SELECT ||
			tool == CanvasItemManipulator::TOOL_MOVE ||
			tool == CanvasItemManipulator::TOOL_SCALE ||
			tool == CanvasItemManipulator::TOOL_ROTATE ||
			tool == CanvasItemManipulator::TOOL_EDIT_PIVOT;
	CanvasItemManipulator::DragType drag = ci_manipulator->get_drag_type();
	bool single = selected_ci_nodes.size() == 1;

	for (HashMap<ObjectID, Dictionary>::ConstIterator kv = selected_ci_nodes.begin(); kv != selected_ci_nodes.end(); ++kv) {
		CanvasItem *ci = ObjectDB::get_instance<CanvasItem>(kv->key);
		if (!ci) {
			selected_ci_nodes.erase(kv->key);
			--kv;
			continue;
		}

		if (!ci->is_inside_tree()) {
			continue;
		}

		Transform2D xform = ci->get_global_transform_with_canvas();
		bool item_locked = ci_manipulator->is_node_locked(ci);

		if (single && node_select_type == NODE_TYPE_2D) {
			// Pivot

			if (transform_tool && !item_locked && ci->_edit_use_pivot()) {
				Transform2D xform_unscaled = (xform * ci->get_transform().affine_inverse() * ci->_edit_get_transform()).orthonormalized();
				Transform2D simple_xform;
				if (ci_manipulator->is_local_space_enabled()) {
					simple_xform *= xform_unscaled;
				} else {
					simple_xform *= Transform2D(0.0f, xform_unscaled.get_origin());
				}

				Size2 pivot_size = pivot_icon->get_size() * scale;
				Rect2 rect(simple_xform.get_origin() - (pivot_size / 2.0).floor(), pivot_size);
				RS::get_singleton()->canvas_item_add_texture_rect(srect_ci, rect, pivot_icon->get_rid());
			}

			if (tool == CanvasItemManipulator::TOOL_SELECT) {
				// Anchors

				Control *control = Object::cast_to<Control>(ci);
				if (control && ci_manipulator->is_node_movable(ci)) {
					// Compute the anchors.
					real_t anchors_values[4];
					anchors_values[0] = control->get_anchor(SIDE_LEFT);
					anchors_values[1] = control->get_anchor(SIDE_TOP);
					anchors_values[2] = control->get_anchor(SIDE_RIGHT);
					anchors_values[3] = control->get_anchor(SIDE_BOTTOM);

					Vector2 anchors_pos[4];
					for (int i = 0; i < 4; i++) {
						Vector2 value = Vector2((i % 2 == 0) ? anchors_values[i] : anchors_values[(i + 1) % 4], (i % 2 == 1) ? anchors_values[i] : anchors_values[(i + 1) % 4]);
						anchors_pos[i] = xform.xform(ci_manipulator->anchor_to_position(control, value));
					}

					// Draw the anchors handles.
					Size2 handle_size = CanvasItemManipulator::ANCHOR_HANDLE_SIZE * scale;
					Rect2 anchor_rects[4];
					if (control->is_layout_rtl()) {
						anchor_rects[0] = Rect2(anchors_pos[0] - Vector2(0.0, handle_size.y), Point2(-handle_size.x, handle_size.y));
						anchor_rects[1] = Rect2(anchors_pos[1] - handle_size, handle_size);
						anchor_rects[2] = Rect2(anchors_pos[2] - Vector2(handle_size.x, 0.0), Point2(handle_size.x, -handle_size.y));
						anchor_rects[3] = Rect2(anchors_pos[3], -handle_size);
					} else {
						anchor_rects[0] = Rect2(anchors_pos[0] - handle_size, handle_size);
						anchor_rects[1] = Rect2(anchors_pos[1] - Vector2(0.0, handle_size.y), Point2(-handle_size.x, handle_size.y));
						anchor_rects[2] = Rect2(anchors_pos[2], -handle_size);
						anchor_rects[3] = Rect2(anchors_pos[3] - Vector2(handle_size.x, 0.0), Point2(handle_size.x, -handle_size.y));
					}

					for (int i = 0; i < 4; i++) {
						RS::get_singleton()->canvas_item_add_texture_rect(srect_ci, anchor_rects[i], anchor_icon->get_rid());
					}
				}

				// Resize

				if (ci->_edit_use_rect() && ci_manipulator->is_node_movable(ci)) {
					Rect2 rect = ci->_edit_get_rect();
					const Point2 endpoints[4] = {
						xform.xform(rect.position),
						xform.xform(rect.position + Point2(rect.size.x, 0)),
						xform.xform(rect.position + rect.size),
						xform.xform(rect.position + Point2(0, rect.size.y))
					};

					Size2 handle_size = Size2(CanvasItemManipulator::RESIZE_HANDLE_DIAMETER, CanvasItemManipulator::RESIZE_HANDLE_DIAMETER) * scale;
					for (int i = 0; i < 4; i++) {
						int prev = (i + 3) % 4;
						int next = (i + 1) % 4;

						Point2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
						ofs *= Math::SQRT2 * (handle_size.width / 2);
						Rect2 icon_rect((endpoints[i] + ofs - (handle_size / 2)).floor(), handle_size);

						RS::get_singleton()->canvas_item_add_texture_rect(srect_ci, icon_rect, resize_icon->get_rid());

						ofs = (endpoints[i] + endpoints[next]) / 2;
						ofs += (endpoints[next] - endpoints[i]).orthogonal().normalized() * (handle_size.width / 2);
						icon_rect.position = (ofs - (handle_size / 2)).floor();

						RS::get_singleton()->canvas_item_add_texture_rect(srect_ci, icon_rect, resize_icon->get_rid());
					}
				}
			}
		}

		// 2D Selection Rectangle

		// Default fallback.
		Rect2 rect = Rect2(Vector2(), Vector2(10, 10));

		if (ci->_edit_use_rect()) {
			rect = ci->_edit_get_rect();
		} else {
#ifndef PHYSICS_2D_DISABLED
			CollisionShape2D *collision_shape = Object::cast_to<CollisionShape2D>(ci);
			if (collision_shape) {
				Ref<Shape2D> shape = collision_shape->get_shape();
				if (shape.is_valid()) {
					rect = shape->get_rect();
				}
			}
#endif // PHYSICS_2D_DISABLED
		}

		Vector2 endpoints[4] = {
			xform.xform(rect.position),
			xform.xform(rect.position + Point2(rect.size.x, 0)),
			xform.xform(rect.position + rect.size),
			xform.xform(rect.position + Point2(0, rect.size.y))
		};

		Color color = item_locked ? srect_locked_color : srect_color;
		for (int i = 0; i < 4; i++) {
			RS::get_singleton()->canvas_item_add_line(srect_ci, endpoints[i], endpoints[(i + 1) % 4], color, sel_2d_scale);
		}
	}

	if (node_select_type == NODE_TYPE_2D) {
		Point2 temp_pivot = ci_manipulator->get_temp_pivot();

		if (!selected_ci_nodes.is_empty() && ci_manipulator->is_showing_transformation_gizmos()) {
			CanvasItem *ci = nullptr;

			// Find the first movable node.
			for (const KeyValue<ObjectID, Dictionary> &kv : selected_ci_nodes) {
				ci = ObjectDB::get_instance<CanvasItem>(kv.key);
				if (ci_manipulator->is_node_movable(ci)) {
					break;
				}
				ci = nullptr;
			}

			if (ci) {
				Input *input = Input::get_singleton();
				bool was_input_disabled = input->is_input_disabled();
				if (was_input_disabled) {
					input->set_disable_input(false);
				}

				bool is_ctrl = Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL);
				bool is_alt = Input::get_singleton()->is_key_pressed(Key::ALT);
				bool is_moving = tool == CanvasItemManipulator::TOOL_MOVE || (tool == CanvasItemManipulator::TOOL_SELECT && is_alt && !is_ctrl);

				Transform2D edit_xform;
				if (!is_moving && !Math::is_inf(temp_pivot.x) && !Math::is_inf(temp_pivot.y)) {
					edit_xform = Transform2D(ci->_edit_get_rotation(), temp_pivot);
				} else {
					edit_xform = ci->_edit_get_transform();
				}

				Transform2D xform_unscaled = (ci->get_global_transform_with_canvas() * ci->get_transform().affine_inverse() * edit_xform).orthonormalized();
				Transform2D simple_xform;
				if (ci_manipulator->is_local_space_enabled()) {
					simple_xform *= xform_unscaled;
				} else {
					simple_xform *= Transform2D(0.0f, xform_unscaled.get_origin());
				}

				RS::get_singleton()->canvas_item_add_set_transform(srect_ci, simple_xform);

				// Move Handles
				if (is_moving) {
					Vector<Point2> points = {
						Point2(CanvasItemManipulator::GIZMO_HANDLE_X_RECT.position.x, CanvasItemManipulator::GIZMO_HANDLE_X_RECT.size.height / 2.0),
						Point2(CanvasItemManipulator::GIZMO_HANDLE_X_RECT.position.x, -CanvasItemManipulator::GIZMO_HANDLE_X_RECT.size.height / 2.0),
						Point2(CanvasItemManipulator::GIZMO_HANDLE_X_RECT.position.x + CanvasItemManipulator::GIZMO_HANDLE_X_RECT.size.width, 0)
					};
					Vector<Color> colors = { axis_x_color };

					RS::get_singleton()->canvas_item_add_polygon(srect_ci, points, colors);
					RS::get_singleton()->canvas_item_add_line(srect_ci, Point2(), Point2(CanvasItemManipulator::GIZMO_HANDLE_DISTANCE, 0), axis_x_color, scale);

					points = {
						Point2(CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.size.width / 2.0, CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.position.y),
						Point2(-CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.size.width / 2.0, CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.position.y),
						Point2(0, CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.position.y + CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.size.height)
					};
					colors = { axis_y_color };

					RS::get_singleton()->canvas_item_add_polygon(srect_ci, points, colors);
					RS::get_singleton()->canvas_item_add_line(srect_ci, Point2(), Point2(0, CanvasItemManipulator::GIZMO_HANDLE_DISTANCE), axis_y_color, scale);
				}

				// Scale Handles
				if (tool == CanvasItemManipulator::TOOL_SCALE || drag == CanvasItemManipulator::DRAG_SCALE_X || drag == CanvasItemManipulator::DRAG_SCALE_Y || (tool == CanvasItemManipulator::TOOL_SELECT && is_alt && is_ctrl)) {
					Size2 scale_factor(CanvasItemManipulator::GIZMO_HANDLE_DISTANCE, CanvasItemManipulator::GIZMO_HANDLE_DISTANCE);
					bool uniform = Input::get_singleton()->is_key_pressed(Key::SHIFT);
					Point2 offset = simple_xform.affine_inverse().xform(ci_manipulator->get_drag_to()) - simple_xform.affine_inverse().xform(ci_manipulator->get_drag_from());

					if (drag == CanvasItemManipulator::DRAG_SCALE_X) {
						scale_factor.x += offset.x;
						if (uniform) {
							scale_factor.y += offset.x;
						}
					} else if (drag == CanvasItemManipulator::DRAG_SCALE_Y) {
						scale_factor.y += offset.y;
						if (uniform) {
							scale_factor.x += offset.y;
						}
					}

					Rect2 handle_x_rect(Point2(scale_factor.x, CanvasItemManipulator::GIZMO_HANDLE_X_RECT.position.y), CanvasItemManipulator::GIZMO_HANDLE_X_RECT.size);
					RS::get_singleton()->canvas_item_add_rect(srect_ci, handle_x_rect, axis_x_color);
					RS::get_singleton()->canvas_item_add_line(srect_ci, Point2(), Point2(scale_factor.x, 0), axis_x_color, scale);

					Rect2 handle_y_rect(Point2(CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.position.x, scale_factor.y), CanvasItemManipulator::GIZMO_HANDLE_Y_RECT.size);
					RS::get_singleton()->canvas_item_add_rect(srect_ci, handle_y_rect, axis_y_color);
					RS::get_singleton()->canvas_item_add_line(srect_ci, Point2(), Point2(0, scale_factor.y), axis_y_color, scale);
				}

				RS::get_singleton()->canvas_item_add_set_transform(srect_ci, Transform2D());

				if (was_input_disabled) {
					input->set_disable_input(true);
				}

				// Rotation Line
				if (drag == CanvasItemManipulator::DRAG_ROTATE) {
					RS::get_singleton()->canvas_item_add_line(srect_ci, ci_manipulator->get_drag_rotation_center(), ci_manipulator->get_drag_to(), accent_color * Color(1, 1, 1, 0.6), 2 * scale);
				}
			}
		}

		if (!Math::is_inf(temp_pivot.x) && !Math::is_inf(temp_pivot.y)) {
			Size2 pivot_size = pivot_icon->get_size() * scale;
			Rect2 rect(((temp_pivot - view_2d_offset) * view_2d_zoom - (pivot_size / 2.0)).floor(), pivot_size);
			RS::get_singleton()->canvas_item_add_texture_rect(srect_ci, rect, pivot_icon->get_rid(), false, accent_color);
		}
	}

#ifndef _3D_DISABLED
	for (HashMap<ObjectID, Ref<SelectionBox>>::ConstIterator kv = selected_3d_nodes.begin(); kv != selected_3d_nodes.end(); ++kv) {
		ObjectID id = kv->key;

		Node3D *n3d = ObjectDB::get_instance<Node3D>(id);
		if (!n3d) {
			selected_3d_nodes.erase(id);
			--kv;
			continue;
		}

		if (!n3d->is_inside_tree()) {
			continue;
		}

		// Fallback.
		AABB bounds(Vector3(-0.5, -0.5, -0.5), Vector3(1, 1, 1));

		VisualInstance3D *visual_instance = Object::cast_to<VisualInstance3D>(n3d);
		if (visual_instance) {
			bounds = visual_instance->get_aabb();
		} else {
#ifndef PHYSICS_3D_DISABLED
			CollisionShape3D *collision_shape = Object::cast_to<CollisionShape3D>(n3d);
			if (collision_shape) {
				Ref<Shape3D> shape = collision_shape->get_shape();
				if (shape.is_valid()) {
					bounds = shape->get_debug_mesh()->get_aabb();
				}
			}
#endif // PHYSICS_3D_DISABLED
		}

		Transform3D xform_to_top_level_parent_space = n3d->get_global_transform().affine_inverse() * n3d->get_global_transform();
		bounds = xform_to_top_level_parent_space.xform(bounds);
		Transform3D t = n3d->get_global_transform();

		Ref<SelectionBox> sb = kv->value;
		if (t == sb->transform && bounds == sb->bounds) {
			continue; // Nothing changed.
		}
		sb->transform = t;
		sb->bounds = bounds;

		Transform3D t_offset = t;

		// Apply AABB scaling before item's global transform.
		{
			const Vector3 offset(0.005, 0.005, 0.005);
			Basis aabb_s;
			aabb_s.scale(bounds.size + offset);
			t.translate_local(bounds.position - offset / 2);
			t.basis = t.basis * aabb_s;
		}
		{
			const Vector3 offset(0.01, 0.01, 0.01);
			Basis aabb_s;
			aabb_s.scale(bounds.size + offset);
			t_offset.translate_local(bounds.position - offset / 2);
			t_offset.basis = t_offset.basis * aabb_s;
		}

		RS::get_singleton()->instance_set_visible(sb->instance, selection_visible);
		RS::get_singleton()->instance_set_visible(sb->instance_ofs, selection_visible);
		RS::get_singleton()->instance_set_visible(sb->instance_xray, selection_visible);
		RS::get_singleton()->instance_set_visible(sb->instance_xray_ofs, selection_visible);

		RS::get_singleton()->instance_set_transform(sb->instance, t);
		RS::get_singleton()->instance_set_transform(sb->instance_ofs, t_offset);
		RS::get_singleton()->instance_set_transform(sb->instance_xray, t);
		RS::get_singleton()->instance_set_transform(sb->instance_xray_ofs, t_offset);
	}
#endif // _3D_DISABLED
}

void RuntimeNodeSelect::_clear_selection(bool p_send_msg) {
	selected_ci_nodes.clear();
	if (draw_canvas.is_valid()) {
		RS::get_singleton()->canvas_item_clear(srect_ci);
	}

#ifndef _3D_DISABLED
	selected_3d_nodes.clear();
#endif // _3D_DISABLED

	if (has_selection && p_send_msg) {
		EngineDebugger::get_singleton()->send_message("remote_nothing_selected", Array());
	}

	has_selection = false;
}

void RuntimeNodeSelect::_update_selection_drag(const Point2 &p_end_pos) {
	RS::get_singleton()->canvas_item_clear(sel_drag_ci);

	if (selection_drag_state != SELECTION_DRAG_MOVE) {
		return;
	}

	selection_drag_area.size = p_end_pos - selection_drag_area.position;

	if (selection_drag_state == SELECTION_DRAG_END) {
		return;
	}

	_set_selection_area(selection_drag_area.abs());
}

void RuntimeNodeSelect::_set_selection_area(const Rect2 &p_area) {
	RS::get_singleton()->canvas_item_clear(sel_drag_ci);
	if (!p_area.has_area()) {
		return;
	}

	const Vector2 endpoints[4] = {
		p_area.position,
		p_area.position + Point2(p_area.size.x, 0),
		p_area.position + p_area.size,
		p_area.position + Point2(0, p_area.size.y)
	};

	// Draw fill.
	RS::get_singleton()->canvas_item_add_rect(sel_drag_ci, p_area, selection_area_fill);
	// Draw outline.
	for (int i = 0; i < 4; i++) {
		RS::get_singleton()->canvas_item_add_line(sel_drag_ci, endpoints[i], endpoints[(i + 1) % 4], selection_area_outline, 1);
	}
}

void RuntimeNodeSelect::_open_selection_list(const Array &p_selection, const Point2 &p_pos, bool p_append) {
	selection_list_appending = p_append;

	selection_list = memnew(PopupMenu);
	selection_list->set_theme(ThemeDB::get_singleton()->get_default_theme());
	selection_list->set_auto_translate_mode(Node::AUTO_TRANSLATE_MODE_DISABLED);
	selection_list->set_force_native(true);
	selection_list->connect("index_pressed", callable_mp(this, &RuntimeNodeSelect::_items_popup_index_pressed).bind(selection_list));
	selection_list->connect("popup_hide", callable_mp(this, &RuntimeNodeSelect::_close_selection_list));

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(selection_list);

	for (const Variant &var : p_selection) {
		Node *node = Object::cast_to<Node>(var);
		int locked = 0;

		if (ci_manipulator->is_node_locked(node)) {
			locked = 1;
		} else {
			Node *parent = node;
			while (parent && parent != root->get_parent()) {
				if (parent->has_meta("_edit_group_")) {
					locked = 2;
				}
				parent = parent->get_parent();
			}
		}

		String suffix;
		if (locked == 1) {
			suffix = " (" + RTR("Locked") + ")";
		} else if (locked == 2) {
			suffix = " (" + RTR("Grouped") + ")";
		}

		selection_list->add_item((String)node->get_name() + suffix);
		selection_list->set_item_metadata(-1, node);
	}

	selection_list->set_position(selection_list->is_embedded() ? p_pos : (Input::get_singleton()->get_mouse_position() + root->get_position()));
	selection_list->reset_size();
	selection_list->popup();

	selection_list->set_content_scale_factor(1);
	selection_list->set_min_size(selection_list->get_contents_minimum_size());
	selection_list->reset_size();

	// FIXME: Ugly hack that stops the popup from hiding when the button is released.
	selection_list->call_deferred(SNAME("set_position"), selection_list->get_position() + Point2(1, 0));
}

void RuntimeNodeSelect::_close_selection_list() {
	selection_list->queue_free();
	selection_list = nullptr;
}

void RuntimeNodeSelect::_set_selection_visible(bool p_visible) {
	selection_visible = p_visible;

	if (has_selection) {
		if (p_visible) {
			_queue_selection_update();
		} else {
			_update_selection();
		}
	}
}

void RuntimeNodeSelect::_set_avoid_locked(bool p_enabled) {
	avoid_locked_nodes = p_enabled;
}

void RuntimeNodeSelect::_set_prefer_group(bool p_enabled) {
	prefer_group_selection = p_enabled;
}

void RuntimeNodeSelect::_pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event) {
	Vector2 scroll = SceneTree::get_singleton()->get_root()->get_screen_transform().affine_inverse().xform(p_scroll_vec);
	view_2d_offset.x -= scroll.x / view_2d_zoom;
	view_2d_offset.y -= scroll.y / view_2d_zoom;

	_update_view_2d();
}

// A very shallow copy of the same function inside CanvasItemEditor.
void RuntimeNodeSelect::_zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event) {
	real_t prev_zoom = view_2d_zoom;
	view_2d_zoom = CLAMP(view_2d_zoom * p_zoom_factor, VIEW_2D_MIN_ZOOM, VIEW_2D_MAX_ZOOM);

	Vector2 pos = SceneTree::get_singleton()->get_root()->get_screen_transform().affine_inverse().xform(p_origin);
	view_2d_offset += pos / prev_zoom - pos / view_2d_zoom;

	// We want to align in-scene pixels to screen pixels, this prevents blurry rendering
	// of small details (texts, lines).
	// This correction adds a jitter movement when zooming, so we correct only when the
	// zoom factor is an integer. (in the other cases, all pixels won't be aligned anyway)
	const real_t closest_zoom_factor = Math::round(view_2d_zoom);
	if (Math::is_zero_approx(view_2d_zoom - closest_zoom_factor)) {
		// Make sure scene pixel at view_offset is aligned on a screen pixel.
		Vector2 view_offset_int = view_2d_offset.floor();
		Vector2 view_offset_frac = view_2d_offset - view_offset_int;
		view_2d_offset = view_offset_int + (view_offset_frac * closest_zoom_factor).round() / closest_zoom_factor;
	}

	_update_view_2d();
}

void RuntimeNodeSelect::_reset_camera_2d() {
	camera_first_override = true;
	Window *root = SceneTree::get_singleton()->get_root();
	Camera2D *game_camera = root->is_camera_2d_override_enabled() ? root->get_overridden_camera_2d() : root->get_camera_2d();
	if (game_camera) {
		// Ideally we should be using Camera2D::get_camera_transform() but it's not so this hack will have to do for now.
		view_2d_offset = game_camera->get_camera_screen_center() - (0.5 * root->get_visible_rect().size);
	} else {
		view_2d_offset = Vector2();
	}

	view_2d_zoom = 1;

	if (root->is_camera_2d_override_enabled()) {
		_update_view_2d();
	}
}

void RuntimeNodeSelect::_update_view_2d() {
	Window *root = SceneTree::get_singleton()->get_root();
	Camera2D *override_camera = root->get_override_camera_2d();
	ERR_FAIL_NULL(override_camera);
	override_camera->set_anchor_mode(Camera2D::ANCHOR_MODE_FIXED_TOP_LEFT);
	override_camera->set_zoom(Vector2(view_2d_zoom, view_2d_zoom));
	override_camera->set_offset(view_2d_offset);

	ci_manipulator->set_zoom(view_2d_zoom);
	ci_manipulator->set_view_offset(view_2d_offset);

	_queue_selection_update();
}

#ifndef _3D_DISABLED

void RuntimeNodeSelect::_find_3d_items_at_pos(const Point2 &p_pos, Vector<DebuggerHelpers::SelectResult> &r_items) {
	Window *root = SceneTree::get_singleton()->get_root();

	Vector3 ray, pos, to;
	Camera3D *camera = root->get_camera_3d();
	if (!camera) {
		return;
	}

	ray = camera->project_ray_normal(p_pos);
	pos = camera->project_ray_origin(p_pos);
	to = pos + ray * camera->get_far();

#ifndef PHYSICS_3D_DISABLED
	// Start with physical objects.
	PhysicsDirectSpaceState3D *ss = root->get_world_3d()->get_direct_space_state();
	PS3DT::RayResult result;
	HashSet<RID> excluded;
	PS3DT::RayParameters ray_params;
	ray_params.from = pos;
	ray_params.to = to;
	ray_params.collide_with_areas = true;
	while (true) {
		ray_params.exclude = excluded;
		if (ss->intersect_ray(ray_params, result)) {
			DebuggerHelpers::SelectResult res;
			res.item = Object::cast_to<Node>(result.collider);
			res.order = -pos.distance_to(result.position);

			// Fetch collision shapes.
			CollisionObject3D *collision = Object::cast_to<CollisionObject3D>(result.collider);
			if (collision) {
				List<uint32_t> owners;
				collision->get_shape_owners(&owners);
				for (uint32_t &I : owners) {
					DebuggerHelpers::SelectResult res_shape;
					res_shape.item = Object::cast_to<Node>(collision->shape_owner_get_owner(I));
					res_shape.order = res.order;
					r_items.push_back(res_shape);
				}
			}

			r_items.push_back(res);

			excluded.insert(result.rid);
		} else {
			break;
		}
	}
#endif // PHYSICS_3D_DISABLED

	// Then go for the meshes.
	Vector<ObjectID> items = RS::get_singleton()->instances_cull_ray(pos, to, root->get_world_3d()->get_scenario());
	for (int i = 0; i < items.size(); i++) {
		Object *obj = ObjectDB::get_instance(items[i]);

		GeometryInstance3D *geo_instance = Object::cast_to<GeometryInstance3D>(obj);
		if (geo_instance) {
			Ref<TriangleMesh> mesh_collision = geo_instance->generate_triangle_mesh();

			if (mesh_collision.is_valid()) {
				Transform3D gt = geo_instance->get_global_transform();
				Transform3D ai = gt.affine_inverse();
				Vector3 point, normal;
				if (mesh_collision->intersect_ray(ai.xform(pos), ai.basis.xform(ray).normalized(), point, normal)) {
					DebuggerHelpers::SelectResult res;
					res.item = Object::cast_to<Node>(obj);
					res.order = -pos.distance_to(gt.xform(point));
					r_items.push_back(res);

					continue;
				}
			}
		}

		items.remove_at(i);
		i--;
	}
}

void RuntimeNodeSelect::_find_3d_items_at_rect(const Rect2 &p_rect, Vector<DebuggerHelpers::SelectResult> &r_items) {
	Window *root = SceneTree::get_singleton()->get_root();
	Camera3D *camera = root->get_camera_3d();
	if (!camera) {
		return;
	}

	Vector3 cam_pos = camera->get_global_position();
	Vector3 dist_pos = camera->project_ray_origin(p_rect.position + p_rect.size / 2);

	real_t znear = camera->get_near();
	real_t zfar = camera->get_far();
	real_t zofs = MAX(0.0, 5.0 - znear);

	const Point2 pos_end = p_rect.position + p_rect.size;
	Vector3 box[4] = {
		Vector3(
				MIN(p_rect.position.x, pos_end.x),
				MIN(p_rect.position.y, pos_end.y),
				zofs),
		Vector3(
				MAX(p_rect.position.x, pos_end.x),
				MIN(p_rect.position.y, pos_end.y),
				zofs),
		Vector3(
				MAX(p_rect.position.x, pos_end.x),
				MAX(p_rect.position.y, pos_end.y),
				zofs),
		Vector3(
				MIN(p_rect.position.x, pos_end.x),
				MAX(p_rect.position.y, pos_end.y),
				zofs)
	};

	Vector<Plane> frustum;
	for (int i = 0; i < 4; i++) {
		Vector3 a = _get_screen_to_space(box[i]);
		Vector3 b = _get_screen_to_space(box[(i + 1) % 4]);
		frustum.push_back(Plane(a, b, cam_pos));
	}

	// Get the camera normal.
	Plane near_plane = Plane(camera->get_global_transform().basis.get_column(2), cam_pos);

	near_plane.d -= znear;
	frustum.push_back(near_plane);

	Plane far_plane = -near_plane;
	far_plane.d += zfar;
	frustum.push_back(far_plane);

	// Keep track of the currently listed nodes, so repeats can be ignored.
	HashSet<Node *> node_list;

#ifndef PHYSICS_3D_DISABLED
	Vector<Vector3> points = Geometry3D::compute_convex_mesh_points(&frustum[0], frustum.size());
	Ref<ConvexPolygonShape3D> shape;
	shape.instantiate();
	shape->set_points(points);

	// Start with physical objects.
	PhysicsDirectSpaceState3D *ss = root->get_world_3d()->get_direct_space_state();
	PS3DT::ShapeResult results[32];
	PS3DT::ShapeParameters shape_params;
	shape_params.shape_rid = shape->get_rid();
	shape_params.collide_with_areas = true;
	const int num_hits = ss->intersect_shape(shape_params, results, 32);
	for (int i = 0; i < num_hits; i++) {
		const PS3DT::ShapeResult &result = results[i];
		if (!result.collider) {
			continue;
		}

		DebuggerHelpers::SelectResult res;
		res.item = Object::cast_to<Node>(result.collider);
		res.order = -dist_pos.distance_to(Object::cast_to<Node3D>(res.item)->get_global_transform().origin);

		// Fetch collision shapes.
		CollisionObject3D *collision = Object::cast_to<CollisionObject3D>(result.collider);
		if (collision) {
			List<uint32_t> owners;
			collision->get_shape_owners(&owners);
			for (uint32_t &I : owners) {
				DebuggerHelpers::SelectResult res_shape;
				res_shape.item = Object::cast_to<Node>(collision->shape_owner_get_owner(I));
				if (!node_list.has(res_shape.item)) {
					node_list.insert(res_shape.item);
					res_shape.order = res.order;
					r_items.push_back(res_shape);
				}
			}
		}

		if (!node_list.has(res.item)) {
			node_list.insert(res.item);
			r_items.push_back(res);
		}
	}
#endif // PHYSICS_3D_DISABLED

	// Then go for the meshes.
	Vector<ObjectID> items = RS::get_singleton()->instances_cull_convex(frustum, root->get_world_3d()->get_scenario());
	for (int i = 0; i < items.size(); i++) {
		Object *obj = ObjectDB::get_instance(items[i]);
		GeometryInstance3D *geo_instance = Object::cast_to<GeometryInstance3D>(obj);
		if (geo_instance) {
			Ref<TriangleMesh> mesh_collision = geo_instance->generate_triangle_mesh();

			if (mesh_collision.is_valid()) {
				Transform3D gt = geo_instance->get_global_transform();
				Vector3 mesh_scale = gt.get_basis().get_scale();
				gt.orthonormalize();

				Transform3D it = gt.affine_inverse();

				Vector<Plane> transformed_frustum;
				int plane_count = frustum.size();
				transformed_frustum.resize(plane_count);

				for (int j = 0; j < plane_count; j++) {
					transformed_frustum.write[j] = it.xform(frustum[j]);
				}
				Vector<Vector3> convex_points = Geometry3D::compute_convex_mesh_points(transformed_frustum.ptr(), plane_count);
				if (mesh_collision->inside_convex_shape(transformed_frustum.ptr(), transformed_frustum.size(), convex_points.ptr(), convex_points.size(), mesh_scale)) {
					DebuggerHelpers::SelectResult res;
					res.item = Object::cast_to<Node>(obj);
					if (!node_list.has(res.item)) {
						node_list.insert(res.item);
						res.order = -dist_pos.distance_to(gt.origin);
						r_items.push_back(res);
					}

					continue;
				}
			}
		}

		items.remove_at(i);
		i--;
	}
}

Vector3 RuntimeNodeSelect::_get_screen_to_space(const Vector3 &p_vector3) {
	Window *root = SceneTree::get_singleton()->get_root();
	Camera3D *camera = root->get_camera_3d();

	Transform3D camera_transform = camera->get_camera_transform();
	Size2 size = root->get_size();
	real_t znear = camera->get_near();
	Projection cm = Projection::create_perspective(camera->get_fov(), size.aspect(), znear + p_vector3.z, camera->get_far());
	Vector2 screen_he = cm.get_viewport_half_extents();
	return camera_transform.xform(Vector3(((p_vector3.x / size.width) * 2.0 - 1.0) * screen_he.x, ((1.0 - (p_vector3.y / size.height)) * 2.0 - 1.0) * screen_he.y, -(znear + p_vector3.z)));
}

void RuntimeNodeSelect::_box_selected_ci(const Array &p_selection) {
	Vector<Node *> nodes;
	for (Variant node : p_selection) {
		nodes.append(Object::cast_to<Node>(node));
	}
	_send_ids(nodes, true);
}

void RuntimeNodeSelect::_save_canvas_state_requested(const Array &p_selection, bool p_save_bones) {
	for (const Variant &var : p_selection) {
		CanvasItem *ci = Object::cast_to<CanvasItem>(var);
		ObjectID id = ci->get_instance_id();
		ERR_CONTINUE(!selected_ci_nodes.has(id));
		selected_ci_nodes[id] = ci->_edit_get_state();
	}
}

void RuntimeNodeSelect::_restore_canvas_state_requested(const Array &p_selection, bool p_restore_bones) {
	for (const Variant &var : p_selection) {
		CanvasItem *ci = Object::cast_to<CanvasItem>(var);
		ObjectID id = ci->get_instance_id();
		ERR_CONTINUE(!selected_ci_nodes.has(id));
		ci->_edit_set_state(selected_ci_nodes[id]);
	}
}

void RuntimeNodeSelect::_commit_canvas_state_requested(const Array &p_selection, const String &p_message, bool p_restore_bones) {
	Dictionary ids;

	for (const Variant &var : p_selection) {
		CanvasItem *ci = Object::cast_to<CanvasItem>(var);
		if (!ci) {
			continue;
		}

		ObjectID id = ci->get_instance_id();

		Dictionary states;
		states["undo"] = selected_ci_nodes[id];
		states["redo"] = ci->_edit_get_state();

		ids[id] = states;
	}

	EngineDebugger::get_singleton()->send_message("remote_undo_redo_action", { p_message, ids });
}

void RuntimeNodeSelect::_fov_scaled() {
	SceneTree::get_singleton()->get_root()->get_override_camera_3d()->set_perspective(camera_fov * view_3d_controller->cursor.fov_scale, camera_znear, camera_zfar);
}

void RuntimeNodeSelect::_cursor_interpolated() {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(!root->is_camera_3d_override_enabled());
	root->get_override_camera_3d()->set_transform(view_3d_controller->interp_to_camera_transform());
}

bool RuntimeNodeSelect::_handle_3d_input(const Ref<InputEvent> &p_event) {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND_V(!root->is_camera_3d_override_enabled(), true);

	Input *input = Input::get_singleton();
	bool was_input_disabled = input->is_input_disabled();
	if (was_input_disabled) {
		input->set_disable_input(false);
	}

	// Reduce all sides of the area by 1, so warping works when windows are maximized/fullscreen.
	bool view_3d_input_received = view_3d_controller->gui_input(p_event, Rect2(Vector2(1, 1), root->get_size() - Vector2(2, 2)));

	if (was_input_disabled) {
		input->set_disable_input(true);
	}

	if (view_3d_input_received) {
		root->get_override_camera_3d()->set_transform(view_3d_controller->interp_to_camera_transform());
		return true;
	}

	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid() && b->get_button_index() == MouseButton::RIGHT) {
		bool enable_freelook = b->is_pressed();
		if (enable_freelook && freelook_modifier != Key::NONE) {
			switch (freelook_modifier) {
				case Key::SHIFT: {
					enable_freelook = b->is_shift_pressed();
				} break;
				case Key::ALT: {
					enable_freelook = b->is_alt_pressed();
				} break;
				case Key::META: {
					enable_freelook = b->is_meta_pressed();
				} break;
				case Key::CTRL: {
					enable_freelook = b->is_ctrl_pressed();
				} break;
				default:
					break;
			}

			if (!enable_freelook) {
				return false;
			}
		}

		view_3d_controller->set_freelook_enabled(enable_freelook);
		return true;
	}

	if (freelook_toggle.is_valid()) {
		const Array shortcuts = freelook_toggle->get_events();
		for (Ref<InputEventKey> k : shortcuts) {
			if (k.is_valid() && p_event->is_match(k) && p_event->is_pressed()) {
				view_3d_controller->set_freelook_enabled(!view_3d_controller->is_freelook_enabled());
				return true;
			}
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->get_physical_keycode() == Key::ESCAPE) {
		view_3d_controller->set_freelook_enabled(false);
		return true;
	}

	return false;
}

void RuntimeNodeSelect::_reset_camera_3d() {
	camera_first_override = true;

	View3DController::Cursor cursor;

	Window *root = SceneTree::get_singleton()->get_root();
	Camera3D *game_camera = root->is_camera_3d_override_enabled() ? root->get_overridden_camera_3d() : root->get_camera_3d();
	if (game_camera) {
		Transform3D transform = game_camera->get_camera_transform();
		transform.translate_local(0, 0, -cursor.distance);
		cursor.pos_x = transform.origin.x;
		cursor.pos_y = transform.origin.y;
		cursor.pos_z = transform.origin.z;

		cursor.x_rot = -game_camera->get_global_rotation().x;
		cursor.y_rot = -game_camera->get_global_rotation().y;
		cursor.unsnapped_x_rot = cursor.x_rot;
		cursor.unsnapped_y_rot = cursor.y_rot;

		cursor.fov_scale = CLAMP(game_camera->get_fov() / camera_fov, View3DControllerConsts::CAMERA_MIN_FOV_SCALE, View3DControllerConsts::CAMERA_MAX_FOV_SCALE);
	}

	view_3d_controller->cursor = cursor;

	if (root->is_camera_3d_override_enabled()) {
		view_3d_controller->update_camera();
		Camera3D *override_camera = root->get_override_camera_3d();
		override_camera->set_transform(view_3d_controller->to_camera_transform());
		override_camera->set_perspective(camera_fov * cursor.fov_scale, camera_znear, camera_zfar);
	}
}

RuntimeNodeSelect::SelectionBox::~SelectionBox() {
	if (instance.is_valid()) {
		RS::get_singleton()->free_rid(instance);
		RS::get_singleton()->free_rid(instance_ofs);
		RS::get_singleton()->free_rid(instance_xray);
		RS::get_singleton()->free_rid(instance_xray_ofs);
	}
}

#endif // _3D_DISABLED

#endif // DEBUG_ENABLED
