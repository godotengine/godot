/**************************************************************************/
/*  path_2d_editor_plugin.cpp                                             */
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

#include "path_2d_editor_plugin.h"

#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/menu_button.h"
#include "scene/resources/mesh.h"

void Path2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			curve_edit->set_button_icon(get_editor_theme_icon(SNAME("CurveEdit")));
			curve_edit_curve->set_button_icon(get_editor_theme_icon(SNAME("CurveCurve")));
			curve_create->set_button_icon(get_editor_theme_icon(SNAME("CurveCreate")));
			curve_del->set_button_icon(get_editor_theme_icon(SNAME("CurveDelete")));
			curve_close->set_button_icon(get_editor_theme_icon(SNAME("CurveClose")));
			curve_clear_points->set_button_icon(get_editor_theme_icon(SNAME("Clear")));

			create_curve_button->set_button_icon(get_editor_theme_icon(SNAME("Curve2D")));
		} break;
	}
}

void Path2DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
		hide();
	}
}

bool Path2DEditor::forward_gui_input(const Ref<InputEvent> &p_event) {
	if (!node) {
		return false;
	}

	if (!node->is_visible_in_tree()) {
		return false;
	}

	Viewport *vp = node->get_viewport();
	if (vp && !vp->is_visible_subviewport()) {
		return false;
	}

	if (node->get_curve().is_null()) {
		return false;
	}

	const real_t grab_threshold = EDITOR_GET("editors/polygon_editor/point_grab_radius");

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_screen_transform();

		Vector2 gpoint = mb->get_position();
		Vector2 cpoint = canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint));
		cpoint = node->to_local(node->get_viewport()->get_popup_base_transform().affine_inverse().xform(cpoint));

		if (mb->is_pressed() && action == ACTION_NONE) {
			Ref<Curve2D> curve = node->get_curve();

			original_mouse_pos = gpoint;

			for (int i = 0; i < curve->get_point_count(); i++) {
				real_t dist_to_p = gpoint.distance_to(xform.xform(curve->get_point_position(i)));
				real_t dist_to_p_out = gpoint.distance_to(xform.xform(curve->get_point_position(i) + curve->get_point_out(i)));
				real_t dist_to_p_in = gpoint.distance_to(xform.xform(curve->get_point_position(i) + curve->get_point_in(i)));

				// Check for point movement start (for point + in/out controls).
				if (mb->get_button_index() == MouseButton::LEFT) {
					if (mode == MODE_EDIT && !mb->is_shift_pressed() && dist_to_p < grab_threshold) {
						// Points can only be moved in edit mode.

						action = ACTION_MOVING_POINT;
						action_point = i;
						moving_from = curve->get_point_position(i);
						moving_screen_from = gpoint;
						return true;
					} else if (mode == MODE_EDIT || mode == MODE_EDIT_CURVE) {
						control_points_in_range = 0;
						// In/out controls can be moved in multiple modes.
						if (dist_to_p_out < grab_threshold && i < (curve->get_point_count() - 1)) {
							action = ACTION_MOVING_OUT;
							action_point = i;
							moving_from = curve->get_point_out(i);
							moving_screen_from = gpoint;
							orig_in_length = curve->get_point_in(action_point).length();
							control_points_in_range += 1;
						}
						if (dist_to_p_in < grab_threshold && i > 0) {
							action = ACTION_MOVING_IN;
							action_point = i;
							moving_from = curve->get_point_in(i);
							moving_screen_from = gpoint;
							orig_out_length = curve->get_point_out(action_point).length();
							control_points_in_range += 1;
						}
						if (control_points_in_range > 0) {
							return true;
						}
					}
				}

				// Check for point deletion.
				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				if ((mb->get_button_index() == MouseButton::RIGHT && (mode == MODE_EDIT || mode == MODE_CREATE)) || (mb->get_button_index() == MouseButton::LEFT && mode == MODE_DELETE)) {
					if (dist_to_p < grab_threshold) {
						undo_redo->create_action(TTR("Remove Point from Curve"));
						undo_redo->add_do_method(curve.ptr(), "remove_point", i);
						undo_redo->add_undo_method(curve.ptr(), "add_point", curve->get_point_position(i), curve->get_point_in(i), curve->get_point_out(i), i);
						undo_redo->add_do_method(canvas_item_editor, "update_viewport");
						undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
						undo_redo->commit_action();
						return true;
					} else if (dist_to_p_out < grab_threshold) {
						undo_redo->create_action(TTR("Remove Out-Control from Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_out", i, Vector2());
						undo_redo->add_undo_method(curve.ptr(), "set_point_out", i, curve->get_point_out(i));
						undo_redo->add_do_method(canvas_item_editor, "update_viewport");
						undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
						undo_redo->commit_action();
						return true;
					} else if (dist_to_p_in < grab_threshold) {
						undo_redo->create_action(TTR("Remove In-Control from Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_in", i, Vector2());
						undo_redo->add_undo_method(curve.ptr(), "set_point_in", i, curve->get_point_in(i));
						undo_redo->add_do_method(canvas_item_editor, "update_viewport");
						undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
						undo_redo->commit_action();
						return true;
					}
				}
			}
		}

		if (action != ACTION_NONE && mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) {
			_cancel_current_action();
			return true;
		}

		// Check for point creation.
		if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && ((mb->is_command_or_control_pressed() && mode == MODE_EDIT) || mode == MODE_CREATE)) {
			Ref<Curve2D> curve = node->get_curve();
			curve->add_point(cpoint);
			moving_from = cpoint;

			action = ACTION_MOVING_NEW_POINT;
			action_point = curve->get_point_count() - 1;
			moving_from = curve->get_point_position(action_point);
			moving_screen_from = gpoint;

			canvas_item_editor->update_viewport();

			return true;
		}

		// Check for segment split.
		if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && mode == MODE_EDIT && on_edge) {
			Vector2 gpoint2 = mb->get_position();
			Ref<Curve2D> curve = node->get_curve();

			int insertion_point = -1;
			float mbLength = curve->get_closest_offset(xform.affine_inverse().xform(gpoint2));
			int len = curve->get_point_count();
			for (int i = 0; i < len - 1; i++) {
				float compareLength = curve->get_closest_offset(curve->get_point_position(i + 1));
				if (mbLength >= curve->get_closest_offset(curve->get_point_position(i)) && mbLength <= compareLength) {
					insertion_point = i;
				}
			}
			if (insertion_point == -1) {
				insertion_point = curve->get_point_count() - 2;
			}

			const Vector2 new_point = xform.affine_inverse().xform(gpoint2);
			curve->add_point(new_point, Vector2(0, 0), Vector2(0, 0), insertion_point + 1);

			action = ACTION_MOVING_NEW_POINT_FROM_SPLIT;
			action_point = insertion_point + 1;
			moving_from = curve->get_point_position(action_point);
			moving_screen_from = gpoint2;

			canvas_item_editor->update_viewport();

			on_edge = false;

			return true;
		}

		// Check for point movement completion.
		if (!mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && action != ACTION_NONE) {
			Ref<Curve2D> curve = node->get_curve();

			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			Vector2 new_pos = moving_from + xform.affine_inverse().basis_xform(gpoint - moving_screen_from);
			switch (action) {
				case ACTION_NONE:
					// N/A, handled in above condition.
					break;

				case ACTION_MOVING_POINT:
					if (original_mouse_pos != gpoint) {
						undo_redo->create_action(TTR("Move Point in Curve"));
						undo_redo->add_undo_method(curve.ptr(), "set_point_position", action_point, moving_from);
						undo_redo->add_do_method(curve.ptr(), "set_point_position", action_point, cpoint);
						undo_redo->add_do_method(canvas_item_editor, "update_viewport");
						undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
						undo_redo->commit_action(false);
					}
					break;
				case ACTION_MOVING_NEW_POINT: {
					undo_redo->create_action(TTR("Add Point to Curve"));
					undo_redo->add_do_method(curve.ptr(), "add_point", cpoint);
					undo_redo->add_do_method(curve.ptr(), "set_point_position", action_point, cpoint);
					undo_redo->add_do_method(canvas_item_editor, "update_viewport");
					undo_redo->add_undo_method(curve.ptr(), "remove_point", curve->get_point_count() - 1);
					undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
					undo_redo->commit_action(false);
				} break;

				case ACTION_MOVING_NEW_POINT_FROM_SPLIT: {
					undo_redo->create_action(TTR("Split Curve"));
					undo_redo->add_do_method(curve.ptr(), "add_point", Vector2(), Vector2(), Vector2(), action_point);
					undo_redo->add_do_method(curve.ptr(), "set_point_position", action_point, cpoint);
					undo_redo->add_undo_method(curve.ptr(), "remove_point", action_point);
					undo_redo->add_do_method(canvas_item_editor, "update_viewport");
					undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
					undo_redo->commit_action(false);
				} break;

				case ACTION_MOVING_IN: {
					if (original_mouse_pos != gpoint) {
						undo_redo->create_action(TTR("Move In-Control in Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_in", action_point, new_pos);
						undo_redo->add_undo_method(curve.ptr(), "set_point_in", action_point, moving_from);

						if (mirror_handle_angle) {
							undo_redo->add_do_method(curve.ptr(), "set_point_out", action_point, mirror_handle_length ? -new_pos : (-new_pos.normalized() * orig_out_length));
							undo_redo->add_undo_method(curve.ptr(), "set_point_out", action_point, mirror_handle_length ? -moving_from : (-moving_from.normalized() * orig_out_length));
						}
						undo_redo->add_do_method(canvas_item_editor, "update_viewport");
						undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
						undo_redo->commit_action();
					}
				} break;

				case ACTION_MOVING_OUT: {
					if (original_mouse_pos != gpoint) {
						undo_redo->create_action(TTR("Move Out-Control in Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_out", action_point, new_pos);
						undo_redo->add_undo_method(curve.ptr(), "set_point_out", action_point, moving_from);

						if (mirror_handle_angle) {
							undo_redo->add_do_method(curve.ptr(), "set_point_in", action_point, mirror_handle_length ? -new_pos : (-new_pos.normalized() * orig_in_length));
							undo_redo->add_undo_method(curve.ptr(), "set_point_in", action_point, mirror_handle_length ? -moving_from : (-moving_from.normalized() * orig_in_length));
						}
						undo_redo->add_do_method(canvas_item_editor, "update_viewport");
						undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
						undo_redo->commit_action();
					}
				} break;
			}

			action = ACTION_NONE;

			return true;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		// When both control points were in range of click,
		// pick the point that drags the curve outwards.
		if (control_points_in_range == 2) {
			control_points_in_range = 0;
			Ref<Curve2D> curve = node->get_curve();
			Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_screen_transform();
			Point2 relative = xform.affine_inverse().basis_xform(mm->get_relative());
			real_t angle_in = relative.angle_to(curve->get_point_position(action_point - 1) - curve->get_point_position(action_point));
			real_t angle_out = relative.angle_to(curve->get_point_position(action_point + 1) - curve->get_point_position(action_point));

			if (Math::abs(angle_in) < Math::abs(angle_out)) {
				action = ACTION_MOVING_IN;
				moving_from = curve->get_point_in(action_point);
				orig_out_length = curve->get_point_out(action_point).length();
			} else {
				action = ACTION_MOVING_OUT;
				moving_from = curve->get_point_out(action_point);
				orig_in_length = curve->get_point_in(action_point).length();
			}
		}

		if (action == ACTION_NONE && mode == MODE_EDIT) {
			// Handle Edge Follow
			bool old_edge = on_edge;

			Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_screen_transform();
			Vector2 gpoint = mm->get_position();

			Ref<Curve2D> curve = node->get_curve();
			if (curve.is_null()) {
				return true;
			}
			if (curve->get_point_count() < 2) {
				return true;
			}

			// Find edge
			edge_point = xform.xform(curve->get_closest_point(xform.affine_inverse().xform(mm->get_position())));
			on_edge = false;
			if (edge_point.distance_to(gpoint) <= grab_threshold) {
				on_edge = true;
			}
			// However, if near a control point or its in-out handles then not on edge
			int len = curve->get_point_count();
			for (int i = 0; i < len; i++) {
				Vector2 pp = curve->get_point_position(i);
				Vector2 p = xform.xform(pp);
				if (p.distance_to(gpoint) <= grab_threshold) {
					on_edge = false;
					break;
				}
				p = xform.xform(pp + curve->get_point_in(i));
				if (p.distance_to(gpoint) <= grab_threshold) {
					on_edge = false;
					break;
				}
				p = xform.xform(pp + curve->get_point_out(i));
				if (p.distance_to(gpoint) <= grab_threshold) {
					on_edge = false;
					break;
				}
			}
			if (on_edge || old_edge != on_edge) {
				canvas_item_editor->update_viewport();
				return true;
			}
		}

		if (action != ACTION_NONE) {
			// Handle point/control movement.
			Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_screen_transform();
			Vector2 gpoint = mm->get_position();
			Vector2 cpoint = canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint));
			cpoint = node->to_local(node->get_viewport()->get_popup_base_transform().affine_inverse().xform(cpoint));

			Ref<Curve2D> curve = node->get_curve();

			Vector2 new_pos = moving_from + xform.affine_inverse().basis_xform(gpoint - moving_screen_from);

			switch (action) {
				case ACTION_NONE:
					// N/A, handled in above condition.
					break;

				case ACTION_MOVING_POINT:
				case ACTION_MOVING_NEW_POINT:
				case ACTION_MOVING_NEW_POINT_FROM_SPLIT: {
					curve->set_point_position(action_point, cpoint);
				} break;

				case ACTION_MOVING_IN: {
					curve->set_point_in(action_point, new_pos);

					if (mirror_handle_angle) {
						curve->set_point_out(action_point, mirror_handle_length ? -new_pos : (-new_pos.normalized() * orig_out_length));
					}
				} break;

				case ACTION_MOVING_OUT: {
					curve->set_point_out(action_point, new_pos);

					if (mirror_handle_angle) {
						curve->set_point_in(action_point, mirror_handle_length ? -new_pos : (-new_pos.normalized() * orig_in_length));
					}
				} break;
			}

			canvas_item_editor->update_viewport();
			return true;
		}
	}

	return false;
}

void Path2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!node || !node->is_visible_in_tree() || node->get_curve().is_null()) {
		return;
	}

	Viewport *vp = node->get_viewport();
	if (vp && !vp->is_visible_subviewport()) {
		return;
	}

	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_screen_transform();

	const Ref<Texture2D> path_sharp_handle = get_editor_theme_icon(SNAME("EditorPathSharpHandle"));
	const Ref<Texture2D> path_smooth_handle = get_editor_theme_icon(SNAME("EditorPathSmoothHandle"));
	// Both handle icons must be of the same size
	const Size2 handle_size = path_sharp_handle->get_size();

	const Ref<Texture2D> curve_handle = get_editor_theme_icon(SNAME("EditorCurveHandle"));
	const Size2 curve_handle_size = curve_handle->get_size();

	Ref<Curve2D> curve = node->get_curve();

	int len = curve->get_point_count();
	Control *vpc = canvas_item_editor->get_viewport_control();

	debug_handle_lines.clear();
	debug_handle_curve_transforms.clear();
	debug_handle_sharp_transforms.clear();
	debug_handle_smooth_transforms.clear();

	Transform2D handle_curve_transform = Transform2D().scaled(curve_handle_size * 0.5);
	Transform2D handle_point_transform = Transform2D().scaled(handle_size * 0.5);

	for (int i = 0; i < len; i++) {
		Vector2 point = xform.xform(curve->get_point_position(i));
		// Determines the point icon to be used
		bool smooth = false;

		if (i < len - 1) {
			Vector2 point_out = xform.xform(curve->get_point_position(i) + curve->get_point_out(i));
			if (point != point_out) {
				smooth = true;
				debug_handle_lines.push_back(point);
				debug_handle_lines.push_back(point_out);
				handle_curve_transform.set_origin(point_out);
				debug_handle_curve_transforms.push_back(handle_curve_transform);
			}
		}

		if (i > 0) {
			Vector2 point_in = xform.xform(curve->get_point_position(i) + curve->get_point_in(i));
			if (point != point_in) {
				smooth = true;
				debug_handle_lines.push_back(point);
				debug_handle_lines.push_back(point_in);
				handle_curve_transform.set_origin(point_in);
				debug_handle_curve_transforms.push_back(handle_curve_transform);
			}
		}

		handle_point_transform.set_origin(point);
		if (smooth) {
			debug_handle_smooth_transforms.push_back(handle_point_transform);
		} else {
			debug_handle_sharp_transforms.push_back(handle_point_transform);
		}
	}

	if (on_edge) {
		Ref<Texture2D> add_handle = get_editor_theme_icon(SNAME("EditorHandleAdd"));
		p_overlay->draw_texture(add_handle, edge_point - add_handle->get_size() * 0.5);
	}

	RenderingServer *rs = RS::get_singleton();
	rs->mesh_clear(debug_mesh_rid);

	if (!debug_handle_lines.is_empty()) {
		Array handles_array;
		handles_array.resize(Mesh::ARRAY_MAX);

		handles_array[Mesh::ARRAY_VERTEX] = Vector<Vector2>(debug_handle_lines);

		rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINES, handles_array, Array(), Dictionary(), RS::ARRAY_FLAG_USE_2D_VERTICES);
		rs->canvas_item_add_mesh(vpc->get_canvas_item(), debug_mesh_rid, Transform2D(), Color(0.5, 0.5, 0.5, 1.0));
	}

	// Add texture rects multimeshes for handle vertices.

	uint32_t handle_curve_count = debug_handle_curve_transforms.size();
	uint32_t handle_sharp_count = debug_handle_sharp_transforms.size();
	uint32_t handle_smooth_count = debug_handle_smooth_transforms.size();

	// Add texture rects for curve handle vertices.

	rs->multimesh_set_visible_instances(debug_handle_curve_multimesh_rid, 0);
	if (handle_curve_count > 0) {
		if (rs->multimesh_get_instance_count(debug_handle_curve_multimesh_rid) != int(handle_curve_count)) {
			rs->multimesh_allocate_data(debug_handle_curve_multimesh_rid, handle_curve_count, RS::MULTIMESH_TRANSFORM_2D);
		}

		Vector<float> multimesh_buffer;
		multimesh_buffer.resize(8 * handle_curve_count);
		float *multimesh_buffer_ptrw = multimesh_buffer.ptrw();

		const Transform2D *debug_handle_transforms_ptr = debug_handle_curve_transforms.ptr();

		for (uint32_t i = 0; i < handle_curve_count; i++) {
			const Transform2D &handle_transform = debug_handle_transforms_ptr[i];

			multimesh_buffer_ptrw[i * 8 + 0] = handle_transform[0][0];
			multimesh_buffer_ptrw[i * 8 + 1] = handle_transform[1][0];
			multimesh_buffer_ptrw[i * 8 + 2] = 0;
			multimesh_buffer_ptrw[i * 8 + 3] = handle_transform[2][0];
			multimesh_buffer_ptrw[i * 8 + 4] = handle_transform[0][1];
			multimesh_buffer_ptrw[i * 8 + 5] = handle_transform[1][1];
			multimesh_buffer_ptrw[i * 8 + 6] = 0;
			multimesh_buffer_ptrw[i * 8 + 7] = handle_transform[2][1];
		}

		rs->multimesh_set_buffer(debug_handle_curve_multimesh_rid, multimesh_buffer);
		rs->multimesh_set_visible_instances(debug_handle_curve_multimesh_rid, handle_curve_count);

		rs->canvas_item_add_multimesh(vpc->get_canvas_item(), debug_handle_curve_multimesh_rid, curve_handle->get_rid());
	}

	// Add texture rects for sharp handle vertices.

	rs->multimesh_set_visible_instances(debug_handle_sharp_multimesh_rid, 0);
	if (handle_sharp_count > 0) {
		if (rs->multimesh_get_instance_count(debug_handle_sharp_multimesh_rid) != int(handle_sharp_count)) {
			rs->multimesh_allocate_data(debug_handle_sharp_multimesh_rid, handle_sharp_count, RS::MULTIMESH_TRANSFORM_2D);
		}

		Vector<float> multimesh_buffer;
		multimesh_buffer.resize(8 * handle_sharp_count);
		float *multimesh_buffer_ptrw = multimesh_buffer.ptrw();

		const Transform2D *debug_handle_transforms_ptr = debug_handle_sharp_transforms.ptr();

		for (uint32_t i = 0; i < handle_sharp_count; i++) {
			const Transform2D &handle_transform = debug_handle_transforms_ptr[i];

			multimesh_buffer_ptrw[i * 8 + 0] = handle_transform[0][0];
			multimesh_buffer_ptrw[i * 8 + 1] = handle_transform[1][0];
			multimesh_buffer_ptrw[i * 8 + 2] = 0;
			multimesh_buffer_ptrw[i * 8 + 3] = handle_transform[2][0];
			multimesh_buffer_ptrw[i * 8 + 4] = handle_transform[0][1];
			multimesh_buffer_ptrw[i * 8 + 5] = handle_transform[1][1];
			multimesh_buffer_ptrw[i * 8 + 6] = 0;
			multimesh_buffer_ptrw[i * 8 + 7] = handle_transform[2][1];
		}

		rs->multimesh_set_buffer(debug_handle_sharp_multimesh_rid, multimesh_buffer);
		rs->multimesh_set_visible_instances(debug_handle_sharp_multimesh_rid, handle_sharp_count);

		rs->canvas_item_add_multimesh(vpc->get_canvas_item(), debug_handle_sharp_multimesh_rid, curve_handle->get_rid());
	}

	// Add texture rects for smooth handle vertices.

	rs->multimesh_set_visible_instances(debug_handle_smooth_multimesh_rid, 0);
	if (handle_smooth_count > 0) {
		if (rs->multimesh_get_instance_count(debug_handle_smooth_multimesh_rid) != int(handle_smooth_count)) {
			rs->multimesh_allocate_data(debug_handle_smooth_multimesh_rid, handle_smooth_count, RS::MULTIMESH_TRANSFORM_2D);
		}

		Vector<float> multimesh_buffer;
		multimesh_buffer.resize(8 * handle_smooth_count);
		float *multimesh_buffer_ptrw = multimesh_buffer.ptrw();

		const Transform2D *debug_handle_transforms_ptr = debug_handle_smooth_transforms.ptr();

		for (uint32_t i = 0; i < handle_smooth_count; i++) {
			const Transform2D &handle_transform = debug_handle_transforms_ptr[i];

			multimesh_buffer_ptrw[i * 8 + 0] = handle_transform[0][0];
			multimesh_buffer_ptrw[i * 8 + 1] = handle_transform[1][0];
			multimesh_buffer_ptrw[i * 8 + 2] = 0;
			multimesh_buffer_ptrw[i * 8 + 3] = handle_transform[2][0];
			multimesh_buffer_ptrw[i * 8 + 4] = handle_transform[0][1];
			multimesh_buffer_ptrw[i * 8 + 5] = handle_transform[1][1];
			multimesh_buffer_ptrw[i * 8 + 6] = 0;
			multimesh_buffer_ptrw[i * 8 + 7] = handle_transform[2][1];
		}

		rs->multimesh_set_buffer(debug_handle_smooth_multimesh_rid, multimesh_buffer);
		rs->multimesh_set_visible_instances(debug_handle_smooth_multimesh_rid, handle_smooth_count);

		rs->canvas_item_add_multimesh(vpc->get_canvas_item(), debug_handle_smooth_multimesh_rid, curve_handle->get_rid());
	}
}

void Path2DEditor::_node_visibility_changed() {
	if (!node) {
		return;
	}

	canvas_item_editor->update_viewport();
	_update_toolbar();
}

void Path2DEditor::_update_toolbar() {
	if (!node) {
		return;
	}
	bool has_curve = node->get_curve().is_valid();
	toolbar->set_visible(has_curve);
	create_curve_button->set_visible(!has_curve);
}

void Path2DEditor::edit(Node *p_path2d) {
	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton();
	}

	if (action != ACTION_NONE) {
		_cancel_current_action();
	}

	if (p_path2d) {
		node = Object::cast_to<Path2D>(p_path2d);
		_update_toolbar();

		if (!node->is_connected(SceneStringName(visibility_changed), callable_mp(this, &Path2DEditor::_node_visibility_changed))) {
			node->connect(SceneStringName(visibility_changed), callable_mp(this, &Path2DEditor::_node_visibility_changed));
		}
	} else {
		// The node may have been deleted at this point.
		if (node && node->is_connected(SceneStringName(visibility_changed), callable_mp(this, &Path2DEditor::_node_visibility_changed))) {
			node->disconnect(SceneStringName(visibility_changed), callable_mp(this, &Path2DEditor::_node_visibility_changed));
		}
		node = nullptr;
	}

	canvas_item_editor->update_viewport();
}

void Path2DEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_toolbar"), &Path2DEditor::_update_toolbar);
	ClassDB::bind_method(D_METHOD("_clear_curve_points"), &Path2DEditor::_clear_curve_points);
	ClassDB::bind_method(D_METHOD("_restore_curve_points"), &Path2DEditor::_restore_curve_points);
}

void Path2DEditor::_mode_selected(int p_mode) {
	if (p_mode == MODE_CREATE) {
		curve_create->set_pressed(true);
		curve_edit->set_pressed(false);
		curve_edit_curve->set_pressed(false);
		curve_del->set_pressed(false);
	} else if (p_mode == MODE_EDIT) {
		curve_create->set_pressed(false);
		curve_edit->set_pressed(true);
		curve_edit_curve->set_pressed(false);
		curve_del->set_pressed(false);
	} else if (p_mode == MODE_EDIT_CURVE) {
		curve_create->set_pressed(false);
		curve_edit->set_pressed(false);
		curve_edit_curve->set_pressed(true);
		curve_del->set_pressed(false);
	} else if (p_mode == MODE_DELETE) {
		curve_create->set_pressed(false);
		curve_edit->set_pressed(false);
		curve_edit_curve->set_pressed(false);
		curve_del->set_pressed(true);
	} else if (p_mode == MODE_CLOSE) {
		if (node->get_curve().is_null()) {
			return;
		}
		if (node->get_curve()->get_point_count() < 3) {
			return;
		}
		Vector2 begin = node->get_curve()->get_point_position(0);
		Vector2 end = node->get_curve()->get_point_position(node->get_curve()->get_point_count() - 1);

		if (begin.is_equal_approx(end)) {
			return;
		}
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

		undo_redo->create_action(TTR("Close the Curve"));
		undo_redo->add_do_method(node->get_curve().ptr(), "add_point", begin);
		undo_redo->add_undo_method(node->get_curve().ptr(), "remove_point", node->get_curve()->get_point_count());
		undo_redo->add_do_method(canvas_item_editor, "update_viewport");
		undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
		undo_redo->commit_action();
		return;
	} else if (p_mode == MODE_CLEAR_POINTS) {
		if (node->get_curve().is_null()) {
			return;
		}
		if (node->get_curve()->get_point_count() == 0) {
			return;
		}
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		PackedVector2Array points = node->get_curve()->get_points().duplicate();

		undo_redo->create_action(TTR("Clear Curve Points"), UndoRedo::MERGE_DISABLE, node);
		undo_redo->add_do_method(this, "_clear_curve_points", node);
		undo_redo->add_undo_method(this, "_restore_curve_points", node, points);
		undo_redo->add_do_method(canvas_item_editor, "update_viewport");
		undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
		undo_redo->commit_action();
		return;
	}
	mode = Mode(p_mode);
}

void Path2DEditor::_handle_option_pressed(int p_option) {
	PopupMenu *pm;
	pm = handle_menu->get_popup();

	switch (p_option) {
		case HANDLE_OPTION_ANGLE: {
			bool is_checked = pm->is_item_checked(HANDLE_OPTION_ANGLE);
			mirror_handle_angle = !is_checked;
			pm->set_item_checked(HANDLE_OPTION_ANGLE, mirror_handle_angle);
			pm->set_item_disabled(HANDLE_OPTION_LENGTH, !mirror_handle_angle);
		} break;
		case HANDLE_OPTION_LENGTH: {
			bool is_checked = pm->is_item_checked(HANDLE_OPTION_LENGTH);
			mirror_handle_length = !is_checked;
			pm->set_item_checked(HANDLE_OPTION_LENGTH, mirror_handle_length);
		} break;
	}
}

void Path2DEditor::_cancel_current_action() {
	ERR_FAIL_NULL(node);
	Ref<Curve2D> curve = node->get_curve();
	ERR_FAIL_COND(curve.is_null());

	switch (action) {
		case ACTION_MOVING_POINT: {
			curve->set_point_position(action_point, moving_from);
		} break;

		case ACTION_MOVING_NEW_POINT: {
			curve->remove_point(curve->get_point_count() - 1);
		} break;

		case ACTION_MOVING_NEW_POINT_FROM_SPLIT: {
			curve->remove_point(action_point);
		} break;

		case ACTION_MOVING_IN: {
			curve->set_point_in(action_point, moving_from);
			curve->set_point_out(action_point, mirror_handle_length ? -moving_from : (-moving_from.normalized() * orig_out_length));
		} break;

		case ACTION_MOVING_OUT: {
			curve->set_point_out(action_point, moving_from);
			curve->set_point_in(action_point, mirror_handle_length ? -moving_from : (-moving_from.normalized() * orig_in_length));
		} break;

		default: {
		}
	}

	canvas_item_editor->update_viewport();
	action = ACTION_NONE;
}

void Path2DEditor::_create_curve() {
	ERR_FAIL_NULL(node);

	Ref<Curve2D> new_curve;
	new_curve.instantiate();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Create Curve in Path2D"));
	undo_redo->add_do_property(node, "curve", new_curve);
	undo_redo->add_undo_property(node, "curve", Ref<Curve2D>());
	undo_redo->add_do_method(this, "_update_toolbar");
	undo_redo->add_undo_method(this, "_update_toolbar");
	undo_redo->commit_action();
}

void Path2DEditor::_confirm_clear_points() {
	if (!node || node->get_curve().is_null()) {
		return;
	}
	if (node->get_curve()->get_point_count() == 0) {
		return;
	}
	clear_points_dialog->reset_size();
	clear_points_dialog->popup_centered();
}

void Path2DEditor::_clear_curve_points(Path2D *p_path2d) {
	if (!p_path2d || p_path2d->get_curve().is_null()) {
		return;
	}
	Ref<Curve2D> curve = p_path2d->get_curve();

	if (curve->get_point_count() == 0) {
		return;
	}
	curve->clear_points();

	if (node == p_path2d) {
		_mode_selected(MODE_CREATE);
	}
}

void Path2DEditor::_restore_curve_points(Path2D *p_path2d, const PackedVector2Array &p_points) {
	if (!p_path2d || p_path2d->get_curve().is_null()) {
		return;
	}
	Ref<Curve2D> curve = p_path2d->get_curve();

	if (curve->get_point_count() > 0) {
		curve->clear_points();
	}

	for (int i = 0; i < p_points.size(); i += 3) {
		curve->add_point(p_points[i + 2], p_points[i], p_points[i + 1]); // The Curve2D::points pattern is [point_in, point_out, point_position].
	}

	if (node == p_path2d) {
		_mode_selected(MODE_EDIT);
	}
}

Path2DEditor::Path2DEditor() {
	toolbar = memnew(HBoxContainer);

	curve_edit = memnew(Button);
	curve_edit->set_theme_type_variation(SceneStringName(FlatButton));
	curve_edit->set_toggle_mode(true);
	curve_edit->set_pressed(true);
	curve_edit->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	curve_edit->set_tooltip_text(TTR("Select Points") + "\n" + TTR("Shift+Drag: Select Control Points") + "\n" + vformat(TTR("%s+Click: Add Point"), keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL)) + "\n" + TTR("Left Click: Split Segment (in curve)") + "\n" + TTR("Right Click: Delete Point"));
	curve_edit->set_accessibility_name(TTRC("Select Points"));
	curve_edit->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_mode_selected).bind(MODE_EDIT));
	toolbar->add_child(curve_edit);

	curve_edit_curve = memnew(Button);
	curve_edit_curve->set_theme_type_variation(SceneStringName(FlatButton));
	curve_edit_curve->set_toggle_mode(true);
	curve_edit_curve->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	curve_edit_curve->set_tooltip_text(TTR("Select Control Points (Shift+Drag)"));
	curve_edit_curve->set_accessibility_name(TTRC("Select Control Points"));
	curve_edit_curve->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_mode_selected).bind(MODE_EDIT_CURVE));
	toolbar->add_child(curve_edit_curve);

	curve_create = memnew(Button);
	curve_create->set_theme_type_variation(SceneStringName(FlatButton));
	curve_create->set_toggle_mode(true);
	curve_create->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	curve_create->set_tooltip_text(TTR("Add Point (in empty space)") + "\n" + TTR("Right Click: Delete Point"));
	curve_create->set_accessibility_name(TTRC("Add Point (in empty space)"));
	curve_create->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_mode_selected).bind(MODE_CREATE));
	toolbar->add_child(curve_create);

	curve_del = memnew(Button);
	curve_del->set_theme_type_variation(SceneStringName(FlatButton));
	curve_del->set_toggle_mode(true);
	curve_del->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	curve_del->set_tooltip_text(TTR("Delete Point"));
	curve_del->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_mode_selected).bind(MODE_DELETE));
	toolbar->add_child(curve_del);

	curve_close = memnew(Button);
	curve_close->set_theme_type_variation(SceneStringName(FlatButton));
	curve_close->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	curve_close->set_tooltip_text(TTR("Close Curve"));
	curve_close->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_mode_selected).bind(MODE_CLOSE));
	toolbar->add_child(curve_close);

	curve_clear_points = memnew(Button);
	curve_clear_points->set_theme_type_variation(SceneStringName(FlatButton));
	curve_clear_points->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	curve_clear_points->set_tooltip_text(TTR("Clear Points"));
	curve_clear_points->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_confirm_clear_points));
	toolbar->add_child(curve_clear_points);

	clear_points_dialog = memnew(ConfirmationDialog);
	clear_points_dialog->set_title(TTR("Please Confirm..."));
	clear_points_dialog->set_text(TTR("Remove all curve points?"));
	clear_points_dialog->connect(SceneStringName(confirmed), callable_mp(this, &Path2DEditor::_mode_selected).bind(MODE_CLEAR_POINTS));
	toolbar->add_child(clear_points_dialog);

	handle_menu = memnew(MenuButton);
	handle_menu->set_flat(false);
	handle_menu->set_theme_type_variation("FlatMenuButton");
	handle_menu->set_text(TTR("Options"));
	toolbar->add_child(handle_menu);

	PopupMenu *menu = handle_menu->get_popup();
	menu->add_check_item(TTR("Mirror Handle Angles"));
	menu->set_item_checked(HANDLE_OPTION_ANGLE, mirror_handle_angle);
	menu->add_check_item(TTR("Mirror Handle Lengths"));
	menu->set_item_checked(HANDLE_OPTION_LENGTH, mirror_handle_length);
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &Path2DEditor::_handle_option_pressed));

	add_child(toolbar);

	create_curve_button = memnew(Button);
	create_curve_button->set_text(TTR("Create Curve"));
	create_curve_button->hide();
	add_child(create_curve_button);
	create_curve_button->connect(SceneStringName(pressed), callable_mp(this, &Path2DEditor::_create_curve));

	ERR_FAIL_NULL(RS::get_singleton());
	RenderingServer *rs = RS::get_singleton();

	debug_mesh_rid = rs->mesh_create();

	{
		debug_handle_mesh_rid = rs->mesh_create();

		Vector<Vector2> vertex_array;
		vertex_array.resize(4);
		Vector2 *vertex_array_ptrw = vertex_array.ptrw();
		vertex_array_ptrw[0] = Vector2(-1.0, -1.0);
		vertex_array_ptrw[1] = Vector2(1.0, -1.0);
		vertex_array_ptrw[2] = Vector2(1.0, 1.0);
		vertex_array_ptrw[3] = Vector2(-1.0, 1.0);

		Vector<Vector2> uv_array;
		uv_array.resize(4);
		Vector2 *uv_array_ptrw = uv_array.ptrw();
		uv_array_ptrw[0] = Vector2(0.0, 0.0);
		uv_array_ptrw[1] = Vector2(1.0, 0.0);
		uv_array_ptrw[2] = Vector2(1.0, 1.0);
		uv_array_ptrw[3] = Vector2(0.0, 1.0);

		Vector<int> index_array;
		index_array.resize(6);
		int *index_array_ptrw = index_array.ptrw();
		index_array_ptrw[0] = 0;
		index_array_ptrw[1] = 1;
		index_array_ptrw[2] = 3;
		index_array_ptrw[3] = 1;
		index_array_ptrw[4] = 2;
		index_array_ptrw[5] = 3;

		Array mesh_arrays;
		mesh_arrays.resize(RS::ARRAY_MAX);
		mesh_arrays[RS::ARRAY_VERTEX] = vertex_array;
		mesh_arrays[RS::ARRAY_TEX_UV] = uv_array;
		mesh_arrays[RS::ARRAY_INDEX] = index_array;

		rs->mesh_add_surface_from_arrays(debug_handle_mesh_rid, RS::PRIMITIVE_TRIANGLES, mesh_arrays, Array(), Dictionary(), RS::ARRAY_FLAG_USE_2D_VERTICES);

		debug_handle_curve_multimesh_rid = rs->multimesh_create();
		debug_handle_sharp_multimesh_rid = rs->multimesh_create();
		debug_handle_smooth_multimesh_rid = rs->multimesh_create();

		rs->multimesh_set_mesh(debug_handle_curve_multimesh_rid, debug_handle_mesh_rid);
		rs->multimesh_set_mesh(debug_handle_sharp_multimesh_rid, debug_handle_mesh_rid);
		rs->multimesh_set_mesh(debug_handle_smooth_multimesh_rid, debug_handle_mesh_rid);
	}
}

Path2DEditor::~Path2DEditor() {
	ERR_FAIL_NULL(RS::get_singleton());
	RS::get_singleton()->free_rid(debug_mesh_rid);
	RS::get_singleton()->free_rid(debug_handle_curve_multimesh_rid);
	RS::get_singleton()->free_rid(debug_handle_sharp_multimesh_rid);
	RS::get_singleton()->free_rid(debug_handle_smooth_multimesh_rid);
	RS::get_singleton()->free_rid(debug_handle_mesh_rid);
}

void Path2DEditorPlugin::edit(Object *p_object) {
	path2d_editor->edit(Object::cast_to<Node>(p_object));
}

bool Path2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Path2D");
}

void Path2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		path2d_editor->show();
	} else {
		path2d_editor->hide();
		path2d_editor->edit(nullptr);
	}
}

Path2DEditorPlugin::Path2DEditorPlugin() {
	path2d_editor = memnew(Path2DEditor);
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(path2d_editor);
	path2d_editor->hide();
}
