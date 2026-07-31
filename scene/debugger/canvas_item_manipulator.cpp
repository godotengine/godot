/**************************************************************************/
/*  canvas_item_manipulator.cpp                                           */
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
#include "canvas_item_manipulator.h"

#include "core/config/engine.h"
#include "core/input/shortcut.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/container.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

bool CanvasItemManipulator::_gui_input_rulers_and_guides(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	if (show_guides && show_rulers && drag_type == DRAG_NONE) {
		Node *start = Object::cast_to<Node>(find_items_start_callback.call());
		if (start) {
			Transform2D xform = (Transform2D)local_transform_callback.call() * (Transform2D)global_transform_callback.call();
			// Retrieve the guide lists.
			Array vguides = start->get_meta("_edit_vertical_guides_", Array());
			Array hguides = start->get_meta("_edit_horizontal_guides_", Array());

			// Hover over guides.
			real_t minimum = 1e20;
			is_hovering_h_guide = false;
			is_hovering_v_guide = false;

			if (m.is_valid() && m->get_position().x < ruler_width) {
				// Check if we are hovering an existing horizontal guide.
				for (int i = 0; i < hguides.size(); i++) {
					if (Math::abs(xform.xform(Point2(0, hguides[i])).y - m->get_position().y) < MIN(minimum, 8)) {
						is_hovering_h_guide = true;
						is_hovering_v_guide = false;
						break;
					}
				}

			} else if (m.is_valid() && m->get_position().y < ruler_width) {
				// Check if we are hovering an existing vertical guide.
				for (int i = 0; i < vguides.size(); i++) {
					if (Math::abs(xform.xform(Point2(vguides[i], 0)).x - m->get_position().x) < MIN(minimum, 8)) {
						is_hovering_v_guide = true;
						is_hovering_h_guide = false;
						break;
					}
				}
			}

			// Start dragging a guide
			if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
				// Press button
				if (b->get_position().x < ruler_width && b->get_position().y < ruler_width) {
					// Drag a new double guide.
					drag_type = DRAG_DOUBLE_GUIDE;
					dragged_guide_index = -1;
					return true;
				} else if (b->get_position().x < ruler_width) {
					// Check if we drag an existing horizontal guide.
					dragged_guide_index = -1;
					for (int i = 0; i < hguides.size(); i++) {
						if (Math::abs(xform.xform(Point2(0, hguides[i])).y - b->get_position().y) < MIN(minimum, 8)) {
							dragged_guide_index = i;
						}
					}

					if (dragged_guide_index >= 0) {
						// Drag an existing horizontal guide.
						drag_type = DRAG_H_GUIDE;
					} else {
						// Drag a new vertical guide.
						drag_type = DRAG_V_GUIDE;
					}
					return true;
				} else if (b->get_position().y < ruler_width) {
					// Check if we drag an existing vertical guide.
					dragged_guide_index = -1;
					for (int i = 0; i < vguides.size(); i++) {
						if (Math::abs(xform.xform(Point2(vguides[i], 0)).x - b->get_position().x) < MIN(minimum, 8)) {
							dragged_guide_index = i;
						}
					}

					if (dragged_guide_index >= 0) {
						// Drag an existing vertical guide.
						drag_type = DRAG_V_GUIDE;
					} else {
						// Drag a new vertical guide.
						drag_type = DRAG_H_GUIDE;
					}
					drag_from = xform.affine_inverse().xform(b->get_position());
					return true;
				}
			}
		}
	}

	if (drag_type == DRAG_DOUBLE_GUIDE || drag_type == DRAG_V_GUIDE || drag_type == DRAG_H_GUIDE) {
		// Move the guide.
		if (m.is_valid()) {
			Transform2D xform = (Transform2D)local_transform_callback.call() * (Transform2D)global_transform_callback.call();
			drag_to = xform.affine_inverse().xform(m->get_position());

			dragged_guide_pos = xform.xform(snap_point(drag_to, SNAP_GRID | SNAP_PIXEL | SNAP_OTHER_NODES));
			emit_signal("update_canvas_requested");
			return true;
		}

		// Confirm the guide move.
		if (show_guides && b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			Node *start = Object::cast_to<Node>(find_items_start_callback.call());
			if (start) {
				Transform2D xform = (Transform2D)local_transform_callback.call() * (Transform2D)global_transform_callback.call();

				// Retrieve the guide lists.
				Array vguides = start->get_meta("_edit_vertical_guides_", Array());
				Array hguides = start->get_meta("_edit_horizontal_guides_", Array());

				Point2 edited = snap_point(xform.affine_inverse().xform(b->get_position()), SNAP_GRID | SNAP_PIXEL | SNAP_OTHER_NODES);
				if (drag_type == DRAG_V_GUIDE) {
					String msg;
					if (b->get_position().x > ruler_width) { // Add/move a vertical guide.
						if (dragged_guide_index >= 0) {
							vguides[dragged_guide_index] = edited.x;
							msg = "Move Vertical Guide";
						} else {
							vguides.push_back(edited.x);
							msg = "Create Vertical Guide";
						}
					} else if (dragged_guide_index >= 0) {
						vguides.remove_at(dragged_guide_index);
						msg = "Remove Vertical Guide";
					}

					Dictionary dict;
					dict["_edit_vertical_guides_"] = vguides;
					emit_signal("commit_guide_meta_requested", RTR(msg), dict);

				} else if (drag_type == DRAG_H_GUIDE) { // Add/move a horizontal guide.
					String msg;
					if (b->get_position().y > ruler_width) {
						if (dragged_guide_index >= 0) {
							hguides[dragged_guide_index] = edited.y;
							msg = "Move Horizontal Guide";
						} else {
							hguides.push_back(edited.y);
							msg = "Create Horizontal Guide";
						}
					} else if (dragged_guide_index >= 0) {
						hguides.remove_at(dragged_guide_index);
						msg = "Remove Horizontal Guide";
					}

					Dictionary dict;
					dict["_edit_horizontal_guides_"] = hguides;
					emit_signal("commit_guide_meta_requested", RTR(msg), dict);

				} else if (drag_type == DRAG_DOUBLE_GUIDE) { // Add/move a double guide.
					if (b->get_position().x > ruler_width && b->get_position().y > ruler_width) {
						vguides.push_back(edited.x);
						hguides.push_back(edited.y);

						Dictionary dict;
						dict["_edit_vertical_guides_"] = vguides;
						dict["_edit_horizontal_guides_"] = hguides;
						emit_signal("commit_guide_meta_requested", RTR("Create Horizontal and Vertical Guides"), dict);
					}
				}
			}
		}

		snap_target[0] = SNAP_TARGET_NONE;
		snap_target[1] = SNAP_TARGET_NONE;
		reset_drag();
		emit_signal("update_canvas_requested");
		return true;
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_open_scene_on_double_click(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;

	// Open a sub-scene on double-click.
	if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && b->is_double_click() && tool == TOOL_SELECT) {
		Array selection;
		get_selection_callback.call(selection);
		if (selection.size() == 1) {
			CanvasItem *ci = Object::cast_to<CanvasItem>(selection.front());
			if (ci->is_instance() && ci != find_items_start_callback.call()) {
				emit_signal("scene_double_clicked", ci->get_scene_file_path());
				return true;
			}
		}
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_scale(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	if (drag_type == DRAG_NONE) {
		// Drag the resize handles.
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() &&
				((tool == TOOL_SELECT && b->is_alt_pressed() && b->is_command_or_control_pressed()) || tool == TOOL_SCALE)) {
			Array selection;
			bool has_locked_items = get_selection_callback.call(selection);

			drag_selection.clear();
			// Remove not movable nodes.
			for (const Variant &var : selection) {
				const Node *node = Object::cast_to<Node>(var);
				if (_is_node_movable(node, true)) {
					drag_selection.push_back(var);
				}
			}

			if (!drag_selection.is_empty()) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());

				Transform2D edit_transform;
				if (!Math::is_inf(temp_pivot.x) || !Math::is_inf(temp_pivot.y)) {
					edit_transform = Transform2D(ci->_edit_get_rotation(), temp_pivot);
				} else {
					edit_transform = ci->_edit_get_transform();
				}

				Transform2D xform = (Transform2D)global_transform_callback.call() * ci->get_screen_transform();
				Transform2D unscaled_transform = (xform * ci->get_transform().affine_inverse() * edit_transform).orthonormalized();
				Transform2D simple_xform;
				if (local_space) {
					simple_xform = (Transform2D)local_transform_callback.call() * unscaled_transform;
				} else {
					Transform2D translation = Transform2D(0.0f, unscaled_transform.get_origin());
					simple_xform = (Transform2D)local_transform_callback.call() * translation;
				}

				drag_type = DRAG_SCALE_BOTH;

				if (show_transformation_gizmos) {
					Size2 scale_factor = Size2(GIZMO_HANDLE_DISTANCE, GIZMO_HANDLE_DISTANCE);
					Rect2 x_handle_rect = Rect2(scale_factor.x * scale, -5 * scale, 10 * scale, 10 * scale);
					if (x_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
						drag_type = DRAG_SCALE_X;
					}
					Rect2 y_handle_rect = Rect2(-5 * scale, scale_factor.y * scale, 10 * scale, 10 * scale);
					if (y_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
						drag_type = DRAG_SCALE_Y;
					}
				}

				drag_from = Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position());
				drag_to = drag_from;
				emit_signal("save_canvas_state_requested", drag_selection, false);

				return true;
			} else {
				if (has_locked_items) {
					emit_signal("locked_items_warn_requested");
				}

				return has_locked_items;
			}
		}
	} else if (drag_type == DRAG_SCALE_BOTH || drag_type == DRAG_SCALE_X || drag_type == DRAG_SCALE_Y) {
		// Resize the node.
		if (m.is_valid()) {
			if (!_validate_drag_selection()) {
				drag_type = DRAG_NONE;
				return true;
			}

			emit_signal("restore_canvas_state_requested", drag_selection, false);

			drag_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());

			Size2 scale_max;
			if (drag_type != DRAG_SCALE_BOTH) {
				for (const Variant &var : drag_selection) {
					CanvasItem *ci = Object::cast_to<CanvasItem>(var);
					Size2 ci_scale = ci->_edit_get_scale();

					if (Math::abs(ci_scale.x) > Math::abs(scale_max.x)) {
						scale_max.x = ci_scale.x;
					}
					if (Math::abs(ci_scale.y) > Math::abs(scale_max.y)) {
						scale_max.y = ci_scale.y;
					}
				}
			}

			Transform2D edit_transform;
			bool using_temp_pivot = !Math::is_inf(temp_pivot.x) || !Math::is_inf(temp_pivot.y);
			CanvasItem *ci_front = Object::cast_to<CanvasItem>(drag_selection.front());
			if (using_temp_pivot) {
				edit_transform = Transform2D(ci_front->_edit_get_rotation(), temp_pivot);
			} else {
				edit_transform = ci_front->_edit_get_transform();
			}

			for (const Variant &var : drag_selection) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				Transform2D parent_xform = ci->get_screen_transform() * ci->get_transform().affine_inverse();
				Transform2D unscaled_transform = ((Transform2D)global_transform_callback.call() * parent_xform * edit_transform).orthonormalized();
				Transform2D simple_xform;

				if (local_space || drag_type == DRAG_SCALE_BOTH) {
					simple_xform = ((Transform2D)local_transform_callback.call() * unscaled_transform).affine_inverse() * (Transform2D)global_transform_callback.call();
				} else {
					Transform2D translation = Transform2D(0.0f, unscaled_transform.get_origin());
					simple_xform = ((Transform2D)local_transform_callback.call() * translation).affine_inverse() * (Transform2D)global_transform_callback.call();
				}

				bool uniform = m->is_shift_pressed();
				bool is_ctrl = m->is_command_or_control_pressed();

				Point2 drag_from_local = simple_xform.xform(drag_from);
				Point2 drag_to_local = simple_xform.xform(drag_to);
				Point2 offset = drag_to_local - drag_from_local;

				Transform2D object_transform = ci->_edit_get_transform();
				if (ci->is_class("Node2D")) {
					object_transform.set_skew(ci->get("skew"));
				}

				Size2 ci_scale = ci->_edit_get_scale();
				Size2 original_scale = ci_scale;
				real_t ratio = ci_scale.y / ci_scale.x;
				if (drag_type == DRAG_SCALE_BOTH) {
					Size2 scale_factor = drag_to_local / drag_from_local;
					if (uniform) {
						ci_scale *= (scale_factor.x + scale_factor.y) / 2.0;
					} else {
						ci_scale *= scale_factor;
					}
				} else {
					Size2 scale_factor = Size2(offset.x, -offset.y) / GIZMO_HANDLE_DISTANCE;
					Size2 parent_scale = parent_xform.get_scale();
					// Take into account the biggest scale, so all nodes are scaled uniformly.
					scale_factor *= Size2(1.0 / parent_scale.x, 1.0 / parent_scale.y) / (scale_max / original_scale);

					if (drag_type == DRAG_SCALE_X) {
						if (!local_space && !uniform) {
							object_transform.set_origin(Point2());
							object_transform.scale(Size2(scale_factor.x + 1.0, 1));
							ci_scale *= object_transform.get_scale();
						} else {
							ci_scale.x += scale_factor.x;
						}
						if (uniform) {
							ci_scale.y = ci_scale.x * ratio;
						}
					} else if (drag_type == DRAG_SCALE_Y) {
						if (!local_space && !uniform) {
							object_transform.set_origin(Point2());
							object_transform.scale(Size2(1, -scale_factor.y + 1.0));
							ci_scale *= object_transform.get_scale();
						} else {
							ci_scale.y -= scale_factor.y;
						}
						if (uniform) {
							ci_scale.x = ci_scale.y / ratio;
						}
					}
				}

				if (snap_scale && !is_ctrl) {
					if (snap_relative) {
						ci_scale.x = original_scale.x * (Math::round((ci_scale.x / original_scale.x) / snap_scale_step) * snap_scale_step);
						ci_scale.y = original_scale.y * (Math::round((ci_scale.y / original_scale.y) / snap_scale_step) * snap_scale_step);
					} else {
						ci_scale.x = Math::round(ci_scale.x / snap_scale_step) * snap_scale_step;
						ci_scale.y = Math::round(ci_scale.y / snap_scale_step) * snap_scale_step;
					}
				}

				ci->_edit_set_scale(ci_scale);
				if (!local_space && !uniform) {
					Node2D *n2d = Object::cast_to<Node2D>(ci);
					if (n2d) {
						n2d->_edit_set_rotation(object_transform.get_rotation());
						n2d->set_skew(object_transform.get_skew());
					}
				}

				if (using_temp_pivot) {
					Point2 ci_origin = ci->_edit_get_transform().get_origin();
					ci->_edit_set_position(ci_origin + (ci_origin - temp_pivot) * ((ci_scale - original_scale) / original_scale));
				}
			}

			return true;
		}

		// Confirm resize.
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			commit_drag();
			return true;
		}

		if (_gui_input_cancel_drag(p_event)) {
			return true;
		}
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_pivot(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseMotion> m = p_event;
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventKey> k = p_event;

	// Drag the pivot (in pivot mode, or with the V key).
	if (drag_type == DRAG_NONE) {
		bool move_temp_pivot = ((b.is_valid() && b->is_shift_pressed()) || (k.is_valid() && k->is_shift_pressed()));

		if ((b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::LEFT && tool == TOOL_EDIT_PIVOT) ||
				(k.is_valid() && k->is_pressed() && !k->is_echo() && k->get_keycode() == Key::V && tool == TOOL_SELECT && (k->get_modifiers_mask().is_empty() || move_temp_pivot))) {
			Array selection;
			get_selection_callback.call(selection);

			drag_selection.clear();
			// Filters the selection with nodes that allow setting the pivot.
			for (const Variant &var : selection) {
				const CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				if (ci->_edit_use_pivot() || move_temp_pivot) {
					drag_selection.push_back(var);
				}
			}

			// Start dragging if we still have nodes.
			if (drag_selection.size() > 0) {
				Vector2 event_pos = b.is_valid() ? b->get_position() : (Point2)local_mouse_pos_callback.call();

				if (move_temp_pivot) {
					drag_type = DRAG_TEMP_PIVOT;
					temp_pivot = Transform2D(global_transform_callback.call()).affine_inverse().xform(event_pos);
					emit_signal("update_canvas_requested");
					return true;
				}

				emit_signal("save_canvas_state_requested", drag_selection, false);
				drag_from = Transform2D(global_transform_callback.call()).affine_inverse().xform(event_pos);
				Vector2 new_pos;
				if (drag_selection.size() == 1) {
					const Node2D *n2d = Object::cast_to<Node2D>(drag_selection.front());
					new_pos = snap_point(drag_from, SNAP_NODE_SIDES | SNAP_NODE_CENTER | SNAP_NODE_ANCHORS | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, n2d);
				} else {
					new_pos = snap_point(drag_from, SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, nullptr, drag_selection);
				}

				for (const Variant &var : drag_selection) {
					CanvasItem *ci = Object::cast_to<CanvasItem>(var);
					ci->_edit_set_pivot(ci->get_screen_transform().affine_inverse().xform(new_pos));
				}

				drag_type = DRAG_PIVOT;
			}

			return true;
		}
	}

	if (drag_type == DRAG_PIVOT) {
		// Move the pivot.
		if (m.is_valid()) {
			if (!_validate_drag_selection()) {
				drag_type = DRAG_NONE;
				return true;
			}

			drag_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());
			emit_signal("restore_canvas_state_requested", drag_selection, false);

			Vector2 new_pos;
			if (drag_selection.size() == 1) {
				const Node2D *n2d = Object::cast_to<Node2D>(drag_selection.front());
				new_pos = snap_point(drag_to, SNAP_NODE_SIDES | SNAP_NODE_CENTER | SNAP_NODE_ANCHORS | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, n2d);
			} else {
				new_pos = snap_point(drag_to, SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL);
			}

			for (const Variant &var : drag_selection) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				ci->_edit_set_pivot(ci->get_screen_transform().affine_inverse().xform(new_pos));
			}

			return true;
		}

		// Confirm the pivot move.
		if (drag_selection.size() >= 1 &&
				((b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT && tool == TOOL_EDIT_PIVOT) ||
						(k.is_valid() && !k->is_pressed() && k->get_keycode() == Key::V))) {
			commit_drag();
			snap_target[0] = SNAP_TARGET_NONE;
			snap_target[1] = SNAP_TARGET_NONE;
			return true;
		}

		if (_gui_input_cancel_drag(p_event, true)) {
			return true;
		}
	}

	if (drag_type == DRAG_TEMP_PIVOT) {
		if (m.is_valid()) {
			temp_pivot = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());
			emit_signal("update_canvas_requested");
			return true;
		}

		if ((b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT && tool == TOOL_EDIT_PIVOT) ||
				(k.is_valid() && !k->is_pressed() && k->get_keycode() == Key::V)) {
			drag_type = DRAG_NONE;
			return true;
		}
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_resize(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	// Drag resize handles.
	if (drag_type == DRAG_NONE && b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && tool == TOOL_SELECT) {
		Array selection;
		get_selection_callback.call(selection);

		if (selection.size() == 1) {
			CanvasItem *ci = Object::cast_to<CanvasItem>(selection.front());
			if (ci->_edit_use_rect() && _is_node_movable(ci)) {
				Rect2 rect = ci->_edit_get_rect();
				Transform2D xform = (Transform2D)global_transform_callback.call() * ci->get_screen_transform();

				const Point2 endpoints[4] = {
					xform.xform(rect.position),
					xform.xform(rect.position + Point2(rect.size.x, 0)),
					xform.xform(rect.position + rect.size),
					xform.xform(rect.position + Point2(0, rect.size.y))
				};

				const DragType dragger[] = {
					DRAG_TOP_LEFT,
					DRAG_TOP,
					DRAG_TOP_RIGHT,
					DRAG_RIGHT,
					DRAG_BOTTOM_RIGHT,
					DRAG_BOTTOM,
					DRAG_BOTTOM_LEFT,
					DRAG_LEFT,
				};

				DragType resize_drag = DRAG_NONE;
				real_t radius = RESIZE_HANDLE_DIAMETER * (1.5f / 2.0f);

				for (int i = 0; i < 4; i++) {
					int prev = (i + 3) % 4;
					int next = (i + 1) % 4;

					Point2 ofs = ((endpoints[i] - endpoints[prev]).normalized() + ((endpoints[i] - endpoints[next]).normalized())).normalized();
					ofs *= RESIZE_HANDLE_DIAMETER / 2;
					ofs += endpoints[i];
					if (ofs.distance_to(b->get_position()) < radius) {
						resize_drag = dragger[i * 2];
					}

					ofs = (endpoints[i] + endpoints[next]) / 2;
					ofs += (endpoints[next] - endpoints[i]).orthogonal().normalized() * (RESIZE_HANDLE_DIAMETER / 2);
					if (ofs.distance_to(b->get_position()) < radius) {
						resize_drag = dragger[i * 2 + 1];
					}
				}

				if (resize_drag != DRAG_NONE) {
					drag_type = resize_drag;
					drag_from = Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position());
					drag_selection.clear();
					drag_selection.push_back(ci);
					emit_signal("save_canvas_state_requested", drag_selection, false);
					return true;
				}
			}
		}
	}

	if (drag_type == DRAG_LEFT || drag_type == DRAG_RIGHT || drag_type == DRAG_TOP || drag_type == DRAG_BOTTOM ||
			drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT) {
		// Resize the node.
		if (m.is_valid()) {
			if (!_validate_drag_selection()) {
				drag_type = DRAG_NONE;
				return true;
			}

			CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());
			Array single = { ci };
			emit_signal("restore_canvas_state_requested", single, false);

			bool uniform = m->is_shift_pressed();
			bool symmetric = m->is_alt_pressed();

			Rect2 local_rect = ci->_edit_get_rect();
			real_t aspect = local_rect.has_area() ? (local_rect.get_size().y / local_rect.get_size().x) : (local_rect.get_size().y + 1.0) / (local_rect.get_size().x + 1.0);
			Point2 current_begin = local_rect.get_position();
			Point2 current_end = local_rect.get_position() + local_rect.get_size();
			Point2 max_begin = (symmetric) ? (current_begin + current_end - ci->_edit_get_minimum_size()) / 2.0 : current_end - ci->_edit_get_minimum_size();
			Point2 min_end = (symmetric) ? (current_begin + current_end + ci->_edit_get_minimum_size()) / 2.0 : current_begin + ci->_edit_get_minimum_size();
			Point2 center = (current_begin + current_end) / 2.0;

			drag_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());

			Transform2D xform = ci->get_screen_transform();

			Point2 drag_to_snapped_begin;
			Point2 drag_to_snapped_end;

			drag_to_snapped_end = snap_point(xform.xform(current_end) + (drag_to - drag_from), SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, ci);
			drag_to_snapped_begin = snap_point(xform.xform(current_begin) + (drag_to - drag_from), SNAP_NODE_ANCHORS | SNAP_NODE_PARENT | SNAP_OTHER_NODES | SNAP_GRID | SNAP_PIXEL, 0, ci);

			Point2 drag_begin = xform.affine_inverse().xform(drag_to_snapped_begin);
			Point2 drag_end = xform.affine_inverse().xform(drag_to_snapped_end);

			// Horizontal resize.
			if (drag_type == DRAG_LEFT || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_BOTTOM_LEFT) {
				current_begin.x = MIN(drag_begin.x, max_begin.x);
			} else if (drag_type == DRAG_RIGHT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_RIGHT) {
				current_end.x = MAX(drag_end.x, min_end.x);
			}

			// Vertical resize.
			if (drag_type == DRAG_TOP || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT) {
				current_begin.y = MIN(drag_begin.y, max_begin.y);
			} else if (drag_type == DRAG_BOTTOM || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT) {
				current_end.y = MAX(drag_end.y, min_end.y);
			}

			// Uniform resize.
			if (uniform) {
				if (drag_type == DRAG_LEFT || drag_type == DRAG_RIGHT) {
					current_end.y = current_begin.y + aspect * (current_end.x - current_begin.x);
				} else if (drag_type == DRAG_TOP || drag_type == DRAG_BOTTOM) {
					current_end.x = current_begin.x + (current_end.y - current_begin.y) / aspect;
				} else {
					if (aspect >= 1.0) {
						if (drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT) {
							current_begin.y = current_end.y - aspect * (current_end.x - current_begin.x);
						} else {
							current_end.y = current_begin.y + aspect * (current_end.x - current_begin.x);
						}
					} else {
						if (drag_type == DRAG_TOP_LEFT || drag_type == DRAG_BOTTOM_LEFT) {
							current_begin.x = current_end.x - (current_end.y - current_begin.y) / aspect;
						} else {
							current_end.x = current_begin.x + (current_end.y - current_begin.y) / aspect;
						}
					}
				}
			}

			// Symmetric resize.
			if (symmetric) {
				if (drag_type == DRAG_LEFT || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_BOTTOM_LEFT) {
					current_end.x = 2.0 * center.x - current_begin.x;
				} else if (drag_type == DRAG_RIGHT || drag_type == DRAG_TOP_RIGHT || drag_type == DRAG_BOTTOM_RIGHT) {
					current_begin.x = 2.0 * center.x - current_end.x;
				}
				if (drag_type == DRAG_TOP || drag_type == DRAG_TOP_LEFT || drag_type == DRAG_TOP_RIGHT) {
					current_end.y = 2.0 * center.y - current_begin.y;
				} else if (drag_type == DRAG_BOTTOM || drag_type == DRAG_BOTTOM_LEFT || drag_type == DRAG_BOTTOM_RIGHT) {
					current_begin.y = 2.0 * center.y - current_end.y;
				}
			}

			if (anchors_mode) {
				Control *control = Object::cast_to<Control>(ci);
				if (control) {
					Size2 parent_rect_size = control->get_parent_anchorable_rect().size;
					if (parent_rect_size.x == 0.0 || parent_rect_size.y == 0.0) {
						emit_signal("anchors_mode_warn_requested");
						return true;
					}
				}
			}

			ci->_edit_set_rect(Rect2(current_begin, current_end - current_begin));
			return true;
		}

		// Confirm resize.
		if (drag_selection.size() >= 1 && b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			commit_drag();
			return true;
		}

		if (_gui_input_cancel_drag(p_event, true)) {
			return true;
		}
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_rotate(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	if (drag_type == DRAG_NONE) {
		// Start rotation.
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
			if ((b->is_command_or_control_pressed() && !b->is_alt_pressed() && tool == TOOL_SELECT) || tool == TOOL_ROTATE) {
				Array selection;
				bool has_locked_items = get_selection_callback.call(selection);

				drag_selection.clear();
				// Remove not movable nodes.
				for (const Variant &var : selection) {
					const Node *node = Object::cast_to<Node>(var);
					if (_is_node_movable(node, true)) {
						drag_selection.push_back(var);
					}
				}

				if (drag_selection.size() > 0) {
					drag_type = DRAG_ROTATE;
					drag_from = Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position());
					drag_to = drag_from;

					CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());
					if (!Math::is_inf(temp_pivot.x) || !Math::is_inf(temp_pivot.y)) {
						drag_rotation_center = temp_pivot;
					} else if (ci->_edit_use_pivot()) {
						drag_rotation_center = ci->get_screen_transform().xform(ci->_edit_get_pivot());
					} else {
						drag_rotation_center = ci->get_screen_transform().get_origin();
					}

					emit_signal("save_canvas_state_requested", drag_selection, false);

					return true;
				} else {
					if (has_locked_items) {
						emit_signal("locked_items_warn_requested");
					}

					return has_locked_items;
				}
			}
		}
	}

	if (drag_type == DRAG_ROTATE) {
		// Rotate the node.
		if (m.is_valid()) {
			if (!_validate_drag_selection()) {
				drag_type = DRAG_NONE;
				return true;
			}

			emit_signal("restore_canvas_state_requested", drag_selection, false);

			for (const Variant &var : drag_selection) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				drag_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());

				// Rotate the opposite way if the canvas item's compounded scale has an uneven number of negative elements.
				bool opposite = (ci->get_global_transform().get_scale().sign().dot(ci->get_transform().get_scale().sign()) == 0);

				real_t prev_rotation = ci->_edit_get_rotation();
				real_t new_rotation = ci->_edit_get_rotation() + (opposite ? -1 : 1) * (drag_from - drag_rotation_center).angle_to(drag_to - drag_rotation_center);
				// Snap angle.
				if (((smart_snap || snap_rotation) != m->is_command_or_control_pressed()) && snap_rotation_step != 0) {
					if (snap_relative) {
						new_rotation = Math::snapped(new_rotation - snap_rotation_offset, snap_rotation_step) + snap_rotation_offset + (prev_rotation - (int)(prev_rotation / snap_rotation_step) * snap_rotation_step);
					} else {
						new_rotation = Math::snapped(new_rotation - snap_rotation_offset, snap_rotation_step) + snap_rotation_offset;
					}
				}

				ci->_edit_set_rotation(new_rotation);
				if (!Math::is_inf(temp_pivot.x) || !Math::is_inf(temp_pivot.y)) {
					Transform2D xform = ci->get_screen_transform() * ci->get_transform().affine_inverse();
					Vector2 radius = xform.xform(ci->_edit_get_position()) - temp_pivot;
					radius = radius.rotated(new_rotation - prev_rotation);
					ci->_edit_set_position(xform.affine_inverse().xform(temp_pivot + radius));
				}

				emit_signal("update_canvas_requested");
			}

			return true;
		}

		// Confirm the node rotation.
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			commit_drag();
			return true;
		}

		if (_gui_input_cancel_drag(p_event)) {
			return true;
		}
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_move(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;
	Ref<InputEventKey> k = p_event;

	// Start moving the nodes.
	if (drag_type == DRAG_NONE && b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
		if ((tool == TOOL_SELECT && b->is_alt_pressed() && !b->is_command_or_control_pressed()) || tool == TOOL_MOVE) {
			Array selection;
			bool has_locked_items = get_selection_callback.call(selection);

			drag_selection.clear();
			// Remove not movable nodes.
			for (const Variant &var : selection) {
				const Node *node = Object::cast_to<Node>(var);
				if (_is_node_movable(node, true)) {
					drag_selection.push_back(var);
				}
			}

			if (selection.size() > 0) {
				drag_type = DRAG_MOVE;

				CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());
				Transform2D parent_xform = ci->get_screen_transform() * ci->get_transform().affine_inverse();
				Transform2D unscaled_transform = ((Transform2D)global_transform_callback.call() * parent_xform * ci->_edit_get_transform()).orthonormalized();
				Transform2D simple_xform;
				if (local_space) {
					simple_xform = (Transform2D)local_transform_callback.call() * unscaled_transform;
				} else {
					Transform2D translation = Transform2D(0.0f, unscaled_transform.get_origin());
					simple_xform = (Transform2D)local_transform_callback.call() * translation;
				}

				if (show_transformation_gizmos) {
					const Rect2 x_handle_rect = Rect2(GIZMO_HANDLE_X_RECT.position * scale, GIZMO_HANDLE_X_RECT.size * scale);
					if (x_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
						drag_type = DRAG_MOVE_X;
					}

					const Rect2 y_handle_rect = Rect2(GIZMO_HANDLE_Y_RECT.position * scale, GIZMO_HANDLE_Y_RECT.size * scale);
					if (y_handle_rect.has_point(simple_xform.affine_inverse().xform(b->get_position()))) {
						drag_type = DRAG_MOVE_Y;
					}
				}

				drag_from = Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position());
				emit_signal("save_canvas_state_requested", drag_selection, true);

				return true;
			} else {
				if (has_locked_items) {
					emit_signal("locked_items_warn_requested");
				}

				return has_locked_items;
			}
		}
	}

	// Move the nodes.
	if (drag_type == DRAG_MOVE || drag_type == DRAG_MOVE_X || drag_type == DRAG_MOVE_Y) {
		if (m.is_valid() && !drag_selection.is_empty()) {
			if (!_validate_drag_selection()) {
				drag_type = DRAG_NONE;
				return true;
			}

			emit_signal("restore_canvas_state_requested", drag_selection, true);

			drag_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());
			Point2 previous_pos;
			if (drag_selection.size() == 1) {
				const CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());
				Transform2D parent_xform = ci->get_screen_transform() * ci->get_transform().affine_inverse();
				previous_pos = parent_xform.xform(ci->_edit_get_position());
			} else {
				previous_pos = _encompass_selection_rect(drag_selection).position;
			}

			Point2 drag_delta = drag_to - drag_from;
			if (drag_type == DRAG_MOVE_X || drag_type == DRAG_MOVE_Y) {
				const CanvasItem *selected = Object::cast_to<CanvasItem>(drag_selection.front());
				Transform2D parent_xform = selected->get_screen_transform() * selected->get_transform().affine_inverse();
				Transform2D unscaled_transform = ((Transform2D)global_transform_callback.call() * parent_xform * selected->_edit_get_transform()).orthonormalized();

				Transform2D simple_xform = local_transform_callback.call();
				if (local_space) {
					simple_xform *= unscaled_transform;
				}

				drag_delta = simple_xform.affine_inverse().basis_xform(drag_delta);
				if (drag_type == DRAG_MOVE_X) {
					drag_delta.y = 0;
				} else {
					drag_delta.x = 0;
				}
				drag_delta = simple_xform.basis_xform(drag_delta);
			}
			Point2 new_pos = snap_point(previous_pos + drag_delta, SNAP_GRID | SNAP_GUIDES | SNAP_PIXEL | SNAP_NODE_PARENT | SNAP_NODE_ANCHORS | SNAP_OTHER_NODES, 0, nullptr, drag_selection);

			bool single_axis = m->is_shift_pressed();
			if (single_axis) {
				if (Math::abs(new_pos.x - previous_pos.x) > Math::abs(new_pos.y - previous_pos.y)) {
					new_pos.y = previous_pos.y;
				} else {
					new_pos.x = previous_pos.x;
				}
			}

			for (const Variant &var : drag_selection) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				Transform2D parent_xform_inv = ci->get_transform() * ci->get_screen_transform().affine_inverse();
				ci->_edit_set_position(ci->_edit_get_position() + parent_xform_inv.basis_xform(new_pos - previous_pos));
			}
			return true;
		}

		// Confirm the move (only if it was moved).
		if (b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT) {
			commit_drag();
			return true;
		}

		if (_gui_input_cancel_drag(p_event, true, true)) {
			return true;
		}
	}

	// Move the canvas items with the arrow keys.
	if (k.is_valid() && k->is_pressed() && (tool == TOOL_SELECT || tool == TOOL_MOVE) &&
			(k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::LEFT || k->get_keycode() == Key::RIGHT)) {
		if (!k->is_echo()) {
			// Start moving the canvas items with the keyboard, if they are movable.
			Array selection;
			get_selection_callback.call(selection);

			drag_selection.clear();
			// Remove not movable nodes.
			for (const Variant &var : selection) {
				const Node *node = Object::cast_to<Node>(var);
				if (_is_node_movable(node, true)) {
					drag_selection.push_back(var);
				}
			}

			drag_type = DRAG_KEY_MOVE;
			drag_from = Vector2();
			drag_to = Vector2();
			emit_signal("save_canvas_state_requested", drag_selection, true);
		}

		if (drag_selection.size() > 0) {
			emit_signal("restore_canvas_state_requested", drag_selection, true);

			bool move_local_base = k->is_alt_pressed();
			bool move_local_base_rotated = k->is_ctrl_pressed() || k->is_meta_pressed();

			Vector2 dir;
			if (k->get_keycode() == Key::UP) {
				dir += Vector2(0, -1);
			} else if (k->get_keycode() == Key::DOWN) {
				dir += Vector2(0, 1);
			} else if (k->get_keycode() == Key::LEFT) {
				dir += Vector2(-1, 0);
			} else if (k->get_keycode() == Key::RIGHT) {
				dir += Vector2(1, 0);
			}
			if (k->is_shift_pressed()) {
				dir *= grid_step * Math::pow(2.0, grid_step_multiplier);
			}

			drag_to += dir;
			if (k->is_shift_pressed()) {
				drag_to = drag_to.snapped(grid_step * Math::pow(2.0, grid_step_multiplier));
			}

			Point2 previous_pos;
			if (drag_selection.size() == 1) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());
				Transform2D xform = ci->get_global_transform_with_canvas() * ci->get_transform().affine_inverse();
				previous_pos = xform.xform(ci->_edit_get_position());
			} else {
				previous_pos = _encompass_selection_rect(drag_selection).position;
			}

			Point2 new_pos;
			if (drag_selection.size() == 1) {
				Node2D *node_2d = Object::cast_to<Node2D>(drag_selection.front());
				if (node_2d && move_local_base_rotated) {
					Transform2D m2;
					m2.rotate(node_2d->get_rotation());
					new_pos += m2.xform(drag_to);
				} else if (move_local_base) {
					new_pos += drag_to;
				} else {
					new_pos = previous_pos + (drag_to - drag_from);
				}
			} else {
				new_pos = previous_pos + (drag_to - drag_from);
			}

			for (const Variant &var : drag_selection) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				Transform2D xform = ci->get_global_transform_with_canvas().affine_inverse() * ci->get_transform();
				ci->_edit_set_position(ci->_edit_get_position() + xform.xform(new_pos) - xform.xform(previous_pos));
			}
		}
		return true;
	}

	// Confirm canvas items move by arrow keys.
	if (k.is_valid() && !k->is_pressed() && drag_type == DRAG_KEY_MOVE && (tool == TOOL_SELECT || tool == TOOL_MOVE) &&
			(k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::LEFT || k->get_keycode() == Key::RIGHT)) {
		commit_drag();
		return true;
	}

	// Accept the key event in any case.
	return (k.is_valid() && (k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::LEFT || k->get_keycode() == Key::RIGHT));
}

bool CanvasItemManipulator::_gui_input_anchors(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	// Start anchor dragging.
	if (drag_type == DRAG_NONE) {
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && b->is_pressed() && tool == TOOL_SELECT) {
			Array selection;
			get_selection_callback.call(selection);

			if (selection.size() == 1) {
				Control *control = Object::cast_to<Control>(selection.front());
				if (control && _is_node_movable(control)) {
					Vector2 anchor_pos[4];
					anchor_pos[0] = Vector2(control->get_anchor(SIDE_LEFT), control->get_anchor(SIDE_TOP));
					anchor_pos[1] = Vector2(control->get_anchor(SIDE_RIGHT), control->get_anchor(SIDE_TOP));
					anchor_pos[2] = Vector2(control->get_anchor(SIDE_RIGHT), control->get_anchor(SIDE_BOTTOM));
					anchor_pos[3] = Vector2(control->get_anchor(SIDE_LEFT), control->get_anchor(SIDE_BOTTOM));

					Rect2 anchor_rects[4];
					for (int i = 0; i < 4; i++) {
						anchor_pos[i] = ((Transform2D)global_transform_callback.call() * control->get_screen_transform()).xform(anchor_to_position(control, anchor_pos[i]));
						anchor_rects[i] = Rect2(anchor_pos[i], ANCHOR_HANDLE_SIZE);
						if (control->is_layout_rtl()) {
							anchor_rects[i].position -= ANCHOR_HANDLE_SIZE * Vector2(real_t(i == 1 || i == 2), real_t(i <= 1));
						} else {
							anchor_rects[i].position -= ANCHOR_HANDLE_SIZE * Vector2(real_t(i == 0 || i == 3), real_t(i <= 1));
						}
					}

					const DragType dragger[] = {
						DRAG_ANCHOR_TOP_LEFT,
						DRAG_ANCHOR_TOP_RIGHT,
						DRAG_ANCHOR_BOTTOM_RIGHT,
						DRAG_ANCHOR_BOTTOM_LEFT,
					};

					for (int i = 0; i < 4; i++) {
						if (anchor_rects[i].has_point(b->get_position())) {
							if ((anchor_pos[0] == anchor_pos[2]) && (anchor_pos[0].distance_to(b->get_position()) < ANCHOR_HANDLE_SIZE.length() / 3.0)) {
								drag_type = DRAG_ANCHOR_ALL;
							} else {
								drag_type = dragger[i];
							}

							drag_from = Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position());
							drag_selection.clear();
							drag_selection.push_back(control);
							emit_signal("save_canvas_state_requested", drag_selection, false);

							return true;
						}
					}
				}
			}
		}
	}

	if (drag_type == DRAG_ANCHOR_TOP_LEFT || drag_type == DRAG_ANCHOR_TOP_RIGHT || drag_type == DRAG_ANCHOR_BOTTOM_RIGHT || drag_type == DRAG_ANCHOR_BOTTOM_LEFT || drag_type == DRAG_ANCHOR_ALL) {
		// Drag the anchor.
		if (m.is_valid()) {
			if (!_validate_drag_selection()) {
				drag_type = DRAG_NONE;
				return true;
			}

			emit_signal("restore_canvas_state_requested", drag_selection, false);
			Control *control = Object::cast_to<Control>(drag_selection.front());

			drag_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());

			Transform2D xform = control->get_screen_transform().affine_inverse();

			Point2 previous_anchor;
			previous_anchor.x = (drag_type == DRAG_ANCHOR_TOP_LEFT || drag_type == DRAG_ANCHOR_BOTTOM_LEFT) ? control->get_anchor(SIDE_LEFT) : control->get_anchor(SIDE_RIGHT);
			previous_anchor.y = (drag_type == DRAG_ANCHOR_TOP_LEFT || drag_type == DRAG_ANCHOR_TOP_RIGHT) ? control->get_anchor(SIDE_TOP) : control->get_anchor(SIDE_BOTTOM);
			previous_anchor = xform.affine_inverse().xform(anchor_to_position(control, previous_anchor));

			Vector2 new_anchor = xform.xform(snap_point(previous_anchor + (drag_to - drag_from), SNAP_GRID | SNAP_OTHER_NODES, SNAP_NODE_PARENT | SNAP_NODE_SIDES | SNAP_NODE_CENTER, control));
			new_anchor = _position_to_anchor(control, new_anchor).snappedf(0.001);

			bool use_single_axis = m->is_shift_pressed();
			Vector2 drag_vector = xform.xform(drag_to) - xform.xform(drag_from);
			bool use_y = Math::abs(drag_vector.y) > Math::abs(drag_vector.x);

			switch (drag_type) {
				case DRAG_ANCHOR_TOP_LEFT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_LEFT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_TOP, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_TOP_RIGHT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_RIGHT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_TOP, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_BOTTOM_RIGHT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_RIGHT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_BOTTOM, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_BOTTOM_LEFT:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_LEFT, new_anchor.x, false, false);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_BOTTOM, new_anchor.y, false, false);
					}
					break;
				case DRAG_ANCHOR_ALL:
					if (!use_single_axis || !use_y) {
						control->set_anchor(SIDE_LEFT, new_anchor.x, false, true);
						control->set_anchor(SIDE_RIGHT, new_anchor.x, false, true);
					}
					if (!use_single_axis || use_y) {
						control->set_anchor(SIDE_TOP, new_anchor.y, false, true);
						control->set_anchor(SIDE_BOTTOM, new_anchor.y, false, true);
					}
					break;
				default:
					break;
			}
			return true;
		}

		// Confirm anchor position.
		if (drag_selection.size() >= 1 && b.is_valid() && b->get_button_index() == MouseButton::LEFT && !b->is_pressed()) {
			commit_drag();
			return true;
		}

		if (_gui_input_cancel_drag(p_event, true)) {
			return true;
		}
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_ruler_tool(const Ref<InputEvent> &p_event) {
	if (tool != TOOL_RULER) {
		ruler_tool_active = false;
		return false;
	}

	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	Point2 previous_origin = ruler_tool_origin;
	if (!ruler_tool_active) {
		ruler_tool_origin = snap_point((Point2)local_mouse_pos_callback.call() / zoom + view_offset);
	}

	if (ruler_tool_active && b.is_valid() && b->get_button_index() == MouseButton::RIGHT) {
		ruler_tool_active = false;
		emit_signal("update_canvas_requested");
		return true;
	}

	if (b.is_valid() && b->get_button_index() == MouseButton::LEFT) {
		if (b->is_pressed()) {
			ruler_tool_active = true;
		} else {
			ruler_tool_active = false;
		}

		emit_signal("update_canvas_requested");
		return true;
	}

	if (m.is_valid() && (ruler_tool_active || (grid_snap && previous_origin != ruler_tool_origin))) {
		emit_signal("update_canvas_requested");
		return true;
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_select(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;
	Ref<InputEventKey> k = p_event;

	if (drag_type == DRAG_NONE || (drag_type == DRAG_BOX_SELECTION && b.is_valid() && !b->is_pressed())) {
		// Popup the selection menu list.
		if (b.is_valid() && b->is_pressed() &&
				((b->get_button_index() == MouseButton::RIGHT && b->is_alt_pressed()) ||
						(b->get_button_index() == MouseButton::LEFT && tool == TOOL_LIST_SELECT))) {
			Vector<DebuggerHelpers::SelectResult> items;
			get_canvas_items_at_pos(Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position()), items, b->is_alt_pressed());

			if (items.size() == 1) {
				// Only one node below the cursor, just select it.
				point_selected_callback.call(items[0].item, b->is_shift_pressed());
				return true;
			}

			if (!items.is_empty()) {
				items.sort(); // Sorts items according the their z-index.

				Array nodes;
				for (const DebuggerHelpers::SelectResult &item : items) {
					nodes.append(item.item);
				}
				emit_signal("selection_menu_requested", nodes, b->get_position(), b->is_shift_pressed());

				return true;
			}
		}

		if (b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::RIGHT && tool != TOOL_SCENE_PAINT) {
			emit_signal("tool_menu_requested", viewport->get_screen_transform().xform(b->get_position()));
			return true;
		}

		// Single item selection.
		if (b.is_valid() && b->get_button_index() == MouseButton::LEFT && (tool == TOOL_SELECT || tool == TOOL_MOVE || tool == TOOL_SCALE || tool == TOOL_ROTATE)) {
			Point2 pos = Transform2D(global_transform_callback.call()).affine_inverse().xform(b->get_position());
			// Allow selecting on release when performed very small box selection (necessary when Shift is pressed, see below).
			if (b->is_pressed() || (drag_type == DRAG_BOX_SELECTION && pos.distance_to(drag_from) <= drag_threshold * scale)) {
				Vector<DebuggerHelpers::SelectResult> items;
				get_canvas_items_at_pos(pos, items);

				CanvasItem *ci = nullptr;

				// Retrieve the canvas items.
				if (!items.is_empty()) {
					items.sort(); // Sorts items according the their z-index.
					ci = Object::cast_to<CanvasItem>(items[0].item);
				}

				// Shift also allows forcing box selection when item was clicked.
				if (!ci || (b->is_shift_pressed() && b->is_pressed())) {
					// Start a box selection.
					if (!b->is_shift_pressed()) {
						// Clear the selection if not additive.
						emit_signal("clear_selection_requested", true);
					}

					if (b->is_pressed()) {
						drag_from = pos;
						drag_type = DRAG_BOX_SELECTION;
						box_selecting_to = drag_from;
						return true;
					}
				} else {
					bool selected = point_selected_callback.call(ci, b->is_shift_pressed());
					// Start dragging.
					if (selected && (tool == TOOL_SELECT || tool == TOOL_MOVE) && b->is_pressed()) {
						// Drag the node(s) if requested.
						drag_start_origin = pos;
						drag_type = DRAG_QUEUED;
					} else if (!b->is_pressed()) {
						reset_drag();
					}

					return true; // Selected the item.
				}
			}
		}
	}

	if (drag_type == DRAG_QUEUED) {
		if (b.is_valid() && !b->is_pressed()) {
			reset_drag();
			return true;
		}

		if (m.is_valid()) {
			Point2 pos = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());
			bool movement_threshold_passed = drag_start_origin.distance_to(pos) > (8 * MAX(1, scale)) / zoom;
			if (m.is_valid() && movement_threshold_passed) {
				Array selection;
				get_selection_callback.call(selection);

				drag_selection.clear();
				for (const Variant &var : selection) {
					const Node *node = Object::cast_to<Node>(var);
					if (_is_node_movable(node, true)) {
						drag_selection.push_back(var);
					}
				}

				if (drag_selection.size() > 0) {
					drag_type = DRAG_MOVE;
					drag_from = drag_start_origin;
					emit_signal("save_canvas_state_requested", drag_selection, false);
				}

				return true;
			}
		}
	}

	if (drag_type == DRAG_BOX_SELECTION) {
		// Update box area.
		Point2 bsfrom = drag_from;
		Point2 bsto = box_selecting_to;
		if (bsfrom.x > bsto.x) {
			SWAP(bsfrom.x, bsto.x);
		}
		if (bsfrom.y > bsto.y) {
			SWAP(bsfrom.y, bsto.y);
		}
		Rect2 area(bsfrom, bsto - bsfrom);

		// Confirm box selection.
		if (b.is_valid() && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT) {
			Node *start = Object::cast_to<Node>(find_items_start_callback.call());
			if (start) {
				Vector<DebuggerHelpers::SelectResult> items;
				find_canvas_items_in_rect(area, start, items);

				if (!items.is_empty()) {
					Array nodes;
					for (const DebuggerHelpers::SelectResult &item : items) {
						nodes.append(item.item);
					}
					emit_signal("box_selected", nodes);
				}
			}

			reset_drag();
			emit_signal("box_selection_updated", Rect2());
			return true;
		}

		// Cancel box selection.
		if (b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::RIGHT) {
			reset_drag();
			emit_signal("box_selection_updated", Rect2());
			return true;
		}

		// Update box selection.
		if (m.is_valid()) {
			box_selecting_to = Transform2D(global_transform_callback.call()).affine_inverse().xform(m->get_position());
			emit_signal("box_selection_updated", area);
			return true;
		}
	}

	// Unselect everything.
	if (k.is_valid() && k->is_action_pressed(SNAME("ui_cancel"), false, true) && drag_type == DRAG_NONE) {
		emit_signal("clear_selection_requested", false);
	}

	return false;
}

bool CanvasItemManipulator::_gui_input_cancel_drag(const Ref<InputEvent> &p_event, bool p_reset_snap_targets, bool p_restore_bones) {
	Ref<InputEventMouseButton> b = p_event;
	if ((b.is_valid() && b->is_pressed() && b->get_button_index() == MouseButton::RIGHT) || DebuggerHelpers::is_shortcut_pressed(SHORTCUT_CANCEL_TRANSFORM, inputs)) {
		if (p_reset_snap_targets) {
			snap_target[0] = SNAP_TARGET_NONE;
			snap_target[1] = SNAP_TARGET_NONE;
		}

		emit_signal("restore_canvas_state_requested", drag_selection, p_restore_bones);
		reset_drag();
		emit_signal("update_canvas_requested");

		return true;
	}

	return false;
}

bool CanvasItemManipulator::_validate_drag_selection() {
	// Check if all nodes are still available.
	for (int i = 0; i < drag_selection.size(); i++) {
		if (!drag_selection[i]) {
			drag_selection.remove_at(i);
			i--;
		}
	}

	return !drag_selection.is_empty();
}

Rect2 CanvasItemManipulator::_encompass_selection_rect(const Array &p_selected) {
	ERR_FAIL_COND_V(p_selected.is_empty(), Rect2());

	// Handles the first element.
	CanvasItem *ci = Object::cast_to<CanvasItem>(p_selected.front());
	Rect2 rect = Rect2(ci->get_global_transform_with_canvas().xform(ci->_edit_get_rect().get_center()), Size2());

	// Expand with the other ones.
	for (const Variant &var : p_selected) {
		CanvasItem *ci2 = Object::cast_to<CanvasItem>(var);
		Transform2D xform = ci2->get_global_transform_with_canvas();
		Rect2 current_rect = ci2->_edit_get_rect();

		rect.expand_to(xform.xform(current_rect.position));
		rect.expand_to(xform.xform(current_rect.position + Vector2(current_rect.size.x, 0)));
		rect.expand_to(xform.xform(current_rect.position + current_rect.size));
		rect.expand_to(xform.xform(current_rect.position + Vector2(0, current_rect.size.y)));
	}

	return rect;
}

Point2 CanvasItemManipulator::anchor_to_position(const Control *p_control, const Vector2 p_anchor) {
	ERR_FAIL_NULL_V(p_control, Point2());

	Transform2D parent_transform = p_control->get_transform().affine_inverse();
	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	if (p_control->is_layout_rtl()) {
		return parent_transform.xform(parent_rect.position + Point2(parent_rect.size.x - parent_rect.size.x * p_anchor.x, parent_rect.size.y * p_anchor.y));
	} else {
		return parent_transform.xform(parent_rect.position + Point2(parent_rect.size.x * p_anchor.x, parent_rect.size.y * p_anchor.y));
	}
}

Point2 CanvasItemManipulator::_position_to_anchor(const Control *p_control, Point2 p_position) {
	ERR_FAIL_NULL_V(p_control, Vector2());

	Rect2 parent_rect = p_control->get_parent_anchorable_rect();

	Vector2 output;
	if (p_control->is_layout_rtl()) {
		output.x = (parent_rect.size.x == 0) ? 0.0 : (parent_rect.size.x - p_control->get_transform().xform(p_position).x - parent_rect.position.x) / parent_rect.size.x;
	} else {
		output.x = (parent_rect.size.x == 0) ? 0.0 : (p_control->get_transform().xform(p_position).x - parent_rect.position.x) / parent_rect.size.x;
	}
	output.y = (parent_rect.size.y == 0) ? 0.0 : (p_control->get_transform().xform(p_position).y - parent_rect.position.y) / parent_rect.size.y;
	return output;
}

void CanvasItemManipulator::_snap_if_closer_float(
		const real_t p_value,
		real_t &r_current_snap, SnapTarget &r_current_snap_target,
		const real_t p_target_value, const SnapTarget p_snap_target,
		const real_t p_radius) {
	const real_t radius = p_radius / zoom;
	const real_t dist = Math::abs(p_value - p_target_value);
	if ((p_radius < 0 || dist < radius) && (r_current_snap_target == SNAP_TARGET_NONE || dist < Math::abs(r_current_snap - p_value))) {
		r_current_snap = p_target_value;
		r_current_snap_target = p_snap_target;
	}
}

void CanvasItemManipulator::_snap_if_closer_point(
		Point2 p_value,
		Point2 &r_current_snap, SnapTarget (&r_current_snap_target)[2],
		Point2 p_target_value, const SnapTarget p_snap_target,
		const real_t rotation,
		const real_t p_radius) {
	Transform2D rot_trans = Transform2D(rotation, Point2());
	p_value = rot_trans.inverse().xform(p_value);
	p_target_value = rot_trans.inverse().xform(p_target_value);
	r_current_snap = rot_trans.inverse().xform(r_current_snap);

	_snap_if_closer_float(
			p_value.x,
			r_current_snap.x,
			r_current_snap_target[0],
			p_target_value.x,
			p_snap_target,
			p_radius);

	_snap_if_closer_float(
			p_value.y,
			r_current_snap.y,
			r_current_snap_target[1],
			p_target_value.y,
			p_snap_target,
			p_radius);

	r_current_snap = rot_trans.xform(r_current_snap);
}

void CanvasItemManipulator::_snap_other_nodes(
		const Point2 p_value,
		const Transform2D p_transform_to_snap,
		Point2 &r_current_snap, SnapTarget (&r_current_snap_target)[2],
		const SnapTarget p_snap_target, const Array &p_exceptions,
		const Node *p_current) {
	const CanvasItem *ci = Object::cast_to<CanvasItem>(p_current);

	// Check if the element is in the exception
	bool exception = false;
	for (const Variant &var : p_exceptions) {
		const CanvasItem *ex = Object::cast_to<CanvasItem>(var);
		if (ex == p_current) {
			exception = true;
			break;
		}
	}

	if (ci && !exception) {
		Transform2D ci_transform = ci->get_screen_transform();
		if (Math::fmod(ci_transform.get_rotation() - p_transform_to_snap.get_rotation(), (real_t)360.0) == 0.0) {
			if (ci->_edit_use_rect()) {
				Point2 begin = ci_transform.xform(ci->_edit_get_rect().get_position());
				Point2 end = ci_transform.xform(ci->_edit_get_rect().get_position() + ci->_edit_get_rect().get_size());

				_snap_if_closer_point(p_value, r_current_snap, r_current_snap_target, begin, p_snap_target, ci_transform.get_rotation());
				_snap_if_closer_point(p_value, r_current_snap, r_current_snap_target, end, p_snap_target, ci_transform.get_rotation());
			} else {
				Point2 position = ci_transform.xform(Point2());
				_snap_if_closer_point(p_value, r_current_snap, r_current_snap_target, position, p_snap_target, ci_transform.get_rotation());
			}
		}
	}
	for (int i = 0; i < p_current->get_child_count(); i++) {
		_snap_other_nodes(p_value, p_transform_to_snap, r_current_snap, r_current_snap_target, p_snap_target, p_exceptions, p_current->get_child(i));
	}
}

Point2 CanvasItemManipulator::snap_point(Point2 p_target, unsigned int p_modes, unsigned int p_forced_modes, const CanvasItem *p_self_canvas_item, const Array &p_other_nodes_exceptions) {
	snap_target[0] = SNAP_TARGET_NONE;
	snap_target[1] = SNAP_TARGET_NONE;

	bool is_snap_active = smart_snap != Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL);

	// Smart snap using the canvas position.
	Vector2 output = p_target;
	real_t rotation = 0.0;

	if (p_self_canvas_item) {
		rotation = p_self_canvas_item->get_screen_transform().get_rotation();

		// Parent sides and center.
		if ((is_snap_active && (snap_mode & SNAP_NODE_PARENT) && (p_modes & SNAP_NODE_PARENT)) || (p_forced_modes & SNAP_NODE_PARENT)) {
			if (const Control *c = Object::cast_to<Control>(p_self_canvas_item)) {
				Point2 begin = p_self_canvas_item->get_screen_transform().xform(anchor_to_position(c, Point2(0, 0)));
				Point2 end = p_self_canvas_item->get_screen_transform().xform(anchor_to_position(c, Point2(1, 1)));
				_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_PARENT, rotation);
				_snap_if_closer_point(p_target, output, snap_target, (begin + end) / 2.0, SNAP_TARGET_PARENT, rotation);
				_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_PARENT, rotation);
			} else if (const CanvasItem *parent_ci = Object::cast_to<CanvasItem>(p_self_canvas_item->get_parent())) {
				if (parent_ci->_edit_use_rect()) {
					Point2 begin = p_self_canvas_item->get_transform().affine_inverse().xform(parent_ci->_edit_get_rect().get_position());
					Point2 end = p_self_canvas_item->get_transform().affine_inverse().xform(parent_ci->_edit_get_rect().get_position() + parent_ci->_edit_get_rect().get_size());
					_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_PARENT, rotation);
					_snap_if_closer_point(p_target, output, snap_target, (begin + end) / 2.0, SNAP_TARGET_PARENT, rotation);
					_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_PARENT, rotation);
				} else {
					Point2 position = p_self_canvas_item->get_transform().affine_inverse().xform(Point2());
					_snap_if_closer_point(p_target, output, snap_target, position, SNAP_TARGET_PARENT, rotation);
				}
			}
		}

		// Self anchors.
		if ((is_snap_active && (snap_mode & SNAP_NODE_ANCHORS) && (p_modes & SNAP_NODE_ANCHORS)) || (p_forced_modes & SNAP_NODE_ANCHORS)) {
			if (const Control *c = Object::cast_to<Control>(p_self_canvas_item)) {
				Point2 begin = p_self_canvas_item->get_screen_transform().xform(anchor_to_position(c, Point2(c->get_anchor(SIDE_LEFT), c->get_anchor(SIDE_TOP))));
				Point2 end = p_self_canvas_item->get_screen_transform().xform(anchor_to_position(c, Point2(c->get_anchor(SIDE_RIGHT), c->get_anchor(SIDE_BOTTOM))));
				_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_SELF_ANCHORS, rotation);
				_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_SELF_ANCHORS, rotation);
			}
		}

		// Self sides.
		if ((is_snap_active && (snap_mode & SNAP_NODE_SIDES) && (p_modes & SNAP_NODE_SIDES)) || (p_forced_modes & SNAP_NODE_SIDES)) {
			if (p_self_canvas_item->_edit_use_rect()) {
				Point2 begin = p_self_canvas_item->get_screen_transform().xform(p_self_canvas_item->_edit_get_rect().get_position());
				Point2 end = p_self_canvas_item->get_screen_transform().xform(p_self_canvas_item->_edit_get_rect().get_position() + p_self_canvas_item->_edit_get_rect().get_size());
				_snap_if_closer_point(p_target, output, snap_target, begin, SNAP_TARGET_SELF, rotation);
				_snap_if_closer_point(p_target, output, snap_target, end, SNAP_TARGET_SELF, rotation);
			}
		}

		// Self center.
		if ((is_snap_active && (snap_mode & SNAP_NODE_CENTER) && (p_modes & SNAP_NODE_CENTER)) || (p_forced_modes & SNAP_NODE_CENTER)) {
			if (p_self_canvas_item->_edit_use_rect()) {
				Point2 center = p_self_canvas_item->get_screen_transform().xform(p_self_canvas_item->_edit_get_rect().get_center());
				_snap_if_closer_point(p_target, output, snap_target, center, SNAP_TARGET_SELF, rotation);
			} else {
				Point2 position = p_self_canvas_item->get_screen_transform().xform(Point2());
				_snap_if_closer_point(p_target, output, snap_target, position, SNAP_TARGET_SELF, rotation);
			}
		}
	}

	// Other nodes sides.
	if ((is_snap_active && (snap_mode & SNAP_OTHER_NODES) && (p_modes & SNAP_OTHER_NODES)) || (p_forced_modes & SNAP_OTHER_NODES)) {
		Transform2D to_snap_transform;
		Array exceptions;
		for (const Variant &var : p_other_nodes_exceptions) {
			exceptions.push_back(var);
		}
		if (p_self_canvas_item) {
			exceptions.push_back(p_self_canvas_item);
			to_snap_transform = p_self_canvas_item->get_screen_transform();
		}

		_snap_other_nodes(
				p_target, to_snap_transform,
				output, snap_target,
				SNAP_TARGET_OTHER_NODE,
				exceptions,
				SceneTree::get_singleton()->get_edited_scene_root());
	}

	// Guides.
	if (((is_snap_active && (snap_mode & SNAP_GUIDES) && (p_modes & SNAP_GUIDES)) || (p_forced_modes & SNAP_GUIDES)) && Math::fmod(rotation, (real_t)360.0) == 0.0) {
		Node *start = Object::cast_to<Node>(find_items_start_callback.call());
		if (start) {
			Array vguides = start->get_meta("_edit_vertical_guides_", Array());
			for (int i = 0; i < vguides.size(); i++) {
				_snap_if_closer_float(p_target.x, output.x, snap_target[0], vguides[i], SNAP_TARGET_GUIDE);
			}

			Array hguides = start->get_meta("_edit_horizontal_guides_", Array());
			for (int i = 0; i < hguides.size(); i++) {
				_snap_if_closer_float(p_target.y, output.y, snap_target[1], hguides[i], SNAP_TARGET_GUIDE);
			}
		}
	}

	// Grid.
	if (((grid_snap && (p_modes & SNAP_GRID)) || (p_forced_modes & SNAP_GRID)) && Math::fmod(rotation, (real_t)360.0) == 0.0) {
		Point2 offset = grid_offset;
		if (snap_relative) {
			if (drag_selection.size() == 1) {
				const Node2D *n2d = Object::cast_to<Node2D>(drag_selection.front());
				offset = n2d->get_global_position();
			} else if (drag_selection.size() > 0) {
				offset = _encompass_selection_rect(drag_selection).position;
			}
		}

		Point2 grid_output;
		grid_output.x = Math::snapped(p_target.x - offset.x, grid_step.x * Math::pow(2.0, grid_step_multiplier)) + offset.x;
		grid_output.y = Math::snapped(p_target.y - offset.y, grid_step.y * Math::pow(2.0, grid_step_multiplier)) + offset.y;
		_snap_if_closer_point(p_target, output, snap_target, grid_output, SNAP_TARGET_GRID, 0.0, -1.0);
	}

	// Pixel.
	if ((((snap_mode & SNAP_PIXEL) && (p_modes & SNAP_PIXEL)) || (p_forced_modes & SNAP_PIXEL)) && rotation == 0.0) {
		output = output.snappedf(1);
	}

	snap_transform = Transform2D(rotation, output);

	return output;
}

void CanvasItemManipulator::commit_drag() {
	if (_validate_drag_selection()) {
		const CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());

		switch (drag_type) {
			// Confirm the pivot move.
			case DRAG_PIVOT: {
				emit_signal("commit_canvas_state_requested",
						drag_selection,
						vformat(RTR("Set CanvasItem \"%s\" Pivot Offset to (%d, %d)"),
								ci->get_name(),
								ci->_edit_get_pivot().x,
								ci->_edit_get_pivot().y),
						false);
			} break;

			// Confirm the node rotation.
			case DRAG_ROTATE: {
				if (drag_selection.size() != 1) {
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Rotate %d CanvasItems"), drag_selection.size()),
							true);
				} else {
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Rotate CanvasItem \"%s\" to %d degrees"),
									ci->get_name(),
									Math::rad_to_deg(ci->_edit_get_rotation())),
							true);
				}
			} break;

			// Confirm new anchor position.
			case DRAG_ANCHOR_TOP_LEFT:
			case DRAG_ANCHOR_TOP_RIGHT:
			case DRAG_ANCHOR_BOTTOM_RIGHT:
			case DRAG_ANCHOR_BOTTOM_LEFT:
			case DRAG_ANCHOR_ALL: {
				emit_signal("commit_canvas_state_requested",
						drag_selection,
						vformat(RTR("Move CanvasItem \"%s\" Anchor"), ci->get_name()),
						false);

				snap_target[0] = SNAP_TARGET_NONE;
				snap_target[1] = SNAP_TARGET_NONE;
			} break;

			// Confirm resize.
			case DRAG_LEFT:
			case DRAG_RIGHT:
			case DRAG_TOP:
			case DRAG_BOTTOM:
			case DRAG_TOP_LEFT:
			case DRAG_TOP_RIGHT:
			case DRAG_BOTTOM_LEFT:
			case DRAG_BOTTOM_RIGHT: {
				const Node2D *node2d = Object::cast_to<Node2D>(ci);
				if (node2d) {
					// Extends from Node2D.
					// Node2D doesn't have an actual stored rect size, unlike Controls.
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Scale Node2D \"%s\" to (%s, %s)"),
									ci->get_name(),
									Math::snapped(ci->_edit_get_scale().x, 0.01),
									Math::snapped(ci->_edit_get_scale().y, 0.01)),
							true);
				} else {
					// Extends from Control.
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(
									RTR("Resize Control \"%s\" to (%d, %d)"),
									ci->get_name(),
									ci->_edit_get_rect().size.x,
									ci->_edit_get_rect().size.y),
							true);
				}

				snap_target[0] = SNAP_TARGET_NONE;
				snap_target[1] = SNAP_TARGET_NONE;
			} break;

			// Confirm resize.
			case DRAG_SCALE_BOTH:
			case DRAG_SCALE_X:
			case DRAG_SCALE_Y: {
				if (drag_selection.size() != 1) {
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Scale %d CanvasItems"), drag_selection.size()),
							true);
				} else {
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Scale CanvasItem \"%s\" to (%s, %s)"),
									ci->get_name(),
									Math::snapped(ci->_edit_get_scale().x, 0.01),
									Math::snapped(ci->_edit_get_scale().y, 0.01)),
							true);
				}
			} break;

			// Confirm the canvas items move.
			case DRAG_MOVE:
			case DRAG_MOVE_X:
			case DRAG_MOVE_Y: {
				ERR_FAIL_NULL(viewport);
				if (Transform2D(global_transform_callback.call()).affine_inverse().xform(viewport->get_mouse_position()) != drag_from) {
					if (drag_selection.size() != 1) {
						emit_signal("commit_canvas_state_requested",
								drag_selection,
								vformat(RTR("Move %d CanvasItems"), drag_selection.size()),
								true);
					} else {
						emit_signal("commit_canvas_state_requested",
								drag_selection,
								vformat(
										RTR("Move CanvasItem \"%s\" to (%d, %d)"),
										ci->get_name(),
										ci->_edit_get_position().x,
										ci->_edit_get_position().y),
								true);
					}
				}

				// Make sure smart snapping lines disappear.
				snap_target[0] = SNAP_TARGET_NONE;
				snap_target[1] = SNAP_TARGET_NONE;
			} break;

			// Confirm the canvas items move by arrow keys.
			case DRAG_KEY_MOVE: {
				if (tool != TOOL_SELECT && tool != TOOL_MOVE) {
					return;
				}

				if (drag_selection.size() > 1) {
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Move %d CanvasItems"), drag_selection.size()),
							true);
				} else if (drag_selection.size() == 1) {
					emit_signal("commit_canvas_state_requested",
							drag_selection,
							vformat(RTR("Move CanvasItem \"%s\" to (%d, %d)"),
									ci->get_name(),
									ci->_edit_get_position().x,
									ci->_edit_get_position().y),
							true);
				}
			} break;

			default:
				break;
		}
	}

	reset_drag();
	emit_signal("update_canvas_requested");
}

bool CanvasItemManipulator::is_node_locked(const Node *p_node) {
	return p_node->get_meta("_edit_lock_", false);
}

bool CanvasItemManipulator::_is_node_movable(const Node *p_node, bool p_popup_warning) {
	if (is_node_locked(p_node)) {
		return false;
	}

	if (Object::cast_to<Control>(p_node) && Object::cast_to<Container>(p_node->get_parent())) {
		if (p_popup_warning) {
			emit_signal("unmovable_items_warn_requested");
		}
		return false;
	}

	return true;
}

void CanvasItemManipulator::_update_cursor_shape() {
	Input::CursorShape new_cursor = Input::CursorShape::CURSOR_ARROW;
	switch (tool) {
		case TOOL_MOVE: {
			new_cursor = Input::CursorShape::CURSOR_MOVE;
		} break;

		case TOOL_EDIT_PIVOT: {
			new_cursor = Input::CursorShape::CURSOR_CROSS;
		} break;

		case TOOL_PAN: {
			new_cursor = Input::CursorShape::CURSOR_DRAG;
		} break;

		case TOOL_RULER: {
			new_cursor = Input::CursorShape::CURSOR_CROSS;
		} break;

		default:
			break;
	}

	if (drag_type == DRAG_NONE) {
		if (is_hovering_h_guide) {
			new_cursor = Input::CursorShape::CURSOR_VSIZE;
		} else if (is_hovering_v_guide) {
			new_cursor = Input::CursorShape::CURSOR_HSIZE;
		}
	} else {
		// Compute an eventual rotation of the cursor.
		const Input::CursorShape rotation_array[4] = {
			Input::CursorShape::CURSOR_HSIZE,
			Input::CursorShape::CURSOR_BDIAGSIZE,
			Input::CursorShape::CURSOR_VSIZE,
			Input::CursorShape::CURSOR_FDIAGSIZE,
		};
		int rotation_array_index = 0;

		if (drag_selection.size() == 1) {
			const CanvasItem *ci = Object::cast_to<CanvasItem>(drag_selection.front());
			const double angle = Math::fposmod((double)ci->get_global_transform_with_canvas().get_rotation(), Math::PI);
			if (angle > Math::PI * 7.0 / 8.0) {
				rotation_array_index = 0;
			} else if (angle > Math::PI * 5.0 / 8.0) {
				rotation_array_index = 1;
			} else if (angle > Math::PI * 3.0 / 8.0) {
				rotation_array_index = 2;
			} else if (angle > Math::PI * 1.0 / 8.0) {
				rotation_array_index = 3;
			} else {
				rotation_array_index = 0;
			}
		}

		switch (drag_type) {
			case DRAG_LEFT:
			case DRAG_RIGHT: {
				new_cursor = rotation_array[rotation_array_index];
			} break;

			case DRAG_V_GUIDE: {
				new_cursor = Input::CursorShape::CURSOR_HSIZE;
			} break;

			case DRAG_TOP:
			case DRAG_BOTTOM: {
				new_cursor = rotation_array[(rotation_array_index + 2) % 4];
			} break;

			case DRAG_H_GUIDE: {
				new_cursor = Input::CursorShape::CURSOR_VSIZE;
			} break;

			case DRAG_TOP_LEFT:
			case DRAG_BOTTOM_RIGHT: {
				new_cursor = rotation_array[(rotation_array_index + 3) % 4];
			} break;

			case DRAG_DOUBLE_GUIDE: {
				new_cursor = Input::CursorShape::CURSOR_FDIAGSIZE;
			} break;

			case DRAG_TOP_RIGHT:
			case DRAG_BOTTOM_LEFT: {
				new_cursor = rotation_array[(rotation_array_index + 1) % 4];
			} break;

			case DRAG_MOVE: {
				new_cursor = Input::CursorShape::CURSOR_MOVE;
			} break;

			default:
				break;
		}
	}

	if (cursor_shape != new_cursor) {
		cursor_shape = new_cursor;
		emit_signal("cursor_shape_changed");
	}
}

void CanvasItemManipulator::get_canvas_items_at_pos(const Point2 &p_pos, Vector<DebuggerHelpers::SelectResult> &r_items, bool p_allow_locked) {
	Node *start = Object::cast_to<Node>(find_items_start_callback.call());
	if (!start) {
		return;
	}

	find_canvas_items_at_pos(p_pos, start, r_items);

	// Remove invalid results.
	bool is_editor = Engine::get_singleton()->is_editor_hint();
	for (int i = 0; i < r_items.size(); i++) {
		Node *node = r_items[i].item;

		// Make sure the selected node is in the current scene, or editable.
		if (is_editor && node && node != SceneTree::get_singleton()->get_edited_scene_root()) {
			node = start->get_deepest_editable_node(node);
		}

		CanvasItem *ci = Object::cast_to<CanvasItem>(node);
		if (!ci) {
			r_items.remove_at(i);
			i--;
			continue;
		}

		if (!p_allow_locked) {
			// Replace the node by the group if grouped.
			while (node && node != start->get_parent()) {
				CanvasItem *ci_tmp = Object::cast_to<CanvasItem>(node);
				if (ci_tmp && node->has_meta("_edit_group_")) {
					ci = ci_tmp;
				}
				node = node->get_parent();
			}
		}

		// Check if the canvas item is already in the list (for groups or scenes).
		bool duplicate = false;
		for (int j = 0; j < i; j++) {
			if (r_items[j].item == ci) {
				duplicate = true;
				break;
			}
		}

		//	Remove the item if invalid.
		bool in_editor_scene = is_editor && ci != start && ci->get_owner() != start && !start->is_editable_instance(ci->get_owner());
		if (duplicate || in_editor_scene || (!p_allow_locked && is_node_locked(ci))) {
			r_items.remove_at(i);
			i--;
		} else {
			r_items.write[i].item = ci;
		}
	}
}

void CanvasItemManipulator::find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<DebuggerHelpers::SelectResult> &r_items, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	bool is_editor = Engine::get_singleton()->is_editor_hint();
	SubViewport *vp = Object::cast_to<SubViewport>(p_node);

	if (!is_editor && vp) {
		return; // FIXME: Make subviewport selection work at runtime.
	}

	Transform2D xform = p_canvas_xform;

	if (is_editor) {
		if (CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node)) {
			xform = cl->get_transform();
		} else if (vp) {
			if (!vp->is_visible_subviewport()) {
				return;
			}
			xform = vp->get_popup_base_transform();
			if (!vp->get_visible_rect().has_point(xform.affine_inverse().xform(p_pos))) {
				return;
			}
		}
	}

	CanvasItem *ci = Object::cast_to<CanvasItem>(p_node);

	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		if (ci) {
			if (!ci->is_set_as_top_level()) {
				find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, p_parent_xform * ci->get_transform(), xform);
			} else {
				find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, ci->get_transform(), xform);
			}
		} else {
			find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, Transform2D(), xform);
		}
	}

	if (!ci || !ci->is_visible_in_tree()) {
		return;
	}

	if (!ci->is_set_as_top_level()) {
		xform *= p_parent_xform;
	}

	Point2 pos = p_pos;

	// Cameras don't affect `CanvasLayer`s.
	// Only check at runtime, as cameras aren't active in the editor.
	if (!is_editor && (!ci->get_canvas_layer_node() || ci->get_canvas_layer_node()->is_following_viewport())) {
		Window *root = SceneTree::get_singleton()->get_root();
		pos = root->get_canvas_transform().affine_inverse().xform(p_pos);
	}

	xform = (xform * ci->get_transform()).affine_inverse();
	const real_t local_grab_distance = xform.basis_xform(Vector2(grab_distance, 0)).length() / zoom;
	if (ci->_edit_is_selected_on_click(xform.xform(pos), local_grab_distance)) {
		Node2D *node = Object::cast_to<Node2D>(ci);

		DebuggerHelpers::SelectResult res;
		res.item = ci;
		res.has_order = node;
		if (is_editor) {
			res.order = node ? node->get_z_index() : 0;
		} else {
			res.order = ci->get_effective_z_index() + ci->get_canvas_layer();
		}

		r_items.push_back(res);
	}
}

void CanvasItemManipulator::find_canvas_items_in_rect(const Rect2 &p_rect, Node *p_node, Vector<DebuggerHelpers::SelectResult> &r_items, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	if (!p_node) {
		return;
	}

	bool is_editor = Engine::get_singleton()->is_editor_hint();
	Viewport *vp = Object::cast_to<Viewport>(p_node);

	if (!is_editor && vp && vp != SceneTree::get_singleton()->get_root()) {
		return; // FIXME: Make subviewport selection work at runtime.
	}

	Transform2D xform = p_canvas_xform;

	if (is_editor) {
		if (CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node)) {
			xform = cl->get_transform();
		} else if (vp) {
			if (!vp->is_visible_subviewport()) {
				return;
			}
			xform = vp->get_popup_base_transform();
			if (!vp->get_visible_rect().intersects(xform.affine_inverse().xform(p_rect))) {
				return;
			}
		}
	}

	CanvasItem *ci = Object::cast_to<CanvasItem>(p_node);

	bool editable = true;
	if (is_editor) {
		Node *start = Object::cast_to<Node>(find_items_start_callback.call());
		editable = !is_editor || p_node == start || p_node->get_owner() == start || p_node == start->get_deepest_editable_node(p_node);
	}

	bool lock_children = p_node->get_meta("_edit_group_", false);
	bool locked = is_node_locked(p_node);

	if (!lock_children || !editable) {
		for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
			if (ci) {
				if (!ci->is_set_as_top_level()) {
					find_canvas_items_in_rect(p_rect, p_node->get_child(i), r_items, p_parent_xform * ci->get_transform(), p_canvas_xform);
				} else {
					find_canvas_items_in_rect(p_rect, p_node->get_child(i), r_items, ci->get_transform(), p_canvas_xform);
				}
			} else {
				CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
				find_canvas_items_in_rect(p_rect, p_node->get_child(i), r_items, Transform2D(), cl ? cl->get_transform() : p_canvas_xform);
			}
		}
	}

	if (!ci || !ci->is_visible_in_tree() || locked || !editable) {
		return;
	}

	if (!ci->is_set_as_top_level()) {
		xform *= p_parent_xform;
	}

	Rect2 rect = p_rect;
	// Cameras don't affect `CanvasLayer`s.
	// Only check at runtime, as cameras aren't active in the editor.
	if (!is_editor && (!ci->get_canvas_layer_node() || ci->get_canvas_layer_node()->is_following_viewport())) {
		Window *root = SceneTree::get_singleton()->get_root();
		rect = root->get_canvas_transform().affine_inverse().xform(p_rect);
	}
	rect = (xform * ci->get_transform()).affine_inverse().xform(rect);

	bool selected = false;
	if (ci->_edit_use_rect()) {
		Rect2 ci_rect = ci->_edit_get_rect();
		if (rect.has_point(ci_rect.position) &&
				rect.has_point(ci_rect.position + Vector2(ci_rect.size.x, 0)) &&
				rect.has_point(ci_rect.position + Vector2(ci_rect.size.x, ci_rect.size.y)) &&
				rect.has_point(ci_rect.position + Vector2(0, ci_rect.size.y))) {
			selected = true;
		}
	} else if (rect.has_point(Point2())) {
		selected = true;
	}

	if (selected) {
		Node2D *node = Object::cast_to<Node2D>(ci);

		DebuggerHelpers::SelectResult res;
		res.item = ci;
		res.has_order = node;
		if (is_editor) {
			res.order = node ? node->get_z_index() : 0;
		} else {
			res.order = ci->get_effective_z_index() + ci->get_canvas_layer();
		}

		r_items.push_back(res);
	}
}

bool CanvasItemManipulator::gui_input(const Ref<InputEvent> &p_event) {
	bool accepted = _gui_input_rulers_and_guides(p_event) ||
			plugin_input_callback.call(p_event) ||
			_gui_input_open_scene_on_double_click(p_event) ||
			_gui_input_scale(p_event) ||
			_gui_input_pivot(p_event) ||
			_gui_input_resize(p_event) ||
			_gui_input_rotate(p_event) ||
			_gui_input_move(p_event) ||
			_gui_input_anchors(p_event) ||
			_gui_input_ruler_tool(p_event) ||
			_gui_input_select(p_event);

	_update_cursor_shape();

	return accepted;
}

void CanvasItemManipulator::reset_drag() {
	drag_type = DRAG_NONE;
	drag_selection.clear();
}

bool CanvasItemManipulator::reset_temp_pivot() {
	if (temp_pivot != Vector2(Math::INF, Math::INF)) {
		temp_pivot = Vector2(Math::INF, Math::INF);
		return true;
	}

	return false;
}

void CanvasItemManipulator::set_tool(Tool p_tool) {
	if (drag_type != DRAG_NONE) {
		commit_drag();
	}

	tool = p_tool;

	if (p_tool == TOOL_EDIT_PIVOT && Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		// Special action that places temporary rotation pivot in the middle of the selection.
		Array selection;
		get_selection_callback.call(selection);

		if (!selection.is_empty()) {
			Vector2 center;
			for (const Variant &var : selection) {
				const CanvasItem *ci = Object::cast_to<CanvasItem>(var);
				center += ci->get_viewport()->get_popup_base_transform().xform(ci->_edit_get_position());
			}
			temp_pivot = center / selection.size();
		}
	}
}

void CanvasItemManipulator::set_shortcut(const ShortcutName p_name, const Ref<Shortcut> &p_shortcut) {
	ERR_FAIL_INDEX(0, SHORTCUT_MAX);
	ERR_FAIL_COND(p_shortcut.is_null());
	inputs[p_name] = p_shortcut;
}

void CanvasItemManipulator::set_callbacks(const Callable p_find_items_start, const Callable p_point_selected, const Callable p_get_selection, const Callable p_local_xform, const Callable p_global_xform, const Callable p_local_mouse_pos, const Callable p_plugin_input) {
	find_items_start_callback = p_find_items_start;
	point_selected_callback = p_point_selected;
	get_selection_callback = p_get_selection;
	local_transform_callback = p_local_xform;
	global_transform_callback = p_global_xform;
	local_mouse_pos_callback = p_local_mouse_pos;
	plugin_input_callback = p_plugin_input;
}

void CanvasItemManipulator::_bind_methods() {
	ADD_SIGNAL(MethodInfo("box_selected", PropertyInfo(Variant::ARRAY, "selection")));
	ADD_SIGNAL(MethodInfo("box_selection_updated", PropertyInfo(Variant::RECT2, "area")));
	ADD_SIGNAL(MethodInfo("clear_selection_requested", PropertyInfo(Variant::BOOL, "selected_from_canvas")));
	ADD_SIGNAL(MethodInfo("selection_menu_requested", PropertyInfo(Variant::ARRAY, "selection"), PropertyInfo(Variant::VECTOR2, "position"), PropertyInfo(Variant::BOOL, "append")));

	ADD_SIGNAL(MethodInfo("update_canvas_requested"));

	ADD_SIGNAL(MethodInfo("tool_menu_requested", PropertyInfo(Variant::VECTOR2, "position")));
	ADD_SIGNAL(MethodInfo("scene_double_clicked", PropertyInfo(Variant::STRING, "path")));

	ADD_SIGNAL(MethodInfo("save_canvas_state_requested", PropertyInfo(Variant::ARRAY, "selection"), PropertyInfo(Variant::BOOL, "save_bones")));
	ADD_SIGNAL(MethodInfo("restore_canvas_state_requested", PropertyInfo(Variant::ARRAY, "selection"), PropertyInfo(Variant::BOOL, "restore_bones")));
	ADD_SIGNAL(MethodInfo("commit_canvas_state_requested", PropertyInfo(Variant::ARRAY, "selection"), PropertyInfo(Variant::STRING, "message"), PropertyInfo(Variant::BOOL, "commit_bones")));

	ADD_SIGNAL(MethodInfo("commit_guide_meta_requested", PropertyInfo(Variant::STRING, "message"), PropertyInfo(Variant::DICTIONARY, "meta")));

	ADD_SIGNAL(MethodInfo("unmovable_items_warn_requested"));
	ADD_SIGNAL(MethodInfo("locked_items_warn_requested"));
	ADD_SIGNAL(MethodInfo("anchors_mode_warn_requested"));

	ADD_SIGNAL(MethodInfo("cursor_shape_changed"));
}
#endif // DEBUG_ENABLED
