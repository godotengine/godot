/**************************************************************************/
/*  curve_editor_plugin.cpp                                               */
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

#include "curve_editor_plugin.h"

#include "core/input/input.h"
#include "core/math/geometry_2d.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/keyboard.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_spin_slider.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/separator.h"
#include "scene/resources/image_texture.h"

CurveEdit::CurveEdit() {
	set_focus_mode(FOCUS_ALL);
	set_clip_contents(true);
}

void CurveEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_selected_indexes", "indexes"), &CurveEdit::set_selected_indexes);
}

void CurveEdit::set_curve(Ref<Curve> p_curve) {
	if (p_curve == curve) {
		return;
	}

	if (curve.is_valid()) {
		curve->disconnect_changed(callable_mp(this, &CurveEdit::_curve_changed));
		curve->disconnect(Curve::SIGNAL_RANGE_CHANGED, callable_mp(this, &CurveEdit::_curve_changed));
		curve->disconnect(Curve::SIGNAL_DOMAIN_CHANGED, callable_mp(this, &CurveEdit::_curve_changed));
	}

	curve = p_curve;

	if (curve.is_valid()) {
		curve->connect_changed(callable_mp(this, &CurveEdit::_curve_changed));
		curve->connect(Curve::SIGNAL_RANGE_CHANGED, callable_mp(this, &CurveEdit::_curve_changed));
		curve->connect(Curve::SIGNAL_DOMAIN_CHANGED, callable_mp(this, &CurveEdit::_curve_changed));
	}

	// Note: if you edit a curve, then set another, and try to undo,
	// it will normally apply on the previous curve, but you won't see it.
}

Ref<Curve> CurveEdit::get_curve() {
	return curve;
}

void CurveEdit::set_snap_enabled(bool p_enabled) {
	snap_enabled = p_enabled;
	queue_redraw();
	if (curve.is_valid()) {
		if (snap_enabled) {
			curve->set_meta(SNAME("_snap_enabled"), true);
		} else {
			curve->remove_meta(SNAME("_snap_enabled"));
		}
	}
}

void CurveEdit::set_snap_count(int p_snap_count) {
	snap_count = p_snap_count;
	queue_redraw();
	if (curve.is_valid()) {
		if (snap_count != CurveEditor::DEFAULT_SNAP) {
			curve->set_meta(SNAME("_snap_count"), snap_count);
		} else {
			curve->remove_meta(SNAME("_snap_count"));
		}
	}
}

Size2 CurveEdit::get_minimum_size() const {
	return Vector2(64, MAX(135, get_size().x * ASPECT_RATIO)) * EDSCALE;
}

Control::CursorShape CurveEdit::get_cursor_shape(const Point2 &p_pos) const {
	if (grabbing == GRAB_MOVE || selected_indexes.has(hovered_index)) {
		return CursorShape::CURSOR_MOVE;
	} else if (grabbing == GRAB_SCALE || scaling_hovered_handle != Vector2i()) {
		Vector2i handle = (grabbing == GRAB_SCALE) ? scaling_handle : scaling_hovered_handle;
		if (handle == Vector2i(1, 1) || handle == Vector2i(-1, -1)) {
			return CURSOR_FDIAGSIZE;
		} else if (handle == Vector2i(1, -1) || handle == Vector2i(-1, 1)) {
			return CURSOR_BDIAGSIZE;
		} else if (abs(handle.x) == 1) {
			return CURSOR_HSIZE;
		} else if (abs(handle.y) == 1) {
			return CURSOR_VSIZE;
		}
	}
	return CURSOR_ARROW;
}

void CurveEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_MOUSE_EXIT: {
			if (hovered_index != -1 || hovered_tangent_index != TANGENT_NONE) {
				hovered_index = -1;
				hovered_tangent_index = TANGENT_NONE;
				queue_redraw();
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			float gizmo_scale = EDITOR_GET("interface/touchscreen/scale_gizmo_handles");
			point_radius = Math::round(BASE_POINT_RADIUS * get_theme_default_base_scale() * gizmo_scale);
			hover_radius = Math::round(BASE_HOVER_RADIUS * get_theme_default_base_scale() * gizmo_scale);
			tangent_radius = Math::round(BASE_TANGENT_RADIUS * get_theme_default_base_scale() * gizmo_scale);
			tangent_hover_radius = Math::round(BASE_TANGENT_HOVER_RADIUS * get_theme_default_base_scale() * gizmo_scale);
			tangent_length = Math::round(BASE_TANGENT_LENGTH * get_theme_default_base_scale());
			accent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
			box_selection_fill_color = get_theme_color(SNAME("box_selection_fill_color"), EditorStringName(Editor));
			box_selection_stroke_color = get_theme_color(SNAME("box_selection_stroke_color"), EditorStringName(Editor));
		} break;
		case NOTIFICATION_DRAW: {
			_redraw();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				grabbing = GRAB_NONE;
			}
		} break;
		case NOTIFICATION_RESIZED: {
			update_scaling_initial_rect();
		} break;
	}
}

void CurveEdit::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());
	if (curve.is_null()) {
		return;
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		// Deleting points or making tangents linear.
		if (k->is_pressed() && k->get_keycode() == Key::KEY_DELETE) {
			if (selected_tangent_index != TANGENT_NONE) {
				toggle_linear(selected_indexes.front()->get(), selected_tangent_index);
			} else if (!selected_indexes.is_empty()) {
				if (grabbing == GRAB_ADD) {
					curve->remove_point(selected_indexes.front()->get()); // Point is temporary, so remove directly from curve.
					set_selected_indexes(PackedInt32Array());
				} else {
					delete_selection();
				}
				grabbing = GRAB_NONE;
				hovered_index = -1;
				hovered_tangent_index = TANGENT_NONE;
			}
			accept_event();
		}

		if (k->get_keycode() == Key::SHIFT || k->get_keycode() == Key::ALT) {
			queue_redraw(); // Redraw to show the axes or constraints.
		}
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed()) {
		Vector2 mpos = mb->get_position();

		if (mb->get_button_index() == MouseButton::RIGHT || mb->get_button_index() == MouseButton::MIDDLE) {
			if (mb->get_button_index() == MouseButton::RIGHT && grabbing == GRAB_MOVE) {
				// Move a point to its old position.
				set_grab_move_offset(Vector2());
				hovered_index = get_point_at(mpos);
				grabbing = GRAB_NONE;
			} else if (mb->get_button_index() == MouseButton::RIGHT && grabbing == GRAB_SCALE) {
				// Move a point to its old position.
				set_grab_scale_factor(Vector2(1.0, 1.0));
				hovered_index = get_point_at(mpos);
				grabbing = GRAB_NONE;
			} else {
				// Remove a point or make a tangent linear.
				selected_tangent_index = get_tangent_at(mpos);
				if (selected_tangent_index != TANGENT_NONE) {
					int selected_idx = selected_indexes.front()->get();
					toggle_linear(selected_idx, selected_tangent_index);
				} else {
					int point_to_remove = get_point_at(mpos);
					if (point_to_remove == -1) {
						set_selected_indexes(PackedInt32Array()); // Nothing on the place of the click, just deselect the points.
					} else {
						if (grabbing == GRAB_ADD) {
							curve->remove_point(point_to_remove); // Point is temporary, so remove directly from curve.
							set_selected_indexes(PackedInt32Array());
						} else {
							remove_point(point_to_remove);
						}
						hovered_index = get_point_at(mpos);
						grabbing = GRAB_NONE;
					}
				}
			}
		}

		// Selecting or creating points.
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (grabbing == GRAB_NONE) {
				selected_tangent_index = get_tangent_at(mpos);
				if (selected_tangent_index == TANGENT_NONE) {
					hovered_index = get_point_at(mpos);
					if (hovered_index != -1) {
						if (mb->is_shift_pressed()) {
							if (selected_indexes.has(hovered_index)) {
								selected_indexes.erase(hovered_index);
							} else {
								selected_indexes.insert(hovered_index);
							}
							update_scaling_initial_rect();
						} else if (mb->is_command_or_control_pressed()) {
							selected_indexes.erase(hovered_index);
						} else {
							update_scaling_initial_rect();
							if (!selected_indexes.has(hovered_index)) { // If is part of selection do not replace it.
								set_selected_indexes(PackedInt32Array({ hovered_index }));
							}
						}
					}
				}
				queue_redraw();
			}

			if (hovered_index != -1) {
				// If an existing point/tangent was grabbed, remember a few things about it.
				grabbing = GRAB_MOVE;
				initialize_grab_state();
				initial_grab_pos = get_world_pos(mb->get_position());
				if (selected_indexes.size() == 1) {
					int selected_idx = selected_indexes.front()->get();
					if (selected_idx > 0) {
						initial_grab_left_tangent = curve->get_point_left_tangent(selected_idx);
					}
					if (selected_indexes.front()->get() < curve->get_point_count() - 1) {
						initial_grab_right_tangent = curve->get_point_right_tangent(selected_idx);
					}
				}
			} else if (scaling_hovered_handle != Vector2i()) {
				grabbing = GRAB_SCALE;
				initialize_grab_state();
				scaling_handle = scaling_hovered_handle;
				scaling_pivot_point.x = scaling_handle.x == 1 ? scaling_initial_rect.position.x : scaling_initial_rect.get_end().x;
				scaling_pivot_point.y = scaling_handle.y == 1 ? scaling_initial_rect.position.y : scaling_initial_rect.get_end().y;
				initial_grab_pos = get_world_pos(mb->get_position());
			} else {
				if (mb->is_double_click()) {
					// Adding a new point. Insert a temporary point for the user to adjust, so it's not in the undo/redo.
					Vector2 new_pos = clamp_world_pos(get_world_pos(mpos));
					if (snap_enabled || mb->is_command_or_control_pressed()) {
						new_pos.x = Math::snapped(new_pos.x - curve->get_min_domain(), curve->get_domain_range() / snap_count) + curve->get_min_domain();
						new_pos.y = Math::snapped(new_pos.y - curve->get_min_value(), curve->get_value_range() / snap_count) + curve->get_min_value();
					}

					new_pos.x = get_offset_without_collision(-1, new_pos.x, mpos.x >= get_view_pos(new_pos).x);

					// Add a temporary point for the user to adjust before adding it permanently.
					initial_grab_pos = new_pos;
					int new_idx = curve->add_point_no_update(new_pos);
					set_selected_indexes(PackedInt32Array({ new_idx }));
					initialize_grab_state();
					grabbing = GRAB_ADD;
				} else {
					// Start box selection.
					initial_grab_pos = clamp_world_pos(get_world_pos(mpos));
					box_selection = Rect2(initial_grab_pos, Size2());
					grabbing = GRAB_SELECT;
				}
				queue_redraw();
			}
		}
	}

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
		if (selected_tangent_index != TANGENT_NONE) {
			// Finish moving a tangent control.
			int selected_idx = selected_indexes.front()->get();
			if (selected_idx == 0) {
				set_point_right_tangent(selected_idx, curve->get_point_right_tangent(selected_idx));
			} else if (selected_idx == curve->get_point_count() - 1) {
				set_point_left_tangent(selected_idx, curve->get_point_left_tangent(selected_idx));
			} else {
				set_point_tangents(selected_idx, curve->get_point_left_tangent(selected_idx), curve->get_point_right_tangent(selected_idx));
			}
			grabbing = GRAB_NONE;
		} else if (grabbing == GRAB_MOVE) {
			if (!initial_grab_pos.is_equal_approx(get_world_pos(mb->get_position()))) {
				finish_grab_transform();
			}
			grabbing = GRAB_NONE;
		} else if (grabbing == GRAB_ADD) {
			// Finish inserting a new point. Remove the temporary point and insert a permanent one in its place.
			int selected_idx = selected_indexes.front()->get();
			Vector2 new_pos = curve->get_point_position(selected_idx);
			curve->remove_point(selected_idx);
			add_point(new_pos);
			grabbing = GRAB_NONE;
		} else if (grabbing == GRAB_SCALE) {
			if (!initial_grab_pos.is_equal_approx(get_world_pos(mb->get_position()))) {
				finish_grab_transform();
			}
			grabbing = GRAB_NONE;
			update_scaling_initial_rect();
		} else if (grabbing == GRAB_SELECT) {
			RBSet<int> new_selection;
			if (mb->is_command_or_control_pressed() || mb->is_shift_pressed()) {
				new_selection = selected_indexes;
			}
			for (int i = 0; i < curve->get_point_count(); i++) {
				if (box_selection.has_point(curve->get_point(i).position)) {
					if (mb->is_command_or_control_pressed()) {
						new_selection.erase(i);
					} else {
						new_selection.insert(i);
					}
				}
			}
			selected_indexes = new_selection;
			grabbing = GRAB_NONE;
			update_scaling_initial_rect();
		}
		queue_redraw();
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		Vector2 mpos = mm->get_position();

		if (grabbing != GRAB_NONE && curve.is_valid()) {
			if (selected_tangent_index == TANGENT_NONE) {
				switch (grabbing) {
					case GRAB_ADD:
					case GRAB_MOVE: {
						Vector2 new_pos = clamp_world_pos(get_world_pos(mpos));

						if (snap_enabled || mm->is_command_or_control_pressed()) {
							new_pos.x = Math::snapped(new_pos.x - curve->get_min_domain(), curve->get_domain_range() / snap_count) + curve->get_min_domain();
							new_pos.y = Math::snapped(new_pos.y - curve->get_min_value(), curve->get_value_range() / snap_count) + curve->get_min_value();
						}

						// Allow to snap to axes with Shift.
						if (mm->is_shift_pressed()) {
							Vector2 initial_mpos = get_view_pos(initial_grab_pos);
							if (Math::abs(mpos.x - initial_mpos.x) > Math::abs(mpos.y - initial_mpos.y)) {
								new_pos.y = initial_grab_pos.y;
							} else {
								new_pos.x = initial_grab_pos.x;
							}
						}

						// Allow to constraint a single point between the adjacent two with Alt.
						if (mm->is_alt_pressed() && selected_indexes.size() == 1) {
							int initial_selected_index = -1;
							for (uint32_t i = 0; i < grab_initial_state.size(); i++) {
								if (grab_initial_state[i].selected) {
									initial_selected_index = i;
									break;
								}
							}
							float prev_point_offset = (initial_selected_index > 0) ? (curve->get_point_position(initial_selected_index - 1).x + 0.00001) : curve->get_min_domain();
							float next_point_offset = (initial_selected_index < curve->get_point_count() - 1) ? (curve->get_point_position(initial_selected_index + 1).x - 0.00001) : curve->get_max_domain();
							new_pos.x = CLAMP(new_pos.x, prev_point_offset, next_point_offset);
						}

						Vector2 offset = new_pos - initial_grab_pos;
						set_grab_move_offset(offset);
					} break;

					case GRAB_SCALE: {
						Vector2 scale_factor = Vector2(1, 1);

						Vector2 ref_pos = mpos - scaling_mouse_offset_from_handle;
						if (snap_enabled || mm->is_command_or_control_pressed()) {
							Vector2 world_pos = get_world_pos(ref_pos);
							world_pos.x = Math::snapped(world_pos.x - curve->get_min_domain(), curve->get_domain_range() / snap_count) + curve->get_min_domain();
							world_pos.y = Math::snapped(world_pos.y - curve->get_min_value(), curve->get_value_range() / snap_count) + curve->get_min_value();
							ref_pos = get_view_pos(world_pos);
						}

						if (scaling_handle.x != 0 && scaling_initial_rect.size.width != 0) {
							real_t pivot = scaling_pivot_point.x;
							real_t distance = ref_pos.x - pivot;
							scale_factor.x = distance / scaling_initial_rect.size.width * scaling_handle.x;
						}
						if (scaling_handle.y != 0 && scaling_initial_rect.size.height != 0) {
							real_t pivot = scaling_pivot_point.y;
							real_t distance = ref_pos.y - pivot;
							scale_factor.y = distance / scaling_initial_rect.size.height * scaling_handle.y;
						}
						set_grab_scale_factor(scale_factor);
					} break;

					case GRAB_SELECT: {
						box_selection = Rect2(initial_grab_pos, Size2());
						box_selection.expand_to(get_world_pos(mpos));
					} break;

					default: {
						ERR_PRINT_ONCE(vformat("Unexpected Grab Mode: %d", grabbing));
					} break;
				}
				queue_redraw();
			} else {
				// Drag tangent.
				int selected_idx = selected_indexes.front()->get();
				const Vector2 new_pos = curve->get_point_position(selected_idx);
				const Vector2 control_pos = get_world_pos(mpos);

				Vector2 dir = (control_pos - new_pos).normalized();
				real_t tangent = dir.y / (dir.x > 0 ? MAX(dir.x, 0.00001) : MIN(dir.x, -0.00001));

				// Must keep track of the hovered index as the cursor might move outside of the editor while dragging.
				hovered_tangent_index = selected_tangent_index;

				// Adjust the tangents.
				if (selected_tangent_index == TANGENT_LEFT) {
					curve->set_point_left_tangent(selected_idx, tangent);

					// Align the other tangent if it isn't linear and Shift is not pressed.
					// If Shift is pressed at any point, restore the initial angle of the other tangent.
					if (selected_idx != (curve->get_point_count() - 1) && curve->get_point_right_mode(selected_idx) != Curve::TANGENT_LINEAR) {
						curve->set_point_right_tangent(selected_idx, mm->is_shift_pressed() ? initial_grab_right_tangent : tangent);
					}

				} else {
					curve->set_point_right_tangent(selected_idx, tangent);

					if (selected_idx != 0 && curve->get_point_left_mode(selected_idx) != Curve::TANGENT_LINEAR) {
						curve->set_point_left_tangent(selected_idx, mm->is_shift_pressed() ? initial_grab_left_tangent : tangent);
					}
				}
			}
		} else {
			// Grab mode is GRAB_NONE, so do hovering logic.
			hovered_index = get_point_at(mpos);
			hovered_tangent_index = get_tangent_at(mpos);
			bool has_selection_rect = selected_indexes.size() >= 2;
			if (has_selection_rect) {
				Rect2 ref_rect = scaling_initial_rect.grow(SCALE_GRAB_SIZE * EDSCALE / 2.0);
				scaling_hovered_handle = Vector2i();
				scaling_mouse_offset_from_handle = Vector2();
				int grab_size = SCALE_GRAB_SIZE * EDSCALE;
				if (Math::abs(mpos.x - ref_rect.position.x) < grab_size) {
					scaling_hovered_handle.x = -1;
					scaling_mouse_offset_from_handle.x = mpos.x - scaling_initial_rect.position.x;
				} else if (Math::abs(mpos.x - ref_rect.get_end().x) < grab_size) {
					scaling_hovered_handle.x = 1;
					scaling_mouse_offset_from_handle.x = mpos.x - scaling_initial_rect.get_end().x;
				}
				if (Math::abs(mpos.y - ref_rect.position.y) < grab_size) {
					scaling_hovered_handle.y = -1;
					scaling_mouse_offset_from_handle.y = mpos.y - scaling_initial_rect.position.y;
				} else if (Math::abs(mpos.y - ref_rect.get_end().y) < grab_size) {
					scaling_hovered_handle.y = 1;
					scaling_mouse_offset_from_handle.y = mpos.y - scaling_initial_rect.get_end().y;
				}
			}
			queue_redraw();
		}
	}
}

void CurveEdit::use_preset(int p_preset_id) {
	ERR_FAIL_COND(p_preset_id < 0 || p_preset_id >= PRESET_COUNT);
	ERR_FAIL_COND(curve.is_null());

	Array previous_data = curve->get_data();
	curve->clear_points();

	const float min_y = curve->get_min_value();
	const float max_y = curve->get_max_value();
	const float min_x = curve->get_min_domain();
	const float max_x = curve->get_max_domain();

	switch (p_preset_id) {
		case PRESET_CONSTANT:
			curve->add_point(Vector2(min_x, (min_y + max_y) / 2.0));
			curve->add_point(Vector2(max_x, (min_y + max_y) / 2.0));
			curve->set_point_right_mode(0, Curve::TANGENT_LINEAR);
			curve->set_point_left_mode(1, Curve::TANGENT_LINEAR);
			break;

		case PRESET_LINEAR:
			curve->add_point(Vector2(min_x, min_y));
			curve->add_point(Vector2(max_x, max_y));
			curve->set_point_right_mode(0, Curve::TANGENT_LINEAR);
			curve->set_point_left_mode(1, Curve::TANGENT_LINEAR);
			break;

		case PRESET_EASE_IN:
			curve->add_point(Vector2(min_x, min_y));
			curve->add_point(Vector2(max_x, max_y), curve->get_value_range() / curve->get_domain_range() * 1.4, 0);
			break;

		case PRESET_EASE_OUT:
			curve->add_point(Vector2(min_x, min_y), 0, curve->get_value_range() / curve->get_domain_range() * 1.4);
			curve->add_point(Vector2(max_x, max_y));
			break;

		case PRESET_SMOOTHSTEP:
			curve->add_point(Vector2(min_x, min_y));
			curve->add_point(Vector2(max_x, max_y));
			break;

		default:
			break;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Load Curve Preset"));
	undo_redo->add_do_method(*curve, "_set_data", curve->get_data());
	undo_redo->add_do_method(this, "set_selected_indexes", PackedInt32Array());
	undo_redo->add_undo_method(*curve, "_set_data", previous_data);
	undo_redo->add_undo_method(this, "set_selected_indexes", get_selected_indexes());
	undo_redo->commit_action();
}

void CurveEdit::_curve_changed() {
	queue_redraw();
	// Point count can change in case of undo.
	for (RBSet<int>::Element *E = selected_indexes.front(); E; E = E->next()) {
		if (E->get() >= curve->get_point_count()) {
			set_selected_indexes(PackedInt32Array());
			break;
		}
	}
}

int CurveEdit::get_point_at(const Vector2 &p_pos) const {
	if (curve.is_null()) {
		return -1;
	}

	// Use a square-shaped hover region. If hovering multiple points, pick the closer one.
	const Rect2 hover_rect = Rect2(p_pos, Vector2(0, 0)).grow(hover_radius);
	int closest_idx = -1;
	float closest_dist_squared = hover_radius * hover_radius * 2;

	for (int i = 0; i < curve->get_point_count(); ++i) {
		Vector2 p = get_view_pos(curve->get_point_position(i));
		if (hover_rect.has_point(p) && p.distance_squared_to(p_pos) < closest_dist_squared) {
			closest_dist_squared = p.distance_squared_to(p_pos);
			closest_idx = i;
		}
	}

	return closest_idx;
}

CurveEdit::TangentIndex CurveEdit::get_tangent_at(const Vector2 &p_pos) const {
	if (curve.is_null() || selected_indexes.size() != 1) {
		return TANGENT_NONE;
	}
	int selected_idx = selected_indexes.front()->get();

	const Rect2 hover_rect = Rect2(p_pos, Vector2(0, 0)).grow(tangent_hover_radius);

	if (selected_idx != 0) {
		Vector2 control_pos = get_tangent_view_pos(selected_idx, TANGENT_LEFT);
		if (hover_rect.has_point(control_pos)) {
			return TANGENT_LEFT;
		}
	}

	if (selected_idx != curve->get_point_count() - 1) {
		Vector2 control_pos = get_tangent_view_pos(selected_idx, TANGENT_RIGHT);
		if (hover_rect.has_point(control_pos)) {
			return TANGENT_RIGHT;
		}
	}

	return TANGENT_NONE;
}

// FIXME: This function should be bounded better.
float CurveEdit::get_offset_without_collision(int p_current_index, float p_offset, bool p_prioritize_right) {
	float safe_offset = p_offset;
	bool prioritizing_right = p_prioritize_right;

	for (int i = 0; i < curve->get_point_count(); i++) {
		if (i == p_current_index) {
			continue;
		}

		if (curve->get_point_position(i).x > safe_offset) {
			break;
		}

		if (curve->get_point_position(i).x == safe_offset) {
			if (prioritizing_right) {
				safe_offset += 0.00001;
				if (safe_offset > 1.0) {
					safe_offset = 1.0;
					prioritizing_right = false;
				}
			} else {
				safe_offset -= 0.00001;
				if (safe_offset < 0.0) {
					safe_offset = 0.0;
					prioritizing_right = true;
				}
			}
			i = -1;
		}
	}

	return safe_offset;
}

void CurveEdit::resolve_point_collisions(LocalVector<CurvePoint> &p_points) const {
	// Note: this function assumes vector is sorted.
	for (uint32_t i = 0; i < p_points.size() - 1; i++) {
		if (p_points[i].position.x == p_points[i + 1].position.x) {
			if (p_points[i].position.x == curve->get_min_domain()) {
				p_points[i + 1].position.x += 0.00001;
			} else {
				p_points[i].position.x -= 0.00001;
			}
			i = -1;
		}
	}
}

void CurveEdit::add_point(const Vector2 &p_pos) {
	ERR_FAIL_COND(curve.is_null());

	// Add a point to get its index, then remove it immediately. Trick to feed the UndoRedo.
	int new_idx = curve->add_point(p_pos);
	curve->remove_point(new_idx);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Curve Point"));
	undo_redo->add_do_method(*curve, "add_point", p_pos);
	undo_redo->add_do_method(this, "set_selected_indexes", PackedInt32Array({ new_idx }));
	undo_redo->add_undo_method(*curve, "remove_point", new_idx);
	undo_redo->add_undo_method(this, "set_selected_indexes", PackedInt32Array());
	undo_redo->commit_action();
}

void CurveEdit::remove_point(int p_index) {
	ERR_FAIL_COND(curve.is_null());
	ERR_FAIL_INDEX_MSG(p_index, curve->get_point_count(), "Curve point is out of bounds.");

	Curve::Point p = curve->get_point(p_index);
	Vector2 old_pos = (grabbing == GRAB_MOVE) ? initial_grab_pos : p.position;

	PackedInt32Array new_selected_indexes;
	// Reselect the old selected point if it's not the deleted one.
	for (int selected : selected_indexes) {
		if (selected > p_index) {
			new_selected_indexes.append(selected - 1);
		} else if (selected == p_index) {
			continue;
		} else {
			new_selected_indexes.append(selected);
		}
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Remove Curve Point"));
	undo_redo->add_do_method(*curve, "remove_point", p_index);
	undo_redo->add_do_method(this, "set_selected_indexes", new_selected_indexes);
	undo_redo->add_undo_method(*curve, "add_point", old_pos, p.left_tangent, p.right_tangent, p.left_mode, p.right_mode);
	undo_redo->add_undo_method(this, "set_selected_indexes", get_selected_indexes());
	undo_redo->commit_action();
}

void CurveEdit::delete_selection() {
	if (selected_indexes.is_empty()) {
		return;
	}
	if (grabbing == GRAB_MOVE) {
		set_grab_move_offset(Vector2());
	} else if (grabbing == GRAB_SCALE) {
		set_grab_scale_factor(Vector2(1.0, 1.0));
	}
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTRN("Remove Curve Point", "Remove Curve Points", selected_indexes.size()));
	// Delete in reverse order.
	for (RBSet<int>::Element *E = selected_indexes.back(); E; E->prev()) {
		undo_redo->add_do_method(*curve, "remove_point", E->get());
	}
	undo_redo->add_do_method(this, "set_selected_indexes", PackedInt32Array());
	// Re-add in order.
	for (int index : selected_indexes) {
		Curve::Point p = curve->get_point(index);
		undo_redo->add_undo_method(*curve, "add_point", p.position, p.left_tangent, p.right_tangent, p.left_mode, p.right_mode);
	}
	undo_redo->add_undo_method(this, "set_selected_indexes", get_selected_indexes());
	undo_redo->commit_action();
}

void CurveEdit::finish_grab_transform() {
	ERR_FAIL_COND(curve.is_null());

	// Note: Changing positions may modify the order.
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTRN("Modify Curve Point", "Modify Curve Points", selected_indexes.size()));

	// Prevent property_list_changed signal
	undo_redo->add_do_method(*curve, "set_block_signals", true);
	undo_redo->add_undo_method(*curve, "set_block_signals", true);

	undo_redo->add_do_method(*curve, "clear_points");
	undo_redo->add_undo_method(*curve, "clear_points");

	PackedInt32Array orig_selected_indexes;
	for (int i = 0; i < curve->get_point_count(); i++) {
		const CurvePoint &op = grab_initial_state[i];
		Curve::Point np = curve->get_point(i);
		undo_redo->add_do_method(*curve, "add_point", np.position, np.left_tangent, np.right_tangent, np.left_mode, np.right_mode);
		undo_redo->add_undo_method(*curve, "add_point", op.position, op.left_tangent, op.right_tangent, op.left_mode, op.right_mode);
		if (grab_initial_state[i].selected) {
			orig_selected_indexes.append(i);
		}
	}
	undo_redo->add_do_method(*curve, "set_block_signals", false);
	undo_redo->add_undo_method(*curve, "set_block_signals", false);

	undo_redo->add_do_method(this, "set_selected_indexes", get_selected_indexes());
	undo_redo->add_undo_method(this, "set_selected_indexes", orig_selected_indexes);

	undo_redo->commit_action();
}

void CurveEdit::set_point_tangents(int p_index, float p_left, float p_right) {
	ERR_FAIL_COND(curve.is_null());
	ERR_FAIL_INDEX_MSG(p_index, curve->get_point_count(), "Curve point is out of bounds.");

	if (initial_grab_left_tangent == p_left) {
		set_point_right_tangent(p_index, p_right);
		return;
	} else if (initial_grab_right_tangent == p_right) {
		set_point_left_tangent(p_index, p_left);
		return;
	}

	curve->set_point_left_tangent(p_index, initial_grab_left_tangent);
	curve->set_point_right_tangent(p_index, initial_grab_right_tangent);
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Modify Curve Point's Tangents"));
	undo_redo->add_do_method(*curve, "set_point_left_tangent", p_index, p_left);
	undo_redo->add_do_method(*curve, "set_point_right_tangent", p_index, p_right);
	undo_redo->add_do_method(this, "set_selected_indexes", PackedInt32Array({ p_index }));
	undo_redo->add_undo_method(*curve, "set_point_left_tangent", p_index, initial_grab_left_tangent);
	undo_redo->add_undo_method(*curve, "set_point_right_tangent", p_index, initial_grab_right_tangent);
	undo_redo->add_undo_method(this, "set_selected_indexes", PackedInt32Array({ p_index }));
	undo_redo->commit_action();
}

void CurveEdit::set_point_left_tangent(int p_index, float p_tangent) {
	ERR_FAIL_COND(curve.is_null());
	ERR_FAIL_INDEX_MSG(p_index, curve->get_point_count(), "Curve point is out of bounds.");

	if (initial_grab_left_tangent == p_tangent) {
		return;
	}

	curve->set_point_left_tangent(p_index, initial_grab_left_tangent);
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Modify Curve Point's Left Tangent"));
	undo_redo->add_do_method(*curve, "set_point_left_tangent", p_index, p_tangent);
	undo_redo->add_do_method(this, "set_selected_indexes", PackedInt32Array({ p_index }));
	undo_redo->add_undo_method(*curve, "set_point_left_tangent", p_index, initial_grab_left_tangent);
	undo_redo->add_undo_method(this, "set_selected_indexes", PackedInt32Array({ p_index }));
	undo_redo->commit_action();
}

void CurveEdit::set_point_right_tangent(int p_index, float p_tangent) {
	ERR_FAIL_COND(curve.is_null());
	ERR_FAIL_INDEX_MSG(p_index, curve->get_point_count(), "Curve point is out of bounds.");

	if (initial_grab_right_tangent == p_tangent) {
		return;
	}

	curve->set_point_right_tangent(p_index, initial_grab_right_tangent);
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Modify Curve Point's Right Tangent"));
	undo_redo->add_do_method(*curve, "set_point_right_tangent", p_index, p_tangent);
	undo_redo->add_do_method(this, "set_selected_indexes", PackedInt32Array({ p_index }));
	undo_redo->add_undo_method(*curve, "set_point_right_tangent", p_index, initial_grab_right_tangent);
	undo_redo->add_undo_method(this, "set_selected_indexes", PackedInt32Array({ p_index }));
	undo_redo->commit_action();
}

void CurveEdit::toggle_linear(int p_index, TangentIndex p_tangent) {
	ERR_FAIL_COND(curve.is_null());
	ERR_FAIL_INDEX_MSG(p_index, curve->get_point_count(), "Curve point is out of bounds.");

	if (p_tangent == TANGENT_NONE) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Toggle Linear Curve Point's Tangent"));

	Curve::TangentMode prev_mode = (p_tangent == TANGENT_LEFT) ? curve->get_point_left_mode(p_index) : curve->get_point_right_mode(p_index);
	Curve::TangentMode mode = (prev_mode == Curve::TANGENT_LINEAR) ? Curve::TANGENT_FREE : Curve::TANGENT_LINEAR;
	float prev_angle = (p_tangent == TANGENT_LEFT) ? curve->get_point_left_tangent(p_index) : curve->get_point_right_tangent(p_index);

	// Add different methods in the UndoRedo based on the tangent passed.
	if (p_tangent == TANGENT_LEFT) {
		undo_redo->add_do_method(*curve, "set_point_left_mode", p_index, mode);
		undo_redo->add_undo_method(*curve, "set_point_left_mode", p_index, prev_mode);
		undo_redo->add_undo_method(*curve, "set_point_left_tangent", p_index, prev_angle);
	} else {
		undo_redo->add_do_method(*curve, "set_point_right_mode", p_index, mode);
		undo_redo->add_undo_method(*curve, "set_point_right_mode", p_index, prev_mode);
		undo_redo->add_undo_method(*curve, "set_point_right_tangent", p_index, prev_angle);
	}

	undo_redo->commit_action();
}

void CurveEdit::initialize_grab_state() {
	grab_initial_state.clear();
	for (int i = 0; i < curve->get_point_count(); i++) {
		CurvePoint p = CurvePoint(curve->get_point(i));
		p.selected = selected_indexes.has(i);
		p.hovered = hovered_index == i;
		grab_initial_state.push_back(p);
	}
}

void CurveEdit::update_scaling_initial_rect() {
	if (grabbing == GRAB_SCALE) {
		// Scaling is ongoing, do not change
		return;
	}
	scaling_initial_rect = Rect2();
	if (selected_indexes.size() > 1) {
		bool first = true;
		for (int i : selected_indexes) {
			if (first) {
				scaling_initial_rect.position = get_view_pos(curve->get_point_position(i));
				first = false;
			} else {
				scaling_initial_rect.expand_to(get_view_pos(curve->get_point_position(i)));
			}
		}
	}
}

void CurveEdit::set_grab_move_offset(const Vector2 &p_offset) {
	LocalVector<CurvePoint> new_values(grab_initial_state);
	for (CurvePoint &p : new_values) {
		if (p.selected) {
			p.position = clamp_world_pos(p.position + p_offset);
		}
	}
	flush_grab_state(new_values);
}

void CurveEdit::set_grab_scale_factor(const Vector2 &p_factor) {
	scaling_factor = p_factor;
	LocalVector<CurvePoint> new_values(grab_initial_state);
	if (scaling_handle != Vector2i() && p_factor != Vector2(1.0, 1.0)) {
		Vector2 pivot_pos = get_world_pos(scaling_pivot_point);
		for (CurvePoint &p : new_values) {
			if (p.selected) {
				Vector2 offset = p.position - pivot_pos;
				Vector2 scaled_pos = clamp_world_pos(pivot_pos + offset * p_factor);
				if (scaling_handle.x != 0) {
					p.position.x = scaled_pos.x;
				}
				if (scaling_handle.y != 0) {
					p.position.y = scaled_pos.y;
				}
			}
		}
	}
	flush_grab_state(new_values);
}

void CurveEdit::flush_grab_state(LocalVector<CurvePoint> &p_new_state) {
	p_new_state.sort_custom<CurveEdit::CompareCurvePoints>();
	resolve_point_collisions(p_new_state);
	PackedInt32Array new_selected_indexes;
	for (uint32_t i = 0; i < p_new_state.size(); i++) {
		curve->set_point_value(i, p_new_state[i].position.y);
		curve->set_point_offset(i, p_new_state[i].position.x);
		if (p_new_state[i].selected) {
			new_selected_indexes.push_back(i);
		}
		if (p_new_state[i].hovered) {
			hovered_index = i;
		}
	}
	set_selected_indexes(new_selected_indexes);
}

void CurveEdit::set_selected_indexes(PackedInt32Array p_indexes) {
	selected_indexes.clear();
	for (uint32_t idx : p_indexes) {
		selected_indexes.insert(idx);
	}
	update_scaling_initial_rect();
	queue_redraw();
}

PackedInt32Array CurveEdit::get_selected_indexes() const {
	PackedInt32Array indexes;
	for (int selected : selected_indexes) {
		indexes.push_back(selected);
	}
	return indexes;
}

void CurveEdit::update_view_transform() {
	Ref<Font> font = get_theme_font(SceneStringName(font), SNAME("Label"));
	int font_size = get_theme_font_size(SceneStringName(font_size), SNAME("Label"));

	const real_t margin = font->get_height(font_size) + 2 * EDSCALE;

	float min_x = curve.is_valid() ? curve->get_min_domain() : 0.0;
	float max_x = curve.is_valid() ? curve->get_max_domain() : 1.0;
	float min_y = curve.is_valid() ? curve->get_min_value() : 0.0;
	float max_y = curve.is_valid() ? curve->get_max_value() : 1.0;

	const Rect2 world_rect = Rect2(min_x, min_y, max_x - min_x, max_y - min_y);
	const Size2 view_margin(margin, margin);
	const Size2 view_size = get_size() - view_margin * 2;
	const Vector2 scale = view_size / world_rect.size;

	Transform2D world_trans;
	world_trans.translate_local(-world_rect.position - Vector2(0, world_rect.size.y));
	world_trans.scale(Vector2(scale.x, -scale.y));

	Transform2D view_trans;
	view_trans.translate_local(view_margin);

	_world_to_view = view_trans * world_trans;
}

Vector2 CurveEdit::get_tangent_view_pos(int p_index, TangentIndex p_tangent) const {
	Vector2 dir;
	if (p_tangent == TANGENT_LEFT) {
		dir = -Vector2(1, curve->get_point_left_tangent(p_index));
	} else {
		dir = Vector2(1, curve->get_point_right_tangent(p_index));
	}

	Vector2 point_pos = curve->get_point_position(p_index);
	Vector2 point_view_pos = get_view_pos(point_pos);
	Vector2 control_view_pos = get_view_pos(point_pos + dir);

	Vector2 distance_from_point = tangent_length * (control_view_pos - point_view_pos).normalized();
	Vector2 tangent_view_pos = point_view_pos + distance_from_point;

	// Since the tangent is long, it might slip outside of the area of the editor for points close to the domain/range boundaries.
	// The code below shrinks the tangent control by up to 50% so it always stays inside the editor for points within the bounds.
	float fraction_inside = 1.0;
	if (distance_from_point.x != 0.0) {
		fraction_inside = MIN(fraction_inside, ((distance_from_point.x > 0 ? get_rect().size.x : 0) - point_view_pos.x) / distance_from_point.x);
	}
	if (distance_from_point.y != 0.0) {
		fraction_inside = MIN(fraction_inside, ((distance_from_point.y > 0 ? get_rect().size.y : 0) - point_view_pos.y) / distance_from_point.y);
	}

	if (fraction_inside < 1.0 && fraction_inside > 0.5) {
		tangent_view_pos = point_view_pos + distance_from_point * fraction_inside;
	}

	return tangent_view_pos;
}

Vector2 CurveEdit::get_view_pos(const Vector2 &p_world_pos) const {
	return _world_to_view.xform(p_world_pos);
}

Vector2 CurveEdit::clamp_world_pos(const Vector2 &p_world_pos) const {
	return p_world_pos.clamp(Vector2(curve->get_min_domain(), curve->get_min_value()), Vector2(curve->get_max_domain(), curve->get_max_value()));
}

Vector2 CurveEdit::get_world_pos(const Vector2 &p_view_pos) const {
	return _world_to_view.affine_inverse().xform(p_view_pos);
}

// Uses non-baked points, but takes advantage of ordered iteration to be faster.
void CurveEdit::plot_curve_accurate(float p_step, const Color &p_line_color, const Color &p_edge_line_color) {
	const real_t min_x = curve->get_min_domain();
	const real_t max_x = curve->get_max_domain();
	if (curve->get_point_count() <= 1) { // Draw single line through entire plot.
		real_t y = curve->sample(0);
		draw_line(get_view_pos(Vector2(min_x, y)) + Vector2(0.5, 0), get_view_pos(Vector2(max_x, y)) - Vector2(1.5, 0), p_line_color, LINE_WIDTH, true);
		return;
	}

	Vector2 first_point = curve->get_point_position(0);
	Vector2 last_point = curve->get_point_position(curve->get_point_count() - 1);

	// Transform pixels-per-step into curve domain. Only works for non-rotated transforms.
	const float world_step_size = p_step / _world_to_view.get_scale().x;

	// Edge lines.
	draw_line(get_view_pos(Vector2(min_x, first_point.y)) + Vector2(0.5, 0), get_view_pos(first_point), p_edge_line_color, LINE_WIDTH, true);
	draw_line(get_view_pos(last_point), get_view_pos(Vector2(max_x, last_point.y)) - Vector2(1.5, 0), p_edge_line_color, LINE_WIDTH, true);

	// Draw section by section, so that we get maximum precision near points.
	// It's an accurate representation, but slower than using the baked one.
	for (int i = 1; i < curve->get_point_count(); ++i) {
		Vector2 a = curve->get_point_position(i - 1);
		Vector2 b = curve->get_point_position(i);

		Vector2 pos = a;
		Vector2 prev_pos = a;

		float samples = (b.x - a.x) / world_step_size;

		for (int j = 1; j < samples; j++) {
			float x = j * world_step_size;
			pos.x = a.x + x;
			pos.y = curve->sample_local_nocheck(i - 1, x);
			draw_line(get_view_pos(prev_pos), get_view_pos(pos), p_line_color, LINE_WIDTH, true);
			prev_pos = pos;
		}

		draw_line(get_view_pos(prev_pos), get_view_pos(b), p_line_color, LINE_WIDTH, true);
	}
}

void CurveEdit::_redraw() {
	if (curve.is_null()) {
		return;
	}

	update_view_transform();

	// Draw background.

	Vector2 view_size = get_rect().size;
	draw_style_box(get_theme_stylebox(SceneStringName(panel), SNAME("Tree")), Rect2(Point2(), view_size));

	// Draw primary grid.
	draw_set_transform_matrix(_world_to_view);

	Vector2 min_edge = get_world_pos(Vector2(0, view_size.y));
	Vector2 max_edge = get_world_pos(Vector2(view_size.x, 0));

	const Color grid_color_primary = get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.25);
	const Color grid_color = get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.1);

	const Vector2i grid_steps = Vector2i(4, 2);
	const Vector2 step_size = Vector2(curve->get_domain_range(), curve->get_value_range()) / grid_steps;

	draw_line(Vector2(min_edge.x, curve->get_min_value()), Vector2(max_edge.x, curve->get_min_value()), grid_color_primary);
	draw_line(Vector2(max_edge.x, curve->get_max_value()), Vector2(min_edge.x, curve->get_max_value()), grid_color_primary);
	draw_line(Vector2(curve->get_min_domain(), min_edge.y), Vector2(curve->get_min_domain(), max_edge.y), grid_color_primary);
	draw_line(Vector2(curve->get_max_domain(), max_edge.y), Vector2(curve->get_max_domain(), min_edge.y), grid_color_primary);

	for (int i = 1; i < grid_steps.x; i++) {
		real_t x = curve->get_min_domain() + i * step_size.x;
		draw_line(Vector2(x, min_edge.y), Vector2(x, max_edge.y), grid_color);
	}

	for (int i = 1; i < grid_steps.y; i++) {
		real_t y = curve->get_min_value() + i * step_size.y;
		draw_line(Vector2(min_edge.x, y), Vector2(max_edge.x, y), grid_color);
	}

	// Draw number markings.
	draw_set_transform_matrix(Transform2D());

	Ref<Font> font = get_theme_font(SceneStringName(font), SNAME("Label"));
	int font_size = get_theme_font_size(SceneStringName(font_size), SNAME("Label"));
	float font_height = font->get_height(font_size);
	Color text_color = get_theme_color(SceneStringName(font_color), EditorStringName(Editor));

	int pad = Math::round(2 * EDSCALE);

	for (int i = 0; i <= grid_steps.x; ++i) {
		real_t x = curve->get_min_domain() + i * step_size.x;
		draw_string(font, get_view_pos(Vector2(x, curve->get_min_value())) + Vector2(pad, font_height - pad), String::num(x, 2), HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, text_color);
	}

	for (int i = 0; i <= grid_steps.y; ++i) {
		real_t y = curve->get_min_value() + i * step_size.y;
		draw_string(font, get_view_pos(Vector2(curve->get_min_domain(), y)) + Vector2(pad, -pad), String::num(y, 2), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, text_color);
	}

	// Draw curve in view coordinates. Curve world-to-view point conversion happens in plot_curve_accurate().

	const Color line_color = get_theme_color(SceneStringName(font_color), EditorStringName(Editor));
	const Color edge_line_color = get_theme_color(SceneStringName(font_color), EditorStringName(Editor)) * Color(1, 1, 1, 0.75);

	plot_curve_accurate(STEP_SIZE, line_color, edge_line_color);

	// Draw points, except for the selected one.

	bool shift_pressed = Input::get_singleton()->is_key_pressed(Key::SHIFT);

	const Color point_color = get_theme_color(SceneStringName(font_color), EditorStringName(Editor));

	for (int i = 0; i < curve->get_point_count(); ++i) {
		Vector2 pos = get_view_pos(curve->get_point_position(i));
		if (!selected_indexes.has(i)) {
			draw_rect(Rect2(pos, Vector2(0, 0)).grow(point_radius), point_color);
		}
		if (hovered_index == i && hovered_tangent_index == TANGENT_NONE) {
			draw_rect(Rect2(pos, Vector2(0, 0)).grow(hover_radius - Math::round(3 * EDSCALE)), line_color, false, Math::round(1 * EDSCALE));
		}
	}

	// Draw selected point and its tangents.

	bool only_one_selected = selected_indexes.size() == 1;
	for (int selected_idx : selected_indexes) {
		const Vector2 point_pos = curve->get_point_position(selected_idx);
		const Color selected_point_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));

		// Draw tangents if not dragging a point, or if holding a point without having moved it yet.
		if (grabbing == GRAB_NONE || initial_grab_pos == point_pos || selected_tangent_index != TANGENT_NONE) {
			const Color selected_tangent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor)).darkened(0.25);
			const Color tangent_color = get_theme_color(SceneStringName(font_color), EditorStringName(Editor)).darkened(0.25);

			if (only_one_selected && selected_idx != 0) {
				Vector2 control_pos = get_tangent_view_pos(selected_idx, TANGENT_LEFT);
				Color left_tangent_color = (selected_tangent_index == TANGENT_LEFT) ? selected_tangent_color : tangent_color;

				draw_line(get_view_pos(point_pos), control_pos, left_tangent_color, 0.5 * EDSCALE, true);
				// Square for linear mode, circle otherwise.
				if (curve->get_point_left_mode(selected_idx) == Curve::TANGENT_FREE) {
					draw_circle(control_pos, tangent_radius, left_tangent_color);
				} else {
					draw_rect(Rect2(control_pos, Vector2(0, 0)).grow(tangent_radius), left_tangent_color);
				}
				// Hover indicator.
				if (hovered_tangent_index == TANGENT_LEFT || (hovered_tangent_index == TANGENT_RIGHT && !shift_pressed && curve->get_point_left_mode(selected_idx) != Curve::TANGENT_LINEAR)) {
					draw_rect(Rect2(control_pos, Vector2(0, 0)).grow(tangent_hover_radius - Math::round(3 * EDSCALE)), tangent_color, false, Math::round(1 * EDSCALE));
				}
			}

			if (only_one_selected && selected_idx != curve->get_point_count() - 1) {
				Vector2 control_pos = get_tangent_view_pos(selected_idx, TANGENT_RIGHT);
				Color right_tangent_color = (selected_tangent_index == TANGENT_RIGHT) ? selected_tangent_color : tangent_color;

				draw_line(get_view_pos(point_pos), control_pos, right_tangent_color, 0.5 * EDSCALE, true);
				// Square for linear mode, circle otherwise.
				if (curve->get_point_right_mode(selected_idx) == Curve::TANGENT_FREE) {
					draw_circle(control_pos, tangent_radius, right_tangent_color);
				} else {
					draw_rect(Rect2(control_pos, Vector2(0, 0)).grow(tangent_radius), right_tangent_color);
				}
				// Hover indicator.
				if (hovered_tangent_index == TANGENT_RIGHT || (hovered_tangent_index == TANGENT_LEFT && !shift_pressed && curve->get_point_right_mode(selected_idx) != Curve::TANGENT_LINEAR)) {
					draw_rect(Rect2(control_pos, Vector2(0, 0)).grow(tangent_hover_radius - Math::round(3 * EDSCALE)), tangent_color, false, Math::round(1 * EDSCALE));
				}
			}
		}

		draw_rect(Rect2(get_view_pos(point_pos), Vector2(0, 0)).grow(point_radius), selected_point_color);
	}

	// Draw help text.

	if (only_one_selected) {
		int selected_idx = selected_indexes.front()->get();
		if (selected_idx > 0 && selected_idx < curve->get_point_count() - 1 && selected_tangent_index == TANGENT_NONE && hovered_tangent_index != TANGENT_NONE && !shift_pressed) {
			float width = view_size.x - 50 * EDSCALE;
			text_color.a *= 0.4;

			draw_multiline_string(font, Vector2(25 * EDSCALE, font_height - Math::round(2 * EDSCALE)), TTR("Hold Shift to edit tangents individually"), HORIZONTAL_ALIGNMENT_CENTER, width, font_size, -1, text_color);

		} else if (selected_idx != -1 && selected_tangent_index == TANGENT_NONE) {
			const Vector2 point_pos = curve->get_point_position(selected_idx);
			float width = view_size.x - 50 * EDSCALE;
			text_color.a *= 0.8;

			draw_string(font, Vector2(25 * EDSCALE, font_height - Math::round(2 * EDSCALE)), vformat("(%.2f, %.2f)", point_pos.x, point_pos.y), HORIZONTAL_ALIGNMENT_CENTER, width, font_size, text_color);

		} else if (selected_idx != -1 && selected_tangent_index != TANGENT_NONE) {
			float width = view_size.x - 50 * EDSCALE;
			text_color.a *= 0.8;
			real_t theta = Math::rad_to_deg(Math::atan(selected_tangent_index == TANGENT_LEFT ? -1 * curve->get_point_left_tangent(selected_idx) : curve->get_point_right_tangent(selected_idx)));

			draw_string(font, Vector2(25 * EDSCALE, font_height - Math::round(2 * EDSCALE)), String::num(theta, 1) + String::utf8(" °"), HORIZONTAL_ALIGNMENT_CENTER, width, font_size, text_color);
		}

		// Draw temporary constraints and snapping axes.
		draw_set_transform_matrix(_world_to_view);

		if (Input::get_singleton()->is_key_pressed(Key::ALT) && grabbing == GRAB_MOVE && selected_tangent_index == TANGENT_NONE) {
			float prev_point_offset = (selected_idx > 0) ? curve->get_point_position(selected_idx - 1).x : curve->get_min_domain();
			float next_point_offset = (selected_idx < curve->get_point_count() - 1) ? curve->get_point_position(selected_idx + 1).x : curve->get_max_domain();

			draw_line(Vector2(prev_point_offset, curve->get_min_value()), Vector2(prev_point_offset, curve->get_max_value()), Color(point_color, 0.6));
			draw_line(Vector2(next_point_offset, curve->get_min_value()), Vector2(next_point_offset, curve->get_max_value()), Color(point_color, 0.6));
		}
	}

	if (shift_pressed && grabbing == GRAB_MOVE && selected_tangent_index == TANGENT_NONE) {
		draw_line(Vector2(initial_grab_pos.x, curve->get_min_value()), Vector2(initial_grab_pos.x, curve->get_max_value()), get_theme_color(SNAME("axis_x_color"), EditorStringName(Editor)).darkened(0.4));
		draw_line(Vector2(curve->get_min_domain(), initial_grab_pos.y), Vector2(curve->get_max_domain(), initial_grab_pos.y), get_theme_color(SNAME("axis_y_color"), EditorStringName(Editor)).darkened(0.4));
	}

	draw_set_transform_matrix(Transform2D());

	bool draw_scale_handles = selected_indexes.size() > 1;
	if (draw_scale_handles) {
		bool first = true;
		Rect2 scale_handle;
		for (int i : selected_indexes) {
			if (first) {
				scale_handle.position = get_view_pos(curve->get_point_position(i));
				first = false;
			} else {
				scale_handle.expand_to(get_view_pos(curve->get_point_position(i)));
			}
		}
		scale_handle.grow_by(SCALE_GRAB_SIZE * EDSCALE / 2.0);
		draw_rect(scale_handle, accent_color, false, Math::round(EDSCALE));
	}

	if (grabbing == GRAB_SELECT && selected_tangent_index == TANGENT_NONE && box_selection.has_area()) {
		draw_rect(_world_to_view.xform(box_selection), box_selection_fill_color, true);
		draw_rect(_world_to_view.xform(box_selection), box_selection_stroke_color, true);
	}
}

///////////////////////

const int CurveEditor::DEFAULT_SNAP = 10;

void CurveEditor::_set_snap_enabled(bool p_enabled) {
	curve_editor_rect->set_snap_enabled(p_enabled);
	snap_count_edit->set_visible(p_enabled);
}

void CurveEditor::_set_snap_count(int p_snap_count) {
	curve_editor_rect->set_snap_count(CLAMP(p_snap_count, 2, 100));
}

void CurveEditor::_on_preset_item_selected(int p_preset_id) {
	curve_editor_rect->use_preset(p_preset_id);
}

void CurveEditor::set_curve(const Ref<Curve> &p_curve) {
	curve_editor_rect->set_curve(p_curve);
}

void CurveEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			spacing = Math::round(BASE_SPACING * get_theme_default_base_scale());
			snap_button->set_button_icon(get_editor_theme_icon(SNAME("SnapGrid")));
			PopupMenu *p = presets_button->get_popup();
			p->clear();
			p->add_icon_item(get_editor_theme_icon(SNAME("CurveConstant")), TTR("Constant"), CurveEdit::PRESET_CONSTANT);
			p->add_icon_item(get_editor_theme_icon(SNAME("CurveLinear")), TTR("Linear"), CurveEdit::PRESET_LINEAR);
			p->add_icon_item(get_editor_theme_icon(SNAME("CurveIn")), TTR("Ease In"), CurveEdit::PRESET_EASE_IN);
			p->add_icon_item(get_editor_theme_icon(SNAME("CurveOut")), TTR("Ease Out"), CurveEdit::PRESET_EASE_OUT);
			p->add_icon_item(get_editor_theme_icon(SNAME("CurveInOut")), TTR("Smoothstep"), CurveEdit::PRESET_SMOOTHSTEP);
		} break;
		case NOTIFICATION_READY: {
			Ref<Curve> curve = curve_editor_rect->get_curve();
			if (curve.is_valid()) {
				// Set snapping settings based on the curve's meta.
				snap_button->set_pressed(curve->get_meta("_snap_enabled", false));
				snap_count_edit->set_value(curve->get_meta("_snap_count", DEFAULT_SNAP));
			}
		} break;
		case NOTIFICATION_RESIZED:
			curve_editor_rect->update_minimum_size();
			break;
	}
}

CurveEditor::CurveEditor() {
	HFlowContainer *toolbar = memnew(HFlowContainer);
	add_child(toolbar);

	snap_button = memnew(Button);
	snap_button->set_tooltip_text(TTR("Toggle Grid Snap"));
	snap_button->set_toggle_mode(true);
	toolbar->add_child(snap_button);
	snap_button->connect(SceneStringName(toggled), callable_mp(this, &CurveEditor::_set_snap_enabled));

	toolbar->add_child(memnew(VSeparator));

	snap_count_edit = memnew(EditorSpinSlider);
	snap_count_edit->set_min(2);
	snap_count_edit->set_max(100);
	snap_count_edit->set_accessibility_name(TTRC("Snap Step"));
	snap_count_edit->set_value(DEFAULT_SNAP);
	snap_count_edit->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
	toolbar->add_child(snap_count_edit);
	snap_count_edit->connect(SceneStringName(value_changed), callable_mp(this, &CurveEditor::_set_snap_count));

	presets_button = memnew(MenuButton);
	presets_button->set_text(TTR("Presets"));
	presets_button->set_switch_on_hover(true);
	presets_button->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_END);
	toolbar->add_child(presets_button);
	presets_button->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &CurveEditor::_on_preset_item_selected));

	curve_editor_rect = memnew(CurveEdit);
	add_child(curve_editor_rect);

	// Some empty space below. Not a part of the curve editor so it can't draw in it.
	Control *empty_space = memnew(Control);
	empty_space->set_custom_minimum_size(Vector2(0, spacing));
	add_child(empty_space);

	set_mouse_filter(MOUSE_FILTER_STOP);
	_set_snap_enabled(snap_button->is_pressed());
	_set_snap_count(snap_count_edit->get_value());
}

///////////////////////

bool EditorInspectorPluginCurve::can_handle(Object *p_object) {
	return Object::cast_to<Curve>(p_object) != nullptr;
}

void EditorInspectorPluginCurve::parse_begin(Object *p_object) {
	Curve *curve = Object::cast_to<Curve>(p_object);
	ERR_FAIL_NULL(curve);
	Ref<Curve> c(curve);

	CurveEditor *editor = memnew(CurveEditor);
	editor->set_curve(c);
	add_custom_control(editor);
}

CurveEditorPlugin::CurveEditorPlugin() {
	Ref<EditorInspectorPluginCurve> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);

	EditorInterface::get_singleton()->get_resource_previewer()->add_preview_generator(memnew(CurvePreviewGenerator));
}

///////////////////////

bool CurvePreviewGenerator::handles(const String &p_type) const {
	return p_type == "Curve";
}

Ref<Texture2D> CurvePreviewGenerator::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Curve> curve = p_from;
	if (curve.is_null()) {
		return Ref<Texture2D>();
	}

	Ref<Image> img_ref;
	img_ref.instantiate();
	Image &im = **img_ref;
	im.initialize_data(p_size.x, p_size.y, false, Image::FORMAT_RGBA8);

	Color line_color = EditorInterface::get_singleton()->get_editor_theme()->get_color(SceneStringName(font_color), EditorStringName(Editor));

	// Set the first pixel of the thumbnail.
	float v = (curve->sample_baked(curve->get_min_domain()) - curve->get_min_value()) / curve->get_value_range();
	int y = CLAMP(im.get_height() - v * im.get_height(), 0, im.get_height() - 1);
	im.set_pixel(0, y, line_color);

	// Plot a line towards the next point.
	int prev_y = y;
	for (int x = 1; x < im.get_width(); ++x) {
		float t = static_cast<float>(x) / im.get_width() * curve->get_domain_range() + curve->get_min_domain();
		v = (curve->sample_baked(t) - curve->get_min_value()) / curve->get_value_range();
		y = CLAMP(im.get_height() - v * im.get_height(), 0, im.get_height() - 1);

		Vector<Point2i> points = Geometry2D::bresenham_line(Point2i(x - 1, prev_y), Point2i(x, y));
		for (Point2i point : points) {
			im.set_pixelv(point, line_color);
		}
		prev_y = y;
	}

	return ImageTexture::create_from_image(img_ref);
}
