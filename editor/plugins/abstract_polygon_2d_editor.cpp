/*************************************************************************/
/*  abstract_polygon_2d_editor.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "abstract_polygon_2d_editor.h"

#include "canvas_item_editor_plugin.h"

void AbstractPolygon2DEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			get_tree()->connect("node_removed", this, "_node_removed");

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}

void AbstractPolygon2DEditor::_node_removed(Node *p_node) {

	if (p_node == _get_node()) {
		edit(NULL);
		hide();

		canvas_item_editor->get_viewport_control()->update();
	}
}

bool AbstractPolygon2DEditor::_is_wip_destructive() const {

	return true;
}

void AbstractPolygon2DEditor::_create_wip_close_action(const Vector<Vector2> &p_wip) {

	undo_redo->create_action(TTR("Create Poly"));
	undo_redo->add_undo_method(_get_node(), "set_polygon", _get_polygon(0));
	undo_redo->add_do_method(_get_node(), "set_polygon", p_wip);
}

void AbstractPolygon2DEditor::_create_edit_poly_action(int p_polygon, const Vector<Vector2> &p_before, const Vector<Vector2> &p_after) {

	undo_redo->create_action(TTR("Edit Poly"));
	undo_redo->add_do_method(_get_node(), "set_polygon", p_after);
	undo_redo->add_undo_method(_get_node(), "set_polygon", p_before);
}

void AbstractPolygon2DEditor::_create_remove_point_action(int p_polygon, int p_point) {

	Vector<Vector2> poly = _get_polygon(p_polygon);
	undo_redo->create_action(TTR("Edit Poly (Remove Point)"));
	undo_redo->add_undo_method(_get_node(), "set_polygon", poly);
	poly.remove(p_point);
	undo_redo->add_do_method(_get_node(), "set_polygon", poly);
}

Color AbstractPolygon2DEditor::_get_current_outline_color() const {

	return Color(1, 0.3, 0.1, 0.8);
}

Color AbstractPolygon2DEditor::_get_previous_outline_color() const {

	return Color(0.5, 0.5, 0.5, 0.8);
}

bool AbstractPolygon2DEditor::_can_input(const Ref<InputEvent> &p_event, bool &p_ret) const {

	return true;
}

bool AbstractPolygon2DEditor::_can_draw() const {

	return true;
}

void AbstractPolygon2DEditor::_wip_close() {

	if (wip.size() >= 3) {

		_create_wip_close_action(wip);
		undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
		undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
		undo_redo->commit_action();

		_enter_edit_mode();
	}

	wip.clear();
	wip_active = false;
	edited_point = -1;
}

bool AbstractPolygon2DEditor::forward_gui_input(const Ref<InputEvent> &p_event) {

	if (!_get_node())
		return false;

	bool ret;
	if (!_can_input(p_event, ret))
		return ret;

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		Transform2D xform = canvas_item_editor->get_canvas_transform() * _get_node()->get_global_transform();

		Vector2 gpoint = mb->get_position();
		Vector2 cpoint = canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint);
		cpoint = canvas_item_editor->snap_point(cpoint);
		cpoint = _get_node()->get_global_transform().affine_inverse().xform(cpoint);

		const Vector2 offset = _get_offset();

		//first check if a point is to be added (segment split)
		real_t grab_threshold = EDITOR_DEF("editors/poly_editor/point_grab_radius", 8);

		if (_is_in_create_mode()) {

			if (mb->get_button_index() == BUTTON_LEFT && mb->is_pressed()) {

				if (!wip_active) {

					wip.clear();
					wip.push_back(cpoint - offset);
					wip_active = true;
					edited_point_pos = cpoint;
					edited_polygon = -1;
					edited_point = 1;
					canvas_item_editor->get_viewport_control()->update();
					return true;
				} else {

					if (wip.size() > 1 && xform.xform(wip[0] + offset).distance_to(gpoint) < grab_threshold) {
						//wip closed
						_wip_close();

						return true;
					} else {

						wip.push_back(cpoint - offset);
						edited_point = wip.size();
						canvas_item_editor->get_viewport_control()->update();
						return true;

						//add wip point
					}
				}
			} else if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed() && wip_active) {
				_wip_close();
			}
		} else if (_is_in_edit_mode()) {

			if (mb->get_button_index() == BUTTON_LEFT) {

				if (mb->is_pressed()) {

					if (mb->get_control()) {

						const int n_polygons = _get_polygon_count();

						if (n_polygons >= 1 && _get_polygon(n_polygons - 1).size() < 3) {

							Vector<Vector2> poly = _get_polygon(n_polygons - 1);
							poly.push_back(cpoint);
							_create_edit_poly_action(n_polygons - 1, _get_polygon(n_polygons - 1), poly);

							undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
							undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
							undo_redo->commit_action();
							return true;
						}

						//search edges
						int closest_poly = -1;
						int closest_idx = -1;
						Vector2 closest_pos;
						real_t closest_dist = 1e10;

						for (int j = 0; j < n_polygons; j++) {

							Vector<Vector2> points = _get_polygon(j);
							const int n_points = points.size();

							for (int i = 0; i < n_points; i++) {

								Vector2 p[2] = { xform.xform(points[i] + offset),
									xform.xform(points[(i + 1) % n_points] + offset) };

								Vector2 cp = Geometry::get_closest_point_to_segment_2d(gpoint, p);
								if (cp.distance_squared_to(points[0]) < CMP_EPSILON2 || cp.distance_squared_to(points[1]) < CMP_EPSILON2)
									continue; //not valid to reuse point

								real_t d = cp.distance_to(gpoint);
								if (d < closest_dist && d < grab_threshold) {
									closest_poly = j;
									closest_dist = d;
									closest_pos = cp;
									closest_idx = i;
								}
							}
						}


						if (closest_idx >= 0) {

							pre_move_edit = _get_polygon(closest_poly);
							Vector<Vector2> poly = pre_move_edit;
							poly.insert(closest_idx + 1, xform.affine_inverse().xform(closest_pos));
							edited_point = closest_idx + 1;
							edited_polygon = closest_poly;
							edited_point_pos = xform.affine_inverse().xform(closest_pos);
							_set_polygon(closest_poly, poly);
							canvas_item_editor->get_viewport_control()->update();
							return true;
						}
					} else {

						//look for points to move
						int closest_poly = -1;
						int closest_idx = -1;
						Vector2 closest_pos;
						real_t closest_dist = 1e10;

						const int n_polygons = _get_polygon_count();

						for (int j = 0; j < n_polygons; j++) {

							Vector<Vector2> points = _get_polygon(j);
							const int n_points = points.size();

							for (int i = 0; i < n_points; i++) {

								Vector2 cp = xform.xform(points[i] + offset);

								real_t d = cp.distance_to(gpoint);
								if (d < closest_dist && d < grab_threshold) {
									closest_poly = j;
									closest_dist = d;
									closest_pos = cp;
									closest_idx = i;
								}
							}
						}

						if (closest_idx >= 0) {

							pre_move_edit = _get_polygon(closest_poly);
							edited_polygon = closest_poly;
							edited_point = closest_idx;
							edited_point_pos = xform.affine_inverse().xform(closest_pos);
							canvas_item_editor->get_viewport_control()->update();
							return true;
						}
					}
				} else {

					if (edited_point != -1) {

						//apply

						Vector<Vector2> poly = _get_polygon(edited_polygon);
						ERR_FAIL_INDEX_V(edited_point, poly.size(), false);
						poly[edited_point] = edited_point_pos - offset;
						_create_edit_poly_action(edited_polygon, pre_move_edit, poly);
						undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->commit_action();

						edited_point = -1;
						return true;
					}
				}
			} else if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed() && edited_point == -1) {

				int closest_poly = -1;
				int closest_idx = -1;
				Vector2 closest_pos;
				real_t closest_dist = 1e10;
				const int n_polygons = _get_polygon_count();

				for (int j = 0; j < n_polygons; j++) {

					Vector<Vector2> points = _get_polygon(j);
					const int n_points = points.size();

					for (int i = 0; i < n_points; i++) {

						Vector2 cp = xform.xform(points[i] + offset);

						real_t d = cp.distance_to(gpoint);
						if (d < closest_dist && d < grab_threshold) {
							closest_poly = j;
							closest_dist = d;
							closest_pos = cp;
							closest_idx = i;
						}
					}
				}

				if (closest_idx >= 0) {

					_create_remove_point_action(closest_poly, closest_idx);
					undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
					undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
					undo_redo->commit_action();
					return true;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		if (edited_point != -1 && (wip_active || mm->get_button_mask() & BUTTON_MASK_LEFT)) {

			Vector2 gpoint = mm->get_position();
			Vector2 cpoint = canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint);
			cpoint = canvas_item_editor->snap_point(cpoint);
			edited_point_pos = _get_node()->get_global_transform().affine_inverse().xform(cpoint);

			if (!wip_active) {

				Vector<Vector2> poly = _get_polygon(edited_polygon);
				ERR_FAIL_INDEX_V(edited_point, poly.size(), false);
				poly[edited_point] = edited_point_pos - _get_offset();
				_set_polygon(edited_polygon, poly);
			}

			canvas_item_editor->get_viewport_control()->update();
		}
	}

	return false;
}

void AbstractPolygon2DEditor::_canvas_draw() {

	if (!_get_node())
		return;

	if (!_can_draw())
		return;

	Control *vpc = canvas_item_editor->get_viewport_control();

	Transform2D xform = canvas_item_editor->get_canvas_transform() * _get_node()->get_global_transform();
	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");
	const Vector2 offset = _get_offset();
	const int n_polygons = _get_polygon_count();

	for (int j = -1; j < n_polygons; j++) {

		if (wip_active && _is_wip_destructive() && j != -1)
			continue;

		Vector<Vector2> poly;

		if (wip_active && j == edited_polygon) {

			poly = wip;
		} else {

			if (j == -1)
				continue;
			poly = _get_polygon(j);
		}

		if (!wip_active && j == edited_polygon && edited_point >= 0 && EDITOR_DEF("editors/poly_editor/show_previous_outline", true)) {

			const Color col = _get_previous_outline_color();
			const int n = pre_move_edit.size();
			for (int i = 0; i < n; i++) {

				Vector2 p, p2;
				p = pre_move_edit[i] + offset;
				p2 = pre_move_edit[(i + 1) % n] + offset;

				Vector2 point = xform.xform(p);
				Vector2 next_point = xform.xform(p2);

				vpc->draw_line(point, next_point, col, 2);
			}
		}

		const int n_points = poly.size();
		Color col = _get_current_outline_color();

		for (int i = 0; i < n_points; i++) {

			Vector2 p, p2;
			p = (j == edited_polygon && i == edited_point) ? edited_point_pos : (poly[i] + offset);
			if (j == edited_polygon && ((wip_active && i == n_points - 1) || (((i + 1) % n_points) == edited_point)))
				p2 = edited_point_pos;
			else
				p2 = poly[(i + 1) % n_points] + offset;

			Vector2 point = xform.xform(p);
			Vector2 next_point = xform.xform(p2);

			vpc->draw_line(point, next_point, col, 2);
			vpc->draw_texture(handle, point - handle->get_size() * 0.5);
		}
	}
}

void AbstractPolygon2DEditor::edit(Node *p_polygon) {

	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton();
	}

	if (p_polygon) {

		_set_node(p_polygon);

		if (!canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->connect("draw", this, "_canvas_draw");

		wip.clear();
		wip_active = false;
		edited_point = -1;

		canvas_item_editor->get_viewport_control()->update();

	} else {

		_set_node(NULL);

		if (canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->disconnect("draw", this, "_canvas_draw");
	}
}

void AbstractPolygon2DEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_canvas_draw"), &AbstractPolygon2DEditor::_canvas_draw);
	ClassDB::bind_method(D_METHOD("_node_removed"), &AbstractPolygon2DEditor::_node_removed);
}

AbstractPolygon2DEditor::AbstractPolygon2DEditor(EditorNode *p_editor) {

	canvas_item_editor = NULL;
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	wip_active = false;
	edited_polygon = -1;
}
