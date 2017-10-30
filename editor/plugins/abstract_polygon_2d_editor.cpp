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

bool AbstractPolygon2DEditor::_is_empty() const {

	if (!_get_node())
		return true;

	const int n = _get_polygon_count();

	for (int i = 0; i < n; i++) {

		Vector<Vector2> vertices = _get_polygon(i);

		if (vertices.size() != 0)
			return false;
	}

	return true;
}

int AbstractPolygon2DEditor::_get_polygon_count() const {

	return 1;
}

Variant AbstractPolygon2DEditor::_get_polygon(int p_idx) const {

	return _get_node()->get("polygon");
}

void AbstractPolygon2DEditor::_set_polygon(int p_idx, const Variant &p_polygon) const {

	_get_node()->set("polygon", p_polygon);
}

void AbstractPolygon2DEditor::_action_set_polygon(int p_idx, const Variant &p_previous, const Variant &p_polygon) {

	Node2D *node = _get_node();
	undo_redo->add_do_method(node, "set_polygon", p_polygon);
	undo_redo->add_undo_method(node, "set_polygon", p_previous);
}

Vector2 AbstractPolygon2DEditor::_get_offset(int p_idx) const {

	return Vector2(0, 0);
}

void AbstractPolygon2DEditor::_commit_action() {

	undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
	undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
	undo_redo->commit_action();
}

void AbstractPolygon2DEditor::_action_add_polygon(const Variant &p_polygon) {

	_action_set_polygon(0, p_polygon);
}

void AbstractPolygon2DEditor::_action_remove_polygon(int p_idx) {

	_action_set_polygon(p_idx, _get_polygon(p_idx), PoolVector<Vector2>());
}

void AbstractPolygon2DEditor::_action_set_polygon(int p_idx, const Variant &p_polygon) {

	_action_set_polygon(p_idx, _get_polygon(p_idx), p_polygon);
}

bool AbstractPolygon2DEditor::_has_resource() const {

	return true;
}

void AbstractPolygon2DEditor::_create_resource() {
}

void AbstractPolygon2DEditor::_menu_option(int p_option) {

	switch (p_option) {

		case MODE_CREATE: {

			mode = MODE_CREATE;
			button_create->set_pressed(true);
			button_edit->set_pressed(false);
		} break;
		case MODE_EDIT: {

			mode = MODE_EDIT;
			button_create->set_pressed(false);
			button_edit->set_pressed(true);
		} break;
	}
}

void AbstractPolygon2DEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			button_create->set_icon(get_icon("Edit", "EditorIcons"));
			button_edit->set_icon(get_icon("MovePoint", "EditorIcons"));
			button_edit->set_pressed(true);

			get_tree()->connect("node_removed", this, "_node_removed");

			create_resource->connect("confirmed", this, "_create_resource");

		} break;
		case NOTIFICATION_PHYSICS_PROCESS: {

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

void AbstractPolygon2DEditor::_wip_close() {

	if (wip.size() >= 3) {

		undo_redo->create_action(TTR("Create Poly"));
		_action_add_polygon(wip);
		_commit_action();

		mode = MODE_EDIT;
		button_edit->set_pressed(true);
		button_create->set_pressed(false);
	}

	wip.clear();
	wip_active = false;
	edited_point = -1;
}

bool AbstractPolygon2DEditor::forward_gui_input(const Ref<InputEvent> &p_event) {

	if (!_get_node())
		return false;

	Ref<InputEventMouseButton> mb = p_event;

	if (!_has_resource()) {

		if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
			create_resource->set_text(String("No polygon resource on this node.\nCreate and assign one?"));
			create_resource->popup_centered_minsize();
		}
		return (mb.is_valid() && mb->get_button_index() == 1);
	}

	if (mb.is_valid()) {

		Transform2D xform = canvas_item_editor->get_canvas_transform() * _get_node()->get_global_transform();

		Vector2 gpoint = mb->get_position();
		Vector2 cpoint = _get_node()->get_global_transform().affine_inverse().xform(canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(mb->get_position())));

		//first check if a point is to be added (segment split)
		real_t grab_threshold = EDITOR_DEF("editors/poly_editor/point_grab_radius", 8);

		if (mode == MODE_CREATE) {

			if (mb->get_button_index() == BUTTON_LEFT && mb->is_pressed()) {

				if (!wip_active) {

					wip.clear();
					wip.push_back(cpoint);
					wip_active = true;
					edited_point_pos = cpoint;
					edited_polygon = -1;
					edited_point = 1;
					canvas_item_editor->get_viewport_control()->update();
					return true;
				} else {

					if (wip.size() > 1 && xform.xform(wip[0]).distance_to(gpoint) < grab_threshold) {
						//wip closed
						_wip_close();

						return true;
					} else {

						wip.push_back(cpoint);
						edited_point = wip.size();
						canvas_item_editor->get_viewport_control()->update();
						return true;

						//add wip point
					}
				}
			} else if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed() && wip_active) {
				_wip_close();
			}
		} else if (mode == MODE_EDIT) {

			if (mb->get_button_index() == BUTTON_LEFT) {

				if (mb->is_pressed()) {

					if (mb->get_control()) {

						const int n_polygons = _get_polygon_count();

						if (n_polygons >= 1) {

							Vector<Vector2> vertices = _get_polygon(n_polygons - 1);

							if (vertices.size() < 3) {

								vertices.push_back(cpoint);
								undo_redo->create_action(TTR("Edit Poly"));
								_action_set_polygon(n_polygons - 1, vertices);
								_commit_action();
								return true;
							}
						}

						//search edges
						int closest_poly = -1;
						int closest_idx = -1;
						Vector2 closest_pos;
						real_t closest_dist = 1e10;

						for (int j = 0; j < n_polygons; j++) {

							PoolVector<Vector2> points = _get_polygon(j);
							const Vector2 offset = _get_offset(j);
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

							Vector<Vector2> vertices = _get_polygon(closest_poly);
							pre_move_edit = vertices;
							vertices.insert(closest_idx + 1, xform.affine_inverse().xform(closest_pos));
							edited_point = closest_idx + 1;
							edited_polygon = closest_poly;
							edited_point_pos = xform.affine_inverse().xform(closest_pos);

							undo_redo->create_action(TTR("Insert Point"));
							_action_set_polygon(closest_poly, vertices);
							_commit_action();

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

							PoolVector<Vector2> points = _get_polygon(j);
							const Vector2 offset = _get_offset(j);
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

						Vector<Vector2> vertices = _get_polygon(edited_polygon);
						ERR_FAIL_INDEX_V(edited_point, vertices.size(), false);
						vertices[edited_point] = edited_point_pos - _get_offset(edited_polygon);

						undo_redo->create_action(TTR("Edit Poly"));
						_action_set_polygon(edited_polygon, pre_move_edit, vertices);
						_commit_action();

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

					PoolVector<Vector2> points = _get_polygon(j);
					const int n_points = points.size();
					const Vector2 offset = _get_offset(j);

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

					PoolVector<Vector2> vertices = _get_polygon(closest_poly);

					if (vertices.size() > 3) {

						vertices.remove(closest_idx);

						undo_redo->create_action(TTR("Edit Poly (Remove Point)"));
						_action_set_polygon(closest_poly, vertices);
						_commit_action();
					} else {

						undo_redo->create_action(TTR("Remove Poly And Point"));
						_action_remove_polygon(closest_poly);
						_commit_action();
					}

					if (_is_empty())
						_menu_option(MODE_CREATE);
					return true;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		if (edited_point != -1 && (wip_active || (mm->get_button_mask() & BUTTON_MASK_LEFT))) {

			Vector2 gpoint = mm->get_position();
			Vector2 cpoint = _get_node()->get_global_transform().affine_inverse().xform(canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(mm->get_position())));
			edited_point_pos = cpoint;

			if (!wip_active) {

				Vector<Vector2> vertices = _get_polygon(edited_polygon);
				ERR_FAIL_INDEX_V(edited_point, vertices.size(), false);
				vertices[edited_point] = cpoint - _get_offset(edited_polygon);
				_set_polygon(edited_polygon, vertices);
			}

			canvas_item_editor->get_viewport_control()->update();
		}
	}

	return false;
}

void AbstractPolygon2DEditor::forward_draw_over_canvas(Control *p_canvas) {
	if (!_get_node())
		return;

	Control *vpc = canvas_item_editor->get_viewport_control();

	Transform2D xform = canvas_item_editor->get_canvas_transform() * _get_node()->get_global_transform();
	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");
	const int n_polygons = _get_polygon_count();

	for (int j = -1; j < n_polygons; j++) {

		if (wip_active && wip_destructive && j != -1)
			continue;

		PoolVector<Vector2> points;
		Vector2 offset;

		if (wip_active && j == edited_polygon) {

			points = Variant(wip);
			offset = Vector2(0, 0);
		} else {

			if (j == -1)
				continue;
			points = _get_polygon(j);
			offset = _get_offset(j);
		}

		if (!wip_active && j == edited_polygon && edited_point >= 0 && EDITOR_DEF("editors/poly_editor/show_previous_outline", true)) {

			const Color col = Color(0.5, 0.5, 0.5); // FIXME polygon->get_outline_color();
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

		const int n_points = points.size();
		const Color col = Color(1, 0.3, 0.1, 0.8);

		for (int i = 0; i < n_points; i++) {

			Vector2 p, p2;
			p = (j == edited_polygon && i == edited_point) ? edited_point_pos : (points[i] + offset);
			if (j == edited_polygon && ((wip_active && i == n_points - 1) || (((i + 1) % n_points) == edited_point)))
				p2 = edited_point_pos;
			else
				p2 = points[(i + 1) % n_points] + offset;

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

		//Enable the pencil tool if the polygon is empty
		if (_is_empty())
			_menu_option(MODE_CREATE);

		wip.clear();
		wip_active = false;
		edited_point = -1;

		canvas_item_editor->get_viewport_control()->update();

	} else {

		_set_node(NULL);
	}
}

void AbstractPolygon2DEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_node_removed"), &AbstractPolygon2DEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_menu_option"), &AbstractPolygon2DEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_create_resource"), &AbstractPolygon2DEditor::_create_resource);
}

AbstractPolygon2DEditor::AbstractPolygon2DEditor(EditorNode *p_editor, bool p_wip_destructive) {

	canvas_item_editor = NULL;
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	wip_active = false;
	edited_polygon = -1;
	wip_destructive = p_wip_destructive;

	add_child(memnew(VSeparator));
	button_create = memnew(ToolButton);
	add_child(button_create);
	button_create->connect("pressed", this, "_menu_option", varray(MODE_CREATE));
	button_create->set_toggle_mode(true);
	button_create->set_tooltip(TTR("Create a new polygon from scratch."));

	button_edit = memnew(ToolButton);
	add_child(button_edit);
	button_edit->connect("pressed", this, "_menu_option", varray(MODE_EDIT));
	button_edit->set_toggle_mode(true);
	button_edit->set_tooltip(TTR("Edit existing polygon:\nLMB: Move Point.\nCtrl+LMB: Split Segment.\nRMB: Erase Point."));

	create_resource = memnew(ConfirmationDialog);
	add_child(create_resource);
	create_resource->get_ok()->set_text(TTR("Create"));

	mode = MODE_EDIT;
}

void AbstractPolygon2DEditorPlugin::edit(Object *p_object) {

	polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool AbstractPolygon2DEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class(klass);
}

void AbstractPolygon2DEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		polygon_editor->show();
	} else {

		polygon_editor->hide();
		polygon_editor->edit(NULL);
	}
}

AbstractPolygon2DEditorPlugin::AbstractPolygon2DEditorPlugin(EditorNode *p_node, AbstractPolygon2DEditor *p_polygon_editor, String p_class) {

	editor = p_node;
	polygon_editor = p_polygon_editor;
	klass = p_class;
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(polygon_editor);

	polygon_editor->hide();
}

AbstractPolygon2DEditorPlugin::~AbstractPolygon2DEditorPlugin() {
}
