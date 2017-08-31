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

void AbstractPolygon2DEditor::_create_resource() {

	if (!node)
		return;

	node->_create_resource(undo_redo);
	_menu_option(MODE_CREATE);
}

bool AbstractPolygon2DEditor::_is_empty() const {

	if (node && node->_has_resource() && node->get_polygon_count() > 0 && node->get_nth_polygon(0)->get_vertices().size() > 0)
		return false;

	return true;
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

			create_res->connect("confirmed", this, "_create_resource");

			get_tree()->connect("node_removed", this, "_node_removed");

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}

void AbstractPolygon2DEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		edit(NULL);
		hide();

		canvas_item_editor->get_viewport_control()->update();
	}
}

void AbstractPolygon2DEditor::_wip_close() {

	if (wip.size() >= 3) {

		undo_redo->create_action(TTR("Create Poly"));
		undo_redo->add_undo_method(node, "remove_polygon", node->get_polygon_count());
		undo_redo->add_do_method(node, "append_polygon", wip);
		undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
		undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
		undo_redo->commit_action();

		mode = MODE_EDIT;
		button_edit->set_pressed(true);
		button_create->set_pressed(false);
	}

	wip.clear();
	wip_active = false;
	edited_point = -1;
}

bool AbstractPolygon2DEditor::forward_gui_input(const Ref<InputEvent> &p_event) {

	if (!node)
		return false;

	if (!node->_has_resource()) {

		Ref<InputEventMouseButton> mb = p_event;

		if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
			create_res->set_text("No matching polygon resource on this node.\nCreate and assign one?");
			create_res->popup_centered_minsize();
		}
		return mb.is_valid() && mb->get_button_index() == 1;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {

		Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

		Vector2 gpoint = mb->get_position();
		Vector2 cpoint = canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint);
		cpoint = canvas_item_editor->snap_point(cpoint);
		cpoint = node->get_global_transform().affine_inverse().xform(cpoint);

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

						const int n_polygons = node->get_polygon_count();

						if (n_polygons >= 1 && node->get_nth_polygon(n_polygons - 1)->get_vertices().size() < 3) {

							Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(n_polygons - 1);
							Vector<Vector2> vertices = polygon->get_vertices();
							vertices.push_back(cpoint);

							undo_redo->create_action(TTR("Edit Poly"));
							undo_redo->add_do_method(node, "set_vertices", n_polygons - 1, vertices);
							undo_redo->add_undo_method(node, "set_vertices", n_polygons - 1, polygon->get_vertices());
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

							Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(j);
							Vector<Vector2> points = polygon->get_vertices();
							const Vector2 offset = polygon->get_offset();
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

							Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(closest_poly);
							pre_move_edit = polygon->get_vertices();
							Vector<Vector2> vertices = pre_move_edit;
							vertices.insert(closest_idx + 1, xform.affine_inverse().xform(closest_pos));
							edited_point = closest_idx + 1;
							edited_polygon = closest_poly;
							edited_point_pos = xform.affine_inverse().xform(closest_pos);
							polygon->set_vertices(vertices);
							canvas_item_editor->get_viewport_control()->update();
							return true;
						}
					} else {

						//look for points to move
						int closest_poly = -1;
						int closest_idx = -1;
						Vector2 closest_pos;
						real_t closest_dist = 1e10;

						const int n_polygons = node->get_polygon_count();

						for (int j = 0; j < n_polygons; j++) {

							Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(j);
							Vector<Vector2> points = polygon->get_vertices();
							const Vector2 offset = polygon->get_offset();
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

							pre_move_edit = node->get_nth_polygon(closest_poly)->get_vertices();
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

						Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(edited_polygon);
						Vector<Vector2> vertices = polygon->get_vertices();
						ERR_FAIL_INDEX_V(edited_point, vertices.size(), false);
						vertices[edited_point] = edited_point_pos - polygon->get_offset();
						undo_redo->create_action(TTR("Edit Poly"));
						undo_redo->add_do_method(node, "set_vertices", edited_polygon, vertices);
						undo_redo->add_undo_method(node, "set_vertices", edited_polygon, pre_move_edit);
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
				const int n_polygons = node->get_polygon_count();

				for (int j = 0; j < n_polygons; j++) {

					Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(j);
					Vector<Vector2> points = polygon->get_vertices();
					const int n_points = points.size();
					const Vector2 offset = polygon->get_offset();

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

					Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(edited_polygon);
					PoolVector<Vector2> vertices = Variant(polygon->get_vertices());

					if (vertices.size() > 3) {

						undo_redo->create_action(TTR("Edit Poly (Remove Point)"));
						undo_redo->add_undo_method(node, "set_vertices", edited_polygon, vertices);
						vertices.remove(closest_idx);
						undo_redo->add_do_method(node, "set_vertices", edited_polygon, vertices);
					} else {

						undo_redo->create_action(TTR("Remove Poly And Point"));
						undo_redo->add_undo_method(node, "add_polygon_at_index", polygon, edited_polygon);
						undo_redo->add_do_method(node, "remove_polygon", edited_polygon);
					}

					undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
					undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
					undo_redo->commit_action();
					if (_is_empty())
						_menu_option(MODE_CREATE);
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
			edited_point_pos = node->get_global_transform().affine_inverse().xform(cpoint);

			if (!wip_active) {

				Ref<AbstractPolygon2D> polygon = node->get_nth_polygon(edited_polygon);
				Vector<Vector2> vertices = polygon->get_vertices();
				ERR_FAIL_INDEX_V(edited_point, vertices.size(), false);
				vertices[edited_point] = edited_point_pos - polygon->get_offset();
				polygon->set_vertices(vertices);
			}

			canvas_item_editor->get_viewport_control()->update();
		}
	}

	return false;
}

void AbstractPolygon2DEditor::_canvas_draw() {

	if (!node)
		return;

	Control *vpc = canvas_item_editor->get_viewport_control();

	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");
	const int n_polygons = node->get_polygon_count();

	for (int j = -1; j < n_polygons; j++) {

		if (wip_active && wip_destructive && j != -1)
			continue;

		Ref<AbstractPolygon2D> polygon;
		Vector<Vector2> points;
		Vector2 offset;

		if (wip_active && j == edited_polygon) {

			points = wip;
			offset = Vector2(0, 0);
		} else {

			if (j == -1)
				continue;
			polygon = node->get_nth_polygon(j);
			points = polygon->get_vertices();
			offset = polygon->get_offset();
		}

		if (!wip_active && j == edited_polygon && edited_point >= 0 && EDITOR_DEF("editors/poly_editor/show_previous_outline", true)) {

			const Color col = polygon->get_outline_color();
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

		node = Object::cast_to<EditablePolygonNode2D>(p_polygon);

		//Enable the pencil tool if the polygon is empty
		if (_is_empty())
			_menu_option(MODE_CREATE);

		if (!canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->connect("draw", this, "_canvas_draw");

		wip.clear();
		wip_active = false;
		edited_point = -1;

		canvas_item_editor->get_viewport_control()->update();

	} else {

		node = NULL;

		if (canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->disconnect("draw", this, "_canvas_draw");
	}
}

void AbstractPolygon2DEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_canvas_draw"), &AbstractPolygon2DEditor::_canvas_draw);
	ClassDB::bind_method(D_METHOD("_node_removed"), &AbstractPolygon2DEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_menu_option"), &AbstractPolygon2DEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_create_resource"), &AbstractPolygon2DEditor::_create_resource);
}

AbstractPolygon2DEditor::AbstractPolygon2DEditor(EditorNode *p_editor, bool p_wip_destructive) {

	canvas_item_editor = NULL;
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	node = NULL;

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

	create_res = memnew(ConfirmationDialog);
	add_child(create_res);
	create_res->get_ok()->set_text(TTR("Create"));

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
