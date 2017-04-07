/*************************************************************************/
/*  polygon_2d_editor_plugin.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "polygon_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_settings.h"
#include "os/file_access.h"
#include "os/input.h"
#include "os/keyboard.h"

void Polygon2DEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			button_create->set_icon(get_icon("Edit", "EditorIcons"));
			button_edit->set_icon(get_icon("MovePoint", "EditorIcons"));
			button_edit->set_pressed(true);
			button_uv->set_icon(get_icon("Uv", "EditorIcons"));

			uv_button[UV_MODE_EDIT_POINT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
			uv_button[UV_MODE_MOVE]->set_icon(get_icon("ToolMove", "EditorIcons"));
			uv_button[UV_MODE_ROTATE]->set_icon(get_icon("ToolRotate", "EditorIcons"));
			uv_button[UV_MODE_SCALE]->set_icon(get_icon("ToolScale", "EditorIcons"));

			b_snap_grid->set_icon(get_icon("Grid", "EditorIcons"));
			b_snap_enable->set_icon(get_icon("Snap", "EditorIcons"));
			uv_icon_zoom->set_texture(get_icon("Zoom", "EditorIcons"));

			get_tree()->connect("node_removed", this, "_node_removed");

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}
void Polygon2DEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		edit(NULL);
		hide();

		canvas_item_editor->get_viewport_control()->update();
	}
}

void Polygon2DEditor::_menu_option(int p_option) {

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
		case MODE_EDIT_UV: {

			if (node->get_texture().is_null()) {

				error->set_text("No texture in this polygon.\nSet a texture to be able to edit UV.");
				error->popup_centered_minsize();
				return;
			}

			PoolVector<Vector2> points = node->get_polygon();
			PoolVector<Vector2> uvs = node->get_uv();
			if (uvs.size() != points.size()) {
				undo_redo->create_action(TTR("Create UV Map"));
				undo_redo->add_do_method(node, "set_uv", points);
				undo_redo->add_undo_method(node, "set_uv", uvs);
				undo_redo->add_do_method(uv_edit_draw, "update");
				undo_redo->add_undo_method(uv_edit_draw, "update");
				undo_redo->commit_action();
			}

			uv_edit->popup_centered_ratio(0.85);
		} break;
		case UVEDIT_POLYGON_TO_UV: {

			PoolVector<Vector2> points = node->get_polygon();
			if (points.size() == 0)
				break;
			PoolVector<Vector2> uvs = node->get_uv();
			undo_redo->create_action(TTR("Create UV Map"));
			undo_redo->add_do_method(node, "set_uv", points);
			undo_redo->add_undo_method(node, "set_uv", uvs);
			undo_redo->add_do_method(uv_edit_draw, "update");
			undo_redo->add_undo_method(uv_edit_draw, "update");
			undo_redo->commit_action();

		} break;
		case UVEDIT_UV_TO_POLYGON: {

			PoolVector<Vector2> points = node->get_polygon();
			PoolVector<Vector2> uvs = node->get_uv();
			if (uvs.size() == 0)
				break;

			undo_redo->create_action(TTR("Create UV Map"));
			undo_redo->add_do_method(node, "set_polygon", uvs);
			undo_redo->add_undo_method(node, "set_polygon", points);
			undo_redo->add_do_method(uv_edit_draw, "update");
			undo_redo->add_undo_method(uv_edit_draw, "update");
			undo_redo->commit_action();

		} break;
		case UVEDIT_UV_CLEAR: {

			PoolVector<Vector2> uvs = node->get_uv();
			if (uvs.size() == 0)
				break;
			undo_redo->create_action(TTR("Create UV Map"));
			undo_redo->add_do_method(node, "set_uv", PoolVector<Vector2>());
			undo_redo->add_undo_method(node, "set_uv", uvs);
			undo_redo->add_do_method(uv_edit_draw, "update");
			undo_redo->add_undo_method(uv_edit_draw, "update");
			undo_redo->commit_action();

		} break;
	}
}

void Polygon2DEditor::_set_use_snap(bool p_use) {
	use_snap = p_use;
}

void Polygon2DEditor::_set_show_grid(bool p_show) {
	snap_show_grid = p_show;
	uv_edit_draw->update();
}

void Polygon2DEditor::_set_snap_off_x(float p_val) {
	snap_offset.x = p_val;
	uv_edit_draw->update();
}

void Polygon2DEditor::_set_snap_off_y(float p_val) {
	snap_offset.y = p_val;
	uv_edit_draw->update();
}

void Polygon2DEditor::_set_snap_step_x(float p_val) {
	snap_step.x = p_val;
	uv_edit_draw->update();
}

void Polygon2DEditor::_set_snap_step_y(float p_val) {
	snap_step.y = p_val;
	uv_edit_draw->update();
}

void Polygon2DEditor::_wip_close() {

	undo_redo->create_action(TTR("Create Poly"));
	undo_redo->add_undo_method(node, "set_polygon", node->get_polygon());
	undo_redo->add_do_method(node, "set_polygon", wip);
	undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
	undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
	undo_redo->commit_action();
	wip.clear();
	wip_active = false;
	mode = MODE_EDIT;
	button_edit->set_pressed(true);
	button_create->set_pressed(false);
	edited_point = -1;
}

bool Polygon2DEditor::forward_gui_input(const InputEvent &p_event) {

	if (node == NULL)
		return false;

	switch (p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &mb = p_event.mouse_button;

			Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

			Vector2 gpoint = Point2(mb.x, mb.y);
			Vector2 cpoint = canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint);
			cpoint = canvas_item_editor->snap_point(cpoint);
			cpoint = node->get_global_transform().affine_inverse().xform(cpoint);

			Vector<Vector2> poly = Variant(node->get_polygon());

			//first check if a point is to be added (segment split)
			real_t grab_treshold = EDITOR_DEF("editors/poly_editor/point_grab_radius", 8);

			switch (mode) {

				case MODE_CREATE: {

					if (mb.button_index == BUTTON_LEFT && mb.pressed) {

						if (!wip_active) {

							wip.clear();
							wip.push_back(cpoint - node->get_offset());
							wip_active = true;
							edited_point_pos = cpoint;
							canvas_item_editor->get_viewport_control()->update();
							edited_point = 1;
							return true;
						} else {

							if (wip.size() > 1 && xform.xform(wip[0] + node->get_offset()).distance_to(gpoint) < grab_treshold) {
								//wip closed
								_wip_close();

								return true;
							} else {

								wip.push_back(cpoint - node->get_offset());
								edited_point = wip.size();
								canvas_item_editor->get_viewport_control()->update();
								return true;

								//add wip point
							}
						}
					} else if (mb.button_index == BUTTON_RIGHT && mb.pressed && wip_active) {
						_wip_close();
					}

				} break;

				case MODE_EDIT: {

					if (mb.button_index == BUTTON_LEFT) {
						if (mb.pressed) {

							if (mb.mod.control) {

								if (poly.size() < 3) {

									undo_redo->create_action(TTR("Edit Poly"));
									undo_redo->add_undo_method(node, "set_polygon", poly);
									poly.push_back(cpoint);
									undo_redo->add_do_method(node, "set_polygon", poly);
									undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
									undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
									undo_redo->commit_action();
									return true;
								}

								//search edges
								int closest_idx = -1;
								Vector2 closest_pos;
								real_t closest_dist = 1e10;
								for (int i = 0; i < poly.size(); i++) {

									Vector2 points[2] = { xform.xform(poly[i] + node->get_offset()),
										xform.xform(poly[(i + 1) % poly.size()] + node->get_offset()) };

									Vector2 cp = Geometry::get_closest_point_to_segment_2d(gpoint, points);
									if (cp.distance_squared_to(points[0]) < CMP_EPSILON2 || cp.distance_squared_to(points[1]) < CMP_EPSILON2)
										continue; //not valid to reuse point

									real_t d = cp.distance_to(gpoint);
									if (d < closest_dist && d < grab_treshold) {
										closest_dist = d;
										closest_pos = cp;
										closest_idx = i;
									}
								}

								if (closest_idx >= 0) {

									pre_move_edit = poly;
									poly.insert(closest_idx + 1, xform.affine_inverse().xform(closest_pos) - node->get_offset());
									edited_point = closest_idx + 1;
									edited_point_pos = xform.affine_inverse().xform(closest_pos);
									node->set_polygon(Variant(poly));
									canvas_item_editor->get_viewport_control()->update();
									return true;
								}
							} else {

								//look for points to move

								int closest_idx = -1;
								Vector2 closest_pos;
								real_t closest_dist = 1e10;
								for (int i = 0; i < poly.size(); i++) {

									Vector2 cp = xform.xform(poly[i] + node->get_offset());

									real_t d = cp.distance_to(gpoint);
									if (d < closest_dist && d < grab_treshold) {
										closest_dist = d;
										closest_pos = cp;
										closest_idx = i;
									}
								}

								if (closest_idx >= 0) {

									pre_move_edit = poly;
									edited_point = closest_idx;
									edited_point_pos = xform.affine_inverse().xform(closest_pos);
									canvas_item_editor->get_viewport_control()->update();
									return true;
								}
							}
						} else {

							if (edited_point != -1) {

								//apply

								ERR_FAIL_INDEX_V(edited_point, poly.size(), false);
								poly[edited_point] = edited_point_pos - node->get_offset();
								undo_redo->create_action(TTR("Edit Poly"));
								undo_redo->add_do_method(node, "set_polygon", poly);
								undo_redo->add_undo_method(node, "set_polygon", pre_move_edit);
								undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
								undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
								undo_redo->commit_action();

								edited_point = -1;
								return true;
							}
						}
					} else if (mb.button_index == BUTTON_RIGHT && mb.pressed && edited_point == -1) {

						int closest_idx = -1;
						Vector2 closest_pos;
						real_t closest_dist = 1e10;
						for (int i = 0; i < poly.size(); i++) {

							Vector2 cp = xform.xform(poly[i] + node->get_offset());

							real_t d = cp.distance_to(gpoint);
							if (d < closest_dist && d < grab_treshold) {
								closest_dist = d;
								closest_pos = cp;
								closest_idx = i;
							}
						}

						if (closest_idx >= 0) {

							undo_redo->create_action(TTR("Edit Poly (Remove Point)"));
							undo_redo->add_undo_method(node, "set_polygon", poly);
							poly.remove(closest_idx);
							undo_redo->add_do_method(node, "set_polygon", poly);
							undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
							undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
							undo_redo->commit_action();
							return true;
						}
					}

				} break;
			}

		} break;
		case InputEvent::MOUSE_MOTION: {

			const InputEventMouseMotion &mm = p_event.mouse_motion;

			if (edited_point != -1 && (wip_active || mm.button_mask & BUTTON_MASK_LEFT)) {

				Vector2 gpoint = Point2(mm.x, mm.y);
				Vector2 cpoint = canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint);
				cpoint = canvas_item_editor->snap_point(cpoint);
				edited_point_pos = node->get_global_transform().affine_inverse().xform(cpoint);

				canvas_item_editor->get_viewport_control()->update();
			}

		} break;
	}

	return false;
}
void Polygon2DEditor::_canvas_draw() {

	if (!node)
		return;

	Control *vpc = canvas_item_editor->get_viewport_control();

	Vector<Vector2> poly;

	if (wip_active)
		poly = wip;
	else
		poly = Variant(node->get_polygon());

	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");

	for (int i = 0; i < poly.size(); i++) {

		Vector2 p, p2;
		p = i == edited_point ? edited_point_pos : (poly[i] + node->get_offset());
		if ((wip_active && i == poly.size() - 1) || (((i + 1) % poly.size()) == edited_point))
			p2 = edited_point_pos;
		else
			p2 = poly[(i + 1) % poly.size()] + node->get_offset();

		Vector2 point = xform.xform(p);
		Vector2 next_point = xform.xform(p2);

		Color col = Color(1, 0.3, 0.1, 0.8);
		vpc->draw_line(point, next_point, col, 2);
		vpc->draw_texture(handle, point - handle->get_size() * 0.5);
	}
}

void Polygon2DEditor::_uv_mode(int p_mode) {

	uv_mode = UVMode(p_mode);
	for (int i = 0; i < UV_MODE_MAX; i++) {
		uv_button[i]->set_pressed(p_mode == i);
	}
}

void Polygon2DEditor::_uv_input(const InputEvent &p_input) {

	Transform2D mtx;
	mtx.elements[2] = -uv_draw_ofs;
	mtx.scale_basis(Vector2(uv_draw_zoom, uv_draw_zoom));

	if (p_input.type == InputEvent::MOUSE_BUTTON) {

		const InputEventMouseButton &mb = p_input.mouse_button;

		if (mb.button_index == BUTTON_LEFT) {

			if (mb.pressed) {

				uv_drag_from = Vector2(mb.x, mb.y);
				uv_drag = true;
				uv_prev = node->get_uv();
				uv_move_current = uv_mode;
				if (uv_move_current == UV_MODE_EDIT_POINT) {

					if (mb.mod.shift && mb.mod.command)
						uv_move_current = UV_MODE_SCALE;
					else if (mb.mod.shift)
						uv_move_current = UV_MODE_MOVE;
					else if (mb.mod.command)
						uv_move_current = UV_MODE_ROTATE;
				}

				if (uv_move_current == UV_MODE_EDIT_POINT) {

					uv_drag_index = -1;
					for (int i = 0; i < uv_prev.size(); i++) {

						Vector2 tuv = mtx.xform(uv_prev[i]);
						if (tuv.distance_to(Vector2(mb.x, mb.y)) < 8) {
							uv_drag_from = tuv;
							uv_drag_index = i;
						}
					}

					if (uv_drag_index == -1) {
						uv_drag = false;
					}
				}
			} else if (uv_drag) {

				undo_redo->create_action(TTR("Transform UV Map"));
				undo_redo->add_do_method(node, "set_uv", node->get_uv());
				undo_redo->add_undo_method(node, "set_uv", uv_prev);
				undo_redo->add_do_method(uv_edit_draw, "update");
				undo_redo->add_undo_method(uv_edit_draw, "update");
				undo_redo->commit_action();

				uv_drag = false;
			}

		} else if (mb.button_index == BUTTON_RIGHT && mb.pressed) {

			if (uv_drag) {

				uv_drag = false;
				node->set_uv(uv_prev);
				uv_edit_draw->update();
			}

		} else if (mb.button_index == BUTTON_WHEEL_UP && mb.pressed) {

			uv_zoom->set_value(uv_zoom->get_value() / 0.9);
		} else if (mb.button_index == BUTTON_WHEEL_DOWN && mb.pressed) {

			uv_zoom->set_value(uv_zoom->get_value() * 0.9);
		}

	} else if (p_input.type == InputEvent::MOUSE_MOTION) {

		const InputEventMouseMotion &mm = p_input.mouse_motion;

		if (mm.button_mask & BUTTON_MASK_MIDDLE || Input::get_singleton()->is_key_pressed(KEY_SPACE)) {

			Vector2 drag(mm.relative_x, mm.relative_y);
			uv_hscroll->set_value(uv_hscroll->get_value() - drag.x);
			uv_vscroll->set_value(uv_vscroll->get_value() - drag.y);

		} else if (uv_drag) {

			Vector2 uv_drag_to = snap_point(Vector2(mm.x, mm.y));
			Vector2 drag = mtx.affine_inverse().xform(uv_drag_to) - mtx.affine_inverse().xform(uv_drag_from);

			switch (uv_move_current) {

				case UV_MODE_EDIT_POINT: {

					PoolVector<Vector2> uv_new = uv_prev;
					uv_new.set(uv_drag_index, uv_new[uv_drag_index] + drag);
					node->set_uv(uv_new);
				} break;
				case UV_MODE_MOVE: {

					PoolVector<Vector2> uv_new = uv_prev;
					for (int i = 0; i < uv_new.size(); i++)
						uv_new.set(i, uv_new[i] + drag);

					node->set_uv(uv_new);

				} break;
				case UV_MODE_ROTATE: {

					Vector2 center;
					PoolVector<Vector2> uv_new = uv_prev;

					for (int i = 0; i < uv_new.size(); i++)
						center += uv_prev[i];
					center /= uv_new.size();

					float angle = (uv_drag_from - mtx.xform(center)).normalized().angle_to((uv_drag_to - mtx.xform(center)).normalized());

					for (int i = 0; i < uv_new.size(); i++) {
						Vector2 rel = uv_prev[i] - center;
						rel = rel.rotated(angle);
						uv_new.set(i, center + rel);
					}

					node->set_uv(uv_new);

				} break;
				case UV_MODE_SCALE: {

					Vector2 center;
					PoolVector<Vector2> uv_new = uv_prev;

					for (int i = 0; i < uv_new.size(); i++)
						center += uv_prev[i];
					center /= uv_new.size();

					float from_dist = uv_drag_from.distance_to(mtx.xform(center));
					float to_dist = uv_drag_to.distance_to(mtx.xform(center));
					if (from_dist < 2)
						break;

					float scale = to_dist / from_dist;

					for (int i = 0; i < uv_new.size(); i++) {
						Vector2 rel = uv_prev[i] - center;
						rel = rel * scale;
						uv_new.set(i, center + rel);
					}

					node->set_uv(uv_new);
				} break;
			}
			uv_edit_draw->update();
		}
	}
}

void Polygon2DEditor::_uv_scroll_changed(float) {

	if (updating_uv_scroll)
		return;

	uv_draw_ofs.x = uv_hscroll->get_value();
	uv_draw_ofs.y = uv_vscroll->get_value();
	uv_draw_zoom = uv_zoom->get_value();
	uv_edit_draw->update();
}

void Polygon2DEditor::_uv_draw() {

	Ref<Texture> base_tex = node->get_texture();
	if (base_tex.is_null())
		return;

	Transform2D mtx;
	mtx.elements[2] = -uv_draw_ofs;
	mtx.scale_basis(Vector2(uv_draw_zoom, uv_draw_zoom));

	VS::get_singleton()->canvas_item_add_set_transform(uv_edit_draw->get_canvas_item(), mtx);
	uv_edit_draw->draw_texture(base_tex, Point2());
	VS::get_singleton()->canvas_item_add_set_transform(uv_edit_draw->get_canvas_item(), Transform2D());

	if (snap_show_grid) {
		Size2 s = uv_edit_draw->get_size();
		int last_cell;

		if (snap_step.x != 0) {
			for (int i = 0; i < s.width; i++) {
				int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(i, 0)).x - snap_offset.x) / snap_step.x));
				if (i == 0)
					last_cell = cell;
				if (last_cell != cell)
					uv_edit_draw->draw_line(Point2(i, 0), Point2(i, s.height), Color(0.3, 0.7, 1, 0.3));
				last_cell = cell;
			}
		}

		if (snap_step.y != 0) {
			for (int i = 0; i < s.height; i++) {
				int cell = Math::fast_ftoi(Math::floor((mtx.affine_inverse().xform(Vector2(0, i)).y - snap_offset.y) / snap_step.y));
				if (i == 0)
					last_cell = cell;
				if (last_cell != cell)
					uv_edit_draw->draw_line(Point2(0, i), Point2(s.width, i), Color(0.3, 0.7, 1, 0.3));
				last_cell = cell;
			}
		}
	}

	PoolVector<Vector2> uvs = node->get_uv();
	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");

	Rect2 rect(Point2(), mtx.basis_xform(base_tex->get_size()));
	rect.expand_to(mtx.basis_xform(uv_edit_draw->get_size()));

	for (int i = 0; i < uvs.size(); i++) {

		int next = (i + 1) % uvs.size();
		uv_edit_draw->draw_line(mtx.xform(uvs[i]), mtx.xform(uvs[next]), Color(0.9, 0.5, 0.5), 2);
		uv_edit_draw->draw_texture(handle, mtx.xform(uvs[i]) - handle->get_size() * 0.5);
		rect.expand_to(mtx.basis_xform(uvs[i]));
	}

	rect = rect.grow(200);
	updating_uv_scroll = true;
	uv_hscroll->set_min(rect.pos.x);
	uv_hscroll->set_max(rect.pos.x + rect.size.x);
	uv_hscroll->set_page(uv_edit_draw->get_size().x);
	uv_hscroll->set_value(uv_draw_ofs.x);
	uv_hscroll->set_step(0.001);

	uv_vscroll->set_min(rect.pos.y);
	uv_vscroll->set_max(rect.pos.y + rect.size.y);
	uv_vscroll->set_page(uv_edit_draw->get_size().y);
	uv_vscroll->set_value(uv_draw_ofs.y);
	uv_vscroll->set_step(0.001);
	updating_uv_scroll = false;
}

void Polygon2DEditor::edit(Node *p_collision_polygon) {

	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton();
	}

	if (p_collision_polygon) {

		node = p_collision_polygon->cast_to<Polygon2D>();
		if (!canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->connect("draw", this, "_canvas_draw");

		wip.clear();
		wip_active = false;
		edited_point = -1;

	} else {

		node = NULL;

		if (canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->disconnect("draw", this, "_canvas_draw");
	}
}

void Polygon2DEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_menu_option"), &Polygon2DEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_canvas_draw"), &Polygon2DEditor::_canvas_draw);
	ClassDB::bind_method(D_METHOD("_uv_mode"), &Polygon2DEditor::_uv_mode);
	ClassDB::bind_method(D_METHOD("_uv_draw"), &Polygon2DEditor::_uv_draw);
	ClassDB::bind_method(D_METHOD("_uv_input"), &Polygon2DEditor::_uv_input);
	ClassDB::bind_method(D_METHOD("_uv_scroll_changed"), &Polygon2DEditor::_uv_scroll_changed);
	ClassDB::bind_method(D_METHOD("_node_removed"), &Polygon2DEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_set_use_snap"), &Polygon2DEditor::_set_use_snap);
	ClassDB::bind_method(D_METHOD("_set_show_grid"), &Polygon2DEditor::_set_show_grid);
	ClassDB::bind_method(D_METHOD("_set_snap_off_x"), &Polygon2DEditor::_set_snap_off_x);
	ClassDB::bind_method(D_METHOD("_set_snap_off_y"), &Polygon2DEditor::_set_snap_off_y);
	ClassDB::bind_method(D_METHOD("_set_snap_step_x"), &Polygon2DEditor::_set_snap_step_x);
	ClassDB::bind_method(D_METHOD("_set_snap_step_y"), &Polygon2DEditor::_set_snap_step_y);
}

inline float _snap_scalar(float p_offset, float p_step, float p_target) {
	return p_step != 0 ? Math::stepify(p_target - p_offset, p_step) + p_offset : p_target;
}

Vector2 Polygon2DEditor::snap_point(Vector2 p_target) const {
	if (use_snap) {
		p_target.x = _snap_scalar(snap_offset.x * uv_draw_zoom - uv_draw_ofs.x, snap_step.x * uv_draw_zoom, p_target.x);
		p_target.y = _snap_scalar(snap_offset.y * uv_draw_zoom - uv_draw_ofs.y, snap_step.y * uv_draw_zoom, p_target.y);
	}

	return p_target;
}

Polygon2DEditor::Polygon2DEditor(EditorNode *p_editor) {

	node = NULL;
	canvas_item_editor = NULL;
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	snap_step = Vector2(10, 10);
	use_snap = false;
	snap_show_grid = false;

	add_child(memnew(VSeparator));
	button_create = memnew(ToolButton);
	add_child(button_create);
	button_create->connect("pressed", this, "_menu_option", varray(MODE_CREATE));
	button_create->set_toggle_mode(true);

	button_edit = memnew(ToolButton);
	add_child(button_edit);
	button_edit->connect("pressed", this, "_menu_option", varray(MODE_EDIT));
	button_edit->set_toggle_mode(true);

	button_uv = memnew(ToolButton);
	add_child(button_uv);
	button_uv->connect("pressed", this, "_menu_option", varray(MODE_EDIT_UV));

//add_constant_override("separation",0);

#if 0
	options = memnew( MenuButton );
	add_child(options);
	options->set_area_as_parent_rect();
	options->set_text("Polygon");
	//options->get_popup()->add_item("Parse BBCode",PARSE_BBCODE);
	options->get_popup()->connect("id_pressed", this,"_menu_option");
#endif

	mode = MODE_EDIT;
	wip_active = false;

	uv_mode = UV_MODE_EDIT_POINT;
	uv_edit = memnew(AcceptDialog);
	add_child(uv_edit);
	uv_edit->set_title(TTR("Polygon 2D UV Editor"));
	uv_edit->set_self_modulate(Color(1, 1, 1, 0.9));

	VBoxContainer *uv_main_vb = memnew(VBoxContainer);
	uv_edit->add_child(uv_main_vb);
	//uv_edit->set_child_rect(uv_main_vb);
	HBoxContainer *uv_mode_hb = memnew(HBoxContainer);
	uv_main_vb->add_child(uv_mode_hb);
	for (int i = 0; i < UV_MODE_MAX; i++) {

		uv_button[i] = memnew(ToolButton);
		uv_button[i]->set_toggle_mode(true);
		uv_mode_hb->add_child(uv_button[i]);
		uv_button[i]->connect("pressed", this, "_uv_mode", varray(i));
		uv_button[i]->set_focus_mode(FOCUS_NONE);
	}

	uv_button[0]->set_tooltip(TTR("Move Point") + "\n" + TTR("Ctrl: Rotate") + "\n" + TTR("Shift: Move All") + "\n" + TTR("Shift+Ctrl: Scale"));
	uv_button[1]->set_tooltip(TTR("Move Polygon"));
	uv_button[2]->set_tooltip(TTR("Rotate Polygon"));
	uv_button[3]->set_tooltip(TTR("Scale Polygon"));

	uv_button[0]->set_pressed(true);
	HBoxContainer *uv_main_hb = memnew(HBoxContainer);
	uv_main_vb->add_child(uv_main_hb);
	uv_edit_draw = memnew(Control);
	uv_main_hb->add_child(uv_edit_draw);
	uv_main_hb->set_v_size_flags(SIZE_EXPAND_FILL);
	uv_edit_draw->set_h_size_flags(SIZE_EXPAND_FILL);
	uv_menu = memnew(MenuButton);
	uv_mode_hb->add_child(uv_menu);
	uv_menu->set_text(TTR("Edit"));
	uv_menu->get_popup()->add_item(TTR("Polygon->UV"), UVEDIT_POLYGON_TO_UV);
	uv_menu->get_popup()->add_item(TTR("UV->Polygon"), UVEDIT_UV_TO_POLYGON);
	uv_menu->get_popup()->add_separator();
	uv_menu->get_popup()->add_item(TTR("Clear UV"), UVEDIT_UV_CLEAR);
	uv_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	uv_mode_hb->add_child(memnew(VSeparator));

	b_snap_enable = memnew(ToolButton);
	uv_mode_hb->add_child(b_snap_enable);
	b_snap_enable->set_text(TTR("Snap"));
	b_snap_enable->set_focus_mode(FOCUS_NONE);
	b_snap_enable->set_toggle_mode(true);
	b_snap_enable->set_pressed(use_snap);
	b_snap_enable->set_tooltip(TTR("Enable Snap"));
	b_snap_enable->connect("toggled", this, "_set_use_snap");

	b_snap_grid = memnew(ToolButton);
	uv_mode_hb->add_child(b_snap_grid);
	b_snap_grid->set_text(TTR("Grid"));
	b_snap_grid->set_focus_mode(FOCUS_NONE);
	b_snap_grid->set_toggle_mode(true);
	b_snap_grid->set_pressed(snap_show_grid);
	b_snap_grid->set_tooltip(TTR("Show Grid"));
	b_snap_grid->connect("toggled", this, "_set_show_grid");

	uv_mode_hb->add_child(memnew(VSeparator));
	uv_mode_hb->add_child(memnew(Label(TTR("Grid Offset:"))));

	SpinBox *sb_off_x = memnew(SpinBox);
	sb_off_x->set_min(-256);
	sb_off_x->set_max(256);
	sb_off_x->set_step(1);
	sb_off_x->set_value(snap_offset.x);
	sb_off_x->set_suffix("px");
	sb_off_x->connect("value_changed", this, "_set_snap_off_x");
	uv_mode_hb->add_child(sb_off_x);

	SpinBox *sb_off_y = memnew(SpinBox);
	sb_off_y->set_min(-256);
	sb_off_y->set_max(256);
	sb_off_y->set_step(1);
	sb_off_y->set_value(snap_offset.y);
	sb_off_y->set_suffix("px");
	sb_off_y->connect("value_changed", this, "_set_snap_off_y");
	uv_mode_hb->add_child(sb_off_y);

	uv_mode_hb->add_child(memnew(VSeparator));
	uv_mode_hb->add_child(memnew(Label(TTR("Grid Step:"))));

	SpinBox *sb_step_x = memnew(SpinBox);
	sb_step_x->set_min(-256);
	sb_step_x->set_max(256);
	sb_step_x->set_step(1);
	sb_step_x->set_value(snap_step.x);
	sb_step_x->set_suffix("px");
	sb_step_x->connect("value_changed", this, "_set_snap_step_x");
	uv_mode_hb->add_child(sb_step_x);

	SpinBox *sb_step_y = memnew(SpinBox);
	sb_step_y->set_min(-256);
	sb_step_y->set_max(256);
	sb_step_y->set_step(1);
	sb_step_y->set_value(snap_step.y);
	sb_step_y->set_suffix("px");
	sb_step_y->connect("value_changed", this, "_set_snap_step_y");
	uv_mode_hb->add_child(sb_step_y);

	uv_mode_hb->add_child(memnew(VSeparator));
	uv_icon_zoom = memnew(TextureRect);
	uv_mode_hb->add_child(uv_icon_zoom);
	uv_zoom = memnew(HSlider);
	uv_zoom->set_min(0.01);
	uv_zoom->set_max(4);
	uv_zoom->set_value(1);
	uv_zoom->set_step(0.01);
	uv_mode_hb->add_child(uv_zoom);
	uv_zoom->set_custom_minimum_size(Size2(200, 0));
	uv_zoom_value = memnew(SpinBox);
	uv_zoom->share(uv_zoom_value);
	uv_zoom_value->set_custom_minimum_size(Size2(50, 0));
	uv_mode_hb->add_child(uv_zoom_value);
	uv_zoom->connect("value_changed", this, "_uv_scroll_changed");

	uv_vscroll = memnew(VScrollBar);
	uv_main_hb->add_child(uv_vscroll);
	uv_vscroll->connect("value_changed", this, "_uv_scroll_changed");
	uv_hscroll = memnew(HScrollBar);
	uv_main_vb->add_child(uv_hscroll);
	uv_hscroll->connect("value_changed", this, "_uv_scroll_changed");

	uv_edit_draw->connect("draw", this, "_uv_draw");
	uv_edit_draw->connect("gui_input", this, "_uv_input");
	uv_draw_zoom = 1.0;
	uv_drag_index = -1;
	uv_drag = false;
	updating_uv_scroll = false;

	error = memnew(AcceptDialog);
	add_child(error);

	uv_edit_draw->set_clip_contents(true);
}

void Polygon2DEditorPlugin::edit(Object *p_object) {

	collision_polygon_editor->edit(p_object->cast_to<Node>());
}

bool Polygon2DEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Polygon2D");
}

void Polygon2DEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		collision_polygon_editor->show();
	} else {

		collision_polygon_editor->hide();
		collision_polygon_editor->edit(NULL);
	}
}

Polygon2DEditorPlugin::Polygon2DEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	collision_polygon_editor = memnew(Polygon2DEditor(p_node));
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(collision_polygon_editor);

	collision_polygon_editor->hide();
}

Polygon2DEditorPlugin::~Polygon2DEditorPlugin() {
}
