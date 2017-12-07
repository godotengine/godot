/*************************************************************************/
/*  polygon_2d_editor_plugin.cpp                                         */
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
#include "polygon_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_settings.h"
#include "os/file_access.h"
#include "os/input.h"
#include "os/keyboard.h"

Node2D *Polygon2DEditor::_get_node() const {

	return node;
}

void Polygon2DEditor::_set_node(Node *p_polygon) {

	node = Object::cast_to<Polygon2D>(p_polygon);
}

Vector2 Polygon2DEditor::_get_offset(int p_idx) const {

	return node->get_offset();
}

void Polygon2DEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			button_uv->set_icon(get_icon("Uv", "EditorIcons"));

			uv_button[UV_MODE_EDIT_POINT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
			uv_button[UV_MODE_MOVE]->set_icon(get_icon("ToolMove", "EditorIcons"));
			uv_button[UV_MODE_ROTATE]->set_icon(get_icon("ToolRotate", "EditorIcons"));
			uv_button[UV_MODE_SCALE]->set_icon(get_icon("ToolScale", "EditorIcons"));

			b_snap_grid->set_icon(get_icon("Grid", "EditorIcons"));
			b_snap_enable->set_icon(get_icon("SnapGrid", "EditorIcons"));
			uv_icon_zoom->set_texture(get_icon("Zoom", "EditorIcons"));

		} break;
		case NOTIFICATION_PHYSICS_PROCESS: {

		} break;
	}
}

void Polygon2DEditor::_menu_option(int p_option) {

	switch (p_option) {

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
		default: {
			AbstractPolygon2DEditor::_menu_option(p_option);
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

void Polygon2DEditor::_uv_mode(int p_mode) {

	uv_mode = UVMode(p_mode);
	for (int i = 0; i < UV_MODE_MAX; i++) {
		uv_button[i]->set_pressed(p_mode == i);
	}
}

void Polygon2DEditor::_uv_input(const Ref<InputEvent> &p_input) {

	Transform2D mtx;
	mtx.elements[2] = -uv_draw_ofs;
	mtx.scale_basis(Vector2(uv_draw_zoom, uv_draw_zoom));

	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid()) {

		if (mb->get_button_index() == BUTTON_LEFT) {

			if (mb->is_pressed()) {

				uv_drag_from = Vector2(mb->get_position().x, mb->get_position().y);
				uv_drag = true;
				uv_prev = node->get_uv();
				uv_move_current = uv_mode;
				if (uv_move_current == UV_MODE_EDIT_POINT) {

					if (mb->get_shift() && mb->get_command())
						uv_move_current = UV_MODE_SCALE;
					else if (mb->get_shift())
						uv_move_current = UV_MODE_MOVE;
					else if (mb->get_command())
						uv_move_current = UV_MODE_ROTATE;
				}

				if (uv_move_current == UV_MODE_EDIT_POINT) {

					uv_drag_index = -1;
					for (int i = 0; i < uv_prev.size(); i++) {

						Vector2 tuv = mtx.xform(uv_prev[i]);
						if (tuv.distance_to(Vector2(mb->get_position().x, mb->get_position().y)) < 8) {
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

		} else if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {

			if (uv_drag) {

				uv_drag = false;
				node->set_uv(uv_prev);
				uv_edit_draw->update();
			}

		} else if (mb->get_button_index() == BUTTON_WHEEL_UP && mb->is_pressed()) {

			uv_zoom->set_value(uv_zoom->get_value() / (1 - (0.1 * mb->get_factor())));
		} else if (mb->get_button_index() == BUTTON_WHEEL_DOWN && mb->is_pressed()) {

			uv_zoom->set_value(uv_zoom->get_value() * (1 - (0.1 * mb->get_factor())));
		}
	}

	Ref<InputEventMouseMotion> mm = p_input;

	if (mm.is_valid()) {

		if ((mm->get_button_mask() & BUTTON_MASK_MIDDLE) || Input::get_singleton()->is_key_pressed(KEY_SPACE)) {

			Vector2 drag(mm->get_relative().x, mm->get_relative().y);
			uv_hscroll->set_value(uv_hscroll->get_value() - drag.x);
			uv_vscroll->set_value(uv_vscroll->get_value() - drag.y);

		} else if (uv_drag) {

			Vector2 uv_drag_to = mm->get_position();
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

	Ref<InputEventMagnifyGesture> magnify_gesture = p_input;
	if (magnify_gesture.is_valid()) {

		uv_zoom->set_value(uv_zoom->get_value() * magnify_gesture->get_factor());
	}

	Ref<InputEventPanGesture> pan_gesture = p_input;
	if (pan_gesture.is_valid()) {

		uv_hscroll->set_value(uv_hscroll->get_value() + uv_hscroll->get_page() * pan_gesture->get_delta().x / 8);
		uv_vscroll->set_value(uv_vscroll->get_value() + uv_vscroll->get_page() * pan_gesture->get_delta().y / 8);
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
		int last_cell = 0;

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
	uv_hscroll->set_min(rect.position.x);
	uv_hscroll->set_max(rect.position.x + rect.size.x);
	uv_hscroll->set_page(uv_edit_draw->get_size().x);
	uv_hscroll->set_value(uv_draw_ofs.x);
	uv_hscroll->set_step(0.001);

	uv_vscroll->set_min(rect.position.y);
	uv_vscroll->set_max(rect.position.y + rect.size.y);
	uv_vscroll->set_page(uv_edit_draw->get_size().y);
	uv_vscroll->set_value(uv_draw_ofs.y);
	uv_vscroll->set_step(0.001);
	updating_uv_scroll = false;
}

void Polygon2DEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_uv_mode"), &Polygon2DEditor::_uv_mode);
	ClassDB::bind_method(D_METHOD("_uv_draw"), &Polygon2DEditor::_uv_draw);
	ClassDB::bind_method(D_METHOD("_uv_input"), &Polygon2DEditor::_uv_input);
	ClassDB::bind_method(D_METHOD("_uv_scroll_changed"), &Polygon2DEditor::_uv_scroll_changed);
	ClassDB::bind_method(D_METHOD("_set_use_snap"), &Polygon2DEditor::_set_use_snap);
	ClassDB::bind_method(D_METHOD("_set_show_grid"), &Polygon2DEditor::_set_show_grid);
	ClassDB::bind_method(D_METHOD("_set_snap_off_x"), &Polygon2DEditor::_set_snap_off_x);
	ClassDB::bind_method(D_METHOD("_set_snap_off_y"), &Polygon2DEditor::_set_snap_off_y);
	ClassDB::bind_method(D_METHOD("_set_snap_step_x"), &Polygon2DEditor::_set_snap_step_x);
	ClassDB::bind_method(D_METHOD("_set_snap_step_y"), &Polygon2DEditor::_set_snap_step_y);
}

Vector2 Polygon2DEditor::snap_point(Vector2 p_target) const {
	if (use_snap) {
		p_target.x = Math::snap_scalar(snap_offset.x * uv_draw_zoom - uv_draw_ofs.x, snap_step.x * uv_draw_zoom, p_target.x);
		p_target.y = Math::snap_scalar(snap_offset.y * uv_draw_zoom - uv_draw_ofs.y, snap_step.y * uv_draw_zoom, p_target.y);
	}

	return p_target;
}

Polygon2DEditor::Polygon2DEditor(EditorNode *p_editor) :
		AbstractPolygon2DEditor(p_editor) {

	snap_step = Vector2(10, 10);
	use_snap = false;
	snap_show_grid = false;

	button_uv = memnew(ToolButton);
	add_child(button_uv);
	button_uv->connect("pressed", this, "_menu_option", varray(MODE_EDIT_UV));

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

Polygon2DEditorPlugin::Polygon2DEditorPlugin(EditorNode *p_node) :
		AbstractPolygon2DEditorPlugin(p_node, memnew(Polygon2DEditor(p_node)), "Polygon2D") {
}
