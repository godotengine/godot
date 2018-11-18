/*************************************************************************/
/*  polygon_2d_editor_plugin.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core/os/file_access.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "editor/editor_settings.h"
#include "scene/2d/skeleton_2d.h"

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

			uv_button[UV_MODE_CREATE]->set_icon(get_icon("Add", "EditorIcons"));
			uv_button[UV_MODE_EDIT_POINT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
			uv_button[UV_MODE_MOVE]->set_icon(get_icon("ToolMove", "EditorIcons"));
			uv_button[UV_MODE_ROTATE]->set_icon(get_icon("ToolRotate", "EditorIcons"));
			uv_button[UV_MODE_SCALE]->set_icon(get_icon("ToolScale", "EditorIcons"));
			uv_button[UV_MODE_ADD_SPLIT]->set_icon(get_icon("AddSplit", "EditorIcons"));
			uv_button[UV_MODE_REMOVE_SPLIT]->set_icon(get_icon("DeleteSplit", "EditorIcons"));
			uv_button[UV_MODE_PAINT_WEIGHT]->set_icon(get_icon("PaintVertex", "EditorIcons"));
			uv_button[UV_MODE_CLEAR_WEIGHT]->set_icon(get_icon("UnpaintVertex", "EditorIcons"));

			b_snap_grid->set_icon(get_icon("Grid", "EditorIcons"));
			b_snap_enable->set_icon(get_icon("SnapGrid", "EditorIcons"));
			uv_icon_zoom->set_texture(get_icon("Zoom", "EditorIcons"));

		} break;
		case NOTIFICATION_PHYSICS_PROCESS: {

		} break;
	}
}

void Polygon2DEditor::_sync_bones() {

	if (!node->has_node(node->get_skeleton())) {
		error->set_text(TTR("The skeleton property of the Polygon2D does not point to a Skeleton2D node"));
		error->popup_centered_minsize();
		return;
	}

	Node *sn = node->get_node(node->get_skeleton());
	Skeleton2D *skeleton = Object::cast_to<Skeleton2D>(sn);

	if (!skeleton) {
		error->set_text(TTR("The skeleton property of the Polygon2D does not point to a Skeleton2D node"));
		error->popup_centered_minsize();
		return;
	}

	Array prev_bones = node->call("_get_bones");
	node->clear_bones();

	for (int i = 0; i < skeleton->get_bone_count(); i++) {
		NodePath path = skeleton->get_path_to(skeleton->get_bone(i));
		PoolVector<float> weights;
		int wc = node->get_polygon().size();

		for (int j = 0; j < prev_bones.size(); j += 2) {
			NodePath pvp = prev_bones[j];
			PoolVector<float> pv = prev_bones[j + 1];
			if (pvp == path && pv.size() == wc) {
				weights = pv;
			}
		}

		if (weights.size() == 0) { //create them
			weights.resize(node->get_polygon().size());
			PoolVector<float>::Write w = weights.write();
			for (int j = 0; j < wc; j++) {
				w[j] = 0.0;
			}
		}

		node->add_bone(path, weights);
	}
	Array new_bones = node->call("_get_bones");

	undo_redo->create_action(TTR("Sync bones"));
	undo_redo->add_do_method(node, "_set_bones", new_bones);
	undo_redo->add_undo_method(node, "_set_bones", prev_bones);
	undo_redo->add_do_method(uv_edit_draw, "update");
	undo_redo->add_undo_method(uv_edit_draw, "update");
	undo_redo->add_do_method(this, "_update_bone_list");
	undo_redo->add_undo_method(this, "_update_bone_list");
	undo_redo->commit_action();
}

void Polygon2DEditor::_update_bone_list() {

	NodePath selected;
	while (bone_scroll_vb->get_child_count()) {
		CheckBox *cb = Object::cast_to<CheckBox>(bone_scroll_vb->get_child(0));
		if (cb && cb->is_pressed()) {
			selected = cb->get_meta("bone_path");
		}
		memdelete(bone_scroll_vb->get_child(0));
	}

	Ref<ButtonGroup> bg;
	bg.instance();
	for (int i = 0; i < node->get_bone_count(); i++) {
		CheckBox *cb = memnew(CheckBox);
		NodePath np = node->get_bone_path(i);
		String name;
		if (np.get_name_count()) {
			name = np.get_name(np.get_name_count() - 1);
		}
		if (name == String()) {
			name = "Bone " + itos(i);
		}
		cb->set_text(name);
		cb->set_button_group(bg);
		cb->set_meta("bone_path", np);
		bone_scroll_vb->add_child(cb);

		if (np == selected)
			cb->set_pressed(true);

		cb->connect("pressed", this, "_bone_paint_selected", varray(i));
	}

	uv_edit_draw->update();
}

void Polygon2DEditor::_bone_paint_selected(int p_index) {
	uv_edit_draw->update();
}

void Polygon2DEditor::_uv_edit_mode_select(int p_mode) {

	if (p_mode == 0) { //uv
		uv_button[UV_MODE_CREATE]->hide();
		for (int i = UV_MODE_MOVE; i <= UV_MODE_SCALE; i++) {
			uv_button[i]->show();
		}
		uv_button[UV_MODE_ADD_SPLIT]->hide();
		uv_button[UV_MODE_REMOVE_SPLIT]->hide();
		uv_button[UV_MODE_PAINT_WEIGHT]->hide();
		uv_button[UV_MODE_CLEAR_WEIGHT]->hide();
		_uv_mode(UV_MODE_EDIT_POINT);

		bone_scroll_main_vb->hide();
		bone_paint_strength->hide();
		bone_paint_radius->hide();
		bone_paint_radius_label->hide();

	} else if (p_mode == 1) { //poly
		for (int i = 0; i <= UV_MODE_SCALE; i++) {
			uv_button[i]->show();
		}
		uv_button[UV_MODE_ADD_SPLIT]->hide();
		uv_button[UV_MODE_REMOVE_SPLIT]->hide();
		uv_button[UV_MODE_PAINT_WEIGHT]->hide();
		uv_button[UV_MODE_CLEAR_WEIGHT]->hide();
		_uv_mode(UV_MODE_EDIT_POINT);

		bone_scroll_main_vb->hide();
		bone_paint_strength->hide();
		bone_paint_radius->hide();
		bone_paint_radius_label->hide();

	} else if (p_mode == 2) { //splits
		for (int i = 0; i <= UV_MODE_SCALE; i++) {
			uv_button[i]->hide();
		}
		uv_button[UV_MODE_ADD_SPLIT]->show();
		uv_button[UV_MODE_REMOVE_SPLIT]->show();
		uv_button[UV_MODE_PAINT_WEIGHT]->hide();
		uv_button[UV_MODE_CLEAR_WEIGHT]->hide();
		_uv_mode(UV_MODE_ADD_SPLIT);

		bone_scroll_main_vb->hide();
		bone_paint_strength->hide();
		bone_paint_radius->hide();
		bone_paint_radius_label->hide();

	} else if (p_mode == 3) { //bones´
		for (int i = 0; i <= UV_MODE_REMOVE_SPLIT; i++) {
			uv_button[i]->hide();
		}
		uv_button[UV_MODE_PAINT_WEIGHT]->show();
		uv_button[UV_MODE_CLEAR_WEIGHT]->show();
		_uv_mode(UV_MODE_PAINT_WEIGHT);

		bone_scroll_main_vb->show();
		bone_paint_strength->show();
		bone_paint_radius->show();
		bone_paint_radius_label->show();
		_update_bone_list();
		bone_paint_pos = Vector2(-100000, -100000); //send brush away when switching
	}

	uv_edit_draw->update();
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
		case UVEDIT_GRID_SETTINGS: {
			grid_settings->popup_centered_minsize();
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

	split_create = false;
	uv_drag = false;
	uv_create = false;

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
				points_prev = node->get_uv();

				if (uv_edit_mode[0]->is_pressed()) { //edit uv
					points_prev = node->get_uv();
				} else { //edit polygon
					points_prev = node->get_polygon();
				}

				uv_move_current = uv_mode;
				if (uv_move_current == UV_MODE_CREATE) {

					if (!uv_create) {
						points_prev.resize(0);
						Vector2 tuv = mtx.affine_inverse().xform(Vector2(mb->get_position().x, mb->get_position().y));
						points_prev.push_back(tuv);
						uv_create_to = tuv;
						point_drag_index = 0;
						uv_drag_from = tuv;
						uv_drag = true;
						uv_create = true;
						uv_create_uv_prev = node->get_uv();
						uv_create_poly_prev = node->get_polygon();
						uv_create_bones_prev = node->call("_get_bones");
						splits_prev = node->get_splits();
						node->set_polygon(points_prev);
						node->set_uv(points_prev);

					} else {
						Vector2 tuv = mtx.affine_inverse().xform(Vector2(mb->get_position().x, mb->get_position().y));

						if (points_prev.size() > 3 && tuv.distance_to(points_prev[0]) < 8) {
							undo_redo->create_action(TTR("Create Polygon & UV"));
							undo_redo->add_do_method(node, "set_uv", node->get_uv());
							undo_redo->add_undo_method(node, "set_uv", points_prev);
							undo_redo->add_do_method(node, "set_polygon", node->get_polygon());
							undo_redo->add_undo_method(node, "set_polygon", points_prev);
							undo_redo->add_do_method(node, "clear_bones");
							undo_redo->add_undo_method(node, "_set_bones", node->call("_get_bones"));
							undo_redo->add_do_method(uv_edit_draw, "update");
							undo_redo->add_undo_method(uv_edit_draw, "update");
							undo_redo->commit_action();
							uv_drag = false;
							uv_create = false;
							_uv_mode(UV_MODE_EDIT_POINT);
						} else {
							points_prev.push_back(tuv);
							point_drag_index = points_prev.size() - 1;
							uv_drag_from = tuv;
						}
						node->set_polygon(points_prev);
						node->set_uv(points_prev);
					}
				}

				if (uv_move_current == UV_MODE_EDIT_POINT) {

					if (mb->get_shift() && mb->get_command())
						uv_move_current = UV_MODE_SCALE;
					else if (mb->get_shift())
						uv_move_current = UV_MODE_MOVE;
					else if (mb->get_command())
						uv_move_current = UV_MODE_ROTATE;
				}

				if (uv_move_current == UV_MODE_EDIT_POINT) {

					point_drag_index = -1;
					for (int i = 0; i < points_prev.size(); i++) {

						Vector2 tuv = mtx.xform(points_prev[i]);
						if (tuv.distance_to(Vector2(mb->get_position().x, mb->get_position().y)) < 8) {
							uv_drag_from = tuv;
							point_drag_index = i;
						}
					}

					if (point_drag_index == -1) {
						uv_drag = false;
					}
				}

				if (uv_move_current == UV_MODE_ADD_SPLIT) {

					int split_to_index = -1;
					split_to_index = -1;
					for (int i = 0; i < points_prev.size(); i++) {

						Vector2 tuv = mtx.xform(points_prev[i]);
						if (tuv.distance_to(Vector2(mb->get_position().x, mb->get_position().y)) < 8) {
							split_to_index = i;
						}
					}

					if (split_to_index == -1) {
						split_create = false;
						return;
					}

					if (split_create) {

						split_create = false;
						if (split_to_index < point_drag_index) {
							SWAP(split_to_index, point_drag_index);
						}
						bool valid = true;
						String split_error;
						if (split_to_index == point_drag_index) {
							split_error = TTR("Split point with itself.");
							valid = false;
						}
						if (split_to_index + 1 == point_drag_index) {
							//not a split,goes along the edge
							split_error = TTR("Split can't form an existing edge.");
							valid = false;
						}
						if (split_to_index == points_prev.size() - 1 && point_drag_index == 0) {
							//not a split,goes along the edge
							split_error = TTR("Split can't form an existing edge.");
							valid = false;
						}

						for (int i = 0; i < splits_prev.size(); i += 2) {

							if (splits_prev[i] == point_drag_index && splits_prev[i + 1] == split_to_index) {
								//already exists
								split_error = TTR("Split already exists.");
								valid = false;
								break;
							}

							int a_state; //-1, outside split, 0 split point, +1, inside split
							if (point_drag_index == splits_prev[i] || point_drag_index == splits_prev[i + 1]) {
								a_state = 0;
							} else if (point_drag_index < splits_prev[i] || point_drag_index > splits_prev[i + 1]) {
								a_state = -1;
							} else {
								a_state = 1;
							}

							int b_state; //-1, outside split, 0 split point, +1, inside split
							if (split_to_index == splits_prev[i] || split_to_index == splits_prev[i + 1]) {
								b_state = 0;
							} else if (split_to_index < splits_prev[i] || split_to_index > splits_prev[i + 1]) {
								b_state = -1;
							} else {
								b_state = 1;
							}

							if (b_state * a_state < 0) {
								//crossing
								split_error = "Split crosses another split.";
								valid = false;
								break;
							}
						}

						if (valid) {

							splits_prev.push_back(point_drag_index);
							splits_prev.push_back(split_to_index);

							undo_redo->create_action(TTR("Add Split"));

							undo_redo->add_do_method(node, "set_splits", splits_prev);
							undo_redo->add_undo_method(node, "set_splits", node->get_splits());
							undo_redo->add_do_method(uv_edit_draw, "update");
							undo_redo->add_undo_method(uv_edit_draw, "update");
							undo_redo->commit_action();
						} else {
							error->set_text(TTR("Invalid Split: ") + split_error);
							error->popup_centered_minsize();
						}

					} else {
						point_drag_index = split_to_index;
						split_create = true;
						splits_prev = node->get_splits();
						uv_create_to = mtx.affine_inverse().xform(Vector2(mb->get_position().x, mb->get_position().y));
					}
				}

				if (uv_move_current == UV_MODE_REMOVE_SPLIT) {

					splits_prev = node->get_splits();
					for (int i = 0; i < splits_prev.size(); i += 2) {
						if (splits_prev[i] < 0 || splits_prev[i] >= points_prev.size())
							continue;
						if (splits_prev[i + 1] < 0 || splits_prev[i] >= points_prev.size())
							continue;
						Vector2 e[2] = { mtx.xform(points_prev[splits_prev[i]]), mtx.xform(points_prev[splits_prev[i + 1]]) };
						Vector2 mp = Vector2(mb->get_position().x, mb->get_position().y);
						Vector2 cp = Geometry::get_closest_point_to_segment_2d(mp, e);
						if (cp.distance_to(mp) < 8) {
							splits_prev.remove(i);
							splits_prev.remove(i);

							undo_redo->create_action(TTR("Remove Split"));

							undo_redo->add_do_method(node, "set_splits", splits_prev);
							undo_redo->add_undo_method(node, "set_splits", node->get_splits());
							undo_redo->add_do_method(uv_edit_draw, "update");
							undo_redo->add_undo_method(uv_edit_draw, "update");
							undo_redo->commit_action();

							break;
						}
					}
				}

				if (uv_move_current == UV_MODE_PAINT_WEIGHT || uv_move_current == UV_MODE_CLEAR_WEIGHT) {

					int bone_selected = -1;
					for (int i = 0; i < bone_scroll_vb->get_child_count(); i++) {
						CheckBox *c = Object::cast_to<CheckBox>(bone_scroll_vb->get_child(i));
						if (c && c->is_pressed()) {
							bone_selected = i;
							break;
						}
					}

					if (bone_selected != -1 && node->get_bone_weights(bone_selected).size() == points_prev.size()) {

						prev_weights = node->get_bone_weights(bone_selected);
						bone_painting = true;
						bone_painting_bone = bone_selected;
					}
				}

			} else if (uv_drag && !uv_create) {

				undo_redo->create_action(TTR("Transform UV Map"));

				if (uv_edit_mode[0]->is_pressed()) { //edit uv
					undo_redo->add_do_method(node, "set_uv", node->get_uv());
					undo_redo->add_undo_method(node, "set_uv", points_prev);
				} else if (uv_edit_mode[1]->is_pressed()) { //edit polygon
					undo_redo->add_do_method(node, "set_polygon", node->get_polygon());
					undo_redo->add_undo_method(node, "set_polygon", points_prev);
				}
				undo_redo->add_do_method(uv_edit_draw, "update");
				undo_redo->add_undo_method(uv_edit_draw, "update");
				undo_redo->commit_action();

				uv_drag = false;
			} else if (bone_painting) {

				undo_redo->create_action(TTR("Paint bone weights"));
				undo_redo->add_do_method(node, "set_bone_weights", bone_painting_bone, node->get_bone_weights(bone_painting_bone));
				undo_redo->add_undo_method(node, "set_bone_weights", bone_painting_bone, prev_weights);
				undo_redo->add_do_method(uv_edit_draw, "update");
				undo_redo->add_undo_method(uv_edit_draw, "update");
				undo_redo->commit_action();
				bone_painting = false;
			}

		} else if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {

			if (uv_create) {

				uv_drag = false;
				uv_create = false;
				node->set_uv(uv_create_uv_prev);
				node->set_polygon(uv_create_poly_prev);
				node->call("_set_bones", uv_create_bones_prev);
				node->set_splits(splits_prev);
				uv_edit_draw->update();
			} else if (uv_drag) {

				uv_drag = false;
				if (uv_edit_mode[0]->is_pressed()) { //edit uv
					node->set_uv(points_prev);
				} else if (uv_edit_mode[1]->is_pressed()) { //edit polygon
					node->set_polygon(points_prev);
				}
				uv_edit_draw->update();
			} else if (split_create) {
				split_create = false;
				uv_edit_draw->update();
			} else if (bone_painting) {
				node->set_bone_weights(bone_painting_bone, prev_weights);
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

				case UV_MODE_CREATE: {
					if (uv_create) {
						uv_create_to = mtx.affine_inverse().xform(Vector2(mm->get_position().x, mm->get_position().y));
					}
				} break;
				case UV_MODE_EDIT_POINT: {

					PoolVector<Vector2> uv_new = points_prev;
					uv_new.set(point_drag_index, uv_new[point_drag_index] + drag);

					if (uv_edit_mode[0]->is_pressed()) { //edit uv
						node->set_uv(uv_new);
					} else if (uv_edit_mode[1]->is_pressed()) { //edit polygon
						node->set_polygon(uv_new);
					}
				} break;
				case UV_MODE_MOVE: {

					PoolVector<Vector2> uv_new = points_prev;
					for (int i = 0; i < uv_new.size(); i++)
						uv_new.set(i, uv_new[i] + drag);

					if (uv_edit_mode[0]->is_pressed()) { //edit uv
						node->set_uv(uv_new);
					} else if (uv_edit_mode[1]->is_pressed()) { //edit polygon
						node->set_polygon(uv_new);
					}

				} break;
				case UV_MODE_ROTATE: {

					Vector2 center;
					PoolVector<Vector2> uv_new = points_prev;

					for (int i = 0; i < uv_new.size(); i++)
						center += points_prev[i];
					center /= uv_new.size();

					float angle = (uv_drag_from - mtx.xform(center)).normalized().angle_to((uv_drag_to - mtx.xform(center)).normalized());

					for (int i = 0; i < uv_new.size(); i++) {
						Vector2 rel = points_prev[i] - center;
						rel = rel.rotated(angle);
						uv_new.set(i, center + rel);
					}

					if (uv_edit_mode[0]->is_pressed()) { //edit uv
						node->set_uv(uv_new);
					} else if (uv_edit_mode[1]->is_pressed()) { //edit polygon
						node->set_polygon(uv_new);
					}

				} break;
				case UV_MODE_SCALE: {

					Vector2 center;
					PoolVector<Vector2> uv_new = points_prev;

					for (int i = 0; i < uv_new.size(); i++)
						center += points_prev[i];
					center /= uv_new.size();

					float from_dist = uv_drag_from.distance_to(mtx.xform(center));
					float to_dist = uv_drag_to.distance_to(mtx.xform(center));
					if (from_dist < 2)
						break;

					float scale = to_dist / from_dist;

					for (int i = 0; i < uv_new.size(); i++) {
						Vector2 rel = points_prev[i] - center;
						rel = rel * scale;
						uv_new.set(i, center + rel);
					}

					if (uv_edit_mode[0]->is_pressed()) { //edit uv
						node->set_uv(uv_new);
					} else if (uv_edit_mode[1]->is_pressed()) { //edit polygon
						node->set_polygon(uv_new);
					}
				} break;
				default: {}
			}

			if (bone_painting) {
				bone_paint_pos = Vector2(mm->get_position().x, mm->get_position().y);
				PoolVector<float> painted_weights = node->get_bone_weights(bone_painting_bone);

				{
					int pc = painted_weights.size();
					float amount = bone_paint_strength->get_value();
					float radius = bone_paint_radius->get_value() * EDSCALE;

					if (uv_mode == UV_MODE_CLEAR_WEIGHT) {
						amount = -amount;
					}

					PoolVector<float>::Write w = painted_weights.write();
					PoolVector<float>::Read r = prev_weights.read();
					PoolVector<Vector2>::Read rv = points_prev.read();

					for (int i = 0; i < pc; i++) {
						if (mtx.xform(rv[i]).distance_to(bone_paint_pos) < radius) {
							w[i] = CLAMP(r[i] + amount, 0, 1);
						}
					}
				}

				node->set_bone_weights(bone_painting_bone, painted_weights);
			}
			uv_edit_draw->update();
		} else if (split_create) {
			uv_create_to = mtx.affine_inverse().xform(Vector2(mm->get_position().x, mm->get_position().y));
			uv_edit_draw->update();
		} else if (uv_mode == UV_MODE_PAINT_WEIGHT || uv_mode == UV_MODE_CLEAR_WEIGHT) {
			bone_paint_pos = Vector2(mm->get_position().x, mm->get_position().y);
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

	String warning;

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

	PoolVector<Vector2> uvs;
	if (uv_edit_mode[0]->is_pressed()) { //edit uv
		uvs = node->get_uv();
	} else { //edit polygon
		uvs = node->get_polygon();
	}

	PoolVector<float>::Read weight_r;

	if (uv_edit_mode[3]->is_pressed()) {
		int bone_selected = -1;
		for (int i = 0; i < bone_scroll_vb->get_child_count(); i++) {
			CheckBox *c = Object::cast_to<CheckBox>(bone_scroll_vb->get_child(i));
			if (c && c->is_pressed()) {
				bone_selected = i;
				break;
			}
		}

		if (bone_selected != -1 && node->get_bone_weights(bone_selected).size() == uvs.size()) {

			weight_r = node->get_bone_weights(bone_selected).read();
		}
	}

	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");

	Rect2 rect(Point2(), mtx.basis_xform(base_tex->get_size()));
	rect.expand_to(mtx.basis_xform(uv_edit_draw->get_size()));

	for (int i = 0; i < uvs.size(); i++) {

		int next = (i + 1) % uvs.size();
		Vector2 next_point = uvs[next];
		if (uv_create && i == uvs.size() - 1) {
			next_point = uv_create_to;
		}
		uv_edit_draw->draw_line(mtx.xform(uvs[i]), mtx.xform(next_point), Color(0.9, 0.5, 0.5), 2);
		if (weight_r.ptr()) {

			Vector2 draw_pos = mtx.xform(uvs[i]);
			float weight = weight_r[i];
			uv_edit_draw->draw_rect(Rect2(draw_pos - Vector2(2, 2) * EDSCALE, Vector2(5, 5) * EDSCALE), Color(weight, weight, weight, 1.0));
		} else {
			uv_edit_draw->draw_texture(handle, mtx.xform(uvs[i]) - handle->get_size() * 0.5);
		}
		rect.expand_to(mtx.basis_xform(uvs[i]));
	}

	if (split_create) {
		Vector2 from = uvs[point_drag_index];
		Vector2 to = uv_create_to;
		uv_edit_draw->draw_line(mtx.xform(from), mtx.xform(to), Color(0.9, 0.5, 0.5), 2);
	}

	PoolVector<int> splits = node->get_splits();

	for (int i = 0; i < splits.size(); i += 2) {
		int idx_from = splits[i];
		int idx_to = splits[i + 1];
		if (idx_from < 0 || idx_to >= uvs.size())
			continue;
		uv_edit_draw->draw_line(mtx.xform(uvs[idx_from]), mtx.xform(uvs[idx_to]), Color(0.9, 0.5, 0.5), 2);
	}

	if (uv_mode == UV_MODE_PAINT_WEIGHT || uv_mode == UV_MODE_CLEAR_WEIGHT) {

		NodePath bone_path;
		for (int i = 0; i < bone_scroll_vb->get_child_count(); i++) {
			CheckBox *c = Object::cast_to<CheckBox>(bone_scroll_vb->get_child(i));
			if (c && c->is_pressed()) {
				bone_path = node->get_bone_path(i);
				break;
			}
		}

		//draw skeleton
		NodePath skeleton_path = node->get_skeleton();
		if (node->has_node(skeleton_path)) {
			Skeleton2D *skeleton = Object::cast_to<Skeleton2D>(node->get_node(skeleton_path));
			if (skeleton) {
				for (int i = 0; i < skeleton->get_bone_count(); i++) {

					Bone2D *bone = skeleton->get_bone(i);
					if (bone->get_rest() == Transform2D(0, 0, 0, 0, 0, 0))
						continue; //not set

					bool current = bone_path == skeleton->get_path_to(bone);

					for (int j = 0; j < bone->get_child_count(); j++) {

						Node2D *n = Object::cast_to<Node2D>(bone->get_child(j));
						if (!n)
							continue;

						bool edit_bone = n->has_meta("_edit_bone_") && n->get_meta("_edit_bone_");
						if (edit_bone) {

							Transform2D bone_xform = node->get_global_transform().affine_inverse() * (skeleton->get_global_transform() * bone->get_skeleton_rest());
							Transform2D endpoint_xform = bone_xform * n->get_transform();

							Color color = current ? Color(1, 1, 1) : Color(0.5, 0.5, 0.5);
							uv_edit_draw->draw_line(mtx.xform(bone_xform.get_origin()), mtx.xform(endpoint_xform.get_origin()), Color(0, 0, 0), current ? 5 : 4);
							uv_edit_draw->draw_line(mtx.xform(bone_xform.get_origin()), mtx.xform(endpoint_xform.get_origin()), color, current ? 3 : 2);
						}
					}
				}
			}
		}

		//draw paint circle
		uv_edit_draw->draw_circle(bone_paint_pos, bone_paint_radius->get_value() * EDSCALE, Color(1, 1, 1, 0.1));
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
	ClassDB::bind_method(D_METHOD("_uv_edit_mode_select"), &Polygon2DEditor::_uv_edit_mode_select);
	ClassDB::bind_method(D_METHOD("_sync_bones"), &Polygon2DEditor::_sync_bones);
	ClassDB::bind_method(D_METHOD("_update_bone_list"), &Polygon2DEditor::_update_bone_list);

	ClassDB::bind_method(D_METHOD("_bone_paint_selected"), &Polygon2DEditor::_bone_paint_selected);
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

	node = NULL;
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

	uv_edit_group.instance();

	uv_edit_mode[0] = memnew(ToolButton);
	uv_mode_hb->add_child(uv_edit_mode[0]);
	uv_edit_mode[0]->set_toggle_mode(true);
	uv_edit_mode[1] = memnew(ToolButton);
	uv_mode_hb->add_child(uv_edit_mode[1]);
	uv_edit_mode[1]->set_toggle_mode(true);
	uv_edit_mode[2] = memnew(ToolButton);
	uv_mode_hb->add_child(uv_edit_mode[2]);
	uv_edit_mode[2]->set_toggle_mode(true);
	uv_edit_mode[3] = memnew(ToolButton);
	uv_mode_hb->add_child(uv_edit_mode[3]);
	uv_edit_mode[3]->set_toggle_mode(true);

	uv_edit_mode[0]->set_text(TTR("UV"));
	uv_edit_mode[0]->set_pressed(true);
	uv_edit_mode[1]->set_text(TTR("Poly"));
	uv_edit_mode[2]->set_text(TTR("Splits"));
	uv_edit_mode[3]->set_text(TTR("Bones"));

	uv_edit_mode[0]->set_button_group(uv_edit_group);
	uv_edit_mode[1]->set_button_group(uv_edit_group);
	uv_edit_mode[2]->set_button_group(uv_edit_group);
	uv_edit_mode[3]->set_button_group(uv_edit_group);

	uv_edit_mode[0]->connect("pressed", this, "_uv_edit_mode_select", varray(0));
	uv_edit_mode[1]->connect("pressed", this, "_uv_edit_mode_select", varray(1));
	uv_edit_mode[2]->connect("pressed", this, "_uv_edit_mode_select", varray(2));
	uv_edit_mode[3]->connect("pressed", this, "_uv_edit_mode_select", varray(3));

	uv_mode_hb->add_child(memnew(VSeparator));

	uv_main_vb->add_child(uv_mode_hb);
	for (int i = 0; i < UV_MODE_MAX; i++) {

		uv_button[i] = memnew(ToolButton);
		uv_button[i]->set_toggle_mode(true);
		uv_mode_hb->add_child(uv_button[i]);
		uv_button[i]->connect("pressed", this, "_uv_mode", varray(i));
		uv_button[i]->set_focus_mode(FOCUS_NONE);
	}

	uv_button[0]->set_tooltip(TTR("Create Polygon"));
	uv_button[1]->set_tooltip(TTR("Move Point") + "\n" + TTR("Ctrl: Rotate") + "\n" + TTR("Shift: Move All") + "\n" + TTR("Shift+Ctrl: Scale"));
	uv_button[2]->set_tooltip(TTR("Move Polygon"));
	uv_button[3]->set_tooltip(TTR("Rotate Polygon"));
	uv_button[4]->set_tooltip(TTR("Scale Polygon"));
	uv_button[5]->set_tooltip(TTR("Connect two points to make a split"));
	uv_button[6]->set_tooltip(TTR("Select a split to erase it"));
	uv_button[7]->set_tooltip(TTR("Paint weights with specified intensity"));
	uv_button[8]->set_tooltip(TTR("UnPaint weights with specified intensity"));

	uv_button[0]->hide();
	uv_button[5]->hide();
	uv_button[6]->hide();
	uv_button[7]->hide();
	uv_button[8]->hide();
	uv_button[1]->set_pressed(true);

	bone_paint_strength = memnew(HSlider);
	uv_mode_hb->add_child(bone_paint_strength);
	bone_paint_strength->set_custom_minimum_size(Size2(75 * EDSCALE, 0));
	bone_paint_strength->set_v_size_flags(SIZE_SHRINK_CENTER);
	bone_paint_strength->set_min(0);
	bone_paint_strength->set_max(1);
	bone_paint_strength->set_step(0.01);
	bone_paint_strength->set_value(0.5);

	bone_paint_radius_label = memnew(Label(" " + TTR("Radius:") + " "));
	uv_mode_hb->add_child(bone_paint_radius_label);
	bone_paint_radius = memnew(SpinBox);
	uv_mode_hb->add_child(bone_paint_radius);

	bone_paint_strength->hide();
	bone_paint_radius->hide();
	bone_paint_radius_label->hide();
	bone_paint_radius->set_min(1);
	bone_paint_radius->set_max(100);
	bone_paint_radius->set_step(1);
	bone_paint_radius->set_value(32);

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
	uv_menu->get_popup()->add_separator();
	uv_menu->get_popup()->add_item(TTR("Grid Settings"), UVEDIT_GRID_SETTINGS);
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

	grid_settings = memnew(AcceptDialog);
	grid_settings->set_title(TTR("Configure Grid:"));
	add_child(grid_settings);
	VBoxContainer *grid_settings_vb = memnew(VBoxContainer);
	grid_settings->add_child(grid_settings_vb);

	SpinBox *sb_off_x = memnew(SpinBox);
	sb_off_x->set_min(-256);
	sb_off_x->set_max(256);
	sb_off_x->set_step(1);
	sb_off_x->set_value(snap_offset.x);
	sb_off_x->set_suffix("px");
	sb_off_x->connect("value_changed", this, "_set_snap_off_x");
	grid_settings_vb->add_margin_child(TTR("Grid Offset X:"), sb_off_x);

	SpinBox *sb_off_y = memnew(SpinBox);
	sb_off_y->set_min(-256);
	sb_off_y->set_max(256);
	sb_off_y->set_step(1);
	sb_off_y->set_value(snap_offset.y);
	sb_off_y->set_suffix("px");
	sb_off_y->connect("value_changed", this, "_set_snap_off_y");
	grid_settings_vb->add_margin_child(TTR("Grid Offset Y:"), sb_off_y);

	SpinBox *sb_step_x = memnew(SpinBox);
	sb_step_x->set_min(-256);
	sb_step_x->set_max(256);
	sb_step_x->set_step(1);
	sb_step_x->set_value(snap_step.x);
	sb_step_x->set_suffix("px");
	sb_step_x->connect("value_changed", this, "_set_snap_step_x");
	grid_settings_vb->add_margin_child(TTR("Grid Step X:"), sb_step_x);

	SpinBox *sb_step_y = memnew(SpinBox);
	sb_step_y->set_min(-256);
	sb_step_y->set_max(256);
	sb_step_y->set_step(1);
	sb_step_y->set_value(snap_step.y);
	sb_step_y->set_suffix("px");
	sb_step_y->connect("value_changed", this, "_set_snap_step_y");
	grid_settings_vb->add_margin_child(TTR("Grid Step Y:"), sb_step_y);

	uv_mode_hb->add_child(memnew(VSeparator));
	uv_icon_zoom = memnew(TextureRect);
	uv_mode_hb->add_child(uv_icon_zoom);
	uv_zoom = memnew(HSlider);
	uv_zoom->set_min(0.01);
	uv_zoom->set_max(4);
	uv_zoom->set_value(1);
	uv_zoom->set_step(0.01);
	uv_zoom->set_v_size_flags(SIZE_SHRINK_CENTER);

	uv_mode_hb->add_child(uv_zoom);
	uv_zoom->set_custom_minimum_size(Size2(80 * EDSCALE, 0));
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

	bone_scroll_main_vb = memnew(VBoxContainer);
	bone_scroll_main_vb->hide();
	sync_bones = memnew(Button(TTR("Sync Bones to Polygon")));
	bone_scroll_main_vb->add_child(sync_bones);
	uv_main_hb->add_child(bone_scroll_main_vb);
	bone_scroll = memnew(ScrollContainer);
	bone_scroll->set_v_scroll(true);
	bone_scroll->set_h_scroll(false);
	bone_scroll_main_vb->add_child(bone_scroll);
	bone_scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	bone_scroll_vb = memnew(VBoxContainer);
	bone_scroll->add_child(bone_scroll_vb);
	sync_bones->connect("pressed", this, "_sync_bones");

	uv_edit_draw->connect("draw", this, "_uv_draw");
	uv_edit_draw->connect("gui_input", this, "_uv_input");
	uv_draw_zoom = 1.0;
	point_drag_index = -1;
	uv_drag = false;
	uv_create = false;
	updating_uv_scroll = false;
	split_create = false;
	bone_painting = false;

	error = memnew(AcceptDialog);
	add_child(error);

	uv_edit_draw->set_clip_contents(true);
}

Polygon2DEditorPlugin::Polygon2DEditorPlugin(EditorNode *p_node) :
		AbstractPolygon2DEditorPlugin(p_node, memnew(Polygon2DEditor(p_node)), "Polygon2D") {
}
