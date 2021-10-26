/*************************************************************************/
/*  collision_polygon_3d_editor_plugin.cpp                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "collision_polygon_3d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "core/input/input.h"
#include "core/io/file_access.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"
#include "editor/editor_settings.h"
#include "node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"

void CollisionPolygon3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			button_create->set_icon(get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
			button_edit->set_icon(get_theme_icon(SNAME("MovePoint"), SNAME("EditorIcons")));
			button_edit->set_pressed(true);
			get_tree()->connect("node_removed", callable_mp(this, &CollisionPolygon3DEditor::_node_removed));

		} break;
		case NOTIFICATION_PROCESS: {
			if (!node) {
				return;
			}

			if (_get_depth() != prev_depth) {
				_polygon_draw();
				prev_depth = _get_depth();
			}

		} break;
	}
}

void CollisionPolygon3DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
		if (imgeom->get_parent() == p_node) {
			p_node->remove_child(imgeom);
		}
		hide();
		set_process(false);
	}
}

void CollisionPolygon3DEditor::_menu_option(int p_option) {
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

void CollisionPolygon3DEditor::_wip_close() {
	undo_redo->create_action(TTR("Create Polygon3D"));
	undo_redo->add_undo_method(node, "set_polygon", node->call("get_polygon"));
	undo_redo->add_do_method(node, "set_polygon", wip);
	undo_redo->add_do_method(this, "_polygon_draw");
	undo_redo->add_undo_method(this, "_polygon_draw");
	wip.clear();
	wip_active = false;
	mode = MODE_EDIT;
	button_edit->set_pressed(true);
	button_create->set_pressed(false);
	edited_point = -1;
	undo_redo->commit_action();
}

EditorPlugin::AfterGUIInput CollisionPolygon3DEditor::forward_spatial_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (!node) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}

	Transform3D gt = node->get_global_transform();
	Transform3D gi = gt.affine_inverse();
	float depth = _get_depth() * 0.5;
	Vector3 n = gt.basis.get_axis(2).normalized();
	Plane p(n, gt.origin + n * depth);

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		Vector2 gpoint = mb->get_position();
		Vector3 ray_from = p_camera->project_ray_origin(gpoint);
		Vector3 ray_dir = p_camera->project_ray_normal(gpoint);

		Vector3 spoint;

		if (!p.intersects_ray(ray_from, ray_dir, &spoint)) {
			return EditorPlugin::AFTER_GUI_INPUT_PASS;
		}

		spoint = gi.xform(spoint);

		Vector2 cpoint(spoint.x, spoint.y);

		//DO NOT snap here, it's confusing in 3D for adding points.
		//Let the snap happen when the point is being moved, instead.
		//cpoint = CanvasItemEditor::get_singleton()->snap_point(cpoint);

		Vector<Vector2> poly = node->call("get_polygon");

		//first check if a point is to be added (segment split)
		real_t grab_threshold = EDITOR_GET("editors/polygon_editor/point_grab_radius");

		switch (mode) {
			case MODE_CREATE: {
				if (mb->get_button_index() == MOUSE_BUTTON_LEFT && mb->is_pressed()) {
					if (!wip_active) {
						wip.clear();
						wip.push_back(cpoint);
						wip_active = true;
						edited_point_pos = cpoint;
						snap_ignore = false;
						_polygon_draw();
						edited_point = 1;
						return EditorPlugin::AFTER_GUI_INPUT_STOP;
					} else {
						if (wip.size() > 1 && p_camera->unproject_position(gt.xform(Vector3(wip[0].x, wip[0].y, depth))).distance_to(gpoint) < grab_threshold) {
							//wip closed
							_wip_close();

							return EditorPlugin::AFTER_GUI_INPUT_STOP;
						} else {
							wip.push_back(cpoint);
							edited_point = wip.size();
							snap_ignore = false;
							_polygon_draw();
							return EditorPlugin::AFTER_GUI_INPUT_STOP;
						}
					}
				} else if (mb->get_button_index() == MOUSE_BUTTON_RIGHT && mb->is_pressed() && wip_active) {
					_wip_close();
				}

			} break;

			case MODE_EDIT: {
				if (mb->get_button_index() == MOUSE_BUTTON_LEFT) {
					if (mb->is_pressed()) {
						if (mb->is_ctrl_pressed()) {
							if (poly.size() < 3) {
								undo_redo->create_action(TTR("Edit Poly"));
								undo_redo->add_undo_method(node, "set_polygon", poly);
								poly.push_back(cpoint);
								undo_redo->add_do_method(node, "set_polygon", poly);
								undo_redo->add_do_method(this, "_polygon_draw");
								undo_redo->add_undo_method(this, "_polygon_draw");
								undo_redo->commit_action();
								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}

							//search edges
							int closest_idx = -1;
							Vector2 closest_pos;
							real_t closest_dist = 1e10;
							for (int i = 0; i < poly.size(); i++) {
								Vector2 points[2] = {
									p_camera->unproject_position(gt.xform(Vector3(poly[i].x, poly[i].y, depth))),
									p_camera->unproject_position(gt.xform(Vector3(poly[(i + 1) % poly.size()].x, poly[(i + 1) % poly.size()].y, depth)))
								};

								Vector2 cp = Geometry2D::get_closest_point_to_segment(gpoint, points);
								if (cp.distance_squared_to(points[0]) < CMP_EPSILON2 || cp.distance_squared_to(points[1]) < CMP_EPSILON2) {
									continue; //not valid to reuse point
								}

								real_t d = cp.distance_to(gpoint);
								if (d < closest_dist && d < grab_threshold) {
									closest_dist = d;
									closest_pos = cp;
									closest_idx = i;
								}
							}

							if (closest_idx >= 0) {
								pre_move_edit = poly;
								poly.insert(closest_idx + 1, cpoint);
								edited_point = closest_idx + 1;
								edited_point_pos = cpoint;
								node->call("set_polygon", poly);
								_polygon_draw();
								snap_ignore = true;

								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}
						} else {
							//look for points to move

							int closest_idx = -1;
							Vector2 closest_pos;
							real_t closest_dist = 1e10;
							for (int i = 0; i < poly.size(); i++) {
								Vector2 cp = p_camera->unproject_position(gt.xform(Vector3(poly[i].x, poly[i].y, depth)));

								real_t d = cp.distance_to(gpoint);
								if (d < closest_dist && d < grab_threshold) {
									closest_dist = d;
									closest_pos = cp;
									closest_idx = i;
								}
							}

							if (closest_idx >= 0) {
								pre_move_edit = poly;
								edited_point = closest_idx;
								edited_point_pos = poly[closest_idx];
								_polygon_draw();
								snap_ignore = false;
								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}
						}
					} else {
						snap_ignore = false;

						if (edited_point != -1) {
							//apply

							ERR_FAIL_INDEX_V(edited_point, poly.size(), EditorPlugin::AFTER_GUI_INPUT_PASS);
							poly.write[edited_point] = edited_point_pos;
							undo_redo->create_action(TTR("Edit Poly"));
							undo_redo->add_do_method(node, "set_polygon", poly);
							undo_redo->add_undo_method(node, "set_polygon", pre_move_edit);
							undo_redo->add_do_method(this, "_polygon_draw");
							undo_redo->add_undo_method(this, "_polygon_draw");
							undo_redo->commit_action();

							edited_point = -1;
							return EditorPlugin::AFTER_GUI_INPUT_STOP;
						}
					}
				}
				if (mb->get_button_index() == MOUSE_BUTTON_RIGHT && mb->is_pressed() && edited_point == -1) {
					int closest_idx = -1;
					Vector2 closest_pos;
					real_t closest_dist = 1e10;
					for (int i = 0; i < poly.size(); i++) {
						Vector2 cp = p_camera->unproject_position(gt.xform(Vector3(poly[i].x, poly[i].y, depth)));

						real_t d = cp.distance_to(gpoint);
						if (d < closest_dist && d < grab_threshold) {
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
						undo_redo->add_do_method(this, "_polygon_draw");
						undo_redo->add_undo_method(this, "_polygon_draw");
						undo_redo->commit_action();
						return EditorPlugin::AFTER_GUI_INPUT_STOP;
					}
				}

			} break;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (edited_point != -1 && (wip_active || mm->get_button_mask() & MOUSE_BUTTON_MASK_LEFT)) {
			Vector2 gpoint = mm->get_position();

			Vector3 ray_from = p_camera->project_ray_origin(gpoint);
			Vector3 ray_dir = p_camera->project_ray_normal(gpoint);

			Vector3 spoint;

			if (!p.intersects_ray(ray_from, ray_dir, &spoint)) {
				return EditorPlugin::AFTER_GUI_INPUT_PASS;
			}

			spoint = gi.xform(spoint);

			Vector2 cpoint(spoint.x, spoint.y);

			if (snap_ignore && !Input::get_singleton()->is_key_pressed(KEY_CTRL)) {
				snap_ignore = false;
			}

			if (!snap_ignore && Node3DEditor::get_singleton()->is_snap_enabled()) {
				cpoint = cpoint.snapped(Vector2(
						Node3DEditor::get_singleton()->get_translate_snap(),
						Node3DEditor::get_singleton()->get_translate_snap()));
			}
			edited_point_pos = cpoint;

			_polygon_draw();
		}
	}

	return EditorPlugin::AFTER_GUI_INPUT_PASS;
}

float CollisionPolygon3DEditor::_get_depth() {
	if (bool(node->call("_has_editable_3d_polygon_no_depth"))) {
		return 0;
	}

	return float(node->call("get_depth"));
}

void CollisionPolygon3DEditor::_polygon_draw() {
	if (!node) {
		return;
	}

	Vector<Vector2> poly;

	if (wip_active) {
		poly = wip;
	} else {
		poly = node->call("get_polygon");
	}

	float depth = _get_depth() * 0.5;

	imesh->clear_surfaces();
	imgeom->set_material_override(line_material);
	imesh->surface_begin(Mesh::PRIMITIVE_LINES);

	Rect2 rect;

	for (int i = 0; i < poly.size(); i++) {
		Vector2 p, p2;
		p = i == edited_point ? edited_point_pos : poly[i];
		if ((wip_active && i == poly.size() - 1) || (((i + 1) % poly.size()) == edited_point)) {
			p2 = edited_point_pos;
		} else {
			p2 = poly[(i + 1) % poly.size()];
		}

		if (i == 0) {
			rect.position = p;
		} else {
			rect.expand_to(p);
		}

		Vector3 point = Vector3(p.x, p.y, depth);
		Vector3 next_point = Vector3(p2.x, p2.y, depth);

		imesh->surface_set_color(Color(1, 0.3, 0.1, 0.8));
		imesh->surface_add_vertex(point);
		imesh->surface_set_color(Color(1, 0.3, 0.1, 0.8));
		imesh->surface_add_vertex(next_point);

		//Color col=Color(1,0.3,0.1,0.8);
		//vpc->draw_line(point,next_point,col,2);
		//vpc->draw_texture(handle,point-handle->get_size()*0.5);
	}

	rect = rect.grow(1);

	AABB r;
	r.position.x = rect.position.x;
	r.position.y = rect.position.y;
	r.position.z = depth;
	r.size.x = rect.size.x;
	r.size.y = rect.size.y;
	r.size.z = 0;

	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position);
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(0.3, 0, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position);
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(0.0, 0.3, 0));

	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(r.size.x, 0, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(r.size.x, 0, 0) - Vector3(0.3, 0, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(r.size.x, 0, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(r.size.x, 0, 0) + Vector3(0, 0.3, 0));

	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(0, r.size.y, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(0, r.size.y, 0) - Vector3(0, 0.3, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(0, r.size.y, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + Vector3(0, r.size.y, 0) + Vector3(0.3, 0, 0));

	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + r.size);
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + r.size - Vector3(0.3, 0, 0));
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + r.size);
	imesh->surface_set_color(Color(0.8, 0.8, 0.8, 0.2));
	imesh->surface_add_vertex(r.position + r.size - Vector3(0.0, 0.3, 0));

	imesh->surface_end();

	if (poly.size() == 0) {
		return;
	}

	Array a;
	a.resize(Mesh::ARRAY_MAX);
	Vector<Vector3> va;
	{
		va.resize(poly.size());
		Vector3 *w = va.ptrw();
		for (int i = 0; i < poly.size(); i++) {
			Vector2 p, p2;
			p = i == edited_point ? edited_point_pos : poly[i];

			Vector3 point = Vector3(p.x, p.y, depth);
			w[i] = point;
		}
	}
	a[Mesh::ARRAY_VERTEX] = va;
	m->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, a);
	m->surface_set_material(0, handle_material);
}

void CollisionPolygon3DEditor::edit(Node *p_collision_polygon) {
	if (p_collision_polygon) {
		node = Object::cast_to<Node3D>(p_collision_polygon);
		//Enable the pencil tool if the polygon is empty
		if (Vector<Vector2>(node->call("get_polygon")).size() == 0) {
			_menu_option(MODE_CREATE);
		}
		wip.clear();
		wip_active = false;
		edited_point = -1;
		p_collision_polygon->add_child(imgeom);
		_polygon_draw();
		set_process(true);
		prev_depth = -1;

	} else {
		node = nullptr;

		if (imgeom->get_parent()) {
			imgeom->get_parent()->remove_child(imgeom);
		}

		set_process(false);
	}
}

void CollisionPolygon3DEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_polygon_draw"), &CollisionPolygon3DEditor::_polygon_draw);
}

CollisionPolygon3DEditor::CollisionPolygon3DEditor(EditorNode *p_editor) {
	node = nullptr;
	editor = p_editor;
	undo_redo = EditorNode::get_undo_redo();

	add_child(memnew(VSeparator));
	button_create = memnew(Button);
	button_create->set_flat(true);
	add_child(button_create);
	button_create->connect("pressed", callable_mp(this, &CollisionPolygon3DEditor::_menu_option), varray(MODE_CREATE));
	button_create->set_toggle_mode(true);

	button_edit = memnew(Button);
	button_edit->set_flat(true);
	add_child(button_edit);
	button_edit->connect("pressed", callable_mp(this, &CollisionPolygon3DEditor::_menu_option), varray(MODE_EDIT));
	button_edit->set_toggle_mode(true);

	mode = MODE_EDIT;
	wip_active = false;
	imgeom = memnew(MeshInstance3D);
	imesh.instantiate();
	imgeom->set_mesh(imesh);
	imgeom->set_transform(Transform3D(Basis(), Vector3(0, 0, 0.00001)));

	line_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	line_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_albedo(Color(1, 1, 1));

	handle_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	handle_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	handle_material->set_flag(StandardMaterial3D::FLAG_USE_POINT_SIZE, true);
	handle_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	handle_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	handle_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	Ref<Texture2D> handle = editor->get_gui_base()->get_theme_icon(SNAME("Editor3DHandle"), SNAME("EditorIcons"));
	handle_material->set_point_size(handle->get_width());
	handle_material->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, handle);

	pointsm = memnew(MeshInstance3D);
	imgeom->add_child(pointsm);
	m.instantiate();
	pointsm->set_mesh(m);
	pointsm->set_transform(Transform3D(Basis(), Vector3(0, 0, 0.00001)));

	snap_ignore = false;
}

CollisionPolygon3DEditor::~CollisionPolygon3DEditor() {
	memdelete(imgeom);
}

void Polygon3DEditorPlugin::edit(Object *p_object) {
	collision_polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool Polygon3DEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Node3D>(p_object) && bool(p_object->call("_is_editable_3d_polygon"));
}

void Polygon3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		collision_polygon_editor->show();
	} else {
		collision_polygon_editor->hide();
		collision_polygon_editor->edit(nullptr);
	}
}

Polygon3DEditorPlugin::Polygon3DEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	collision_polygon_editor = memnew(CollisionPolygon3DEditor(p_node));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(collision_polygon_editor);

	collision_polygon_editor->hide();
}

Polygon3DEditorPlugin::~Polygon3DEditorPlugin() {
}
