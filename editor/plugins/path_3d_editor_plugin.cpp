/**************************************************************************/
/*  path_3d_editor_plugin.cpp                                             */
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

#include "path_3d_editor_plugin.h"

#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "node_3d_editor_plugin.h"
#include "scene/gui/menu_button.h"
#include "scene/resources/curve.h"

String Path3DGizmo::get_handle_name(int p_id, bool p_secondary) const {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return "";
	}

	// Primary handles: position.
	if (!p_secondary) {
		return TTR("Curve Point #") + itos(p_id);
	}

	// Secondary handles: in, out, tilt.
	const HandleInfo info = _secondary_handles_info[p_id];
	switch (info.type) {
		case HandleType::HANDLE_TYPE_IN:
			return TTR("Handle In #") + itos(info.point_idx);
		case HandleType::HANDLE_TYPE_OUT:
			return TTR("Handle Out #") + itos(info.point_idx);
		case HandleType::HANDLE_TYPE_TILT:
			return TTR("Handle Tilt #") + itos(info.point_idx);
	}

	return "";
}

Variant Path3DGizmo::get_handle_value(int p_id, bool p_secondary) const {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return Variant();
	}

	// Primary handles: position.
	if (!p_secondary) {
		original = c->get_point_position(p_id);
		return original;
	}

	// Secondary handles: in, out, tilt.
	const HandleInfo info = _secondary_handles_info[p_id];
	Vector3 ofs;
	switch (info.type) {
		case HandleType::HANDLE_TYPE_TILT:
			return c->get_point_tilt(info.point_idx);
		case HandleType::HANDLE_TYPE_IN:
			ofs = c->get_point_in(info.point_idx);
			break;
		case HandleType::HANDLE_TYPE_OUT:
			ofs = c->get_point_out(info.point_idx);
			break;
	}

	original = ofs + c->get_point_position(info.point_idx);
	return ofs;
}

void Path3DGizmo::set_handle(int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	const Transform3D gt = path->get_global_transform();
	const Transform3D gi = gt.affine_inverse();
	const Vector3 ray_from = p_camera->project_ray_origin(p_point);
	const Vector3 ray_dir = p_camera->project_ray_normal(p_point);
	const Plane p = Plane(p_camera->get_transform().basis.get_column(2), gt.xform(original));

	// Primary handles: position.
	if (!p_secondary) {
		Vector3 inters;
		// Special cas for primary handle, the handle id equals control point id.
		const int idx = p_id;
		if (p.intersects_ray(ray_from, ray_dir, &inters)) {
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				float snap = Node3DEditor::get_singleton()->get_translate_snap();
				inters.snap(Vector3(snap, snap, snap));
			}

			Vector3 local = gi.xform(inters);
			c->set_point_position(idx, local);
		}

		return;
	}

	// Secondary handles: in, out, tilt.
	const HandleInfo info = _secondary_handles_info[p_id];
	switch (info.type) {
		case HandleType::HANDLE_TYPE_OUT:
		case HandleType::HANDLE_TYPE_IN: {
			const int idx = info.point_idx;
			const Vector3 base = c->get_point_position(idx);

			Vector3 inters;
			if (p.intersects_ray(ray_from, ray_dir, &inters)) {
				if (!Path3DEditorPlugin::singleton->is_handle_clicked()) {
					orig_in_length = c->get_point_in(idx).length();
					orig_out_length = c->get_point_out(idx).length();
					Path3DEditorPlugin::singleton->set_handle_clicked(true);
				}

				Vector3 local = gi.xform(inters) - base;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					float snap = Node3DEditor::get_singleton()->get_translate_snap();
					local.snap(Vector3(snap, snap, snap));
				}

				if (info.type == HandleType::HANDLE_TYPE_IN) {
					c->set_point_in(idx, local);
					if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
						c->set_point_out(idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -local : (-local.normalized() * orig_out_length));
					}
				} else {
					c->set_point_out(idx, local);
					if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
						c->set_point_in(idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -local : (-local.normalized() * orig_in_length));
					}
				}
			}
			break;
		}
		case HandleType::HANDLE_TYPE_TILT: {
			const int idx = info.point_idx;
			const Vector3 position = c->get_point_position(idx);
			const Basis posture = c->get_point_baked_posture(idx);
			const Vector3 tangent = -posture.get_column(2);
			const Vector3 up = posture.get_column(1);
			const Plane p_tilt = Plane(tangent, position);

			Vector3 intersection;

			if (p_tilt.intersects_ray(ray_from, ray_dir, &intersection)) {
				Vector3 direction = intersection - position;
				direction.normalize(); // FIXME: redundant?
				real_t tilt_angle = up.signed_angle_to(direction, tangent);

				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					real_t snap = Node3DEditor::get_singleton()->get_rotate_snap();

					tilt_angle = Math::rad_to_deg(tilt_angle) + snap * 0.5; // Else it won't reach +180.
					tilt_angle -= Math::fmod(tilt_angle, snap);
					tilt_angle = Math::deg_to_rad(tilt_angle);
				}

				c->set_point_tilt(idx, tilt_angle);
			}
			break;
		}
	}
}

void Path3DGizmo::commit_handle(int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();

	// Primary handles: position.
	if (!p_secondary) {
		// Special cas for primary handle, the handle id equals control point id.
		const int idx = p_id;
		if (p_cancel) {
			c->set_point_position(idx, p_restore);
			return;
		}
		ur->create_action(TTR("Set Curve Point Position"));
		ur->add_do_method(c.ptr(), "set_point_position", idx, c->get_point_position(idx));
		ur->add_undo_method(c.ptr(), "set_point_position", idx, p_restore);
		ur->commit_action();

		return;
	}

	// Secondary handles: in, out, tilt.
	const HandleInfo info = _secondary_handles_info[p_id];
	const int idx = info.point_idx;
	switch (info.type) {
		case HandleType::HANDLE_TYPE_OUT: {
			if (p_cancel) {
				c->set_point_out(idx, p_restore);

				return;
			}

			ur->create_action(TTR("Set Curve Out Position"));
			ur->add_do_method(c.ptr(), "set_point_out", idx, c->get_point_out(idx));
			ur->add_undo_method(c.ptr(), "set_point_out", idx, p_restore);

			if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
				ur->add_do_method(c.ptr(), "set_point_in", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -c->get_point_out(idx) : (-c->get_point_out(idx).normalized() * orig_in_length));
				ur->add_undo_method(c.ptr(), "set_point_in", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -static_cast<Vector3>(p_restore) : (-static_cast<Vector3>(p_restore).normalized() * orig_in_length));
			}
			ur->commit_action();
			break;
		}
		case HandleType::HANDLE_TYPE_IN: {
			if (p_cancel) {
				c->set_point_in(idx, p_restore);
				return;
			}

			ur->create_action(TTR("Set Curve In Position"));
			ur->add_do_method(c.ptr(), "set_point_in", idx, c->get_point_in(idx));
			ur->add_undo_method(c.ptr(), "set_point_in", idx, p_restore);

			if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
				ur->add_do_method(c.ptr(), "set_point_out", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -c->get_point_in(idx) : (-c->get_point_in(idx).normalized() * orig_out_length));
				ur->add_undo_method(c.ptr(), "set_point_out", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -static_cast<Vector3>(p_restore) : (-static_cast<Vector3>(p_restore).normalized() * orig_out_length));
			}
			ur->commit_action();
			break;
		}
		case HandleType::HANDLE_TYPE_TILT: {
			if (p_cancel) {
				c->set_point_tilt(idx, p_restore);
				return;
			}
			ur->create_action(TTR("Set Curve Point Tilt"));
			ur->add_do_method(c.ptr(), "set_point_tilt", idx, c->get_point_tilt(idx));
			ur->add_undo_method(c.ptr(), "set_point_tilt", idx, p_restore);
			ur->commit_action();
			break;
		}
	}
}

void Path3DGizmo::redraw() {
	clear();

	Ref<StandardMaterial3D> path_material = gizmo_plugin->get_material("path_material", this);
	Ref<StandardMaterial3D> path_thin_material = gizmo_plugin->get_material("path_thin_material", this);
	Ref<StandardMaterial3D> path_tilt_material = gizmo_plugin->get_material("path_tilt_material", this);
	Ref<StandardMaterial3D> handles_material = gizmo_plugin->get_material("handles");
	Ref<StandardMaterial3D> sec_handles_material = gizmo_plugin->get_material("sec_handles");

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	real_t interval = 0.1;
	const real_t length = c->get_baked_length();

	// 1. Draw curve and bones.
	if (length > CMP_EPSILON) {
		const int sample_count = int(length / interval) + 2;
		interval = length / (sample_count - 1); // Recalculate real interval length.

		Vector<Transform3D> frames;
		frames.resize(sample_count);

		{
			Transform3D *w = frames.ptrw();

			for (int i = 0; i < sample_count; i++) {
				w[i] = c->sample_baked_with_rotation(i * interval, true, true);
			}
		}

		const Transform3D *r = frames.ptr();
		Vector<Vector3> _collision_segments;
		Vector<Vector3> bones;
		Vector<Vector3> ribbon;
		for (int i = 0; i < sample_count; i++) {
			const Vector3 p1 = r[i].origin;
			const Vector3 side = r[i].basis.get_column(0);
			const Vector3 up = r[i].basis.get_column(1);
			const Vector3 forward = r[i].basis.get_column(2);

			// Collision segments.
			if (i != sample_count) {
				const Vector3 p2 = r[i + 1].origin;
				_collision_segments.push_back(p1);
				_collision_segments.push_back(p2);
			}

			// Path3D as a ribbon.
			ribbon.push_back(p1);

			// Fish Bone.
			const Vector3 p_left = p1 + (side + forward - up * 0.3) * 0.06;
			const Vector3 p_right = p1 + (-side + forward - up * 0.3) * 0.06;
			bones.push_back(p1);
			bones.push_back(p_left);

			bones.push_back(p1);
			bones.push_back(p_right);
		}

		add_collision_segments(_collision_segments);
		add_lines(bones, path_material);
		add_vertices(ribbon, path_material, Mesh::PRIMITIVE_LINE_STRIP);
	}

	// 2. Draw handles when selected.
	if (Path3DEditorPlugin::singleton->get_edited_path() == path) {
		PackedVector3Array handle_lines;
		PackedVector3Array tilt_handle_lines;
		PackedVector3Array primary_handle_points;
		PackedVector3Array secondary_handle_points;
		PackedInt32Array collected_secondary_handle_ids; // Avoid shadowing member on Node3DEditorGizmo.

		_secondary_handles_info.resize(c->get_point_count() * 3);

		for (int idx = 0; idx < c->get_point_count(); idx++) {
			// Collect primary-handles.
			const Vector3 pos = c->get_point_position(idx);
			primary_handle_points.append(pos);

			HandleInfo info;
			info.point_idx = idx;

			// Collect in-handles except for the first point.
			if (idx > 0) {
				info.type = HandleType::HANDLE_TYPE_IN;
				const int handle_idx = idx * 3 + 0;
				collected_secondary_handle_ids.append(handle_idx);
				_secondary_handles_info.write[handle_idx] = info;

				const Vector3 in = c->get_point_in(idx);
				secondary_handle_points.append(pos + in);
				handle_lines.append(pos);
				handle_lines.append(pos + in);
			}

			// Collect out-handles except for the last point.
			if (idx < c->get_point_count() - 1) {
				info.type = HandleType::HANDLE_TYPE_OUT;
				const int handle_idx = idx * 3 + 1;
				collected_secondary_handle_ids.append(handle_idx);
				_secondary_handles_info.write[handle_idx] = info;

				const Vector3 out = c->get_point_out(idx);
				secondary_handle_points.append(pos + out);
				handle_lines.append(pos);
				handle_lines.append(pos + out);
			}

			// Collect tilt-handles.
			{
				{
					info.type = HandleType::HANDLE_TYPE_TILT;
					const int handle_idx = idx * 3 + 2;
					collected_secondary_handle_ids.append(handle_idx);
					_secondary_handles_info.write[handle_idx] = info;

					const Basis posture = c->get_point_baked_posture(idx, true);
					const Vector3 up = posture.get_column(1);
					secondary_handle_points.append(pos + up * disk_size);
					tilt_handle_lines.append(pos);
					tilt_handle_lines.append(pos + up * disk_size);
				}

				// Tilt disk.
				{
					const Basis posture = c->get_point_baked_posture(idx, false);
					const Vector3 up = posture.get_column(1);
					const Vector3 side = posture.get_column(0);

					PackedVector3Array disk;
					disk.append(pos);

					const int n = 36;
					for (int i = 0; i <= n; i++) {
						const float a = Math_TAU * i / n;
						const Vector3 edge = sin(a) * side + cos(a) * up;
						disk.append(pos + edge * disk_size);
					}
					add_vertices(disk, path_tilt_material, Mesh::PRIMITIVE_LINE_STRIP);
				}
			}
		}

		if (handle_lines.size() > 1) {
			add_lines(handle_lines, path_thin_material);
		}

		if (tilt_handle_lines.size() > 1) {
			add_lines(tilt_handle_lines, path_tilt_material);
		}

		if (primary_handle_points.size()) {
			add_handles(primary_handle_points, handles_material);
		}
		if (secondary_handle_points.size()) {
			add_handles(secondary_handle_points, sec_handles_material, collected_secondary_handle_ids, false, true);
		}
	}
}

Path3DGizmo::Path3DGizmo(Path3D *p_path, float p_disk_size) {
	path = p_path;
	disk_size = p_disk_size;
	set_node_3d(p_path);
	orig_in_length = 0;
	orig_out_length = 0;
}

EditorPlugin::AfterGUIInput Path3DEditorPlugin::forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (!path) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}
	Transform3D gt = path->get_global_transform();
	Transform3D it = gt.affine_inverse();

	static const int click_dist = 10; //should make global

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		Point2 mbpos(mb->get_position().x, mb->get_position().y);

		if (!mb->is_pressed()) {
			set_handle_clicked(false);
		}

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT && (curve_create->is_pressed() || (curve_edit->is_pressed() && mb->is_command_or_control_pressed()))) {
			//click into curve, break it down
			Vector<Vector3> v3a = c->tessellate();
			int rc = v3a.size();
			int closest_seg = -1;
			Vector3 closest_seg_point;

			if (rc >= 2) {
				int idx = 0;
				const Vector3 *r = v3a.ptr();
				float closest_d = 1e20;

				if (p_camera->unproject_position(gt.xform(c->get_point_position(0))).distance_to(mbpos) < click_dist) {
					return EditorPlugin::AFTER_GUI_INPUT_PASS; //nope, existing
				}

				for (int i = 0; i < c->get_point_count() - 1; i++) {
					//find the offset and point index of the place to break up
					int j = idx;
					if (p_camera->unproject_position(gt.xform(c->get_point_position(i + 1))).distance_to(mbpos) < click_dist) {
						return EditorPlugin::AFTER_GUI_INPUT_PASS; //nope, existing
					}

					while (j < rc && c->get_point_position(i + 1) != r[j]) {
						Vector3 from = r[j];
						Vector3 to = r[j + 1];
						real_t cdist = from.distance_to(to);
						from = gt.xform(from);
						to = gt.xform(to);
						if (cdist > 0) {
							Vector2 s[2];
							s[0] = p_camera->unproject_position(from);
							s[1] = p_camera->unproject_position(to);
							Vector2 inters = Geometry2D::get_closest_point_to_segment(mbpos, s);
							float d = inters.distance_to(mbpos);

							if (d < 10 && d < closest_d) {
								closest_d = d;
								closest_seg = i;
								Vector3 ray_from = p_camera->project_ray_origin(mbpos);
								Vector3 ray_dir = p_camera->project_ray_normal(mbpos);

								Vector3 ra, rb;
								Geometry3D::get_closest_points_between_segments(ray_from, ray_from + ray_dir * 4096, from, to, ra, rb);

								closest_seg_point = it.xform(rb);
							}
						}
						j++;
					}
					if (idx == j) {
						idx++; //force next
					} else {
						idx = j; //swap
					}

					if (j == rc) {
						break;
					}
				}
			}

			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			if (closest_seg != -1) {
				//subdivide

				ur->create_action(TTR("Split Path"));
				ur->add_do_method(c.ptr(), "add_point", closest_seg_point, Vector3(), Vector3(), closest_seg + 1);
				ur->add_undo_method(c.ptr(), "remove_point", closest_seg + 1);
				ur->commit_action();
				return EditorPlugin::AFTER_GUI_INPUT_STOP;

			} else {
				Vector3 origin;
				if (c->get_point_count() == 0) {
					origin = path->get_transform().get_origin();
				} else {
					origin = gt.xform(c->get_point_position(c->get_point_count() - 1));
				}
				Plane p(p_camera->get_transform().basis.get_column(2), origin);
				Vector3 ray_from = p_camera->project_ray_origin(mbpos);
				Vector3 ray_dir = p_camera->project_ray_normal(mbpos);

				Vector3 inters;
				if (p.intersects_ray(ray_from, ray_dir, &inters)) {
					ur->create_action(TTR("Add Point to Curve"));
					ur->add_do_method(c.ptr(), "add_point", it.xform(inters), Vector3(), Vector3(), -1);
					ur->add_undo_method(c.ptr(), "remove_point", c->get_point_count());
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}

				//add new at pos
			}

		} else if (mb->is_pressed() && ((mb->get_button_index() == MouseButton::LEFT && curve_del->is_pressed()) || (mb->get_button_index() == MouseButton::RIGHT && curve_edit->is_pressed()))) {
			for (int i = 0; i < c->get_point_count(); i++) {
				real_t dist_to_p = p_camera->unproject_position(gt.xform(c->get_point_position(i))).distance_to(mbpos);
				real_t dist_to_p_out = p_camera->unproject_position(gt.xform(c->get_point_position(i) + c->get_point_out(i))).distance_to(mbpos);
				real_t dist_to_p_in = p_camera->unproject_position(gt.xform(c->get_point_position(i) + c->get_point_in(i))).distance_to(mbpos);
				real_t dist_to_p_up = p_camera->unproject_position(gt.xform(c->get_point_position(i) + c->get_point_baked_posture(i, true).get_column(1) * disk_size)).distance_to(mbpos);

				// Find the offset and point index of the place to break up.
				// Also check for the control points.
				if (dist_to_p < click_dist) {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Remove Path Point"));
					ur->add_do_method(c.ptr(), "remove_point", i);
					ur->add_undo_method(c.ptr(), "add_point", c->get_point_position(i), c->get_point_in(i), c->get_point_out(i), i);
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (dist_to_p_out < click_dist) {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Reset Out-Control Point"));
					ur->add_do_method(c.ptr(), "set_point_out", i, Vector3());
					ur->add_undo_method(c.ptr(), "set_point_out", i, c->get_point_out(i));
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (dist_to_p_in < click_dist) {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Reset In-Control Point"));
					ur->add_do_method(c.ptr(), "set_point_in", i, Vector3());
					ur->add_undo_method(c.ptr(), "set_point_in", i, c->get_point_in(i));
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (dist_to_p_up < click_dist) {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Reset Point Tilt"));
					ur->add_do_method(c.ptr(), "set_point_tilt", i, 0.0f);
					ur->add_undo_method(c.ptr(), "set_point_tilt", i, c->get_point_tilt(i));
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}
			}
		}
		if (curve_edit_curve->is_pressed()) {
			mb->set_shift_pressed(true);
		}
	}

	return EditorPlugin::AFTER_GUI_INPUT_PASS;
}

void Path3DEditorPlugin::edit(Object *p_object) {
	if (p_object) {
		path = Object::cast_to<Path3D>(p_object);
		if (path) {
			if (path->get_curve().is_valid()) {
				path->get_curve()->emit_signal(SNAME("changed"));
			}
		}
	} else {
		Path3D *pre = path;
		path = nullptr;
		if (pre) {
			pre->get_curve()->emit_signal(SNAME("changed"));
		}
	}

	update_overlays();
	//collision_polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool Path3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Path3D");
}

void Path3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		topmenu_bar->show();
	} else {
		topmenu_bar->hide();

		{
			Path3D *pre = path;
			path = nullptr;
			if (pre && pre->get_curve().is_valid()) {
				pre->get_curve()->emit_signal(SNAME("changed"));
			}
		}
	}
}

void Path3DEditorPlugin::_mode_changed(int p_mode) {
	curve_create->set_pressed(p_mode == MODE_CREATE);
	curve_edit_curve->set_pressed(p_mode == MODE_EDIT_CURVE);
	curve_edit->set_pressed(p_mode == MODE_EDIT);
	curve_del->set_pressed(p_mode == MODE_DELETE);
}

void Path3DEditorPlugin::_close_curve() {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}
	if (c->get_point_count() < 2) {
		return;
	}
	if (c->get_point_position(0) == c->get_point_position(c->get_point_count() - 1)) {
		return;
	}
	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Close Curve"));
	ur->add_do_method(c.ptr(), "add_point", c->get_point_position(0), c->get_point_in(0), c->get_point_out(0), -1);
	ur->add_undo_method(c.ptr(), "remove_point", c->get_point_count());
	ur->commit_action();
}

void Path3DEditorPlugin::_handle_option_pressed(int p_option) {
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

void Path3DEditorPlugin::_update_theme() {
	// TODO: Split the EditorPlugin instance from the UI instance and connect this properly.
	// See the 2D path editor for inspiration.
	curve_edit->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("CurveEdit"), EditorStringName(EditorIcons)));
	curve_edit_curve->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("CurveCurve"), EditorStringName(EditorIcons)));
	curve_create->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("CurveCreate"), EditorStringName(EditorIcons)));
	curve_del->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("CurveDelete"), EditorStringName(EditorIcons)));
	curve_close->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("CurveClose"), EditorStringName(EditorIcons)));
}

void Path3DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			curve_create->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed).bind(MODE_CREATE));
			curve_edit_curve->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed).bind(MODE_EDIT_CURVE));
			curve_edit->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed).bind(MODE_EDIT));
			curve_del->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed).bind(MODE_DELETE));
			curve_close->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_close_curve));

			_update_theme();
		} break;

		case NOTIFICATION_READY: {
			Node3DEditor::get_singleton()->connect("theme_changed", callable_mp(this, &Path3DEditorPlugin::_update_theme));
		} break;
	}
}

void Path3DEditorPlugin::_bind_methods() {
}

Path3DEditorPlugin *Path3DEditorPlugin::singleton = nullptr;

Path3DEditorPlugin::Path3DEditorPlugin() {
	path = nullptr;
	singleton = this;
	mirror_handle_angle = true;
	mirror_handle_length = true;

	disk_size = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_settings/path3d_tilt_disk_size", 0.8);

	Ref<Path3DGizmoPlugin> gizmo_plugin = memnew(Path3DGizmoPlugin(disk_size));
	Node3DEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);

	topmenu_bar = memnew(HBoxContainer);
	topmenu_bar->hide();
	Node3DEditor::get_singleton()->add_control_to_menu_panel(topmenu_bar);

	curve_edit = memnew(Button);
	curve_edit->set_theme_type_variation("FlatButton");
	curve_edit->set_toggle_mode(true);
	curve_edit->set_focus_mode(Control::FOCUS_NONE);
	curve_edit->set_tooltip_text(TTR("Select Points") + "\n" + TTR("Shift+Drag: Select Control Points") + "\n" + keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL) + TTR("Click: Add Point") + "\n" + TTR("Right Click: Delete Point"));
	topmenu_bar->add_child(curve_edit);

	curve_edit_curve = memnew(Button);
	curve_edit_curve->set_theme_type_variation("FlatButton");
	curve_edit_curve->set_toggle_mode(true);
	curve_edit_curve->set_focus_mode(Control::FOCUS_NONE);
	curve_edit_curve->set_tooltip_text(TTR("Select Control Points (Shift+Drag)"));
	topmenu_bar->add_child(curve_edit_curve);

	curve_create = memnew(Button);
	curve_create->set_theme_type_variation("FlatButton");
	curve_create->set_toggle_mode(true);
	curve_create->set_focus_mode(Control::FOCUS_NONE);
	curve_create->set_tooltip_text(TTR("Add Point (in empty space)") + "\n" + TTR("Split Segment (in curve)"));
	topmenu_bar->add_child(curve_create);

	curve_del = memnew(Button);
	curve_del->set_theme_type_variation("FlatButton");
	curve_del->set_toggle_mode(true);
	curve_del->set_focus_mode(Control::FOCUS_NONE);
	curve_del->set_tooltip_text(TTR("Delete Point"));
	topmenu_bar->add_child(curve_del);

	curve_close = memnew(Button);
	curve_close->set_theme_type_variation("FlatButton");
	curve_close->set_focus_mode(Control::FOCUS_NONE);
	curve_close->set_tooltip_text(TTR("Close Curve"));
	topmenu_bar->add_child(curve_close);

	PopupMenu *menu;

	handle_menu = memnew(MenuButton);
	handle_menu->set_flat(false);
	handle_menu->set_theme_type_variation("FlatMenuButton");
	handle_menu->set_text(TTR("Options"));
	topmenu_bar->add_child(handle_menu);

	menu = handle_menu->get_popup();
	menu->add_check_item(TTR("Mirror Handle Angles"));
	menu->set_item_checked(HANDLE_OPTION_ANGLE, mirror_handle_angle);
	menu->add_check_item(TTR("Mirror Handle Lengths"));
	menu->set_item_checked(HANDLE_OPTION_LENGTH, mirror_handle_length);
	menu->connect("id_pressed", callable_mp(this, &Path3DEditorPlugin::_handle_option_pressed));

	curve_edit->set_pressed(true);
}

Path3DEditorPlugin::~Path3DEditorPlugin() {
}

Ref<EditorNode3DGizmo> Path3DGizmoPlugin::create_gizmo(Node3D *p_spatial) {
	Ref<Path3DGizmo> ref;

	Path3D *path = Object::cast_to<Path3D>(p_spatial);
	if (path) {
		ref = Ref<Path3DGizmo>(memnew(Path3DGizmo(path, disk_size)));
	}

	return ref;
}

String Path3DGizmoPlugin::get_gizmo_name() const {
	return "Path3D";
}

int Path3DGizmoPlugin::get_priority() const {
	return -1;
}

Path3DGizmoPlugin::Path3DGizmoPlugin(float p_disk_size) {
	Color path_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/path", Color(0.5, 0.5, 1.0, 0.9));
	Color path_tilt_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/path_tilt", Color(1.0, 1.0, 0.4, 0.9));
	disk_size = p_disk_size;

	create_material("path_material", path_color);
	create_material("path_thin_material", Color(0.6, 0.6, 0.6));
	create_material("path_tilt_material", path_tilt_color);
	create_handle_material("handles", false, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("EditorPathSmoothHandle"), EditorStringName(EditorIcons)));
	create_handle_material("sec_handles", false, EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("EditorCurveHandle"), EditorStringName(EditorIcons)));
}
