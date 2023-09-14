/**************************************************************************/
/*  gizmo_3d_helper.cpp                                                   */
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

#include "gizmo_3d_helper.h"

#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"

void Gizmo3DHelper::initialize_handle_action(const Variant &p_initial_value, const Transform3D &p_initial_transform) {
	initial_value = p_initial_value;
	initial_transform = p_initial_transform;
}

void Gizmo3DHelper::get_segment(Camera3D *p_camera, const Point2 &p_point, Vector3 *r_segment) {
	Transform3D gt = initial_transform;
	Transform3D gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	r_segment[0] = gi.xform(ray_from);
	r_segment[1] = gi.xform(ray_from + ray_dir * 4096);
}

Vector<Vector3> Gizmo3DHelper::box_get_handles(const Vector3 &p_box_size) {
	Vector<Vector3> handles;
	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = p_box_size[i] / 2;
		handles.push_back(ax);
		handles.push_back(-ax);
	}
	return handles;
}

String Gizmo3DHelper::box_get_handle_name(int p_id) const {
	switch (p_id) {
		case 0:
		case 1:
			return "Size X";
		case 2:
		case 3:
			return "Size Y";
		case 4:
		case 5:
			return "Size Z";
	}
	return "";
}

void Gizmo3DHelper::box_set_handle(const Vector3 p_segment[2], int p_id, Vector3 &r_box_size, Vector3 &r_box_position) {
	Vector3 axis;
	axis[p_id / 2] = 1.0;
	Vector3 ra, rb;
	int sign = p_id % 2 * -2 + 1;
	Vector3 initial_size = initial_value;

	Geometry3D::get_closest_points_between_segments(Vector3(), axis * 4096 * sign, p_segment[0], p_segment[1], ra, rb);
	if (ra[p_id / 2] == 0) {
		// Point before half of the shape. Needs to be calculated in opposite direction.
		Geometry3D::get_closest_points_between_segments(Vector3(), axis * 4096 * -sign, p_segment[0], p_segment[1], ra, rb);
	}

	float d = ra[p_id / 2] * sign;

	Vector3 he = r_box_size;
	he[p_id / 2] = d * 2;
	if (Node3DEditor::get_singleton()->is_snap_enabled()) {
		he[p_id / 2] = Math::snapped(he[p_id / 2], Node3DEditor::get_singleton()->get_translate_snap());
	}

	if (Input::get_singleton()->is_key_pressed(Key::ALT)) {
		he[p_id / 2] = MAX(he[p_id / 2], 0.001);
		r_box_size = he;
		r_box_position = initial_transform.get_origin();
	} else {
		he[p_id / 2] = MAX(he[p_id / 2], -initial_size[p_id / 2] + 0.002);
		r_box_size = (initial_size + (he - initial_size) * 0.5).abs();
		Vector3 pos = initial_transform.affine_inverse().xform(initial_transform.get_origin());
		pos += (r_box_size - initial_size) * 0.5 * sign;
		r_box_position = initial_transform.xform(pos);
	}
}

void Gizmo3DHelper::box_commit_handle(const String &p_action_name, bool p_cancel, Object *p_position_object, Object *p_size_object, const StringName &p_position_property, const StringName &p_size_property) {
	if (!p_size_object) {
		p_size_object = p_position_object;
	}

	if (p_cancel) {
		p_size_object->set(p_size_property, initial_value);
		p_position_object->set(p_position_property, initial_transform.get_origin());
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(p_action_name);
	ur->add_do_property(p_size_object, p_size_property, p_size_object->get(p_size_property));
	ur->add_do_property(p_position_object, p_position_property, p_position_object->get(p_position_property));
	ur->add_undo_property(p_size_object, p_size_property, initial_value);
	ur->add_undo_property(p_position_object, p_position_property, initial_transform.get_origin());
	ur->commit_action();
}
