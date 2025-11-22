/**************************************************************************/
/*  visible_on_screen_notifier_3d_gizmo_plugin.cpp                        */
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

#include "visible_on_screen_notifier_3d_gizmo_plugin.h"

#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/3d/visible_on_screen_notifier_3d.h"

VisibleOnScreenNotifier3DGizmoPlugin::VisibleOnScreenNotifier3DGizmoPlugin() {
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/visibility_notifier");
	create_material("visibility_notifier_material", gizmo_color);
	gizmo_color.a = 0.1;
	create_material("visibility_notifier_solid_material", gizmo_color);
	create_handle_material("handles");
}

bool VisibleOnScreenNotifier3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<VisibleOnScreenNotifier3D>(p_spatial) != nullptr;
}

String VisibleOnScreenNotifier3DGizmoPlugin::get_gizmo_name() const {
	return "VisibleOnScreenNotifier3D";
}

int VisibleOnScreenNotifier3DGizmoPlugin::get_priority() const {
	return -1;
}

String VisibleOnScreenNotifier3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	switch (p_id) {
		case 0:
			return "+X Face";
		case 1:
			return "-X Face";
		case 2:
			return "+Y Face";
		case 3:
			return "-Y Face";
		case 4:
			return "+Z Face";
		case 5:
			return "-Z Face";
	}

	return "";
}

Variant VisibleOnScreenNotifier3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());
	return notifier->get_aabb();
}

void VisibleOnScreenNotifier3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());

	Transform3D gt = notifier->get_global_transform();

	Transform3D gi = gt.affine_inverse();

	// Determine the face 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
	int axis = p_id / 2;
	bool positive = (p_id % 2 ) == 0;

	AABB aabb = notifier->get_aabb();

	Vector3 ray_from = gi.xform(p_camera->project_ray_origin(p_point));
	Vector3 ray_to = gi.xform(p_camera->project_ray_origin(p_point) + p_camera->project_ray_normal(p_point) * 4096);

	Vector3 handle_pos = aabb.get_center();
	handle_pos[axis] += (positive ? 1.0f : -1.0f) * (aabb.size[axis] * 0.5f);

	Vector3 axis_vec;
	axis_vec[axis] = 1.0f;

	Vector3 ra, rb;
	Geometry3D::get_closest_points_between_segments(handle_pos - axis_vec * 4096, handle_pos + axis_vec * 4096, ray_from, ray_to, ra, rb);

	float move_to = ra[axis];
	if (Node3DEditor::get_singleton()->is_snap_enabled()){
		move_to = Math::snapped(move_to, Node3DEditor::get_singleton()->get_translate_snap());
	}

	if (positive) {
		float new_size = move_to - aabb.position[axis];
		aabb.size[axis] = MAX(new_size, 0.001f);
	} else {
		float new_pos = move_to;
		float old_end = aabb.position[axis] + aabb.size[axis];
		aabb.position[axis] = new_pos;
		aabb.size[axis] = MAX(old_end - new_pos, 0.001f);
	}

	notifier->set_aabb(aabb);
}

void VisibleOnScreenNotifier3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());

	if (p_cancel) {
		notifier->set_aabb(p_restore);
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Change Notifier AABB"));
	ur->add_do_method(notifier, "set_aabb", notifier->get_aabb());
	ur->add_undo_method(notifier, "set_aabb", p_restore);
	ur->commit_action();
}

void VisibleOnScreenNotifier3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	AABB aabb = notifier->get_aabb();
	Vector3 center = aabb.get_center();
	Vector3 extents = aabb.get_size() * 0.5;

	Vector<Vector3> lines;
	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	Vector<Vector3> handles = {
		center + Vector3(extents.x, 0, 0),
		center - Vector3(extents.x, 0, 0),
		center + Vector3(0, extents.y, 0),
		center - Vector3(0, extents.y, 0),
		center + Vector3(0, 0, extents.z),
		center - Vector3(0, 0, extents.z)
	};

	Ref<Material> material = get_material("visibility_notifier_material", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("visibility_notifier_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, aabb.get_size(), aabb.get_center());
	}

	p_gizmo->add_handles(handles, get_material("handles"));
}
