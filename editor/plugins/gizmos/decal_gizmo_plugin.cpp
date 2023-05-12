/**************************************************************************/
/*  decal_gizmo_plugin.cpp                                                */
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

#include "decal_gizmo_plugin.h"

#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/decal.h"

DecalGizmoPlugin::DecalGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/decal", Color(0.6, 0.5, 1.0));

	create_material("decal_material", gizmo_color);

	create_handle_material("handles");
}

bool DecalGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<Decal>(p_spatial) != nullptr;
}

String DecalGizmoPlugin::get_gizmo_name() const {
	return "Decal";
}

int DecalGizmoPlugin::get_priority() const {
	return -1;
}

String DecalGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	switch (p_id) {
		case 0:
			return "Size X";
		case 1:
			return "Size Y";
		case 2:
			return "Size Z";
	}

	return "";
}

Variant DecalGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());
	return decal->get_size();
}

void DecalGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());
	Transform3D gt = decal->get_global_transform();

	Transform3D gi = gt.affine_inverse();

	Vector3 size = decal->get_size();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

	Vector3 axis;
	axis[p_id] = 1.0;

	Vector3 ra, rb;
	Geometry3D::get_closest_points_between_segments(Vector3(), axis * 16384, sg[0], sg[1], ra, rb);
	float d = ra[p_id] * 2;
	if (Node3DEditor::get_singleton()->is_snap_enabled()) {
		d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
	}

	if (d < 0.001) {
		d = 0.001;
	}

	size[p_id] = d;
	decal->set_size(size);
}

void DecalGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());

	Vector3 restore = p_restore;

	if (p_cancel) {
		decal->set_size(restore);
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Change Decal Size"));
	ur->add_do_method(decal, "set_size", decal->get_size());
	ur->add_undo_method(decal, "set_size", restore);
	ur->commit_action();
}

void DecalGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector3 size = decal->get_size();

	AABB aabb;
	aabb.position = -size / 2;
	aabb.size = size;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		if (a.y == b.y) {
			lines.push_back(a);
			lines.push_back(b);
		} else {
			Vector3 ah = a.lerp(b, 0.2);
			lines.push_back(a);
			lines.push_back(ah);
			Vector3 bh = b.lerp(a, 0.2);
			lines.push_back(b);
			lines.push_back(bh);
		}
	}

	float half_size_y = size.y / 2;
	lines.push_back(Vector3(0, half_size_y, 0));
	lines.push_back(Vector3(0, half_size_y * 1.2, 0));

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	Ref<Material> material = get_material("decal_material", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_handles(handles, get_material("handles"));
}
