/**************************************************************************/
/*  primitive_geometry_3d_gizmo_plugin.cpp                                */
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

#include "primitive_geometry_3d_gizmo_plugin.h"

#include "core/math/geometry_3d.h"
#include "core/math/triangle_mesh.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/3d/gizmos/gizmo_3d_helper.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "scene/3d/primitive_geometry_3d.h"

PrimitiveGeometry3DGizmoPlugin::PrimitiveGeometry3DGizmoPlugin() {
	helper.instantiate();
	create_handle_material("handles");
}

bool PrimitiveGeometry3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<PrimitiveGeometry3D>(p_spatial) != nullptr;
}

String PrimitiveGeometry3DGizmoPlugin::get_gizmo_name() const {
	return "PrimitiveGeometry3D";
}

int PrimitiveGeometry3DGizmoPlugin::get_priority() const {
	return -1;
}

String PrimitiveGeometry3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	const PrimitiveGeometry3D *cs = Object::cast_to<PrimitiveGeometry3D>(p_gizmo->get_node_3d());


	if (Object::cast_to<Sphere3D>(cs)) {
		return "Radius";
	}

	if (Object::cast_to<Box3D>(cs)) {
		return helper->box_get_handle_name(p_id);
	}

	if (Object::cast_to<Capsule3D>(cs)) {
		return helper->capsule_get_handle_name(p_id);
	}

	if (Object::cast_to<Cylinder3D>(cs)) {
		return helper->cylinder_get_handle_name(p_id);
	}

	return "";
}

Variant PrimitiveGeometry3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	PrimitiveGeometry3D *cs = Object::cast_to<PrimitiveGeometry3D>(p_gizmo->get_node_3d());

	if (Object::cast_to<Sphere3D>(cs)) {
		Sphere3D* ss = (Sphere3D*)cs;
		return ss->get_radius();
	}

	if (Object::cast_to<Box3D>(cs)) {
		Box3D* ss = (Box3D*)cs;
		return ss->get_size();
	}

	if (Object::cast_to<Capsule3D>(cs)) {
		Capsule3D* ss = (Capsule3D*)cs;
		return Vector2(ss->get_radius(), ss->get_height());
	}

	if (Object::cast_to<Cylinder3D>(cs)) {
		Cylinder3D* ss = (Cylinder3D*)cs;
		return Vector2(ss->get_radius(), ss->get_height());
	}

	return Variant();
}

void PrimitiveGeometry3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void PrimitiveGeometry3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	PrimitiveGeometry3D *cs = Object::cast_to<PrimitiveGeometry3D>(p_gizmo->get_node_3d());

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	if (Object::cast_to<Sphere3D>(cs)) {
		Sphere3D* ss = (Sphere3D*)cs;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = ra.x;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		ss->set_radius(d);
	}

	if (Object::cast_to<Box3D>(cs)) {
		Box3D* bs = (Box3D*)cs;
		Vector3 size = bs->get_size();
		Vector3 position;
		helper->box_set_handle(sg, p_id, size, position);
		bs->set_size(size);
		cs->set_global_position(position);
	}

	if (Object::cast_to<Capsule3D>(cs)) {
		Capsule3D* cs2 = (Capsule3D*)cs;

		real_t height = cs2->get_height();
		real_t radius = cs2->get_radius();
		Vector3 position;
		helper->capsule_set_handle(sg, p_id, height, radius, position);
		cs2->set_height(height);
		cs2->set_radius(radius);
		cs->set_global_position(position);
	}

	if (Object::cast_to<Cylinder3D>(cs)) {
		Cylinder3D* cs2 = (Cylinder3D*)cs;

		real_t height = cs2->get_height();
		real_t radius = cs2->get_radius();
		Vector3 position;
		helper->cylinder_set_handle(sg, p_id, height, radius, position);
		cs2->set_height(height);
		cs2->set_radius(radius);
		cs->set_global_position(position);
	}
}

void PrimitiveGeometry3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	PrimitiveGeometry3D *cs = Object::cast_to<PrimitiveGeometry3D>(p_gizmo->get_node_3d());

	// there's no helper->commit_sphere_handle so manual implementation
	if (Object::cast_to<Sphere3D>(cs)) {
		Sphere3D* ss = (Sphere3D*)cs;
		if (p_cancel) {
			ss->set_radius(p_restore);
			return;
		}

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Change Sphere Radius"));
		ur->add_do_method(ss, "set_radius", ss->get_radius());
		ur->add_undo_method(ss, "set_radius", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<Box3D>(cs)) {
		helper->box_commit_handle(TTR("Change Box Size"), p_cancel, cs, cs);
	}

	if (Object::cast_to<Capsule3D>(cs)) {
		helper->cylinder_commit_handle(p_id, TTR("Change Capsule Radius"), TTR("Change Capsule Height"), p_cancel, cs, cs, cs);
	}

	if (Object::cast_to<Cylinder3D>(cs)) {
		helper->cylinder_commit_handle(p_id, TTR("Change Cylinder Radius"), TTR("Change Cylinder Height"), p_cancel, cs, cs, cs);
	}
}

void PrimitiveGeometry3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	PrimitiveGeometry3D *cs = Object::cast_to<PrimitiveGeometry3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	const Ref<Material> handles_material = get_material("handles");

	if (Object::cast_to<Sphere3D>(cs)) {
		Sphere3D* sp = (Sphere3D*)cs;

		Vector<Vector3> handles;
		handles.push_back(Vector3(sp->get_radius(), 0.f, 0.f));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<Box3D>(cs)) {
		Box3D* bs = (Box3D*)cs;

		const Vector<Vector3> handles = helper->box_get_handles(bs->get_size());
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<Capsule3D>(cs)) {
		Capsule3D* cs2 = (Capsule3D*)cs;

		Vector<Vector3> handles = helper->capsule_get_handles(cs2->get_height(), cs2->get_radius());
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<Cylinder3D>(cs)) {
		Cylinder3D* cs2 = (Cylinder3D*)cs;

		Vector<Vector3> handles = helper->cylinder_get_handles(cs2->get_height(), cs2->get_radius());
		p_gizmo->add_handles(handles, handles_material);
	}

	Ref<TriangleMesh> triangle_mesh = cs->generate_triangle_mesh();
	if (triangle_mesh.is_valid()) {
		p_gizmo->add_collision_triangles(triangle_mesh);
	}
}

void PrimitiveGeometry3DGizmoPlugin::set_show_only_when_selected(bool p_enabled) {
	show_only_when_selected = p_enabled;
}
