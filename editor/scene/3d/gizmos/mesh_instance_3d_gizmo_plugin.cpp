/**************************************************************************/
/*  mesh_instance_3d_gizmo_plugin.cpp                                     */
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

#include "mesh_instance_3d_gizmo_plugin.h"

#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics/soft_body_3d.h"
#include "scene/resources/3d/primitive_meshes.h"

bool MeshInstance3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<MeshInstance3D>(p_spatial) != nullptr && Object::cast_to<SoftBody3D>(p_spatial) == nullptr;
}

String MeshInstance3DGizmoPlugin::get_gizmo_name() const {
	return "MeshInstance3D";
}

int MeshInstance3DGizmoPlugin::get_priority() const {
	return -1;
}

bool MeshInstance3DGizmoPlugin::can_be_hidden() const {
	return false;
}

String MeshInstance3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_gizmo->get_node_3d());

	if (Object::cast_to<CapsuleMesh>(*mesh->get_mesh())) {
		return helper->capsule_get_handle_name(p_id);
	}

	if (Object::cast_to<CylinderMesh>(*mesh->get_mesh())) {
		return helper->cone_frustum_get_handle_name(p_id);
	}

	if (Object::cast_to<BoxMesh>(*mesh->get_mesh())) {
		return helper->box_get_handle_name(p_id);
	}

	return "";
}

Variant MeshInstance3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_gizmo->get_node_3d());

	const Ref<CapsuleMesh> capsule_mesh = mesh->get_mesh();
	if (capsule_mesh.is_valid()) {
		return Vector2(capsule_mesh->get_radius(), capsule_mesh->get_height());
	}

	const Ref<CylinderMesh> cylinder_mesh = mesh->get_mesh();
	if (cylinder_mesh.is_valid()) {
		return Vector3(cylinder_mesh->get_top_radius(), cylinder_mesh->get_bottom_radius(), cylinder_mesh->get_height());
	}

	const Ref<BoxMesh> box_mesh = mesh->get_mesh();
	if (box_mesh.is_valid()) {
		return box_mesh->get_size();
	}

	return Variant();
}

void MeshInstance3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void MeshInstance3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_gizmo->get_node_3d());

	Vector3 segment[2];
	helper->get_segment(p_camera, p_point, segment);

	const Ref<CapsuleMesh> capsule_mesh = mesh->get_mesh();
	if (capsule_mesh.is_valid()) {
		real_t height, radius;
		Vector3 position;
		helper->capsule_set_handle(segment, p_id, height, radius, position);
		capsule_mesh->set_height(height);
		capsule_mesh->set_radius(radius);
		mesh->set_global_position(position);
	}

	const Ref<CylinderMesh> cylinder_mesh = mesh->get_mesh();
	if (cylinder_mesh.is_valid()) {
		real_t height, radius_top, radius_bottom;
		Vector3 position;
		helper->cone_frustum_set_handle(segment, p_id, height, radius_top, radius_bottom, position);
		cylinder_mesh->set_height(height);
		cylinder_mesh->set_top_radius(radius_top);
		cylinder_mesh->set_bottom_radius(radius_bottom);
		mesh->set_global_position(position);
	}

	const Ref<BoxMesh> box_mesh = mesh->get_mesh();
	if (box_mesh.is_valid()) {
		Vector3 box_size, position;
		helper->box_set_handle(segment, p_id, box_size, position);
		box_mesh->set_size(box_size);
		mesh->set_global_position(position);
	}
}

void MeshInstance3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_gizmo->get_node_3d());

	const Ref<CapsuleMesh> capsule_mesh = mesh->get_mesh();
	if (capsule_mesh.is_valid()) {
		helper->cylinder_commit_handle(p_id, TTR("Change Capsule Mesh Radius"), TTR("Change Capsule Mesh Height"), p_cancel, mesh, *capsule_mesh, *capsule_mesh);
	}

	const Ref<CylinderMesh> cylinder_mesh = mesh->get_mesh();
	if (cylinder_mesh.is_valid()) {
		helper->cone_frustum_commit_handle(p_id, TTR("Change Cylinder Mesh Radius"), TTR("Change Cylinder Mesh Height"), p_cancel, mesh, *cylinder_mesh, *cylinder_mesh, *cylinder_mesh);
	}

	const Ref<BoxMesh> box_mesh = mesh->get_mesh();
	if (box_mesh.is_valid()) {
		helper->box_commit_handle(TTR("Change Box Mesh Size"), p_cancel, mesh, *box_mesh);
	}
}

void MeshInstance3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Ref<Mesh> m = mesh->get_mesh();

	const Ref<Material> handles_material = get_material("handles");

	if (m.is_null()) {
		return; //none
	}

	Ref<TriangleMesh> tm;

	Ref<PlaneMesh> plane_mesh = mesh->get_mesh();
	if (plane_mesh.is_valid() && (plane_mesh->get_subdivide_depth() > 0 || plane_mesh->get_subdivide_width() > 0)) {
		// PlaneMesh subdiv makes gizmo redraw very slow due to TriangleMesh BVH calculation for every face.
		// For gizmo collision this is very much unnecessary since a PlaneMesh is always flat, 2 faces is enough.
		Ref<PlaneMesh> simple_plane_mesh;
		simple_plane_mesh.instantiate();
		simple_plane_mesh->set_orientation(plane_mesh->get_orientation());
		simple_plane_mesh->set_size(plane_mesh->get_size());
		simple_plane_mesh->set_center_offset(plane_mesh->get_center_offset());
		tm = simple_plane_mesh->generate_triangle_mesh();
	} else {
		tm = m->generate_triangle_mesh();
	}

	if (tm.is_valid()) {
		p_gizmo->add_collision_triangles(tm);
	}

	const Ref<CapsuleMesh> capsule_mesh = mesh->get_mesh();
	if (capsule_mesh.is_valid()) {
		const Vector<Vector3> handles = helper->capsule_get_handles(capsule_mesh->get_height(), capsule_mesh->get_radius());
		p_gizmo->add_handles(handles, handles_material);
	}

	const Ref<CylinderMesh> cylinder_mesh = mesh->get_mesh();
	if (cylinder_mesh.is_valid()) {
		const Vector<Vector3> handles = helper->cone_frustum_get_handles(cylinder_mesh->get_height(), cylinder_mesh->get_top_radius(), cylinder_mesh->get_bottom_radius());
		p_gizmo->add_handles(handles, handles_material);
	}

	const Ref<BoxMesh> box_mesh = mesh->get_mesh();
	if (box_mesh.is_valid()) {
		const Vector<Vector3> handles = helper->box_get_handles(box_mesh->get_size());
		p_gizmo->add_handles(handles, handles_material);
	}
}
