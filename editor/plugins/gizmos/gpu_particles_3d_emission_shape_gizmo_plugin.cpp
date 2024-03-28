/**************************************************************************/
/*  gpu_particles_3d_emission_shape_gizmo_plugin.cpp                      */
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

#include "gpu_particles_3d_emission_shape_gizmo_plugin.h"

#include "core/math/transform_3d.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/resources/particle_process_material.h"
#include "scene/resources/primitive_meshes.h"

GPUParticles3DEmissionShapeGizmoPlugin::GPUParticles3DEmissionShapeGizmoPlugin() {
	helper.instantiate();

	Color gizmo_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/particles_emission_shape", Color(0.5, 0.7, 1));
	create_material("particles_emission_shape_material", gizmo_color);

	create_handle_material("handles");
}

bool GPUParticles3DEmissionShapeGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<GPUParticles3D>(p_spatial) != nullptr;
}

String GPUParticles3DEmissionShapeGizmoPlugin::get_gizmo_name() const {
	return "GPUParticles3DEmissionShape";
}

int GPUParticles3DEmissionShapeGizmoPlugin::get_priority() const {
	return 0;
}

bool GPUParticles3DEmissionShapeGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

String GPUParticles3DEmissionShapeGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	if (p_id == 0) {
		return "Radius";
	} else if (p_id == 1 || p_id == 2 || p_id == 3) {
		return "Box Extends";
	} else if (p_id == 4) {
		return "Ring Height";
	} else if (p_id == 5) {
		return "Ring Radius";
	} else if (p_id == 6) {
		return "Ring Inner Radius";
	} else {
		return "";
	}
}

Variant GPUParticles3DEmissionShapeGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
	Ref<ParticleProcessMaterial> mat = particles->get_process_material();

	if (p_id == 0) {
		return mat->get_emission_sphere_radius();
	} else if (p_id == 1 || p_id == 2 || p_id == 3) {
		return mat->get_emission_box_extents();
	} else if (p_id == 4) {
		return mat->get_emission_ring_height();
	} else if (p_id == 5) {
		return mat->get_emission_ring_radius();
	} else if (p_id == 6) {
		return mat->get_emission_ring_inner_radius();
	} else {
		return Variant();
	}
}

void GPUParticles3DEmissionShapeGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
	Ref<ParticleProcessMaterial> mat = particles->get_process_material();
	Transform3D gt = particles->get_global_transform();
	Transform3D gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };
	Vector3 ra, rb;
	float d;

	switch (p_id) {
		case 0:
			Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), s[0], s[1], ra, rb);

			d = -ra.z;
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
			}

			if (d <= 0) { // Equal is here for negative zero.
				d = 0;
			}

			mat->set_emission_sphere_radius(d);
		case 1:
			Geometry3D::get_closest_points_between_segments(Vector3(-4096, 0, 0), Vector3(4096, 0, 0), s[0], s[1], ra, rb);

			d = -ra.x;
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
			}

			if (d <= 0) { // Equal is here for negative zero.
				d = 0;
			}

			mat->set_emission_box_extents(Vector3(d, mat->get_emission_box_extents().y, mat->get_emission_box_extents().z));
		case 2:
			Geometry3D::get_closest_points_between_segments(Vector3(0, -4096, 0), Vector3(0, 4096, 0), s[0], s[1], ra, rb);

			d = -ra.y;
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
			}

			if (d <= 0) { // Equal is here for negative zero.
				d = 0;
			}

			mat->set_emission_box_extents(Vector3(mat->get_emission_box_extents().x, d, mat->get_emission_box_extents().z));
		case 3:
			Geometry3D::get_closest_points_between_segments(Vector3(0, 0, -4096), Vector3(0, 0, 4096), s[0], s[1], ra, rb);

			d = -ra.z;
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
			}

			if (d <= 0) { // Equal is here for negative zero.
				d = 0;
			}

			mat->set_emission_box_extents(Vector3(mat->get_emission_box_extents().x, mat->get_emission_box_extents().y, d));
		case 4:
			Geometry3D::get_closest_points_between_segments(Vector3(0, 0, -4096), Vector3(0, 0, 4096), s[0], s[1], ra, rb);
			// TODO: Ring cases
		case 5:
			// TODO: Ring cases
		case 6:
			// TODO: Ring cases
	}
}

void GPUParticles3DEmissionShapeGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	// TODO: commit handling
	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
	Ref<ParticleProcessMaterial> mat = particles->get_process_material();

	if (p_id == 0) {
		if (p_cancel) {
			mat->set_emission_sphere_radius(p_restore);
		}
		/*EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Set emission_sphere_radius"));
		ur->add_do_method(mat, "set_emission_sphere_radius", mat->get_emission_sphere_radius());
		ur->add_undo_method(mat, "set_mission_sphere_radius", p_restore);
		ur->commit_action();*/
	} else if (p_id == 1 || p_id == 2 || p_id == 3) {
		if (p_cancel) {
			mat->set_emission_box_extents(p_restore);
		}
		/*EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Set emission_box_extents"));
		ur->add_do_method(mat, "set_emission_box_extents", mat->get_emission_box_extents());
		ur->add_undo_method(mat, "set_emission_box_extents", p_restore);
		ur->commit_action();*/
	} else if (p_id == 4) {
		if (p_cancel) {
			mat->set_emission_ring_height(p_restore);
		}
		/*EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Set emission_ring_height"));
		ur->add_do_method(mat, "set_emission_ring_height", mat->get_emission_ring_height());
		ur->add_undo_method(mat, "set_emission_ring_height", p_restore);
		ur->commit_action();*/
	} else if (p_id == 5) {
		if (p_cancel) {
			mat->set_emission_ring_radius(p_restore);
		}
		/*EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Set emission_ring_radius"));
		ur->add_do_method(mat, "set_emission_ring_radius", mat->get_emission_ring_radius());
		ur->add_undo_method(mat, "set_emission_ring_radius", p_restore);
		ur->commit_action();*/
	} else if (p_id == 6) {
		if (p_cancel) {
			mat->set_emission_ring_inner_radius(p_restore);
		}
		/*EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Set emission_ring_inner_radius"));
		ur->add_do_method(mat, "set_emission_ring_inner_radius", mat->get_emission_ring_inner_radius());
		ur->add_undo_method(mat, "set_emission_ring_inner_radius", p_restore);
		ur->commit_action();*/
	}
}

void GPUParticles3DEmissionShapeGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());

	if (particles->get_process_material() != nullptr) {
		Ref<ParticleProcessMaterial> mat = particles->get_process_material();
		ParticleProcessMaterial::EmissionShape shape = mat->get_emission_shape();

		WARN_PRINT(vformat("Current Emission Shape: %d", shape));

		const Ref<Material> material =
				get_material("particles_emission_shape_material", p_gizmo);
		Ref<Material> handles_material = get_material("handles");

		if (shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE || shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE_SURFACE) {
			WARN_PRINT("Emission shape changed to SPHERE");
			float r = mat->get_emission_sphere_radius();

			Vector<Vector3> points;

			for (int i = 0; i <= 360; i++) {
				float ra = Math::deg_to_rad((float)i);
				float rb = Math::deg_to_rad((float)i + 1);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
				Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

				points.push_back(Vector3(a.x, 0, a.y));
				points.push_back(Vector3(b.x, 0, b.y));
				points.push_back(Vector3(0, a.x, a.y));
				points.push_back(Vector3(0, b.x, b.y));
				points.push_back(Vector3(a.x, a.y, 0));
				points.push_back(Vector3(b.x, b.y, 0));
			}

			Vector<Vector3> handles;
			Vector<int> ids;
			handles.push_back(Vector3(r, 0, 0));
			ids.push_back(0);

			p_gizmo->add_lines(points, material);
			p_gizmo->add_handles(handles, handles_material, ids);
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_BOX) {
			WARN_PRINT("Emission shape changed to BOX");
			Vector3 box_extents = mat->get_emission_box_extents();
			Ref<BoxMesh> box = memnew(BoxMesh);
			AABB box_aabb = box->get_aabb();
			Vector<Vector3> lines;

			for (int i = 0; i < 12; i++) {
				Vector3 a, b;
				box_aabb.get_edge(i, a, b);
				lines.push_back(a);
				lines.push_back(b);
			}

			Vector<Vector3> handles;
			Vector<int> ids;
			handles.push_back(Vector3(box_extents.x, 0.0, 0.0));
			ids.push_back(1);
			handles.push_back(Vector3(0.0, box_extents.y, 0.0));
			ids.push_back(2);
			handles.push_back(Vector3(0.0, 0.0, box_extents.z));
			ids.push_back(3);

			p_gizmo->add_handles(handles, handles_material, ids);
			p_gizmo->add_lines(lines, material);
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_RING) {
			WARN_PRINT("Emission shape changed to RING");
			float ring_height = mat->get_emission_ring_height();
			float half_ring_height = ring_height / 2;
			float ring_radius = mat->get_emission_ring_radius();
			float ring_inner_radius = mat->get_emission_ring_inner_radius();

			Vector<Vector3> points;

			for (int i = 0; i <= 360; i++) {
				float ra = Math::deg_to_rad((float)i);
				float rb = Math::deg_to_rad((float)i + 1);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_radius;
				Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * ring_radius;
				Point2 inner_a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_inner_radius;
				Point2 inner_b = Vector2(Math::sin(rb), Math::cos(rb)) * ring_inner_radius;

				// outer top ring cap
				points.push_back(Vector3(a.x, half_ring_height, a.y));
				points.push_back(Vector3(b.x, half_ring_height, b.y));
				// outer bottom ring cap
				points.push_back(Vector3(a.x, -half_ring_height, a.y));
				points.push_back(Vector3(b.x, -half_ring_height, b.y));

				// inner top ring cap
				points.push_back(Vector3(inner_a.x, half_ring_height, inner_a.y));
				points.push_back(Vector3(inner_b.x, half_ring_height, inner_b.y));
				// inner bottom ring cap
				points.push_back(Vector3(inner_a.x, -half_ring_height, inner_a.y));
				points.push_back(Vector3(inner_b.x, -half_ring_height, inner_b.y));
			}

			for (int i = 0; i <= 360; i=i+90) {
				float ra = Math::deg_to_rad((float)i);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_radius;
				Point2 inner_a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_inner_radius;

				// outer 90 degrees vertical lines
				points.push_back(Vector3(a.x, half_ring_height, a.y));
				points.push_back(Vector3(a.x, -half_ring_height, a.y));

				// inner 90 degrees vertical lines
				points.push_back(Vector3(inner_a.x, half_ring_height, inner_a.y));
				points.push_back(Vector3(inner_a.x, -half_ring_height, inner_a.y));
			}

			Transform3D transform = Transform3D();
			transform = transform.scaled(Vector3(2,2,2));
			transform = transform.looking_at(transform.origin + Vector3(0.0, -1.0, 0.0), Vector3(0.0, 0.0, 1.0), true);

			p_gizmo->add_lines(points, material);
		}
	}
}
