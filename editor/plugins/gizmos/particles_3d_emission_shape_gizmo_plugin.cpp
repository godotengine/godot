/**************************************************************************/
/*  particles_3d_emission_shape_gizmo_plugin.cpp                          */
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

#include "particles_3d_emission_shape_gizmo_plugin.h"

#include "core/math/transform_3d.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/cpu_particles_3d.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/particle_process_material.h"

Particles3DEmissionShapeGizmoPlugin::Particles3DEmissionShapeGizmoPlugin() {
	helper.instantiate();

	Color gizmo_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/particles_emission_shape", Color(0.5, 0.7, 1));
	create_material("particles_emission_shape_material", gizmo_color);

	create_handle_material("handles");
}

bool Particles3DEmissionShapeGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<GPUParticles3D>(p_spatial) || Object::cast_to<CPUParticles3D>(p_spatial) != nullptr;
}

String Particles3DEmissionShapeGizmoPlugin::get_gizmo_name() const {
	return "Particles3DEmissionShape";
}

int Particles3DEmissionShapeGizmoPlugin::get_priority() const {
	return -1;
}

bool Particles3DEmissionShapeGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

String Particles3DEmissionShapeGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	if (Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d())) {
		GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
		Ref<ParticleProcessMaterial> mat = particles->get_process_material();
		ParticleProcessMaterial::EmissionShape shape = mat->get_emission_shape();

		if (shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE || shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE_SURFACE) {
			return "Radius";
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_BOX) {
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
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_RING) {
			if (p_id == 0) {
				return "Axis";
			} else if (p_id == 1) {
				return "Height";
			} else if (p_id == 2) {
				return "Radius";
			} else {
				return "Inner Radius";
			}
		}
	} else if (Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d())) {
		CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d());
		CPUParticles3D::EmissionShape shape = particles->get_emission_shape();

		if (shape == CPUParticles3D::EMISSION_SHAPE_SPHERE || shape == CPUParticles3D::EMISSION_SHAPE_SPHERE_SURFACE) {
			return "Radius";
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_BOX) {
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
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_RING) {
			if (p_id == 0) {
				return "Axis";
			} else if (p_id == 1) {
				return "Height";
			} else if (p_id == 2) {
				return "Radius";
			} else {
				return "Inner Radius";
			}
		}
	}

	return "";
}

Variant Particles3DEmissionShapeGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	if (Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d())) {
		GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
		Ref<ParticleProcessMaterial> mat = particles->get_process_material();
		ParticleProcessMaterial::EmissionShape shape = mat->get_emission_shape();

		if (shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE || shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE_SURFACE) {
			return mat->get_emission_sphere_radius();
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_BOX) {
			return mat->get_emission_box_extents();
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_RING) {
			if (p_id == 0) {
				return mat->get_emission_ring_axis();
			} else if (p_id == 1) {
				return mat->get_emission_ring_height();
			} else if (p_id == 2) {
				return mat->get_emission_ring_radius();
			} else {
				return mat->get_emission_ring_inner_radius();
			}
		}
	} else if (Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d())) {
		CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d());
		CPUParticles3D::EmissionShape shape = particles->get_emission_shape();

		if (shape == CPUParticles3D::EMISSION_SHAPE_SPHERE || shape == CPUParticles3D::EMISSION_SHAPE_SPHERE_SURFACE) {
			return particles->get_emission_sphere_radius();
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_BOX) {
			return particles->get_emission_box_extents();
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_RING) {
			if (p_id == 0) {
				return particles->get_emission_ring_axis();
			} else if (p_id == 1) {
				return particles->get_emission_ring_height();
			} else if (p_id == 2) {
				return particles->get_emission_ring_radius();
			} else {
				return particles->get_emission_ring_inner_radius();
			}
		}
	}

	return Variant();
}

void Particles3DEmissionShapeGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	if (Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d())) {
		GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
		Ref<ParticleProcessMaterial> mat = particles->get_process_material();
		ParticleProcessMaterial::EmissionShape shape = mat->get_emission_shape();

		Transform3D gt = particles->get_global_transform();
		Transform3D gi = gt.affine_inverse();
		Vector3 ray_from = p_camera->project_ray_origin(p_point);
		Vector3 ray_dir = p_camera->project_ray_normal(p_point);
		Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

		Vector3 ra, rb;
		float d;

		if (shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE || shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE_SURFACE) {
			Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), s[0], s[1], ra, rb);
			d = ra.x;
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
			}
			d = MAX(d, 0.001);
			mat->set_emission_sphere_radius(d);
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_BOX) {
			Vector3 initial_size = mat->get_emission_box_extents();
			int axis = p_id / 2;
			int sign = p_id % 2 * -2 + 1;
			float neg_end = initial_size[axis] * -0.5;
			float pos_end = initial_size[axis] * 0.5;

			Vector3 axis_segment[2] = { Vector3(), Vector3() };
			axis_segment[0][axis] = 4096.0;
			axis_segment[1][axis] = -4096.0;
			Geometry3D::get_closest_points_between_segments(axis_segment[0], axis_segment[1], s[0], s[1], ra, rb);

			Vector3 r_box_size = initial_size;
			ra[axis] = ra[axis] * 0.5;
			if (Input::get_singleton()->is_key_pressed(Key::ALT)) {
				r_box_size[axis] = ra[axis] * sign * 2;
			} else {
				r_box_size[axis] = sign > 0 ? ra[axis] - neg_end : pos_end - ra[axis];
			}

			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				r_box_size[axis] = Math::snapped(r_box_size[axis], Node3DEditor::get_singleton()->get_translate_snap());
			}
			r_box_size[axis] = MAX(r_box_size[axis], 0.001);

			Vector3 offset;
			offset[axis] = (pos_end + neg_end) * 0.5;
			Vector3 r_box_position = gt.xform(offset);

			mat->set_emission_box_extents(r_box_size);
			mat->set_emission_shape_offset(r_box_position);
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_RING) {
			Vector3 ring_axis = mat->get_emission_ring_axis();
			Basis axis_basis;
			axis_basis.rows[1] = ring_axis.normalized();
			axis_basis.rows[0] = Vector3(axis_basis[1][1], -axis_basis[1][2], -axis_basis[1][0]).normalized();
			axis_basis.rows[0] = (axis_basis[0] - axis_basis[0].dot(axis_basis[1]) * axis_basis[1]).normalized();
			axis_basis[2] = axis_basis[0].cross(axis_basis[1]).normalized();

			if (p_id == 1) {
				Geometry3D::get_closest_points_between_segments(Vector3(0, 0, -4096), Vector3(0, 0, 4096), s[0], s[1], ra, rb);
				d = ra.z;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
				}
				d = MAX(d, 0.001);
				// Times 2 because of using the half_height for drawing
				mat->set_emission_ring_height(2.0 * d);
			} else if (p_id == 2) {
				Geometry3D::get_closest_points_between_segments(Vector3(-4096, 0, 0), Vector3(4096, 0, 0), s[0], s[1], ra, rb);
				d = ra.x;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
				}
				d = MAX(d, 0.001);
				mat->set_emission_ring_radius(d);
			} else {
				Geometry3D::get_closest_points_between_segments(Vector3(-4096, 0, 0), Vector3(4096, 0, 0), s[0], s[1], ra, rb);
				d = ra.x;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
				}
				d = MAX(d, 0.001);
				mat->set_emission_ring_inner_radius(d);
			}
		}

	} else if (Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d())) {
		CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d());
		CPUParticles3D::EmissionShape shape = particles->get_emission_shape();

		Transform3D gt = particles->get_global_transform();
		Transform3D gi = gt.affine_inverse();
		Vector3 ray_from = p_camera->project_ray_origin(p_point);
		Vector3 ray_dir = p_camera->project_ray_normal(p_point);
		Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

		Vector3 ra, rb;
		float d;

		if (shape == CPUParticles3D::EMISSION_SHAPE_SPHERE || shape == CPUParticles3D::EMISSION_SHAPE_SPHERE_SURFACE) {
			Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), s[0], s[1], ra, rb);
			d = ra.x;
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
			}
			d = MAX(d, 0.001);
			particles->set_emission_sphere_radius(d);
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_BOX) {
			Vector3 initial_size = particles->get_emission_box_extents();
			int axis = p_id / 2;
			int sign = p_id % 2 * -2 + 1;
			float neg_end = initial_size[axis] * -0.5;
			float pos_end = initial_size[axis] * 0.5;

			Vector3 axis_segment[2] = { Vector3(), Vector3() };
			axis_segment[0][axis] = 4096.0;
			axis_segment[1][axis] = -4096.0;
			Geometry3D::get_closest_points_between_segments(axis_segment[0], axis_segment[1], s[0], s[1], ra, rb);

			Vector3 r_box_size = initial_size;
			ra[axis] = ra[axis] * 0.5;
			if (Input::get_singleton()->is_key_pressed(Key::ALT)) {
				r_box_size[axis] = ra[axis] * sign * 2;
			} else {
				r_box_size[axis] = sign > 0 ? ra[axis] - neg_end : pos_end - ra[axis];
			}

			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				r_box_size[axis] = Math::snapped(r_box_size[axis], Node3DEditor::get_singleton()->get_translate_snap());
			}
			r_box_size[axis] = MAX(r_box_size[axis], 0.001);

			// TODO: Remove unused variable
			//Vector3 offset;
			//offset[axis] = (pos_end + neg_end) * 0.5;
			//Vector3 r_box_position = gt.xform(offset);

			particles->set_emission_box_extents(r_box_size);
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_RING) {
			Vector3 ring_axis = particles->get_emission_ring_axis();
			Basis axis_basis;
			axis_basis.rows[1] = ring_axis.normalized();
			axis_basis.rows[0] = Vector3(axis_basis[1][1], -axis_basis[1][2], -axis_basis[1][0]).normalized();
			axis_basis.rows[0] = (axis_basis[0] - axis_basis[0].dot(axis_basis[1]) * axis_basis[1]).normalized();
			axis_basis[2] = axis_basis[0].cross(axis_basis[1]).normalized();

			if (p_id == 1) {
				Geometry3D::get_closest_points_between_segments(Vector3(0, 0, -4096), Vector3(0, 0, 4096), s[0], s[1], ra, rb);
				d = ra.z;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
				}
				d = MAX(d, 0.001);
				// Times 2 because of using the half_height for drawing
				particles->set_emission_ring_height(2.0 * d);
			} else if (p_id == 2) {
				Geometry3D::get_closest_points_between_segments(Vector3(-4096, 0, 0), Vector3(4096, 0, 0), s[0], s[1], ra, rb);
				d = ra.x;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
				}
				d = MAX(d, 0.001);
				particles->set_emission_ring_radius(d);
			} else {
				Geometry3D::get_closest_points_between_segments(Vector3(-4096, 0, 0), Vector3(4096, 0, 0), s[0], s[1], ra, rb);
				d = ra.x;
				if (Node3DEditor::get_singleton()->is_snap_enabled()) {
					d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
				}
				d = MAX(d, 0.001);
				particles->set_emission_ring_inner_radius(d);
			}
		}
	}
}

void Particles3DEmissionShapeGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	if (Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d())) {
		GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());
		Ref<ParticleProcessMaterial> mat = particles->get_process_material();
		ParticleProcessMaterial *process_mat = Object::cast_to<ParticleProcessMaterial>(mat.ptr());
		ParticleProcessMaterial::EmissionShape shape = mat->get_emission_shape();

		if (shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE || shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE_SURFACE) {
			if (p_cancel) {
				mat->set_emission_sphere_radius(p_restore);
			} else {
				EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
				ur->create_action(TTR("Change emission sphere radius"));
				ur->add_do_property(process_mat, "emission_sphere_radius", process_mat->get_emission_sphere_radius());
				ur->add_undo_property(process_mat, "emission_sphere_radius", p_restore);
				ur->commit_action();
			}
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_BOX) {
			if (p_cancel) {
				mat->set_emission_box_extents(p_restore);
				return;
			} else {
				EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
				ur->create_action(TTR("Change emission box extents"));
				ur->add_do_property(process_mat, "emission_box_extents", process_mat->get_emission_box_extents());
				ur->add_undo_property(process_mat, "emission_box_extents", p_restore);
				ur->commit_action();
			}
		} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_RING) {
			if (p_id == 0) {
				if (p_cancel) {
					mat->set_emission_ring_axis(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring axis"));
					ur->add_do_property(process_mat, "emission_ring_axis", process_mat->get_emission_ring_axis());
					ur->add_undo_property(process_mat, "emission_ring_axis", p_restore);
					ur->commit_action();
				}
			} else if (p_id == 1) {
				if (p_cancel) {
					mat->set_emission_ring_height(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring height"));
					ur->add_do_property(process_mat, "emission_ring_height", process_mat->get_emission_ring_height());
					ur->add_undo_property(process_mat, "emission_ring_height", p_restore);
					ur->commit_action();
				}
			} else if (p_id == 2) {
				if (p_cancel) {
					mat->set_emission_ring_radius(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring radius"));
					ur->add_do_property(process_mat, "emission_ring_radius", process_mat->get_emission_ring_radius());
					ur->add_undo_property(process_mat, "emission_ring_radius", p_restore);
					ur->commit_action();
				}
			} else {
				if (p_cancel) {
					mat->set_emission_ring_inner_radius(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring inner radius"));
					ur->add_do_property(process_mat, "emission_ring_inner_radius", process_mat->get_emission_ring_inner_radius());
					ur->add_undo_property(process_mat, "emission_ring_inner_radius", p_restore);
					ur->commit_action();
				}
			}
		}
	} else if (Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d())) {
		CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d());
		CPUParticles3D::EmissionShape shape = particles->get_emission_shape();

		if (shape == CPUParticles3D::EMISSION_SHAPE_SPHERE || shape == CPUParticles3D::EMISSION_SHAPE_SPHERE_SURFACE) {
			if (p_cancel) {
				particles->set_emission_sphere_radius(p_restore);
			} else {
				EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
				ur->create_action(TTR("Change emission sphere radius"));
				ur->add_do_property(particles, "emission_sphere_radius", particles->get_emission_sphere_radius());
				ur->add_undo_property(particles, "emission_sphere_radius", p_restore);
				ur->commit_action();
			}
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_BOX) {
			if (p_cancel) {
				particles->set_emission_box_extents(p_restore);
				return;
			} else {
				EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
				ur->create_action(TTR("Change emission box extents"));
				ur->add_do_property(particles, "emission_box_extents", particles->get_emission_box_extents());
				ur->add_undo_property(particles, "emission_box_extents", p_restore);
				ur->commit_action();
			}
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_RING) {
			if (p_id == 0) {
				if (p_cancel) {
					particles->set_emission_ring_axis(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring axis"));
					ur->add_do_property(particles, "emission_ring_axis", particles->get_emission_ring_axis());
					ur->add_undo_property(particles, "emission_ring_axis", p_restore);
					ur->commit_action();
				}
			} else if (p_id == 1) {
				if (p_cancel) {
					particles->set_emission_ring_height(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring height"));
					ur->add_do_property(particles, "emission_ring_height", particles->get_emission_ring_height());
					ur->add_undo_property(particles, "emission_ring_height", p_restore);
					ur->commit_action();
				}
			} else if (p_id == 2) {
				if (p_cancel) {
					particles->set_emission_ring_radius(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring radius"));
					ur->add_do_property(particles, "emission_ring_radius", particles->get_emission_ring_radius());
					ur->add_undo_property(particles, "emission_ring_radius", p_restore);
					ur->commit_action();
				}
			} else {
				if (p_cancel) {
					particles->set_emission_ring_inner_radius(p_restore);
				} else {
					EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
					ur->create_action(TTR("Change emission ring inner radius"));
					ur->add_do_property(particles, "emission_ring_inner_radius", particles->get_emission_ring_inner_radius());
					ur->add_undo_property(particles, "emission_ring_inner_radius", p_restore);
					ur->commit_action();
				}
			}
		}
	}
}

void Particles3DEmissionShapeGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	if (Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d())) {
		GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());

		if (particles->get_process_material() != nullptr) {
			Ref<ParticleProcessMaterial> mat = particles->get_process_material();
			ParticleProcessMaterial::EmissionShape shape = mat->get_emission_shape();

			const Ref<Material> material = get_material("particles_emission_shape_material", p_gizmo);
			Ref<Material> handles_material = get_material("handles");

			if (shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE || shape == ParticleProcessMaterial::EMISSION_SHAPE_SPHERE_SURFACE) {
				Vector3 offset = mat->get_emission_shape_offset();
				Vector3 scale = mat->get_emission_shape_scale();

				float r = mat->get_emission_sphere_radius();
				Vector<Vector3> points;
				for (int i = 0; i <= 360; i++) {
					float ra = Math::deg_to_rad((float)i);
					float rb = Math::deg_to_rad((float)i + 1);
					Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
					Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

					points.push_back(Vector3(a.x * scale.x + offset.x, offset.y, a.y * scale.z + offset.z));
					points.push_back(Vector3(b.x * scale.x + offset.x, offset.y, b.y * scale.z + offset.z));
					points.push_back(Vector3(offset.x, a.x * scale.y + offset.y, a.y * scale.z + offset.z));
					points.push_back(Vector3(offset.x, b.x * scale.y + offset.y, b.y * scale.z + offset.z));
					points.push_back(Vector3(a.x * scale.x + offset.x, a.y * scale.y + offset.y, offset.z));
					points.push_back(Vector3(b.x * scale.x + offset.x, b.y * scale.y + offset.y, offset.z));
				}

				Vector<Vector3> handles;
				Vector<int> ids;
				handles.push_back(Vector3(r, 0, 0));
				ids.push_back(0);

				p_gizmo->add_lines(points, material);
				p_gizmo->add_handles(handles, handles_material, ids);
			} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_BOX) {
				Vector3 offset = mat->get_emission_shape_offset();
				Vector3 scale = mat->get_emission_shape_scale();

				Vector3 box_extents = mat->get_emission_box_extents();
				Ref<BoxMesh> box = memnew(BoxMesh);
				AABB box_aabb = box->get_aabb();
				Vector<Vector3> lines;

				for (int i = 0; i < 12; i++) {
					Vector3 a, b;
					box_aabb.get_edge(i, a, b);
					// Multiplication by 2 due to the extents being only half of the box size
					lines.push_back(a * 2.0 * scale * box_extents + offset);
					lines.push_back(b * 2.0 * scale * box_extents + offset);
				}

				Vector<Vector3> handles = helper->box_get_handles(mat->get_emission_box_extents() * 2.0);

				p_gizmo->add_lines(lines, material);
				p_gizmo->add_handles(handles, handles_material);
			} else if (shape == ParticleProcessMaterial::EMISSION_SHAPE_RING) {
				Vector3 offset = mat->get_emission_shape_offset();
				Vector3 scale = mat->get_emission_shape_scale();

				float ring_height = mat->get_emission_ring_height();
				float half_ring_height = ring_height / 2;
				float ring_radius = mat->get_emission_ring_radius();
				float ring_inner_radius = mat->get_emission_ring_inner_radius();
				Vector3 ring_axis = mat->get_emission_ring_axis();

				Vector<Vector3> points;

				Basis basis;
				basis.rows[1] = ring_axis.normalized();
				basis.rows[0] = Vector3(basis[1][1], -basis[1][2], -basis[1][0]).normalized();
				basis.rows[0] = (basis[0] - basis[0].dot(basis[1]) * basis[1]).normalized();
				basis[2] = basis[0].cross(basis[1]).normalized();
				basis = basis.inverse();

				for (int i = 0; i <= 360; i++) {
					float ra = Math::deg_to_rad((float)i);
					float rb = Math::deg_to_rad((float)i + 1);
					Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_radius;
					Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * ring_radius;
					Point2 inner_a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_inner_radius;
					Point2 inner_b = Vector2(Math::sin(rb), Math::cos(rb)) * ring_inner_radius;

					// outer top ring cap
					points.push_back(basis.xform(Vector3(a.x * scale.x + offset.x, half_ring_height * scale.y + offset.y, a.y * scale.z + offset.z)));
					points.push_back(basis.xform(Vector3(b.x * scale.x + offset.x, half_ring_height * scale.y + offset.y, b.y * scale.z + offset.z)));
					// outer bottom ring cap
					points.push_back(basis.xform(Vector3(a.x * scale.x + offset.x, -half_ring_height * scale.y + offset.y, a.y * scale.z + offset.z)));
					points.push_back(basis.xform(Vector3(b.x * scale.x + offset.x, -half_ring_height * scale.y + offset.y, b.y * scale.z + offset.z)));

					// inner top ring cap
					points.push_back(basis.xform(Vector3(inner_a.x * scale.x + offset.x, half_ring_height * scale.y + offset.y, inner_a.y * scale.z + offset.z)));
					points.push_back(basis.xform(Vector3(inner_b.x * scale.x + offset.x, half_ring_height * scale.y + offset.y, inner_b.y * scale.z + offset.z)));
					// inner bottom ring cap
					points.push_back(basis.xform(Vector3(inner_a.x * scale.x + offset.x, -half_ring_height * scale.y + offset.y, inner_a.y * scale.z + offset.z)));
					points.push_back(basis.xform(Vector3(inner_b.x * scale.x + offset.x, -half_ring_height * scale.y + offset.y, inner_b.y * scale.z + offset.z)));
				}

				for (int i = 0; i <= 360; i = i + 90) {
					float ra = Math::deg_to_rad((float)i);
					Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_radius;
					Point2 inner_a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_inner_radius;

					// outer 90 degrees vertical lines
					points.push_back(basis.xform(Vector3(a.x * scale.x + offset.x, half_ring_height * scale.y + offset.y, a.y * scale.z + offset.z)));
					points.push_back(basis.xform(Vector3(a.x * scale.x + offset.x, -half_ring_height * scale.y + offset.y, a.y * scale.z + offset.z)));

					// inner 90 degrees vertical lines
					points.push_back(basis.xform(Vector3(inner_a.x * scale.x + offset.x, half_ring_height * scale.y + offset.y, inner_a.y * scale.z + offset.z)));
					points.push_back(basis.xform(Vector3(inner_a.x * scale.x + offset.x, -half_ring_height * scale.y + offset.y, inner_a.y * scale.z + offset.z)));
				}

				Vector<Vector3> handles;
				Vector<int> ids;
				handles.push_back(Vector3(0, 0, half_ring_height));
				ids.push_back(1);
				handles.push_back(Vector3(ring_radius, 0, 0));
				ids.push_back(2);
				handles.push_back(Vector3(ring_inner_radius, 0, 0));
				ids.push_back(3);

				p_gizmo->add_lines(points, material);
				p_gizmo->add_handles(handles, handles_material, ids);
			}
		}
	} else if (Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d())) {
		CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(p_gizmo->get_node_3d());
		CPUParticles3D::EmissionShape shape = particles->get_emission_shape();

		const Ref<Material> material = get_material("particles_emission_shape_material", p_gizmo);
		Ref<Material> handles_material = get_material("handles");

		if (shape == CPUParticles3D::EMISSION_SHAPE_SPHERE || shape == CPUParticles3D::EMISSION_SHAPE_SPHERE_SURFACE) {
			float r = particles->get_emission_sphere_radius();
			Vector<Vector3> points;
			for (int i = 0; i <= 360; i++) {
				float ra = Math::deg_to_rad((float)i);
				float rb = Math::deg_to_rad((float)i + 1);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
				Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

				points.push_back(Vector3(a.x, 0.0, a.y));
				points.push_back(Vector3(b.x, 0.0, b.y));
				points.push_back(Vector3(0.0, a.x, a.y));
				points.push_back(Vector3(0.0, b.x, b.y));
				points.push_back(Vector3(a.x, a.y, 0.0));
				points.push_back(Vector3(b.x, b.y, 0.0));
			}

			Vector<Vector3> handles;
			Vector<int> ids;
			handles.push_back(Vector3(r, 0, 0));
			ids.push_back(0);

			p_gizmo->add_lines(points, material);
			p_gizmo->add_handles(handles, handles_material, ids);
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_BOX) {
			Vector3 box_extents = particles->get_emission_box_extents();
			Ref<BoxMesh> box = memnew(BoxMesh);
			AABB box_aabb = box->get_aabb();
			Vector<Vector3> lines;

			for (int i = 0; i < 12; i++) {
				Vector3 a, b;
				box_aabb.get_edge(i, a, b);
				// Multiplication by 2 due to the extents being only half of the box size
				lines.push_back(a * 2.0 * box_extents);
				lines.push_back(b * 2.0 * box_extents);
			}

			Vector<Vector3> handles = helper->box_get_handles(particles->get_emission_box_extents() * 2.0);

			p_gizmo->add_lines(lines, material);
			p_gizmo->add_handles(handles, handles_material);
		} else if (shape == CPUParticles3D::EMISSION_SHAPE_RING) {
			float ring_height = particles->get_emission_ring_height();
			float half_ring_height = ring_height / 2;
			float ring_radius = particles->get_emission_ring_radius();
			float ring_inner_radius = particles->get_emission_ring_inner_radius();
			Vector3 ring_axis = particles->get_emission_ring_axis();

			Vector<Vector3> points;

			Basis basis;
			basis.rows[1] = ring_axis.normalized();
			basis.rows[0] = Vector3(basis[1][1], -basis[1][2], -basis[1][0]).normalized();
			basis.rows[0] = (basis[0] - basis[0].dot(basis[1]) * basis[1]).normalized();
			basis[2] = basis[0].cross(basis[1]).normalized();
			basis = basis.inverse();

			for (int i = 0; i <= 360; i++) {
				float ra = Math::deg_to_rad((float)i);
				float rb = Math::deg_to_rad((float)i + 1);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_radius;
				Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * ring_radius;
				Point2 inner_a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_inner_radius;
				Point2 inner_b = Vector2(Math::sin(rb), Math::cos(rb)) * ring_inner_radius;

				// outer top ring cap
				points.push_back(basis.xform(Vector3(a.x, half_ring_height, a.y)));
				points.push_back(basis.xform(Vector3(b.x, half_ring_height, b.y)));
				// outer bottom ring cap
				points.push_back(basis.xform(Vector3(a.x, -half_ring_height, a.y)));
				points.push_back(basis.xform(Vector3(b.x, -half_ring_height, b.y)));

				// inner top ring cap
				points.push_back(basis.xform(Vector3(inner_a.x, half_ring_height, inner_a.y)));
				points.push_back(basis.xform(Vector3(inner_b.x, half_ring_height, inner_b.y)));
				// inner bottom ring cap
				points.push_back(basis.xform(Vector3(inner_a.x, -half_ring_height, inner_a.y)));
				points.push_back(basis.xform(Vector3(inner_b.x, -half_ring_height, inner_b.y)));
			}

			for (int i = 0; i <= 360; i = i + 90) {
				float ra = Math::deg_to_rad((float)i);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_radius;
				Point2 inner_a = Vector2(Math::sin(ra), Math::cos(ra)) * ring_inner_radius;

				// outer 90 degrees vertical lines
				points.push_back(basis.xform(Vector3(a.x, half_ring_height, a.y)));
				points.push_back(basis.xform(Vector3(a.x, -half_ring_height, a.y)));

				// inner 90 degrees vertical lines
				points.push_back(basis.xform(Vector3(inner_a.x, half_ring_height, inner_a.y)));
				points.push_back(basis.xform(Vector3(inner_a.x, -half_ring_height, inner_a.y)));
			}

			Vector<Vector3> handles;
			Vector<int> ids;
			handles.push_back(Vector3(0, 0, half_ring_height));
			ids.push_back(1);
			handles.push_back(Vector3(ring_radius, 0, 0));
			ids.push_back(2);
			handles.push_back(Vector3(ring_inner_radius, 0, 0));
			ids.push_back(3);

			p_gizmo->add_lines(points, material);
			p_gizmo->add_handles(handles, handles_material, ids);
		}
	}
}
