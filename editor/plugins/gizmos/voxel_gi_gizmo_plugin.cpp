/**************************************************************************/
/*  voxel_gi_gizmo_plugin.cpp                                             */
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

#include "voxel_gi_gizmo_plugin.h"

#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/voxel_gi.h"

VoxelGIGizmoPlugin::VoxelGIGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/voxel_gi", Color(0.5, 1, 0.6));

	create_material("voxel_gi_material", gizmo_color);

	// This gizmo draws a lot of lines. Use a low opacity to make it not too intrusive.
	gizmo_color.a = 0.1;
	create_material("voxel_gi_internal_material", gizmo_color);

	gizmo_color.a = 0.05;
	create_material("voxel_gi_solid_material", gizmo_color);

	create_icon_material("voxel_gi_icon", Node3DEditor::get_singleton()->get_theme_icon(SNAME("GizmoVoxelGI"), SNAME("EditorIcons")));
	create_handle_material("handles");
}

bool VoxelGIGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<VoxelGI>(p_spatial) != nullptr;
}

String VoxelGIGizmoPlugin::get_gizmo_name() const {
	return "VoxelGI";
}

int VoxelGIGizmoPlugin::get_priority() const {
	return -1;
}

String VoxelGIGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
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

Variant VoxelGIGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	VoxelGI *probe = Object::cast_to<VoxelGI>(p_gizmo->get_node_3d());
	return probe->get_size();
}

void VoxelGIGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	VoxelGI *probe = Object::cast_to<VoxelGI>(p_gizmo->get_node_3d());

	Transform3D gt = probe->get_global_transform();
	Transform3D gi = gt.affine_inverse();

	Vector3 size = probe->get_size();

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
	probe->set_size(size);
}

void VoxelGIGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	VoxelGI *probe = Object::cast_to<VoxelGI>(p_gizmo->get_node_3d());

	Vector3 restore = p_restore;

	if (p_cancel) {
		probe->set_size(restore);
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Change Probe Size"));
	ur->add_do_method(probe, "set_size", probe->get_size());
	ur->add_undo_method(probe, "set_size", restore);
	ur->commit_action();
}

void VoxelGIGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	VoxelGI *probe = Object::cast_to<VoxelGI>(p_gizmo->get_node_3d());

	Ref<Material> material = get_material("voxel_gi_material", p_gizmo);
	Ref<Material> icon = get_material("voxel_gi_icon", p_gizmo);
	Ref<Material> material_internal = get_material("voxel_gi_internal_material", p_gizmo);

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector3 size = probe->get_size();

	static const int subdivs[VoxelGI::SUBDIV_MAX] = { 64, 128, 256, 512 };

	AABB aabb = AABB(-size / 2, size);
	int subdiv = subdivs[probe->get_subdiv()];
	float cell_size = aabb.get_longest_axis_size() / subdiv;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	p_gizmo->add_lines(lines, material);

	lines.clear();

	for (int i = 1; i < subdiv; i++) {
		for (int j = 0; j < 3; j++) {
			if (cell_size * i > aabb.size[j]) {
				continue;
			}

			int j_n1 = (j + 1) % 3;
			int j_n2 = (j + 2) % 3;

			for (int k = 0; k < 4; k++) {
				Vector3 from = aabb.position, to = aabb.position;
				from[j] += cell_size * i;
				to[j] += cell_size * i;

				if (k & 1) {
					to[j_n1] += aabb.size[j_n1];
				} else {
					to[j_n2] += aabb.size[j_n2];
				}

				if (k & 2) {
					from[j_n1] += aabb.size[j_n1];
					from[j_n2] += aabb.size[j_n2];
				}

				lines.push_back(from);
				lines.push_back(to);
			}
		}
	}

	p_gizmo->add_lines(lines, material_internal);

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("voxel_gi_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, aabb.get_size());
	}

	p_gizmo->add_unscaled_billboard(icon, 0.05);
	p_gizmo->add_handles(handles, get_material("handles"));
}
