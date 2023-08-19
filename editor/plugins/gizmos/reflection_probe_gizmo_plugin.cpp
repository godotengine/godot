/**************************************************************************/
/*  reflection_probe_gizmo_plugin.cpp                                     */
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

#include "reflection_probe_gizmo_plugin.h"

#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/reflection_probe.h"

ReflectionProbeGizmoPlugin::ReflectionProbeGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/reflection_probe", Color(0.6, 1, 0.5));

	create_material("reflection_probe_material", gizmo_color);

	gizmo_color.a = 0.5;
	create_material("reflection_internal_material", gizmo_color);

	gizmo_color.a = 0.1;
	create_material("reflection_probe_solid_material", gizmo_color);

	create_icon_material("reflection_probe_icon", Node3DEditor::get_singleton()->get_theme_icon(SNAME("GizmoReflectionProbe"), SNAME("EditorIcons")));
	create_handle_material("handles");
}

bool ReflectionProbeGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<ReflectionProbe>(p_spatial) != nullptr;
}

String ReflectionProbeGizmoPlugin::get_gizmo_name() const {
	return "ReflectionProbe";
}

int ReflectionProbeGizmoPlugin::get_priority() const {
	return -1;
}

String ReflectionProbeGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	switch (p_id) {
		case 0:
			return "Size X";
		case 1:
			return "Size Y";
		case 2:
			return "Size Z";
		case 3:
			return "Origin X";
		case 4:
			return "Origin Y";
		case 5:
			return "Origin Z";
	}

	return "";
}

Variant ReflectionProbeGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_node_3d());
	return AABB(probe->get_origin_offset(), probe->get_size());
}

void ReflectionProbeGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_node_3d());
	Transform3D gt = probe->get_global_transform();

	Transform3D gi = gt.affine_inverse();

	if (p_id < 3) {
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
	} else {
		p_id -= 3;

		Vector3 origin = probe->get_origin_offset();
		origin[p_id] = 0;

		Vector3 ray_from = p_camera->project_ray_origin(p_point);
		Vector3 ray_dir = p_camera->project_ray_normal(p_point);

		Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

		Vector3 axis;
		axis[p_id] = 1.0;

		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(origin - axis * 16384, origin + axis * 16384, sg[0], sg[1], ra, rb);
		// Adjust the actual position to account for the gizmo handle position
		float d = ra[p_id] + 0.25;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		origin[p_id] = d;
		probe->set_origin_offset(origin);
	}
}

void ReflectionProbeGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_node_3d());

	AABB restore = p_restore;

	if (p_cancel) {
		probe->set_origin_offset(restore.position);
		probe->set_size(restore.size);
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Change Probe Size"));
	ur->add_do_method(probe, "set_size", probe->get_size());
	ur->add_do_method(probe, "set_origin_offset", probe->get_origin_offset());
	ur->add_undo_method(probe, "set_size", restore.size);
	ur->add_undo_method(probe, "set_origin_offset", restore.position);
	ur->commit_action();
}

void ReflectionProbeGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector<Vector3> internal_lines;
	Vector3 size = probe->get_size();

	AABB aabb;
	aabb.position = -size / 2;
	aabb.size = size;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	for (int i = 0; i < 8; i++) {
		Vector3 ep = aabb.get_endpoint(i);
		internal_lines.push_back(probe->get_origin_offset());
		internal_lines.push_back(ep);
	}

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	for (int i = 0; i < 3; i++) {
		Vector3 orig_handle = probe->get_origin_offset();
		orig_handle[i] -= 0.25;
		lines.push_back(orig_handle);
		handles.push_back(orig_handle);

		orig_handle[i] += 0.5;
		lines.push_back(orig_handle);
	}

	Ref<Material> material = get_material("reflection_probe_material", p_gizmo);
	Ref<Material> material_internal = get_material("reflection_internal_material", p_gizmo);
	Ref<Material> icon = get_material("reflection_probe_icon", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_lines(internal_lines, material_internal);

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("reflection_probe_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, probe->get_size());
	}

	p_gizmo->add_unscaled_billboard(icon, 0.05);
	p_gizmo->add_handles(handles, get_material("handles"));
}
