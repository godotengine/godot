/**************************************************************************/
/*  fog_volume_gizmo_plugin.cpp                                           */
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

#include "fog_volume_gizmo_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/fog_volume.h"

FogVolumeGizmoPlugin::FogVolumeGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/fog_volume", Color(0.5, 0.7, 1));
	create_material("shape_material", gizmo_color);
	gizmo_color.a = 0.15;
	create_material("shape_material_internal", gizmo_color);

	create_icon_material("fog_volume_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoFogVolume"), EditorStringName(EditorIcons)));

	create_handle_material("handles");
}

bool FogVolumeGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return (Object::cast_to<FogVolume>(p_spatial) != nullptr);
}

String FogVolumeGizmoPlugin::get_gizmo_name() const {
	return "FogVolume";
}

int FogVolumeGizmoPlugin::get_priority() const {
	return -1;
}

String FogVolumeGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return "Size";
}

Variant FogVolumeGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return Vector3(p_gizmo->get_node_3d()->call("get_size"));
}

void FogVolumeGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	Node3D *sn = p_gizmo->get_node_3d();

	Transform3D gt = sn->get_global_transform();
	Transform3D gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	Vector3 axis;
	axis[p_id] = 1.0;
	Vector3 ra, rb;
	Geometry3D::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
	float d = ra[p_id] * 2;
	if (Node3DEditor::get_singleton()->is_snap_enabled()) {
		d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
	}

	if (d < 0.001) {
		d = 0.001;
	}

	Vector3 he = sn->call("get_size");
	he[p_id] = d;
	sn->call("set_size", he);
}

void FogVolumeGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	Node3D *sn = p_gizmo->get_node_3d();

	if (p_cancel) {
		sn->call("set_size", p_restore);
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Change Fog Volume Size"));
	ur->add_do_method(sn, "set_size", sn->call("get_size"));
	ur->add_undo_method(sn, "set_size", p_restore);
	ur->commit_action();
}

void FogVolumeGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	Node3D *cs = p_gizmo->get_node_3d();

	p_gizmo->clear();

	if (RS::FogVolumeShape(int(p_gizmo->get_node_3d()->call("get_shape"))) != RS::FOG_VOLUME_SHAPE_WORLD) {
		const Ref<Material> material =
				get_material("shape_material", p_gizmo);
		const Ref<Material> material_internal =
				get_material("shape_material_internal", p_gizmo);

		Ref<Material> handles_material = get_material("handles");

		Vector<Vector3> lines;
		AABB aabb;
		aabb.size = cs->call("get_size").operator Vector3();
		aabb.position = aabb.size / -2;

		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			aabb.get_edge(i, a, b);
			lines.push_back(a);
			lines.push_back(b);
		}

		Vector<Vector3> handles;

		for (int i = 0; i < 3; i++) {
			Vector3 ax;
			ax[i] = cs->call("get_size").operator Vector3()[i] / 2;
			handles.push_back(ax);
		}

		p_gizmo->add_lines(lines, material);
		p_gizmo->add_collision_segments(lines);
		const Ref<Material> icon = get_material("fog_volume_icon", p_gizmo);
		p_gizmo->add_unscaled_billboard(icon, 0.05);
		p_gizmo->add_handles(handles, handles_material);
	}
}
