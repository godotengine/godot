/**************************************************************************/
/*  lightmap_probe_gizmo_plugin.cpp                                       */
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

#include "lightmap_probe_gizmo_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/gizmos/gizmo_3d_helper.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/lightmap_probe.h"

LightmapProbeGizmoPlugin::LightmapProbeGizmoPlugin() {
	helper.instantiate();
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/lightprobe_lines");

	create_material("lightmap_probe_material", gizmo_color);

	gizmo_color.a = 0.1;
	create_material("lightmap_probe_solid_material", gizmo_color);

	create_icon_material("lightmap_probe_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoLightmapProbe"), EditorStringName(EditorIcons)));
	create_handle_material("handles");
}

LightmapProbeGizmoPlugin::~LightmapProbeGizmoPlugin() {
}

bool LightmapProbeGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<LightmapProbe>(p_spatial) != nullptr;
}

String LightmapProbeGizmoPlugin::get_gizmo_name() const {
	return "LightmapProbe";
}

int LightmapProbeGizmoPlugin::get_priority() const {
	return -1;
}

String LightmapProbeGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return helper->box_get_handle_name(p_id);
}

Variant LightmapProbeGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	LightmapProbe *probe = Object::cast_to<LightmapProbe>(p_gizmo->get_node_3d());
	return probe->get_size();
}

void LightmapProbeGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void LightmapProbeGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	LightmapProbe *probe = Object::cast_to<LightmapProbe>(p_gizmo->get_node_3d());

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	Vector3 size = probe->get_size();
	Vector3 position;
	helper->box_set_handle(sg, p_id, size, position);
	probe->set_size(size);
	probe->set_global_position(position);
}

void LightmapProbeGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	helper->box_commit_handle(TTR("Change LightmapProbe Size"), p_cancel, p_gizmo->get_node_3d());
}

void LightmapProbeGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	if (p_gizmo->is_selected()) {
		LightmapProbe *probe = Object::cast_to<LightmapProbe>(p_gizmo->get_node_3d());
		Vector<Vector3> lines;
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

		Vector<Vector3> handles = helper->box_get_handles(probe->get_size());

		Ref<Material> material = get_material("lightmap_probe_material", p_gizmo);

		p_gizmo->add_lines(lines, material);

		if (p_gizmo->is_selected()) {
			Ref<Material> solid_material = get_material("lightmap_probe_solid_material", p_gizmo);
			p_gizmo->add_solid_box(solid_material, probe->get_size());
		}

		p_gizmo->add_handles(handles, get_material("handles"));
	}

	Ref<Material> icon = get_material("lightmap_probe_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}
