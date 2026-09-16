/**************************************************************************/
/*  foliage_spawner_3d_gizmo_plugin.cpp                                   */
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

#include "foliage_spawner_3d_gizmo_plugin.h"

#include "editor/scene/3d/gizmos/gizmo_3d_helper.h"
#include "editor/settings/editor_settings.h"
#include "scene/3d/foliage_spawner_3d.h"

FoliageSpawner3DGizmoPlugin::FoliageSpawner3DGizmoPlugin() {
	helper.instantiate();
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/foliage_spawner");
	create_material("shape_material", gizmo_color);
	gizmo_color.a = 0.15;
	create_material("shape_material_internal", gizmo_color);

	create_handle_material("handles");
}

bool FoliageSpawner3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return (Object::cast_to<FoliageSpawner3D>(p_spatial) != nullptr);
}

String FoliageSpawner3DGizmoPlugin::get_gizmo_name() const {
	return "FoliageSpawner3D";
}

int FoliageSpawner3DGizmoPlugin::get_priority() const {
	return -1;
}

String FoliageSpawner3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return helper->box_get_handle_name(p_id);
}

Variant FoliageSpawner3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return Vector3(p_gizmo->get_node_3d()->call("get_volume_size"));
}

void FoliageSpawner3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void FoliageSpawner3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	FoliageSpawner3D *spawner = Object::cast_to<FoliageSpawner3D>(p_gizmo->get_node_3d());
	Vector3 size = spawner->get_volume_size();

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	Vector3 position;
	helper->box_set_handle(sg, p_id, size, position);
	spawner->set_volume_size(size);
	spawner->set_global_position(position);
}

void FoliageSpawner3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	helper->box_commit_handle(TTR("Change Foliage Spawner Volume Size"), p_cancel, p_gizmo->get_node_3d(), nullptr, "global_position", "volume_size");
}

void FoliageSpawner3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	FoliageSpawner3D *spawner = Object::cast_to<FoliageSpawner3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	const Ref<Material> material = get_material("shape_material", p_gizmo);
	Ref<Material> handles_material = get_material("handles");

	Vector<Vector3> lines;
	AABB aabb;
	aabb.size = spawner->get_volume_size();
	aabb.position = aabb.size / -2;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	Vector<Vector3> handles = helper->box_get_handles(spawner->get_volume_size());

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);
	p_gizmo->add_handles(handles, handles_material);
}
