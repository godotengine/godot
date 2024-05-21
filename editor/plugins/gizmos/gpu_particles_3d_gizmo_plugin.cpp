/**************************************************************************/
/*  gpu_particles_3d_gizmo_plugin.cpp                                     */
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

#include "gpu_particles_3d_gizmo_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/gpu_particles_3d.h"

GPUParticles3DGizmoPlugin::GPUParticles3DGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/particles", Color(0.8, 0.7, 0.4));
	create_material("particles_material", gizmo_color);
	gizmo_color.a = MAX((gizmo_color.a - 0.2) * 0.02, 0.0);
	create_icon_material("particles_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoGPUParticles3D"), EditorStringName(EditorIcons)));
}

bool GPUParticles3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<GPUParticles3D>(p_spatial) != nullptr;
}

String GPUParticles3DGizmoPlugin::get_gizmo_name() const {
	return "GPUParticles3D";
}

int GPUParticles3DGizmoPlugin::get_priority() const {
	return -1;
}

bool GPUParticles3DGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

void GPUParticles3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	if (p_gizmo->is_selected()) {
		GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(p_gizmo->get_node_3d());

		Vector<Vector3> lines;
		AABB aabb = particles->get_visibility_aabb();

		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			aabb.get_edge(i, a, b);
			lines.push_back(a);
			lines.push_back(b);
		}

		Ref<Material> material = get_material("particles_material", p_gizmo);

		p_gizmo->add_lines(lines, material);
	}

	Ref<Material> icon = get_material("particles_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}
