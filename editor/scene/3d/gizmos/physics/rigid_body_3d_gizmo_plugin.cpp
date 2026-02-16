/**************************************************************************/
/*  rigid_body_3d_gizmo_plugin.cpp                                        */
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

#include "rigid_body_3d_gizmo_plugin.h"

#include "scene/3d/physics/rigid_body_3d.h"

RigidBody3DGizmoPlugin::RigidBody3DGizmoPlugin() {
	// Materials for center of mass crosshair.
	create_material("center_of_mass_material_custom", Color(1.0, 0.0, 1.0)); // Magenta
	create_material("center_of_mass_material_auto", Color(1.0, 0.6, 0.0)); // Orange
}

bool RigidBody3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<RigidBody3D>(p_spatial) != nullptr;
}

String RigidBody3DGizmoPlugin::get_gizmo_name() const {
	return "RigidBody3D";
}

int RigidBody3DGizmoPlugin::get_priority() const {
	return -1;
}

void RigidBody3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	RigidBody3D *rigid_body = Object::cast_to<RigidBody3D>(p_gizmo->get_node_3d());
	if (!rigid_body) {
		return;
	}

	p_gizmo->clear();

	if (!p_gizmo->is_selected()) {
		return;
	}

	const Vector3 center_of_mass = rigid_body->get_center_of_mass();

	// Length of the crosshair lines.
	constexpr float extents = 0.1;

	// Draw a crosshair at the center of mass position.
	Vector<Vector3> lines;
	// X line.
	lines.push_back(center_of_mass + Vector3(-extents, 0, 0));
	lines.push_back(center_of_mass + Vector3(+extents, 0, 0));
	// Y line.
	lines.push_back(center_of_mass + Vector3(0, -extents, 0));
	lines.push_back(center_of_mass + Vector3(0, +extents, 0));
	// Z line.
	lines.push_back(center_of_mass + Vector3(0, 0, -extents));
	lines.push_back(center_of_mass + Vector3(0, 0, +extents));

	// Color of the center of mass indicator.
	String mat_name = (rigid_body->get_center_of_mass_mode() == RigidBody3D::CENTER_OF_MASS_MODE_CUSTOM)
			? "center_of_mass_material_custom"
			: "center_of_mass_material_auto";  

	Ref<Material> material = get_material(mat_name, p_gizmo);
	p_gizmo->add_lines(lines, material);
}
