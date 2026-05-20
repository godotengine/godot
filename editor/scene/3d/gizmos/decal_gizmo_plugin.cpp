/**************************************************************************/
/*  decal_gizmo_plugin.cpp                                                */
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

#include "decal_gizmo_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/scene/3d/gizmos/gizmo_3d_helper.h"
#include "editor/settings/editor_settings.h"
#include "scene/3d/decal.h"

DecalGizmoPlugin::DecalGizmoPlugin() {
	helper.instantiate();
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/decal");

	create_icon_material("decal_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoDecal"), EditorStringName(EditorIcons)));

	create_material("decal_material", gizmo_color);

	create_handle_material("handles");
}

bool DecalGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<Decal>(p_spatial) != nullptr;
}

String DecalGizmoPlugin::get_gizmo_name() const {
	return "Decal";
}

int DecalGizmoPlugin::get_priority() const {
	return -1;
}

String DecalGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return helper->box_get_handle_name(p_id);
}

Variant DecalGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());
	return decal->get_size();
}

void DecalGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void DecalGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());
	Vector3 size = decal->get_size();

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	Vector3 position;
	helper->box_set_handle(sg, p_id, size, position);
	decal->set_size(size);
	decal->set_global_position(position);
}

void DecalGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	helper->box_commit_handle(TTR("Change Decal Size"), p_cancel, p_gizmo->get_node_3d());
}

void DecalGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	Decal *decal = Object::cast_to<Decal>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	if (p_gizmo->is_selected()) {
		Vector<Vector3> lines;
		Vector3 size = decal->get_size();

		AABB aabb;
		aabb.position = -size / 2;
		aabb.size = size;

		// Draw the decal's AABB with lines.
		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			aabb.get_edge(i, a, b);
			if (a.y == b.y) {
				lines.push_back(a);
				lines.push_back(b);
			} else {
				Vector3 ah = a.lerp(b, 0.2);
				lines.push_back(a);
				lines.push_back(ah);
				Vector3 bh = b.lerp(a, 0.2);
				lines.push_back(b);
				lines.push_back(bh);
			}
		}

		// Draw a directional arrow at the decal's origin.
		constexpr int arrow_points = 7;
		const float arrow_length = size.y * 0.5;

		const Vector3 arrow[arrow_points] = {
			Vector3(0, 0, -1),
			Vector3(0, 0.8, 0),
			Vector3(0, 0.3, 0),
			Vector3(0, 0.3, arrow_length),
			Vector3(0, -0.3, arrow_length),
			Vector3(0, -0.3, 0),
			Vector3(0, -0.8, 0)
		};

		constexpr int arrow_sides = 2;

		for (int i = 0; i < arrow_sides; i++) {
			for (int j = 0; j < arrow_points; j++) {
				// Rotate by 90 degrees on the X axis to match the decal orientation.
				const Basis rotation = Basis(Vector3(1, 0, 0), -Math::PI * 0.5) * Basis(Vector3(0, 0, 1), Math::PI * i / arrow_sides);
				const Basis scale = Basis::from_scale(size * 0.125);
				// Move the arrow to start at the top of the decal (when the decal points downwards).
				// This ensures the arrow is not within surface geometry, since decals are sometimes placed inside surfaces.
				const Transform3D transform = Transform3D(scale * rotation, Vector3(0, size.y * 0.5, 0));

				Vector3 v1 = arrow[j] - Vector3(0, 0, arrow_length);
				Vector3 v2 = arrow[(j + 1) % arrow_points] - Vector3(0, 0, arrow_length);

				lines.push_back(transform.xform(v1));
				lines.push_back(transform.xform(v2));
			}
		}

		Vector<Vector3> handles = helper->box_get_handles(decal->get_size());
		Ref<Material> material = get_material("decal_material", p_gizmo);

		p_gizmo->add_lines(lines, material);
		p_gizmo->add_handles(handles, get_material("handles"));
	}

	const Ref<Material> icon = get_material("decal_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}
