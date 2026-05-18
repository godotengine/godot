/**************************************************************************/
/*  marker_3d_gizmo_plugin.cpp                                            */
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

#include "marker_3d_gizmo_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "scene/3d/marker_3d.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/texture.h"
#include "servers/rendering/rendering_server_enums.h"

Marker3DGizmoPlugin::Marker3DGizmoPlugin() {
	pos3d_mesh.instantiate();

	Vector<Vector3> cursor_points;
	Vector<Color> cursor_colors;
	const float cs = 1.0;
	// Add more points to create a "hard stop" in the color gradient.
	cursor_points.push_back(Vector3(+cs, 0, 0));
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3(-cs, 0, 0));

	cursor_points.push_back(Vector3(0, +cs, 0));
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3(0, -cs, 0));

	cursor_points.push_back(Vector3(0, 0, +cs));
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3(0, 0, -cs));

	// Use the axis color which is brighter for the positive axis.
	// Use a darkened axis color for the negative axis.
	// This makes it possible to see in which direction the Marker3D node is rotated
	// (which can be important depending on how it's used).
	const Color color_x = EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("axis_x_color"), EditorStringName(Editor));
	cursor_colors.push_back(color_x);
	cursor_colors.push_back(color_x);
	cursor_colors.push_back(color_x.darkened(0.5));
	cursor_colors.push_back(color_x.darkened(0.5));

	const Color color_y = EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("axis_y_color"), EditorStringName(Editor));
	cursor_colors.push_back(color_y);
	cursor_colors.push_back(color_y);
	cursor_colors.push_back(color_y.darkened(0.5));
	cursor_colors.push_back(color_y.darkened(0.5));

	const Color color_z = EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("axis_z_color"), EditorStringName(Editor));
	cursor_colors.push_back(color_z);
	cursor_colors.push_back(color_z);
	cursor_colors.push_back(color_z.darkened(0.5));
	cursor_colors.push_back(color_z.darkened(0.5));

	Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mat->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

	Array d;
	d.resize(RSE::ARRAY_MAX);
	d[Mesh::ARRAY_VERTEX] = cursor_points;
	d[Mesh::ARRAY_COLOR] = cursor_colors;
	pos3d_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, d);
	pos3d_mesh->surface_set_material(0, mat);

	//Texture mesh if used
	tex3d_mesh.instantiate();
	Vector<Vector3> tex3d_points;
	Vector<Vector2> uv = {
		Vector2(0, 0), Vector2(1, 0), Vector2(1, 1),
		Vector2(0, 0), Vector2(1, 1), Vector2(0, 1)
	};
	tex3d_points.push_back(Vector3(-cs, cs, 0));
	tex3d_points.push_back(Vector3(cs, cs, 0));
	tex3d_points.push_back(Vector3(cs, -cs, 0));

	tex3d_points.push_back(Vector3(-cs, cs, 0));
	tex3d_points.push_back(Vector3(cs, -cs, 0));
	tex3d_points.push_back(Vector3(-cs, -cs, 0));

	Array d2;
	d2.resize(RSE::ARRAY_MAX);
	d2[Mesh::ARRAY_VERTEX] = tex3d_points;
	d2[Mesh::ARRAY_TEX_UV] = uv;
	tex3d_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, d2);
}

bool Marker3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<Marker3D>(p_spatial) != nullptr;
}

String Marker3DGizmoPlugin::get_gizmo_name() const {
	return "Marker3D";
}

int Marker3DGizmoPlugin::get_priority() const {
	return -1;
}

void Marker3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	const Marker3D *marker = Object::cast_to<Marker3D>(p_gizmo->get_node_3d());
	const real_t extents = marker->get_gizmo_extents();

	p_gizmo->clear();

	Ref<Texture2D> tex = marker->get_gizmo_texture();
	const Transform3D xform(Basis::from_scale(Vector3(extents, extents, extents)));
	if (tex.is_valid()) {
		ObjectID tex_id = tex->get_instance_id();
		if (!tex3d_materials.has(tex_id)) {
			Ref<StandardMaterial3D> m = memnew(StandardMaterial3D);
			m->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
			m->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
			m->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			m->set_billboard_mode(StandardMaterial3D::BILLBOARD_ENABLED);
			m->set_flag(StandardMaterial3D::FLAG_BILLBOARD_KEEP_SCALE, true);
			m->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, tex);
			tex3d_materials[tex_id] = m;
		}
		p_gizmo->add_mesh(tex3d_mesh, tex3d_materials[tex_id], xform);
	} else {
		p_gizmo->add_mesh(pos3d_mesh, Ref<Material>(), xform);
	}

	const Vector<Vector3> points = {
		Vector3(-extents, 0, 0),
		Vector3(+extents, 0, 0),
		Vector3(0, -extents, 0),
		Vector3(0, +extents, 0),
		Vector3(0, 0, -extents),
		Vector3(0, 0, +extents),
	};
	p_gizmo->add_collision_segments(points);
}
