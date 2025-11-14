/**************************************************************************/
/*  two_bone_ik_3d_gizmo_plugin.cpp                                       */
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

#include "two_bone_ik_3d_gizmo_plugin.h"

#include "editor/settings/editor_settings.h"

TwoBoneIK3DGizmoPlugin::SelectionMaterials TwoBoneIK3DGizmoPlugin::selection_materials;

TwoBoneIK3DGizmoPlugin::TwoBoneIK3DGizmoPlugin() {
	selection_materials.unselected_mat.instantiate();
	selection_materials.unselected_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	selection_materials.unselected_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	selection_materials.unselected_mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	selection_materials.unselected_mat->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	selection_materials.unselected_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);

	selection_materials.selected_mat.instantiate();
	Ref<Shader> sh;
	sh.instantiate();
	sh->set_code(R"(
// Skeleton 3D gizmo bones shader.

shader_type spatial;
render_mode unshaded, shadows_disabled;

void vertex() {
	if (!OUTPUT_IS_SRGB) {
		COLOR.rgb = mix(pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), COLOR.rgb * (1.0 / 12.92), lessThan(COLOR.rgb,vec3(0.04045)));
	}
	VERTEX = VERTEX;
	POSITION = PROJECTION_MATRIX * VIEW_MATRIX * MODEL_MATRIX * vec4(VERTEX.xyz, 1.0);
	POSITION.z = mix(POSITION.z, POSITION.w, 0.998);
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	selection_materials.selected_mat->set_shader(sh);
}

TwoBoneIK3DGizmoPlugin::~TwoBoneIK3DGizmoPlugin() {
	selection_materials.unselected_mat.unref();
	selection_materials.selected_mat.unref();
}

bool TwoBoneIK3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<TwoBoneIK3D>(p_spatial) != nullptr;
}

String TwoBoneIK3DGizmoPlugin::get_gizmo_name() const {
	return "TwoBoneIK3D";
}

int TwoBoneIK3DGizmoPlugin::get_priority() const {
	return -1;
}

void TwoBoneIK3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	TwoBoneIK3D *ik = Object::cast_to<TwoBoneIK3D>(p_gizmo->get_node_3d());
	p_gizmo->clear();

	if (!ik->get_setting_count()) {
		return;
	}

	Skeleton3D *skeleton = ik->get_skeleton();
	if (!skeleton) {
		return;
	}

	Ref<ArrayMesh> mesh = get_joints_mesh(skeleton, ik, p_gizmo->is_selected());
	Transform3D skel_tr = ik->get_global_transform().inverse() * skeleton->get_global_transform();
	p_gizmo->add_mesh(mesh, Ref<Material>(), skel_tr, skeleton->register_skin(skeleton->create_skin_from_rest_transforms()));
}

Ref<ArrayMesh> TwoBoneIK3DGizmoPlugin::get_joints_mesh(Skeleton3D *p_skeleton, TwoBoneIK3D *p_ik, bool p_is_selected) {
	Color bone_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/ik_chain");

	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_LINES);

	if (p_is_selected) {
		surface_tool->set_material(selection_materials.selected_mat);
	} else {
		selection_materials.unselected_mat->set_albedo(bone_color);
		surface_tool->set_material(selection_materials.unselected_mat);
	}

	PackedInt32Array bones;
	PackedFloat32Array weights;
	bones.resize(4);
	weights.resize(4);
	for (int i = 0; i < 4; i++) {
		bones.write[i] = 0;
		weights.write[i] = 0;
	}
	weights.write[0] = 1;

	for (int i = 0; i < p_ik->get_setting_count(); i++) {
		if (!p_ik->is_valid(i)) {
			continue; // Skip invalid settings.
		}

		int root_bone = p_ik->get_root_bone(i);
		int middle_bone = p_ik->get_middle_bone(i);

		Transform3D root_gp = p_skeleton->get_bone_global_rest(root_bone);
		Transform3D mid_gp = p_skeleton->get_bone_global_rest(middle_bone);
		Vector3 root_vec = p_ik->get_root_bone_vector(i);
		Vector3 mid_vec = p_ik->get_middle_bone_vector(i);

		bones.write[0] = root_bone;
		surface_tool->set_bones(bones);
		surface_tool->set_weights(weights);
		draw_line(surface_tool, root_gp.origin, root_gp.translated_local(root_vec).origin, bone_color);

		bones.write[0] = middle_bone;
		surface_tool->set_bones(bones);
		surface_tool->set_weights(weights);
		draw_line(surface_tool, mid_gp.origin, mid_gp.translated_local(mid_vec).origin, bone_color);

		Vector3 pole_vector = p_ik->get_pole_direction_vector(i);
		if (pole_vector.is_zero_approx()) {
			continue;
		}

		float pole_length = MIN(root_vec.length(), mid_vec.length()) * 0.25;
		draw_arrow(surface_tool, mid_gp.origin, mid_gp.basis.get_rotation_quaternion().xform(pole_vector).normalized(), pole_length, bone_color);
	}

	return surface_tool->commit();
}

void TwoBoneIK3DGizmoPlugin::draw_line(Ref<SurfaceTool> &p_surface_tool, const Vector3 &p_begin_pos, const Vector3 &p_end_pos, const Color &p_color) {
	p_surface_tool->set_color(p_color);
	p_surface_tool->add_vertex(p_begin_pos);
	p_surface_tool->set_color(p_color);
	p_surface_tool->add_vertex(p_end_pos);
}

void TwoBoneIK3DGizmoPlugin::draw_arrow(Ref<SurfaceTool> &p_surface_tool, const Vector3 &p_origin, const Vector3 &p_direction, real_t p_length, const Color &p_color) {
	static const float HALF_PI = Math::PI * 0.5;

	p_surface_tool->set_color(p_color);

	LocalVector<Vector3> arrow_body;
	arrow_body.resize(2);
	arrow_body[0] = Vector3(0, 0, 0);
	arrow_body[1] = Vector3(0, p_length, 0);

	LocalVector<Vector3> arrow_head;
	arrow_head.resize(2);
	arrow_head[0] = Vector3(0, p_length, 0);
	arrow_head[1] = Vector3(p_length * 0.25, p_length * 0.75, 0);

	Quaternion dir = Quaternion(Vector3(0, 1, 0), p_direction);

	// Draw the arrow body.
	p_surface_tool->add_vertex(arrow_body[0] + p_origin);
	p_surface_tool->add_vertex(dir.xform(arrow_body[1]) + p_origin);

	// Draw the arrow head 4 times, rotate around the arrow body.
	for (int i = 0; i < 4; i++) {
		Quaternion rotation = dir * Quaternion(Vector3(0, 1, 0), HALF_PI * i);
		for (int j = 0; j < 2; j++) {
			Vector3 v = arrow_head[j];
			Vector3 rotated_v = rotation.xform(v);
			p_surface_tool->add_vertex(rotated_v + p_origin);
		}
	}
}
