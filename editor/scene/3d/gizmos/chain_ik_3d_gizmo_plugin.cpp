/**************************************************************************/
/*  chain_ik_3d_gizmo_plugin.cpp                                          */
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

#include "chain_ik_3d_gizmo_plugin.h"

#include "editor/settings/editor_settings.h"

ChainIK3DGizmoPlugin::SelectionMaterials ChainIK3DGizmoPlugin::selection_materials;

ChainIK3DGizmoPlugin::ChainIK3DGizmoPlugin() {
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

ChainIK3DGizmoPlugin::~ChainIK3DGizmoPlugin() {
	selection_materials.unselected_mat.unref();
	selection_materials.selected_mat.unref();
}

bool ChainIK3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<ChainIK3D>(p_spatial) != nullptr;
}

String ChainIK3DGizmoPlugin::get_gizmo_name() const {
	return "ChainIK3D";
}

int ChainIK3DGizmoPlugin::get_priority() const {
	return -1;
}

void ChainIK3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	ChainIK3D *ik = Object::cast_to<ChainIK3D>(p_gizmo->get_node_3d());
	p_gizmo->clear();

	if (!ik->get_setting_count()) {
		return;
	}

	Skeleton3D *skeleton = ik->get_skeleton();
	if (!skeleton) {
		return;
	}

	Ref<ArrayMesh> skeleton_mesh;
	Ref<ArrayMesh> mesh;
	get_joints_mesh(skeleton, ik, p_gizmo->is_selected(), skeleton_mesh, mesh);
	Transform3D skel_tr = ik->get_global_transform().inverse() * skeleton->get_global_transform();
	p_gizmo->add_mesh(skeleton_mesh, Ref<Material>(), skel_tr, skeleton->register_skin(skeleton->create_skin_from_rest_transforms()));
	p_gizmo->add_mesh(mesh, Ref<Material>(), skel_tr);
}

void ChainIK3DGizmoPlugin::get_joints_mesh(Skeleton3D *p_skeleton, ChainIK3D *p_ik, bool p_is_selected, Ref<ArrayMesh> &r_skinned_mesh, Ref<ArrayMesh> &r_mesh) {
	Color bone_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/ik_chain");
	static const Color limitation_x_axis_color = Color(1, 0, 0, 1);
	static const Color limitation_z_axis_color = Color(0, 0, 1, 1);

	IterateIK3D *it_ik = Object::cast_to<IterateIK3D>(p_ik);

	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_LINES);

	Ref<SurfaceTool> surface_tool_without_skin;
	surface_tool_without_skin.instantiate();
	surface_tool_without_skin->begin(Mesh::PRIMITIVE_LINES);

	if (p_is_selected) {
		surface_tool->set_material(selection_materials.selected_mat);
		surface_tool_without_skin->set_material(selection_materials.selected_mat);
	} else {
		selection_materials.unselected_mat->set_albedo(bone_color);
		surface_tool->set_material(selection_materials.unselected_mat);
		surface_tool_without_skin->set_material(selection_materials.unselected_mat);
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
		int current_bone = -1;
		int prev_bone = -1;
		int joint_end = p_ik->get_joint_count(i) - 1;
		float prev_length = INFINITY;
		bool is_extended = p_ik->is_end_bone_extended(i) && p_ik->get_end_bone_length(i) > 0;
		Transform3D anc_global_pose = p_ik->get_chain_root_global_rest(i);
		for (int j = 0; j <= joint_end; j++) {
			current_bone = p_ik->get_joint_bone(i, j);
			if (j > 0) {
				int prev_joint = j - 1;
				Transform3D parent_global_pose = p_skeleton->get_bone_global_rest(prev_bone);
				Vector3 bone_vector = p_ik->get_bone_vector(i, prev_joint);
				float current_length = bone_vector.length();
				Vector3 center = parent_global_pose.translated_local(bone_vector).origin;
				draw_line(surface_tool, parent_global_pose.origin, center, bone_color);

				if (it_ik) {
					// Draw rotation axis vector if not ROTATION_AXIS_ALL.
					if (j != joint_end || (j == joint_end && is_extended)) {
						SkeletonModifier3D::RotationAxis rotation_axis = it_ik->get_joint_rotation_axis(i, j);
						if (rotation_axis != SkeletonModifier3D::ROTATION_AXIS_ALL) {
							Vector3 axis_vector = it_ik->get_joint_rotation_axis_vector(i, j);
							if (!axis_vector.is_zero_approx()) {
								float rot_axis_length = bone_vector.length() * 0.2; // Use 20% of the bone length for the rotation axis vector.
								Vector3 axis = parent_global_pose.basis.xform(axis_vector.normalized()) * rot_axis_length;
								draw_line(surface_tool, center - axis, center + axis, bone_color);
							}
						}
					}

					// Draw parent limitation shape.
					Ref<JointLimitation3D> lim = it_ik->get_joint_limitation(i, prev_joint);
					if (lim.is_valid() && prev_bone >= 0) {
						// Limitation space should bind parent bone rest.
						int parent = p_skeleton->get_bone_parent(prev_bone);
						Ref<SurfaceTool> limitation_surface_tool = parent >= 0 ? surface_tool : surface_tool_without_skin;
						if (parent >= 0) {
							bones.write[0] = parent;
							limitation_surface_tool->set_bones(bones);
							limitation_surface_tool->set_weights(weights);
						}
						Transform3D tr = anc_global_pose;
						tr.basis *= it_ik->get_joint_limitation_space(i, prev_joint, bone_vector.normalized());
						float sl = MIN(current_length, prev_length);
						lim->draw_shape(limitation_surface_tool, tr, sl, bone_color);
						sl *= 0.1;
						Vector3 x_axis = tr.basis.get_column(Vector3::AXIS_X).normalized() * sl;
						Vector3 z_axis = tr.basis.get_column(Vector3::AXIS_Z).normalized() * sl;
						draw_line(limitation_surface_tool, tr.origin + x_axis * 2, tr.origin + x_axis * 3, limitation_x_axis_color); // Offset 20%.
						draw_line(limitation_surface_tool, tr.origin + z_axis * 2, tr.origin + z_axis * 3, limitation_z_axis_color); // Offset 20%.
					}
				}
				prev_length = current_length;
				Transform3D tr = p_skeleton->get_bone_rest(current_bone);
				tr.origin = bone_vector;
				parent_global_pose *= tr;
				anc_global_pose = parent_global_pose;
			}
			if (j == joint_end && is_extended) {
				Transform3D current_global_pose = p_skeleton->get_bone_global_rest(current_bone);
				Vector3 bone_vector = p_ik->get_bone_vector(i, j);
				if (bone_vector.is_zero_approx()) {
					continue;
				}
				float current_length = bone_vector.length();
				bones.write[0] = current_bone;
				surface_tool->set_bones(bones);
				surface_tool->set_weights(weights);
				Vector3 center = current_global_pose.translated_local(bone_vector).origin;
				draw_line(surface_tool, current_global_pose.origin, center, bone_color);

				if (it_ik) {
					// Draw limitation shape.
					Ref<JointLimitation3D> lim = it_ik->get_joint_limitation(i, j);
					if (lim.is_valid() && current_bone >= 0) {
						// Limitation space should bind parent bone rest.
						int parent = p_skeleton->get_bone_parent(current_bone);
						Ref<SurfaceTool> limitation_surface_tool = parent >= 0 ? surface_tool : surface_tool_without_skin;
						if (parent >= 0) {
							bones.write[0] = parent;
							limitation_surface_tool->set_bones(bones);
							limitation_surface_tool->set_weights(weights);
						}
						Transform3D tr = anc_global_pose;
						tr.basis *= it_ik->get_joint_limitation_space(i, j, bone_vector.normalized());
						float sl = MIN(current_length, prev_length);
						lim->draw_shape(limitation_surface_tool, tr, sl, bone_color);
						sl *= 0.1;
						Vector3 x_axis = tr.basis.get_column(Vector3::AXIS_X).normalized() * sl;
						Vector3 z_axis = tr.basis.get_column(Vector3::AXIS_Z).normalized() * sl;
						draw_line(limitation_surface_tool, tr.origin + x_axis * 2, tr.origin + x_axis * 3, limitation_x_axis_color); // Offset 20%.
						draw_line(limitation_surface_tool, tr.origin + z_axis * 2, tr.origin + z_axis * 3, limitation_z_axis_color); // Offset 20%.
					}
				}
			} else {
				bones.write[0] = current_bone;
				surface_tool->set_bones(bones);
				surface_tool->set_weights(weights);
				if (j == 0) {
					// Check if the next bone exists.
					int count = p_ik->get_joint_count(i);
					if (count < 2) {
						continue;
					}
					if (it_ik) {
						// Draw rotation axis vector if not ROTATION_AXIS_ALL.
						SkeletonModifier3D::RotationAxis rotation_axis = it_ik->get_joint_rotation_axis(i, j);
						if (rotation_axis != SkeletonModifier3D::ROTATION_AXIS_ALL) {
							Vector3 axis_vector = it_ik->get_joint_rotation_axis_vector(i, j);
							if (!axis_vector.is_zero_approx()) {
								Vector3 bone_vector = p_ik->get_bone_vector(i, j);
								float rot_axis_length = bone_vector.length() * 0.2; // Use 20% of the bone length for the rotation axis vector.
								Vector3 axis = anc_global_pose.basis.xform(axis_vector.normalized()) * rot_axis_length;
								draw_line(surface_tool, anc_global_pose.origin - axis, anc_global_pose.origin + axis, bone_color);
							}
						}
					}
				}
			}
			prev_bone = current_bone;
		}
	}

	r_skinned_mesh = surface_tool->commit();
	r_mesh = surface_tool_without_skin->commit();
}

void ChainIK3DGizmoPlugin::draw_line(Ref<SurfaceTool> &p_surface_tool, const Vector3 &p_begin_pos, const Vector3 &p_end_pos, const Color &p_color) {
	p_surface_tool->set_color(p_color);
	p_surface_tool->add_vertex(p_begin_pos);
	p_surface_tool->set_color(p_color);
	p_surface_tool->add_vertex(p_end_pos);
}
