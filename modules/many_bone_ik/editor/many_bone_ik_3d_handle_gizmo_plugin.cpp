/**************************************************************************/
/*  many_bone_ik_3d_handle_gizmo_plugin.cpp                               */
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

#include "many_bone_ik_3d_handle_gizmo_plugin.h"

#include "modules/many_bone_ik/src/ik_kusudama_3d.h"

#include "core/io/resource_saver.h"
#include "core/math/transform_3d.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/editor_scale.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/joint_3d.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/capsule_shape_3d.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/surface_tool.h"
#include "scene/scene_string_names.h"

void ManyBoneIK3DHandleGizmoPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_gizmo_name"), &ManyBoneIK3DHandleGizmoPlugin::get_gizmo_name);
}

bool ManyBoneIK3DHandleGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return cast_to<ManyBoneIK3D>(p_spatial);
}

String ManyBoneIK3DHandleGizmoPlugin::get_gizmo_name() const {
	return "ManyBoneIK3DHandle";
}

void ManyBoneIK3DHandleGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	if (!p_gizmo) {
		return;
	}
	p_gizmo->clear();
	if (!p_gizmo->is_selected()) {
		return;
	}
	Node3D *node_3d = p_gizmo->get_node_3d();
	if (!node_3d) {
		return;
	}
	if (!node_3d->is_visible_in_tree()) {
		return;
	}
	Node *root = node_3d->get_tree()->get_edited_scene_root();
	TypedArray<Node> nodes = root->find_children("*", "ManyBoneIK3D");
	for (int32_t node_i = 0; node_i < nodes.size(); node_i++) {
		ManyBoneIK3D *many_bone_ik = cast_to<ManyBoneIK3D>(nodes[node_i]);
		if (!many_bone_ik) {
			return;
		}
		Skeleton3D *many_bone_ik_skeleton = many_bone_ik->get_skeleton();
		if (!many_bone_ik_skeleton) {
			return;
		}
		if (!many_bone_ik_skeleton->is_connected(SceneStringNames::get_singleton()->pose_updated, callable_mp(node_3d, &Node3D::update_gizmos))) {
			many_bone_ik_skeleton->connect(SceneStringNames::get_singleton()->pose_updated, callable_mp(node_3d, &Node3D::update_gizmos));
		}
		Vector<int> bones_to_process = many_bone_ik_skeleton->get_parentless_bones();
		int bones_to_process_i = 0;
		Vector<BoneId> processing_bones;
		TypedArray<IKBoneSegment3D> bone_segments = many_bone_ik->get_child_segments();
		for (int32_t segment_i = 0; segment_i < bone_segments.size(); segment_i++) {
			Ref<IKBoneSegment3D> bone_segment = bone_segments[segment_i];
			if (bone_segment.is_null()) {
				continue;
			}
			while (bones_to_process_i < bones_to_process.size()) {
				int current_bone_idx = bones_to_process[bones_to_process_i];
				processing_bones.push_back(current_bone_idx);
				Vector<int> child_bones_vector = many_bone_ik_skeleton->get_bone_children(current_bone_idx);
				for (int child_bone_idx : child_bones_vector) {
					bones_to_process.push_back(child_bone_idx);
				}
				bones_to_process_i++;
			}
			Color current_bone_color = bone_color;
			for (BoneId bone_i : bones_to_process) {
				Ref<IKBone3D> ik_bone = bone_segment->find_ik_bone(bone_i);
				if (ik_bone.is_null()) {
					continue;
				}
				if (ik_bone->is_axially_constrained()) {
					create_gizmo_handles(bone_i, ik_bone, p_gizmo, current_bone_color, many_bone_ik_skeleton, many_bone_ik);
					create_twist_gizmo_handles(bone_i, ik_bone, p_gizmo, current_bone_color, many_bone_ik_skeleton, many_bone_ik);
				}
			}
		}
	}
}

ManyBoneIK3DHandleGizmoPlugin::ManyBoneIK3DHandleGizmoPlugin() {
	create_material("lines_primary", Color(0.93725490570068, 0.19215686619282, 0.22352941334248), true, true, true);
	Ref<Texture2D> handle_center = Node3DEditor::get_singleton()->get_theme_icon(SNAME("EditorPivot"), SNAME("EditorIcons"));
	create_handle_material("handles", false, handle_center);
	Ref<Texture2D> handle_radius = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Editor3DHandle"), SNAME("EditorIcons"));
	create_handle_material("handles_radius", false, handle_radius);
	create_handle_material("handles_billboard", true);
	Ref<Texture2D> handle_axial_from = Node3DEditor::get_singleton()->get_theme_icon(SNAME("SpringArm3D"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_from", false, handle_axial_from);
	Ref<Texture2D> handle_axial_middle = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Node"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_middle", false, handle_axial_middle);
	Ref<Texture2D> handle_axial_to = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Node"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_to", false, handle_axial_to);
	Ref<Texture2D> handle_axial_current = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Node2D"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_current", false, handle_axial_current);
	kusudama_shader.instantiate();
	kusudama_shader->set_code(MANY_BONE_IKKUSUDAMA_SHADER);
}

int32_t ManyBoneIK3DHandleGizmoPlugin::get_priority() const {
	return -1;
}

void ManyBoneIK3DHandleGizmoPlugin::create_gizmo_handles(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *many_bone_ik_skeleton, ManyBoneIK3D *p_many_bone_ik) {
	// TEST PLAN: You will also want to make sure it's robust to translations of the skeleton node and root bone
	Ref<IKKusudama3D> ik_kusudama = ik_bone->get_constraint();
	if (ik_kusudama.is_null()) {
		return;
	}
	BoneId parent_idx = many_bone_ik_skeleton->get_bone_parent(current_bone_idx);
	LocalVector<int> bones;
	LocalVector<float> weights;
	bones.resize(4);
	weights.resize(4);
	for (int i = 0; i < 4; i++) {
		bones[i] = 0;
		weights[i] = 0;
	}
	bones[0] = parent_idx;
	weights[0] = 1;
	Transform3D constraint_transform = p_many_bone_ik->get_relative_transform(p_many_bone_ik->get_owner()).affine_inverse() * many_bone_ik_skeleton->get_relative_transform(many_bone_ik_skeleton->get_owner()) * p_many_bone_ik->get_godot_skeleton_transform_inverse() * ik_bone->get_constraint_orientation_transform()->get_global_transform();

	PackedFloat32Array kusudama_limit_cones;
	if (current_bone_idx >= many_bone_ik_skeleton->get_bone_count()) {
		return;
	}
	if (current_bone_idx <= -1) {
		return;
	}
	if (parent_idx >= many_bone_ik_skeleton->get_bone_count()) {
		return;
	}
	if (parent_idx <= -1) {
		return;
	}
	Vector<Vector3> center_handles;
	Vector<Vector3> radius_handles;
	const TypedArray<IKLimitCone3D> &limit_cones = ik_kusudama->get_limit_cones();
	for (int32_t cone_i = 0; cone_i < limit_cones.size(); cone_i++) {
		Ref<IKLimitCone3D> limit_cone = limit_cones[cone_i];
		Vector3 control_point = limit_cone->get_control_point();
		Vector<float> new_kusudama_limit_cones;
		new_kusudama_limit_cones.resize(4 * 3);
		new_kusudama_limit_cones.fill(0.0f);
		new_kusudama_limit_cones.write[0] = control_point.x;
		new_kusudama_limit_cones.write[1] = control_point.y;
		new_kusudama_limit_cones.write[2] = control_point.z;
		float radius = limit_cone->get_radius();
		new_kusudama_limit_cones.write[3] = radius;

		Vector3 tangent_center_1 = limit_cone->get_tangent_circle_center_next_1();
		new_kusudama_limit_cones.write[4] = tangent_center_1.x;
		new_kusudama_limit_cones.write[5] = tangent_center_1.y;
		new_kusudama_limit_cones.write[6] = tangent_center_1.z;
		float tangent_radius = limit_cone->get_tangent_circle_radius_next();
		new_kusudama_limit_cones.write[7] = tangent_radius;

		Vector3 tangent_center_2 = limit_cone->get_tangent_circle_center_next_2();
		new_kusudama_limit_cones.write[8] = tangent_center_2.x;
		new_kusudama_limit_cones.write[9] = tangent_center_2.y;
		new_kusudama_limit_cones.write[10] = tangent_center_2.z;
		new_kusudama_limit_cones.write[11] = tangent_radius;

		kusudama_limit_cones.append_array(new_kusudama_limit_cones);
	}
	Vector3 v0 = many_bone_ik_skeleton->get_bone_global_rest(current_bone_idx).origin;
	Vector3 v1 = many_bone_ik_skeleton->get_bone_global_rest(parent_idx).origin;
	real_t dist = v0.distance_to(v1);
	float radius = dist / 5.0;
	int32_t current_cone = 0;
	for (int32_t cone_i = 0; cone_i < ik_kusudama->get_limit_cones().size() * (3 * 4); cone_i = cone_i + (3 * 4)) {
		Vector3 center = Vector3(kusudama_limit_cones[cone_i + 0], kusudama_limit_cones[cone_i + 1], kusudama_limit_cones[cone_i + 2]);
		float cone_radius = kusudama_limit_cones[cone_i + 3];
		if (Math::is_zero_approx(center.length_squared())) {
			center = Vector3(0.0f, 1.0f, 0.0f);
			cone_radius = 0.0;
		}
		{
			Transform3D handle_relative_to_mesh;
			handle_relative_to_mesh.origin = center * radius;
			Transform3D handle_transform = constraint_transform * handle_relative_to_mesh;
			center_handles.push_back(handle_transform.origin);
		}
		{
			Ref<IKLimitCone3D> limit_cone = ik_kusudama->get_limit_cones()[current_cone];
			Vector3 perpendicular = limit_cone->get_tangent_circle_center_next_1();
			Vector3 maw_axis = center.cross(perpendicular);
			Quaternion maw_rotation = IKKusudama3D::quaternion_axis_angle_normalized(maw_axis, cone_radius);
			Transform3D handle_relative_to_mesh;
			handle_relative_to_mesh.origin = maw_rotation.xform(center) * radius;
			Transform3D handle_transform = constraint_transform * handle_relative_to_mesh;
			radius_handles.push_back(handle_transform.origin);
		}
		current_cone++;
	}
	if (center_handles.size()) {
		p_gizmo->add_handles(center_handles, get_material("handles"), Vector<int>(), false, true);
	}
	if (radius_handles.size()) {
		p_gizmo->add_handles(radius_handles, get_material("handles_radius"), Vector<int>(), false, true);
	}
}

void ManyBoneIK3DHandleGizmoPlugin::create_twist_gizmo_handles(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *many_bone_ik_skeleton, ManyBoneIK3D *p_many_bone_ik) {
	Ref<IKKusudama3D> ik_kusudama = ik_bone->get_constraint();
	if (ik_kusudama.is_null()) {
		return;
	}
	BoneId parent_idx = many_bone_ik_skeleton->get_bone_parent(current_bone_idx);
	LocalVector<int> bones;
	LocalVector<float> weights;
	bones.resize(4);
	weights.resize(4);
	for (int i = 0; i < 4; i++) {
		bones[i] = 0;
		weights[i] = 0;
	}
	bones[0] = parent_idx;
	weights[0] = 1;
	PackedFloat32Array kusudama_limit_cones;
	Ref<IKKusudama3D> kusudama = ik_bone->get_constraint();
	if (kusudama.is_null()) {
		return;
	}
	if (current_bone_idx >= many_bone_ik_skeleton->get_bone_count()) {
		return;
	}
	if (current_bone_idx <= -1) {
		return;
	}
	if (parent_idx >= many_bone_ik_skeleton->get_bone_count()) {
		return;
	}
	if (parent_idx <= -1) {
		return;
	}
	Vector<Vector3> axial_from_handles;
	TypedArray<Vector3> axial_middle_handles;
	TypedArray<Vector3> axial_current_handles;
	Vector<Vector3> axial_to_handles;

	Transform3D constraint_twist_transform = p_many_bone_ik->get_relative_transform(p_many_bone_ik->get_owner()).affine_inverse() * many_bone_ik_skeleton->get_relative_transform(many_bone_ik_skeleton->get_owner()) * p_many_bone_ik->get_godot_skeleton_transform_inverse() * ik_bone->get_constraint_twist_transform()->get_global_transform();
	float cone_radius = Math::deg_to_rad(90.0f);
	Vector3 v0 = many_bone_ik_skeleton->get_bone_global_rest(current_bone_idx).origin;
	Vector3 v1 = many_bone_ik_skeleton->get_bone_global_rest(parent_idx).origin;
	real_t dist = v0.distance_to(v1);
	float radius = dist / 5.0;
	float w = radius * Math::sin(cone_radius);
	float d = radius * Math::cos(cone_radius);
	{
		const float ra = (float)kusudama->get_min_axial_angle();
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Transform3D axial_from_relative_to_mesh;
		Transform3D center_relative_to_mesh;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, -d, a.y));
		Transform3D axial_transform = constraint_twist_transform * axial_from_relative_to_mesh;
		axial_from_handles.push_back((axial_transform).origin);
	}
	float start_angle = kusudama->get_min_axial_angle();
	bool negative_range = kusudama->get_range_angle() < real_t(0.0);
	if (negative_range) {
		start_angle = start_angle + kusudama->get_range_angle();
	}
	float end_angle = start_angle + Math::abs(kusudama->get_range_angle());
	float gaps = Math::deg_to_rad(15.0f);
	for (float theta = start_angle; theta < end_angle; theta += gaps) {
		const float ra = theta;
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Transform3D axial_from_relative_to_mesh;
		Transform3D center_relative_to_mesh;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, -d, a.y));
		Transform3D axial_transform = constraint_twist_transform * axial_from_relative_to_mesh;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, -d, a.y));
		axial_middle_handles.push_back((axial_transform).origin);
	}
	axial_middle_handles.pop_front();
	{
		const float ra = kusudama->get_min_axial_angle() + (float)(kusudama->get_range_angle());
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Transform3D axial_from_relative_to_mesh;
		Transform3D center_relative_to_mesh;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, -d, a.y));
		Transform3D axial_transform = constraint_twist_transform * axial_from_relative_to_mesh;
		axial_to_handles.push_back((axial_transform).origin);
	}
	if (axial_from_handles.size() && axial_to_handles.size()) {
		p_gizmo->add_handles(axial_from_handles, get_material("handles_axial_from"), Vector<int>(), true, false);
		p_gizmo->add_handles(axial_to_handles, get_material("handles_axial_to"), Vector<int>(), true, false);
	}
	if (axial_middle_handles.size()) {
		Vector<Vector3> handles;
		for (int32_t handle_i = 0; handle_i < axial_middle_handles.size(); handle_i++) {
			handles.push_back(axial_middle_handles[handle_i]);
		}
		p_gizmo->add_handles(handles, get_material("handles_axial_middle"), Vector<int>(), true, true);
	}
}
