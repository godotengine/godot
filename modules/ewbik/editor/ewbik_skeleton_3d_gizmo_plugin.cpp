/*************************************************************************/
/*  ewbik_skeleton_3d_gizmo_plugin.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "ewbik_skeleton_3d_gizmo_plugin.h"

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

#include "../src/ik_kusudama.h"

bool EWBIK3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return cast_to<Skeleton3D>(p_spatial);
}

String EWBIK3DGizmoPlugin::get_gizmo_name() const {
	return "Bone Constraints";
}

void EWBIK3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	if (!p_gizmo) {
		return;
	}
	Node3D *node_3d = p_gizmo->get_node_3d();
	if (!node_3d) {
		return;
	}
	Node *owner_node = node_3d->get_owner();
	if (!owner_node) {
		return;
	}
	TypedArray<Node> nodes = owner_node->find_children("*", "SkeletonModification3DNBoneIK");
	p_gizmo->clear();
	for (int32_t node_i = 0; node_i < nodes.size(); node_i++) {
		SkeletonModification3DNBoneIK *ewbik = cast_to<SkeletonModification3DNBoneIK>(nodes[node_i]);
		if (!ewbik) {
			continue;
		}
		Skeleton3D *ewbik_skeleton = ewbik->get_skeleton();
		if (!ewbik_skeleton) {
			continue;
		}
		if (cast_to<Skeleton3D>(node_3d) != ewbik_skeleton) {
			continue;
		}
		Vector<int> bones_to_process = ewbik_skeleton->get_parentless_bones();
		kusudama_shader.instantiate();
		kusudama_shader->set_code(EWBIK_KUSUDAMA_SHADER);
		int bones_to_process_i = 0;
		Vector<BoneId> processing_bones;
		Ref<IKBoneSegment> bone_segment = ewbik->get_segmented_skeleton();
		if (bone_segment.is_null()) {
			return;
		}
		while (bones_to_process_i < bones_to_process.size()) {
			int current_bone_idx = bones_to_process[bones_to_process_i];
			processing_bones.push_back(current_bone_idx);
			Vector<int> child_bones_vector = ewbik_skeleton->get_bone_children(current_bone_idx);
			for (int child_bone_idx : child_bones_vector) {
				bones_to_process.push_back(child_bone_idx);
			}
			bones_to_process_i++;
		}
		Color current_bone_color = bone_color;
		for (BoneId current_bone_idx : processing_bones) {
			Ref<IKBone3D> ik_bone = bone_segment->get_ik_bone(current_bone_idx);
			if (ik_bone.is_null() || ik_bone->get_bone_id() != current_bone_idx) {
				continue;
			}
			create_gizmo_mesh(current_bone_idx, ik_bone, p_gizmo, current_bone_color, ewbik_skeleton);
			create_gizmo_handles(current_bone_idx, ik_bone, p_gizmo, current_bone_color, ewbik_skeleton);
		}
	}
}

void EWBIK3DGizmoPlugin::create_gizmo_mesh(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *ewbik_skeleton) {
	Ref<IKKusudama> ik_kusudama = ik_bone->get_constraint();
	if (ik_kusudama.is_null()) {
		return;
	}
	BoneId parent_idx = ewbik_skeleton->get_bone_parent(current_bone_idx);
	Vector<Vector3> handles;
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
	Transform3D constraint_relative_to_the_skeleton = ewbik_skeleton->get_transform().affine_inverse() * ik_bone->get_constraint_transform()->get_global_transform();
	Transform3D constraint_relative_to_the_universe = ewbik_skeleton->get_global_transform() * constraint_relative_to_the_skeleton;
	PackedFloat32Array kusudama_limit_cones;
	Ref<IKKusudama> kusudama = ik_bone->get_constraint();
	kusudama_limit_cones.resize(KUSUDAMA_MAX_CONES * 4);
	kusudama_limit_cones.fill(0.0f);
	int out_idx = 0;
	const TypedArray<IKLimitCone> &limit_cones = ik_kusudama->get_limit_cones();
	for (int32_t cone_i = 0; cone_i < limit_cones.size(); cone_i++) {
		Ref<IKLimitCone> limit_cone = limit_cones[cone_i];
		Vector3 control_point = limit_cone->get_control_point();
		kusudama_limit_cones.write[out_idx + 0] = control_point.x;
		kusudama_limit_cones.write[out_idx + 1] = control_point.y;
		kusudama_limit_cones.write[out_idx + 2] = control_point.z;
		float radius = limit_cone->get_radius();
		kusudama_limit_cones.write[out_idx + 3] = radius;
		out_idx += 4;

		Vector3 tangent_center_1 = limit_cone->get_tangent_circle_center_next_1();
		kusudama_limit_cones.write[out_idx + 0] = tangent_center_1.x;
		kusudama_limit_cones.write[out_idx + 1] = tangent_center_1.y;
		kusudama_limit_cones.write[out_idx + 2] = tangent_center_1.z;
		float tangent_radius = limit_cone->get_tangent_circle_radius_next();
		kusudama_limit_cones.write[out_idx + 3] = tangent_radius;
		out_idx += 4;

		Vector3 tangent_center_2 = limit_cone->get_tangent_circle_center_next_2();
		kusudama_limit_cones.write[out_idx + 0] = tangent_center_2.x;
		kusudama_limit_cones.write[out_idx + 1] = tangent_center_2.y;
		kusudama_limit_cones.write[out_idx + 2] = tangent_center_2.z;
		kusudama_limit_cones.write[out_idx + 3] = tangent_radius;
		out_idx += 4;
	}
	Vector3 v0 = ewbik_skeleton->get_bone_global_rest(current_bone_idx).origin;
	Vector3 v1 = ewbik_skeleton->get_bone_global_rest(parent_idx).origin;
	real_t dist = v0.distance_to(v1);
	float radius = dist / 5.0;
	// Code copied from the SphereMesh.
	float height = dist / 2.5;
	int rings = 32;

	int i, j, prevrow, thisrow, point;
	float x, y, z;

	float scale = height * 0.5;

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<int> indices;
	point = 0;

	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		int radial_segments = 32;
		float v = j;
		float w;

		v /= (rings + 1);
		w = sin(Math_PI * v);
		y = scale * cos(Math_PI * v);

		for (i = 0; i <= radial_segments; i++) {
			float u = i;
			u /= radial_segments;

			x = sin(u * Math_TAU);
			z = cos(u * Math_TAU);

			Vector3 p = Vector3(x * scale * w, y, z * scale * w);
			points.push_back(p);
			Vector3 normal = Vector3(x * w * scale, radius * (y / scale), z * w * scale);
			normals.push_back(normal.normalized());
			point++;

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			};
		};

		prevrow = thisrow;
		thisrow = point;
	}
	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	const int32_t MESH_CUSTOM_0 = 0;
	surface_tool->set_custom_format(MESH_CUSTOM_0, SurfaceTool::CustomFormat::CUSTOM_RGBA_HALF);
	for (int32_t point_i = 0; point_i < points.size(); point_i++) {
		surface_tool->set_bones(bones);
		surface_tool->set_weights(weights);
		Color c;
		c.r = normals[point_i].x;
		c.g = normals[point_i].y;
		c.b = normals[point_i].z;
		c.a = 0;
		surface_tool->set_custom(MESH_CUSTOM_0, c);
		surface_tool->set_normal(normals[point_i]);
		surface_tool->add_vertex(points[point_i]);
	}
	for (int32_t index_i : indices) {
		surface_tool->add_index(index_i);
	}
	Ref<ShaderMaterial> kusudama_material;
	kusudama_material.instantiate();
	kusudama_material->set_shader(kusudama_shader);
	kusudama_material->set_shader_parameter("cone_sequence", kusudama_limit_cones);
	int32_t cone_count = kusudama->get_limit_cones().size();
	kusudama_material->set_shader_parameter("cone_count", cone_count);
	kusudama_material->set_shader_parameter("kusudama_color", current_bone_color);
	p_gizmo->add_mesh(
			surface_tool->commit(Ref<Mesh>(), RS::ARRAY_CUSTOM_RGBA_HALF << RS::ARRAY_FORMAT_CUSTOM0_SHIFT),
			kusudama_material, constraint_relative_to_the_universe);
}

EWBIK3DGizmoPlugin::EWBIK3DGizmoPlugin() {
	// Enable vertex colors for the materials below as the gizmo color depends on the light color.
	create_material("lines_primary", Color(0.93725490570068, 0.19215686619282, 0.22352941334248), true, true, true);
	// Need a textured2d handle for yellow dot, blue dot and turqouise dot and be icons.
	Ref<Texture2D> handle_center = Node3DEditor::get_singleton()->get_theme_icon(SNAME("EditorPivot"), SNAME("EditorIcons"));
	create_handle_material("handles", false, handle_center);
	Ref<Texture2D> handle_radius = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Editor3DHandle"), SNAME("EditorIcons"));
	create_handle_material("handles_radius", false, handle_radius);
	create_handle_material("handles_billboard", true);
	Ref<Texture2D> handle_axial_from = Node3DEditor::get_singleton()->get_theme_icon(SNAME("SpringArm3D"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_from", false, handle_axial_from);
	Ref<Texture2D> handle_axial_middle = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Node"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_middle", false, handle_axial_middle);
	Ref<Texture2D> handle_axial_to = Node3DEditor::get_singleton()->get_theme_icon(SNAME("Node3D"), SNAME("EditorIcons"));
	create_handle_material("handles_axial_to", false, handle_axial_to);
}

void EWBIK3DGizmoPlugin::create_gizmo_handles(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *ewbik_skeleton) {
	Ref<IKKusudama> ik_kusudama = ik_bone->get_constraint();
	if (ik_kusudama.is_null()) {
		return;
	}
	BoneId parent_idx = ewbik_skeleton->get_bone_parent(current_bone_idx);
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
	Transform3D constraint_relative_to_the_skeleton = ewbik_skeleton->get_transform().affine_inverse() * ik_bone->get_constraint_transform()->get_global_transform();
	Transform3D constraint_relative_to_the_universe = ewbik_skeleton->get_global_transform() * constraint_relative_to_the_skeleton;
	PackedFloat32Array kusudama_limit_cones;
	Ref<IKKusudama> kusudama = ik_bone->get_constraint();
	kusudama_limit_cones.resize(KUSUDAMA_MAX_CONES * 4);
	kusudama_limit_cones.fill(0.0f);
	int out_idx = 0;
	const TypedArray<IKLimitCone> &limit_cones = ik_kusudama->get_limit_cones();
	for (int32_t cone_i = 0; cone_i < limit_cones.size(); cone_i++) {
		Ref<IKLimitCone> limit_cone = limit_cones[cone_i];
		Vector3 control_point = limit_cone->get_control_point();
		kusudama_limit_cones.write[out_idx + 0] = control_point.x;
		kusudama_limit_cones.write[out_idx + 1] = control_point.y;
		kusudama_limit_cones.write[out_idx + 2] = control_point.z;
		float radius = limit_cone->get_radius();
		kusudama_limit_cones.write[out_idx + 3] = radius;
		out_idx += 4;

		Vector3 tangent_center_1 = limit_cone->get_tangent_circle_center_next_1();
		kusudama_limit_cones.write[out_idx + 0] = tangent_center_1.x;
		kusudama_limit_cones.write[out_idx + 1] = tangent_center_1.y;
		kusudama_limit_cones.write[out_idx + 2] = tangent_center_1.z;
		float tangent_radius = limit_cone->get_tangent_circle_radius_next();
		kusudama_limit_cones.write[out_idx + 3] = tangent_radius;
		out_idx += 4;

		Vector3 tangent_center_2 = limit_cone->get_tangent_circle_center_next_2();
		kusudama_limit_cones.write[out_idx + 0] = tangent_center_2.x;
		kusudama_limit_cones.write[out_idx + 1] = tangent_center_2.y;
		kusudama_limit_cones.write[out_idx + 2] = tangent_center_2.z;
		kusudama_limit_cones.write[out_idx + 3] = tangent_radius;
		out_idx += 4;
	}
	Vector3 v0 = ewbik_skeleton->get_bone_global_rest(current_bone_idx).origin;
	Vector3 v1 = ewbik_skeleton->get_bone_global_rest(parent_idx).origin;
	real_t dist = v0.distance_to(v1);
	float radius = dist / 5.0;
	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_LINES);
	Vector<Vector3> center_handles;
	Vector<Vector3> radius_handles;
	Vector<Vector3> axial_from_handles;
	Vector<Vector3> axial_middle_handles;
	Vector<Vector3> axial_to_handles;
	float r = radius;
	Basis mesh_orientation = Basis::from_euler(Vector3(Math::deg_to_rad(90.0f), 0, 0));
	for (int32_t cone_i = 0; cone_i < kusudama_limit_cones.size(); cone_i = cone_i + (3 * 4)) {
		Vector3 center = Vector3(kusudama_limit_cones[cone_i + 0], kusudama_limit_cones[cone_i + 1], kusudama_limit_cones[cone_i + 2]);
		if (Math::is_zero_approx(center.length())) {
			break;
		}
		{
			Transform3D handle_relative_to_mesh;
			handle_relative_to_mesh.origin = center * radius;
			Transform3D handle_relative_to_universe = constraint_relative_to_the_universe * handle_relative_to_mesh;
			center_handles.push_back(handle_relative_to_universe.origin);
		}
		{
			float cone_radius = kusudama_limit_cones[cone_i + 3];
			float w = r * Math::sin(cone_radius);
			float d = r * Math::cos(cone_radius);
			const float ra = (float)(0 * 3);
			const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
			Transform3D handle_border_relative_to_mesh;
			Transform3D center_relative_to_mesh = Transform3D(Quaternion(Vector3(0, 1, 0), center)) * mesh_orientation;
			handle_border_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, a.y, -d));
			Transform3D handle_border_relative_to_skeleton = constraint_relative_to_the_skeleton * handle_border_relative_to_mesh;
			Transform3D handle_border_relative_to_universe = ewbik_skeleton->get_global_transform() * handle_border_relative_to_skeleton;
			radius_handles.push_back(handle_border_relative_to_universe.origin);
		}
	}
	const Vector3 axial_center = Vector3(0, 1, 0);
	float cone_radius = Math::deg_to_rad(90.0f);
	float w = r * Math::sin(cone_radius);
	float d = r * Math::cos(cone_radius);
	{
		const float ra = (float)kusudama->get_min_axial_angle();
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Transform3D axial_from_relative_to_mesh;
		Transform3D center_relative_to_mesh = Transform3D(Quaternion(Vector3(0, 1, 0), axial_center)) * mesh_orientation;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, a.y, -d));
		Transform3D axial_relative_to_skeleton = constraint_relative_to_the_skeleton * axial_from_relative_to_mesh;
		Transform3D axial_relative_to_universe = ewbik_skeleton->get_global_transform() * axial_relative_to_skeleton;
		axial_from_handles.push_back(axial_relative_to_universe.origin);
	}
	const int32_t segment_count = int(Math::rad_to_deg(IKKusudama::to_tau(kusudama->get_max_axial_angle() - kusudama->get_min_axial_angle()))) / 20;
	for (int32_t segment_i = 1; segment_i < segment_count; segment_i++) {
		const float ra = Math::lerp((float)kusudama->get_max_axial_angle(), (float)kusudama->get_min_axial_angle(), (float)segment_i / segment_count);
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Transform3D axial_from_relative_to_mesh;
		Transform3D center_relative_to_mesh = Transform3D(Quaternion(Vector3(0, 1, 0), axial_center)) * mesh_orientation;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, a.y, -d));
		Transform3D axial_relative_to_skeleton = constraint_relative_to_the_skeleton * axial_from_relative_to_mesh;
		Transform3D axial_relative_to_universe = ewbik_skeleton->get_global_transform() * axial_relative_to_skeleton;
		axial_middle_handles.push_back(axial_relative_to_universe.origin);
	}
	{
		const float ra = (float)(kusudama->get_max_axial_angle());
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Transform3D axial_from_relative_to_mesh;
		Transform3D center_relative_to_mesh = Transform3D(Quaternion(Vector3(0, 1, 0), axial_center)) * mesh_orientation;
		axial_from_relative_to_mesh.origin = center_relative_to_mesh.xform(Vector3(a.x, a.y, -d));
		Transform3D axial_relative_to_skeleton = constraint_relative_to_the_skeleton * axial_from_relative_to_mesh;
		Transform3D axial_relative_to_universe = ewbik_skeleton->get_global_transform() * axial_relative_to_skeleton;
		axial_to_handles.push_back(axial_relative_to_universe.origin);
	}
	p_gizmo->add_handles(center_handles, get_material("handles"), Vector<int>(), false, false);
	p_gizmo->add_handles(radius_handles, get_material("handles_radius"), Vector<int>(), false, true);
	p_gizmo->add_handles(axial_from_handles, get_material("handles_axial_from"), Vector<int>(), false, false);
	p_gizmo->add_handles(axial_middle_handles, get_material("handles_axial_middle"), Vector<int>(), false, true);
	p_gizmo->add_handles(axial_to_handles, get_material("handles_axial_to"), Vector<int>(), false, false);
}
