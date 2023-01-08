/**************************************************************************/
/*  many_bone_ik_3d_gizmo_plugin.cpp                                      */
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

#include "many_bone_ik_3d_gizmo_plugin.h"

#include "core/variant/typed_array.h"
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

void ManyBoneIK3DGizmoPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_gizmo_name"), &ManyBoneIK3DGizmoPlugin::get_gizmo_name);
}

bool ManyBoneIK3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return cast_to<ManyBoneIK3D>(p_spatial);
}

String ManyBoneIK3DGizmoPlugin::get_gizmo_name() const {
	return "ManyBoneIK3D";
}

void ManyBoneIK3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	if (!p_gizmo) {
		return;
	}
	Node3D *node_3d = p_gizmo->get_node_3d();
	if (!node_3d) {
		return;
	}
	if (!node_3d->is_visible_in_tree()) {
		return;
	}
	p_gizmo->clear();
	if (!p_gizmo->is_selected()) {
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
				if (ik_bone->is_orientationally_constrained()) {
					create_gizmo_mesh(bone_i, ik_bone, p_gizmo, current_bone_color, many_bone_ik_skeleton, many_bone_ik);
				}
			}
		}
	}
}

void ManyBoneIK3DGizmoPlugin::create_gizmo_mesh(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *many_bone_ik_skeleton, ManyBoneIK3D *p_many_bone_ik) {
	Ref<IKKusudama3D> ik_kusudama = ik_bone->get_constraint();
	if (ik_kusudama.is_null()) {
		return;
	}
	const TypedArray<IKLimitCone3D> &limit_cones = ik_kusudama->get_limit_cones();
	if (!limit_cones.size()) {
		return;
	}
	BoneId parent_idx = many_bone_ik_skeleton->get_bone_parent(current_bone_idx);
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

	Transform3D constraint_relative_to_the_skeleton = p_many_bone_ik->get_relative_transform(p_many_bone_ik->get_owner()).affine_inverse() * many_bone_ik_skeleton->get_relative_transform(many_bone_ik_skeleton->get_owner()) * p_many_bone_ik->get_godot_skeleton_transform_inverse() * ik_bone->get_constraint_orientation_transform()->get_global_transform();
	PackedFloat32Array kusudama_limit_cones;
	Ref<IKKusudama3D> kusudama = ik_bone->get_constraint();
	for (int32_t cone_i = 0; cone_i < limit_cones.size(); cone_i++) {
		Ref<IKLimitCone3D> limit_cone = limit_cones[cone_i];
		Vector3 control_point = limit_cone->get_control_point();
		PackedFloat32Array new_kusudama_limit_cones;
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
	Vector3 v0 = many_bone_ik_skeleton->get_bone_global_rest(current_bone_idx).origin;
	Vector3 v1 = many_bone_ik_skeleton->get_bone_global_rest(parent_idx).origin;
	real_t dist = v0.distance_to(v1);
	float radius = dist / 5.0;
	// Code copied from the SphereMesh.
	float height = dist / 2.5;
	int rings = 8;

	int i = 0, j = 0, prevrow = 0, thisrow = 0, point = 0;
	float x, y, z;

	float scale = height * 0.5;

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<int> indices;
	point = 0;

	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		int radial_segments = 16;
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
	if (!indices.size()) {
		return;
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
			kusudama_material, constraint_relative_to_the_skeleton);
}

ManyBoneIK3DGizmoPlugin::ManyBoneIK3DGizmoPlugin() {
	create_material("lines_primary", Color(0.93725490570068, 0.19215686619282, 0.22352941334248), true, true, true);
	kusudama_shader.instantiate();
	kusudama_shader->set_code(MANY_BONE_IKKUSUDAMA_SHADER);
}

int32_t ManyBoneIK3DGizmoPlugin::get_priority() const {
	return -1;
}

EditorPluginManyBoneIK::EditorPluginManyBoneIK() {
	Ref<ManyBoneIK3DGizmoPlugin> many_bone_ik_gizmo_plugin;
	many_bone_ik_gizmo_plugin.instantiate();
	Node3DEditor::get_singleton()->add_gizmo_plugin(many_bone_ik_gizmo_plugin);
}
