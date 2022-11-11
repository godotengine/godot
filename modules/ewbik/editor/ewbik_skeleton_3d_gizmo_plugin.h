/*************************************************************************/
/*  ewbik_skeleton_3d_gizmo_plugin.h                                     */
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

#ifndef EWBIK_SKELETON_3D_GIZMO_PLUGIN_H
#define EWBIK_SKELETON_3D_GIZMO_PLUGIN_H

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "modules/ewbik/src/ik_bone_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/immediate_mesh.h"

#include "../src/ik_ewbik.h"

class Joint;
class PhysicalBone3D;
class EWBIKSkeleton3DEditorPlugin;
class Button;

class EWBIK3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(EWBIK3DGizmoPlugin, EditorNode3DGizmoPlugin);

	SkeletonModification3DNBoneIK *ewbik = nullptr;
	Ref<Shader> kusudama_shader;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("_get_gizmo_name"), &EWBIK3DGizmoPlugin::get_gizmo_name);
	}

public:
	const Color bone_color = EditorSettings::get_singleton()->get("editors/3d_gizmos/gizmo_colors/skeleton");
	const int32_t KUSUDAMA_MAX_CONES = 30;
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;
	EWBIK3DGizmoPlugin();
	void create_gizmo_mesh(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *ewbik_skeleton);
	void create_gizmo_handles(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *ewbik_skeleton);
};

class EditorPluginEWBIK : public EditorPlugin {
	GDCLASS(EditorPluginEWBIK, EditorPlugin);

public:
	EditorPluginEWBIK() {
		Ref<EWBIK3DGizmoPlugin> ewbik_gizmo_plugin;
		ewbik_gizmo_plugin.instantiate();
		Node3DEditor::get_singleton()->add_gizmo_plugin(ewbik_gizmo_plugin);
	}
};

static String EWBIK_KUSUDAMA_SHADER = R"(
// Skeleton 3D gizmo kusudama constraint shader.
shader_type spatial;
render_mode depth_prepass_alpha, cull_disabled;

uniform vec4 kusudama_color : source_color = vec4(0.58039218187332, 0.27058824896812, 0.00784313771874, 1.0);
uniform int cone_count = 0;

// 0,0,0 is the center of the kusudama. The kusudamas have their own bases that automatically get reoriented such that +y points in the direction that is the weighted average of the limitcones on the kusudama.
// But, if you have a kusuduma with just 1 limit_cone, then in general that limit_cone should be 0,1,0 in the kusudama's basis unless the user has specifically specified otherwise.

uniform vec4 cone_sequence[30];

// This shader can display up to 30 cones (represented by 30 4d vectors)
// Each group of 4 represents the xyz coordinates of the cone direction
// vector in model space and the fourth element represents radius

// TODO: fire 2022-05-26
// Use a texture to store bone parameters.
// Use the uv to get the row of the bone.

varying vec3 normal_model_dir;
varying vec4 vert_model_color;

bool is_in_inter_cone_path(in vec3 normal_dir, in vec4 tangent_1, in vec4 cone_1, in vec4 tangent_2, in vec4 cone_2) {
	vec3 c1xc2 = cross(cone_1.xyz, cone_2.xyz);
	float c1c2dir = dot(normal_dir, c1xc2);

	if (c1c2dir < 0.0) {
		vec3 c1xt1 = cross(cone_1.xyz, tangent_1.xyz);
		vec3 t1xc2 = cross(tangent_1.xyz, cone_2.xyz);
		float c1t1dir = dot(normal_dir, c1xt1);
		float t1c2dir = dot(normal_dir, t1xc2);

	 	return (c1t1dir > 0.0 && t1c2dir > 0.0);

	} else {
		vec3 t2xc1 = cross(tangent_2.xyz, cone_1.xyz);
		vec3 c2xt2 = cross(cone_2.xyz, tangent_2.xyz);
		float t2c1dir = dot(normal_dir, t2xc1);
		float c2t2dir = dot(normal_dir, c2xt2);

		return (c2t2dir > 0.0 && t2c1dir > 0.0);
	}
	return false;
}

//determines the current draw condition based on the desired draw condition in the setToArgument
// -3 = disallowed entirely;
// -2 = disallowed and on tangent_cone boundary
// -1 = disallowed and on control_cone boundary
// 0 =  allowed and empty;
// 1 =  allowed and on control_cone boundary
// 2  = allowed and on tangent_cone boundary
int get_allowability_condition(in int current_condition, in int set_to) {
	if((current_condition == -1 || current_condition == -2)
		&& set_to >= 0) {
		return current_condition *= -1;
	} else if(current_condition == 0 && (set_to == -1 || set_to == -2)) {
		return set_to *=-2;
	}
	return max(current_condition, set_to);
}

// returns 1 if normal_dir is beyond (cone.a) radians from the cone.rgb
// returns 0 if normal_dir is within (cone.a + boundary_width) radians from the cone.rgb
// return -1 if normal_dir is less than (cone.a) radians from the cone.rgb
int is_in_cone(in vec3 normal_dir, in vec4 cone, in float boundary_width) {
	float arc_dist_to_cone = acos(dot(normal_dir, cone.rgb));
	if (arc_dist_to_cone > (cone.a+(boundary_width/2.))) {
		return 1;
	}
	if (arc_dist_to_cone < (cone.a-(boundary_width/2.))) {
		return -1;
	}
	return 0;
}

// Returns a color corresponding to the allowability of this region,
// or otherwise the boundaries corresponding
// to various cones and tangent_cone.
vec4 color_allowed(in vec3 normal_dir,  in int cone_counts, in float boundary_width) {
	int current_condition = -3;
	if (cone_counts == 1) {
		vec4 cone = cone_sequence[0];
		int in_cone = is_in_cone(normal_dir, cone, boundary_width);
		bool is_in_cone = in_cone == 0;
		if (is_in_cone) {
			in_cone = -1;
		} else {
			if (in_cone < 0) {
				in_cone = 0;
			} else {
				in_cone = -3;
			}
		}
		current_condition = get_allowability_condition(current_condition, in_cone);
	} else {
		for(int i=0; i < (cone_counts-1)*3; i=i+3) {
			normal_dir = normalize(normal_dir);

			vec4 cone_1 = cone_sequence[i+0];
			vec4 tangent_1 = cone_sequence[i+1];
			vec4 tangent_2 = cone_sequence[i+2];
			vec4 cone_2 = cone_sequence[i+3];

			int inCone1 = is_in_cone(normal_dir, cone_1, boundary_width);
			if (inCone1 == 0) {
				inCone1 = -1;
			} else {
				if (inCone1 < 0) {
					inCone1 = 0;
				} else {
					inCone1 = -3;
				}
			}
			current_condition = get_allowability_condition(current_condition, inCone1);

			int inCone2 = is_in_cone(normal_dir, cone_2, boundary_width);
			if (inCone2 == 0) {
				inCone2 = -1;
			} else {
				if (inCone2 < 0) {
					inCone2 = 0;
				} else {
					inCone2 = -3;
				}
			}
			current_condition = get_allowability_condition(current_condition, inCone2);

			int in_tan_1 = is_in_cone(normal_dir, tangent_1, boundary_width);
			int in_tan_2 = is_in_cone(normal_dir, tangent_2, boundary_width);

			if (float(in_tan_1) < 1. || float(in_tan_2) < 1.) {
				in_tan_1 = in_tan_1 == 0 ? -2 : -3;
				current_condition = get_allowability_condition(current_condition, in_tan_1);
				in_tan_2 = in_tan_2 == 0 ? -2 : -3;
				current_condition = get_allowability_condition(current_condition, in_tan_2);
			} else {
				bool in_intercone = is_in_inter_cone_path(normal_dir, tangent_1, cone_1, tangent_2, cone_2);
				int intercone_condition = in_intercone ? 0 : -3;
				current_condition = get_allowability_condition(current_condition, intercone_condition);
			}
		}
	}
	vec4 result = vert_model_color;
	if (current_condition != 0 && current_condition != 2) {
		float on_cone_boundary = current_condition == 1 ? -0.3 : 0.0;
		result += vec4(0.0, on_cone_boundary, 0, 0.0);
	} else {
		return vec4(0.0, 0.0, 0.0, 0.0);
	}
	return result;
}

void vertex() {
	normal_model_dir = CUSTOM0.rgb;
	vert_model_color.rgb = kusudama_color.rgb;
}

void fragment() {
	vec4 result_color_allowed = vec4(0.0, 0.0, 0.0, 0.0);
	if (cone_sequence.length() == 30) {
		result_color_allowed = color_allowed(normal_model_dir, cone_count, 0.02);
	}
	ALBEDO = kusudama_color.rgb;
	if (result_color_allowed.a == 0.0) {
		discard;
	}
	ALBEDO = result_color_allowed.rgb;
	ALPHA = result_color_allowed.a;
}
)";

#endif // EWBIK_SKELETON_3D_GIZMO_PLUGIN_H
