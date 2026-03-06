/**************************************************************************/
/*  kusudama_gizmo_shader.h                                               */
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

#pragma once

// Kusudama constraint gizmo shader (from many_bone_ik; color set by caller).
static const char KUSUDAMA_GIZMO_SHADER[] = R"(
shader_type spatial;
render_mode depth_draw_always;

uniform vec4 kusudama_color : source_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform vec4 boundary_outline_color : source_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform int cone_count = 0;
uniform vec4 cone_sequence[30];

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
}

int get_allowability_condition(in int current_condition, in int set_to) {
	if ((current_condition == -1 || current_condition == -2)
			&& set_to >= 0) {
		return current_condition *= -1;
	} else if (current_condition == 0 && (set_to == -1 || set_to == -2)) {
		return set_to *= -2;
	}
	return max(current_condition, set_to);
}

int is_in_cone(in vec3 normal_dir, in vec4 cone, in float boundary_width) {
	float arc_dist_to_cone = acos(dot(normal_dir, cone.rgb));
	if (arc_dist_to_cone > (cone.a + (boundary_width / 2.0))) {
		return 1;
	}
	if (arc_dist_to_cone < (cone.a - (boundary_width / 2.0))) {
		return -1;
	}
	return 0;
}

// Returns -3 disallowed, -2 tangent boundary, -1 control boundary, 0 allowed.
int get_condition(in vec3 normal_dir, in int cone_counts, in float boundary_width) {
	int current_condition = -3;
	if (cone_counts == 1) {
		vec4 cone = cone_sequence[0];
		int in_cone = is_in_cone(normal_dir, cone, boundary_width);
		bool in_cone_bool = in_cone == 0;
		if (in_cone_bool) {
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
		for(int i = 0; i < (cone_counts - 1) * 3; i = i + 3) {
			normal_dir = normalize(normal_dir);

			vec4 cone_1 = cone_sequence[i + 0];
			vec4 tangent_1 = cone_sequence[i + 1];
			vec4 tangent_2 = cone_sequence[i + 2];
			vec4 cone_2 = cone_sequence[i + 3];

			int in_cone_1 = is_in_cone(normal_dir, cone_1, boundary_width);
			if (in_cone_1 == 0) {
				in_cone_1 = -1;
			} else {
				if (in_cone_1 < 0) {
					in_cone_1 = 0;
				} else {
					in_cone_1 = -3;
				}
			}
			current_condition = get_allowability_condition(current_condition, in_cone_1);

			int in_cone_2 = is_in_cone(normal_dir, cone_2, boundary_width);
			if (in_cone_2 == 0) {
				in_cone_2 = -1;
			} else {
				if (in_cone_2 < 0) {
					in_cone_2 = 0;
				} else {
					in_cone_2 = -3;
				}
			}
			current_condition = get_allowability_condition(current_condition, in_cone_2);

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
	return current_condition;
}

vec4 color_allowed(in vec3 normal_dir, in int cone_counts, in float boundary_width) {
	int current_condition = get_condition(normal_dir, cone_counts, boundary_width);
	vec4 result = vert_model_color;
	if (current_condition == -3 || current_condition == -2 || current_condition == -1) {
		return result;
	} else {
		return vec4(0.0, 0.0, 0.0, 0.0);
	}
}

void vertex() {
	normal_model_dir = CUSTOM0.rgb;
	vert_model_color.rgb = kusudama_color.rgb;
	VERTEX = VERTEX;
	POSITION = PROJECTION_MATRIX * VIEW_MATRIX * MODEL_MATRIX * vec4(VERTEX.xyz, 1.0);
	POSITION.z = mix(POSITION.z, POSITION.w, 0.999);
}

void fragment() {
	const float boundary_width = 0.02;
	const float outline_width = 0.05;
	int cond = get_condition(normal_model_dir, cone_count, boundary_width);
	vec4 result_color_allowed;
	if (cond == -3 || cond == -1 || cond == -2) {
		result_color_allowed = vert_model_color;
	} else {
		int cond_near = get_condition(normal_model_dir, cone_count, outline_width);
		if (cond_near == -1 || cond_near == -2) {
			result_color_allowed = boundary_outline_color;
		} else {
			result_color_allowed = vec4(0.0, 0.0, 0.0, 0.0);
		}
	}
	ALBEDO = result_color_allowed.rgb;
	ALPHA = 0.8;
}
)";
