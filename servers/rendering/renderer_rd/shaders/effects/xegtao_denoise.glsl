///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
// SPDX-License-Identifier: MIT
// XeGTAO edge-aware denoiser, adapted to Godot's RenderingDevice compute shaders.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_working;
layout(set = 0, binding = 1) uniform sampler2D source_ao;
layout(set = 0, binding = 2) uniform sampler2D source_bent_normal;
layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D dest_final;
layout(rgba8, set = 1, binding = 1) uniform restrict writeonly image2D dest_bent_normal;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	float blur_beta;
	float output_scale;
}
params;

vec4 unpack_edges(float p_packed) {
	uint packed = uint(p_packed * 255.5);
	return vec4(
		float((packed >> 6) & 0x03u),
		float((packed >> 4) & 0x03u),
		float((packed >> 2) & 0x03u),
		float((packed >> 0) & 0x03u)) / 3.0;
}

vec2 load_working(ivec2 p_coord) {
	return texelFetch(source_working, clamp(p_coord, ivec2(0), params.screen_size - 1), 0).xy;
}

float load_ao(ivec2 p_coord) {
	return texelFetch(source_ao, clamp(p_coord, ivec2(0), params.screen_size - 1), 0).r;
}

vec3 load_bent_normal(ivec2 p_coord) {
	vec3 encoded = texelFetch(source_bent_normal, clamp(p_coord, ivec2(0), params.screen_size - 1), 0).xyz;
	return normalize(encoded * 2.0 - 1.0);
}

void main() {
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(coord, params.screen_size))) {
		return;
	}

	vec2 center = load_working(coord);
	vec4 edges_center = unpack_edges(center.y);
	vec2 value_left = load_working(coord + ivec2(-1, 0));
	vec2 value_right = load_working(coord + ivec2(1, 0));
	vec2 value_top = load_working(coord + ivec2(0, -1));
	vec2 value_bottom = load_working(coord + ivec2(0, 1));
	vec4 edges_left = unpack_edges(value_left.y);
	vec4 edges_right = unpack_edges(value_right.y);
	vec4 edges_top = unpack_edges(value_top.y);
	vec4 edges_bottom = unpack_edges(value_bottom.y);

	// Match opposite edge estimates before filtering to avoid blur bleeding across silhouettes.
	edges_center *= vec4(edges_left.y, edges_right.x, edges_top.w, edges_bottom.z);
	float edginess = clamp(4.0 - 2.5 - dot(edges_center, vec4(1.0)), 0.0, 1.0) / 1.5;
	edges_center = clamp(edges_center + edginess * 0.5, 0.0, 1.0);

	float diagonal_weight = 0.85 * 0.5;
	float weight_tl = diagonal_weight * (edges_center.x * edges_left.z + edges_center.z * edges_top.x);
	float weight_tr = diagonal_weight * (edges_center.z * edges_top.y + edges_center.y * edges_right.z);
	float weight_bl = diagonal_weight * (edges_center.w * edges_bottom.x + edges_center.x * edges_left.w);
	float weight_br = diagonal_weight * (edges_center.y * edges_right.w + edges_center.w * edges_bottom.y);

	float sum = load_ao(coord) * max(params.blur_beta, 0.001);
	float sum_weight = max(params.blur_beta, 0.001);
	float weight_left = edges_center.x;
	float weight_right = edges_center.y;
	float weight_top = edges_center.z;
	float weight_bottom = edges_center.w;
	sum += load_ao(coord + ivec2(-1, 0)) * weight_left + load_ao(coord + ivec2(1, 0)) * weight_right + load_ao(coord + ivec2(0, -1)) * weight_top + load_ao(coord + ivec2(0, 1)) * weight_bottom;
	sum_weight += weight_left + weight_right + weight_top + weight_bottom;

	vec2 value_tl = load_working(coord + ivec2(-1, -1));
	vec2 value_tr = load_working(coord + ivec2(1, -1));
	vec2 value_bl = load_working(coord + ivec2(-1, 1));
	vec2 value_br = load_working(coord + ivec2(1, 1));
	sum += load_ao(coord + ivec2(-1, -1)) * weight_tl + load_ao(coord + ivec2(1, -1)) * weight_tr + load_ao(coord + ivec2(-1, 1)) * weight_bl + load_ao(coord + ivec2(1, 1)) * weight_br;
	sum_weight += weight_tl + weight_tr + weight_bl + weight_br;

	vec3 bent_sum = load_bent_normal(coord) * max(params.blur_beta, 0.001);
	bent_sum += load_bent_normal(coord + ivec2(-1, 0)) * weight_left + load_bent_normal(coord + ivec2(1, 0)) * weight_right + load_bent_normal(coord + ivec2(0, -1)) * weight_top + load_bent_normal(coord + ivec2(0, 1)) * weight_bottom;
	bent_sum += load_bent_normal(coord + ivec2(-1, -1)) * weight_tl + load_bent_normal(coord + ivec2(1, -1)) * weight_tr + load_bent_normal(coord + ivec2(-1, 1)) * weight_bl + load_bent_normal(coord + ivec2(1, 1)) * weight_br;

	float visibility = clamp((sum / max(sum_weight, 0.0001)) * params.output_scale, 0.0, 1.0);
	imageStore(dest_final, coord, vec4(visibility));
	vec3 bent_normal = normalize(bent_sum / max(sum_weight, 0.0001));
	imageStore(dest_bent_normal, coord, vec4(bent_normal * 0.5 + 0.5, 1.0));
}
