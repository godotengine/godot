///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
// SPDX-License-Identifier: MIT
// XeGTAO depth prefilter, adapted to Godot's RenderingDevice compute shaders.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	vec2 depth_unpack;
	vec2 near_far;
	uint orthogonal;
	float effect_radius;
	float falloff_range;
	float pad;
}
params;

#ifdef LINEARIZE
layout(set = 0, binding = 0) uniform sampler2D source_depth;
#else
layout(set = 0, binding = 0) uniform sampler2D source_depth_mip;
#endif
layout(r16f, set = 1, binding = 0) uniform restrict writeonly image2D dest_depth;

float screen_space_to_view_space_depth(float p_depth) {
	if (params.orthogonal != 0u) {
		float depth = p_depth * 2.0 - 1.0;
		return -(depth * (params.near_far.y - params.near_far.x) - (params.near_far.y + params.near_far.x)) * 0.5;
	}
	float denominator = params.depth_unpack.y - p_depth;
	if (abs(denominator) <= 0.0000001) {
		return 65504.0;
	}
	return clamp(params.depth_unpack.x / denominator, 0.0, 65504.0);
}

float depth_mip_filter(vec4 p_depths) {
	float max_depth = max(max(p_depths.x, p_depths.y), max(p_depths.z, p_depths.w));
	float effect_radius = 0.75 * params.effect_radius * 1.457;
	float falloff = max(0.0001, params.falloff_range * effect_radius);
	float falloff_from = effect_radius * (1.0 - params.falloff_range);
	float falloff_mul = -1.0 / falloff;
	float falloff_add = falloff_from / falloff + 1.0;
	vec4 weights = clamp((max_depth - p_depths) * falloff_mul + falloff_add, 0.0, 1.0);
	float weight_sum = max(dot(weights, vec4(1.0)), 0.0001);
	return dot(weights, p_depths) / weight_sum;
}

void main() {
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(coord, params.screen_size))) {
		return;
	}

#ifdef LINEARIZE
	ivec2 source_size = textureSize(source_depth, 0);
	ivec2 source_coord = ivec2((vec2(coord) + 0.5) * vec2(source_size) / vec2(params.screen_size));
	source_coord = clamp(source_coord, ivec2(0), source_size - 1);
	float depth = texelFetch(source_depth, source_coord, 0).r;
	imageStore(dest_depth, coord, vec4(screen_space_to_view_space_depth(depth)));
#else
	ivec2 source_size = textureSize(source_depth_mip, 0);
	ivec2 source_coord = coord * 2;
	ivec2 max_coord = source_size - 1;
	vec4 depths = vec4(
		texelFetch(source_depth_mip, min(source_coord + ivec2(0, 0), max_coord), 0).r,
		texelFetch(source_depth_mip, min(source_coord + ivec2(1, 0), max_coord), 0).r,
		texelFetch(source_depth_mip, min(source_coord + ivec2(0, 1), max_coord), 0).r,
		texelFetch(source_depth_mip, min(source_coord + ivec2(1, 1), max_coord), 0).r);
	imageStore(dest_depth, coord, vec4(depth_mip_filter(depths)));
#endif
}
