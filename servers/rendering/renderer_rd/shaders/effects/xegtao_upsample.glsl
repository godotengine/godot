///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
// SPDX-License-Identifier: MIT
// Normal-aware upsample for half-resolution XeGTAO output.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ao;
layout(rgba8, set = 0, binding = 1) uniform restrict readonly image2D source_normal;
layout(set = 0, binding = 2) uniform sampler2D source_bent_normal;
layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D dest_final;
layout(rgba8, set = 1, binding = 1) uniform restrict writeonly image2D dest_bent_normal;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	ivec2 source_size;
}
params;

vec3 load_normal(ivec2 p_coord) {
	ivec2 coord = clamp(p_coord, ivec2(0), params.screen_size - 1);
	return normalize(imageLoad(source_normal, coord).xyz * 2.0 - 1.0);
}

vec3 load_bent_normal(ivec2 p_coord) {
	ivec2 coord = clamp(p_coord, ivec2(0), params.source_size - 1);
	return normalize(texelFetch(source_bent_normal, coord, 0).xyz * 2.0 - 1.0);
}

void main() {
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(coord, params.screen_size))) {
		return;
	}

	vec2 low_position = (vec2(coord) + 0.5) * vec2(params.source_size) / vec2(params.screen_size) - 0.5;
	ivec2 base = ivec2(floor(low_position));
	vec2 fraction = fract(low_position);
	vec3 center_normal = load_normal(coord);
	float sum = 0.0;
	float weight_sum = 0.0;
	vec3 bent_sum = vec3(0.0);

	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			ivec2 sample_coord = clamp(base + ivec2(x, y), ivec2(0), params.source_size - 1);
			vec2 axis_weight = mix(vec2(1.0) - fraction, fraction, vec2(x, y));
			float spatial_weight = axis_weight.x * axis_weight.y;
			ivec2 normal_coord = ivec2((vec2(sample_coord) + 0.5) * vec2(params.screen_size) / vec2(params.source_size));
			float normal_weight = pow(max(dot(center_normal, load_normal(normal_coord)), 0.0), 16.0);
			float weight = spatial_weight * normal_weight;
			sum += texelFetch(source_ao, sample_coord, 0).r * weight;
			bent_sum += load_bent_normal(sample_coord) * weight;
			weight_sum += weight;
		}
	}

	ivec2 fallback_coord = clamp(ivec2(round(low_position)), ivec2(0), params.source_size - 1);
	float fallback_ao = texelFetch(source_ao, fallback_coord, 0).r;
	float ao = weight_sum > 0.0001 ? sum / weight_sum : fallback_ao;
	vec3 bent_normal = weight_sum > 0.0001 ? normalize(bent_sum / weight_sum) : load_bent_normal(fallback_coord);
	imageStore(dest_final, coord, vec4(clamp(ao * 1.5, 0.0, 1.0)));
	imageStore(dest_bent_normal, coord, vec4(bent_normal * 0.5 + 0.5, 1.0));
}
