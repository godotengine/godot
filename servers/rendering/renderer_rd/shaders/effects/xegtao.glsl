///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
// SPDX-License-Identifier: MIT
// XeGTAO main pass, adapted to Godot's RenderingDevice compute shaders.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_viewspace_depth;
layout(rgba8, set = 0, binding = 1) uniform restrict readonly image2D source_normal;
layout(rg8, set = 1, binding = 0) uniform restrict writeonly image2D dest_working;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	vec2 pixel_size;
	vec2 ndc_to_view_mul;
	vec2 ndc_to_view_add;
	vec4 ao_params; // effect radius, falloff range, radius multiplier, sample distribution power
	vec4 ao_params2; // thin occluder compensation, final value power, depth MIP offset, intensity
	vec4 ao_params3; // detail, horizon, quality, pad
	vec4 fadeout_params; // distance fadeout multiplier and offset
	uint noise_index; // frame index modulo 64 when TAA is used
}
params;

const float XE_GTAO_PI = 3.14159265358979323846;
const float XE_GTAO_PI_HALF = 1.57079632679489661923;
const float XE_GTAO_OCCLUSION_TERM_SCALE = 1.5;

vec3 compute_viewspace_position(vec2 p_screen_pos, float p_viewspace_depth) {
	return vec3((params.ndc_to_view_mul * p_screen_pos + params.ndc_to_view_add) * p_viewspace_depth, p_viewspace_depth);
}

vec4 calculate_edges(float p_center_z, float p_left_z, float p_right_z, float p_top_z, float p_bottom_z) {
	vec4 edges = vec4(p_left_z, p_right_z, p_top_z, p_bottom_z) - p_center_z;
	vec2 slope = vec2((edges.y - edges.x) * 0.5, (edges.w - edges.z) * 0.5);
	vec4 slope_adjusted = edges + vec4(slope.x, -slope.x, slope.y, -slope.y);
	edges = min(abs(edges), abs(slope_adjusted));
	return clamp(1.25 - edges / max(p_center_z * 0.011, 0.000001), 0.0, 1.0);
}

// Godot's SSAO adds a high-frequency term from the four immediate neighbors.
// Reuse the same idea so close-range depth detail does not disappear in the
// wider GTAO hemisphere integration.
float calculate_detail_obscurance(vec3 p_normal, vec3 p_delta, float p_falloff_sq, float p_horizon) {
	float length_sq = dot(p_delta, p_delta);
	float n_dot_d = dot(p_normal, p_delta) * inversesqrt(max(length_sq, 0.00000001));
	float falloff = max(0.0, length_sq * p_falloff_sq + 1.0);
	return max(0.0, n_dot_d - p_horizon) * falloff;
}

float pack_edges(vec4 p_edges) {
	vec4 quantized = round(clamp(p_edges, 0.0, 1.0) * 2.9);
	return dot(quantized, vec4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

vec3 safe_normalize(vec3 p_value) {
	return p_value * inversesqrt(max(dot(p_value, p_value), 0.00000001));
}

float fast_acos(float p_value) {
	const float PI = 3.141593;
	const float HALF_PI = 1.570796;
	float x = abs(clamp(p_value, -1.0, 1.0));
	float result = -0.156583 * x + HALF_PI;
	result *= sqrt(max(1.0 - x, 0.0));
	return (p_value >= 0.0) ? result : PI - result;
}

uint hilbert_index(uint p_x, uint p_y) {
	uint index = 0u;
	for (uint level = 32u; level > 0u; level >>= 1u) {
		uint region_x = (p_x & level) > 0u ? 1u : 0u;
		uint region_y = (p_y & level) > 0u ? 1u : 0u;
		index += level * level * ((3u * region_x) ^ region_y);
		if (region_y == 0u) {
			if (region_x == 1u) {
				p_x = 63u - p_x;
				p_y = 63u - p_y;
			}
			uint temp = p_x;
			p_x = p_y;
			p_y = temp;
		}
	}
	return index;
}

vec2 spatio_temporal_noise(uvec2 p_coord, uint p_temporal_index) {
	uint index = hilbert_index(p_coord.x & 63u, p_coord.y & 63u);
	index += 288u * (p_temporal_index & 63u);
	return fract(vec2(0.5) + float(index) * vec2(0.7548776662466928, 0.5698402909980533));
}

float sample_viewspace_depth(vec2 p_screen_pos, float p_mip) {
	vec2 uv = clamp(p_screen_pos, params.pixel_size * 0.5, vec2(1.0) - params.pixel_size * 0.5);
	float max_mip = float(max(textureQueryLevels(source_viewspace_depth) - 1, 0));
	return textureLod(source_viewspace_depth, uv, clamp(p_mip, 0.0, max_mip)).r;
}

void main() {
	ivec2 pix_coord = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pix_coord, params.screen_size))) {
		return;
	}

	vec2 normalized_screen_pos = (vec2(pix_coord) + 0.5) * params.pixel_size;
	ivec2 left_coord = max(pix_coord + ivec2(-1, 0), ivec2(0));
	ivec2 right_coord = min(pix_coord + ivec2(1, 0), params.screen_size - 1);
	ivec2 top_coord = max(pix_coord + ivec2(0, -1), ivec2(0));
	ivec2 bottom_coord = min(pix_coord + ivec2(0, 1), params.screen_size - 1);
	float viewspace_z = texelFetch(source_viewspace_depth, pix_coord, 0).r;
	float pixel_left_z = texelFetch(source_viewspace_depth, left_coord, 0).r;
	float pixel_right_z = texelFetch(source_viewspace_depth, right_coord, 0).r;
	float pixel_top_z = texelFetch(source_viewspace_depth, top_coord, 0).r;
	float pixel_bottom_z = texelFetch(source_viewspace_depth, bottom_coord, 0).r;
	vec4 edges = calculate_edges(viewspace_z, pixel_left_z, pixel_right_z, pixel_top_z, pixel_bottom_z);

	ivec2 normal_size = imageSize(source_normal);
	ivec2 normal_coord = ivec2((vec2(pix_coord) + 0.5) * vec2(normal_size) / vec2(params.screen_size));
	normal_coord = clamp(normal_coord, ivec2(0), normal_size - 1);
	vec3 normal = normalize(imageLoad(source_normal, normal_coord).xyz * 2.0 - 1.0);
	normal.z = -normal.z;
	if (viewspace_z >= 65000.0 || viewspace_z <= 0.0) {
		imageStore(dest_working, pix_coord, vec4(1.0 / XE_GTAO_OCCLUSION_TERM_SCALE, pack_edges(vec4(1.0)), 0.0, 0.0));
		return;
	}

	// Move the center toward the camera to reduce half-float depth precision artifacts.
	viewspace_z *= 0.9992;
	vec3 pixel_center_pos = compute_viewspace_position(normalized_screen_pos, viewspace_z);
	vec3 view_vec = safe_normalize(-pixel_center_pos);

	float effect_radius = max(params.ao_params.x * params.ao_params.z, 0.0001);
	float falloff_range = clamp(params.ao_params.y, 0.01, 1.0) * effect_radius;
	float falloff_from = effect_radius * (1.0 - clamp(params.ao_params.y, 0.01, 1.0));
	float falloff_mul = -1.0 / max(falloff_range, 0.0001);
	float falloff_add = falloff_from / max(falloff_range, 0.0001) + 1.0;
	float sample_distribution_power = max(params.ao_params.w, 1.0);
	float thin_occluder_compensation = clamp(params.ao_params2.x, 0.0, 0.7);
	float depth_mip_sampling_offset = params.ao_params2.z;
	float final_value_power = max(params.ao_params2.y, 0.01);
	float intensity = max(params.ao_params2.w, 0.0);

	// High-frequency detail from the four immediate neighbors, mirroring the
	// detail term in Godot's SSAO gather pass.
	int quality = int(params.ao_params3.z + 0.5);
	float detail_obscurance = 0.0;
	float detail_strength = clamp(params.ao_params3.x, 0.0, 1.0);
	// SSAO enables the detail term from the Low preset and disables it for the
	// reference (adaptive) preset.
	if (detail_strength > 0.0 && quality >= 1 && quality != 4) {
		float detail_radius = max(params.ao_params.x, 0.0001);
		float detail_falloff_sq = -4.0 / (detail_radius * detail_radius);
		float detail_horizon = params.ao_params3.y;
		// SSAO compares against a center moved 1% towards the camera, which acts
		// as a threshold that keeps tiny depth noise from producing detail AO.
		vec3 detail_center_pos = pixel_center_pos * 0.99;
		vec2 left_ndc = (vec2(left_coord) + 0.5) * params.pixel_size;
		vec2 right_ndc = (vec2(right_coord) + 0.5) * params.pixel_size;
		vec2 top_ndc = (vec2(top_coord) + 0.5) * params.pixel_size;
		vec2 bottom_ndc = (vec2(bottom_coord) + 0.5) * params.pixel_size;
		detail_obscurance += edges.x * calculate_detail_obscurance(normal, compute_viewspace_position(left_ndc, pixel_left_z) - detail_center_pos, detail_falloff_sq, detail_horizon);
		detail_obscurance += edges.y * calculate_detail_obscurance(normal, compute_viewspace_position(right_ndc, pixel_right_z) - detail_center_pos, detail_falloff_sq, detail_horizon);
		detail_obscurance += edges.z * calculate_detail_obscurance(normal, compute_viewspace_position(top_ndc, pixel_top_z) - detail_center_pos, detail_falloff_sq, detail_horizon);
		detail_obscurance += edges.w * calculate_detail_obscurance(normal, compute_viewspace_position(bottom_ndc, pixel_bottom_z) - detail_center_pos, detail_falloff_sq, detail_horizon);
		// SSAO normalizes the detail term by its sample weight sum (roughly 10 for
		// the Low/Medium presets) and then scales the whole obscurance by intensity.
		// XeGTAO's intensity is calibrated at 1/4 of SSAO's, so the term needs the
		// inverse factor to end up with the same contrast.
		detail_obscurance *= detail_strength * 0.4;
	}

	int slice_count = 3;
	int steps_per_slice = 3;
	if (quality <= 0) {
		slice_count = 1;
		steps_per_slice = 2;
	} else if (quality == 1) {
		slice_count = 2;
		steps_per_slice = 2;
	} else if (quality == 3) {
		slice_count = 6;
		steps_per_slice = 3;
	} else if (quality >= 4) {
		slice_count = 6 + int(round(clamp(params.ao_params3.w, 0.0, 1.0) * 3.0));
		steps_per_slice = 3;
	}

	float pixel_dir_x = max(abs(viewspace_z * params.ndc_to_view_mul.x * params.pixel_size.x), 0.000001);
	float screen_space_radius = effect_radius / pixel_dir_x;
	float min_sample = 1.3 / max(screen_space_radius, 0.0001);
	vec2 noise = spatio_temporal_noise(uvec2(pix_coord), params.noise_index);
	float visibility = clamp((10.0 - screen_space_radius) / 100.0, 0.0, 1.0) * 0.5;

	for (int slice = 0; slice < slice_count; slice++) {
		float slice_k = (float(slice) + noise.x) / float(slice_count);
		float phi = slice_k * XE_GTAO_PI;
		float cos_phi = cos(phi);
		float sin_phi = sin(phi);
		vec2 omega = vec2(cos_phi, -sin_phi) * screen_space_radius;
		vec3 direction_vec = vec3(cos_phi, sin_phi, 0.0);
		vec3 ortho_direction_vec = direction_vec - dot(direction_vec, view_vec) * view_vec;
		vec3 axis_vec = safe_normalize(cross(ortho_direction_vec, view_vec));
		vec3 projected_normal_vec = normal - axis_vec * dot(normal, axis_vec);
		float sign_norm = sign(dot(ortho_direction_vec, projected_normal_vec));
		float projected_normal_length = max(length(projected_normal_vec), 0.0001);
		float cos_norm = clamp(dot(projected_normal_vec, view_vec) / projected_normal_length, 0.0, 1.0);
		float n = sign_norm * fast_acos(cos_norm);
		float low_horizon_cos0 = cos(n + XE_GTAO_PI_HALF);
		float low_horizon_cos1 = cos(n - XE_GTAO_PI_HALF);
		float horizon_cos0 = low_horizon_cos0;
		float horizon_cos1 = low_horizon_cos1;

		for (int step = 0; step < steps_per_slice; step++) {
			float step_base_noise = float(slice + step * steps_per_slice) * 0.6180339887498948;
			float step_noise = fract(noise.y + step_base_noise);
			float sample_distance = pow((float(step) + step_noise) / float(steps_per_slice), sample_distribution_power) + min_sample;
			vec2 sample_offset = round(sample_distance * omega) * params.pixel_size;
			float sample_offset_length = length(sample_offset / params.pixel_size);
			float mip_level = clamp(log2(max(sample_offset_length, 1.0)) - depth_mip_sampling_offset, 0.0, float(max(textureQueryLevels(source_viewspace_depth) - 1, 0)));

			vec2 sample_screen_pos0 = normalized_screen_pos + sample_offset;
			float sample_z0 = sample_viewspace_depth(sample_screen_pos0, mip_level);
			vec3 sample_pos0 = compute_viewspace_position(sample_screen_pos0, sample_z0);
			vec3 sample_delta0 = sample_pos0 - pixel_center_pos;
			float sample_dist0 = max(length(sample_delta0), 0.0001);
			vec3 sample_horizon_vec0 = sample_delta0 / sample_dist0;
			float falloff_base0 = length(vec3(sample_delta0.xy, sample_delta0.z * (1.0 + thin_occluder_compensation)));
			float weight0 = clamp(falloff_base0 * falloff_mul + falloff_add, 0.0, 1.0);
			float sample_horizon_cos0 = dot(sample_horizon_vec0, view_vec);
			sample_horizon_cos0 = mix(low_horizon_cos0, sample_horizon_cos0, weight0);
			horizon_cos0 = max(horizon_cos0, sample_horizon_cos0);

			vec2 sample_screen_pos1 = normalized_screen_pos - sample_offset;
			float sample_z1 = sample_viewspace_depth(sample_screen_pos1, mip_level);
			vec3 sample_pos1 = compute_viewspace_position(sample_screen_pos1, sample_z1);
			vec3 sample_delta1 = sample_pos1 - pixel_center_pos;
			float sample_dist1 = max(length(sample_delta1), 0.0001);
			vec3 sample_horizon_vec1 = sample_delta1 / sample_dist1;
			float falloff_base1 = length(vec3(sample_delta1.xy, sample_delta1.z * (1.0 + thin_occluder_compensation)));
			float weight1 = clamp(falloff_base1 * falloff_mul + falloff_add, 0.0, 1.0);
			float sample_horizon_cos1 = dot(sample_horizon_vec1, view_vec);
			sample_horizon_cos1 = mix(low_horizon_cos1, sample_horizon_cos1, weight1);
			horizon_cos1 = max(horizon_cos1, sample_horizon_cos1);
		}

		projected_normal_length = mix(projected_normal_length, 1.0, 0.05);
		float h0 = -fast_acos(horizon_cos1);
		float h1 = fast_acos(horizon_cos0);
		float iarc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
		float iarc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;
		visibility += projected_normal_length * (iarc0 + iarc1);
	}

	visibility /= float(slice_count);
	visibility = max(0.03, visibility);
	float fadeout = clamp(viewspace_z * params.fadeout_params.x + params.fadeout_params.y, 0.0, 1.0);
	float obscurance = clamp((1.0 - visibility + detail_obscurance) * intensity, 0.0, 0.97);
	visibility = pow(clamp(1.0 - obscurance * fadeout, 0.0, 1.0), final_value_power);
	imageStore(dest_working, pix_coord, vec4(clamp(visibility / XE_GTAO_OCCLUSION_TERM_SCALE, 0.0, 1.0), pack_edges(edges), 0.0, 0.0));
}
