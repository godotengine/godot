///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
// Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D input_current;
layout(set = 0, binding = 1) uniform sampler2D input_history;
layout(set = 0, binding = 2) uniform sampler2D input_depth;
layout(set = 0, binding = 3) uniform sampler2D input_normal;
layout(set = 0, binding = 4) uniform sampler2D input_velocity;
layout(rgba16, set = 0, binding = 5) uniform restrict writeonly image2D output_image;

layout(push_constant, std430) uniform Params {
	vec4 pixel_params; // xy: pixel_size, z: history_weight, w: luma_reject
	vec4 thresholds; // x: depth_threshold, y: normal_power, z: velocity_weight, w: clamp_sigma
	vec4 camera; // x: z_near, y: z_far, z: orthogonal
	vec4 temporal_params; // x: depth_reject, y: radius_reject, z: radius
	vec4 disocclusion_params; // x: depth_reject, y: normal_dot_min, z: velocity_max_px
	vec4 stability_params; // x: luma_delta_reject, y: stability_strength, z: blur_threshold, w: blur_strength
}
params;

vec3 decode_normal(vec3 p_encoded) {
	vec3 n = normalize(p_encoded * 2.0 - 1.0);
	n.z = -n.z;
	return n;
}

bool is_bad_vec3(vec3 v) {
	return any(isnan(v)) || any(isinf(v));
}

vec3 sanitize_color(vec3 v) {
	if (is_bad_vec3(v)) {
		return vec3(0.0);
	}
	return clamp(v, vec3(0.0), vec3(1000.0));
}

float linearize_depth(float p_depth) {
	float ndc = p_depth * 2.0 - 1.0;
	if (params.camera.z > 0.5) {
		return -(ndc * (params.camera.y - params.camera.x) - (params.camera.y + params.camera.x)) * 0.5;
	}
	return 2.0 * params.camera.x * params.camera.y / (params.camera.y + params.camera.x + ndc * (params.camera.y - params.camera.x));
}

vec3 clamp_history(vec3 p_history, vec2 p_uv, vec2 p_pixel_size) {
	vec3 min_v = vec3(1e9);
	vec3 max_v = vec3(-1e9);
	vec3 mean = vec3(0.0);
	vec3 mean_sq = vec3(0.0);
	int count = 0;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec2 uv = p_uv + vec2(float(x), float(y)) * p_pixel_size;
			if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
				continue;
			}
			vec3 v = textureLod(input_current, uv, 0.0).rgb;
			min_v = min(min_v, v);
			max_v = max(max_v, v);
			mean += v;
			mean_sq += v * v;
			count++;
		}
	}
	if (count > 0) {
		mean /= float(count);
		vec3 variance = max(mean_sq / float(count) - mean * mean, vec3(0.0));
		vec3 sigma = sqrt(variance) * params.thresholds.w;
		vec3 min_v_sigma = mean - sigma;
		vec3 max_v_sigma = mean + sigma;
		return clamp(p_history, max(min_v, min_v_sigma), min(max_v, max_v_sigma));
	}
	return clamp(p_history, min_v, max_v);
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = ivec2(1.0 / params.pixel_params.xy);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	vec2 uv = (vec2(pixel) + vec2(0.5)) * params.pixel_params.xy;
	vec3 current = sanitize_color(textureLod(input_current, uv, 0.0).rgb);

	vec2 velocity = textureLod(input_velocity, uv, 0.0).xy;
	vec2 prev_uv = uv + velocity;
	if (any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)))) {
		imageStore(output_image, pixel, vec4(current, 1.0));
		return;
	}

	vec4 history_sample = textureLod(input_history, prev_uv, 0.0);
	vec3 history = sanitize_color(history_sample.rgb);
	float history_confidence = history_sample.a;

	float center_depth = linearize_depth(textureLod(input_depth, uv, 0.0).r);
	float prev_depth = linearize_depth(textureLod(input_depth, prev_uv, 0.0).r);
	float depth_delta = abs(prev_depth - center_depth);
	float depth_weight = 1.0 - depth_delta * params.thresholds.x;
	depth_weight = clamp(depth_weight, 0.0, 1.0);

	vec3 center_normal = decode_normal(textureLod(input_normal, uv, 0.0).xyz);
	vec3 prev_normal = decode_normal(textureLod(input_normal, prev_uv, 0.0).xyz);
	float normal_dot = clamp(dot(center_normal, prev_normal), 0.0, 1.0);
	float normal_weight = pow(normal_dot, params.thresholds.y);

	float current_luma = dot(current, vec3(0.299, 0.587, 0.114));
	float history_luma = dot(history, vec3(0.299, 0.587, 0.114));
	float luma_delta = abs(history_luma - current_luma);
	float stability = clamp(1.0 - luma_delta * params.stability_params.x, 0.0, 1.0);
	float luma_weight = clamp(1.0 - luma_delta * params.pixel_params.w, 0.0, 1.0);
	float reactive = clamp((current_luma - history_luma) * params.pixel_params.w, 0.0, 1.0);

	float velocity_px = length(velocity / params.pixel_params.xy);
	float velocity_weight = clamp(1.0 - velocity_px * params.thresholds.z, 0.0, 1.0);
	float depth_factor = clamp(1.0 - (center_depth / max(params.camera.y, 0.001)) * params.temporal_params.x, 0.0, 1.0);
	float radius_factor = clamp(1.0 - (params.temporal_params.z / 16.0) * params.temporal_params.y, 0.0, 1.0);
	float history_weight = clamp(params.pixel_params.z * depth_weight * normal_weight * velocity_weight * luma_weight * depth_factor * radius_factor, 0.0, 1.0);
	history_weight *= clamp(history_confidence, 0.0, 1.0);
	history_weight *= (1.0 - reactive);

	if (depth_delta > params.disocclusion_params.x || normal_dot < params.disocclusion_params.y || velocity_px > params.disocclusion_params.z) {
		history_weight = 0.0;
	}

	history_weight *= mix(1.0 - params.stability_params.y, 1.0, stability);
	float new_confidence = clamp(mix(history_confidence, stability, 0.2), 0.0, 1.0);

	if (stability < params.stability_params.z) {
		vec3 sum = vec3(0.0);
		float wsum = 0.0;
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				vec2 uv = prev_uv + vec2(float(x), float(y)) * params.pixel_params.xy;
				if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
					continue;
				}
				sum += textureLod(input_history, uv, 0.0).rgb;
				wsum += 1.0;
			}
		}
		if (wsum > 0.0) {
			vec3 blurred = sum / wsum;
			history = mix(history, blurred, params.stability_params.w);
		}
	}

	history = clamp_history(history, uv, params.pixel_params.xy);
	vec3 result = mix(current, history, history_weight);
	result = sanitize_color(result);

	if (history_weight <= 0.0) {
		new_confidence = 0.0;
	}
	imageStore(output_image, pixel, vec4(result, new_confidence));
}
