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

layout(set = 0, binding = 0) uniform sampler2D input_ssgi;
layout(set = 0, binding = 1) uniform sampler2D input_depth;
layout(set = 0, binding = 2) uniform sampler2D input_normal;
layout(rgba16, set = 0, binding = 3) uniform restrict writeonly image2D output_image;

layout(push_constant, std430) uniform Params {
	vec4 pixel_dir; // xy: pixel_size, zw: direction
	vec4 thresholds; // x: depth_threshold, y: normal_power
	vec4 camera; // x: z_near, y: z_far, z: orthogonal
}
params;

vec3 decode_normal(vec3 p_encoded) {
	vec3 n = normalize(p_encoded * 2.0 - 1.0);
	n.z = -n.z;
	return n;
}

float linearize_depth(float p_depth) {
	float ndc = p_depth * 2.0 - 1.0;
	if (params.camera.z > 0.5) {
		return -(ndc * (params.camera.y - params.camera.x) - (params.camera.y + params.camera.x)) * 0.5;
	}
	return 2.0 * params.camera.x * params.camera.y / (params.camera.y + params.camera.x + ndc * (params.camera.y - params.camera.x));
}

float depth_weight(float p_center, float p_sample, float p_threshold) {
	float w = 1.0 - abs(p_sample - p_center) * p_threshold;
	return clamp(w, 0.0, 1.0);
}

float normal_weight(vec3 p_center, vec3 p_sample, float p_power) {
	return pow(clamp(dot(p_center, p_sample), 0.0, 1.0), p_power);
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = ivec2(1.0 / params.pixel_dir.xy);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	vec2 uv = (vec2(pixel) + vec2(0.5)) * params.pixel_dir.xy;
	float center_depth = linearize_depth(textureLod(input_depth, uv, 0.0).r);
	vec3 center_normal = decode_normal(textureLod(input_normal, uv, 0.0).xyz);

	vec2 dir = params.pixel_dir.zw * params.pixel_dir.xy;

	float weights[13] = float[](
		0.000244,
		0.00293,
		0.01611,
		0.05371,
		0.12085,
		0.19336,
		0.22559,
		0.19336,
		0.12085,
		0.05371,
		0.01611,
		0.00293,
		0.000244);

	vec3 accum = vec3(0.0);
	float weight_sum = 0.0;

	for (int i = -6; i <= 6; i++) {
		vec2 uv_sample = uv + dir * float(i);
		if (any(lessThan(uv_sample, vec2(0.0))) || any(greaterThan(uv_sample, vec2(1.0)))) {
			continue;
		}

		float sample_depth = linearize_depth(textureLod(input_depth, uv_sample, 0.0).r);
		vec3 sample_normal = decode_normal(textureLod(input_normal, uv_sample, 0.0).xyz);
		float w = weights[i + 6];
		w *= depth_weight(center_depth, sample_depth, params.thresholds.x * 0.5);
		w *= normal_weight(center_normal, sample_normal, params.thresholds.y * 0.5);

		accum += textureLod(input_ssgi, uv_sample, 0.0).rgb * w;
		weight_sum += w;
	}

	vec3 result = (weight_sum > 0.0) ? (accum / weight_sum) : textureLod(input_ssgi, uv, 0.0).rgb;
	imageStore(output_image, pixel, vec4(result, 1.0));
}
