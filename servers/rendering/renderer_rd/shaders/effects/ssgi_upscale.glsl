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
	vec4 pixel_sizes; // xy: full pixel size, zw: half pixel size
	vec4 thresholds; // x: depth_threshold, y: normal_power
	vec4 camera; // x: z_near, y: z_far, z: orthogonal
}
params;

const int TILE_BORDER = 2;
const int TILE_SIZE = 8 + TILE_BORDER * 2;
const int HALF_TILE_SIZE = TILE_SIZE / 2 + 1;

shared float cache_depth[TILE_SIZE * TILE_SIZE];
shared vec4 cache_normal[TILE_SIZE * TILE_SIZE];
shared vec4 cache_ssgi[HALF_TILE_SIZE * HALF_TILE_SIZE];

int cache_index(ivec2 p_coord) {
	ivec2 clamped = clamp(p_coord, ivec2(0), ivec2(TILE_SIZE - 1));
	return clamped.y * TILE_SIZE + clamped.x;
}

int half_cache_index(ivec2 p_coord) {
	ivec2 clamped = clamp(p_coord, ivec2(0), ivec2(HALF_TILE_SIZE - 1));
	return clamped.y * HALF_TILE_SIZE + clamped.x;
}

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

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 full_size = ivec2(1.0 / params.pixel_sizes.xy);
	if (any(greaterThanEqual(pixel, full_size))) {
		return;
	}

	ivec2 half_size = ivec2(1.0 / params.pixel_sizes.zw);

	ivec2 local_id = ivec2(gl_LocalInvocationID.xy);
	ivec2 tile_origin = ivec2(gl_WorkGroupID.xy) * ivec2(8, 8) - ivec2(TILE_BORDER);
	ivec2 half_origin = tile_origin / 2;

	for (int y = local_id.y; y < TILE_SIZE; y += 8) {
		for (int x = local_id.x; x < TILE_SIZE; x += 8) {
			ivec2 tile_coord = tile_origin + ivec2(x, y);
			ivec2 clamped = clamp(tile_coord, ivec2(0), full_size - ivec2(1));
			vec2 tile_uv = (vec2(clamped) + vec2(0.5)) * params.pixel_sizes.xy;

			int idx = cache_index(ivec2(x, y));
			cache_depth[idx] = linearize_depth(textureLod(input_depth, tile_uv, 0.0).r);
			cache_normal[idx] = vec4(decode_normal(textureLod(input_normal, tile_uv, 0.0).xyz), 0.0);
		}
	}

	for (int y = local_id.y; y < HALF_TILE_SIZE; y += 8) {
		for (int x = local_id.x; x < HALF_TILE_SIZE; x += 8) {
			ivec2 half_coord = half_origin + ivec2(x, y);
			ivec2 clamped = clamp(half_coord, ivec2(0), half_size - ivec2(1));
			vec2 half_uv = (vec2(clamped) + vec2(0.5)) * params.pixel_sizes.zw;
			int idx = half_cache_index(ivec2(x, y));
			cache_ssgi[idx] = vec4(textureLod(input_ssgi, half_uv, 0.0).rgb, 0.0);
		}
	}

	barrier();

	vec2 uv_center = (vec2(pixel) + vec2(0.5)) * params.pixel_sizes.xy;
	vec3 center_normal;
	float center_depth;
	ivec2 center_cache = pixel - tile_origin;
	if (all(greaterThanEqual(center_cache, ivec2(0))) && all(lessThan(center_cache, ivec2(TILE_SIZE)))) {
		int idx = cache_index(center_cache);
		center_depth = cache_depth[idx];
		center_normal = cache_normal[idx].xyz;
	} else {
		center_depth = linearize_depth(textureLod(input_depth, uv_center, 0.0).r);
		center_normal = decode_normal(textureLod(input_normal, uv_center, 0.0).xyz);
	}

	vec3 accum = vec3(0.0);
	float weight_sum = 0.0;

	const float kernel[3] = float[3](1.0, 2.0, 1.0);

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			ivec2 full_coord = clamp(pixel + ivec2(x, y), ivec2(0), full_size - ivec2(1));

			vec3 sample_normal;
			float sample_depth;
			ivec2 full_cache = full_coord - tile_origin;
			if (all(greaterThanEqual(full_cache, ivec2(0))) && all(lessThan(full_cache, ivec2(TILE_SIZE)))) {
				int idx = cache_index(full_cache);
				sample_depth = cache_depth[idx];
				sample_normal = cache_normal[idx].xyz;
			} else {
				vec2 uv_full = (vec2(full_coord) + vec2(0.5)) * params.pixel_sizes.xy;
				sample_depth = linearize_depth(textureLod(input_depth, uv_full, 0.0).r);
				sample_normal = decode_normal(textureLod(input_normal, uv_full, 0.0).xyz);
			}

			float depth_weight = 1.0 - abs(sample_depth - center_depth) * params.thresholds.x;
			depth_weight = clamp(depth_weight, 0.0, 1.0);
			float normal_weight = pow(clamp(dot(center_normal, sample_normal), 0.0, 1.0), params.thresholds.y);

			float spatial_weight = kernel[abs(x)] * kernel[abs(y)];
			float weight = depth_weight * normal_weight * spatial_weight;

			vec2 uv_full = (vec2(full_coord) + vec2(0.5)) * params.pixel_sizes.xy;
			vec3 ssgi_sample = textureLod(input_ssgi, uv_full, 0.0).rgb;

			accum += ssgi_sample * weight;
			weight_sum += weight;
		}
	}

	vec3 base_sample = textureLod(input_ssgi, uv_center, 0.0).rgb;
	vec3 weighted = (weight_sum > 0.0) ? (accum / weight_sum) : base_sample;
	vec3 result = mix(base_sample, weighted, 0.85);
	imageStore(output_image, pixel, vec4(result, 1.0));
}
