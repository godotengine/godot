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

layout(set = 0, binding = 0) uniform sampler2D input_color;
layout(set = 0, binding = 1) uniform sampler2D input_depth;
layout(set = 0, binding = 2) uniform sampler2D input_normal;
layout(rgba16, set = 0, binding = 3) uniform restrict writeonly image2D output_image;
layout(set = 0, binding = 4) uniform sampler2D input_color_2;
layout(set = 0, binding = 5) uniform sampler2D input_color_4;
layout(set = 0, binding = 6) uniform sampler2D input_color_8;
layout(set = 0, binding = 7) uniform sampler2D input_color_16;

layout(push_constant, std430) uniform Params {
	vec4 pixel_intensity; // xy: pixel_size, z: intensity, w: depth_threshold
	vec4 thresholds; // x: normal_power
	ivec4 radius_pad; // x: radius
	vec4 camera; // x: z_near, y: z_far, z: orthogonal
	vec4 multirez; // x: enabled, y: max_level
	vec4 multirez_dist; // x: dist2, y: dist4, z: dist8, w: dist16
	vec4 jitter; // xy: jitter in pixels
	ivec4 sample_params; // x: sample_range, y: max_samples, z: quality
}
params;

const int TILE_BORDER = 4;
const int TILE_SIZE = 8 + TILE_BORDER * 2;

shared float cache_depth[TILE_SIZE * TILE_SIZE];
shared vec4 cache_color[TILE_SIZE * TILE_SIZE];
shared vec4 cache_normal[TILE_SIZE * TILE_SIZE];

int cache_index(ivec2 p_coord) {
	ivec2 clamped = clamp(p_coord, ivec2(0), ivec2(TILE_SIZE - 1));
	return clamped.y * TILE_SIZE + clamped.x;
}

vec3 apply_occlusion(ivec2 p_origin, ivec2 p_target, float p_origin_depth, float p_target_depth, vec3 p_color) {
	int dx = p_target.x - p_origin.x;
	int dy = p_target.y - p_origin.y;
	int step = max(abs(dx), abs(dy));
	if (step <= 1) {
		return p_color;
	}

	step = (step + 1) / 2;
	float step_rcp = 1.0 / float(step);
	float x_incr = float(dx) * step_rcp;
	float y_incr = float(dy) * step_rcp;
	float x = float(p_origin.x);
	float y = float(p_origin.y);

	for (int i = 1; i < step; i++) {
		x += x_incr;
		y += y_incr;

		ivec2 loc = ivec2(x, y);
		int idx = cache_index(loc);
		float z = mix(p_origin_depth, p_target_depth, float(i) * step_rcp);
		float cached = cache_depth[idx];
		if (cached < z - 0.05) {
			return cache_color[idx].xyz;
		}
	}

	return p_color;
}

vec3 sample_multirez_color(vec2 p_uv, float p_dist, bool p_use_cache, vec3 p_cache_color) {
	if (params.multirez.x < 0.5) {
		return p_use_cache ? p_cache_color : textureLod(input_color, p_uv, 0.0).rgb;
	}

	float max_level = params.multirez.y;
	if (p_dist >= params.multirez_dist.w && max_level >= 16.0) {
		return textureLod(input_color_16, p_uv, 0.0).rgb;
	}
	if (p_dist >= params.multirez_dist.z && max_level >= 8.0) {
		return textureLod(input_color_8, p_uv, 0.0).rgb;
	}
	if (p_dist >= params.multirez_dist.y && max_level >= 4.0) {
		return textureLod(input_color_4, p_uv, 0.0).rgb;
	}
	if (p_dist >= params.multirez_dist.x && max_level >= 2.0) {
		return textureLod(input_color_2, p_uv, 0.0).rgb;
	}

	return p_use_cache ? p_cache_color : textureLod(input_color, p_uv, 0.0).rgb;
}

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

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, ivec2(1.0 / params.pixel_intensity.xy)))) {
		return;
	}

	vec2 uv = (vec2(pixel) + vec2(0.5)) * params.pixel_intensity.xy;
	ivec2 half_size = ivec2(1.0 / params.pixel_intensity.xy);

	ivec2 local_id = ivec2(gl_LocalInvocationID.xy);
	ivec2 tile_origin = ivec2(gl_WorkGroupID.xy) * ivec2(8, 8) - ivec2(TILE_BORDER);

	for (int y = local_id.y; y < TILE_SIZE; y += 8) {
		for (int x = local_id.x; x < TILE_SIZE; x += 8) {
			ivec2 tile_coord = tile_origin + ivec2(x, y);
			ivec2 clamped = clamp(tile_coord, ivec2(0), half_size - ivec2(1));
			vec2 tile_uv = (vec2(clamped) + vec2(0.5)) * params.pixel_intensity.xy;

			int idx = cache_index(ivec2(x, y));
			cache_depth[idx] = linearize_depth(textureLod(input_depth, tile_uv, 0.0).r);
			cache_normal[idx] = vec4(decode_normal(textureLod(input_normal, tile_uv, 0.0).xyz), 0.0);
			cache_color[idx] = vec4(textureLod(input_color, tile_uv, 0.0).rgb, 0.0);
		}
	}

	barrier();

	float center_depth = linearize_depth(textureLod(input_depth, uv, 0.0).r);
	vec3 center_normal = decode_normal(textureLod(input_normal, uv, 0.0).xyz);

	vec3 accum = vec3(0.0);
	float weight_sum = 0.0;

	int radius = params.radius_pad.x;
	float inv_radius = radius > 0 ? 1.0 / float(radius) : 0.0;
	const int MAX_SAMPLES = 96;
	int max_samples = (params.sample_params.y > 0) ? params.sample_params.y : MAX_SAMPLES;
	int max_range = int(floor((sqrt(float(max_samples)) - 1.0) * 0.5));
	int range = max(params.sample_params.x, 1);
	range = max(min(range, max_range), 1);
	float quality = float(params.sample_params.z);
	float quality_scale = clamp(1.0 - 0.08 * quality, 0.7, 1.0);
	float spread = max(float(radius) / float(range), 1.0) * quality_scale;
	float dist_exp = clamp(1.2 - 0.2 * quality, 0.6, 1.2);
	float radius_falloff = 1.0 + float(radius) * 0.02;
	for (int y = -range; y <= range; y++) {
		for (int x = -range; x <= range; x++) {
			vec2 offset_px = vec2(float(x), float(y)) * spread;
			float dist = length(offset_px);
			float dist_norm = dist * inv_radius;
			float dist_weight = smoothstep(1.5, 0.0, dist_norm);
			dist_weight = pow(dist_weight, dist_exp * 0.5);
			float layer_weight = clamp(1.0 - dist_norm, 0.0, 1.0);
			dist_weight *= pow(layer_weight, radius_falloff);
			if (dist_weight <= 0.0) {
				continue;
			}

			ivec2 sample_pixel = pixel + ivec2(floor(offset_px + vec2(0.5)));
			if (any(lessThan(sample_pixel, ivec2(0))) || any(greaterThanEqual(sample_pixel, half_size))) {
				continue;
			}

			vec3 sample_color;
			vec3 sample_normal;
			float sample_depth;
			ivec2 cache_coord = sample_pixel - tile_origin;
			bool in_cache = all(greaterThanEqual(cache_coord, ivec2(0))) && all(lessThan(cache_coord, ivec2(TILE_SIZE)));
			if (in_cache) {
				int idx = cache_index(cache_coord);
				sample_depth = cache_depth[idx];
				sample_normal = cache_normal[idx].xyz;
				vec3 cached = cache_color[idx].xyz;
				vec2 sample_uv = (vec2(sample_pixel) + vec2(0.5) + params.jitter.xy) * params.pixel_intensity.xy;
				float dist = length(offset_px);
				sample_color = sanitize_color(sample_multirez_color(sample_uv, dist, true, cached));
				sample_color = apply_occlusion(pixel - tile_origin, cache_coord, center_depth, sample_depth, sample_color);
			} else {
				vec2 sample_uv = (vec2(sample_pixel) + vec2(0.5) + params.jitter.xy) * params.pixel_intensity.xy;
				sample_depth = linearize_depth(textureLod(input_depth, sample_uv, 0.0).r);
				sample_normal = decode_normal(textureLod(input_normal, sample_uv, 0.0).xyz);
				float dist = length(offset_px);
				sample_color = sanitize_color(sample_multirez_color(sample_uv, dist, false, vec3(0.0)));
			}

			float depth_weight = 1.0 - abs(sample_depth - center_depth) * params.pixel_intensity.w;
			depth_weight = clamp(depth_weight, 0.0, 1.0);
			float normal_weight = pow(clamp(dot(center_normal, sample_normal), 0.0, 1.0), params.thresholds.x);

			float weight = depth_weight * normal_weight * dist_weight;
			if (weight <= 0.0) {
				continue;
			}

			float luma = dot(sample_color, vec3(0.299, 0.587, 0.114));
			float light_weight = smoothstep(0.01, 0.35, luma);
			light_weight = pow(light_weight, 0.6);
			if (light_weight <= 0.0) {
				continue;
			}

			float weighted = weight * light_weight;
			accum += sample_color * weighted;
			weight_sum += weighted;

			if (radius >= 2) {
				vec2 offset_px2 = offset_px * 2.0;
				float dist2 = length(offset_px2);
				float dist_norm2 = dist2 / max(float(radius) * 4.0, 1.0);
				float dist_weight2 = smoothstep(1.5, 0.0, dist_norm2);
				dist_weight2 = pow(dist_weight2, dist_exp * 0.5);
				float layer_weight2 = clamp(1.0 - dist_norm2, 0.0, 1.0);
				dist_weight2 *= pow(layer_weight2, radius_falloff);
				if (dist_weight2 > 0.0) {
					ivec2 sample_pixel2 = pixel + ivec2(floor(offset_px2 + vec2(0.5)));
					if (all(greaterThanEqual(sample_pixel2, ivec2(0))) && all(lessThan(sample_pixel2, half_size))) {
						vec3 sample_color2;
						vec3 sample_normal2;
						float sample_depth2;
						ivec2 cache_coord2 = sample_pixel2 - tile_origin;
						bool in_cache2 = all(greaterThanEqual(cache_coord2, ivec2(0))) && all(lessThan(cache_coord2, ivec2(TILE_SIZE)));
						if (in_cache2) {
							int idx2 = cache_index(cache_coord2);
							sample_depth2 = cache_depth[idx2];
							sample_normal2 = cache_normal[idx2].xyz;
							vec3 cached2 = cache_color[idx2].xyz;
							vec2 sample_uv2 = (vec2(sample_pixel2) + vec2(0.5) + params.jitter.xy) * params.pixel_intensity.xy;
							float dist2 = length(offset_px2);
							sample_color2 = sanitize_color(sample_multirez_color(sample_uv2, dist2, true, cached2));
							sample_color2 = apply_occlusion(pixel - tile_origin, cache_coord2, center_depth, sample_depth2, sample_color2);
						} else {
							vec2 sample_uv2 = (vec2(sample_pixel2) + vec2(0.5) + params.jitter.xy) * params.pixel_intensity.xy;
							sample_depth2 = linearize_depth(textureLod(input_depth, sample_uv2, 0.0).r);
							sample_normal2 = decode_normal(textureLod(input_normal, sample_uv2, 0.0).xyz);
							float dist2 = length(offset_px2);
							sample_color2 = sanitize_color(sample_multirez_color(sample_uv2, dist2, false, vec3(0.0)));
						}

						float depth_weight2 = 1.0 - abs(sample_depth2 - center_depth) * params.pixel_intensity.w;
						depth_weight2 = clamp(depth_weight2, 0.0, 1.0);
						float normal_weight2 = pow(clamp(dot(center_normal, sample_normal2), 0.0, 1.0), params.thresholds.x);
						float weight2 = depth_weight2 * normal_weight2 * dist_weight2 * 0.45;
						if (weight2 > 0.0) {
							float luma2 = dot(sample_color2, vec3(0.299, 0.587, 0.114));
							float light_weight2 = smoothstep(0.01, 0.35, luma2);
							light_weight2 = pow(light_weight2, 0.6);
							if (light_weight2 > 0.0) {
								float weighted2 = weight2 * light_weight2;
								accum += sample_color2 * weighted2;
								weight_sum += weighted2;
							}
						}
					}
				}
			}

			if (radius >= 4) {
				vec2 offset_px4 = offset_px * 4.0;
				float dist4 = length(offset_px4);
				float dist_norm4 = dist4 / max(float(radius) * 8.0, 1.0);
				float dist_weight4 = smoothstep(1.5, 0.0, dist_norm4);
				dist_weight4 = pow(dist_weight4, dist_exp * 0.5);
				float layer_weight4 = clamp(1.0 - dist_norm4, 0.0, 1.0);
				dist_weight4 *= pow(layer_weight4, radius_falloff);
				if (dist_weight4 > 0.0) {
					ivec2 sample_pixel4 = pixel + ivec2(floor(offset_px4 + vec2(0.5)));
					if (all(greaterThanEqual(sample_pixel4, ivec2(0))) && all(lessThan(sample_pixel4, half_size))) {
						vec3 sample_color4;
						vec3 sample_normal4;
						float sample_depth4;
						ivec2 cache_coord4 = sample_pixel4 - tile_origin;
						bool in_cache4 = all(greaterThanEqual(cache_coord4, ivec2(0))) && all(lessThan(cache_coord4, ivec2(TILE_SIZE)));
						if (in_cache4) {
							int idx4 = cache_index(cache_coord4);
							sample_depth4 = cache_depth[idx4];
							sample_normal4 = cache_normal[idx4].xyz;
							vec3 cached4 = cache_color[idx4].xyz;
							vec2 sample_uv4 = (vec2(sample_pixel4) + vec2(0.5) + params.jitter.xy) * params.pixel_intensity.xy;
							float dist4 = length(offset_px4);
							sample_color4 = sanitize_color(sample_multirez_color(sample_uv4, dist4, true, cached4));
							sample_color4 = apply_occlusion(pixel - tile_origin, cache_coord4, center_depth, sample_depth4, sample_color4);
						} else {
							vec2 sample_uv4 = (vec2(sample_pixel4) + vec2(0.5) + params.jitter.xy) * params.pixel_intensity.xy;
							sample_depth4 = linearize_depth(textureLod(input_depth, sample_uv4, 0.0).r);
							sample_normal4 = decode_normal(textureLod(input_normal, sample_uv4, 0.0).xyz);
							float dist4 = length(offset_px4);
							sample_color4 = sanitize_color(sample_multirez_color(sample_uv4, dist4, false, vec3(0.0)));
						}

						float depth_weight4 = 1.0 - abs(sample_depth4 - center_depth) * params.pixel_intensity.w;
						depth_weight4 = clamp(depth_weight4, 0.0, 1.0);
						float normal_weight4 = pow(clamp(dot(center_normal, sample_normal4), 0.0, 1.0), params.thresholds.x);
						float weight4 = depth_weight4 * normal_weight4 * dist_weight4 * 0.25;
						if (weight4 > 0.0) {
							float luma4 = dot(sample_color4, vec3(0.299, 0.587, 0.114));
							float light_weight4 = smoothstep(0.01, 0.35, luma4);
							light_weight4 = pow(light_weight4, 0.6);
							if (light_weight4 > 0.0) {
								float weighted4 = weight4 * light_weight4;
								accum += sample_color4 * weighted4;
								weight_sum += weighted4;
							}
						}
					}
				}
			}
		}
	}

	vec3 result = (weight_sum > 0.0) ? (accum / weight_sum) : vec3(0.0);
	result = sanitize_color(result);
	imageStore(output_image, pixel, vec4(result, 1.0));
}
