#[compute]

#version 450

#VERSION_DEFINES

#if defined(MODE_IRRADIANCE_CACHE_QUERY) || defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)
#define HDDAGI_IRRADIANCE_CACHE_SET 2
#include "hddagi_screen_probe_irradiance_cache_inc.glsl"
#endif

#include "../oct_inc.glsl"

#ifdef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#endif

const uint SCREEN_PROBE_FLAG_DETAIL_TRACE = 1u << 0u;
const uint SCREEN_PROBE_FLAG_GUIDED_SAMPLING = 1u << 1u;
const uint SCREEN_PROBE_FLAG_DIRECTIONAL_SURFACE_FOOTPRINT = 1u << 2u;
const uint SCREEN_PROBE_FLAG_DIRECTIONAL_HISTORY_VALID = 1u << 3u;
const uint SCREEN_PROBE_SKY_COLOR = 1u;
const uint SCREEN_PROBE_SKY_TEXTURE = 2u;
const float SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE = 2.0;

const int HDDAGI_REGION_SIZE = 8;
const int HDDAGI_HDDA_FP_BITS = 10;
const uint HDDAGI_LIGHT_CELL_VALID_BIT = 1u << 26u;
const uint HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK = HDDAGI_LIGHT_CELL_VALID_BIT - 1u;
const float TAU = 6.283185307179586;

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_FILTER) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_DIRECTIONAL_RESOLVE)
#ifndef SCREEN_PROBE_DIRECTIONAL_SIZE
#error "Directional screen probes require SCREEN_PROBE_DIRECTIONAL_SIZE"
#endif
#if SCREEN_PROBE_DIRECTIONAL_SIZE != 8
#error "Directional screen probes require an 8x8 workgroup"
#endif
const int SCREEN_PROBE_DIRECTIONAL_TILE_SIZE = SCREEN_PROBE_DIRECTIONAL_SIZE;
const int SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
const float SCREEN_PROBE_DIRECTIONAL_IRRADIANCE_FACTOR = 4.0 / float(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT);
const float SCREEN_PROBE_DIRECTIONAL_FP16_MAX = 65504.0;
const uint SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE = 8u;
const float SCREEN_PROBE_DIRECTIONAL_RADIANCE_MAX = 10.0;
#endif

#ifndef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
layout(push_constant, std430) uniform Params {
	ivec2 gi_size;
	ivec2 screen_size;

	int probe_size;
	uint view_index;
	uint frame_index;
	uint flags;

	float normal_bias;
	float spatial_depth_tolerance_scale;
	float history_depth_tolerance;
	float history_normal_threshold;

	uint candidate_count;
	uint sky_mode;
	float sky_energy;
	uint detail_trace_mip_count;

	vec4 sky_color;
#ifdef MODE_SVGF_PREPARE
	float denoising_range;
	float scene_to_svgf_scale;
	float input_radiance_max;
	float input_hit_distance_max;
	uint svgf_reserved[4];
#endif
}
params;
#endif

struct ScreenProbeCascadeData {
	vec3 position;
	float to_probe;

	ivec3 region_world_offset;
	float to_cell;

	vec3 blend_position;
	float exposure_normalization;

	uvec4 pad2;
};

struct ScreenProbeSceneData {
	mat4 inv_projection[2];
	mat4 cam_transform;
	mat4 projection[2];
	mat3 radiance_inverse_xform;
	mat4 previous_cam_inv_transform;
	mat4 previous_inv_projection[2];
	mat4 temporal_projection[2];
	mat4 previous_temporal_projection[2];
};

#ifdef MODE_SURFACE

layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 2) uniform sampler nearest_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform restrict writeonly uimage2D screen_probe_surface_output;

#endif

#if defined(MODE_TRACE) || defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)

#ifdef MODE_TRACE
layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D screen_probe_surface_input;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D raw_radiance_output;
#endif
layout(rg32ui, set = 0, binding = 2) uniform restrict readonly uimage3D hddagi_voxel_cascades;
layout(r8ui, set = 0, binding = 3) uniform restrict readonly uimage3D hddagi_voxel_region_cascades;
layout(set = 0, binding = 4) uniform texture3D hddagi_light_cascades;
layout(set = 0, binding = 5) uniform sampler linear_sampler;
layout(r32ui, set = 0, binding = 6) uniform restrict readonly uimage3D hddagi_voxel_neighbours;
layout(set = 0, binding = 7, std140) uniform HDDAGIData {
	ivec3 grid_size;
	int max_cascades;

	float normal_bias;
	float energy;
	float y_mult;
	float pad1;

	ivec3 probe_axis_size;
	float esm_strength;

	ivec4 pad3;

	ScreenProbeCascadeData cascades[8];
}
hddagi;
layout(r8ui, set = 0, binding = 8) uniform restrict readonly uimage3D hddagi_voxel_disocclusion;
layout(set = 0, binding = 9, std140) uniform SceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
#ifdef MODE_TRACE
layout(set = 0, binding = 10) uniform texture2D detail_hiz_buffer;
layout(set = 0, binding = 11) uniform texture2D detail_normal_roughness_buffer;
#endif
#ifdef MODE_DIRECTIONAL_TRACE
layout(set = 0, binding = 12) uniform texture2D directional_previous_radiance_input;
layout(rgba32ui, set = 0, binding = 13) uniform restrict readonly uimage2D directional_previous_surface_input;
layout(r8ui, set = 0, binding = 14) uniform restrict readonly uimage2D directional_previous_history_age_input;
layout(r8ui, set = 0, binding = 15) uniform restrict writeonly uimage2D directional_history_age_output;
#endif
#ifdef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
layout(r16ui, set = 0, binding = 10) uniform restrict readonly uimage3D hddagi_albedo_cascades;
#endif

#ifdef USE_RADIANCE_OCTMAP_ARRAY
layout(set = 1, binding = 0) uniform texture2DArray sky_radiance;
#else
layout(set = 1, binding = 0) uniform texture2D sky_radiance;
#endif
layout(set = 1, binding = 1) uniform sampler sky_sampler;
#ifdef MODE_TRACE
layout(set = 1, binding = 2) uniform texture2DArray hddagi_lightprobe_specular;
#endif

#endif

#ifdef MODE_DIRECTIONAL_FILTER

layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D directional_filter_surface_input;
layout(set = 0, binding = 1) uniform texture2D directional_filter_source_input;
layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D directional_filter_output;
layout(set = 0, binding = 3) uniform sampler directional_filter_nearest_sampler;
layout(set = 0, binding = 4, std140) uniform DirectionalFilterSceneDataBuffer {
	ScreenProbeSceneData scene_data;
};

#endif

#ifdef MODE_DIRECTIONAL_IRRADIANCE

layout(set = 0, binding = 0) uniform texture2D directional_irradiance_source_input;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D directional_irradiance_output;
layout(set = 0, binding = 2) uniform sampler directional_irradiance_nearest_sampler;

#endif

#ifdef MODE_RESOLVE

layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D screen_probe_surface_input;
layout(set = 0, binding = 1) uniform texture2D raw_radiance_input;
layout(set = 0, binding = 2) uniform texture2D depth_buffer;
layout(set = 0, binding = 3) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D resolved_radiance_output;
layout(set = 0, binding = 7, std140) uniform SceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
#ifdef MODE_DIRECTIONAL_RESOLVE
layout(set = 0, binding = 8) uniform texture2D directional_base_ambient_input;
#endif
#ifdef MODE_SVGF_PREPARE
layout(rgba8, set = 0, binding = 6) uniform restrict writeonly image2D svgf_normal_roughness_output;
layout(set = 0, binding = 9) uniform texture2D velocity_buffer;
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image2D svgf_signal_output;
layout(r32f, set = 0, binding = 11) uniform restrict writeonly image2D svgf_view_z_output;
layout(rgba16f, set = 0, binding = 12) uniform restrict writeonly image2D svgf_motion_output;
#endif

#endif

#ifdef MODE_APPLY

layout(set = 0, binding = 0) uniform texture2D resolved_radiance_input;
layout(set = 0, binding = 1) uniform sampler nearest_sampler;
layout(r32ui, set = 0, binding = 2) uniform restrict writeonly uimage2D ambient_output;
layout(set = 0, binding = 3) uniform texture2D base_ambient_input;

#endif

uint hash_uvec3(uvec3 value) {
	value = value * 1664525u + 1013904223u;
	value.x += value.y * value.z;
	value.y += value.z * value.x;
	value.z += value.x * value.y;
	value ^= value >> 16u;
	value.x += value.y * value.z;
	return value.x ^ value.y ^ value.z;
}

float uint_to_unit_float(uint value) {
	return float(value >> 8u) * (1.0 / 16777216.0);
}

vec2 sample_r2_sequence(uvec2 position, uint sequence_index) {
	const uvec2 r2_increment = uvec2(3242174889u, 2447445413u);
	uvec2 scramble = uvec2(
			hash_uvec3(uvec3(position, 0xe145f4edu)),
			hash_uvec3(uvec3(position, 0x0c8f49b7u)));
	uvec2 sequence = scramble + uvec2(sequence_index) * r2_increment;
	return vec2(uint_to_unit_float(sequence.x), uint_to_unit_float(sequence.y));
}

vec3 cosine_sample_hemisphere(vec2 sample_position) {
	float radius = sqrt(sample_position.x);
	float angle = TAU * sample_position.y;
	return vec3(radius * cos(angle), radius * sin(angle), sqrt(max(0.0, 1.0 - sample_position.x)));
}

vec3 tangent_to_world(vec3 local_direction, vec3 normal) {
	vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(up, normal));
	vec3 bitangent = cross(normal, tangent);
	return normalize(tangent * local_direction.x + bitangent * local_direction.y + normal * local_direction.z);
}

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_FILTER) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_DIRECTIONAL_RESOLVE)

bool directional_finite(vec3 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

bool directional_finite(vec4 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

vec4 directional_sanitize_fp16(vec4 value) {
	if (!directional_finite(value)) {
		return vec4(0.0);
	}
	return clamp(value, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX));
}

vec3 directional_clamp_radiance(vec3 value) {
	if (!directional_finite(value)) {
		return vec3(0.0);
	}
	value = max(value, vec3(0.0));
	float maximum_channel = max(value.r, max(value.g, value.b));
	if (maximum_channel > SCREEN_PROBE_DIRECTIONAL_RADIANCE_MAX) {
		value *= SCREEN_PROBE_DIRECTIONAL_RADIANCE_MAX / maximum_channel;
	}
	return value;
}

vec3 directional_square_to_sphere(vec2 uv) {
	vec2 square = uv * 2.0 - 1.0;
	float diagonal = 1.0 - (abs(square.x) + abs(square.y));
	float radius = 1.0 - abs(diagonal);
	float angle = radius > 1e-8 ? (TAU * 0.125) * ((abs(square.y) - abs(square.x)) / radius + 1.0) : 0.0;
	float planar_radius = radius * sqrt(max(2.0 - radius * radius, 0.0));
	vec2 square_sign = vec2(square.x < 0.0 ? -1.0 : 1.0, square.y < 0.0 ? -1.0 : 1.0);
	float diagonal_sign = diagonal < 0.0 ? -1.0 : 1.0;
	return normalize(vec3(
			planar_radius * square_sign.x * abs(cos(angle)),
			planar_radius * square_sign.y * abs(sin(angle)),
			diagonal_sign * (1.0 - radius * radius)));
}

vec3 directional_bucket_to_world(ivec2 bucket) {
	vec2 bucket_uv = (vec2(bucket) + vec2(0.5)) / float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE);
	return directional_square_to_sphere(bucket_uv);
}

vec2 directional_bucket_jitter(ivec2 probe_position) {
	uvec2 random_seed;
	random_seed.x = hash_uvec3(uvec3(uvec2(probe_position), 0x68bc21ebu));
	random_seed.y = hash_uvec3(uvec3(uvec2(probe_position.yx), 0x02e5be93u));
	random_seed &= uvec2(0xffffu);
	uint phase = params.frame_index & 7u;
	float sequence_x = fract(float(phase) * (1.0 / 8.0) + float(random_seed.x) * (1.0 / 65536.0));
	float sequence_y = float((bitfieldReverse(phase) >> 16u) ^ random_seed.y) * (1.0 / 65536.0);
	return vec2(sequence_x, sequence_y);
}

vec3 directional_jittered_bucket_to_world(ivec2 probe_position, ivec2 bucket) {
	vec2 bucket_uv = (vec2(bucket) + directional_bucket_jitter(probe_position)) / float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE);
	return directional_square_to_sphere(bucket_uv);
}

vec2 directional_sphere_to_square(vec3 direction) {
	float direction_length_squared = dot(direction, direction);
	if (!directional_finite(direction) || !(direction_length_squared > 1e-8)) {
		return vec2(0.5);
	}
	direction *= inversesqrt(direction_length_squared);
	vec3 absolute_direction = abs(direction);
	float radius = sqrt(max(1.0 - absolute_direction.z, 0.0));
	float ratio = min(absolute_direction.x, absolute_direction.y) / max(max(absolute_direction.x, absolute_direction.y), 5.42101086243e-20);
	float angle = atan(ratio) * (4.0 / TAU);
	if (absolute_direction.x < absolute_direction.y) {
		angle = 1.0 - angle;
	}
	vec2 square = vec2(radius * (1.0 - angle), radius * angle);
	if (direction.z < 0.0) {
		square = vec2(1.0 - square.y, 1.0 - square.x);
	}
	square *= vec2(direction.x < 0.0 ? -1.0 : 1.0, direction.y < 0.0 ? -1.0 : 1.0);
	return clamp(square * 0.5 + 0.5, vec2(0.0), vec2(1.0));
}

ivec2 directional_wrap_texel(ivec2 texel) {
	const int size = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	if (texel.x < 0) {
		texel.x = -texel.x - 1;
		texel.y = size - 1 - texel.y;
	} else if (texel.x >= size) {
		texel.x = 2 * size - texel.x - 1;
		texel.y = size - 1 - texel.y;
	}
	if (texel.y < 0) {
		texel.y = -texel.y - 1;
		texel.x = size - 1 - texel.x;
	} else if (texel.y >= size) {
		texel.y = 2 * size - texel.y - 1;
		texel.x = size - 1 - texel.x;
	}
	return clamp(texel, ivec2(0), ivec2(size - 1));
}

#endif

uint rgbe_encode(vec3 rgb) {
	const float rgbe_max = uintBitsToFloat(0x477f8000);
	const float rgbe_min = uintBitsToFloat(0x37800000);
	rgb = clamp(rgb, vec3(0.0), vec3(rgbe_max));
	float max_channel = max(max(rgbe_min, rgb.r), max(rgb.g, rgb.b));
	float bias = uintBitsToFloat((floatBitsToUint(max_channel) + 0x07804000u) & 0x7f800000u);
	uvec3 encoded_rgb = floatBitsToUint(rgb + bias);
	uint exponent = (floatBitsToUint(bias) << 4u) + 0x10000000u;
	return exponent | (encoded_rgb.b << 18u) | (encoded_rgb.g << 9u) | (encoded_rgb.r & 0x1ffu);
}

#if defined(MODE_TRACE) || defined(MODE_RESOLVE) || defined(MODE_DIRECTIONAL_FILTER)

vec3 compute_view_position(vec3 screen_position) {
	vec4 position = vec4(screen_position.xy * 2.0 - 1.0, screen_position.z, 1.0);
	position = scene_data.inv_projection[params.view_index] * position;
	return position.xyz / position.w;
}

#endif

bool decode_normal(vec3 encoded, out vec3 r_normal) {
	r_normal = encoded * 2.0 - 1.0;
	float length_squared = dot(r_normal, r_normal);
	if (!(length_squared > 0.001) || any(isnan(r_normal)) || any(isinf(r_normal))) {
		return false;
	}
	r_normal *= inversesqrt(length_squared);
	return true;
}

uint pack_surface_normal(vec3 normal, bool dynamic_surface) {
	float length_squared = dot(normal, normal);
	if (!(length_squared > 1e-10) || any(isnan(normal)) || any(isinf(normal))) {
		normal = vec3(0.0, 0.0, 1.0);
	} else {
		normal *= inversesqrt(length_squared);
	}
	vec2 octahedral = clamp(vec3_to_oct(normal), vec2(0.0), vec2(1.0));
	uvec2 packed = uvec2(roundEven(octahedral * vec2(65535.0, 32767.0)));
	return packed.x | (packed.y << 16u) | (dynamic_surface ? 0x80000000u : 0u);
}

vec3 unpack_surface_normal(uint packed) {
	vec2 octahedral = vec2(float(packed & 0xffffu) / 65535.0, float((packed >> 16u) & 0x7fffu) / 32767.0);
	return oct_to_vec3(octahedral * 2.0 - 1.0);
}

bool surface_is_dynamic(uint packed) {
	return (packed & 0x80000000u) != 0u;
}

#ifdef MODE_SURFACE

ivec2 gi_to_screen(ivec2 gi_position) {
	return clamp(gi_position * params.screen_size / params.gi_size, ivec2(0), params.screen_size - ivec2(1));
}

bool load_surface(ivec2 screen_position, out float r_depth, out vec3 r_normal, out bool r_dynamic) {
	if (any(lessThan(screen_position, ivec2(0))) || any(greaterThanEqual(screen_position, params.screen_size))) {
		return false;
	}
	r_depth = texelFetch(sampler2D(depth_buffer, nearest_sampler), screen_position, 0).r;
	if (!(r_depth > 0.0)) {
		return false;
	}
	vec4 normal_roughness = texelFetch(sampler2D(normal_roughness_buffer, nearest_sampler), screen_position, 0);
	r_dynamic = normal_roughness.a > 0.5;
	return decode_normal(normal_roughness.xyz, r_normal);
}

void screen_probe_surface_main() {
	ivec2 probe_position = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(probe_position, imageSize(screen_probe_surface_output)))) {
		return;
	}

	ivec2 gi_begin = probe_position * params.probe_size;
	ivec2 gi_end = min(gi_begin + ivec2(params.probe_size), params.gi_size);
	ivec2 tile_extent = gi_end - gi_begin;
	ivec2 best_screen_position = ivec2(0);
	ivec2 best_gi_position = ivec2(0);
	float best_depth = 0.0;
	vec3 best_normal = vec3(0.0);
	bool best_dynamic = false;
	uint best_distance_squared = 0xffffffffu;
	bool found = false;

	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_SURFACE_FOOTPRINT) != 0u) {
		ivec2 screen_begin = (gi_begin * params.screen_size + params.gi_size - ivec2(1)) / params.gi_size;
		ivec2 screen_end = (gi_end * params.screen_size + params.gi_size - ivec2(1)) / params.gi_size;
		screen_begin = clamp(screen_begin, ivec2(0), params.screen_size - ivec2(1));
		screen_end = clamp(screen_end, screen_begin + ivec2(1), params.screen_size);
		ivec2 screen_center_twice = screen_begin + screen_end - ivec2(1);
		for (int y = screen_begin.y; y < screen_end.y; y++) {
			for (int x = screen_begin.x; x < screen_end.x; x++) {
				ivec2 screen_position = ivec2(x, y);
				float depth;
				vec3 normal;
				bool dynamic_surface;
				if (!load_surface(screen_position, depth, normal, dynamic_surface)) {
					continue;
				}
				ivec2 center_delta_twice = screen_position * 2 - screen_center_twice;
				uint distance_squared = uint(dot(center_delta_twice, center_delta_twice));
				bool wins_tie = distance_squared == best_distance_squared &&
						(screen_position.y < best_screen_position.y || (screen_position.y == best_screen_position.y && screen_position.x < best_screen_position.x));
				if (!found || distance_squared < best_distance_squared || wins_tie) {
					found = true;
					best_distance_squared = distance_squared;
					best_screen_position = screen_position;
					best_depth = depth;
					best_normal = normal;
					best_dynamic = dynamic_surface;
				}
			}
		}
	} else {
		ivec2 tile_center_twice = tile_extent - ivec2(1);
		for (int y = 0; y < tile_extent.y; y++) {
			for (int x = 0; x < tile_extent.x; x++) {
				ivec2 gi_position = gi_begin + ivec2(x, y);
				ivec2 screen_position = gi_to_screen(gi_position);
				float depth;
				vec3 normal;
				bool dynamic_surface;
				if (!load_surface(screen_position, depth, normal, dynamic_surface)) {
					continue;
				}
				ivec2 center_delta_twice = ivec2(x, y) * 2 - tile_center_twice;
				uint distance_squared = uint(dot(center_delta_twice, center_delta_twice));
				bool wins_tie = distance_squared == best_distance_squared &&
						(gi_position.y < best_gi_position.y || (gi_position.y == best_gi_position.y && gi_position.x < best_gi_position.x));
				if (!found || distance_squared < best_distance_squared || wins_tie) {
					found = true;
					best_distance_squared = distance_squared;
					best_gi_position = gi_position;
					best_screen_position = screen_position;
					best_depth = depth;
					best_normal = normal;
					best_dynamic = dynamic_surface;
				}
			}
		}
	}

	if (!found) {
		imageStore(screen_probe_surface_output, probe_position, uvec4(0xffffffffu));
		return;
	}
	imageStore(screen_probe_surface_output, probe_position, uvec4(uvec2(best_screen_position), floatBitsToUint(best_depth), pack_surface_normal(best_normal, best_dynamic)));
}

#endif

#ifdef MODE_TRACE

bool detail_trace_project(vec3 view_position, out vec3 r_screen_position) {
	vec4 clip = scene_data.projection[params.view_index] * vec4(view_position, 1.0);
	if (!(clip.w > 1e-6) || any(isnan(clip)) || any(isinf(clip))) {
		r_screen_position = vec3(0.0);
		return false;
	}
	r_screen_position = vec3(clip.xy / clip.w * 0.5 + 0.5, clip.z / clip.w);
	return !any(isnan(r_screen_position)) && !any(isinf(r_screen_position));
}

bool detail_trace_load_view_position(ivec2 pixel, out vec3 r_view_position) {
	if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, params.screen_size))) {
		r_view_position = vec3(0.0);
		return false;
	}
	float depth = texelFetch(sampler2D(detail_hiz_buffer, linear_sampler), pixel, 0).r;
	if (!(depth > 0.0)) {
		r_view_position = vec3(0.0);
		return false;
	}
	vec2 uv = (vec2(pixel) + 0.5) / vec2(params.screen_size);
	r_view_position = compute_view_position(vec3(uv, depth));
	return !any(isnan(r_view_position)) && !any(isinf(r_view_position));
}

bool detail_trace_geometric_normal(ivec2 pixel, vec3 center_view, vec3 shading_normal_view, bool shading_normal_valid, out vec3 r_normal_view) {
	vec3 left_view;
	vec3 right_view;
	vec3 up_view;
	vec3 down_view;
	bool left_valid = detail_trace_load_view_position(pixel + ivec2(-1, 0), left_view);
	bool right_valid = detail_trace_load_view_position(pixel + ivec2(1, 0), right_view);
	bool up_valid = detail_trace_load_view_position(pixel + ivec2(0, -1), up_view);
	bool down_valid = detail_trace_load_view_position(pixel + ivec2(0, 1), down_view);

	vec3 horizontal_derivative = vec3(0.0);
	vec3 vertical_derivative = vec3(0.0);
	bool horizontal_valid = left_valid || right_valid;
	bool vertical_valid = up_valid || down_valid;
	if (horizontal_valid) {
		bool use_left = left_valid && (!right_valid || abs(left_view.z - center_view.z) <= abs(right_view.z - center_view.z));
		horizontal_derivative = use_left ? center_view - left_view : right_view - center_view;
	}
	if (vertical_valid) {
		bool use_up = up_valid && (!down_valid || abs(up_view.z - center_view.z) <= abs(down_view.z - center_view.z));
		vertical_derivative = use_up ? center_view - up_view : down_view - center_view;
	}

	vec3 geometric_normal = cross(vertical_derivative, horizontal_derivative);
	float length_squared = dot(geometric_normal, geometric_normal);
	if (horizontal_valid && vertical_valid && length_squared > 1e-12 && !any(isnan(geometric_normal)) && !any(isinf(geometric_normal))) {
		geometric_normal *= inversesqrt(length_squared);
		if (shading_normal_valid) {
			geometric_normal *= dot(geometric_normal, shading_normal_view) < 0.0 ? -1.0 : 1.0;
		} else if (dot(geometric_normal, center_view) > 0.0) {
			geometric_normal = -geometric_normal;
		}
		r_normal_view = geometric_normal;
		return true;
	}
	if (shading_normal_valid) {
		r_normal_view = shading_normal_view;
		return true;
	}
	r_normal_view = vec3(0.0, 0.0, 1.0);
	return false;
}

bool trace_screen_detail(vec3 origin_view, vec3 ray_direction_view, float distance_limit, out vec3 r_endpoint_world, out vec3 r_endpoint_normal_world) {
	r_endpoint_world = vec3(0.0);
	r_endpoint_normal_world = vec3(0.0, 0.0, 1.0);

	float max_distance = min(SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE, distance_limit);
	if ((params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) == 0u || !(max_distance > 0.0) || params.detail_trace_mip_count == 0u || any(lessThanEqual(params.screen_size, ivec2(0)))) {
		return false;
	}
	if (any(isnan(ray_direction_view)) || any(isinf(ray_direction_view))) {
		return false;
	}
	if (ray_direction_view.z > 0.0) {
		max_distance = min(max_distance, (-0.001 - origin_view.z) / ray_direction_view.z);
	}
	if (!(max_distance > 0.02)) {
		return false;
	}

	vec3 screen_start;
	vec3 screen_end;
	if (!detail_trace_project(origin_view, screen_start) || !detail_trace_project(origin_view + ray_direction_view * max_distance, screen_end)) {
		return false;
	}
	vec3 screen_delta = screen_end - screen_start;
	if (abs(screen_delta.z) < 1e-5 || all(lessThan(abs(screen_delta.xy), vec2(1e-7)))) {
		return false;
	}

	float segment_t = abs(screen_delta.z);
	vec3 screen_direction = screen_delta / segment_t;
	bool facing_camera = screen_direction.z >= 0.0;
	vec2 direction_xy = screen_direction.xy;
	vec2 safe_direction_xy = mix(direction_xy, vec2(1e-20), lessThan(abs(direction_xy), vec2(1e-20)));
	vec2 inverse_direction_xy = 1.0 / safe_direction_xy;
	vec2 cell_step = mix(vec2(1.0), vec2(-1.0), lessThan(direction_xy, vec2(0.0)));
	vec2 positive_cell_step = clamp(cell_step, vec2(0.0), vec2(1.0));
	vec2 cell_boundary_bias = cell_step * 1e-6;
	vec2 detail_size = vec2(params.screen_size);
	vec2 inverse_detail_size = 1.0 / detail_size;

	vec2 edge_t0 = -screen_start.xy * inverse_direction_xy;
	vec2 edge_t1 = (vec2(1.0) - screen_start.xy) * inverse_direction_xy;
	vec2 positive_edge_t = max(edge_t0, edge_t1);
	float trace_t_max = min(segment_t, min(positive_edge_t.x, positive_edge_t.y));
	if (!(trace_t_max > 0.0)) {
		return false;
	}

	vec2 start_cell = floor(screen_start.xy * detail_size);
	vec2 next_cell_uv = (start_cell + positive_cell_step) * inverse_detail_size + cell_boundary_bias;
	vec2 start_cell_t = (next_cell_uv - screen_start.xy) * inverse_direction_xy;
	float trace_t = max(min(start_cell_t.x, start_cell_t.y), 0.0);
	int mip_level = 0;
	int iterations_remaining = 48;
	int max_mip = int(params.detail_trace_mip_count) - 1;
	float receiver_exclusion = max(0.02, abs(origin_view.z) * 2.0 / detail_size.y);

	while (mip_level >= 0 && iterations_remaining > 0 && trace_t < trace_t_max) {
		vec3 current_screen = screen_start + screen_direction * trace_t;
		ivec2 mip_size = max(params.screen_size >> mip_level, ivec2(1));
		ivec2 cell_index = clamp(ivec2(floor(current_screen.xy * vec2(mip_size))), ivec2(0), mip_size - ivec2(1));
		float cell_depth = texelFetch(sampler2D(detail_hiz_buffer, linear_sampler), cell_index, mip_level).r;

		vec2 next_mip_uv = (vec2(cell_index) + positive_cell_step) / vec2(mip_size) + cell_boundary_bias;
		vec2 mip_cell_t = (next_mip_uv - screen_start.xy) * inverse_direction_xy;
		float cell_exit_t = min(mip_cell_t.x, mip_cell_t.y);

		bool possible_hit = cell_depth > 0.0;
		float depth_t = trace_t;
		if (possible_hit) {
			depth_t = (cell_depth - screen_start.z) / screen_direction.z;
			possible_hit = facing_camera ? trace_t <= depth_t : depth_t <= cell_exit_t;
		}

		if (possible_hit && mip_level == 0) {
			float candidate_t = facing_camera ? trace_t : max(trace_t, depth_t);
			vec3 candidate_screen = screen_start + screen_direction * candidate_t;
			ivec2 candidate_pixel = ivec2(floor(candidate_screen.xy * detail_size));
			bool candidate_valid = all(greaterThanEqual(candidate_pixel, ivec2(1))) && all(lessThan(candidate_pixel, params.screen_size - ivec2(1)));
			vec3 candidate_view = vec3(0.0);
			vec3 candidate_normal_view = vec3(0.0, 0.0, 1.0);
			if (candidate_valid) {
				float candidate_depth = any(notEqual(candidate_pixel, cell_index)) ? texelFetch(sampler2D(detail_hiz_buffer, linear_sampler), candidate_pixel, 0).r : cell_depth;
				vec3 shading_normal_view;
				bool shading_normal_valid = decode_normal(texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), candidate_pixel, 0).xyz, shading_normal_view);
				candidate_valid = candidate_depth > 0.0;
				if (candidate_valid) {
					vec2 candidate_uv = (vec2(candidate_pixel) + 0.5) * inverse_detail_size;
					candidate_view = compute_view_position(vec3(candidate_uv, candidate_depth));
					candidate_valid = !any(isnan(candidate_view)) && !any(isinf(candidate_view)) && detail_trace_geometric_normal(candidate_pixel, candidate_view, shading_normal_view, shading_normal_valid, candidate_normal_view) && dot(ray_direction_view, candidate_normal_view) < -0.05;
				}
				if (candidate_valid) {
					float candidate_distance = dot(candidate_view - origin_view, ray_direction_view);
					float thickness = clamp(abs(candidate_view.z) * 0.0015, 0.025, 0.10);
					vec3 ray_closest_view = origin_view + ray_direction_view * candidate_distance;
					vec3 candidate_ray_view = compute_view_position(candidate_screen);
					vec3 ray_error = candidate_view - ray_closest_view;
					float normal_thickness_error = abs(dot(candidate_view - candidate_ray_view, candidate_normal_view));
					candidate_valid = candidate_distance > receiver_exclusion && candidate_distance <= max_distance && dot(ray_error, ray_error) <= thickness * thickness && normal_thickness_error <= thickness && !any(isnan(candidate_ray_view)) && !any(isinf(candidate_ray_view));
				}
			}

			if (candidate_valid) {
				vec3 endpoint_world = (scene_data.cam_transform * vec4(candidate_view, 1.0)).xyz;
				vec3 endpoint_normal_world = normalize(mat3(scene_data.cam_transform) * candidate_normal_view);
				if (!any(isnan(endpoint_world)) && !any(isinf(endpoint_world)) && !any(isnan(endpoint_normal_world)) && !any(isinf(endpoint_normal_world))) {
					r_endpoint_world = endpoint_world;
					r_endpoint_normal_world = endpoint_normal_world;
					return true;
				}
			}

			trace_t = max(cell_exit_t, candidate_t + 1e-6);
			mip_level = 0;
			iterations_remaining--;
			continue;
		}
		if (possible_hit) {
			mip_level--;
			if (!facing_camera) {
				trace_t = max(trace_t, depth_t);
			}
		} else {
			trace_t = max(cell_exit_t, trace_t + 1e-6);
			mip_level = min(mip_level + 1, max_mip);
		}
		iterations_remaining--;
	}
	return false;
}

#endif

#if defined(MODE_TRACE) || defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)

#ifndef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
uvec4 load_probe_surface_texel(ivec2 position) {
	return imageLoad(screen_probe_surface_input, position);
}
#endif

uvec2 load_voxel_cascade(ivec3 position) {
	return imageLoad(hddagi_voxel_cascades, position).rg;
}

uint load_voxel_region(ivec3 position) {
	return imageLoad(hddagi_voxel_region_cascades, position).r;
}

uint load_voxel_neighbours(ivec3 position) {
	return imageLoad(hddagi_voxel_neighbours, position).r;
}

uint load_voxel_disocclusion(ivec3 position) {
	return imageLoad(hddagi_voxel_disocclusion, position).r;
}

#ifndef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
bool load_probe_surface(ivec2 probe_position, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal) {
	uvec4 packed = load_probe_surface_texel(probe_position);
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	return r_depth > 0.0 && all(greaterThanEqual(r_screen_position, ivec2(0))) && all(lessThan(r_screen_position, params.screen_size));
}
#endif

float point_to_ray_distance(vec3 point, vec3 ray_origin, vec3 ray_direction) {
	vec3 point_to_ray = point - ray_origin;
	vec3 projected = ray_origin + ray_direction * dot(point_to_ray, ray_direction);
	return length(point - projected);
}

bool trace_ray_hdda(vec3 ray_position, vec3 ray_direction, int first_cascade, bool exclude_initial_cell, out ivec3 r_cell, out ivec3 r_side, out int r_cascade) {
	const int level_cascade = -1;
	const int level_region = 0;
	const int level_block = 1;
	const int level_voxel = 2;
	const int max_level = 3;
	const int fp_bits = HDDAGI_HDDA_FP_BITS;
	const int fp_block_bits = fp_bits + 2;
	const int fp_region_bits = fp_block_bits + 1;

	bvec3 limit_direction = greaterThan(ray_direction, vec3(0.0));
	ivec3 step = mix(ivec3(0), ivec3(1), limit_direction);
	ivec3 ray_sign = ivec3(sign(ray_direction));
	bvec3 ray_zero = lessThan(abs(ray_direction), vec3(1.0 / 127.0));
	ivec3 ray_direction_fp = ivec3(ray_direction * float(1 << fp_bits));
	vec3 reciprocal_divisor = mix(ray_direction, vec3(1.0), ray_zero);
	ivec3 inverse_ray_direction_fp = ivec3(vec3(float(1 << fp_bits)) / reciprocal_divisor);
	const ivec3 level_masks[max_level] = ivec3[](
			ivec3(1 << fp_region_bits) - ivec3(1),
			ivec3(1 << fp_block_bits) - ivec3(1),
			ivec3(1 << fp_bits) - ivec3(1));

	ivec3 region_offset_mask = hddagi.grid_size / HDDAGI_REGION_SIZE - ivec3(1);
	ivec3 limits[max_level];
	limits[level_region] = ((hddagi.grid_size << fp_bits) - ivec3(1)) * step;
	int level = level_cascade;
	int cascade = first_cascade - 1;
	ivec3 cascade_base;
	ivec3 region_base;
	uvec2 block;
	bool hit = false;
	ivec3 position;
	int excluded_cascade = -1;
	ivec3 excluded_cell = ivec3(0);

	while (true) {
		if (level == level_voxel) {
			ivec3 block_local = (position & level_masks[level_block]) >> fp_bits;
			uint block_index = uint(block_local.z * 16 + block_local.y * 4 + block_local.x);
			bool occupied;
			if (block_index < 32u) {
				occupied = bool(block.x & (1u << block_index));
			} else {
				block_index -= 32u;
				occupied = bool(block.y & (1u << block_index));
			}
			if (occupied) {
				ivec3 current_cell = position >> fp_bits;
				if (cascade != excluded_cascade || any(notEqual(current_cell, excluded_cell))) {
					hit = true;
					break;
				}
			}
		} else if (level == level_block) {
			ivec3 block_local = (position & level_masks[level_region]) >> fp_block_bits;
			block = load_voxel_cascade(region_base + block_local);
			if (block != uvec2(0)) {
				level = level_voxel;
				limits[level_voxel] = position - (position & level_masks[level_block]) + step * (level_masks[level_block] + ivec3(1));
				continue;
			}
		} else if (level == level_region) {
			ivec3 region = position >> fp_region_bits;
			region = (hddagi.cascades[cascade].region_world_offset + region) & region_offset_mask;
			region += cascade_base;
			if (load_voxel_region(region) > 0u) {
				region_base = region << 1;
				level = level_block;
				limits[level_block] = position - (position & level_masks[level_region]) + step * (level_masks[level_region] + ivec3(1));
				continue;
			}
		} else {
			if (cascade >= first_cascade) {
				ray_position = vec3(position) / float(1 << fp_bits);
				ray_position /= hddagi.cascades[cascade].to_cell;
				ray_position += hddagi.cascades[cascade].position;
			}
			cascade++;
			if (cascade == hddagi.max_cascades) {
				break;
			}
			ray_position = (ray_position - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
			position = ivec3(ray_position * float(1 << fp_bits));
			if (any(lessThan(position, ivec3(0))) || any(greaterThanEqual(position, hddagi.grid_size << fp_bits))) {
				continue;
			}
			if (exclude_initial_cell && excluded_cascade < 0) {
				excluded_cascade = cascade;
				excluded_cell = position >> fp_bits;
			}
			cascade_base = ivec3(0, hddagi.grid_size.y / HDDAGI_REGION_SIZE * cascade, 0);
			level = level_region;
			continue;
		}

		ivec3 mask = level_masks[level];
		ivec3 box = mask * step;
		ivec3 position_difference = box - (position & mask);
		ivec3 multiplied = (position_difference * inverse_ray_direction_fp) >> fp_bits;
		ivec3 axis_time = mix(multiplied, ivec3(0x7fffffff), ray_zero);
		int advance_time = min(axis_time.x, min(axis_time.y, axis_time.z));
		ivec3 box_advance = position_difference + ray_sign;
		ivec3 ray_advance = (ray_direction_fp * advance_time) >> fp_bits;
		position += mix(ray_advance, box_advance, equal(ivec3(advance_time), axis_time));
		while (true) {
			bvec3 limit = lessThan(position, limits[level]);
			if (all(equal(limit, limit_direction))) {
				break;
			}
			level--;
			if (level == level_cascade) {
				break;
			}
		}
	}

	if (hit) {
		ivec3 mask = level_masks[level_voxel];
		ivec3 box = mask * (step ^ ivec3(1));
		ivec3 position_difference = box - (position & mask);
		ivec3 multiplied = position_difference * -inverse_ray_direction_fp;
		ivec3 axis_time = mix(multiplied, ivec3(0x7fffffff), ray_zero);
		int minimum_time;
		if (axis_time.x < axis_time.y) {
			r_side = ivec3(1, 0, 0);
			minimum_time = axis_time.x;
		} else {
			r_side = ivec3(0, 1, 0);
			minimum_time = axis_time.y;
		}
		if (axis_time.z < minimum_time) {
			r_side = ivec3(0, 0, 1);
		}
		r_side *= -ray_sign;
		r_cell = position >> fp_bits;
		r_cascade = cascade;
	}
	return hit;
}

ivec3 light_texture_position(ivec3 light_cell, int cascade) {
	ivec3 read_cell = (light_cell + hddagi.cascades[cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
	return read_cell + ivec3(0, hddagi.grid_size.y * cascade, 0);
}

bool sample_endpoint_light(ivec3 light_cell, int cascade, vec3 endpoint_cell, vec3 normal_scaled, out vec3 r_light, out float r_weight, out uint r_metadata) {
	r_light = vec3(0.0);
	r_weight = 0.0;
	r_metadata = 0u;
	if (any(lessThan(light_cell, ivec3(0))) || any(greaterThanEqual(light_cell, hddagi.grid_size))) {
		return false;
	}

	ivec3 texture_position = light_texture_position(light_cell, cascade);
	r_metadata = load_voxel_neighbours(texture_position);
	if (!bool(r_metadata & HDDAGI_LIGHT_CELL_VALID_BIT)) {
		return false;
	}

	uint disocclusion = load_voxel_disocclusion(texture_position) & 0x3fu;
	const ivec3 outward_directions[6] = ivec3[](
			ivec3(-1, 0, 0), ivec3(1, 0, 0),
			ivec3(0, -1, 0), ivec3(0, 1, 0),
			ivec3(0, 0, -1), ivec3(0, 0, 1));
	const uint inward_bits[6] = uint[](1u, 0u, 3u, 2u, 5u, 4u);
	float surface_alignment = 0.0;
	for (int i = 0; i < 6; i++) {
		bool outward_is_empty = bool(disocclusion & (1u << uint(i)));
		bool inward_is_solid = !bool(disocclusion & (1u << inward_bits[i]));
		if (outward_is_empty && inward_is_solid) {
			surface_alignment = max(surface_alignment, dot(normal_scaled, vec3(outward_directions[i])));
		}
	}
	if (!(surface_alignment >= 0.35)) {
		return false;
	}

	vec3 surface_delta = vec3(light_cell) + 0.5 - endpoint_cell;
	float normal_distance = dot(surface_delta, normal_scaled);
	vec3 tangent_delta = surface_delta - normal_scaled * normal_distance;
	float normal_weight = max(0.0, 1.0 - abs(normal_distance - 0.75) / 1.25);
	float tangent_weight = max(0.0, 1.0 - dot(tangent_delta, tangent_delta) / 2.25);
	r_weight = normal_weight * tangent_weight * surface_alignment * surface_alignment;
	if (!(r_weight > 1e-5) || isnan(r_weight) || isinf(r_weight)) {
		return false;
	}

	r_light = texelFetch(sampler3D(hddagi_light_cascades, linear_sampler), texture_position, 0).rgb;
	return !any(isnan(r_light)) && !any(isinf(r_light)) && !any(lessThan(r_light, vec3(0.0)));
}

bool query_endpoint_radiance(vec3 endpoint_world, vec3 endpoint_normal_world, vec3 trace_origin_world, out vec3 r_radiance) {
	r_radiance = vec3(0.0);
#ifdef MODE_IRRADIANCE_CACHE_QUERY
	HDDAGIIrradianceCacheLookup irradiance_cache_lookup = hddagi_irradiance_cache_lookup(endpoint_world, endpoint_normal_world, trace_origin_world);
	if (irradiance_cache_lookup.has_radiance && (hddagi_irradiance_cache_multibounce_enabled() || !irradiance_cache_lookup.needs_refresh)) {
		r_radiance = irradiance_cache_lookup.radiance;
		return true;
	}
#endif
	vec3 endpoint_scaled = endpoint_world - scene_data.cam_transform[3].xyz;
	endpoint_scaled.y *= hddagi.y_mult;
	vec3 normal_scaled = endpoint_normal_world;
	normal_scaled.y *= hddagi.y_mult;
	float normal_length_squared = dot(normal_scaled, normal_scaled);
	if (!(normal_length_squared > 1e-8) || any(isnan(endpoint_scaled)) || any(isinf(endpoint_scaled)) || any(isnan(normal_scaled)) || any(isinf(normal_scaled))) {
		return false;
	}
	normal_scaled *= inversesqrt(normal_length_squared);
	const ivec3 neighbour_directions[26] = ivec3[](
			ivec3(-1, 0, 0), ivec3(1, 0, 0), ivec3(0, -1, 0), ivec3(0, 1, 0), ivec3(0, 0, -1), ivec3(0, 0, 1),
			ivec3(-1, -1, -1), ivec3(-1, -1, 0), ivec3(-1, -1, 1), ivec3(-1, 0, -1), ivec3(-1, 0, 1),
			ivec3(-1, 1, -1), ivec3(-1, 1, 0), ivec3(-1, 1, 1), ivec3(0, -1, -1), ivec3(0, -1, 1),
			ivec3(0, 1, -1), ivec3(0, 1, 1), ivec3(1, -1, -1), ivec3(1, -1, 0), ivec3(1, -1, 1),
			ivec3(1, 0, -1), ivec3(1, 0, 1), ivec3(1, 1, -1), ivec3(1, 1, 0), ivec3(1, 1, 1));

	for (int cascade = 0; cascade < hddagi.max_cascades; cascade++) {
		vec3 endpoint_cell = (endpoint_scaled - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
		if (any(lessThan(endpoint_cell, vec3(0.0))) || any(greaterThanEqual(endpoint_cell, vec3(hddagi.grid_size)))) {
			continue;
		}

		ivec3 search_center = ivec3(floor(endpoint_cell + normal_scaled * 0.75));
		ivec3 anchor_cell = ivec3(0);
		vec3 anchor_light = vec3(0.0);
		float anchor_weight = 0.0;
		uint anchor_metadata = 0u;
		bool anchor_found = false;
		for (int z = -1; z <= 1; z++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					ivec3 candidate_cell = search_center + ivec3(x, y, z);
					vec3 candidate_light;
					float candidate_weight;
					uint candidate_metadata;
					if (sample_endpoint_light(candidate_cell, cascade, endpoint_cell, normal_scaled, candidate_light, candidate_weight, candidate_metadata) && (!anchor_found || candidate_weight > anchor_weight)) {
						anchor_found = true;
						anchor_cell = candidate_cell;
						anchor_light = candidate_light;
						anchor_weight = candidate_weight;
						anchor_metadata = candidate_metadata;
					}
				}
			}
		}
		if (!anchor_found) {
			continue;
		}

		vec3 radiance_sum = anchor_light * anchor_weight;
		float weight_sum = anchor_weight;
		uint neighbour_bits = anchor_metadata & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
		while (neighbour_bits != 0u) {
			uint bit = findLSB(neighbour_bits);
			ivec3 candidate_cell = anchor_cell + neighbour_directions[bit];
			vec3 candidate_light;
			float candidate_weight;
			uint candidate_metadata;
			if (sample_endpoint_light(candidate_cell, cascade, endpoint_cell, normal_scaled, candidate_light, candidate_weight, candidate_metadata)) {
				radiance_sum += candidate_light * candidate_weight;
				weight_sum += candidate_weight;
			}
			neighbour_bits &= ~(1u << bit);
		}

		vec3 filtered_radiance = radiance_sum / weight_sum * hddagi.cascades[cascade].exposure_normalization;
		if (!any(isnan(filtered_radiance)) && !any(isinf(filtered_radiance)) && !any(lessThan(filtered_radiance, vec3(0.0)))) {
#ifdef MODE_IRRADIANCE_CACHE_QUERY
			if (hddagi_irradiance_cache_should_submit_endpoint_sample(irradiance_cache_lookup)) {
				hddagi_irradiance_cache_submit(irradiance_cache_lookup, endpoint_world, endpoint_normal_world, filtered_radiance);
			}
			r_radiance = irradiance_cache_lookup.has_radiance ? irradiance_cache_lookup.radiance : filtered_radiance;
#else
			r_radiance = filtered_radiance;
#endif
			return true;
		}
	}
#ifdef MODE_IRRADIANCE_CACHE_QUERY
	if (irradiance_cache_lookup.has_radiance) {
		r_radiance = irradiance_cache_lookup.radiance;
		return true;
	}
#endif
	return false;
}

#ifdef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE

bool sample_irradiance_cache_representative(vec3 position_world, vec3 normal_world, out vec3 r_direct_radiance, out vec3 r_albedo) {
	r_direct_radiance = vec3(0.0);
	r_albedo = vec3(0.0);
	vec3 endpoint_scaled = position_world - scene_data.cam_transform[3].xyz;
	endpoint_scaled.y *= hddagi.y_mult;
	vec3 normal_scaled = normal_world;
	normal_scaled.y *= hddagi.y_mult;
	float normal_length_squared = dot(normal_scaled, normal_scaled);
	if (!(normal_length_squared > 1e-8) || any(isnan(endpoint_scaled)) || any(isinf(endpoint_scaled)) || any(isnan(normal_scaled)) || any(isinf(normal_scaled))) {
		return false;
	}
	normal_scaled *= inversesqrt(normal_length_squared);
	const ivec3 neighbour_directions[26] = ivec3[](
			ivec3(-1, 0, 0), ivec3(1, 0, 0), ivec3(0, -1, 0), ivec3(0, 1, 0), ivec3(0, 0, -1), ivec3(0, 0, 1),
			ivec3(-1, -1, -1), ivec3(-1, -1, 0), ivec3(-1, -1, 1), ivec3(-1, 0, -1), ivec3(-1, 0, 1),
			ivec3(-1, 1, -1), ivec3(-1, 1, 0), ivec3(-1, 1, 1), ivec3(0, -1, -1), ivec3(0, -1, 1),
			ivec3(0, 1, -1), ivec3(0, 1, 1), ivec3(1, -1, -1), ivec3(1, -1, 0), ivec3(1, -1, 1),
			ivec3(1, 0, -1), ivec3(1, 0, 1), ivec3(1, 1, -1), ivec3(1, 1, 0), ivec3(1, 1, 1));

	for (int cascade = 0; cascade < hddagi.max_cascades; cascade++) {
		vec3 endpoint_cell = (endpoint_scaled - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
		if (any(lessThan(endpoint_cell, vec3(0.0))) || any(greaterThanEqual(endpoint_cell, vec3(hddagi.grid_size)))) {
			continue;
		}

		ivec3 search_center = ivec3(floor(endpoint_cell + normal_scaled * 0.75));
		ivec3 anchor_cell = ivec3(0);
		vec3 anchor_light = vec3(0.0);
		float anchor_weight = 0.0;
		uint anchor_metadata = 0u;
		bool anchor_found = false;
		for (int z = -1; z <= 1; z++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					ivec3 candidate_cell = search_center + ivec3(x, y, z);
					vec3 candidate_light;
					float candidate_weight;
					uint candidate_metadata;
					if (sample_endpoint_light(candidate_cell, cascade, endpoint_cell, normal_scaled, candidate_light, candidate_weight, candidate_metadata) && (!anchor_found || candidate_weight > anchor_weight)) {
						anchor_found = true;
						anchor_cell = candidate_cell;
						anchor_light = candidate_light;
						anchor_weight = candidate_weight;
						anchor_metadata = candidate_metadata;
					}
				}
			}
		}
		if (!anchor_found) {
			continue;
		}

		vec3 radiance_sum = anchor_light * anchor_weight;
		float weight_sum = anchor_weight;
		uint neighbour_bits = anchor_metadata & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
		while (neighbour_bits != 0u) {
			uint bit = findLSB(neighbour_bits);
			ivec3 candidate_cell = anchor_cell + neighbour_directions[bit];
			vec3 candidate_light;
			float candidate_weight;
			uint candidate_metadata;
			if (sample_endpoint_light(candidate_cell, cascade, endpoint_cell, normal_scaled, candidate_light, candidate_weight, candidate_metadata)) {
				radiance_sum += candidate_light * candidate_weight;
				weight_sum += candidate_weight;
			}
			neighbour_bits &= ~(1u << bit);
		}

		r_direct_radiance = radiance_sum / weight_sum * hddagi.cascades[cascade].exposure_normalization;
		uint packed_albedo = imageLoad(hddagi_albedo_cascades, light_texture_position(anchor_cell, cascade)).r;
		r_albedo = vec3((uvec3(packed_albedo) >> uvec3(0, 5, 11)) & uvec3(0x1f, 0x3f, 0x1f)) / vec3(0x1f, 0x3f, 0x1f);
		return !any(isnan(r_direct_radiance)) && !any(isinf(r_direct_radiance)) && !any(lessThan(r_direct_radiance, vec3(0.0))) &&
				!any(isnan(r_albedo)) && !any(isinf(r_albedo));
	}
	return false;
}

#endif

vec3 sample_environment(vec3 ray_direction_view) {
#ifdef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
	uint sky_mode = hddagi_irradiance_cache_params.training.z;
	vec4 sky = hddagi_irradiance_cache_params.sky_color_or_border_energy;
	if (sky_mode == SCREEN_PROBE_SKY_COLOR) {
		return max(sky.rgb * sky.w, vec3(0.0));
	}
	if (sky_mode != SCREEN_PROBE_SKY_TEXTURE) {
		return vec3(0.0);
	}
	float border = clamp(sky.x, 0.0, 0.499);
	float sky_energy = sky.w;
#else
	if (params.sky_mode == SCREEN_PROBE_SKY_COLOR) {
		return max(params.sky_color.rgb * params.sky_energy, vec3(0.0));
	}
	if (params.sky_mode != SCREEN_PROBE_SKY_TEXTURE) {
		return vec3(0.0);
	}

	float border = clamp(params.sky_color.w, 0.0, 0.499);
	float sky_energy = params.sky_energy;
#endif
	vec3 sky_direction = scene_data.radiance_inverse_xform * ray_direction_view;
	float direction_length_squared = dot(sky_direction, sky_direction);
	if (!(direction_length_squared > 1e-8) || any(isnan(sky_direction)) || any(isinf(sky_direction))) {
		return vec3(0.0);
	}
	sky_direction *= inversesqrt(direction_length_squared);
	vec2 sky_uv = vec3_to_oct_with_border(sky_direction, vec2(border, 1.0 - border * 2.0));
#ifdef USE_RADIANCE_OCTMAP_ARRAY
	vec3 radiance = textureLod(sampler2DArray(sky_radiance, sky_sampler), vec3(sky_uv, 0.0), 0.0).rgb * sky_energy;
#else
	vec3 radiance = textureLod(sampler2D(sky_radiance, sky_sampler), sky_uv, 0.0).rgb * sky_energy;
#endif
	if (any(isnan(radiance)) || any(isinf(radiance))) {
		return vec3(0.0);
	}
	return max(radiance, vec3(0.0));
}

#ifdef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE

bool trace_irradiance_cache_segment(vec3 origin_world, vec3 origin_normal_world, vec3 direction_world, out vec3 r_direct_radiance, out vec3 r_endpoint_world, out vec3 r_endpoint_normal_world, out vec3 r_surface_albedo) {
	r_direct_radiance = vec3(0.0);
	r_endpoint_world = vec3(0.0);
	r_endpoint_normal_world = vec3(0.0, 0.0, 1.0);
	r_surface_albedo = vec3(0.0);

	vec3 ray_position = origin_world - scene_data.cam_transform[3].xyz;
	vec3 ray_direction = normalize(direction_world);
	vec3 hddagi_normal = normalize(origin_normal_world);
	ray_position.y *= hddagi.y_mult;
	ray_direction.y *= hddagi.y_mult;
	hddagi_normal.y *= hddagi.y_mult;
	ray_direction = normalize(ray_direction);
	hddagi_normal = normalize(hddagi_normal);
	if (any(isnan(ray_position)) || any(isinf(ray_position)) || any(isnan(ray_direction)) || any(isinf(ray_direction)) || any(isnan(hddagi_normal)) || any(isinf(hddagi_normal))) {
		return false;
	}

	int cascade = hddagi.max_cascades;
	for (int candidate_cascade = 0; candidate_cascade < hddagi.max_cascades; candidate_cascade++) {
		vec3 cascade_position = (ray_position - hddagi.cascades[candidate_cascade].position) * hddagi.cascades[candidate_cascade].to_cell;
		if (all(greaterThanEqual(cascade_position, vec3(0.0))) && all(lessThan(cascade_position, vec3(hddagi.grid_size)))) {
			cascade = candidate_cascade;
			break;
		}
	}
	if (cascade >= hddagi.max_cascades) {
		return false;
	}

	vec3 start_cell = (ray_position - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
	vec3 absolute_normal = abs(hddagi_normal);
	vec3 ray_bias = hddagi_normal / max(max(absolute_normal.x, max(absolute_normal.y, absolute_normal.z)), 1e-5);
	start_cell += ray_bias * 0.01;
	ray_position = start_cell / hddagi.cascades[cascade].to_cell + hddagi.cascades[cascade].position;

	ivec3 geometry_hit_cell;
	ivec3 hit_face;
	int hit_cascade;
	if (!trace_ray_hdda(ray_position, ray_direction, cascade, true, geometry_hit_cell, hit_face, hit_cascade) || dot(vec3(hit_face), vec3(hit_face)) != 1.0) {
		return false;
	}

	vec3 face_position = hddagi.cascades[hit_cascade].position + (vec3(geometry_hit_cell) + vec3(0.5) + vec3(hit_face) * 0.5) / hddagi.cascades[hit_cascade].to_cell;
	float face_denominator = dot(ray_direction, vec3(hit_face));
	if (abs(face_denominator) <= 1e-6) {
		return false;
	}
	float face_distance = dot(face_position - ray_position, vec3(hit_face)) / face_denominator;
	if (!(face_distance > 0.0) || isnan(face_distance) || isinf(face_distance)) {
		return false;
	}
	vec3 endpoint_relative = ray_position + ray_direction * face_distance;
	endpoint_relative.y /= max(abs(hddagi.y_mult), 1e-6);
	r_endpoint_world = endpoint_relative + scene_data.cam_transform[3].xyz;
	r_endpoint_normal_world = normalize(vec3(hit_face));
	if (any(isnan(r_endpoint_world)) || any(isinf(r_endpoint_world))) {
		return false;
	}

	ivec3 light_cell = geometry_hit_cell + hit_face;
	ivec3 texture_position = light_texture_position(light_cell, hit_cascade);
	vec3 light = texelFetch(sampler3D(hddagi_light_cascades, linear_sampler), texture_position, 0).rgb;
	uint light_metadata = load_voxel_neighbours(texture_position);
	if (bool(light_metadata & HDDAGI_LIGHT_CELL_VALID_BIT)) {
		uint packed_albedo = imageLoad(hddagi_albedo_cascades, texture_position).r;
		r_surface_albedo = vec3((uvec3(packed_albedo) >> uvec3(0, 5, 11)) & uvec3(0x1f, 0x3f, 0x1f)) / vec3(0x1f, 0x3f, 0x1f);
	}

	uint neighbour_bits = light_metadata & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
	vec3 cascade_offset = hddagi.cascades[hit_cascade].position;
	float to_cell = hddagi.cascades[hit_cascade].to_cell;
	float cell_size = 1.0 / to_cell;
	const ivec3 directions[26] = ivec3[](
			ivec3(-1, 0, 0), ivec3(1, 0, 0), ivec3(0, -1, 0), ivec3(0, 1, 0), ivec3(0, 0, -1), ivec3(0, 0, 1),
			ivec3(-1, -1, -1), ivec3(-1, -1, 0), ivec3(-1, -1, 1), ivec3(-1, 0, -1), ivec3(-1, 0, 1),
			ivec3(-1, 1, -1), ivec3(-1, 1, 0), ivec3(-1, 1, 1), ivec3(0, -1, -1), ivec3(0, -1, 1),
			ivec3(0, 1, -1), ivec3(0, 1, 1), ivec3(1, -1, -1), ivec3(1, -1, 0), ivec3(1, -1, 1),
			ivec3(1, 0, -1), ivec3(1, 0, 1), ivec3(1, 1, -1), ivec3(1, 1, 0), ivec3(1, 1, 1));
	vec3 light_cell_position = (vec3(light_cell) + 0.5) * cell_size + cascade_offset;
	float center_weight = max(0.0, 1.0 - point_to_ray_distance(light_cell_position, ray_position, ray_direction) * to_cell);
	vec4 light_accumulator = vec4(light, 1.0) * center_weight;
	while (neighbour_bits != 0u) {
		uint bit = findLSB(neighbour_bits);
		vec3 neighbour_position = light_cell_position + vec3(directions[bit]) * cell_size;
		float weight = max(0.0, 1.0 - point_to_ray_distance(neighbour_position, ray_position, ray_direction) * to_cell);
		if (weight > 0.0) {
			ivec3 neighbour_cell = light_cell + directions[bit];
			vec3 neighbour_light = texelFetch(sampler3D(hddagi_light_cascades, linear_sampler), light_texture_position(neighbour_cell, hit_cascade), 0).rgb;
			light_accumulator += vec4(neighbour_light, 1.0) * weight;
		}
		neighbour_bits &= ~(1u << bit);
	}
	if (light_accumulator.a > 0.0) {
		light = light_accumulator.rgb / light_accumulator.a;
	}
	r_direct_radiance = light * hddagi.cascades[hit_cascade].exposure_normalization;
	return hddagi_irradiance_cache_is_finite(r_direct_radiance) && !any(lessThan(r_direct_radiance, vec3(0.0))) && hddagi_irradiance_cache_is_finite(r_surface_albedo);
}

void irradiance_cache_update_multibounce_main() {
	if (!hddagi_irradiance_cache_multibounce_enabled()) {
		return;
	}
	uint entry_index = gl_GlobalInvocationID.x;
	uint update_stride = hddagi_irradiance_cache_training_stride();
	if ((entry_index + hddagi_irradiance_cache_params.control.x) % update_stride != 0u) {
		return;
	}

	HDDAGIIrradianceCacheLookup root_lookup = hddagi_irradiance_cache_load_entry(entry_index);
	if (!root_lookup.valid || !root_lookup.has_radiance) {
		return;
	}

	vec3 root_direct_radiance;
	vec3 root_albedo;
	if (!sample_irradiance_cache_representative(root_lookup.sample_position, root_lookup.sample_normal, root_direct_radiance, root_albedo)) {
		return;
	}
	root_albedo = clamp(root_albedo, vec3(0.0), vec3(1.0));

	uint update_sequence = hddagi_irradiance_cache_params.control.x / update_stride;
	vec2 random_sample = sample_r2_sequence(uvec2(entry_index, root_lookup.generation), update_sequence);
	vec3 direction_world = tangent_to_world(cosine_sample_hemisphere(random_sample), root_lookup.sample_normal);
	vec3 secondary_direct_radiance;
	vec3 endpoint_world;
	vec3 endpoint_normal_world;
	vec3 ignored_albedo;
	vec3 incoming_radiance;
	if (trace_irradiance_cache_segment(root_lookup.sample_position, root_lookup.sample_normal, direction_world, secondary_direct_radiance, endpoint_world, endpoint_normal_world, ignored_albedo)) {
		HDDAGIIrradianceCacheLookup secondary_lookup = hddagi_irradiance_cache_lookup(endpoint_world, endpoint_normal_world, root_lookup.sample_position);
		if (secondary_lookup.has_radiance) {
			incoming_radiance = secondary_lookup.radiance;
		} else {
			incoming_radiance = secondary_direct_radiance;
			if (hddagi_irradiance_cache_should_submit_endpoint_sample(secondary_lookup)) {
				hddagi_irradiance_cache_submit(secondary_lookup, endpoint_world, endpoint_normal_world, secondary_direct_radiance);
			}
		}
	} else {
		vec3 direction_view = normalize(transpose(mat3(scene_data.cam_transform)) * direction_world);
		incoming_radiance = sample_environment(direction_view);
	}

	vec3 total_radiance = root_direct_radiance + root_albedo * incoming_radiance;
	if (!hddagi_irradiance_cache_is_finite(total_radiance) || any(lessThan(total_radiance, vec3(0.0)))) {
		return;
	}
	hddagi_irradiance_cache_submit(root_lookup, root_lookup.sample_position, root_lookup.sample_normal, min(total_radiance, vec3(HDDAGI_IRRADIANCE_CACHE_RADIANCE_CLAMP)));
}

#else

bool trace_hddagi_sample(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_view, float detail_distance_limit, out vec3 r_radiance, out int r_hit_cascade, out vec3 r_endpoint_world, out vec3 r_endpoint_normal_world) {
	r_radiance = vec3(0.0);
	r_hit_cascade = -1;
	r_endpoint_world = vec3(0.0);
	r_endpoint_normal_world = vec3(0.0, 0.0, 1.0);
	vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
	vec3 origin_view = compute_view_position(vec3(origin_uv, origin_depth));
	ray_direction_view = normalize(ray_direction_view);
	if (any(isnan(origin_view)) || any(isinf(origin_view)) || any(isnan(ray_direction_view)) || any(isinf(ray_direction_view))) {
		return false;
	}
	mat3 camera_basis = mat3(scene_data.cam_transform);
	vec3 ray_position = camera_basis * origin_view;
	vec3 trace_origin_world = ray_position + scene_data.cam_transform[3].xyz;

	if (trace_screen_detail(origin_view, ray_direction_view, detail_distance_limit, r_endpoint_world, r_endpoint_normal_world)) {
		r_hit_cascade = -2;
		if (!query_endpoint_radiance(r_endpoint_world, r_endpoint_normal_world, trace_origin_world, r_radiance)) {
			r_radiance = sample_environment(ray_direction_view);
		}
		return true;
	}

	vec3 ray_direction = normalize(camera_basis * ray_direction_view);
	vec3 hddagi_normal = normalize(camera_basis * origin_normal);
	ray_position.y *= hddagi.y_mult;
	ray_direction.y *= hddagi.y_mult;
	hddagi_normal.y *= hddagi.y_mult;
	ray_direction = normalize(ray_direction);
	hddagi_normal = normalize(hddagi_normal);
	if (any(isnan(ray_direction)) || any(isinf(ray_direction)) || any(isnan(hddagi_normal)) || any(isinf(hddagi_normal))) {
		return false;
	}

	int cascade = hddagi.max_cascades;
	for (int i = 0; i < hddagi.max_cascades; i++) {
		vec3 cascade_position = (ray_position - hddagi.cascades[i].position) * hddagi.cascades[i].to_cell;
		if (all(greaterThanEqual(cascade_position, vec3(0.0))) && all(lessThan(cascade_position, vec3(hddagi.grid_size)))) {
			cascade = i;
			break;
		}
	}
	if (cascade >= hddagi.max_cascades) {
		return false;
	}

	vec3 start_cell = (ray_position - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
	vec3 absolute_normal = abs(hddagi_normal);
	float dominant_normal = max(absolute_normal.x, max(absolute_normal.y, absolute_normal.z));
	if (!(dominant_normal > 1e-6)) {
		return false;
	}
	vec3 ray_bias = hddagi_normal / dominant_normal;
	start_cell += ray_bias * params.normal_bias;
	ray_position = start_cell / hddagi.cascades[cascade].to_cell + hddagi.cascades[cascade].position;

	ivec3 hit_cell;
	ivec3 hit_face;
	int hit_cascade;
	bool exclude_receiver_cell = params.normal_bias >= 0.0 && dot(ray_direction_view, origin_normal) > 1e-4;
#ifndef MODE_DIRECTIONAL_TRACE
	exclude_receiver_cell = exclude_receiver_cell && (params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) != 0u;
#endif
	if (!trace_ray_hdda(ray_position, ray_direction, cascade, exclude_receiver_cell, hit_cell, hit_face, hit_cascade)) {
		return false;
	}

	bool reconnectable_endpoint = !(hit_cascade == cascade && all(equal(ivec3(start_cell), hit_cell)));
	bool disoccluded = false;
	if (!reconnectable_endpoint) {
		ivec3 read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
		uint disocclusion = load_voxel_disocclusion(read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0));
		if (disocclusion == 0u) {
			vec3 local_position = fract(start_cell) - 0.5;
			vec3 absolute_origin_normal = abs(hddagi_normal);
			int closest_axis = absolute_origin_normal.y > absolute_origin_normal.x ? 1 : 0;
			if (absolute_origin_normal.z > absolute_origin_normal[closest_axis]) {
				closest_axis = 2;
			}
			const vec3 axes[5] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
			vec3 axis_a = axes[closest_axis + 1];
			vec3 axis_b = axes[closest_axis + 2];
			vec3 advance = abs(dot(axis_a, local_position)) > abs(dot(axis_b, local_position)) ? axis_a * sign(local_position) : axis_b * sign(local_position);
			start_cell += advance;
			hit_cell += ivec3(advance);
			read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
			disocclusion = load_voxel_disocclusion(read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0));
		}
		if (disocclusion != 0u) {
			vec3 local_position = fract(start_cell) - 0.5;
			const vec3 directions[6] = vec3[](vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0));
			int best_axis = 0;
			float best_distance = -20.0;
			for (int i = 0; i < 6; i++) {
				if (bool(disocclusion & (1u << uint(i)))) {
					float distance = dot(local_position, directions[i]);
					if (distance > best_distance) {
						best_axis = i;
						best_distance = distance;
					}
				}
			}
			hit_face = ivec3(directions[best_axis]);
			disoccluded = true;
		}
	}

	if (reconnectable_endpoint && dot(vec3(hit_face), vec3(hit_face)) == 1.0) {
		vec3 face_position = hddagi.cascades[hit_cascade].position + (vec3(hit_cell) + vec3(0.5) + vec3(hit_face) * 0.5) / hddagi.cascades[hit_cascade].to_cell;
		float face_denominator = dot(ray_direction, vec3(hit_face));
		if (abs(face_denominator) > 1e-6) {
			float face_distance = dot(face_position - ray_position, vec3(hit_face)) / face_denominator;
			if (!isnan(face_distance) && !isinf(face_distance) && face_distance > 0.0) {
				vec3 endpoint_relative = ray_position + ray_direction * face_distance;
				endpoint_relative.y /= max(abs(hddagi.y_mult), 1e-6);
				vec3 endpoint_world = endpoint_relative + scene_data.cam_transform[3].xyz;
				if (!any(isnan(endpoint_world)) && !any(isinf(endpoint_world))) {
					r_hit_cascade = hit_cascade;
					r_endpoint_world = endpoint_world;
					r_endpoint_normal_world = normalize(vec3(hit_face));
				}
			}
		}
	}

#ifdef MODE_IRRADIANCE_CACHE_QUERY
	HDDAGIIrradianceCacheLookup irradiance_cache_lookup = hddagi_irradiance_cache_empty_lookup();
	if (r_hit_cascade >= 0) {
		irradiance_cache_lookup = hddagi_irradiance_cache_lookup(r_endpoint_world, r_endpoint_normal_world, trace_origin_world);
		if (irradiance_cache_lookup.has_radiance && (hddagi_irradiance_cache_multibounce_enabled() || !irradiance_cache_lookup.needs_refresh)) {
			r_radiance = irradiance_cache_lookup.radiance;
			return true;
		}
	}
#endif

	hit_cell += hit_face;
	ivec3 read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
	ivec3 texture_position = read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0);
	vec3 light = texelFetch(sampler3D(hddagi_light_cascades, linear_sampler), texture_position, 0).rgb;
	uint neighbour_bits = disoccluded ? 0u : load_voxel_neighbours(texture_position) & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
	vec3 cascade_offset = hddagi.cascades[hit_cascade].position;
	float to_cell = hddagi.cascades[hit_cascade].to_cell;
	float cell_size = 1.0 / to_cell;
	const ivec3 directions[26] = ivec3[](
			ivec3(-1, 0, 0), ivec3(1, 0, 0), ivec3(0, -1, 0), ivec3(0, 1, 0), ivec3(0, 0, -1), ivec3(0, 0, 1),
			ivec3(-1, -1, -1), ivec3(-1, -1, 0), ivec3(-1, -1, 1), ivec3(-1, 0, -1), ivec3(-1, 0, 1),
			ivec3(-1, 1, -1), ivec3(-1, 1, 0), ivec3(-1, 1, 1), ivec3(0, -1, -1), ivec3(0, -1, 1),
			ivec3(0, 1, -1), ivec3(0, 1, 1), ivec3(1, -1, -1), ivec3(1, -1, 0), ivec3(1, -1, 1),
			ivec3(1, 0, -1), ivec3(1, 0, 1), ivec3(1, 1, -1), ivec3(1, 1, 0), ivec3(1, 1, 1));
	vec3 light_cell_position = (vec3(hit_cell) + 0.5) * cell_size + cascade_offset;
	float center_weight = max(0.0, 1.0 - point_to_ray_distance(light_cell_position, ray_position, ray_direction) * to_cell);
	vec4 light_accumulator = vec4(light, 1.0) * center_weight;
	while (neighbour_bits != 0u) {
		uint bit = findLSB(neighbour_bits);
		vec3 neighbour_position = light_cell_position + vec3(directions[bit]) * cell_size;
		float weight = max(0.0, 1.0 - point_to_ray_distance(neighbour_position, ray_position, ray_direction) * to_cell);
		if (weight > 0.0) {
			ivec3 neighbour_cell = hit_cell + directions[bit];
			read_cell = (neighbour_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
			vec3 neighbour_light = texelFetch(sampler3D(hddagi_light_cascades, linear_sampler), read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0), 0).rgb;
			light_accumulator += vec4(neighbour_light, 1.0) * weight;
		}
		neighbour_bits &= ~(1u << bit);
	}
	if (light_accumulator.a > 0.0) {
		light = light_accumulator.rgb / light_accumulator.a;
	}
	r_radiance = light * hddagi.cascades[hit_cascade].exposure_normalization;
#ifdef MODE_IRRADIANCE_CACHE_QUERY
	if (hddagi_irradiance_cache_is_finite(r_radiance) && !any(lessThan(r_radiance, vec3(0.0)))) {
		if (hddagi_irradiance_cache_should_submit_endpoint_sample(irradiance_cache_lookup)) {
			hddagi_irradiance_cache_submit(irradiance_cache_lookup, r_endpoint_world, r_endpoint_normal_world, r_radiance);
		}
		if (irradiance_cache_lookup.has_radiance) {
			r_radiance = irradiance_cache_lookup.radiance;
		}
	} else if (irradiance_cache_lookup.has_radiance) {
		r_radiance = irradiance_cache_lookup.radiance;
	}
#endif
	return !any(isnan(r_radiance)) && !any(isinf(r_radiance)) && !any(lessThan(r_radiance, vec3(0.0)));
}

bool trace_hddagi_radiance(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_view, out vec3 r_radiance, out float r_hit_distance) {
	int hit_cascade;
	vec3 endpoint_world;
	vec3 endpoint_normal_world;
	bool hit = trace_hddagi_sample(origin_position, origin_depth, origin_normal, ray_direction_view, SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE, r_radiance, hit_cascade, endpoint_world, endpoint_normal_world);
	r_hit_distance = 0.0;
	if (hit && (hit_cascade >= 0 || hit_cascade == -2) && !any(isnan(endpoint_world)) && !any(isinf(endpoint_world))) {
		vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
		vec3 origin_world = (scene_data.cam_transform * vec4(compute_view_position(vec3(origin_uv, origin_depth)), 1.0)).xyz;
		r_hit_distance = clamp(length(endpoint_world - origin_world), 0.0, 65504.0);
		if (isnan(r_hit_distance) || isinf(r_hit_distance)) {
			r_hit_distance = 0.0;
		}
	}
	return hit;
}

const int GUIDED_SAMPLING_BIN_COUNT = 8;
const float GUIDED_SAMPLING_BASELINE_MIX = 0.5;
const float PI = 3.141592653589793;

struct GuidedSamplingDistribution {
	vec4 cdf_low;
	vec4 cdf_high;
};

float guided_sampling_cdf_at(GuidedSamplingDistribution distribution, int bin) {
	return bin < 4 ? distribution.cdf_low[bin] : distribution.cdf_high[bin - 4];
}

ivec3 positive_mod(ivec3 value, ivec3 divisor) {
	return ((value % divisor) + divisor) % divisor;
}

GuidedSamplingDistribution build_guided_sampling_distribution(ivec2 origin_position, float origin_depth, vec3 origin_normal) {
	GuidedSamplingDistribution distribution;
	distribution.cdf_low = vec4(0.0);
	distribution.cdf_high = vec4(0.0);
	float guide_weights[GUIDED_SAMPLING_BIN_COUNT];
	float guide_weight_sum = 0.0;
	for (int bin = 0; bin < GUIDED_SAMPLING_BIN_COUNT; bin++) {
		guide_weights[bin] = 0.0;
	}

	vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
	vec3 receiver_position = mat3(scene_data.cam_transform) * compute_view_position(vec3(origin_uv, origin_depth));
	receiver_position.y *= hddagi.y_mult;
	vec3 receiver_normal = mat3(scene_data.cam_transform) * origin_normal;
	receiver_normal.y *= hddagi.y_mult;
	float normal_length_squared = dot(receiver_normal, receiver_normal);
	int cascade = -1;
	ivec3 nearest_probe = ivec3(0);
	if (!any(isnan(receiver_position)) && !any(isinf(receiver_position)) && !any(isnan(receiver_normal)) && !any(isinf(receiver_normal)) && normal_length_squared > 1e-12) {
		receiver_normal *= inversesqrt(normal_length_squared);
		for (int candidate_cascade = 0; candidate_cascade < hddagi.max_cascades; candidate_cascade++) {
			vec3 cascade_position = (receiver_position - hddagi.cascades[candidate_cascade].position) * hddagi.cascades[candidate_cascade].to_cell;
			if (all(greaterThanEqual(cascade_position, vec3(0.0))) && all(lessThan(cascade_position, vec3(hddagi.grid_size)))) {
				vec3 absolute_normal = abs(receiver_normal);
				vec3 ray_bias = receiver_normal / max(max(absolute_normal.x, absolute_normal.y), absolute_normal.z);
				vec3 biased_position = cascade_position + ray_bias * params.normal_bias;
				nearest_probe = clamp(ivec3(floor(biased_position / float(HDDAGI_REGION_SIZE) + 0.5)), ivec3(0), hddagi.probe_axis_size - ivec3(1));
				cascade = candidate_cascade;
				break;
			}
		}
	}

	if (cascade >= 0) {
		ivec3 wrapped_probe = positive_mod(hddagi.cascades[cascade].region_world_offset + nearest_probe, hddagi.probe_axis_size);
		ivec2 probe_tile = wrapped_probe.xy + ivec2(0, wrapped_probe.z * hddagi.probe_axis_size.y);
		vec2 atlas_size = vec2(textureSize(sampler2DArray(hddagi_lightprobe_specular, sky_sampler), 0).xy);
		vec2 atlas_base = vec2(probe_tile * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1));
		for (int bin = 0; bin < GUIDED_SAMPLING_BIN_COUNT; bin++) {
			int radial_bin = bin / 4;
			int azimuth_bin = bin - radial_bin * 4;
			vec2 representative_sample = vec2((float(radial_bin) + 0.5) * 0.5, (float(azimuth_bin) + 0.5) * 0.25);
			vec3 direction_view = tangent_to_world(cosine_sample_hemisphere(representative_sample), origin_normal);
			vec3 direction_camera = mat3(scene_data.cam_transform) * direction_view;
			direction_camera.y *= hddagi.y_mult;
			direction_camera = normalize(direction_camera);
			vec2 atlas_uv = (atlas_base + vec3_to_oct(direction_camera) * float(LIGHTPROBE_OCT_SIZE)) / atlas_size;
			vec3 guide_radiance = textureLod(sampler2DArray(hddagi_lightprobe_specular, sky_sampler), vec3(atlas_uv, float(cascade)), 0.0).rgb;
			float guide_luminance = dot(max(guide_radiance, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
			float guide_weight = !isnan(guide_luminance) && !isinf(guide_luminance) ? sqrt(max(guide_luminance, 0.0)) : 0.0;
			guide_weights[bin] = guide_weight;
			guide_weight_sum += guide_weight;
		}
	}

	bool uniform_guide = cascade < 0 || isnan(guide_weight_sum) || isinf(guide_weight_sum) || guide_weight_sum <= 1e-8;
	float cumulative_probability = 0.0;
	for (int bin = 0; bin < GUIDED_SAMPLING_BIN_COUNT; bin++) {
		float guide_probability = uniform_guide ? 1.0 / float(GUIDED_SAMPLING_BIN_COUNT) : guide_weights[bin] / guide_weight_sum;
		float probability = GUIDED_SAMPLING_BASELINE_MIX / float(GUIDED_SAMPLING_BIN_COUNT) + (1.0 - GUIDED_SAMPLING_BASELINE_MIX) * guide_probability;
		cumulative_probability += probability;
		if (bin < 4) {
			distribution.cdf_low[bin] = cumulative_probability;
		} else {
			distribution.cdf_high[bin - 4] = cumulative_probability;
		}
	}
	distribution.cdf_high.w = 1.0;
	return distribution;
}

vec3 sample_guided_direction(GuidedSamplingDistribution distribution, vec2 random_sample, vec3 normal, out float r_proposal_pdf) {
	float selector = min(random_sample.x, uintBitsToFloat(0x3f7fffffu));
	int selected_bin = GUIDED_SAMPLING_BIN_COUNT - 1;
	for (int bin = 0; bin < GUIDED_SAMPLING_BIN_COUNT - 1; bin++) {
		if (selector < guided_sampling_cdf_at(distribution, bin)) {
			selected_bin = bin;
			break;
		}
	}
	float cdf_min = selected_bin == 0 ? 0.0 : guided_sampling_cdf_at(distribution, selected_bin - 1);
	float bin_probability = max(guided_sampling_cdf_at(distribution, selected_bin) - cdf_min, 1e-8);
	float within_radial_bin = clamp((selector - cdf_min) / bin_probability, 0.0, uintBitsToFloat(0x3f7fffffu));
	int radial_bin = selected_bin / 4;
	int azimuth_bin = selected_bin - radial_bin * 4;
	vec2 cosine_sample = vec2((float(radial_bin) + within_radial_bin) * 0.5, (float(azimuth_bin) + min(random_sample.y, uintBitsToFloat(0x3f7fffffu))) * 0.25);
	vec3 direction = tangent_to_world(cosine_sample_hemisphere(cosine_sample), normal);
	float cosine_pdf = max(dot(normal, direction), 0.0) / PI;
	r_proposal_pdf = max(cosine_pdf * float(GUIDED_SAMPLING_BIN_COUNT) * bin_probability, 1e-8);
	return direction;
}

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_RESOLVE)
float directional_probe_axis_texel(float grid_uv, int gi_extent, int probe_extent) {
	int safe_probe_extent = max(probe_extent, 1);
	int atlas_extent = max((gi_extent + safe_probe_extent - 1) / safe_probe_extent, 1);
	float grid_pixel = grid_uv * float(gi_extent);
	float last_origin = float((atlas_extent - 1) * safe_probe_extent);
	float last_center = (last_origin + float(gi_extent)) * 0.5;
	if (atlas_extent <= 1) {
		float tile_extent = float(max(min(safe_probe_extent, gi_extent), 1));
		return (grid_pixel - last_center) / tile_extent;
	}
	float previous_center = (float(atlas_extent) - 1.5) * float(safe_probe_extent);
	if (grid_pixel >= previous_center) {
		float last_spacing = max(last_center - previous_center, 1e-6);
		return float(atlas_extent - 2) + (grid_pixel - previous_center) / last_spacing;
	}
	return grid_pixel / float(safe_probe_extent) - 0.5;
}

vec2 directional_grid_uv_to_probe_texel(vec2 grid_uv) {
	return vec2(
			directional_probe_axis_texel(grid_uv.x, params.gi_size.x, params.probe_size),
			directional_probe_axis_texel(grid_uv.y, params.gi_size.y, params.probe_size));
}
#endif

#ifdef MODE_DIRECTIONAL_TRACE

shared ivec2 directional_history_probe;
shared uint directional_history_age;
shared uint directional_history_valid;

bool directional_decode_history_surface(uvec4 packed, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal_view, out bool r_dynamic) {
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal_view = unpack_surface_normal(packed.w);
	r_dynamic = surface_is_dynamic(packed.w);
	return r_depth > 0.0 && !isnan(r_depth) && !isinf(r_depth) &&
			all(greaterThanEqual(r_screen_position, ivec2(0))) && all(lessThan(r_screen_position, params.screen_size)) &&
			directional_finite(r_normal_view);
}

bool directional_previous_surface_world(uvec4 packed, mat4 previous_cam_transform, out vec3 r_position_world, out vec3 r_normal_world, out bool r_dynamic) {
	ivec2 screen_position;
	float depth;
	vec3 normal_view;
	if (!directional_decode_history_surface(packed, screen_position, depth, normal_view, r_dynamic)) {
		return false;
	}
	vec2 uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec4 previous_view = scene_data.previous_inv_projection[params.view_index] * vec4(uv * 2.0 - 1.0, depth, 1.0);
	if (!(abs(previous_view.w) > 1e-8) || !directional_finite(previous_view)) {
		return false;
	}
	previous_view.xyz /= previous_view.w;
	r_position_world = (previous_cam_transform * vec4(previous_view.xyz, 1.0)).xyz;
	r_normal_world = normalize(mat3(previous_cam_transform) * normal_view);
	return directional_finite(r_position_world) && directional_finite(r_normal_world);
}

bool directional_select_history(ivec2 probe_position, ivec2 current_screen_position, float current_depth, vec3 current_normal_view, bool current_dynamic, out ivec2 r_previous_probe, out uint r_previous_age) {
	r_previous_probe = ivec2(-1);
	r_previous_age = 0u;
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_HISTORY_VALID) == 0u || current_dynamic) {
		return false;
	}

	vec2 current_uv = (vec2(current_screen_position) + 0.5) / vec2(params.screen_size);
	vec3 current_view = compute_view_position(vec3(current_uv, current_depth));
	vec3 current_world = (scene_data.cam_transform * vec4(current_view, 1.0)).xyz;
	vec3 current_normal_world = normalize(mat3(scene_data.cam_transform) * current_normal_view);
	if (!directional_finite(current_view) || !directional_finite(current_world) || !directional_finite(current_normal_world)) {
		return false;
	}

	ivec2 gi_begin = probe_position * params.probe_size;
	ivec2 gi_end = min(gi_begin + ivec2(params.probe_size), params.gi_size);
	vec2 current_grid_uv = (vec2(gi_begin) + vec2(gi_end)) * 0.5 / vec2(params.gi_size);
	vec4 current_clip = scene_data.temporal_projection[params.view_index] * vec4(current_view, 1.0);
	vec3 previous_receiver_view = (scene_data.previous_cam_inv_transform * vec4(current_world, 1.0)).xyz;
	vec4 previous_clip = scene_data.previous_temporal_projection[params.view_index] * vec4(previous_receiver_view, 1.0);
	if (!(current_clip.w > 1e-6) || !(previous_clip.w > 1e-6) || !directional_finite(current_clip) || !directional_finite(previous_clip)) {
		return false;
	}
	vec2 current_stable_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
	vec2 previous_stable_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
	vec2 previous_grid_uv = current_grid_uv + previous_stable_uv - current_stable_uv;
	if (!directional_finite(vec3(previous_grid_uv, 0.0)) || any(lessThan(previous_grid_uv, vec2(0.0))) || any(greaterThanEqual(previous_grid_uv, vec2(1.0)))) {
		return false;
	}

	vec2 previous_probe_texel = directional_grid_uv_to_probe_texel(previous_grid_uv);
	if (!directional_finite(vec3(previous_probe_texel, 0.0))) {
		return false;
	}
	ivec2 previous_surface_size = imageSize(directional_previous_surface_input);
	ivec2 previous_age_size = imageSize(directional_previous_history_age_input);
	ivec2 search_base = ivec2(floor(previous_probe_texel));
	vec2 search_fraction = fract(previous_probe_texel);
	mat4 previous_cam_transform = inverse(scene_data.previous_cam_inv_transform);
	const float relative_plane_tolerance = 0.01823;
	float current_scene_depth = max(abs(current_view.z), 1e-3);
	float normal_threshold = max(params.history_normal_threshold, 0.8);
	float best_score = -1.0;

	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 candidate = search_base + ivec2(x, y);
			vec2 axis_weight = vec2(x == 0 ? 1.0 - search_fraction.x : search_fraction.x, y == 0 ? 1.0 - search_fraction.y : search_fraction.y);
			float footprint_weight = axis_weight.x * axis_weight.y;
			if (!(footprint_weight > 1e-8) || any(lessThan(candidate, ivec2(0))) ||
					any(greaterThanEqual(candidate, previous_surface_size)) || any(greaterThanEqual(candidate, previous_age_size))) {
				continue;
			}
			uint candidate_age = imageLoad(directional_previous_history_age_input, candidate).r;
			if (candidate_age == 0u) {
				continue;
			}

			vec3 previous_world;
			vec3 previous_normal_world;
			bool previous_dynamic;
			if (!directional_previous_surface_world(imageLoad(directional_previous_surface_input, candidate), previous_cam_transform, previous_world, previous_normal_world, previous_dynamic) ||
					previous_dynamic) {
				continue;
			}
			vec3 receiver_delta = previous_world - current_world;
			float relative_plane_distance = abs(dot(receiver_delta, current_normal_world)) / current_scene_depth;
			float normal_similarity = dot(current_normal_world, previous_normal_world);
			if (isnan(relative_plane_distance) || isinf(relative_plane_distance) || relative_plane_distance > relative_plane_tolerance || normal_similarity < normal_threshold) {
				continue;
			}
			float plane_score = 1.0 - clamp(relative_plane_distance / relative_plane_tolerance, 0.0, 1.0);
			float normal_score = clamp((normal_similarity - normal_threshold) / max(1.0 - normal_threshold, 1e-4), 0.0, 1.0);
			float score = footprint_weight * (0.25 + 0.75 * plane_score) * (0.25 + 0.75 * normal_score);
			bool coordinate_wins = abs(score - best_score) <= 1e-8 && (candidate.y < r_previous_probe.y || (candidate.y == r_previous_probe.y && candidate.x < r_previous_probe.x));
			if (score > best_score + 1e-8 || coordinate_wins) {
				best_score = score;
				r_previous_probe = candidate;
				r_previous_age = min(candidate_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE);
			}
		}
	}
	return best_score >= 0.0;
}

void directional_trace_main() {
	ivec2 probe_position = ivec2(gl_WorkGroupID.xy);
	ivec2 direction_texel = ivec2(gl_LocalInvocationID.xy);
	ivec2 atlas_position = probe_position * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
	ivec2 atlas_size = imageSize(raw_radiance_output);
	ivec2 probe_count = atlas_size / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 surface_size = imageSize(screen_probe_surface_input);
	if (any(greaterThanEqual(probe_position, probe_count)) || any(greaterThanEqual(probe_position, surface_size))) {
		return;
	}

	ivec2 origin_position;
	float origin_depth;
	vec3 origin_normal;
	if (!load_probe_surface(probe_position, origin_position, origin_depth, origin_normal)) {
		if (gl_LocalInvocationIndex == 0u) {
			imageStore(directional_history_age_output, probe_position, uvec4(0u));
		}
		barrier();
		imageStore(raw_radiance_output, atlas_position, vec4(0.0));
		return;
	}

	if (gl_LocalInvocationIndex == 0u) {
		directional_history_probe = ivec2(-1);
		directional_history_age = 0u;
		directional_history_valid = 0u;
		uvec4 current_surface = imageLoad(screen_probe_surface_input, probe_position);
		uint previous_age;
		ivec2 previous_probe;
		if (directional_select_history(probe_position, origin_position, origin_depth, origin_normal, surface_is_dynamic(current_surface.w), previous_probe, previous_age)) {
			directional_history_probe = previous_probe;
			directional_history_age = min(previous_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE - 1u);
			directional_history_valid = 1u;
		}
		uint current_age = directional_history_valid != 0u ? directional_history_age + 1u : 1u;
		imageStore(directional_history_age_output, probe_position, uvec4(current_age, 0u, 0u, 0u));
	}
	barrier();

	vec3 ray_direction_world = directional_jittered_bucket_to_world(probe_position, direction_texel);
	vec3 ray_direction_view = normalize(transpose(mat3(scene_data.cam_transform)) * ray_direction_world);
	vec3 incident_radiance;
	float hit_distance;
	bool synthetic_self_hit = false;
	if (!trace_hddagi_radiance(origin_position, origin_depth, origin_normal, ray_direction_view, incident_radiance, hit_distance)) {
		incident_radiance = sample_environment(ray_direction_view);
		hit_distance = SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
	} else if (!(hit_distance > 1e-4)) {
		incident_radiance = vec3(0.0);
		hit_distance = 0.0;
		synthetic_self_hit = true;
	}

	vec4 directional_sample = directional_sanitize_fp16(vec4(directional_clamp_radiance(incident_radiance * hddagi.energy), hit_distance));
	if (!synthetic_self_hit && directional_history_valid != 0u) {
		ivec2 previous_atlas_position = directional_history_probe * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
		ivec2 previous_atlas_size = textureSize(sampler2D(directional_previous_radiance_input, linear_sampler), 0);
		if (all(greaterThanEqual(previous_atlas_position, ivec2(0))) && all(lessThan(previous_atlas_position, previous_atlas_size))) {
			vec4 previous_sample = texelFetch(sampler2D(directional_previous_radiance_input, linear_sampler), previous_atlas_position, 0);
			if (directional_finite(previous_sample)) {
				previous_sample = clamp(previous_sample, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX));
				directional_sample.rgb = mix(previous_sample.rgb, directional_sample.rgb, 0.5);
			}
		}
	}
	imageStore(raw_radiance_output, atlas_position, directional_sanitize_fp16(directional_sample));
}

#endif

void screen_probe_trace_main() {
	ivec2 probe_position = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(probe_position, imageSize(raw_radiance_output)))) {
		return;
	}

	ivec2 origin_position;
	float origin_depth;
	vec3 origin_normal;
	if (!load_probe_surface(probe_position, origin_position, origin_depth, origin_normal)) {
		imageStore(raw_radiance_output, probe_position, vec4(0.0));
		return;
	}

	uint candidate_count = clamp(params.candidate_count, 1u, 8u);
	bool guided_sampling = (params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) != 0u;
	GuidedSamplingDistribution guided_distribution;
	if (guided_sampling) {
		guided_distribution = build_guided_sampling_distribution(origin_position, origin_depth, origin_normal);
	}
	vec3 radiance = vec3(0.0);
	float hit_distance = 0.0;
	for (uint candidate = 0u; candidate < candidate_count; candidate++) {
		vec2 sample_position = sample_r2_sequence(uvec2(probe_position), params.frame_index * candidate_count + candidate);
		float proposal_pdf;
		vec3 ray_direction;
		if (guided_sampling) {
			ray_direction = sample_guided_direction(guided_distribution, sample_position, origin_normal, proposal_pdf);
		} else {
			ray_direction = tangent_to_world(cosine_sample_hemisphere(sample_position), origin_normal);
			proposal_pdf = max(dot(origin_normal, ray_direction), 0.0) / PI;
		}
		vec3 candidate_radiance;
		float candidate_hit_distance;
		if (!trace_hddagi_radiance(origin_position, origin_depth, origin_normal, ray_direction, candidate_radiance, candidate_hit_distance)) {
			candidate_radiance = sample_environment(ray_direction);
			candidate_hit_distance = 65504.0;
		}
		float cosine_pdf = max(dot(origin_normal, ray_direction), 0.0) / PI;
		radiance += candidate_radiance * hddagi.energy * (cosine_pdf / max(proposal_pdf, 1e-8));
		hit_distance += candidate_hit_distance;
	}
	radiance /= float(candidate_count);
	hit_distance /= float(candidate_count);
	if (any(isnan(radiance)) || any(isinf(radiance)) || isnan(hit_distance) || isinf(hit_distance)) {
		imageStore(raw_radiance_output, probe_position, vec4(0.0));
		return;
	}
	imageStore(raw_radiance_output, probe_position, clamp(vec4(radiance, hit_distance), vec4(0.0), vec4(65504.0)));
}


#endif

#endif

#ifdef MODE_DIRECTIONAL_FILTER

bool directional_load_filter_surface(ivec2 probe_position, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal, out vec3 r_view_position, out bool r_dynamic) {
	uvec4 packed = imageLoad(directional_filter_surface_input, probe_position);
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	r_dynamic = surface_is_dynamic(packed.w);
	if (!(r_depth > 0.0) || any(lessThan(r_screen_position, ivec2(0))) || any(greaterThanEqual(r_screen_position, params.screen_size)) || !directional_finite(r_normal)) {
		return false;
	}
	vec2 screen_uv = (vec2(r_screen_position) + 0.5) / vec2(params.screen_size);
	r_view_position = compute_view_position(vec3(screen_uv, r_depth));
	return directional_finite(r_view_position);
}

void directional_filter_main() {
	ivec2 atlas_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 atlas_size = imageSize(directional_filter_output);
	if (any(greaterThanEqual(atlas_position, atlas_size))) {
		return;
	}

	ivec2 probe_position = atlas_position / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 direction_texel = atlas_position - probe_position * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 surface_size = imageSize(directional_filter_surface_input);
	ivec2 source_size = textureSize(sampler2D(directional_filter_source_input, directional_filter_nearest_sampler), 0);
	ivec2 probe_count = min(surface_size, min(atlas_size, source_size) / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE);
	if (any(greaterThanEqual(probe_position, probe_count))) {
		imageStore(directional_filter_output, atlas_position, vec4(0.0));
		return;
	}

	ivec2 center_screen_position;
	float center_depth;
	vec3 center_normal;
	vec3 center_view_position;
	bool center_dynamic;
	if (!directional_load_filter_surface(probe_position, center_screen_position, center_depth, center_normal, center_view_position, center_dynamic)) {
		imageStore(directional_filter_output, atlas_position, vec4(0.0));
		return;
	}
	ivec2 center_source_position = probe_position * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
	vec4 center_source_value = directional_sanitize_fp16(texelFetch(sampler2D(directional_filter_source_input, directional_filter_nearest_sampler), center_source_position, 0));
	vec3 direction_world = directional_jittered_bucket_to_world(probe_position, direction_texel);
	vec3 direction_view = normalize(transpose(mat3(scene_data.cam_transform)) * direction_world);

	vec3 radiance_sum = vec3(0.0);
	float radiance_weight_sum = 0.0;
	float hit_distance_sum = 0.0;
	float hit_distance_weight_sum = 0.0;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			if (abs(x) + abs(y) > 1) {
				continue;
			}
			ivec2 neighbor_probe = probe_position + ivec2(x, y);
			if (any(lessThan(neighbor_probe, ivec2(0))) || any(greaterThanEqual(neighbor_probe, probe_count))) {
				continue;
			}

			ivec2 neighbor_screen_position;
			float neighbor_depth;
			vec3 neighbor_normal;
			vec3 neighbor_view_position;
			bool neighbor_dynamic;
			if (!directional_load_filter_surface(neighbor_probe, neighbor_screen_position, neighbor_depth, neighbor_normal, neighbor_view_position, neighbor_dynamic) ||
					neighbor_dynamic != center_dynamic) {
				continue;
			}

			vec3 receiver_delta = neighbor_view_position - center_view_position;
			float receiver_separation = length(receiver_delta);
			float receiver_plane_distance = abs(dot(receiver_delta, center_normal));
			float receiver_tolerance = max(params.history_depth_tolerance + params.spatial_depth_tolerance_scale * receiver_separation, 1e-4);
			float receiver_plane_weight = 1.0 - smoothstep(0.0, receiver_tolerance, receiver_plane_distance);
			float kernel_weight = exp(-0.5 * float(x * x + y * y));
			float weight = receiver_plane_weight * kernel_weight;
			if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
				continue;
			}

			ivec2 source_position = neighbor_probe * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
			vec4 source_value = texelFetch(sampler2D(directional_filter_source_input, directional_filter_nearest_sampler), source_position, 0);
			if (!directional_finite(source_value)) {
				continue;
			}
			source_value = clamp(source_value, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX));
			float endpoint_angle_weight = 1.0;
			if (x != 0 || y != 0) {
				float neighbor_hit_distance = min(source_value.a, center_source_value.a);
				vec3 neighbor_hit_position = neighbor_view_position + direction_view * neighbor_hit_distance;
				vec3 to_neighbor_hit = neighbor_hit_position - center_view_position;
				float to_neighbor_hit_length_squared = dot(to_neighbor_hit, to_neighbor_hit);
				if (!(to_neighbor_hit_length_squared > 1e-8) || !directional_finite(to_neighbor_hit)) {
					endpoint_angle_weight = 0.0;
				} else {
					float direction_cosine = clamp(dot(to_neighbor_hit, direction_view) * inversesqrt(to_neighbor_hit_length_squared), -1.0, 1.0);
					float endpoint_angle = acos(direction_cosine);
					const float maximum_endpoint_angle = TAU / 36.0;
					endpoint_angle_weight = 1.0 - clamp(endpoint_angle / maximum_endpoint_angle, 0.0, 1.0);
				}
			}
			float radiance_weight = weight * endpoint_angle_weight;
			if (!(radiance_weight > 0.0)) {
				continue;
			}
			radiance_sum += source_value.rgb * radiance_weight;
			radiance_weight_sum += radiance_weight;
			if (source_value.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
				hit_distance_sum += source_value.a * radiance_weight;
				hit_distance_weight_sum += radiance_weight;
			}
		}
	}

	vec4 filtered = vec4(0.0);
	if (radiance_weight_sum > 0.0) {
		filtered.rgb = radiance_sum / radiance_weight_sum;
		filtered.a = hit_distance_weight_sum > 0.0 ? hit_distance_sum / hit_distance_weight_sum : SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
	}
	imageStore(directional_filter_output, atlas_position, directional_sanitize_fp16(filtered));
}

#endif

#ifdef MODE_DIRECTIONAL_IRRADIANCE

void directional_irradiance_main() {
	ivec2 atlas_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 atlas_size = imageSize(directional_irradiance_output);
	if (any(greaterThanEqual(atlas_position, atlas_size))) {
		return;
	}

	ivec2 probe_position = atlas_position / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 output_direction_texel = atlas_position - probe_position * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 source_size = textureSize(sampler2D(directional_irradiance_source_input, directional_irradiance_nearest_sampler), 0);
	ivec2 probe_count = min(atlas_size, source_size) / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	if (any(greaterThanEqual(probe_position, probe_count))) {
		imageStore(directional_irradiance_output, atlas_position, vec4(0.0));
		return;
	}

	vec3 output_normal = directional_bucket_to_world(output_direction_texel);
	vec3 radiance_sum = vec3(0.0);
	float radiance_hit_distance_sum = 0.0;
	float radiance_hit_distance_weight = 0.0;
	float cosine_hit_distance_sum = 0.0;
	float cosine_hit_distance_weight = 0.0;
	bool has_valid_sample = false;
	for (int direction_y = 0; direction_y < SCREEN_PROBE_DIRECTIONAL_TILE_SIZE; direction_y++) {
		for (int direction_x = 0; direction_x < SCREEN_PROBE_DIRECTIONAL_TILE_SIZE; direction_x++) {
			ivec2 input_direction_texel = ivec2(direction_x, direction_y);
			vec3 input_direction = directional_bucket_to_world(input_direction_texel);
			float cosine_weight = max(dot(output_normal, input_direction), 0.0);
			if (!(cosine_weight > 0.0)) {
				continue;
			}

			ivec2 source_position = probe_position * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + input_direction_texel;
			vec4 source_value = texelFetch(sampler2D(directional_irradiance_source_input, directional_irradiance_nearest_sampler), source_position, 0);
			if (!directional_finite(source_value)) {
				continue;
			}
			source_value = clamp(source_value, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX));
			radiance_sum += source_value.rgb * cosine_weight;
			if (source_value.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
				float radiance_luminance = dot(source_value.rgb, vec3(0.2126, 0.7152, 0.0722));
				if (radiance_luminance > 1e-6) {
					float radiance_hit_weight = cosine_weight * radiance_luminance;
					radiance_hit_distance_sum += source_value.a * radiance_hit_weight;
					radiance_hit_distance_weight += radiance_hit_weight;
				}
				cosine_hit_distance_sum += source_value.a * cosine_weight;
				cosine_hit_distance_weight += cosine_weight;
			}
			has_valid_sample = true;
		}
	}

	vec3 diffuse_irradiance_over_pi = radiance_sum * SCREEN_PROBE_DIRECTIONAL_IRRADIANCE_FACTOR;
	float representative_hit_distance = 0.0;
	if (has_valid_sample) {
		if (radiance_hit_distance_weight > 0.0) {
			representative_hit_distance = radiance_hit_distance_sum / radiance_hit_distance_weight;
		} else if (cosine_hit_distance_weight > 0.0) {
			representative_hit_distance = cosine_hit_distance_sum / cosine_hit_distance_weight;
		} else {
			representative_hit_distance = SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
		}
	}
	imageStore(directional_irradiance_output, atlas_position, directional_sanitize_fp16(vec4(diffuse_irradiance_over_pi, representative_hit_distance)));
}

#endif

#ifdef MODE_RESOLVE

bool load_full_resolution_surface(ivec2 screen_position, out float r_depth, out vec3 r_normal, out float r_roughness, out bool r_dynamic) {
	r_depth = texelFetch(sampler2D(depth_buffer, nearest_sampler), screen_position, 0).r;
	if (!(r_depth > 0.0)) {
		return false;
	}
	vec4 normal_roughness = texelFetch(sampler2D(normal_roughness_buffer, nearest_sampler), screen_position, 0);
	r_dynamic = normal_roughness.w > 0.5;
	float encoded_roughness = r_dynamic ? 1.0 - normal_roughness.w : normal_roughness.w;
	r_roughness = clamp(encoded_roughness / (127.0 / 255.0), 0.0, 1.0);
	return decode_normal(normal_roughness.xyz, r_normal);
}

bool select_full_resolution_surface(ivec2 resolve_position, ivec2 resolve_size, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal, out float r_roughness, out bool r_dynamic) {
	ivec2 footprint_begin = (resolve_position * params.screen_size + resolve_size - ivec2(1)) / resolve_size;
	ivec2 footprint_end = ((resolve_position + ivec2(1)) * params.screen_size + resolve_size - ivec2(1)) / resolve_size;
	footprint_begin = clamp(footprint_begin, ivec2(0), params.screen_size - ivec2(1));
	footprint_end = clamp(footprint_end, footprint_begin + ivec2(1), params.screen_size);
	r_screen_position = clamp((footprint_begin + footprint_end - ivec2(1)) / 2, ivec2(0), params.screen_size - ivec2(1));
	r_depth = 0.0;
	r_normal = vec3(0.0, 0.0, 1.0);
	r_roughness = 1.0;
	r_dynamic = false;

	bool found = false;
	for (int y = footprint_begin.y; y < footprint_end.y; y++) {
		for (int x = footprint_begin.x; x < footprint_end.x; x++) {
			ivec2 candidate_position = ivec2(x, y);
			float candidate_depth;
			vec3 candidate_normal;
			float candidate_roughness;
			bool candidate_dynamic;
			if (!load_full_resolution_surface(candidate_position, candidate_depth, candidate_normal, candidate_roughness, candidate_dynamic)) {
				continue;
			}
			if (!found || candidate_depth > r_depth) {
				found = true;
				r_screen_position = candidate_position;
				r_depth = candidate_depth;
				r_normal = candidate_normal;
				r_roughness = candidate_roughness;
				r_dynamic = candidate_dynamic;
			}
		}
	}
	return found;
}

#ifdef MODE_SVGF_PREPARE

bool svgf_prepare_finite(float value) {
	return !isnan(value) && !isinf(value);
}

bool svgf_prepare_finite(vec3 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

vec4 pack_svgf_normal_roughness(vec3 world_normal, float roughness) {
	float maximum_component = max(abs(world_normal.x), max(abs(world_normal.y), abs(world_normal.z)));
	world_normal /= max(maximum_component, 1e-6);
	return vec4(world_normal * 0.5 + 0.5, roughness);
}

void store_svgf_prepare_outputs(ivec2 resolve_position, ivec2 screen_position, bool valid_surface, float linear_depth, vec3 view_normal, float roughness, vec4 raw_signal) {
	valid_surface = valid_surface && svgf_prepare_finite(linear_depth) && abs(linear_depth) <= params.denoising_range;
	float view_z = valid_surface ? abs(linear_depth) : params.denoising_range + 1.0;

	float normal_length_squared = dot(view_normal, view_normal);
	valid_surface = valid_surface && svgf_prepare_finite(view_normal) && normal_length_squared > 1e-6;
	view_normal = valid_surface ? view_normal * inversesqrt(max(normal_length_squared, 1e-6)) : vec3(0.0, 0.0, 1.0);
	vec3 world_normal = normalize(mat3(scene_data.cam_transform) * view_normal);
	if (!svgf_prepare_finite(world_normal)) {
		world_normal = vec3(0.0, 1.0, 0.0);
	}
	roughness = valid_surface ? roughness : 1.0;

	vec2 velocity = valid_surface ? texelFetch(sampler2D(velocity_buffer, nearest_sampler), screen_position, 0).xy : vec2(0.0);
	if (any(isnan(velocity)) || any(isinf(velocity))) {
		velocity = vec2(0.0);
	}
	vec2 resolve_uv = (vec2(resolve_position) + 0.5) / vec2(imageSize(svgf_motion_output));
	vec2 surface_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec2 surface_uv_offset = valid_surface ? surface_uv - resolve_uv : vec2(0.0);

	bool valid_signal = !any(isnan(raw_signal)) && !any(isinf(raw_signal));
#ifndef MODE_DIRECTIONAL_RESOLVE
	valid_signal = valid_signal && valid_surface;
#endif
	if (!valid_signal) {
		raw_signal = vec4(0.0);
	}
	raw_signal = max(raw_signal, vec4(0.0));
	raw_signal.rgb = min(raw_signal.rgb * params.scene_to_svgf_scale, vec3(params.input_radiance_max));
	raw_signal.a = min(raw_signal.a, params.input_hit_distance_max);

	imageStore(svgf_signal_output, resolve_position, raw_signal);
	imageStore(svgf_normal_roughness_output, resolve_position, pack_svgf_normal_roughness(world_normal, roughness));
	imageStore(svgf_view_z_output, resolve_position, vec4(view_z, 0.0, 0.0, 0.0));
	imageStore(svgf_motion_output, resolve_position, vec4(velocity, surface_uv_offset));
}

#endif

bool load_resolve_probe_surface(ivec2 probe_position, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal, out bool r_dynamic) {
	uvec4 packed = imageLoad(screen_probe_surface_input, probe_position);
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	r_dynamic = surface_is_dynamic(packed.w);
	return r_depth > 0.0;
}

#ifdef MODE_DIRECTIONAL_RESOLVE

struct DirectionalLookup {
	ivec2 texel_00;
	ivec2 texel_10;
	ivec2 texel_01;
	ivec2 texel_11;
	vec2 blend;
};

bool directional_prepare_lookup(vec3 world_normal, out DirectionalLookup r_lookup) {
	float normal_length_squared = dot(world_normal, world_normal);
	if (!directional_finite(world_normal) || !(normal_length_squared > 1e-8)) {
		return false;
	}
	world_normal *= inversesqrt(normal_length_squared);
	vec2 atlas_position = directional_sphere_to_square(world_normal) * float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE) - 0.5;
	ivec2 base_texel = ivec2(floor(atlas_position));
	r_lookup.texel_00 = directional_wrap_texel(base_texel);
	r_lookup.texel_10 = directional_wrap_texel(base_texel + ivec2(1, 0));
	r_lookup.texel_01 = directional_wrap_texel(base_texel + ivec2(0, 1));
	r_lookup.texel_11 = directional_wrap_texel(base_texel + ivec2(1, 1));
	r_lookup.blend = fract(atlas_position);
	return true;
}

vec4 directional_resolve_probe(ivec2 probe_position, DirectionalLookup lookup) {
	ivec2 tile_origin = probe_position * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	vec4 sample_00 = directional_sanitize_fp16(texelFetch(sampler2D(raw_radiance_input, nearest_sampler), tile_origin + lookup.texel_00, 0));
	vec4 sample_10 = directional_sanitize_fp16(texelFetch(sampler2D(raw_radiance_input, nearest_sampler), tile_origin + lookup.texel_10, 0));
	vec4 sample_01 = directional_sanitize_fp16(texelFetch(sampler2D(raw_radiance_input, nearest_sampler), tile_origin + lookup.texel_01, 0));
	vec4 sample_11 = directional_sanitize_fp16(texelFetch(sampler2D(raw_radiance_input, nearest_sampler), tile_origin + lookup.texel_11, 0));
	vec4 row_0 = mix(sample_00, sample_10, lookup.blend.x);
	vec4 row_1 = mix(sample_01, sample_11, lookup.blend.x);
	vec3 irradiance = mix(row_0.rgb, row_1.rgb, lookup.blend.y);

	vec4 bilinear_weights = vec4(
			(1.0 - lookup.blend.x) * (1.0 - lookup.blend.y),
			lookup.blend.x * (1.0 - lookup.blend.y),
			(1.0 - lookup.blend.x) * lookup.blend.y,
			lookup.blend.x * lookup.blend.y);
	float hit_distance_sum = 0.0;
	float hit_distance_weight_sum = 0.0;
	if (sample_00.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
		hit_distance_sum += sample_00.a * bilinear_weights.x;
		hit_distance_weight_sum += bilinear_weights.x;
	}
	if (sample_10.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
		hit_distance_sum += sample_10.a * bilinear_weights.y;
		hit_distance_weight_sum += bilinear_weights.y;
	}
	if (sample_01.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
		hit_distance_sum += sample_01.a * bilinear_weights.z;
		hit_distance_weight_sum += bilinear_weights.z;
	}
	if (sample_11.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
		hit_distance_sum += sample_11.a * bilinear_weights.w;
		hit_distance_weight_sum += bilinear_weights.w;
	}
	float hit_distance = hit_distance_weight_sum > 0.0 ? hit_distance_sum / hit_distance_weight_sum : SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
	return directional_sanitize_fp16(vec4(irradiance, hit_distance));
}

float directional_receiver_weight(ivec2 probe_position, vec3 pixel_view_position, vec3 pixel_normal, bool pixel_dynamic) {
	ivec2 probe_screen_position;
	float probe_depth;
	vec3 probe_normal;
	bool probe_dynamic;
	if (!load_resolve_probe_surface(probe_position, probe_screen_position, probe_depth, probe_normal, probe_dynamic) || probe_dynamic != pixel_dynamic) {
		return 0.0;
	}
	vec2 probe_uv = (vec2(probe_screen_position) + 0.5) / vec2(params.screen_size);
	vec3 probe_view_position = compute_view_position(vec3(probe_uv, probe_depth));
	if (!directional_finite(probe_view_position)) {
		return 0.0;
	}
	vec3 receiver_delta = probe_view_position - pixel_view_position;
	float receiver_separation = length(receiver_delta);
	float receiver_plane_distance = abs(dot(receiver_delta, pixel_normal));
	float receiver_plane_tolerance = max(params.history_depth_tolerance + params.spatial_depth_tolerance_scale * receiver_separation, 1e-4);
	return 1.0 - smoothstep(receiver_plane_tolerance, receiver_plane_tolerance * 2.0, receiver_plane_distance);
}

vec3 directional_base_ambient(ivec2 resolve_position, ivec2 resolve_size) {
	vec2 resolve_uv = (vec2(resolve_position) + 0.5) / vec2(resolve_size);
	vec3 base_ambient = textureLod(sampler2D(directional_base_ambient_input, nearest_sampler), resolve_uv, 0.0).rgb;
	return directional_finite(base_ambient) ? max(base_ambient, vec3(0.0)) : vec3(0.0);
}

#endif

void screen_probe_resolve_main() {
	ivec2 resolve_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 resolve_size = imageSize(resolved_radiance_output);
	if (any(greaterThanEqual(resolve_position, resolve_size))) {
		return;
	}

	ivec2 screen_position = ivec2(0);
	float pixel_depth;
	vec3 pixel_normal;
	float pixel_roughness;
	bool pixel_dynamic;
	if (!select_full_resolution_surface(resolve_position, resolve_size, screen_position, pixel_depth, pixel_normal, pixel_roughness, pixel_dynamic)) {
		vec4 fallback_signal = vec4(0.0);
#ifdef MODE_DIRECTIONAL_RESOLVE
		fallback_signal.rgb = directional_base_ambient(resolve_position, resolve_size);
#endif
		imageStore(resolved_radiance_output, resolve_position, fallback_signal);
#ifdef MODE_SVGF_PREPARE
		store_svgf_prepare_outputs(resolve_position, screen_position, false, 0.0, vec3(0.0, 0.0, 1.0), 1.0, fallback_signal);
#endif
		return;
	}
	vec2 pixel_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec3 pixel_view_position = compute_view_position(vec3(pixel_uv, pixel_depth));
	float pixel_linear_depth = pixel_view_position.z;
	if (any(isnan(pixel_view_position)) || any(isinf(pixel_view_position))) {
		vec4 fallback_signal = vec4(0.0);
#ifdef MODE_DIRECTIONAL_RESOLVE
		fallback_signal.rgb = directional_base_ambient(resolve_position, resolve_size);
#endif
		imageStore(resolved_radiance_output, resolve_position, fallback_signal);
#ifdef MODE_SVGF_PREPARE
		store_svgf_prepare_outputs(resolve_position, screen_position, false, 0.0, pixel_normal, pixel_roughness, fallback_signal);
#endif
		return;
	}

#ifdef MODE_DIRECTIONAL_RESOLVE
	vec3 pixel_world_normal = normalize(mat3(scene_data.cam_transform) * pixel_normal);
	DirectionalLookup directional_lookup;
	if (!directional_prepare_lookup(pixel_world_normal, directional_lookup)) {
		vec4 fallback_signal = vec4(directional_base_ambient(resolve_position, resolve_size), 0.0);
		imageStore(resolved_radiance_output, resolve_position, fallback_signal);
#ifdef MODE_SVGF_PREPARE
		store_svgf_prepare_outputs(resolve_position, screen_position, false, pixel_linear_depth, pixel_normal, pixel_roughness, fallback_signal);
#endif
		return;
	}
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(pixel_uv);
	ivec2 probe_base = ivec2(floor(probe_grid_position));
	vec2 probe_blend = fract(probe_grid_position);
	ivec2 atlas_probe_count = textureSize(sampler2D(raw_radiance_input, nearest_sampler), 0) / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 expected_probe_count = (params.gi_size + ivec2(max(params.probe_size, 1)) - ivec2(1)) / max(params.probe_size, 1);
	ivec2 probe_count = min(atlas_probe_count, expected_probe_count);
	vec3 radiance_sum = vec3(0.0);
	float hit_distance_sum = 0.0;
	float hit_distance_weight_sum = 0.0;
	float bilinear_weight_sum = 0.0;
#else
	ivec2 gi_position = clamp(resolve_position * params.gi_size / resolve_size, ivec2(0), params.gi_size - ivec2(1));
	ivec2 probe_base = gi_position / max(params.probe_size, 1);
	ivec2 probe_count = textureSize(sampler2D(raw_radiance_input, nearest_sampler), 0);
	vec2 probe_screen_extent = vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size);
	float probe_extent = max((probe_screen_extent.x + probe_screen_extent.y) * 0.5, 1.0);
	vec4 radiance_hit_distance_sum = vec4(0.0);
#endif
	float weight_sum = 0.0;

#ifdef MODE_DIRECTIONAL_RESOLVE
	const int probe_offset_begin = 0;
#else
	const int probe_offset_begin = -1;
#endif
	const int probe_offset_end = 1;
	for (int y = probe_offset_begin; y <= probe_offset_end; y++) {
		for (int x = probe_offset_begin; x <= probe_offset_end; x++) {
			ivec2 probe_position = probe_base + ivec2(x, y);
			if (any(lessThan(probe_position, ivec2(0))) || any(greaterThanEqual(probe_position, probe_count))) {
				continue;
			}
#ifdef MODE_DIRECTIONAL_RESOLVE
			float bilinear_weight = (x == 0 ? 1.0 - probe_blend.x : probe_blend.x) * (y == 0 ? 1.0 - probe_blend.y : probe_blend.y);
			bilinear_weight_sum += bilinear_weight;
			float weight = bilinear_weight * directional_receiver_weight(probe_position, pixel_view_position, pixel_normal, pixel_dynamic);
			if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
				continue;
			}
			vec4 raw_radiance = directional_resolve_probe(probe_position, directional_lookup);
			radiance_sum += raw_radiance.rgb * weight;
			if (raw_radiance.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
				hit_distance_sum += raw_radiance.a * weight;
				hit_distance_weight_sum += weight;
			}
#else
			ivec2 probe_screen_position;
			float probe_depth;
			vec3 probe_normal;
			bool probe_dynamic;
			if (!load_resolve_probe_surface(probe_position, probe_screen_position, probe_depth, probe_normal, probe_dynamic)) {
				continue;
			}
			vec4 raw_radiance = texelFetch(sampler2D(raw_radiance_input, nearest_sampler), probe_position, 0);
			if (any(isnan(raw_radiance)) || any(isinf(raw_radiance))) {
				continue;
			}
			vec2 probe_uv = (vec2(probe_screen_position) + 0.5) / vec2(params.screen_size);
			float probe_linear_depth = compute_view_position(vec3(probe_uv, probe_depth)).z;
			if (isnan(probe_linear_depth) || isinf(probe_linear_depth)) {
				continue;
			}
			float normal_similarity = max(dot(pixel_normal, probe_normal), 0.0);
			float normal_weight = pow(normal_similarity, 6.0);
			float depth_scale = max(0.015, abs(pixel_linear_depth) * mix(0.012, 0.04, smoothstep(0.75, 0.98, normal_similarity)));
			float depth_weight = 1.0 - smoothstep(0.0, depth_scale, abs(pixel_linear_depth - probe_linear_depth));
			float distance_weight = 1.0 / (1.0 + length(vec2(screen_position - probe_screen_position)) / probe_extent);
			float weight = normal_weight * depth_weight * distance_weight;
			if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
				continue;
			}
			radiance_hit_distance_sum += max(raw_radiance, vec4(0.0)) * weight;
#endif
			weight_sum += weight;
		}
	}

#ifdef MODE_DIRECTIONAL_RESOLVE
	vec4 resolved = vec4(0.0, 0.0, 0.0, SCREEN_PROBE_DIRECTIONAL_FP16_MAX);
	float coverage = 0.0;
	if (weight_sum > 0.0) {
		resolved.rgb = radiance_sum / weight_sum;
		resolved.a = hit_distance_weight_sum > 0.0 ? hit_distance_sum / hit_distance_weight_sum : SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
		float confidence = clamp(weight_sum / max(bilinear_weight_sum, 1e-4), 0.0, 1.0);
		coverage = smoothstep(0.05, 0.55, confidence);
	}
#else
	vec4 resolved = weight_sum > 0.0 ? radiance_hit_distance_sum / weight_sum : vec4(0.0);
#endif
	if (any(isnan(resolved)) || any(isinf(resolved))) {
#ifdef MODE_DIRECTIONAL_RESOLVE
		resolved = vec4(0.0, 0.0, 0.0, SCREEN_PROBE_DIRECTIONAL_FP16_MAX);
		coverage = 0.0;
#else
		resolved = vec4(0.0);
#endif
	} else {
		resolved = clamp(resolved, vec4(0.0), vec4(65504.0));
	}
#ifdef MODE_DIRECTIONAL_RESOLVE
	resolved.rgb = mix(directional_base_ambient(resolve_position, resolve_size), resolved.rgb, clamp(coverage, 0.0, 1.0));
#endif
	imageStore(resolved_radiance_output, resolve_position, resolved);
#ifdef MODE_SVGF_PREPARE
	store_svgf_prepare_outputs(resolve_position, screen_position, true, pixel_linear_depth, pixel_normal, pixel_roughness, resolved);
#endif
}

#endif

#ifdef MODE_APPLY

void screen_probe_apply_main() {
	ivec2 output_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 output_size = imageSize(ambient_output);
	if (any(greaterThanEqual(output_position, output_size))) {
		return;
	}
	ivec2 radiance_size = textureSize(sampler2D(resolved_radiance_input, nearest_sampler), 0);
	ivec2 radiance_position = clamp(output_position * radiance_size / output_size, ivec2(0), radiance_size - ivec2(1));
	vec4 resolved_radiance = texelFetch(sampler2D(resolved_radiance_input, nearest_sampler), radiance_position, 0);
#ifdef MODE_SVGF_APPLY
	const float radiance_scale = 512.0;
#else
	const float radiance_scale = 1.0;
#endif
	bool valid_radiance = !any(isnan(resolved_radiance)) && !any(isinf(resolved_radiance));
	vec3 diffuse_irradiance_over_pi = valid_radiance ? max(resolved_radiance.rgb, vec3(0.0)) * radiance_scale : vec3(0.0);
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_SURFACE_FOOTPRINT) != 0u) {
		if (!valid_radiance) {
			vec2 output_uv = (vec2(output_position) + 0.5) / vec2(output_size);
			diffuse_irradiance_over_pi = textureLod(sampler2D(base_ambient_input, nearest_sampler), output_uv, 0.0).rgb;
			if (any(isnan(diffuse_irradiance_over_pi)) || any(isinf(diffuse_irradiance_over_pi))) {
				diffuse_irradiance_over_pi = vec3(0.0);
			}
		}
		imageStore(ambient_output, output_position, uvec4(rgbe_encode(max(diffuse_irradiance_over_pi, vec3(0.0)))));
		return;
	}
	if (!valid_radiance) {
		return;
	}
	imageStore(ambient_output, output_position, uvec4(rgbe_encode(diffuse_irradiance_over_pi)));
}

#endif

void main() {
#ifdef MODE_SURFACE
	screen_probe_surface_main();
#elif defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)
	irradiance_cache_update_multibounce_main();
#elif defined(MODE_DIRECTIONAL_TRACE)
	directional_trace_main();
#elif defined(MODE_TRACE)
	screen_probe_trace_main();
#elif defined(MODE_DIRECTIONAL_FILTER)
	directional_filter_main();
#elif defined(MODE_DIRECTIONAL_IRRADIANCE)
	directional_irradiance_main();
#elif defined(MODE_RESOLVE)
	screen_probe_resolve_main();
#elif defined(MODE_APPLY)
	screen_probe_apply_main();
#endif
}
