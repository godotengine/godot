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
const uint SCREEN_PROBE_FLAG_DIRECTIONAL_ADAPTIVE = 1u << 4u;
const uint SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID = 1u << 5u;
const uint SCREEN_PROBE_FLAG_SPECULAR_MOTION_VALID = 1u << 4u;
const uint SCREEN_PROBE_FLAG_SPECULAR_SCREEN_RADIANCE_VALID = 1u << 5u;
const uint SCREEN_PROBE_SKY_COLOR = 1u;
const uint SCREEN_PROBE_SKY_TEXTURE = 2u;
const uint SCREEN_PROBE_DEBUG_VALID = 1u << 0u;
const uint SCREEN_PROBE_DEBUG_DETAIL_HIT = 1u << 1u;
const uint SCREEN_PROBE_DEBUG_DETAIL_MISS = 1u << 2u;
const uint SCREEN_PROBE_DEBUG_DETAIL_REJECTED = 1u << 3u;
const uint SCREEN_PROBE_DEBUG_SCREEN_HDDAGI = 1u << 4u;
const uint SCREEN_PROBE_DEBUG_HDDA_VOXEL = 1u << 5u;
const uint SCREEN_PROBE_DEBUG_SKY = 1u << 6u;
const uint SCREEN_PROBE_DEBUG_SCREEN_RADIANCE_FALLBACK = 1u << 7u;
const uint SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE = 1u << 8u;
const uint SCREEN_PROBE_DEBUG_SCREEN_IRRADIANCE_CACHE = 1u << 9u;
const uint SCREEN_PROBE_DEBUG_HDDA_IRRADIANCE_CACHE = 1u << 10u;
const uint SCREEN_PROBE_DEBUG_SOURCE_MASK = SCREEN_PROBE_DEBUG_SCREEN_HDDAGI | SCREEN_PROBE_DEBUG_HDDA_VOXEL | SCREEN_PROBE_DEBUG_SKY | SCREEN_PROBE_DEBUG_SCREEN_IRRADIANCE_CACHE | SCREEN_PROBE_DEBUG_HDDA_IRRADIANCE_CACHE;
const uint SCREEN_PROBE_DEBUG_SOURCE_STATE_MASK = SCREEN_PROBE_DEBUG_SOURCE_MASK | SCREEN_PROBE_DEBUG_SCREEN_RADIANCE_FALLBACK;
const float SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE = 2.0;
const float SCREEN_PROBE_SPECULAR_DETAIL_TRACE_MAX_DISTANCE = 50.0;

const int HDDAGI_TRACE_SCREEN_ENDPOINT = -2;
const int HDDAGI_TRACE_SCREEN_ENDPOINT_SKY_FALLBACK = -3;

const int HDDAGI_REGION_SIZE = 8;
const int HDDAGI_HDDA_FP_BITS = 10;
const uint HDDAGI_LIGHT_CELL_VALID_BIT = 1u << 26u;
const uint HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK = HDDAGI_LIGHT_CELL_VALID_BIT - 1u;
const float TAU = 6.283185307179586;

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_FILTER) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_DIRECTIONAL_RESOLVE) || defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)
#ifndef SCREEN_PROBE_DIRECTIONAL_SIZE
#error "Directional screen probes require SCREEN_PROBE_DIRECTIONAL_SIZE"
#endif
#if SCREEN_PROBE_DIRECTIONAL_SIZE != 8
#error "Directional screen probes require an 8x8 workgroup"
#endif
const int SCREEN_PROBE_DIRECTIONAL_TILE_SIZE = SCREEN_PROBE_DIRECTIONAL_SIZE;
const int SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
const int SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_RESOLUTION = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE * 2;
const uint SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_UNIFORM_LEVEL = 1u;
const float SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_MIN_PDF = 0.1;
const float SCREEN_PROBE_DIRECTIONAL_IRRADIANCE_FACTOR = 4.0 / float(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT);
const float SCREEN_PROBE_DIRECTIONAL_FP16_MAX = 65504.0;
const uint SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE = 8u;
const float SCREEN_PROBE_DIRECTIONAL_RADIANCE_MAX = 10.0;
const uint SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT = 8u;
const uint SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE = 8u;
const float SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE = 0.01;
const float SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_RETIRE_COVERAGE = 0.06;
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

#if defined(MODE_SPECULAR_TRACE) || defined(MODE_SPECULAR_APPLY)
	vec4 specular_tuning;
	vec4 specular_eye_offset_exposure;
#endif
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

#if defined(MODE_DEBUG_MONTAGE) && defined(MODE_TRACE)
layout(r32ui, set = 3, binding = 0) uniform restrict writeonly uimage2D trace_debug_output;
#endif

#ifdef MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE
void screen_probe_debug_mark(uint bits) {}
void screen_probe_debug_set_source(uint source) {}
#endif

#ifdef MODE_SURFACE

layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 2) uniform sampler nearest_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform restrict writeonly uimage2D screen_probe_surface_output;

#endif

#if defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)

layout(set = 0, binding = 0) uniform texture2D adaptive_depth_buffer;
layout(set = 0, binding = 1) uniform texture2D adaptive_normal_roughness_buffer;
layout(set = 0, binding = 2) uniform sampler adaptive_nearest_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform coherent uimage2D adaptive_probe_surface;
#ifdef MODE_DIRECTIONAL_ADAPTIVE_MARK
layout(r8ui, set = 0, binding = 4) uniform restrict writeonly uimage2D adaptive_candidate_mark_output;
#else
layout(r8ui, set = 0, binding = 4) uniform restrict readonly uimage2D adaptive_candidate_mark_input;
#endif
layout(rgba32ui, set = 0, binding = 5) uniform coherent uimage2D adaptive_tile_data_output;
layout(r32ui, set = 0, binding = 6) uniform coherent uimage2D adaptive_probe_counter;
layout(set = 0, binding = 7, std140) uniform AdaptiveSceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
layout(rgba32ui, set = 0, binding = 8) uniform restrict readonly uimage2D adaptive_previous_probe_surface;
layout(rgba32ui, set = 0, binding = 9) uniform restrict readonly uimage2D adaptive_previous_tile_data_input;

#endif

#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE) || defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)

#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE)
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
#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE)
layout(set = 0, binding = 10) uniform texture2D detail_hiz_buffer;
layout(set = 0, binding = 11) uniform texture2D detail_normal_roughness_buffer;
#endif
#ifdef MODE_DIRECTIONAL_TRACE
layout(set = 0, binding = 12) uniform texture2D directional_previous_radiance_input;
layout(rgba32ui, set = 0, binding = 13) uniform restrict readonly uimage2D directional_previous_surface_input;
layout(r8ui, set = 0, binding = 14) uniform restrict readonly uimage2D directional_previous_history_age_input;
layout(r8ui, set = 0, binding = 15) uniform restrict writeonly uimage2D directional_history_age_output;
layout(rgba32ui, set = 0, binding = 16) uniform restrict readonly uimage2D directional_previous_adaptive_tile_data_input;
layout(set = 0, binding = 17) uniform texture2D directional_velocity_buffer;
layout(set = 0, binding = 18) uniform texture2D directional_depth_buffer;
layout(set = 0, binding = 19) uniform texture2D directional_previous_filtered_radiance_input;
layout(r8ui, set = 0, binding = 20) uniform restrict writeonly uimage2D directional_trace_count_output;
#endif
#ifdef MODE_SPECULAR_TRACE
layout(set = 0, binding = 12) uniform texture2D specular_depth_buffer;
layout(set = 0, binding = 13) uniform texture2D specular_velocity_buffer;
layout(rgba8, set = 0, binding = 14) uniform restrict writeonly image2D specular_normal_roughness_output;
layout(r32f, set = 0, binding = 15) uniform restrict writeonly image2D specular_view_z_output;
layout(rgba16f, set = 0, binding = 16) uniform restrict writeonly image2D specular_motion_output;
layout(set = 0, binding = 17) uniform texture2D specular_screen_radiance_buffer;
layout(set = 0, binding = 18) uniform texture2D specular_previous_depth_buffer;
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
#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE)
layout(set = 1, binding = 2) uniform texture2DArray hddagi_lightprobe_specular;
#endif

#endif

#ifdef MODE_SPECULAR_APPLY

layout(set = 0, binding = 0) uniform texture2D specular_radiance_input;
layout(set = 0, binding = 1) uniform sampler specular_nearest_sampler;
layout(r32ui, set = 0, binding = 2) uniform coherent uimage2D reflection_output;
layout(set = 0, binding = 3) uniform texture2D specular_normal_roughness_input;

#endif

#ifdef MODE_DIRECTIONAL_FILTER

layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D directional_filter_surface_input;
layout(set = 0, binding = 1) uniform texture2D directional_filter_source_input;
layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D directional_filter_output;
layout(set = 0, binding = 3) uniform sampler directional_filter_nearest_sampler;
layout(set = 0, binding = 4, std140) uniform DirectionalFilterSceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
layout(r8ui, set = 0, binding = 5) uniform restrict readonly uimage2D directional_filter_trace_count_input;

#endif

#ifdef MODE_DIRECTIONAL_IRRADIANCE

layout(set = 0, binding = 0) uniform texture2D directional_irradiance_source_input;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D directional_irradiance_output;
layout(set = 0, binding = 2) uniform sampler directional_irradiance_nearest_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform restrict readonly uimage2D directional_irradiance_surface_input;
layout(r8ui, set = 0, binding = 4) uniform restrict readonly uimage2D directional_irradiance_trace_count_input;

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
layout(set = 0, binding = 13) uniform utexture2D directional_adaptive_tile_data_input;
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

float hash_float(uvec3 value) {
	return float(hash_uvec3(value) & 0x00ffffffu) / float(0x01000000u);
}

const uvec2 R2_WEYL_INCREMENT = uvec2(3242174889u, 2447445413u);

float uint_to_unit_float(uint value) {
	return float(value >> 8u) * (1.0 / 16777216.0);
}

vec2 sample_r2_sequence(uvec2 position, uint sequence_index, uint stream) {
	uvec2 scramble = uvec2(
			hash_uvec3(uvec3(position, stream ^ 0xa511e9b3u)),
			hash_uvec3(uvec3(position, stream ^ 0x63d83595u)));
	uvec2 sequence = scramble + uvec2(sequence_index) * R2_WEYL_INCREMENT;
	return vec2(uint_to_unit_float(sequence.x), uint_to_unit_float(sequence.y));
}

vec2 sample_r2_sequence(uvec2 position, uint sequence_index) {
	uvec2 scramble = uvec2(
			hash_uvec3(uvec3(position, 0xe145f4edu)),
			hash_uvec3(uvec3(position, 0x0c8f49b7u)));
	uvec2 sequence = scramble + uvec2(sequence_index) * R2_WEYL_INCREMENT;
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

vec3 world_to_tangent(vec3 world_direction, vec3 normal) {
	vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(up, normal));
	vec3 bitangent = cross(normal, tangent);
	return vec3(dot(world_direction, tangent), dot(world_direction, bitangent), dot(world_direction, normal));
}

vec3 sample_bounded_ggx_vndf_reflection(vec3 view_local, float alpha, vec2 random_sample) {
	alpha = clamp(alpha, 1e-6, 1.0);
	vec3 stretched_view = normalize(vec3(alpha * view_local.xy, max(view_local.z, 1e-6)));

	float phi = TAU * random_sample.x;
	float scale = 1.0 + length(view_local.xy);
	float alpha_squared = alpha * alpha;
	float scale_squared = scale * scale;
	float bound = (1.0 - alpha_squared) * scale_squared /
			max(scale_squared + alpha_squared * view_local.z * view_local.z, 1e-8);
	float cap = bound * stretched_view.z;
	float reflected_z_stretched = (1.0 - random_sample.y) * (1.0 + cap) - cap;
	float reflected_xy_length = sqrt(clamp(1.0 - reflected_z_stretched * reflected_z_stretched, 0.0, 1.0));
	vec3 reflected_stretched = vec3(
			reflected_xy_length * cos(phi),
			reflected_xy_length * sin(phi),
			reflected_z_stretched);

	vec3 microfacet_stretched = stretched_view + reflected_stretched;
	vec3 microfacet_normal = normalize(vec3(alpha * microfacet_stretched.xy, microfacet_stretched.z));
	return reflect(-view_local, microfacet_normal);
}

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_FILTER) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_DIRECTIONAL_RESOLVE) || defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)

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

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_RESOLVE) || defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)

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

#ifdef MODE_SPECULAR_APPLY
vec3 rgbe_decode(uint packed_rgbe) {
	vec4 rgbe = vec4((uvec4(packed_rgbe) >> uvec4(0, 9, 18, 27)) & uvec4(0x1ff, 0x1ff, 0x1ff, 0x1f));
	return rgbe.rgb * pow(2.0, rgbe.a - 15.0 - 9.0);
}
#endif

#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE) || defined(MODE_RESOLVE) || defined(MODE_DIRECTIONAL_FILTER) || defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)

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

#if defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)

shared uint directional_adaptive_mask;
shared uint directional_adaptive_spawn_mask;
shared uint directional_adaptive_allocation_start;
shared uint directional_adaptive_allocation_count;
shared uint directional_adaptive_retained_count;
shared uvec4 directional_adaptive_candidate_surfaces[8];
shared uvec4 directional_adaptive_retained_surfaces[8];

ivec2 directional_adaptive_base_count() {
	int safe_probe_size = max(params.probe_size, 1);
	return (params.gi_size + ivec2(safe_probe_size) - ivec2(1)) / safe_probe_size;
}

bool directional_adaptive_load_screen_surface(ivec2 screen_position, out uvec4 r_packed, out vec3 r_view_position, out vec3 r_normal, out bool r_dynamic) {
	r_packed = uvec4(0xffffffffu);
	r_view_position = vec3(0.0);
	r_normal = vec3(0.0, 0.0, 1.0);
	r_dynamic = false;
	if (any(lessThan(screen_position, ivec2(0))) || any(greaterThanEqual(screen_position, params.screen_size))) {
		return false;
	}
	float depth = texelFetch(sampler2D(adaptive_depth_buffer, adaptive_nearest_sampler), screen_position, 0).r;
	vec4 normal_roughness = texelFetch(sampler2D(adaptive_normal_roughness_buffer, adaptive_nearest_sampler), screen_position, 0);
	if (!(depth > 0.0) || !decode_normal(normal_roughness.xyz, r_normal)) {
		return false;
	}
	r_dynamic = normal_roughness.w > 0.5;
	vec2 uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	r_view_position = compute_view_position(vec3(uv, depth));
	if (!directional_finite(r_view_position)) {
		return false;
	}
	r_packed = uvec4(uvec2(screen_position), floatBitsToUint(depth), pack_surface_normal(r_normal, r_dynamic));
	return true;
}

bool directional_adaptive_decode_surface(uvec4 packed, out ivec2 r_screen_position, out vec3 r_view_position, out vec3 r_normal, out bool r_dynamic) {
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	float depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	r_dynamic = surface_is_dynamic(packed.w);
	if (!(depth > 0.0) || any(lessThan(r_screen_position, ivec2(0))) || any(greaterThanEqual(r_screen_position, params.screen_size)) || !directional_finite(r_normal)) {
		return false;
	}
	vec2 uv = (vec2(r_screen_position) + 0.5) / vec2(params.screen_size);
	r_view_position = compute_view_position(vec3(uv, depth));
	return directional_finite(r_view_position);
}

ivec2 directional_adaptive_candidate_position(ivec2 tile, uint candidate_index) {
	int safe_probe_size = max(params.probe_size, 1);
	ivec2 gi_begin = tile * safe_probe_size;
	ivec2 gi_end = min(gi_begin + ivec2(safe_probe_size), params.gi_size);
	ivec2 screen_begin = (gi_begin * params.screen_size + params.gi_size - ivec2(1)) / params.gi_size;
	ivec2 screen_end = (gi_end * params.screen_size + params.gi_size - ivec2(1)) / params.gi_size;
	screen_begin = clamp(screen_begin, ivec2(0), params.screen_size - ivec2(1));
	screen_end = clamp(screen_end, screen_begin + ivec2(1), params.screen_size);
	ivec2 screen_extent = max(screen_end - screen_begin, ivec2(1));

	uint scramble = hash_uvec3(uvec3(uvec2(tile), 0x91e10da5u));
	uint permuted = (candidate_index + (scramble & 7u)) & 7u;
	uvec2 lattice = uvec2(permuted & 3u, permuted >> 2u);
	uint phase = params.frame_index & 7u;
	vec2 jitter = vec2(
			hash_float(uvec3(uvec2(tile), phase ^ 0x68bc21ebu)),
			hash_float(uvec3(uvec2(tile.yx), phase ^ 0x02e5be93u)));
	vec2 candidate_uv = (vec2(lattice) + vec2(0.2) + jitter * 0.6) / vec2(4.0, 2.0);
	ivec2 offset = min(ivec2(floor(candidate_uv * vec2(screen_extent))), screen_extent - ivec2(1));
	return screen_begin + offset;
}

float directional_adaptive_receiver_weight(vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic, uvec4 packed_probe, float spatial_weight) {
	ivec2 probe_screen_position;
	vec3 probe_view;
	vec3 probe_normal;
	bool probe_dynamic;
	if (!directional_adaptive_decode_surface(packed_probe, probe_screen_position, probe_view, probe_normal, probe_dynamic) || probe_dynamic != receiver_dynamic) {
		return 0.0;
	}
	vec3 receiver_delta = probe_view - receiver_view;
	float receiver_separation = length(receiver_delta);
	float receiver_plane_distance = abs(dot(receiver_delta, receiver_normal));
	float receiver_tolerance = max(0.02 + 0.02 * receiver_separation, 1e-4);
	float plane_weight = 1.0 - smoothstep(receiver_tolerance, receiver_tolerance * 2.0, receiver_plane_distance);
	return max(spatial_weight, 0.0) * plane_weight;
}

float directional_adaptive_uniform_coverage(ivec2 screen_position, vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic) {
	ivec2 base_count = directional_adaptive_base_count();
	vec2 screen_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(screen_uv);
	ivec2 probe_base = ivec2(floor(probe_grid_position));
	vec2 probe_blend = fract(probe_grid_position);
	float coverage = 0.0;
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 probe_position = probe_base + ivec2(x, y);
			if (any(lessThan(probe_position, ivec2(0))) || any(greaterThanEqual(probe_position, base_count))) {
				continue;
			}
			float bilinear_weight = (x == 0 ? 1.0 - probe_blend.x : probe_blend.x) * (y == 0 ? 1.0 - probe_blend.y : probe_blend.y);
			coverage += directional_adaptive_receiver_weight(receiver_view, receiver_normal, receiver_dynamic, imageLoad(adaptive_probe_surface, probe_position), bilinear_weight);
		}
	}
	return coverage;
}

bool directional_adaptive_candidate_surface(ivec2 tile, uint candidate_index, out uvec4 r_packed, out ivec2 r_screen_position, out vec3 r_view_position, out vec3 r_normal, out bool r_dynamic) {
	r_screen_position = directional_adaptive_candidate_position(tile, candidate_index);
	return directional_adaptive_load_screen_surface(r_screen_position, r_packed, r_view_position, r_normal, r_dynamic);
}

#ifdef MODE_DIRECTIONAL_ADAPTIVE_SPAWN

ivec2 directional_adaptive_owner_tile(ivec2 screen_position) {
	ivec2 gi_position = clamp(screen_position * params.gi_size / params.screen_size, ivec2(0), params.gi_size - ivec2(1));
	return clamp(gi_position / max(params.probe_size, 1), ivec2(0), directional_adaptive_base_count() - ivec2(1));
}

ivec2 directional_adaptive_history_tile_offset(int index) {
	const ivec2 offsets[9] = ivec2[](
			ivec2(0, 0), ivec2(-1, 0), ivec2(1, 0),
			ivec2(0, -1), ivec2(0, 1), ivec2(-1, -1),
			ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));
	return offsets[index];
}

bool directional_adaptive_previous_owner_from_current_surface(uvec4 current_packed, out ivec2 r_previous_owner) {
	r_previous_owner = ivec2(-1);
	ivec2 current_screen_position;
	vec3 current_view;
	vec3 current_normal;
	bool current_dynamic;
	if (!directional_adaptive_decode_surface(current_packed, current_screen_position, current_view, current_normal, current_dynamic) || current_dynamic) {
		return false;
	}
	vec3 current_world = (scene_data.cam_transform * vec4(current_view, 1.0)).xyz;
	vec3 previous_view = (scene_data.previous_cam_inv_transform * vec4(current_world, 1.0)).xyz;
	vec4 previous_clip = inverse(scene_data.previous_inv_projection[params.view_index]) * vec4(previous_view, 1.0);
	if (!(previous_clip.w > 1e-6) || !directional_finite(previous_clip)) {
		return false;
	}
	vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
	float previous_device_depth = previous_clip.z / previous_clip.w;
	if (any(lessThan(previous_uv, vec2(0.0))) || any(greaterThanEqual(previous_uv, vec2(1.0))) ||
			isnan(previous_device_depth) || isinf(previous_device_depth) || previous_device_depth < 0.0 || previous_device_depth > 1.0) {
		return false;
	}
	ivec2 previous_screen_position = clamp(ivec2(floor(previous_uv * vec2(params.screen_size))), ivec2(0), params.screen_size - ivec2(1));
	r_previous_owner = directional_adaptive_owner_tile(previous_screen_position);
	return true;
}

bool directional_adaptive_reproject_previous_surface(uvec4 previous_packed, ivec2 expected_current_owner, mat4 previous_cam_transform, mat4 current_cam_inv_transform, out uvec4 r_current_packed, out ivec2 r_current_screen_position, out vec3 r_current_view_position, out vec3 r_current_normal, out bool r_current_dynamic) {
	r_current_packed = uvec4(0xffffffffu);
	r_current_screen_position = ivec2(-1);
	r_current_view_position = vec3(0.0);
	r_current_normal = vec3(0.0, 0.0, 1.0);
	r_current_dynamic = false;
	if (all(equal(previous_packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}

	ivec2 previous_screen_position = ivec2(previous_packed.xy);
	float previous_depth = uintBitsToFloat(previous_packed.z);
	vec3 previous_normal_view = unpack_surface_normal(previous_packed.w);
	bool previous_dynamic = surface_is_dynamic(previous_packed.w);
	if (!(previous_depth > 0.0) || previous_dynamic ||
			any(lessThan(previous_screen_position, ivec2(0))) || any(greaterThanEqual(previous_screen_position, params.screen_size)) ||
			!directional_finite(previous_normal_view)) {
		return false;
	}

	vec2 previous_uv = (vec2(previous_screen_position) + 0.5) / vec2(params.screen_size);
	vec4 previous_view_h = scene_data.previous_inv_projection[params.view_index] * vec4(previous_uv * 2.0 - 1.0, previous_depth, 1.0);
	if (!(abs(previous_view_h.w) > 1e-8) || !directional_finite(previous_view_h)) {
		return false;
	}
	vec3 previous_view = previous_view_h.xyz / previous_view_h.w;
	vec3 previous_world = (previous_cam_transform * vec4(previous_view, 1.0)).xyz;
	vec3 previous_normal_world = normalize(mat3(previous_cam_transform) * previous_normal_view);
	vec3 projected_current_view = (current_cam_inv_transform * vec4(previous_world, 1.0)).xyz;
	vec4 current_clip = scene_data.projection[params.view_index] * vec4(projected_current_view, 1.0);
	if (!(current_clip.w > 1e-6) || !directional_finite(current_clip) || !directional_finite(previous_world) || !directional_finite(previous_normal_world)) {
		return false;
	}
	vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
	float projected_device_depth = current_clip.z / current_clip.w;
	if (any(lessThan(current_uv, vec2(0.0))) || any(greaterThanEqual(current_uv, vec2(1.0))) ||
			isnan(projected_device_depth) || isinf(projected_device_depth) || projected_device_depth < 0.0 || projected_device_depth > 1.0) {
		return false;
	}

	vec2 projected_pixel = current_uv * vec2(params.screen_size) - 0.5;
	ivec2 projected_base = ivec2(floor(projected_pixel));
	const float relative_plane_tolerance = 0.01823;
	float normal_threshold = max(params.history_normal_threshold, 0.85);
	float best_score = -1.0;
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 candidate_screen_position = projected_base + ivec2(x, y);
			if (any(lessThan(candidate_screen_position, ivec2(0))) || any(greaterThanEqual(candidate_screen_position, params.screen_size)) ||
					any(notEqual(directional_adaptive_owner_tile(candidate_screen_position), expected_current_owner))) {
				continue;
			}
			uvec4 candidate_packed;
			vec3 candidate_view;
			vec3 candidate_normal;
			bool candidate_dynamic;
			if (!directional_adaptive_load_screen_surface(candidate_screen_position, candidate_packed, candidate_view, candidate_normal, candidate_dynamic) || candidate_dynamic) {
				continue;
			}
			vec3 current_world = (scene_data.cam_transform * vec4(candidate_view, 1.0)).xyz;
			vec3 current_normal_world = normalize(mat3(scene_data.cam_transform) * candidate_normal);
			vec3 surface_delta = current_world - previous_world;
			float depth_scale = max(abs(candidate_view.z), 1e-3);
			float current_plane_distance = abs(dot(surface_delta, current_normal_world)) / depth_scale;
			float previous_plane_distance = abs(dot(surface_delta, previous_normal_world)) / depth_scale;
			float normal_similarity = dot(current_normal_world, previous_normal_world);
			if (!directional_finite(current_world) || !directional_finite(current_normal_world) ||
					isnan(current_plane_distance) || isinf(current_plane_distance) || isnan(previous_plane_distance) || isinf(previous_plane_distance) ||
					current_plane_distance > relative_plane_tolerance || previous_plane_distance > relative_plane_tolerance || normal_similarity < normal_threshold) {
				continue;
			}
			float pixel_distance = length(vec2(candidate_screen_position) - projected_pixel);
			float screen_score = 1.0 - clamp(pixel_distance * 0.70710678, 0.0, 1.0);
			float plane_score = 1.0 - clamp(max(current_plane_distance, previous_plane_distance) / relative_plane_tolerance, 0.0, 1.0);
			float normal_score = clamp((normal_similarity - normal_threshold) / max(1.0 - normal_threshold, 1e-4), 0.0, 1.0);
			float score = (0.25 + 0.75 * screen_score) * (0.25 + 0.75 * plane_score) * (0.25 + 0.75 * normal_score);
			if (score > best_score) {
				best_score = score;
				r_current_packed = candidate_packed;
				r_current_screen_position = candidate_screen_position;
				r_current_view_position = candidate_view;
				r_current_normal = candidate_normal;
				r_current_dynamic = false;
			}
		}
	}
	return best_score >= 0.0;
}

float directional_adaptive_retained_coverage(ivec2 screen_position, vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic) {
	vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
	float coverage = 0.0;
	for (uint retained_index = 0u; retained_index < directional_adaptive_retained_count; retained_index++) {
		ivec2 retained_screen_position;
		vec3 retained_view;
		vec3 retained_normal;
		bool retained_dynamic;
		if (!directional_adaptive_decode_surface(directional_adaptive_retained_surfaces[retained_index], retained_screen_position, retained_view, retained_normal, retained_dynamic)) {
			continue;
		}
		vec2 axis_distance = abs(vec2(retained_screen_position - screen_position)) / probe_screen_extent;
		float spatial_weight = 1.0 - clamp(max(axis_distance.x, axis_distance.y), 0.0, 1.0);
		coverage = max(coverage, directional_adaptive_receiver_weight(receiver_view, receiver_normal, receiver_dynamic, directional_adaptive_retained_surfaces[retained_index], spatial_weight));
	}
	return coverage;
}

void directional_adaptive_collect_retained(ivec2 tile) {
	directional_adaptive_retained_count = 0u;
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_HISTORY_VALID) == 0u) {
		return;
	}

	ivec2 base_count = directional_adaptive_base_count();
	uint base_probe_total = uint(base_count.x * base_count.y);
	uint adaptive_capacity = base_probe_total / 2u;
	ivec2 previous_surface_size = imageSize(adaptive_previous_probe_surface);
	mat4 previous_cam_transform = inverse(scene_data.previous_cam_inv_transform);
	mat4 current_cam_inv_transform = inverse(scene_data.cam_transform);

	ivec2 previous_owners[9];
	int previous_owner_count = 0;
	for (int anchor_index = -1; anchor_index < int(SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT); anchor_index++) {
		uvec4 anchor_packed = imageLoad(adaptive_probe_surface, tile);
		if (anchor_index >= 0) {
			ivec2 anchor_screen_position;
			vec3 anchor_view;
			vec3 anchor_normal;
			bool anchor_dynamic;
			if (!directional_adaptive_candidate_surface(tile, uint(anchor_index), anchor_packed, anchor_screen_position, anchor_view, anchor_normal, anchor_dynamic)) {
				continue;
			}
		}
		ivec2 previous_owner;
		if (!directional_adaptive_previous_owner_from_current_surface(anchor_packed, previous_owner)) {
			continue;
		}
		bool duplicate_owner = false;
		for (int previous_owner_index = 0; previous_owner_index < previous_owner_count; previous_owner_index++) {
			duplicate_owner = duplicate_owner || all(equal(previous_owner, previous_owners[previous_owner_index]));
		}
		if (!duplicate_owner) {
			previous_owners[previous_owner_count++] = previous_owner;
		}
	}

	// Compact indices are not persistent identities; reproject the previous surfaces.
	for (int previous_owner_index = 0; previous_owner_index < previous_owner_count; previous_owner_index++) {
		for (int history_tile_index = 0; history_tile_index < 9; history_tile_index++) {
			ivec2 previous_tile = previous_owners[previous_owner_index] + directional_adaptive_history_tile_offset(history_tile_index);
			if (any(lessThan(previous_tile, ivec2(0))) || any(greaterThanEqual(previous_tile, base_count))) {
				continue;
			}
			bool visited = false;
			for (int prior_owner_index = 0; prior_owner_index < previous_owner_index; prior_owner_index++) {
				visited = visited || all(lessThanEqual(abs(previous_tile - previous_owners[prior_owner_index]), ivec2(1)));
			}
			if (visited) {
				continue;
			}
			uvec4 previous_tile_data = imageLoad(adaptive_previous_tile_data_input, previous_tile);
			if (previous_tile_data.x >= adaptive_capacity) {
				continue;
			}
			uint previous_count = min(min(previous_tile_data.y, SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE), adaptive_capacity - previous_tile_data.x);
			for (uint previous_offset = 0u; previous_offset < previous_count; previous_offset++) {
				uint previous_physical_index = base_probe_total + previous_tile_data.x + previous_offset;
				ivec2 previous_physical_position = ivec2(int(previous_physical_index % uint(previous_surface_size.x)), int(previous_physical_index / uint(previous_surface_size.x)));
				if (any(lessThan(previous_physical_position, ivec2(0))) || any(greaterThanEqual(previous_physical_position, previous_surface_size))) {
					continue;
				}
				uvec4 current_packed;
				ivec2 current_screen_position;
				vec3 current_view;
				vec3 current_normal;
				bool current_dynamic;
				if (!directional_adaptive_reproject_previous_surface(imageLoad(adaptive_previous_probe_surface, previous_physical_position), tile, previous_cam_transform, current_cam_inv_transform, current_packed, current_screen_position, current_view, current_normal, current_dynamic) ||
						any(notEqual(directional_adaptive_owner_tile(current_screen_position), tile)) ||
						directional_adaptive_uniform_coverage(current_screen_position, current_view, current_normal, current_dynamic) >= SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_RETIRE_COVERAGE ||
						directional_adaptive_retained_coverage(current_screen_position, current_view, current_normal, current_dynamic) >= SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
					continue;
				}
				directional_adaptive_retained_surfaces[directional_adaptive_retained_count++] = current_packed;
				if (directional_adaptive_retained_count >= SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE) {
					return;
				}
			}
		}
	}
}

float directional_adaptive_prior_coverage(ivec2 tile, uint candidate_index, ivec2 screen_position, vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic) {
	ivec2 base_count = directional_adaptive_base_count();
	vec2 screen_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(screen_uv);
	ivec2 search_base = ivec2(floor(probe_grid_position));
	vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
	vec4 corner_coverage = vec4(0.0);
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 prior_tile = search_base + ivec2(x, y);
			if (any(lessThan(prior_tile, ivec2(0))) || any(greaterThanEqual(prior_tile, base_count))) {
				continue;
			}
			uint prior_mask = imageLoad(adaptive_candidate_mark_input, prior_tile).r;
			for (uint prior_candidate = 0u; prior_candidate < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT; prior_candidate++) {
				if (prior_candidate >= candidate_index) {
					break;
				}
				if ((prior_mask & (1u << prior_candidate)) == 0u) {
					continue;
				}
				uvec4 prior_packed;
				ivec2 prior_screen_position;
				vec3 prior_view;
				vec3 prior_normal;
				bool prior_dynamic;
				if (!directional_adaptive_candidate_surface(prior_tile, prior_candidate, prior_packed, prior_screen_position, prior_view, prior_normal, prior_dynamic)) {
					continue;
				}
				vec2 axis_distance = abs(vec2(prior_screen_position - screen_position)) / probe_screen_extent;
				float spatial_weight = 1.0 - clamp(min(axis_distance.x, axis_distance.y), 0.0, 1.0);
				int corner_index = y * 2 + x;
				corner_coverage[corner_index] = max(corner_coverage[corner_index], directional_adaptive_receiver_weight(receiver_view, receiver_normal, receiver_dynamic, prior_packed, spatial_weight));
			}
		}
	}
	return dot(corner_coverage, vec4(1.0));
}

#endif

#ifdef MODE_DIRECTIONAL_ADAPTIVE_MARK

void directional_adaptive_mark_main() {
	ivec2 tile = ivec2(gl_WorkGroupID.xy);
	ivec2 base_count = directional_adaptive_base_count();
	if (any(greaterThanEqual(tile, base_count))) {
		return;
	}
	uint lane = gl_LocalInvocationIndex;
	if (lane == 0u) {
		directional_adaptive_mask = 0u;
		imageStore(adaptive_tile_data_output, tile, uvec4(0u));
		if (all(equal(tile, ivec2(0)))) {
			imageStore(adaptive_probe_counter, ivec2(0), uvec4(0u));
		}
	}
	barrier();
	if (lane < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT) {
		uvec4 packed;
		ivec2 screen_position;
		vec3 view_position;
		vec3 normal;
		bool dynamic_surface;
		if (directional_adaptive_candidate_surface(tile, lane, packed, screen_position, view_position, normal, dynamic_surface) &&
				directional_adaptive_uniform_coverage(screen_position, view_position, normal, dynamic_surface) < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
			atomicOr(directional_adaptive_mask, 1u << lane);
		}
	}
	barrier();
	if (lane == 0u) {
		imageStore(adaptive_candidate_mark_output, tile, uvec4(directional_adaptive_mask, 0u, 0u, 0u));
	}
}

#endif

#ifdef MODE_DIRECTIONAL_ADAPTIVE_SPAWN

void directional_adaptive_spawn_main() {
	ivec2 tile = ivec2(gl_WorkGroupID.xy);
	ivec2 base_count = directional_adaptive_base_count();
	if (any(greaterThanEqual(tile, base_count))) {
		return;
	}
	uint lane = gl_LocalInvocationIndex;
	if (lane == 0u) {
		directional_adaptive_spawn_mask = 0u;
		directional_adaptive_allocation_start = 0u;
		directional_adaptive_allocation_count = 0u;
		directional_adaptive_retained_count = 0u;
	}
	if (lane < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT) {
		directional_adaptive_candidate_surfaces[lane] = uvec4(0xffffffffu);
		directional_adaptive_retained_surfaces[lane] = uvec4(0xffffffffu);
	}
	barrier();
	if (lane == 0u) {
		directional_adaptive_collect_retained(tile);
	}
	barrier();

	uint marked = imageLoad(adaptive_candidate_mark_input, tile).r;
	if (lane < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT && (marked & (1u << lane)) != 0u) {
		uvec4 packed;
		ivec2 screen_position;
		vec3 view_position;
		vec3 normal;
		bool dynamic_surface;
		if (directional_adaptive_candidate_surface(tile, lane, packed, screen_position, view_position, normal, dynamic_surface)) {
			directional_adaptive_candidate_surfaces[lane] = packed;
			if (directional_adaptive_retained_coverage(screen_position, view_position, normal, dynamic_surface) < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE &&
					directional_adaptive_prior_coverage(tile, lane, screen_position, view_position, normal, dynamic_surface) < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
				atomicOr(directional_adaptive_spawn_mask, 1u << lane);
			}
		}
	}
	barrier();

	uint lower_mask = lane < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT && lane > 0u ? directional_adaptive_spawn_mask & ((1u << lane) - 1u) : 0u;
	uint local_rank = bitCount(lower_mask);
	bool accepted = lane < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT &&
			(directional_adaptive_spawn_mask & (1u << lane)) != 0u &&
			directional_adaptive_retained_count + local_rank < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE;
	if (lane == 0u) {
		uint requested = min(directional_adaptive_retained_count + uint(bitCount(directional_adaptive_spawn_mask)), SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE);
		uint base_probe_total = uint(base_count.x * base_count.y);
		uint adaptive_capacity = base_probe_total / 2u;
		uint allocation_start = imageAtomicAdd(adaptive_probe_counter, ivec2(0), requested);
		uint allocation_count = allocation_start < adaptive_capacity ? min(requested, adaptive_capacity - allocation_start) : 0u;
		directional_adaptive_allocation_start = allocation_start;
		directional_adaptive_allocation_count = allocation_count;
		imageStore(adaptive_tile_data_output, tile, uvec4(allocation_start, allocation_count, min(directional_adaptive_retained_count, allocation_count), requested));
	}
	barrier();

	uint base_probe_total = uint(base_count.x * base_count.y);
	ivec2 surface_size = imageSize(adaptive_probe_surface);
	if (lane < directional_adaptive_retained_count && lane < directional_adaptive_allocation_count) {
		uint physical_index = base_probe_total + directional_adaptive_allocation_start + lane;
		ivec2 physical_position = ivec2(int(physical_index % uint(surface_size.x)), int(physical_index / uint(surface_size.x)));
		if (all(lessThan(physical_position, surface_size))) {
			imageStore(adaptive_probe_surface, physical_position, directional_adaptive_retained_surfaces[lane]);
		}
	}
	if (accepted) {
		uint output_rank = directional_adaptive_retained_count + local_rank;
		if (output_rank < directional_adaptive_allocation_count) {
			uint physical_index = base_probe_total + directional_adaptive_allocation_start + output_rank;
			ivec2 physical_position = ivec2(int(physical_index % uint(surface_size.x)), int(physical_index / uint(surface_size.x)));
			if (all(lessThan(physical_position, surface_size))) {
				imageStore(adaptive_probe_surface, physical_position, directional_adaptive_candidate_surfaces[lane]);
			}
		}
	}
}

#endif

#endif

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
	if (any(greaterThanEqual(gi_begin, params.gi_size))) {
		imageStore(screen_probe_surface_output, probe_position, uvec4(0xffffffffu));
		return;
	}
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

#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE)

#ifdef MODE_DEBUG_MONTAGE

uint screen_probe_debug_bits;
uint screen_probe_debug_source;

void screen_probe_debug_begin() {
	screen_probe_debug_bits = 0u;
	screen_probe_debug_source = 0u;
}

void screen_probe_debug_mark(uint bits) {
	screen_probe_debug_bits |= bits;
}

void screen_probe_debug_mark_source(uint bits) {
	screen_probe_debug_source |= bits & SCREEN_PROBE_DEBUG_SOURCE_STATE_MASK;
}

void screen_probe_debug_begin_source() {
	screen_probe_debug_source = 0u;
}

void screen_probe_debug_set_source(uint source) {
	screen_probe_debug_source = (screen_probe_debug_source & ~SCREEN_PROBE_DEBUG_SOURCE_MASK) | (source & SCREEN_PROBE_DEBUG_SOURCE_MASK);
}

void screen_probe_debug_commit_source() {
	screen_probe_debug_bits |= screen_probe_debug_source;
}

void screen_probe_debug_select_source() {
	screen_probe_debug_bits = (screen_probe_debug_bits & ~SCREEN_PROBE_DEBUG_SOURCE_STATE_MASK) | screen_probe_debug_source;
}

void screen_probe_debug_store(ivec2 probe_position) {
	if (all(greaterThanEqual(probe_position, ivec2(0))) && all(lessThan(probe_position, imageSize(trace_debug_output)))) {
		imageStore(trace_debug_output, probe_position, uvec4(screen_probe_debug_bits, 0u, 0u, 0u));
	}
}

#else

void screen_probe_debug_begin() {}
void screen_probe_debug_mark(uint bits) {}
void screen_probe_debug_mark_source(uint bits) {}
void screen_probe_debug_begin_source() {}
void screen_probe_debug_set_source(uint source) {}
void screen_probe_debug_commit_source() {}
void screen_probe_debug_select_source() {}
void screen_probe_debug_store(ivec2 probe_position) {}

#endif

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

#ifdef MODE_SPECULAR_TRACE
	float max_distance = min(SCREEN_PROBE_SPECULAR_DETAIL_TRACE_MAX_DISTANCE, distance_limit);
#else
	float max_distance = min(SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE, distance_limit);
#endif
	if ((params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) == 0u || !(max_distance > 0.0) || params.detail_trace_mip_count == 0u || any(lessThanEqual(params.screen_size, ivec2(0)))) {
		return false;
	}
	if (any(isnan(ray_direction_view)) || any(isinf(ray_direction_view))) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_DETAIL_MISS | SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
		return false;
	}
	if (ray_direction_view.z > 0.0) {
		max_distance = min(max_distance, (-0.001 - origin_view.z) / ray_direction_view.z);
	}
	if (!(max_distance > 0.02)) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_DETAIL_MISS);
		return false;
	}

	vec3 screen_start;
	vec3 screen_end;
	if (!detail_trace_project(origin_view, screen_start) || !detail_trace_project(origin_view + ray_direction_view * max_distance, screen_end)) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_DETAIL_MISS);
		return false;
	}
	vec3 screen_delta = screen_end - screen_start;
	if (abs(screen_delta.z) < 1e-5 || all(lessThan(abs(screen_delta.xy), vec2(1e-7)))) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_DETAIL_MISS);
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
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_DETAIL_MISS);
		return false;
	}

	vec2 start_cell = floor(screen_start.xy * detail_size);
	vec2 next_cell_uv = (start_cell + positive_cell_step) * inverse_detail_size + cell_boundary_bias;
	vec2 start_cell_t = (next_cell_uv - screen_start.xy) * inverse_direction_xy;
	float trace_t = max(min(start_cell_t.x, start_cell_t.y), 0.0);
	int max_mip = int(params.detail_trace_mip_count) - 1;
#ifdef MODE_SPECULAR_TRACE
	int mip_level = min(2, max_mip);
	int iterations_remaining = 50;
	float receiver_exclusion = max(0.01, abs(origin_view.z) / detail_size.y);
	const float candidate_normal_threshold = -0.01;
#else
	int mip_level = 0;
	int iterations_remaining = 48;
	float receiver_exclusion = max(0.02, abs(origin_view.z) * 2.0 / detail_size.y);
	const float candidate_normal_threshold = -0.05;
#endif
	bool rejected_candidate = false;

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
					candidate_valid = !any(isnan(candidate_view)) && !any(isinf(candidate_view)) && detail_trace_geometric_normal(candidate_pixel, candidate_view, shading_normal_view, shading_normal_valid, candidate_normal_view) && dot(ray_direction_view, candidate_normal_view) < candidate_normal_threshold;
				}
				if (candidate_valid) {
					float candidate_distance = dot(candidate_view - origin_view, ray_direction_view);
#ifdef MODE_SPECULAR_TRACE
					float thickness = clamp(abs(candidate_view.z) * 0.004, 0.04, 0.25);
#else
					float thickness = clamp(abs(candidate_view.z) * 0.0015, 0.025, 0.10);
#endif
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
					screen_probe_debug_mark(SCREEN_PROBE_DEBUG_DETAIL_HIT);
					return true;
				}
			}
			rejected_candidate = true;

			trace_t = max(cell_exit_t, candidate_t + 1e-6);
#ifdef MODE_SPECULAR_TRACE
			mip_level = min(2, max_mip);
#else
			mip_level = 0;
#endif
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
	screen_probe_debug_mark(rejected_candidate ? SCREEN_PROBE_DEBUG_DETAIL_REJECTED : SCREEN_PROBE_DEBUG_DETAIL_MISS);
	return false;
}

#endif

#if defined(MODE_TRACE) || defined(MODE_SPECULAR_TRACE) || defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)

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
		if (any(isnan(r_radiance)) || any(isinf(r_radiance)) || any(lessThan(r_radiance, vec3(0.0)))) {
			screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
		}
		screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_SCREEN_IRRADIANCE_CACHE);
		return true;
	}
#endif
	vec3 endpoint_scaled = endpoint_world - scene_data.cam_transform[3].xyz;
	endpoint_scaled.y *= hddagi.y_mult;
	vec3 normal_scaled = endpoint_normal_world;
	normal_scaled.y *= hddagi.y_mult;
	float normal_length_squared = dot(normal_scaled, normal_scaled);
	if (!(normal_length_squared > 1e-8) || any(isnan(endpoint_scaled)) || any(isinf(endpoint_scaled)) || any(isnan(normal_scaled)) || any(isinf(normal_scaled))) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
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
			if (any(isnan(r_radiance)) || any(isinf(r_radiance)) || any(lessThan(r_radiance, vec3(0.0)))) {
				screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
			}
			screen_probe_debug_set_source(irradiance_cache_lookup.has_radiance ? SCREEN_PROBE_DEBUG_SCREEN_IRRADIANCE_CACHE : SCREEN_PROBE_DEBUG_SCREEN_HDDAGI);
#else
			r_radiance = filtered_radiance;
			screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_SCREEN_HDDAGI);
#endif
			return true;
		}
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
	}
#ifdef MODE_IRRADIANCE_CACHE_QUERY
	if (irradiance_cache_lookup.has_radiance) {
		r_radiance = irradiance_cache_lookup.radiance;
		if (any(isnan(r_radiance)) || any(isinf(r_radiance)) || any(lessThan(r_radiance, vec3(0.0)))) {
			screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
		}
		screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_SCREEN_IRRADIANCE_CACHE);
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
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
		return false;
	}
	mat3 camera_basis = mat3(scene_data.cam_transform);
	vec3 ray_position = camera_basis * origin_view;
	vec3 trace_origin_world = ray_position + scene_data.cam_transform[3].xyz;

	if (trace_screen_detail(origin_view, ray_direction_view, detail_distance_limit, r_endpoint_world, r_endpoint_normal_world)) {
		r_hit_cascade = HDDAGI_TRACE_SCREEN_ENDPOINT;
		if (!query_endpoint_radiance(r_endpoint_world, r_endpoint_normal_world, trace_origin_world, r_radiance)) {
			r_radiance = sample_environment(ray_direction_view);
			r_hit_cascade = HDDAGI_TRACE_SCREEN_ENDPOINT_SKY_FALLBACK;
			screen_probe_debug_mark_source(SCREEN_PROBE_DEBUG_SCREEN_RADIANCE_FALLBACK);
			screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_SKY);
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
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
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
			if (!hddagi_irradiance_cache_is_finite(r_radiance) || any(lessThan(r_radiance, vec3(0.0)))) {
				screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
			}
			screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_HDDA_IRRADIANCE_CACHE);
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
	screen_probe_debug_set_source(irradiance_cache_lookup.has_radiance ? SCREEN_PROBE_DEBUG_HDDA_IRRADIANCE_CACHE : SCREEN_PROBE_DEBUG_HDDA_VOXEL);
#else
	screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_HDDA_VOXEL);
#endif
	bool radiance_valid = !any(isnan(r_radiance)) && !any(isinf(r_radiance)) && !any(lessThan(r_radiance, vec3(0.0)));
	if (!radiance_valid) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
	}
	return radiance_valid;
}

bool trace_hddagi_radiance(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_view, out vec3 r_radiance, out float r_hit_distance) {
	int hit_cascade;
	vec3 endpoint_world;
	vec3 endpoint_normal_world;
#ifdef MODE_SPECULAR_TRACE
	const float detail_distance_limit = SCREEN_PROBE_SPECULAR_DETAIL_TRACE_MAX_DISTANCE;
#else
	const float detail_distance_limit = SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE;
#endif
	bool hit = trace_hddagi_sample(origin_position, origin_depth, origin_normal, ray_direction_view, detail_distance_limit, r_radiance, hit_cascade, endpoint_world, endpoint_normal_world);
	r_hit_distance = 0.0;
	if (hit && (hit_cascade >= 0 || hit_cascade <= HDDAGI_TRACE_SCREEN_ENDPOINT) && !any(isnan(endpoint_world)) && !any(isinf(endpoint_world))) {
		vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
		vec3 origin_world = (scene_data.cam_transform * vec4(compute_view_position(vec3(origin_uv, origin_depth)), 1.0)).xyz;
		r_hit_distance = clamp(length(endpoint_world - origin_world), 0.0, 65504.0);
		if (isnan(r_hit_distance) || isinf(r_hit_distance)) {
			screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
			r_hit_distance = 0.0;
		}
	} else if (hit && (any(isnan(endpoint_world)) || any(isinf(endpoint_world)))) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
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

#ifdef MODE_DIRECTIONAL_TRACE

shared ivec2 directional_history_probe;
shared uint directional_history_age;
shared uint directional_history_valid;
shared vec4 directional_importance_footprint_normals[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared float directional_importance_pdf[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared float directional_importance_lighting[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared float directional_importance_lighting_sum;
shared uint directional_importance_sorted_indices[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared uint directional_importance_sorted_positions[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared vec4 directional_importance_trace_samples[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared uint directional_importance_synthetic_hits[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];

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

bool directional_temporal_reproject_jitter_neutral(vec2 current_grid_uv, vec3 current_view_position, vec3 world_position, out vec2 r_previous_uv, out vec3 r_previous_view_position) {
	vec4 current_stable_clip = scene_data.temporal_projection[params.view_index] * vec4(current_view_position, 1.0);
	r_previous_view_position = (scene_data.previous_cam_inv_transform * vec4(world_position, 1.0)).xyz;
	vec4 previous_stable_clip = scene_data.previous_temporal_projection[params.view_index] * vec4(r_previous_view_position, 1.0);
	if (current_stable_clip.w <= 1e-6 || previous_stable_clip.w <= 1e-6 || !directional_finite(current_stable_clip) || !directional_finite(previous_stable_clip)) {
		r_previous_uv = vec2(0.0);
		return false;
	}
	vec2 current_stable_uv = current_stable_clip.xy / current_stable_clip.w * 0.5 + 0.5;
	vec2 previous_stable_uv = previous_stable_clip.xy / previous_stable_clip.w * 0.5 + 0.5;
	r_previous_uv = current_grid_uv + previous_stable_uv - current_stable_uv;
	return directional_finite(vec3(r_previous_uv, 0.0));
}

bool directional_select_history(ivec2 probe_position, ivec2 current_screen_position, float current_depth, vec3 current_normal_view, bool current_dynamic, out ivec2 r_previous_probe, out uint r_previous_age) {
	r_previous_probe = ivec2(-1);
	r_previous_age = 0u;
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_HISTORY_VALID) == 0u) {
		return false;
	}

	vec2 current_uv = (vec2(current_screen_position) + 0.5) / vec2(params.screen_size);
	vec3 current_view = compute_view_position(vec3(current_uv, current_depth));
	vec3 current_world = (scene_data.cam_transform * vec4(current_view, 1.0)).xyz;
	vec3 current_normal_world = normalize(mat3(scene_data.cam_transform) * current_normal_view);
	if (!directional_finite(current_view) || !directional_finite(current_world) || !directional_finite(current_normal_world)) {
		return false;
	}

	ivec2 base_probe_count = (params.gi_size + ivec2(max(params.probe_size, 1)) - ivec2(1)) / max(params.probe_size, 1);
	uint base_probe_total = uint(base_probe_count.x * base_probe_count.y);
	ivec2 current_surface_size = imageSize(screen_probe_surface_input);
	uint physical_index = uint(probe_position.y * current_surface_size.x + probe_position.x);
	bool adaptive_probe = (params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_ADAPTIVE) != 0u && physical_index >= base_probe_total;
	ivec2 gi_origin = min(probe_position, base_probe_count - ivec2(1)) * params.probe_size;
	ivec2 gi_end = min(gi_origin + ivec2(params.probe_size), params.gi_size);
	vec2 current_grid_uv = adaptive_probe ? current_uv : (vec2(gi_origin) + vec2(gi_end)) * 0.5 / vec2(params.gi_size);
	vec2 previous_grid_uv;
	vec3 previous_receiver_view = (scene_data.previous_cam_inv_transform * vec4(current_world, 1.0)).xyz;
	vec2 receiver_motion = vec2(0.0);
	vec2 previous_receiver_raster_uv = vec2(-1.0);
	bool previous_receiver_raster_valid = false;
	if (current_dynamic) {
		if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID) == 0u) {
			return false;
		}
		receiver_motion = texelFetch(sampler2D(directional_velocity_buffer, linear_sampler), current_screen_position, 0).xy;
		if (any(isnan(receiver_motion)) || any(isinf(receiver_motion))) {
			return false;
		}
		previous_grid_uv = current_grid_uv + receiver_motion;
	} else if (!directional_temporal_reproject_jitter_neutral(current_grid_uv, current_view, current_world, previous_grid_uv, previous_receiver_view)) {
		return false;
	}
	if (any(lessThan(previous_grid_uv, vec2(0.0))) || any(greaterThanEqual(previous_grid_uv, vec2(1.0)))) {
		return false;
	}

	if (adaptive_probe || current_dynamic) {
		vec4 previous_raster_clip = inverse(scene_data.previous_inv_projection[params.view_index]) * vec4(previous_receiver_view, 1.0);
		if (previous_raster_clip.w > 1e-6 && directional_finite(previous_raster_clip)) {
			vec2 previous_raster_uv = previous_raster_clip.xy / previous_raster_clip.w * 0.5 + 0.5;
			float previous_raster_depth = previous_raster_clip.z / previous_raster_clip.w;
			if (current_dynamic) {
				vec4 current_stable_clip = scene_data.temporal_projection[params.view_index] * vec4(current_view, 1.0);
				vec4 previous_stable_clip = scene_data.previous_temporal_projection[params.view_index] * vec4(previous_receiver_view, 1.0);
				if (!(current_stable_clip.w > 1e-6) || !(previous_stable_clip.w > 1e-6) || !directional_finite(current_stable_clip) || !directional_finite(previous_stable_clip)) {
					return false;
				}
				vec2 current_stable_uv = current_stable_clip.xy / current_stable_clip.w * 0.5 + 0.5;
				vec2 previous_stable_uv = previous_stable_clip.xy / previous_stable_clip.w * 0.5 + 0.5;
				previous_receiver_raster_uv = current_stable_uv + receiver_motion + previous_raster_uv - previous_stable_uv;
			} else {
				previous_receiver_raster_uv = previous_raster_uv;
			}
			previous_receiver_raster_valid = previous_raster_depth >= 0.0 && previous_raster_depth <= 1.0 &&
					!isnan(previous_raster_depth) && !isinf(previous_raster_depth) && directional_finite(vec3(previous_receiver_raster_uv, 0.0)) &&
					all(greaterThanEqual(previous_receiver_raster_uv, vec2(0.0))) && all(lessThan(previous_receiver_raster_uv, vec2(1.0)));
		}
		if (current_dynamic && !previous_receiver_raster_valid) {
			return false;
		}
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

	if (adaptive_probe && previous_receiver_raster_valid) {
		vec2 adaptive_previous_probe_texel = directional_grid_uv_to_probe_texel(previous_receiver_raster_uv);
		ivec2 adaptive_search_base = ivec2(floor(adaptive_previous_probe_texel));
		ivec2 previous_tile_data_size = imageSize(directional_previous_adaptive_tile_data_input);
		uint adaptive_capacity = base_probe_total / 2u;
		vec2 previous_screen_position = previous_receiver_raster_uv * vec2(params.screen_size) - 0.5;
		vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
		const float adaptive_relative_plane_tolerance = 0.01;
		const float adaptive_relative_view_z_tolerance = 0.015;
		const float adaptive_screen_footprint_limit = 0.75;
		float adaptive_normal_threshold = max(params.history_normal_threshold, 0.9);
		float dynamic_normal_threshold = max(params.history_normal_threshold, 0.9);
		uint best_age = 0u;
		uint best_physical_index = 0xffffffffu;

		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 2; x++) {
				ivec2 owner_tile = adaptive_search_base + ivec2(x, y);
				if (any(lessThan(owner_tile, ivec2(0))) || any(greaterThanEqual(owner_tile, base_probe_count)) || any(greaterThanEqual(owner_tile, previous_tile_data_size))) {
					continue;
				}
				uvec4 previous_tile_data = imageLoad(directional_previous_adaptive_tile_data_input, owner_tile);
				if (previous_tile_data.x >= adaptive_capacity) {
					continue;
				}
				uint previous_count = min(min(previous_tile_data.y, SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE), adaptive_capacity - previous_tile_data.x);
				for (uint previous_offset = 0u; previous_offset < previous_count; previous_offset++) {
					uint previous_physical_index = base_probe_total + previous_tile_data.x + previous_offset;
					ivec2 previous_probe = ivec2(int(previous_physical_index % uint(previous_surface_size.x)), int(previous_physical_index / uint(previous_surface_size.x)));
					if (any(lessThan(previous_probe, ivec2(0))) || any(greaterThanEqual(previous_probe, previous_surface_size)) || any(greaterThanEqual(previous_probe, previous_age_size))) {
						continue;
					}
					uint candidate_age = imageLoad(directional_previous_history_age_input, previous_probe).r;
					if (candidate_age == 0u) {
						continue;
					}

					uvec4 previous_packed = imageLoad(directional_previous_surface_input, previous_probe);
					ivec2 previous_screen_position_packed;
					float previous_depth;
					vec3 previous_normal_view;
					bool previous_dynamic;
					if (!directional_decode_history_surface(previous_packed, previous_screen_position_packed, previous_depth, previous_normal_view, previous_dynamic) || previous_dynamic != current_dynamic) {
						continue;
					}
					vec2 screen_axis_distance = abs(vec2(previous_screen_position_packed) - previous_screen_position) / probe_screen_extent;
					float screen_distance = max(screen_axis_distance.x, screen_axis_distance.y);
					if (screen_distance > adaptive_screen_footprint_limit) {
						continue;
					}

					vec3 previous_world;
					vec3 previous_normal_world;
					if (!directional_previous_surface_world(previous_packed, previous_cam_transform, previous_world, previous_normal_world, previous_dynamic) || previous_dynamic != current_dynamic) {
						continue;
					}
					vec3 candidate_previous_view = (scene_data.previous_cam_inv_transform * vec4(previous_world, 1.0)).xyz;
					float view_z_distance = abs(candidate_previous_view.z - previous_receiver_view.z);
					float relative_view_z = view_z_distance / max(abs(previous_receiver_view.z), 1e-3);
					float normal_similarity = dot(current_normal_world, previous_normal_world);
					float screen_score = 1.0 - clamp(screen_distance / adaptive_screen_footprint_limit, 0.0, 1.0);
					float score;
					if (current_dynamic) {
						float dynamic_view_z_tolerance = 0.02 + 0.02 * max(abs(candidate_previous_view.z), abs(previous_receiver_view.z));
						if (isnan(view_z_distance) || isinf(view_z_distance) || view_z_distance > dynamic_view_z_tolerance || normal_similarity < dynamic_normal_threshold) {
							continue;
						}
						float view_z_score = 1.0 - clamp(view_z_distance / dynamic_view_z_tolerance, 0.0, 1.0);
						float normal_score = clamp((normal_similarity - dynamic_normal_threshold) / max(1.0 - dynamic_normal_threshold, 1e-4), 0.0, 1.0);
						score = (0.25 + 0.75 * view_z_score) * (0.25 + 0.75 * normal_score) * (0.25 + 0.75 * screen_score);
					} else {
						vec3 receiver_delta = previous_world - current_world;
						float current_plane_distance = abs(dot(receiver_delta, current_normal_world)) / current_scene_depth;
						float previous_plane_distance = abs(dot(receiver_delta, previous_normal_world)) / current_scene_depth;
						if (isnan(relative_view_z) || isinf(relative_view_z) || relative_view_z > adaptive_relative_view_z_tolerance ||
								isnan(current_plane_distance) || isinf(current_plane_distance) || isnan(previous_plane_distance) || isinf(previous_plane_distance) ||
								current_plane_distance > adaptive_relative_plane_tolerance || previous_plane_distance > adaptive_relative_plane_tolerance ||
								normal_similarity < adaptive_normal_threshold) {
							continue;
						}
						float plane_score = 1.0 - clamp(max(current_plane_distance, previous_plane_distance) / adaptive_relative_plane_tolerance, 0.0, 1.0);
						float normal_score = clamp((normal_similarity - adaptive_normal_threshold) / max(1.0 - adaptive_normal_threshold, 1e-4), 0.0, 1.0);
						score = (0.25 + 0.75 * plane_score) * (0.25 + 0.75 * normal_score) * (0.25 + 0.75 * screen_score);
					}
					if (score < 0.1) {
						continue;
					}
					bool age_wins = abs(score - best_score) <= 1e-8 && candidate_age > best_age;
					bool index_wins = abs(score - best_score) <= 1e-8 && candidate_age == best_age && previous_physical_index < best_physical_index;
					if (score > best_score + 1e-8 || age_wins || index_wins) {
						best_score = score;
						best_age = candidate_age;
						best_physical_index = previous_physical_index;
						r_previous_probe = previous_probe;
						r_previous_age = min(candidate_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE);
					}
				}
			}
		}
		if (best_score >= 0.0) {
			return true;
		}
	}

	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 candidate = search_base + ivec2(x, y);
			vec2 axis_weight = vec2(x == 0 ? 1.0 - search_fraction.x : search_fraction.x, y == 0 ? 1.0 - search_fraction.y : search_fraction.y);
			float footprint_weight = axis_weight.x * axis_weight.y;
			if (!(footprint_weight > 1e-8) || any(lessThan(candidate, ivec2(0))) || any(greaterThanEqual(candidate, base_probe_count)) ||
					any(greaterThanEqual(candidate, previous_surface_size)) || any(greaterThanEqual(candidate, previous_age_size))) {
				continue;
			}
			uint candidate_age = imageLoad(directional_previous_history_age_input, candidate).r;
			if (candidate_age == 0u) {
				continue;
			}

			uvec4 previous_packed = imageLoad(directional_previous_surface_input, candidate);
			ivec2 previous_screen_position;
			float previous_depth;
			vec3 previous_normal_view;
			bool previous_dynamic;
			if (!directional_decode_history_surface(previous_packed, previous_screen_position, previous_depth, previous_normal_view, previous_dynamic) || previous_dynamic != current_dynamic) {
				continue;
			}
			vec3 previous_world;
			vec3 previous_normal_world;
			if (!directional_previous_surface_world(previous_packed, previous_cam_transform, previous_world, previous_normal_world, previous_dynamic) || previous_dynamic != current_dynamic) {
				continue;
			}
			float normal_similarity = dot(current_normal_world, previous_normal_world);
			float score = -1.0;
			if (current_dynamic) {
				vec2 previous_screen_position_expected = previous_receiver_raster_uv * vec2(params.screen_size) - 0.5;
				vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
				vec2 screen_axis_distance = abs(vec2(previous_screen_position) - previous_screen_position_expected) / probe_screen_extent;
				float screen_distance = max(screen_axis_distance.x, screen_axis_distance.y);
				vec3 candidate_previous_view = (scene_data.previous_cam_inv_transform * vec4(previous_world, 1.0)).xyz;
				float view_z_distance = abs(candidate_previous_view.z - previous_receiver_view.z);
				float dynamic_view_z_tolerance = 0.02 + 0.02 * max(abs(candidate_previous_view.z), abs(previous_receiver_view.z));
				const float dynamic_screen_footprint_limit = 0.75;
				float dynamic_normal_threshold = max(params.history_normal_threshold, 0.85);
				if (!isnan(screen_distance) && !isinf(screen_distance) && screen_distance <= dynamic_screen_footprint_limit &&
						!isnan(view_z_distance) && !isinf(view_z_distance) && view_z_distance <= dynamic_view_z_tolerance && normal_similarity >= dynamic_normal_threshold) {
					float screen_score = 1.0 - clamp(screen_distance / dynamic_screen_footprint_limit, 0.0, 1.0);
					float view_z_score = 1.0 - clamp(view_z_distance / dynamic_view_z_tolerance, 0.0, 1.0);
					float normal_score = clamp((normal_similarity - dynamic_normal_threshold) / max(1.0 - dynamic_normal_threshold, 1e-4), 0.0, 1.0);
					float dynamic_score = footprint_weight * (0.25 + 0.75 * screen_score) * (0.25 + 0.75 * view_z_score) * (0.25 + 0.75 * normal_score);
					score = dynamic_score >= 0.1 ? dynamic_score : -1.0;
				}
			} else {
				vec3 receiver_delta = previous_world - current_world;
				float relative_plane_distance = abs(dot(receiver_delta, current_normal_world)) / current_scene_depth;
				if (!isnan(relative_plane_distance) && !isinf(relative_plane_distance) && relative_plane_distance <= relative_plane_tolerance && normal_similarity >= normal_threshold) {
					float plane_score = 1.0 - clamp(relative_plane_distance / relative_plane_tolerance, 0.0, 1.0);
					float normal_score = clamp((normal_similarity - normal_threshold) / max(1.0 - normal_threshold, 1e-4), 0.0, 1.0);
					score = footprint_weight * (0.25 + 0.75 * plane_score) * (0.25 + 0.75 * normal_score);
				}
			}
			if (score >= 0.0) {
				bool coordinate_wins = abs(score - best_score) <= 1e-8 && (candidate.y < r_previous_probe.y || (candidate.y == r_previous_probe.y && candidate.x < r_previous_probe.x));
				if (score > best_score + 1e-8 || coordinate_wins) {
					best_score = score;
					r_previous_probe = candidate;
					r_previous_age = adaptive_probe ? 0u : min(candidate_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE);
				}
			}
		}
	}
	return best_score >= 0.0;
}

uint directional_pack_importance_ray_info(uvec2 texel_coordinate, uint level) {
	return (texel_coordinate.x & 0x3fu) | ((texel_coordinate.y & 0x3fu) << 6u) | ((level & 0xfu) << 12u);
}

void directional_unpack_importance_ray_info(uint ray_info, out uvec2 r_texel_coordinate, out uint r_level) {
	r_texel_coordinate = uvec2(ray_info & 0x3fu, (ray_info >> 6u) & 0x3fu);
	r_level = (ray_info >> 12u) & 0xfu;
}

vec3 directional_importance_ray_to_world(ivec2 jitter_key, uint ray_info) {
	uvec2 texel_coordinate;
	uint level;
	directional_unpack_importance_ray_info(ray_info, texel_coordinate, level);
	uint mip_size = uint(SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_RESOLUTION) >> level;
	vec2 ray_uv = (vec2(texel_coordinate) + directional_bucket_jitter(jitter_key)) / float(max(mip_size, 1u));
	return directional_square_to_sphere(ray_uv);
}

vec4 directional_trace_world_direction(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_world, out bool r_synthetic_self_hit) {
	vec3 ray_direction_view = normalize(transpose(mat3(scene_data.cam_transform)) * ray_direction_world);
	vec3 incident_radiance;
	float hit_distance;
	r_synthetic_self_hit = false;
	if (!trace_hddagi_radiance(origin_position, origin_depth, origin_normal, ray_direction_view, incident_radiance, hit_distance)) {
		incident_radiance = sample_environment(ray_direction_view);
		hit_distance = SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
	} else if (!(hit_distance > 1e-4)) {
		incident_radiance = vec3(0.0);
		hit_distance = 0.0;
		r_synthetic_self_hit = true;
	}
	vec3 scene_radiance = directional_clamp_radiance(incident_radiance * hddagi.energy);
	return directional_sanitize_fp16(vec4(scene_radiance, hit_distance));
}

vec4 directional_apply_stable_history(vec4 current_sample, ivec2 direction_texel, bool bypass_history) {
	if (!bypass_history && directional_history_valid != 0u) {
		ivec2 previous_atlas_position = directional_history_probe * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
		ivec2 previous_atlas_size = textureSize(sampler2D(directional_previous_radiance_input, linear_sampler), 0);
		if (all(greaterThanEqual(previous_atlas_position, ivec2(0))) && all(lessThan(previous_atlas_position, previous_atlas_size))) {
			vec4 previous_sample = texelFetch(sampler2D(directional_previous_radiance_input, linear_sampler), previous_atlas_position, 0);
			if (directional_finite(previous_sample)) {
				previous_sample = clamp(previous_sample, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX));
				current_sample.rgb = mix(previous_sample.rgb, current_sample.rgb, 0.5);
			}
		}
	}
	return directional_sanitize_fp16(current_sample);
}

vec4 directional_importance_load_footprint_normal(ivec2 origin_position, float origin_depth, vec3 origin_normal, ivec2 footprint_texel) {
	bool center_sample = all(equal(footprint_texel, ivec2(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE / 2)));
	if (center_sample) {
		vec3 center_normal_world = normalize(mat3(scene_data.cam_transform) * origin_normal);
		return directional_finite(center_normal_world) ? vec4(center_normal_world, 1.0) : vec4(0.0);
	}

	vec2 gi_to_screen = vec2(params.screen_size) / vec2(max(params.gi_size, ivec2(1)));
	vec2 footprint_radius = float(max(params.probe_size, 1)) * gi_to_screen;
	vec2 footprint_unit = (vec2(footprint_texel) + 0.5) / float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE) * 2.0 - 1.0;
	ivec2 sample_position = clamp(origin_position + ivec2(round(footprint_unit * footprint_radius)), ivec2(0), params.screen_size - ivec2(1));

	float sample_depth = texelFetch(sampler2D(directional_depth_buffer, linear_sampler), sample_position, 0).r;
	vec3 sample_normal;
	vec4 normal_roughness = texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), sample_position, 0);
	if (!(sample_depth > 0.0) || !decode_normal(normal_roughness.xyz, sample_normal)) {
		return vec4(0.0);
	}

	vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
	vec2 sample_uv = (vec2(sample_position) + 0.5) / vec2(params.screen_size);
	vec3 origin_view_position = compute_view_position(vec3(origin_uv, origin_depth));
	vec3 sample_view_position = compute_view_position(vec3(sample_uv, sample_depth));
	if (!directional_finite(origin_view_position) || !directional_finite(sample_view_position)) {
		return vec4(0.0);
	}

	float plane_distance = abs(dot(origin_view_position - sample_view_position, sample_normal));
	float relative_plane_distance = plane_distance / max(abs(origin_view_position.z), 1e-3);
	float depth_weight = exp2(-10000.0 * relative_plane_distance * relative_plane_distance);
	if (!(depth_weight > 0.01) || isnan(depth_weight) || isinf(depth_weight)) {
		return vec4(0.0);
	}
	vec3 sample_normal_world = normalize(mat3(scene_data.cam_transform) * sample_normal);
	return directional_finite(sample_normal_world) ? vec4(sample_normal_world, 1.0) : vec4(0.0);
}

float directional_importance_brdf_pdf(vec3 direction_world) {
	float brdf_sum = 0.0;
	float valid_count = 0.0;
	for (uint sample_index = 0u; sample_index < uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT); sample_index++) {
		vec4 footprint_normal = directional_importance_footprint_normals[sample_index];
		if (footprint_normal.w > 0.5) {
			brdf_sum += max(dot(footprint_normal.xyz, direction_world), 0.0);
			valid_count += 1.0;
		}
	}
	return valid_count > 0.0 ? brdf_sum / valid_count : 1.0;
}

bool directional_importance_merge_active(uint merge_index) {
	uint low_index_2 = merge_index * 3u + 2u;
	uint high_index = uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT - 1) - merge_index;
	if (low_index_2 >= high_index) {
		return false;
	}
	float low_pdf = directional_importance_pdf[directional_importance_sorted_indices[low_index_2]];
	float high_pdf = directional_importance_pdf[directional_importance_sorted_indices[high_index]];
	return low_pdf < SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_MIN_PDF && high_pdf >= SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_MIN_PDF;
}

uint directional_importance_build_ray_info(uint sorted_position) {
	uint source_index = directional_importance_sorted_indices[sorted_position];
	uvec2 source_texel = uvec2(source_index % uint(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE), source_index / uint(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE));
	uint source_level = SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_UNIFORM_LEVEL;
	uint low_merge_index = sorted_position / 3u;
	if (directional_importance_merge_active(low_merge_index)) {
		uint high_sorted_position = uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT - 1) - low_merge_index;
		uint high_source_index = directional_importance_sorted_indices[high_sorted_position];
		uvec2 high_source_texel = uvec2(high_source_index % uint(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE), high_source_index / uint(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE));
		uint child_index = sorted_position % 3u + 1u;
		uvec2 child_offset = uvec2(child_index & 1u, child_index >> 1u);
		source_texel = high_source_texel * 2u + child_offset;
		source_level = 0u;
	} else {
		uint high_merge_index = uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT - 1) - sorted_position;
		if (directional_importance_merge_active(high_merge_index)) {
			source_texel *= 2u;
			source_level = 0u;
		}
	}
	return directional_pack_importance_ray_info(source_texel, source_level);
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
		if ((params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) != 0u) {
			imageStore(directional_trace_count_output, atlas_position, uvec4(0u));
		}
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

	ivec2 base_probe_count = (params.gi_size + ivec2(max(params.probe_size, 1)) - ivec2(1)) / max(params.probe_size, 1);
	uint base_probe_total = uint(base_probe_count.x * base_probe_count.y);
	uint physical_index = uint(probe_position.y * surface_size.x + probe_position.x);
	ivec2 jitter_key = (params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_ADAPTIVE) != 0u && physical_index >= base_probe_total ? origin_position : probe_position;
	bool structured_importance = (params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) != 0u && directional_history_valid != 0u;
	if (!structured_importance) {
		bool synthetic_self_hit;
		vec3 ray_direction_world = directional_jittered_bucket_to_world(jitter_key, direction_texel);
		vec4 directional_sample = directional_trace_world_direction(origin_position, origin_depth, origin_normal, ray_direction_world, synthetic_self_hit);
		directional_sample = directional_apply_stable_history(directional_sample, direction_texel, synthetic_self_hit);
		if ((params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) != 0u) {
			imageStore(directional_trace_count_output, atlas_position, uvec4(1u));
		}
		imageStore(raw_radiance_output, atlas_position, directional_sample);
		return;
	}

	uint lane = gl_LocalInvocationIndex;
	directional_importance_footprint_normals[lane] = directional_importance_load_footprint_normal(origin_position, origin_depth, origin_normal, direction_texel);
	barrier();

	vec3 coarse_direction_world = directional_bucket_to_world(direction_texel);
	float brdf_pdf = directional_importance_brdf_pdf(coarse_direction_world);
	float lighting = 0.0;
	ivec2 previous_filtered_position = directional_history_probe * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
	ivec2 previous_filtered_size = textureSize(sampler2D(directional_previous_filtered_radiance_input, linear_sampler), 0);
	if (all(greaterThanEqual(previous_filtered_position, ivec2(0))) && all(lessThan(previous_filtered_position, previous_filtered_size))) {
		vec3 previous_filtered = texelFetch(sampler2D(directional_previous_filtered_radiance_input, linear_sampler), previous_filtered_position, 0).rgb;
		if (directional_finite(previous_filtered)) {
			lighting = max(dot(max(previous_filtered, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722)), 0.0);
		}
	}
	directional_importance_lighting[lane] = lighting;
	barrier();

	if (lane == 0u) {
		directional_importance_lighting_sum = 0.0;
		for (uint lighting_index = 0u; lighting_index < uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT); lighting_index++) {
			directional_importance_lighting_sum += directional_importance_lighting[lighting_index];
		}
	}
	barrier();
	float lighting_sum = directional_importance_lighting_sum;
	float combined_pdf = brdf_pdf;
	if (lighting_sum > 1e-6 && !isnan(lighting_sum) && !isinf(lighting_sum)) {
		combined_pdf *= lighting * float(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT) / lighting_sum;
		if (brdf_pdf >= SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_MIN_PDF) {
			combined_pdf = max(combined_pdf, SCREEN_PROBE_DIRECTIONAL_IMPORTANCE_MIN_PDF);
		}
	}
	if (isnan(combined_pdf) || isinf(combined_pdf)) {
		combined_pdf = 1.0;
	}
	directional_importance_pdf[lane] = max(combined_pdf, 0.0);
	barrier();

	float lane_pdf = directional_importance_pdf[lane];
	uint sorted_rank = 0u;
	for (uint other_lane = 0u; other_lane < uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT); other_lane++) {
		float other_pdf = directional_importance_pdf[other_lane];
		if (other_pdf < lane_pdf || (other_pdf == lane_pdf && other_lane > lane)) {
			sorted_rank++;
		}
	}
	directional_importance_sorted_indices[sorted_rank] = lane;
	directional_importance_sorted_positions[lane] = sorted_rank;
	barrier();

	uint ray_info = directional_importance_build_ray_info(lane);
	bool synthetic_self_hit;
	vec3 ray_direction_world = directional_importance_ray_to_world(jitter_key, ray_info);
	directional_importance_trace_samples[lane] = directional_trace_world_direction(origin_position, origin_depth, origin_normal, ray_direction_world, synthetic_self_hit);
	directional_importance_synthetic_hits[lane] = synthetic_self_hit ? 1u : 0u;
	barrier();

	vec3 scattered_radiance = vec3(0.0);
	float scattered_hit_distance = SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
	uint scattered_trace_count = 0u;
	bool scattered_synthetic_self_hit = false;
	uint source_sorted_position = directional_importance_sorted_positions[lane];
	uint low_merge_index = source_sorted_position / 3u;
	if (!directional_importance_merge_active(low_merge_index)) {
		uint high_merge_index = uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT - 1) - source_sorted_position;
		if (directional_importance_merge_active(high_merge_index)) {
			for (uint child_index = 0u; child_index < 4u; child_index++) {
				uint trace_lane = child_index == 0u ? source_sorted_position : high_merge_index * 3u + child_index - 1u;
				vec4 trace_sample = directional_importance_trace_samples[trace_lane];
				scattered_radiance += trace_sample.rgb * 0.25;
				scattered_hit_distance = min(scattered_hit_distance, trace_sample.a);
				scattered_trace_count++;
				scattered_synthetic_self_hit = scattered_synthetic_self_hit || directional_importance_synthetic_hits[trace_lane] != 0u;
			}
		} else {
			vec4 trace_sample = directional_importance_trace_samples[source_sorted_position];
			scattered_radiance = trace_sample.rgb;
			scattered_hit_distance = trace_sample.a;
			scattered_trace_count = 1u;
			scattered_synthetic_self_hit = directional_importance_synthetic_hits[source_sorted_position] != 0u;
		}
	}

	vec4 directional_sample = vec4(directional_clamp_radiance(scattered_radiance), scattered_hit_distance);
	if (scattered_trace_count > 0u) {
		directional_sample = directional_apply_stable_history(directional_sample, direction_texel, scattered_synthetic_self_hit);
	}
	imageStore(directional_trace_count_output, atlas_position, uvec4(scattered_trace_count));
	imageStore(raw_radiance_output, atlas_position, directional_sanitize_fp16(directional_sample));
}

#endif

#ifdef MODE_SPECULAR_TRACE

const float SPECULAR_INVALID_VIEW_Z = 65505.0;

bool specular_load_full_resolution_surface(ivec2 screen_position, out float r_depth, out vec3 r_normal, out float r_roughness, out bool r_dynamic) {
	r_depth = texelFetch(sampler2D(specular_depth_buffer, linear_sampler), screen_position, 0).r;
	r_dynamic = false;
	if (r_depth <= 0.0) {
		return false;
	}
	vec4 normal_roughness = texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), screen_position, 0);
	r_dynamic = normal_roughness.w > 0.5;
	float encoded_roughness = r_dynamic ? 1.0 - normal_roughness.w : normal_roughness.w;
	r_roughness = clamp(encoded_roughness / (127.0 / 255.0), 0.0, 1.0);
	return decode_normal(normal_roughness.xyz, r_normal);
}

bool specular_select_full_resolution_surface(ivec2 output_position, ivec2 output_size, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal, out float r_roughness, out bool r_dynamic) {
	ivec2 footprint_begin = (output_position * params.screen_size + output_size - ivec2(1)) / output_size;
	ivec2 footprint_end = ((output_position + ivec2(1)) * params.screen_size + output_size - ivec2(1)) / output_size;
	footprint_begin = clamp(footprint_begin, ivec2(0), params.screen_size - ivec2(1));
	footprint_end = clamp(footprint_end, footprint_begin + ivec2(1), params.screen_size);
	r_screen_position = clamp((footprint_begin + footprint_end - ivec2(1)) / 2, ivec2(0), params.screen_size - ivec2(1));
	r_depth = 0.0;
	r_normal = vec3(0.0, 0.0, 1.0);
	r_roughness = 1.0;
	r_dynamic = false;
	bool found_surface = false;
	for (int y = footprint_begin.y; y < footprint_end.y; y++) {
		for (int x = footprint_begin.x; x < footprint_end.x; x++) {
			ivec2 candidate_position = ivec2(x, y);
			float candidate_depth;
			vec3 candidate_normal;
			float candidate_roughness;
			bool candidate_dynamic;
			if (!specular_load_full_resolution_surface(candidate_position, candidate_depth, candidate_normal, candidate_roughness, candidate_dynamic)) {
				continue;
			}
			if (!found_surface || candidate_depth > r_depth) {
				found_surface = true;
				r_screen_position = candidate_position;
				r_depth = candidate_depth;
				r_normal = candidate_normal;
				r_roughness = candidate_roughness;
				r_dynamic = candidate_dynamic;
			}
		}
	}
	return found_surface;
}

bool specular_clip_to_uv(vec4 clip_position, out vec2 r_uv) {
	if (!(clip_position.w > 1e-6) || any(isnan(clip_position)) || any(isinf(clip_position))) {
		r_uv = vec2(0.0);
		return false;
	}
	r_uv = clip_position.xy / clip_position.w * 0.5 + 0.5;
	return !any(isnan(r_uv)) && !any(isinf(r_uv));
}

bool specular_sample_screen_radiance(vec3 endpoint_world, out vec3 r_radiance, out float r_confidence) {
	r_radiance = vec3(0.0);
	r_confidence = 0.0;
	if ((params.flags & SCREEN_PROBE_FLAG_SPECULAR_SCREEN_RADIANCE_VALID) == 0u || any(isnan(endpoint_world)) || any(isinf(endpoint_world))) {
		return false;
	}

	vec3 current_view = transpose(mat3(scene_data.cam_transform)) * (endpoint_world - scene_data.cam_transform[3].xyz);
	vec3 previous_view = (scene_data.previous_cam_inv_transform * vec4(endpoint_world, 1.0)).xyz;
	vec4 previous_clip = inverse(scene_data.previous_inv_projection[params.view_index]) * vec4(previous_view, 1.0);
	vec2 current_raster_uv;
	vec2 current_stable_uv;
	vec2 previous_raster_uv;
	vec2 previous_stable_uv;
	if (!specular_clip_to_uv(scene_data.projection[params.view_index] * vec4(current_view, 1.0), current_raster_uv) ||
			!specular_clip_to_uv(scene_data.temporal_projection[params.view_index] * vec4(current_view, 1.0), current_stable_uv) ||
			!specular_clip_to_uv(previous_clip, previous_raster_uv) ||
			!specular_clip_to_uv(scene_data.previous_temporal_projection[params.view_index] * vec4(previous_view, 1.0), previous_stable_uv)) {
		return false;
	}
	float expected_previous_depth = previous_clip.z / previous_clip.w;
	if (isnan(expected_previous_depth) || isinf(expected_previous_depth) || expected_previous_depth < 0.0 || expected_previous_depth > 1.0) {
		return false;
	}

	ivec2 history_size = textureSize(sampler2D(specular_screen_radiance_buffer, linear_sampler), 0);
	if (any(lessThanEqual(history_size, ivec2(0)))) {
		return false;
	}

	ivec2 endpoint_pixel = clamp(ivec2(floor(current_raster_uv * vec2(params.screen_size))), ivec2(0), params.screen_size - ivec2(1));
	float endpoint_roughness_dynamic = texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), endpoint_pixel, 0).a;
	bool dynamic_endpoint = endpoint_roughness_dynamic > 0.5;
	vec2 history_uv = previous_raster_uv;
	float object_motion_pixels = 0.0;
	if ((params.flags & SCREEN_PROBE_FLAG_SPECULAR_MOTION_VALID) != 0u) {
		vec2 endpoint_velocity = texelFetch(sampler2D(specular_velocity_buffer, linear_sampler), endpoint_pixel, 0).xy;
		if (!any(isnan(endpoint_velocity)) && !any(isinf(endpoint_velocity))) {
			vec2 current_jitter_uv = current_raster_uv - current_stable_uv;
			vec2 previous_jitter_uv = previous_raster_uv - previous_stable_uv;
			history_uv = current_raster_uv + endpoint_velocity + previous_jitter_uv - current_jitter_uv;
			object_motion_pixels = length((history_uv - previous_raster_uv) * vec2(history_size));
		}
	} else if (dynamic_endpoint) {
		return false;
	}

	vec2 half_texel = 0.5 / vec2(history_size);
	if (any(lessThan(history_uv, half_texel)) || any(greaterThan(history_uv, vec2(1.0) - half_texel))) {
		return false;
	}
	ivec2 previous_depth_size = textureSize(sampler2D(specular_previous_depth_buffer, linear_sampler), 0);
	if (any(notEqual(previous_depth_size, history_size))) {
		return false;
	}
	ivec2 previous_depth_position = clamp(ivec2(floor(history_uv * vec2(previous_depth_size))), ivec2(0), previous_depth_size - ivec2(1));
	float actual_previous_depth = texelFetch(sampler2D(specular_previous_depth_buffer, linear_sampler), previous_depth_position, 0).r;
	if (isnan(actual_previous_depth) || isinf(actual_previous_depth) || actual_previous_depth <= 0.0 || abs(actual_previous_depth - expected_previous_depth) >= 0.005) {
		return false;
	}

	vec3 screen_radiance = textureLod(sampler2D(specular_screen_radiance_buffer, linear_sampler), history_uv, 0.0).rgb;
	if (any(isnan(screen_radiance)) || any(isinf(screen_radiance)) || any(lessThan(screen_radiance, vec3(0.0)))) {
		return false;
	}
	float exposure_scale = clamp(params.specular_eye_offset_exposure.w, 1.0 / 16.0, 16.0);
	r_radiance = screen_radiance * exposure_scale;
	vec2 edge_texels = min(history_uv, vec2(1.0) - history_uv) * vec2(history_size);
	float edge_confidence = smoothstep(0.5, 2.0, min(edge_texels.x, edge_texels.y));
	float motion_confidence = mix(1.0, 0.2, smoothstep(1.0, 24.0, object_motion_pixels));
	if (dynamic_endpoint) {
		motion_confidence *= 0.9;
	}
	r_confidence = edge_confidence * motion_confidence;
	return r_confidence > 0.0;
}

vec4 specular_pack_normal_roughness(vec3 world_normal, float roughness, bool screen_radiance_used) {
	float maximum_component = max(abs(world_normal.x), max(abs(world_normal.y), abs(world_normal.z)));
	world_normal /= max(maximum_component, 1e-6);
	float roughness_bin = floor(clamp(roughness, 0.0, 1.0) * 127.0 + 0.5);
	float packed_roughness_source = (roughness_bin + (screen_radiance_used ? 128.0 : 0.0)) / 255.0;
	return vec4(world_normal * 0.5 + 0.5, packed_roughness_source);
}

void specular_store_guides(ivec2 output_position, ivec2 screen_position, bool valid_surface, bool dynamic_surface, vec3 view_position, vec3 view_normal, float roughness, bool screen_radiance_used) {
	float normal_length_squared = dot(view_normal, view_normal);
	valid_surface = valid_surface && !any(isnan(view_position)) && !any(isinf(view_position)) &&
			!any(isnan(view_normal)) && !any(isinf(view_normal)) && normal_length_squared > 1e-6;
	view_normal = valid_surface ? view_normal * inversesqrt(max(normal_length_squared, 1e-6)) : vec3(0.0, 0.0, 1.0);
	vec3 world_normal = normalize(mat3(scene_data.cam_transform) * view_normal);
	if (any(isnan(world_normal)) || any(isinf(world_normal))) {
		world_normal = vec3(0.0, 1.0, 0.0);
		valid_surface = false;
	}

	vec2 velocity = vec2(0.0);
	if (valid_surface && (params.flags & SCREEN_PROBE_FLAG_SPECULAR_MOTION_VALID) != 0u) {
		velocity = texelFetch(sampler2D(specular_velocity_buffer, linear_sampler), screen_position, 0).xy;
		if (any(isnan(velocity)) || any(isinf(velocity))) {
			velocity = vec2(0.0);
		}
	}
	vec2 resolve_uv = (vec2(output_position) + 0.5) / vec2(imageSize(specular_motion_output));
	vec2 surface_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec2 surface_uv_offset = valid_surface ? surface_uv - resolve_uv : vec2(0.0);
	if (valid_surface && dynamic_surface && (params.flags & SCREEN_PROBE_FLAG_SPECULAR_MOTION_VALID) != 0u) {
		surface_uv_offset.y += 2.0;
	}
	float view_z = valid_surface ? abs(view_position.z) : SPECULAR_INVALID_VIEW_Z;
	roughness = valid_surface ? roughness : 1.0;

	imageStore(specular_normal_roughness_output, output_position, specular_pack_normal_roughness(world_normal, roughness, screen_radiance_used));
	imageStore(specular_view_z_output, output_position, vec4(view_z, 0.0, 0.0, 0.0));
	imageStore(specular_motion_output, output_position, vec4(velocity, surface_uv_offset));
}

void specular_trace_main() {
	ivec2 output_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 output_size = imageSize(raw_radiance_output);
	if (any(greaterThanEqual(output_position, output_size))) {
		return;
	}

	ivec2 screen_position = ivec2(0);
	float origin_depth;
	vec3 origin_normal;
	float roughness;
	bool dynamic_surface;
	if (!specular_select_full_resolution_surface(output_position, output_size, screen_position, origin_depth, origin_normal, roughness, dynamic_surface)) {
		imageStore(raw_radiance_output, output_position, vec4(0.0));
		specular_store_guides(output_position, screen_position, false, false, vec3(0.0), vec3(0.0, 0.0, 1.0), 1.0, false);
		return;
	}

	vec2 origin_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec3 view_position = compute_view_position(vec3(origin_uv, origin_depth));
	vec3 eye_to_surface = view_position - params.specular_eye_offset_exposure.xyz;
	float eye_distance_squared = dot(eye_to_surface, eye_to_surface);
	vec3 view_direction = eye_distance_squared > 1e-10 ? -eye_to_surface * inversesqrt(eye_distance_squared) : vec3(0.0, 0.0, 1.0);
	bool valid_orientation = !any(isnan(view_direction)) && !any(isinf(view_direction)) && dot(origin_normal, view_direction) > 1e-5;
	if (!valid_orientation || roughness >= params.specular_tuning.y) {
		imageStore(raw_radiance_output, output_position, vec4(0.0));
		specular_store_guides(output_position, screen_position, valid_orientation, dynamic_surface, view_position, origin_normal, roughness, false);
		return;
	}

	vec3 view_local = world_to_tangent(view_direction, origin_normal);
	view_local.z = max(view_local.z, 1e-5);
	view_local = normalize(view_local);
	uint specular_stream = 0x53504543u ^ (params.view_index * 0x9e3779b9u);
	vec3 ray_local = vec3(0.0);
	bool valid_ray = false;
	if (roughness < 0.001) {
		ray_local = vec3(-view_local.xy, view_local.z);
		valid_ray = ray_local.z > 1e-5;
	} else {
		float alpha = max(roughness * roughness, params.specular_tuning.w);
		for (uint attempt = 0u; attempt < 8u; attempt++) {
			uint attempt_stream = specular_stream ^ (attempt * 0x85ebca6bu);
			vec2 random_sample = sample_r2_sequence(uvec2(output_position), params.frame_index, attempt_stream);
			random_sample.y *= 0.9;
			ray_local = sample_bounded_ggx_vndf_reflection(view_local, alpha, random_sample);
			valid_ray = ray_local.z > 1e-5 && !any(isnan(ray_local)) && !any(isinf(ray_local));
			if (valid_ray) {
				break;
			}
		}
	}
	if (!valid_ray) {
		imageStore(raw_radiance_output, output_position, vec4(0.0));
		specular_store_guides(output_position, screen_position, true, dynamic_surface, view_position, origin_normal, roughness, false);
		return;
	}
	vec3 ray_direction = tangent_to_world(ray_local, origin_normal);

	vec3 sample_radiance = vec3(0.0);
	float hit_distance = 0.0;
	vec3 endpoint_world = vec3(0.0);
	bool screen_radiance_used = false;
	if ((params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) != 0u &&
			(params.flags & SCREEN_PROBE_FLAG_SPECULAR_SCREEN_RADIANCE_VALID) != 0u) {
		vec3 endpoint_normal_world;
		if (trace_screen_detail(view_position, ray_direction, SCREEN_PROBE_SPECULAR_DETAIL_TRACE_MAX_DISTANCE, endpoint_world, endpoint_normal_world)) {
			float screen_confidence;
			if (specular_sample_screen_radiance(endpoint_world, sample_radiance, screen_confidence) && screen_confidence >= 0.75) {
				vec3 receiver_world = (scene_data.cam_transform * vec4(view_position, 1.0)).xyz;
				hit_distance = clamp(length(endpoint_world - receiver_world), 0.0, 65504.0);
				screen_radiance_used = true;
			}
		}
	}
	if (!screen_radiance_used) {
		int hit_cascade;
		vec3 endpoint_normal_world;
		bool trace_hit = trace_hddagi_sample(screen_position, origin_depth, origin_normal, ray_direction, 0.0,
				sample_radiance, hit_cascade, endpoint_world, endpoint_normal_world);
		if (trace_hit && hit_cascade >= 0 && !any(isnan(endpoint_world)) && !any(isinf(endpoint_world))) {
			vec3 receiver_world = (scene_data.cam_transform * vec4(view_position, 1.0)).xyz;
			hit_distance = clamp(length(endpoint_world - receiver_world), 0.0, 65504.0);
		}
		if (!trace_hit) {
			sample_radiance = sample_environment(ray_direction);
			hit_distance = 65504.0;
		}
	}
	vec3 traced_radiance = sample_radiance * (screen_radiance_used ? 1.0 : hddagi.energy);
	specular_store_guides(output_position, screen_position, true, dynamic_surface, view_position, origin_normal, roughness, screen_radiance_used);

	vec4 radiance_hit_distance = vec4(traced_radiance, hit_distance);
	if (any(isnan(radiance_hit_distance)) || any(isinf(radiance_hit_distance))) {
		radiance_hit_distance = vec4(0.0);
	} else {
		radiance_hit_distance.rgb = max(radiance_hit_distance.rgb, vec3(0.0));
		float maximum_radiance_component = max(radiance_hit_distance.r, max(radiance_hit_distance.g, radiance_hit_distance.b));
		if (maximum_radiance_component > params.specular_tuning.z) {
			radiance_hit_distance.rgb *= params.specular_tuning.z / maximum_radiance_component;
		}
		radiance_hit_distance.a = clamp(radiance_hit_distance.a, 0.0, 65504.0);
	}
	imageStore(raw_radiance_output, output_position, radiance_hit_distance);
}

#endif

void screen_probe_trace_main() {
	ivec2 probe_position = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(probe_position, imageSize(raw_radiance_output)))) {
		return;
	}
	screen_probe_debug_begin();

	ivec2 origin_position;
	float origin_depth;
	vec3 origin_normal;
	if (!load_probe_surface(probe_position, origin_position, origin_depth, origin_normal)) {
		imageStore(raw_radiance_output, probe_position, vec4(0.0));
		screen_probe_debug_store(probe_position);
		return;
	}
	screen_probe_debug_mark(SCREEN_PROBE_DEBUG_VALID);

	uint candidate_count = clamp(params.candidate_count, 1u, 8u);
	bool guided_sampling = (params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) != 0u;
	GuidedSamplingDistribution guided_distribution;
	if (guided_sampling) {
		guided_distribution = build_guided_sampling_distribution(origin_position, origin_depth, origin_normal);
	}
	vec3 radiance = vec3(0.0);
	float hit_distance = 0.0;
	for (uint candidate = 0u; candidate < candidate_count; candidate++) {
		vec2 sample_position = sample_r2_sequence(uvec2(probe_position), params.frame_index * candidate_count + candidate, 0x44544330u);
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
		screen_probe_debug_begin_source();
		if (!trace_hddagi_radiance(origin_position, origin_depth, origin_normal, ray_direction, candidate_radiance, candidate_hit_distance)) {
			candidate_radiance = sample_environment(ray_direction);
			candidate_hit_distance = 65504.0;
			screen_probe_debug_set_source(SCREEN_PROBE_DEBUG_SKY);
		}
		screen_probe_debug_commit_source();
		float cosine_pdf = max(dot(origin_normal, ray_direction), 0.0) / PI;
		radiance += candidate_radiance * hddagi.energy * (cosine_pdf / max(proposal_pdf, 1e-8));
		hit_distance += candidate_hit_distance;
	}
	radiance /= float(candidate_count);
	hit_distance /= float(candidate_count);
	if (any(isnan(radiance)) || any(isinf(radiance)) || isnan(hit_distance) || isinf(hit_distance)) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
		imageStore(raw_radiance_output, probe_position, vec4(0.0));
		screen_probe_debug_store(probe_position);
		return;
	}
#ifdef MODE_DEBUG_MONTAGE
	if (any(lessThan(radiance, vec3(0.0))) || hit_distance < 0.0) {
		screen_probe_debug_mark(SCREEN_PROBE_DEBUG_INVALID_OR_NONFINITE);
	}
#endif
	imageStore(raw_radiance_output, probe_position, clamp(vec4(radiance, hit_distance), vec4(0.0), vec4(65504.0)));
	screen_probe_debug_store(probe_position);
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
	bool center_has_current_trace = (params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) == 0u || imageLoad(directional_filter_trace_count_input, center_source_position).r > 0u;
	ivec2 base_probe_count = (params.gi_size + ivec2(max(params.probe_size, 1)) - ivec2(1)) / max(params.probe_size, 1);
	uint base_probe_total = uint(base_probe_count.x * base_probe_count.y);
	uint physical_index = uint(probe_position.y * surface_size.x + probe_position.x);
	bool adaptive_center = (params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_ADAPTIVE) != 0u && physical_index >= base_probe_total;
	ivec2 center_gi_position = clamp(center_screen_position * params.gi_size / params.screen_size, ivec2(0), params.gi_size - ivec2(1));
	ivec2 owner_probe = adaptive_center ? center_gi_position / max(params.probe_size, 1) : probe_position;
	ivec2 jitter_key = adaptive_center ? center_screen_position : probe_position;
	vec3 direction_world = directional_jittered_bucket_to_world(jitter_key, direction_texel);
	vec3 direction_view = normalize(transpose(mat3(scene_data.cam_transform)) * direction_world);

	vec3 radiance_sum = vec3(0.0);
	float radiance_weight_sum = 0.0;
	float hit_distance_sum = 0.0;
	float hit_distance_weight_sum = 0.0;
	// Packed adaptive probes borrow the owner base probe and its cardinal neighbors.
	for (int tap = 0; tap < 6; tap++) {
		if (!adaptive_center && tap == 5) {
			continue;
		}
		ivec2 neighbor_probe = probe_position;
		float kernel_weight = 1.0;
		bool neighbor_is_center = tap == 0;
		if (!neighbor_is_center) {
			ivec2 offset;
			if (adaptive_center) {
				if (tap == 1) {
					offset = ivec2(0);
				} else if (tap == 2) {
					offset = ivec2(-1, 0);
				} else if (tap == 3) {
					offset = ivec2(1, 0);
				} else if (tap == 4) {
					offset = ivec2(0, -1);
				} else {
					offset = ivec2(0, 1);
				}
			} else if (tap == 1) {
				offset = ivec2(-1, 0);
			} else if (tap == 2) {
				offset = ivec2(1, 0);
			} else if (tap == 3) {
				offset = ivec2(0, -1);
			} else {
				offset = ivec2(0, 1);
			}
			neighbor_probe = owner_probe + offset;
			if (any(lessThan(neighbor_probe, ivec2(0))) || any(greaterThanEqual(neighbor_probe, base_probe_count))) {
				continue;
			}
			kernel_weight = exp(-0.5 * float(dot(offset, offset)));
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
		float weight = receiver_plane_weight * kernel_weight;
		if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
			continue;
		}

		ivec2 source_position = neighbor_probe * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
		if ((params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) != 0u && imageLoad(directional_filter_trace_count_input, source_position).r == 0u) {
			continue;
		}
		vec4 source_value = texelFetch(sampler2D(directional_filter_source_input, directional_filter_nearest_sampler), source_position, 0);
		if (!directional_finite(source_value)) {
			continue;
		}
		source_value = clamp(source_value, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX));
		float endpoint_angle_weight = 1.0;
		if (!neighbor_is_center) {
			float neighbor_hit_distance = center_has_current_trace ? min(source_value.a, center_source_value.a) : source_value.a;
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
	ivec2 surface_size = imageSize(directional_irradiance_surface_input);
	if (any(greaterThanEqual(probe_position, probe_count)) || any(greaterThanEqual(probe_position, surface_size)) ||
			all(equal(imageLoad(directional_irradiance_surface_input, probe_position).xy, uvec2(0xffffffffu)))) {
		imageStore(directional_irradiance_output, atlas_position, vec4(0.0));
		return;
	}

	vec3 output_normal = directional_bucket_to_world(output_direction_texel);
	vec3 radiance_sum = vec3(0.0);
	float radiance_weighted_hit_distance_sum = 0.0;
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
			bool source_has_current_trace = (params.flags & SCREEN_PROBE_FLAG_GUIDED_SAMPLING) == 0u || imageLoad(directional_irradiance_trace_count_input, source_position).r > 0u;
			if (source_has_current_trace && source_value.a < SCREEN_PROBE_DIRECTIONAL_FP16_MAX) {
				float radiance_luminance = dot(source_value.rgb, vec3(0.2126, 0.7152, 0.0722));
				if (radiance_luminance > 1e-6) {
					float radiance_hit_weight = cosine_weight * radiance_luminance;
					radiance_weighted_hit_distance_sum += source_value.a * radiance_hit_weight;
					radiance_hit_distance_weight += radiance_hit_weight;
				}
				cosine_hit_distance_sum += source_value.a * cosine_weight;
				cosine_hit_distance_weight += cosine_weight;
			}
			has_valid_sample = has_valid_sample || source_has_current_trace;
		}
	}

	vec3 diffuse_irradiance_over_pi = radiance_sum * SCREEN_PROBE_DIRECTIONAL_IRRADIANCE_FACTOR;
	float representative_hit_distance = SCREEN_PROBE_DIRECTIONAL_FP16_MAX;
	if (has_valid_sample) {
		if (radiance_hit_distance_weight > 0.0) {
			representative_hit_distance = radiance_weighted_hit_distance_sum / radiance_hit_distance_weight;
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

void store_svgf_prepare_outputs(ivec2 resolve_position, ivec2 screen_position, bool valid_surface, bool dynamic_surface, float linear_depth, vec3 view_normal, float roughness, vec4 raw_signal) {
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
#ifdef MODE_DIRECTIONAL_RESOLVE
	if (valid_surface && dynamic_surface && (params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID) != 0u) {
		surface_uv_offset.y += 2.0;
	}
#endif

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

float directional_receiver_weight(ivec2 probe_position, vec3 pixel_view_position, vec3 pixel_normal, bool pixel_dynamic, out ivec2 r_probe_screen_position) {
	r_probe_screen_position = ivec2(0);
	float probe_depth;
	vec3 probe_normal;
	bool probe_dynamic;
	if (!load_resolve_probe_surface(probe_position, r_probe_screen_position, probe_depth, probe_normal, probe_dynamic) || probe_dynamic != pixel_dynamic) {
		return 0.0;
	}
	vec2 probe_uv = (vec2(r_probe_screen_position) + 0.5) / vec2(params.screen_size);
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
		store_svgf_prepare_outputs(resolve_position, screen_position, false, false, 0.0, vec3(0.0, 0.0, 1.0), 1.0, fallback_signal);
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
		store_svgf_prepare_outputs(resolve_position, screen_position, false, false, 0.0, pixel_normal, pixel_roughness, fallback_signal);
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
		store_svgf_prepare_outputs(resolve_position, screen_position, false, false, pixel_linear_depth, pixel_normal, pixel_roughness, fallback_signal);
#endif
		return;
	}
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(pixel_uv);
	ivec2 probe_base = ivec2(floor(probe_grid_position));
	vec2 probe_blend = fract(probe_grid_position);
	ivec2 physical_probe_count = textureSize(sampler2D(raw_radiance_input, nearest_sampler), 0) / SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	ivec2 probe_count = (params.gi_size + ivec2(max(params.probe_size, 1)) - ivec2(1)) / max(params.probe_size, 1);
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
			ivec2 selected_probe = probe_position;
			ivec2 selected_screen_position;
			float selected_receiver_weight = directional_receiver_weight(selected_probe, pixel_view_position, pixel_normal, pixel_dynamic, selected_screen_position);
			float selected_weight = bilinear_weight * selected_receiver_weight;
			if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_ADAPTIVE) != 0u && selected_weight < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
				uvec2 adaptive_tile_data = texelFetch(usampler2D(directional_adaptive_tile_data_input, nearest_sampler), probe_position, 0).rg;
				uint base_probe_total = uint(probe_count.x * probe_count.y);
				vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
				for (uint adaptive_offset = 0u; adaptive_offset < min(adaptive_tile_data.y, SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE); adaptive_offset++) {
					uint physical_index = base_probe_total + adaptive_tile_data.x + adaptive_offset;
					ivec2 adaptive_probe = ivec2(int(physical_index % uint(physical_probe_count.x)), int(physical_index / uint(physical_probe_count.x)));
					if (any(lessThan(adaptive_probe, ivec2(0))) || any(greaterThanEqual(adaptive_probe, physical_probe_count))) {
						continue;
					}
					ivec2 adaptive_screen_position;
					float adaptive_receiver_weight = directional_receiver_weight(adaptive_probe, pixel_view_position, pixel_normal, pixel_dynamic, adaptive_screen_position);
					vec2 axis_distance = abs(vec2(adaptive_screen_position - screen_position)) / probe_screen_extent;
					float adaptive_spatial_weight = 1.0 - clamp(min(axis_distance.x, axis_distance.y), 0.0, 1.0);
					float adaptive_weight = adaptive_receiver_weight * adaptive_spatial_weight;
					if (adaptive_weight > selected_weight) {
						selected_weight = adaptive_weight;
						selected_probe = adaptive_probe;
						selected_screen_position = adaptive_screen_position;
					}
				}
			}
			float weight = selected_weight;
			if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
				continue;
			}
			vec4 raw_radiance = directional_resolve_probe(selected_probe, directional_lookup);
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
	store_svgf_prepare_outputs(resolve_position, screen_position, true, pixel_dynamic, pixel_linear_depth, pixel_normal, pixel_roughness, resolved);
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

#ifdef MODE_SPECULAR_APPLY

void specular_apply_main() {
	ivec2 output_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 reflection_size = imageSize(reflection_output);
	if (any(greaterThanEqual(output_position, reflection_size))) {
		return;
	}

	ivec2 signal_size = textureSize(sampler2D(specular_radiance_input, specular_nearest_sampler), 0);
	ivec2 signal_position = clamp(output_position * signal_size / reflection_size, ivec2(0), signal_size - ivec2(1));
	float packed_roughness_source = floor(texelFetch(sampler2D(specular_normal_roughness_input, specular_nearest_sampler), signal_position, 0).a * 255.0 + 0.5);
	float roughness = (packed_roughness_source - floor(packed_roughness_source / 128.0) * 128.0) / 127.0;
	float authority = 1.0 - smoothstep(params.specular_tuning.x, params.specular_tuning.y, roughness);
	if (authority <= 0.0) {
		return;
	}

	vec3 traced_radiance = texelFetch(sampler2D(specular_radiance_input, specular_nearest_sampler), signal_position, 0).rgb;
	if (any(isnan(traced_radiance)) || any(isinf(traced_radiance))) {
		return;
	}
	vec3 fallback_radiance = rgbe_decode(imageLoad(reflection_output, output_position).r);
	imageStore(reflection_output, output_position, uvec4(rgbe_encode(mix(fallback_radiance, max(traced_radiance, vec3(0.0)), authority))));
}

#endif

void main() {
#ifdef MODE_SURFACE
	screen_probe_surface_main();
#elif defined(MODE_DIRECTIONAL_ADAPTIVE_MARK)
	directional_adaptive_mark_main();
#elif defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)
	directional_adaptive_spawn_main();
#elif defined(MODE_IRRADIANCE_CACHE_UPDATE_MULTIBOUNCE)
	irradiance_cache_update_multibounce_main();
#elif defined(MODE_DIRECTIONAL_TRACE)
	directional_trace_main();
#elif defined(MODE_SPECULAR_TRACE)
	specular_trace_main();
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
#elif defined(MODE_SPECULAR_APPLY)
	specular_apply_main();
#endif
}
