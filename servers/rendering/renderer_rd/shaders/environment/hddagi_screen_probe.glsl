#[compute]

#version 460

#VERSION_DEFINES

#if defined(MODE_DIRECTIONAL_TRACE)
#define HDDAGI_DATA_BINDING 4
#define HDDAGI_PAGE_BINDING 6
#elif defined(MODE_RESOLVE)
#define HDDAGI_DATA_BINDING 18
#define HDDAGI_PAGE_BINDING 19
#endif
#ifdef HDDAGI_DATA_BINDING
#include "hddagi_voxel_pages_inc.glsl"
#endif

#include "../oct_inc.glsl"
#include "hddagi_radiance_inc.glsl"
#include "hddagi_voxel_metadata_inc.glsl"

#if defined(MODE_DIRECTIONAL_ADAPTIVE_MARK) || defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)
#define MODE_ADAPTIVE_PLACEMENT
layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#endif

const uint SCREEN_PROBE_FLAG_DETAIL_TRACE = 1u << 0u;
const uint SCREEN_PROBE_FLAG_DIRECTIONAL_HISTORY_VALID = 1u << 3u;
const uint SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID = 1u << 5u;
const uint SCREEN_PROBE_DETAIL_TRACE_QUALITY_SHIFT = 7u;
const uint SCREEN_PROBE_DETAIL_TRACE_QUALITY_MASK = 3u << SCREEN_PROBE_DETAIL_TRACE_QUALITY_SHIFT;
const uint SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER = 1u << 9u;
const uint SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT = 8u;
const uint SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE = 8u;
const float SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE = 0.01;
const float SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_RETIRE_COVERAGE = 0.06;

const uint SCREEN_PROBE_SKY_COLOR = 1u;
const uint SCREEN_PROBE_SKY_TEXTURE = 2u;
const float SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE = 2.0;

const int HDDAGI_REGION_SIZE = 8;
const float TAU = 6.283185307179586;

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_RESOLVE)
const int SCREEN_PROBE_DIRECTIONAL_TILE_SIZE = int(gl_WorkGroupSize.x);
const int SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
const float SCREEN_PROBE_DIRECTIONAL_FP16_MAX = 65504.0;
const uint SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE = 8u;
const float SCREEN_PROBE_DIRECTIONAL_RADIANCE_MAX = 10.0;
#endif

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
}
params;

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
	vec4 taa_jitter_uv;
	mat4 previous_projection[2];
	mat4 previous_cam_transform;
	vec4 receiver_camera_offset; // Camera relative to the receiver history origin; w resets the history.
};

#ifdef MODE_SURFACE

layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 2) uniform sampler surface_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform restrict writeonly uimage2D screen_probe_surface_output;

#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_RESOLVE)
#ifdef MODE_RESOLVE
layout(set = 0, binding = 20, std140)
#else
layout(set = 0, binding = 7, std140)
#endif
		uniform HDDAGIData {
	ivec3 grid_size;
	int max_cascades;

	float normal_bias;
	float energy;
	float y_mult;
	float pad1;

	ivec3 probe_axis_size;
	uint pad;

	ivec4 pad3;

	ScreenProbeCascadeData cascades[8];
}
hddagi;
#endif

#if defined(MODE_DIRECTIONAL_TRACE)

#ifdef MODE_DIRECTIONAL_TRACE
layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D screen_probe_surface_input;
#endif
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D raw_radiance_output;
layout(rg32ui, set = 0, binding = 2) uniform restrict readonly uimage3D hddagi_voxel_cascades;
layout(r8ui, set = 0, binding = 3) uniform restrict readonly uimage3D hddagi_voxel_region_cascades;
layout(set = 0, binding = 5) uniform sampler linear_sampler;
layout(set = 0, binding = 9, std140) uniform SceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
layout(set = 0, binding = 10) uniform texture2D detail_hiz_buffer;
layout(set = 0, binding = 11) uniform texture2D detail_normal_roughness_buffer;
layout(set = 0, binding = 12) uniform texture2D directional_previous_radiance_input;
layout(rgba32ui, set = 0, binding = 13) uniform restrict readonly uimage2D directional_previous_surface_input;
layout(r8ui, set = 0, binding = 14) uniform restrict readonly uimage2D directional_previous_history_age_input;
layout(r8ui, set = 0, binding = 15) uniform restrict writeonly uimage2D directional_history_age_output;
#ifdef MODE_ADAPTIVE
layout(rg32ui, set = 0, binding = 16) uniform restrict readonly uimage2D directional_previous_adaptive_tile_data_input;
#endif
layout(set = 0, binding = 17) uniform texture2D velocity_buffer;
layout(set = 0, binding = 23) uniform texture2D directional_previous_lighting_response_input;
layout(r8, set = 0, binding = 24) uniform restrict writeonly image2D directional_lighting_response_output;
layout(set = 0, binding = 25) uniform texture2D directional_previous_lighting_statistics_input;
layout(rgba16f, set = 0, binding = 26) uniform restrict writeonly image2D directional_lighting_statistics_output;
layout(rg16f, set = 0, binding = 28) uniform restrict image2D receiver_luminance;
layout(rgba32ui, set = 0, binding = 29) uniform restrict uimage2D receiver_geometry;
#ifdef MODE_DIRECTIONAL_TRACE_IMPORTANCE
layout(set = 0, binding = 27) uniform texture2DArray directional_world_radiance;
#endif

#endif

#if defined(MODE_DIRECTIONAL_TRACE)
#ifdef USE_RADIANCE_OCTMAP_ARRAY
layout(set = 1, binding = 0) uniform texture2DArray sky_radiance;
#else
layout(set = 1, binding = 0) uniform texture2D sky_radiance;
#endif
layout(set = 1, binding = 1) uniform sampler sky_sampler;

#endif

#ifdef MODE_DIRECTIONAL_IRRADIANCE

layout(set = 0, binding = 0) uniform texture2D directional_irradiance_source_input;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D directional_irradiance_output;
layout(set = 0, binding = 2) uniform sampler directional_irradiance_nearest_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform restrict readonly uimage2D directional_irradiance_surface_input;
layout(set = 0, binding = 4) uniform texture2D directional_irradiance_velocity;
layout(r8ui, set = 0, binding = 5) uniform restrict readonly uimage2D directional_irradiance_history_age;
layout(set = 0, binding = 7, std140) uniform IrradianceSceneDataBuffer {
	ScreenProbeSceneData scene_data;
};

#endif

#ifdef MODE_RESOLVE

layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D screen_probe_surface_input;
layout(set = 0, binding = 1) uniform texture2D directional_irradiance_input;
layout(set = 0, binding = 2) uniform texture2D depth_buffer;
layout(set = 0, binding = 3) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 4) uniform sampler surface_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D resolved_radiance_output;
layout(set = 0, binding = 7, std140) uniform SceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
layout(set = 0, binding = 8) uniform texture2D directional_base_ambient_input;
#ifdef MODE_ADAPTIVE
layout(set = 0, binding = 13) uniform utexture2D directional_adaptive_tile_data_input;
#endif
layout(set = 0, binding = 14) uniform texture2D directional_resolve_lighting_response_input;
layout(rg16f, set = 0, binding = 15) uniform restrict readonly image2D receiver_luminance;
layout(rgba32ui, set = 0, binding = 16) uniform restrict readonly uimage2D receiver_geometry;
#ifdef MODE_SVGF_PREPARE
layout(rgba8, set = 0, binding = 6) uniform restrict writeonly image2D svgf_normal_roughness_output;
layout(r32f, set = 0, binding = 11) uniform restrict writeonly image2D svgf_view_z_output;
#endif

#endif

#ifdef MODE_APPLY

layout(set = 0, binding = 0) uniform texture2D resolved_radiance_input;
layout(set = 0, binding = 1) uniform sampler nearest_sampler;
layout(r32ui, set = 0, binding = 2) uniform restrict writeonly uimage2D ambient_output;
layout(set = 0, binding = 3) uniform texture2D base_ambient_input;
#endif

#ifdef MODE_ADAPTIVE_PLACEMENT

layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 2) uniform sampler surface_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform coherent uimage2D adaptive_probe_surface;
#ifdef MODE_DIRECTIONAL_ADAPTIVE_MARK
layout(r8ui, set = 0, binding = 4) uniform restrict writeonly uimage2D adaptive_candidate_mark_output;
#else
layout(r8ui, set = 0, binding = 4) uniform restrict readonly uimage2D adaptive_candidate_mark_input;
#endif
layout(r32ui, set = 0, binding = 6) uniform coherent uimage2D adaptive_probe_counter;
layout(set = 0, binding = 7, std140) uniform AdaptiveSceneDataBuffer {
	ScreenProbeSceneData scene_data;
};
#ifdef MODE_DIRECTIONAL_ADAPTIVE_SPAWN
layout(rg32ui, set = 0, binding = 5) uniform restrict writeonly uimage2D adaptive_tile_data_output;
layout(rgba32ui, set = 0, binding = 8) uniform restrict readonly uimage2D adaptive_previous_probe_surface;
layout(rg32ui, set = 0, binding = 9) uniform restrict readonly uimage2D adaptive_previous_tile_data_input;
#endif

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

const uvec2 R2_WEYL_INCREMENT = uvec2(3242174889u, 2447445413u);

float uint_to_unit_float(uint value) {
	return float(value >> 8u) * (1.0 / 16777216.0);
}

vec2 sample_temporally_permuted_r2_sequence(uvec2 position, uint sequence_index, uint stream) {
	uvec2 scramble = uvec2(
			hash_uvec3(uvec3(position, stream ^ 0xa511e9b3u)),
			hash_uvec3(uvec3(position, stream ^ 0x63d83595u)));
	uint temporal_permutation = hash_uvec3(uvec3(position, stream ^ 0xd1b54a35u));
	uint permuted_index = sequence_index ^ temporal_permutation;
	uvec2 sequence = scramble + uvec2(permuted_index) * R2_WEYL_INCREMENT;
	return vec2(uint_to_unit_float(sequence.x), uint_to_unit_float(sequence.y));
}

vec2 sample_hammersley_8(uint sequence_index) {
	uint index = (0x54e8f8u >> ((sequence_index & 7u) * 3u)) & 7u;
	uint reversed_index = ((index & 1u) << 2u) | (index & 2u) | ((index & 4u) >> 2u);
	return (vec2(index, reversed_index) + 0.5) * (1.0 / 8.0);
}

vec2 screen_probe_uniform_placement_offset(uint frame_index, uint flags) {
	return (flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u ? sample_hammersley_8(frame_index) : vec2(0.5);
}

bool directional_finite(vec3 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

bool directional_finite(vec4 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_RESOLVE)

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

vec2 directional_bucket_jitter(ivec2 probe_position, ivec2 bucket) {
	uvec2 sample_key = uvec2(probe_position) * 32u + uvec2(bucket);
	return sample_temporally_permuted_r2_sequence(sample_key, params.frame_index, 0xf68b5852u);
}

#ifndef MODE_MULTISAMPLE
vec3 directional_jittered_bucket_to_world(ivec2 probe_position, ivec2 direction_texel) {
	vec2 jitter = directional_bucket_jitter(probe_position, direction_texel);
	return directional_square_to_sphere((vec2(direction_texel) + jitter) / float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE));
}
#endif

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

#endif

#if defined(MODE_SURFACE) || defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_RESOLVE) || defined(MODE_ADAPTIVE_PLACEMENT)

float screen_probe_uniform_axis_anchor(int probe_index, int gi_extent, int screen_extent, int probe_extent, float sample_offset) {
	int safe_probe_extent = max(probe_extent, 1);
	int safe_gi_extent = max(gi_extent, 1);
	int atlas_extent = max((safe_gi_extent + safe_probe_extent - 1) / safe_probe_extent, 1);
	int clamped_index = clamp(probe_index, 0, atlas_extent - 1);
	int origin = clamped_index * safe_probe_extent;
	int end = min(origin + safe_probe_extent, safe_gi_extent);
	int safe_screen_extent = max(screen_extent, 1);
	int screen_begin;
	int screen_end;
	if (safe_screen_extent == safe_gi_extent || safe_screen_extent == safe_gi_extent * 2) {
		int scale = safe_screen_extent == safe_gi_extent ? 1 : 2;
		screen_begin = origin * scale;
		screen_end = end * scale;
	} else {
		screen_begin = (origin * safe_screen_extent + safe_gi_extent - 1) / safe_gi_extent;
		screen_end = (end * safe_screen_extent + safe_gi_extent - 1) / safe_gi_extent;
	}
	float target_twice = float(screen_begin + screen_end - 1);
	if ((params.flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u) {
		float integer_offset = floor(sample_offset * float(safe_probe_extent));
		float gi_sample = min(float(origin) + integer_offset + 0.5, float(safe_gi_extent) - 0.5);
		target_twice = round(2.0 * (gi_sample * float(safe_screen_extent) / float(safe_gi_extent) - 0.5));
	}
	// Match the lower-coordinate tie break in surface placement.
	float screen_sample = clamp(floor(target_twice * 0.5), float(screen_begin), float(max(screen_begin, screen_end - 1)));
	return (screen_sample + 0.5) * float(safe_gi_extent) / float(safe_screen_extent);
}

float directional_probe_axis_texel(float grid_uv, int gi_extent, int screen_extent, int probe_extent, float sample_offset) {
	int safe_probe_extent = max(probe_extent, 1);
	int atlas_extent = max((gi_extent + safe_probe_extent - 1) / safe_probe_extent, 1);
	float grid_pixel = grid_uv * float(gi_extent);
	float first_sample = screen_probe_uniform_axis_anchor(0, gi_extent, screen_extent, safe_probe_extent, sample_offset);
	if (atlas_extent <= 1) {
		return (grid_pixel - first_sample) / float(max(gi_extent, 1));
	}
	float previous_sample = screen_probe_uniform_axis_anchor(atlas_extent - 2, gi_extent, screen_extent, safe_probe_extent, sample_offset);
	float last_sample = screen_probe_uniform_axis_anchor(atlas_extent - 1, gi_extent, screen_extent, safe_probe_extent, sample_offset);
	if (grid_pixel >= previous_sample) {
		float last_spacing = max(last_sample - previous_sample, 1e-6);
		return float(atlas_extent - 2) + (grid_pixel - previous_sample) / last_spacing;
	}
	if (screen_extent != gi_extent && screen_extent != gi_extent * 2) {
		// Odd half-resolution sizes have uneven spacing after pixel snapping.
		int lower = clamp(int(floor((grid_pixel - first_sample) / float(safe_probe_extent))), 0, atlas_extent - 2);
		float lower_sample = screen_probe_uniform_axis_anchor(lower, gi_extent, screen_extent, safe_probe_extent, sample_offset);
		float upper_sample = screen_probe_uniform_axis_anchor(lower + 1, gi_extent, screen_extent, safe_probe_extent, sample_offset);
		if (grid_pixel < lower_sample && lower > 0) {
			lower--;
			upper_sample = lower_sample;
			lower_sample = screen_probe_uniform_axis_anchor(lower, gi_extent, screen_extent, safe_probe_extent, sample_offset);
		} else if (grid_pixel > upper_sample && lower < atlas_extent - 2) {
			lower++;
			lower_sample = upper_sample;
			upper_sample = screen_probe_uniform_axis_anchor(lower + 1, gi_extent, screen_extent, safe_probe_extent, sample_offset);
		}
		return float(lower) + (grid_pixel - lower_sample) / max(upper_sample - lower_sample, 1e-6);
	}
	return (grid_pixel - first_sample) / float(safe_probe_extent);
}

ivec2 directional_base_probe_count() {
	return (params.gi_size + ivec2(params.probe_size - 1)) / params.probe_size;
}

vec2 directional_grid_uv_to_probe_texel(vec2 grid_uv, uint frame_index) {
	vec2 sample_offset = screen_probe_uniform_placement_offset(frame_index, params.flags);
	return vec2(
			directional_probe_axis_texel(grid_uv.x, params.gi_size.x, params.screen_size.x, params.probe_size, sample_offset.x),
			directional_probe_axis_texel(grid_uv.y, params.gi_size.y, params.screen_size.y, params.probe_size, sample_offset.y));
}

vec2 directional_grid_uv_to_probe_texel(vec2 grid_uv) {
	return directional_grid_uv_to_probe_texel(grid_uv, params.frame_index);
}

vec2 directional_probe_nominal_grid_uv(ivec2 probe_position, uint frame_index) {
	vec2 sample_offset = screen_probe_uniform_placement_offset(frame_index, params.flags);
	return vec2(
			screen_probe_uniform_axis_anchor(probe_position.x, params.gi_size.x, params.screen_size.x, params.probe_size, sample_offset.x) / float(params.gi_size.x),
			screen_probe_uniform_axis_anchor(probe_position.y, params.gi_size.y, params.screen_size.y, params.probe_size, sample_offset.y) / float(params.gi_size.y));
}

#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_RESOLVE)

const vec3 DIRECTIONAL_LUMINANCE = vec3(0.2126, 0.7152, 0.0722);

vec3 directional_receiver_position(vec3 view_position) {
	return mat3(scene_data.cam_transform) * view_position + scene_data.receiver_camera_offset.xyz;
}

float directional_receiver_spacing(vec3 view_position) {
	float clip_w = scene_data.projection[params.view_index][3][3] == 1.0 ? 1.0 : abs(view_position.z);
	return 2.0 * abs(scene_data.inv_projection[params.view_index][1][1]) * clip_w * float(params.probe_size) / float(params.gi_size.y);
}

// Low four bits hold the visit count (zero invalidates the slot), the next twelve the frame.
uint directional_receiver_age(uint metadata) {
	return (params.frame_index - (metadata >> 4u)) & 0xfffu;
}

vec3 directional_receiver_normal(uint metadata) {
	return oct_to_vec3(unpackUnorm4x8(metadata).zw * 2.0 - 1.0);
}

#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_DIRECTIONAL_IRRADIANCE) || defined(MODE_RESOLVE) || defined(MODE_ADAPTIVE_PLACEMENT)
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

#if defined(MODE_DIRECTIONAL_TRACE)
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
#endif

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

#if defined(MODE_SURFACE) || defined(MODE_RESOLVE) || defined(MODE_ADAPTIVE_PLACEMENT)
bool load_surface(ivec2 screen_position, out float r_depth, out vec3 r_normal, out bool r_dynamic) {
	r_depth = texelFetch(sampler2D(depth_buffer, surface_sampler), screen_position, 0).r;
	if (!(r_depth > 0.0)) {
		return false;
	}
	vec4 normal_roughness = texelFetch(sampler2D(normal_roughness_buffer, surface_sampler), screen_position, 0);
	r_dynamic = normal_roughness.a > 0.5;
	return decode_normal(normal_roughness.xyz, r_normal);
}
#endif

#if defined(MODE_SURFACE) || defined(MODE_ADAPTIVE_PLACEMENT)
void screen_probe_tile_bounds(ivec2 tile, out ivec2 r_begin, out ivec2 r_end) {
	ivec2 gi_begin = tile * params.probe_size;
	ivec2 gi_end = min(gi_begin + ivec2(params.probe_size), params.gi_size);
	r_begin = (gi_begin * params.screen_size + params.gi_size - ivec2(1)) / params.gi_size;
	r_end = (gi_end * params.screen_size + params.gi_size - ivec2(1)) / params.gi_size;
}
#endif

#if defined(MODE_ADAPTIVE) || defined(MODE_ADAPTIVE_PLACEMENT)
ivec2 directional_adaptive_probe_position(ivec2 base_count, uint slot) {
	return ivec2(slot % uint(base_count.x), slot / uint(base_count.x)) + ivec2(0, base_count.y);
}
#endif

#ifdef MODE_ADAPTIVE_PLACEMENT
#include "hddagi_screen_probe_adaptive_inc.glsl"
#endif

#ifdef MODE_SURFACE

shared uint surface_best_key;

void screen_probe_surface_main() {
	ivec2 probe_position = ivec2(gl_WorkGroupID.xy);
	ivec2 screen_begin;
	ivec2 screen_end;
	screen_probe_tile_bounds(probe_position, screen_begin, screen_end);
	ivec2 screen_target_twice = screen_begin + screen_end - ivec2(1);
	if ((params.flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u) {
		vec2 target_screen_position = directional_probe_nominal_grid_uv(probe_position, params.frame_index) * vec2(params.screen_size) - 0.5;
		screen_target_twice = ivec2(round(target_screen_position * 2.0));
	}
	ivec2 extent = screen_end - screen_begin;
	uint area = uint(extent.x * extent.y);
	uint best_key = 0xffffffffu;
	uvec4 best_surface = uvec4(0xffffffffu);
	for (int y = int(gl_LocalInvocationID.y); y < extent.y; y += int(gl_WorkGroupSize.y)) {
		for (int x = int(gl_LocalInvocationID.x); x < extent.x; x += int(gl_WorkGroupSize.x)) {
			ivec2 screen_position = screen_begin + ivec2(x, y);
			ivec2 delta = screen_position * 2 - screen_target_twice;
			uint key = uint(dot(delta, delta)) * area + uint(y * extent.x + x);
			float depth;
			vec3 normal;
			bool dynamic_surface;
			if (key < best_key && load_surface(screen_position, depth, normal, dynamic_surface)) {
				best_key = key;
				best_surface = uvec4(uvec2(screen_position), floatBitsToUint(depth), pack_surface_normal(normal, dynamic_surface));
			}
		}
	}

	if (gl_LocalInvocationIndex == 0u) {
		surface_best_key = 0xffffffffu;
	}
	barrier();
	if (best_key != 0xffffffffu) {
		atomicMin(surface_best_key, best_key);
	}
	barrier();
	if (best_key == surface_best_key && (best_key != 0xffffffffu || gl_LocalInvocationIndex == 0u)) {
		imageStore(screen_probe_surface_output, probe_position, best_surface);
	}
}

#endif

#include "hddagi_screen_probe_trace_inc.glsl"

#ifdef MODE_DIRECTIONAL_TRACE
shared ivec2 directional_history_base;
shared vec4 directional_history_weights;
shared uint directional_history_age;
shared uint directional_history_valid;
shared float directional_lighting_response;
shared vec3 directional_lighting_variance[SCREEN_PROBE_DIRECTIONAL_TILE_SIZE];
shared vec4 directional_lighting_current[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared vec4 directional_lighting_previous[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];

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

bool directional_select_history(ivec2 probe_position, ivec2 current_screen_position, float current_depth, vec3 current_normal_view, bool current_dynamic, out ivec2 r_previous_base, out vec4 r_previous_weights, out uint r_previous_age) {
	r_previous_base = ivec2(-1);
	r_previous_weights = vec4(0.0);
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

	ivec2 base_probe_count = directional_base_probe_count();
	bool use_surface_uv = (params.flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u;
#ifdef MODE_ADAPTIVE
	bool adaptive_probe = probe_position.y >= base_probe_count.y;
	use_surface_uv = use_surface_uv || adaptive_probe;
#endif
	vec3 previous_receiver_view = (scene_data.previous_cam_inv_transform * vec4(current_world, 1.0)).xyz;
	vec4 previous_clip = scene_data.previous_projection[params.view_index] * vec4(previous_receiver_view, 1.0);
	if (!(previous_clip.w > 1e-6) || !directional_finite(previous_clip)) {
		return false;
	}
	if ((current_dynamic || use_surface_uv) && (previous_clip.z < 0.0 || previous_clip.z > previous_clip.w)) {
		return false;
	}
	vec2 previous_grid_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
	vec2 jitter_delta = scene_data.taa_jitter_uv.zw - scene_data.taa_jitter_uv.xy;
	if (current_dynamic) {
		if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID) == 0u) {
			return false;
		}
		vec2 motion = texelFetch(sampler2D(velocity_buffer, linear_sampler), current_screen_position, 0).xy;
		previous_grid_uv = current_uv + motion + jitter_delta;
	} else if (!use_surface_uv) {
		vec2 grid_uv = directional_probe_nominal_grid_uv(min(probe_position, base_probe_count - ivec2(1)), params.frame_index);
		previous_grid_uv += grid_uv - current_uv - jitter_delta;
	}
	if (any(lessThan(previous_grid_uv, vec2(0.0))) || any(greaterThanEqual(previous_grid_uv, vec2(1.0)))) {
		return false;
	}

	vec2 previous_probe_texel = directional_grid_uv_to_probe_texel(previous_grid_uv, params.frame_index - 1u);
	if (!directional_finite(vec3(previous_probe_texel, 0.0))) {
		return false;
	}
	ivec2 previous_surface_size = imageSize(directional_previous_surface_input);
	ivec2 previous_age_size = imageSize(directional_previous_history_age_input);
	ivec2 search_base = ivec2(floor(previous_probe_texel));
	vec2 search_fraction = fract(previous_probe_texel);
	mat4 previous_cam_transform = scene_data.previous_cam_transform;
	const float relative_plane_tolerance = 0.01823;
	float current_scene_depth = max(abs(current_view.z), 1e-3);
	float normal_threshold = max(params.history_normal_threshold, 0.8);
	vec2 previous_screen_position_expected = previous_grid_uv * vec2(params.screen_size) - 0.5;
	vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
#ifdef MODE_ADAPTIVE
	if (adaptive_probe) {
		float best_score = -1.0;
		const float adaptive_relative_plane_tolerance = 0.01;
		const float adaptive_relative_view_z_tolerance = 0.015;
		const float adaptive_screen_footprint_limit = 0.75;
		float adaptive_normal_threshold = max(params.history_normal_threshold, 0.9);
		uint best_age = 0u;
		uint best_slot = 0xffffffffu;

		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 2; x++) {
				ivec2 owner_tile = search_base + ivec2(x, y);
				if (any(lessThan(owner_tile, ivec2(0))) || any(greaterThanEqual(owner_tile, base_probe_count))) {
					continue;
				}
				uvec2 tile = imageLoad(directional_previous_adaptive_tile_data_input, owner_tile).rg;
				for (uint i = 0u; i < tile.y; i++) {
					uint slot = tile.x + i;
					ivec2 previous_probe = directional_adaptive_probe_position(base_probe_count, slot);
					uint candidate_age = imageLoad(directional_previous_history_age_input, previous_probe).r;
					if (candidate_age == 0u) {
						continue;
					}

					uvec4 previous_packed = imageLoad(directional_previous_surface_input, previous_probe);
					vec2 screen_axis_distance = abs(vec2(previous_packed.xy) - previous_screen_position_expected) / probe_screen_extent;
					float screen_distance = max(screen_axis_distance.x, screen_axis_distance.y);
					if (screen_distance > adaptive_screen_footprint_limit) {
						continue;
					}

					vec3 previous_world;
					vec3 previous_normal_world;
					bool previous_dynamic;
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
						if (isnan(view_z_distance) || isinf(view_z_distance) || view_z_distance > dynamic_view_z_tolerance || normal_similarity < adaptive_normal_threshold) {
							continue;
						}
						float view_z_score = 1.0 - clamp(view_z_distance / dynamic_view_z_tolerance, 0.0, 1.0);
						float normal_score = clamp((normal_similarity - adaptive_normal_threshold) / max(1.0 - adaptive_normal_threshold, 1e-4), 0.0, 1.0);
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
					bool index_wins = abs(score - best_score) <= 1e-8 && candidate_age == best_age && slot < best_slot;
					if (score > best_score + 1e-8 || age_wins || index_wins) {
						best_score = score;
						best_age = candidate_age;
						best_slot = slot;
						r_previous_base = previous_probe;
						r_previous_weights = vec4(1.0, 0.0, 0.0, 0.0);
						r_previous_age = min(candidate_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE);
					}
				}
			}
		}
		if (best_score >= 0.0) {
			return true;
		}
	}
#endif
	float history_weight_sum = 0.0;
	float history_age_sum = 0.0;

	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 candidate = search_base + ivec2(x, y);
			vec2 axis_weight = vec2(x == 0 ? 1.0 - search_fraction.x : search_fraction.x, y == 0 ? 1.0 - search_fraction.y : search_fraction.y);
			float footprint_weight = axis_weight.x * axis_weight.y;
			if (!(footprint_weight > 1e-8) || any(lessThan(candidate, ivec2(0))) || any(greaterThanEqual(candidate, base_probe_count)) || any(greaterThanEqual(candidate, previous_surface_size)) || any(greaterThanEqual(candidate, previous_age_size))) {
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
			if (!directional_decode_history_surface(previous_packed, previous_screen_position, previous_depth, previous_normal_view, previous_dynamic)) {
				continue;
			}
			vec3 previous_world;
			vec3 previous_normal_world;
			if (!directional_previous_surface_world(previous_packed, previous_cam_transform, previous_world, previous_normal_world, previous_dynamic)) {
				continue;
			}
			float normal_similarity = dot(current_normal_world, previous_normal_world);
			bool compatible = false;
			if (current_dynamic) {
				vec2 screen_axis_distance = abs(vec2(previous_screen_position) - previous_screen_position_expected) / probe_screen_extent;
				float screen_distance = max(screen_axis_distance.x, screen_axis_distance.y);
				vec3 candidate_previous_view = (scene_data.previous_cam_inv_transform * vec4(previous_world, 1.0)).xyz;
				float view_z_distance = abs(candidate_previous_view.z - previous_receiver_view.z);
				float dynamic_view_z_tolerance = 0.02 + 0.02 * max(abs(candidate_previous_view.z), abs(previous_receiver_view.z));
				const float dynamic_screen_footprint_limit = 1.0;
				float dynamic_normal_threshold = max(params.history_normal_threshold, 0.85);
				if (!isnan(screen_distance) && !isinf(screen_distance) && screen_distance <= dynamic_screen_footprint_limit && !isnan(view_z_distance) && !isinf(view_z_distance) && view_z_distance <= dynamic_view_z_tolerance && normal_similarity >= dynamic_normal_threshold) {
					compatible = true;
				}
			} else {
				vec3 receiver_delta = previous_world - current_world;
				float relative_plane_distance = abs(dot(receiver_delta, current_normal_world)) / current_scene_depth;
				vec2 screen_axis_distance = abs(vec2(previous_screen_position) - previous_screen_position_expected) / probe_screen_extent;
				float screen_distance = max(screen_axis_distance.x, screen_axis_distance.y);
				const float uniform_screen_footprint_limit = 1.0;
				bool screen_footprint_valid = !use_surface_uv || (!isnan(screen_distance) && !isinf(screen_distance) && screen_distance <= uniform_screen_footprint_limit);
				if (screen_footprint_valid && !isnan(relative_plane_distance) && !isinf(relative_plane_distance) && relative_plane_distance <= relative_plane_tolerance && normal_similarity >= normal_threshold) {
					compatible = true;
				}
			}
			if (compatible) {
				// Renormalize the bilinear footprint to avoid rounding subpixel motion on every frame.
				r_previous_weights[y * 2 + x] = footprint_weight;
				history_weight_sum += footprint_weight;
				history_age_sum += float(min(candidate_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE)) * footprint_weight;
			}
		}
	}
	if (history_weight_sum <= 1e-5) {
		return false;
	}
	r_previous_base = search_base;
	r_previous_weights /= history_weight_sum;
	float effective_age = history_age_sum / history_weight_sum * min(history_weight_sum, 1.0);
	r_previous_age = uint(clamp(round(effective_age), 1.0, float(SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE)));
	return true;
}

vec4 directional_trace_world_direction(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_world) {
	vec3 ray_direction_view = normalize(transpose(mat3(scene_data.cam_transform)) * ray_direction_world);
	vec3 incident_radiance;
	float unused_hit_distance;
	if (!trace_hddagi_radiance(origin_position, origin_depth, origin_normal, ray_direction_view, incident_radiance, unused_hit_distance)) {
		incident_radiance = sample_environment(ray_direction_view);
	}
	return vec4(directional_clamp_radiance(incident_radiance * hddagi.energy), 0.0);
}

#ifdef MODE_MULTISAMPLE
vec3 directional_sample_direction(ivec2 bucket, vec2 rotation, uint index, uint count) {
	vec2 offset = vec2(float(index) / float(count), float(bitfieldReverse(index)) * (1.0 / 4294967296.0));
	return directional_square_to_sphere((vec2(bucket) + fract(rotation + offset)) / float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE));
}

vec4 directional_trace_samples(ivec2 bucket, vec2 rotation, uint sample_count, ivec2 origin_position, float origin_depth, vec3 origin_normal) {
	vec3 radiance = vec3(0.0);
	for (uint i = 0u; i < sample_count; i++) {
		radiance += directional_trace_world_direction(origin_position, origin_depth, origin_normal, directional_sample_direction(bucket, rotation, i, sample_count)).rgb;
	}
	return vec4(radiance / float(sample_count), 0.0);
}
#endif

vec4 directional_apply_stable_history(vec4 current_sample, vec4 previous_sample, bool sample_history_valid) {
	if (sample_history_valid) {
		float history_weight = 1.0 / float(directional_history_age + 1u);
		float current_weight = mix(history_weight, max(history_weight, 0.5), directional_lighting_response);
		current_sample.rgb = mix(previous_sample.rgb, current_sample.rgb, current_weight);
	}
	return directional_sanitize_fp16(current_sample);
}

bool directional_load_previous_radiance(ivec2 direction_texel, out vec4 r_previous_sample) {
	r_previous_sample = vec4(0.0);
	if (directional_history_valid == 0u) {
		return false;
	}
	ivec2 previous_atlas_size = textureSize(sampler2D(directional_previous_radiance_input, linear_sampler), 0);
	float weight_sum = 0.0;
	for (int tap = 0; tap < 4; tap++) {
		float weight = directional_history_weights[tap];
		if (weight <= 0.0) {
			continue;
		}
		ivec2 previous_probe = directional_history_base + ivec2(tap & 1, tap >> 1);
		ivec2 previous_atlas_position = previous_probe * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + direction_texel;
		if (any(lessThan(previous_atlas_position, ivec2(0))) || any(greaterThanEqual(previous_atlas_position, previous_atlas_size))) {
			continue;
		}
		vec4 previous_sample = texelFetch(sampler2D(directional_previous_radiance_input, linear_sampler), previous_atlas_position, 0);
		if (directional_finite(previous_sample)) {
			r_previous_sample += clamp(previous_sample, vec4(0.0), vec4(SCREEN_PROBE_DIRECTIONAL_FP16_MAX)) * weight;
			weight_sum += weight;
		}
	}
	if (weight_sum <= 1e-5) {
		return false;
	}
	r_previous_sample /= weight_sum;
	return true;
}

bool directional_uses_receiver_history(ivec2 probe) {
#ifdef MODE_ADAPTIVE
	if (probe.y >= directional_base_probe_count().y) {
		return false;
	}
#endif
	return (params.flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u;
}

ivec2 directional_receiver_slot(ivec2 probe) {
	ivec2 subcell = ivec2(sample_hammersley_8(params.frame_index) * 4.0);
	return probe * ivec2(2, 4) + ivec2(subcell.x >> 1, subcell.y);
}

void directional_store_receiver(ivec2 probe, vec3 world_normal, vec2 luminance) {
	ivec2 slot = directional_receiver_slot(probe);
	uvec4 surface = imageLoad(screen_probe_surface_input, probe);
	if (surface_is_dynamic(surface.w)) {
		imageStore(receiver_geometry, slot, uvec4(0u));
		return;
	}
	vec2 uv = (vec2(surface.xy) + 0.5) / vec2(params.screen_size);
	vec3 view_position = compute_view_position(vec3(uv, uintBitsToFloat(surface.z)));
	vec3 view_normal = transpose(mat3(scene_data.cam_transform)) * world_normal;
	float spacing = directional_receiver_spacing(view_position);
	ivec2 probe_count = directional_base_probe_count();
	float contact = 0.0;
	for (int i = 0; i < 4; i++) {
		ivec2 axis = i < 2 ? ivec2(1, 0) : ivec2(0, 1);
		ivec2 neighbor = probe + ((i & 1) == 0 ? axis : -axis);
		if (any(lessThan(neighbor, ivec2(0))) || any(greaterThanEqual(neighbor, probe_count))) {
			continue;
		}
		uvec4 other = imageLoad(screen_probe_surface_input, neighbor);
		float depth = uintBitsToFloat(other.z);
		if (depth > 0.0 && !surface_is_dynamic(other.w)) {
			vec2 other_uv = (vec2(other.xy) + 0.5) / vec2(params.screen_size);
			vec3 delta = compute_view_position(vec3(other_uv, depth)) - view_position;
			float height = dot(delta, view_normal) / spacing;
			contact = max(contact, smoothstep(0.0, 0.25, height) * (1.0 - smoothstep(1.0, 2.0, height)));
		}
	}
	luminance.x = mix(luminance.y, luminance.x, contact);

	vec3 position = directional_receiver_position(view_position);
	float visits = 1.0;
	if (scene_data.receiver_camera_offset.w == 0.0) {
		uvec4 previous = imageLoad(receiver_geometry, slot);
		uint count = previous.w & 15u;
		if (count > 0u && directional_receiver_age(previous.w) == 8u &&
				distance(position, uintBitsToFloat(previous.xyz)) < 0.23 * spacing &&
				dot(world_normal, directional_receiver_normal(previous.w)) > 0.95) {
			visits = min(float(count) + 1.0, mix(14.0, 1.0, directional_lighting_response));
			luminance = mix(imageLoad(receiver_luminance, slot).rg, luminance, 1.0 / visits);
		}
	}
	uint metadata = packUnorm4x8(vec4(0.0, 0.0, vec3_to_oct(world_normal))) | ((params.frame_index & 0xfffu) << 4u) | uint(round(visits));
	imageStore(receiver_geometry, slot, uvec4(floatBitsToUint(position), metadata));
	imageStore(receiver_luminance, slot, vec4(luminance, 0.0, 0.0));
}

void directional_update_lighting_response(ivec2 probe_position, ivec2 direction_texel, vec3 normal_world, vec4 current_sample, vec4 previous_sample, bool sample_history_valid) {
	float cosine_weight = max(dot(normal_world, directional_bucket_to_world(direction_texel)), 0.0);
	directional_lighting_current[gl_LocalInvocationIndex] = vec4(current_sample.rgb, 1.0) * cosine_weight;
	directional_lighting_previous[gl_LocalInvocationIndex] = sample_history_valid ? vec4(previous_sample.rgb, 1.0) * cosine_weight : vec4(0.0);
	barrier();
	uint row = gl_LocalInvocationID.y;
	uint first = row * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
	if (gl_LocalInvocationID.x == 0u) {
		vec4 current = vec4(0.0);
		vec4 previous = vec4(0.0);
		vec3 variance = vec3(0.0);
		for (uint i = first; i < first + SCREEN_PROBE_DIRECTIONAL_TILE_SIZE; i++) {
			current += directional_lighting_current[i];
			previous += directional_lighting_previous[i];
			vec3 difference = directional_lighting_current[i].rgb - directional_lighting_previous[i].rgb;
			variance += difference * difference;
		}
		directional_lighting_current[first] = current;
		directional_lighting_previous[first] = previous;
		directional_lighting_variance[row] = variance;
	}
	barrier();
	if (gl_LocalInvocationIndex == 0u) {
		vec4 current_sum = vec4(0.0);
		vec4 previous_sum = vec4(0.0);
		vec3 directional_variance = vec3(0.0);
		for (uint i = 0u; i < SCREEN_PROBE_DIRECTIONAL_TILE_SIZE; i++) {
			current_sum += directional_lighting_current[i * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE];
			previous_sum += directional_lighting_previous[i * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE];
			directional_variance += directional_lighting_variance[i];
		}
		vec3 current_mean = current_sum.rgb / max(current_sum.w, 1e-6);
		vec4 statistics = vec4(current_mean, 0.0);
		if (directional_history_valid != 0u && current_sum.w > 0.0 && previous_sum.w > 0.0) {
			vec4 previous_statistics = vec4(0.0);
			for (int tap = 0; tap < 4; tap++) {
				if (directional_history_weights[tap] > 0.0) {
					ivec2 previous_probe = directional_history_base + ivec2(tap & 1, tap >> 1);
					previous_statistics += texelFetch(sampler2D(directional_previous_lighting_statistics_input, linear_sampler), previous_probe, 0) * directional_history_weights[tap];
				}
			}
			vec3 previous_mean = previous_sum.rgb / previous_sum.w;
			// Estimate temporal noise separately from variation across directions.
			statistics.rgb = mix(previous_statistics.rgb, current_mean, 0.5);
			vec3 variance_sample = (current_mean - previous_statistics.rgb) * (current_mean - statistics.rgb);
			statistics.a = mix(previous_statistics.a, max(variance_sample.r, max(variance_sample.g, variance_sample.b)), 0.5);
			vec3 signal_scale = max(statistics.rgb, previous_mean);
			float relative_threshold = 0.05 * max(signal_scale.r, max(signal_scale.g, signal_scale.b)) + 1e-4;
			// EMA weights 1/2 and 1/8 correspond to 3 and 15 independent samples.
			float noise_threshold = 2.5 * sqrt(max(statistics.a, 0.0) * (1.0 / 3.0 + 1.0 / 15.0));
			directional_variance /= current_sum.w * current_sum.w;
			float sample_variance = max(directional_variance.r, max(directional_variance.g, directional_variance.b));
			noise_threshold = max(noise_threshold, 2.0 * sqrt(sample_variance * (1.0 / 3.0 + 1.0 / 15.0)));
			vec3 difference = abs(statistics.rgb - previous_mean);
			float disagreement = max(difference.r, max(difference.g, difference.b)) / max(noise_threshold, relative_threshold);
			directional_lighting_response = max(directional_lighting_response, smoothstep(1.0, 2.0, disagreement));
		}
		imageStore(directional_lighting_statistics_output, probe_position, directional_sanitize_fp16(statistics));
		if (directional_uses_receiver_history(probe_position)) {
			vec3 broad_mean = previous_sum.w > 1e-6 ? previous_sum.rgb / previous_sum.w : current_mean;
			directional_store_receiver(probe_position, normal_world, vec2(dot(current_mean, DIRECTIONAL_LUMINANCE), dot(broad_mean, DIRECTIONAL_LUMINANCE)));
		}
		imageStore(directional_lighting_response_output, probe_position, vec4(directional_lighting_response));
	}
	barrier();
}
#ifdef MODE_DIRECTIONAL_TRACE_IMPORTANCE

const uint DIRECTIONAL_HISTORY_RAY_COUNT = 32u;
const uint DIRECTIONAL_GUIDE_HISTORY_AGE = 4u;
shared vec2 directional_importance_guide[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared uint directional_importance_end[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared uint directional_importance_bins[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared vec3 directional_importance_samples[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];
shared vec4 directional_importance_rows[SCREEN_PROBE_DIRECTIONAL_TILE_SIZE];
shared float directional_importance_average;
shared float directional_importance_scale;

vec2 directional_importance_random(uvec2 position, uint stream) {
	uvec2 value = uvec2(hash_uvec3(uvec3(position, params.frame_index ^ stream)),
			hash_uvec3(uvec3(position.yx, params.frame_index ^ stream ^ 0x63d83595u)));
	return vec2(uint_to_unit_float(value.x), uint_to_unit_float(value.y));
}

float directional_world_guide_luminance(ivec2 origin_position, float origin_depth, vec3 normal_world, vec3 direction) {
	vec2 uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
	vec3 position = mat3(scene_data.cam_transform) * compute_view_position(vec3(uv, origin_depth));
	position.y *= hddagi.y_mult;
	vec3 normal_grid = normalize(normal_world * vec3(1.0, hddagi.y_mult, 1.0));
	for (int cascade = 0; cascade < hddagi.max_cascades; cascade++) {
		vec3 cell = (position - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(hddagi.grid_size)))) {
			continue;
		}
		ivec3 probe = clamp(ivec3(floor(cell / float(HDDAGI_REGION_SIZE) + normal_grid * 0.5 + 0.5)), ivec3(0), hddagi.probe_axis_size - 1);
		ivec3 world_probe = probe + hddagi.cascades[cascade].region_world_offset;
		ivec3 wrapped = hddagi_wrap_probe(world_probe, hddagi.probe_axis_size);
		ivec2 tile = wrapped.xy + ivec2(0, wrapped.z * hddagi.probe_axis_size.y);
		vec2 texel = vec2(tile * (LIGHTPROBE_OCT_SIZE + 2) + 1) + vec3_to_oct(direction) * float(LIGHTPROBE_OCT_SIZE);
		vec2 atlas_size = vec2(textureSize(sampler2DArray(directional_world_radiance, linear_sampler), 0).xy);
		vec3 radiance = textureLod(sampler2DArray(directional_world_radiance, linear_sampler), vec3(texel / atlas_size, float(cascade)), 0.0).rgb;
		return dot(directional_clamp_radiance(radiance * hddagi.cascades[cascade].exposure_normalization * hddagi.energy), DIRECTIONAL_LUMINANCE);
	}
	return 0.0;
}

vec4 directional_trace_importance(ivec2 probe_position, ivec2 sample_key, ivec2 direction_texel,
		ivec2 origin_position, float origin_depth, vec3 origin_normal,
		vec4 previous_sample) {
	uint lane = gl_LocalInvocationIndex;
	// Stratify the rotations across the four probes used by the resolve.
	uvec2 block = uvec2(sample_key) >> 1u;
	uvec2 quadrant = uvec2(sample_key) & 1u;
	float lighting = dot(previous_sample.rgb, DIRECTIONAL_LUMINANCE);
	vec3 normal_world = normalize(mat3(scene_data.cam_transform) * origin_normal);
	vec3 direction_center = directional_bucket_to_world(direction_texel);
	bool young = directional_history_age < DIRECTIONAL_GUIDE_HISTORY_AGE;
	uint ray_count = young ? uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT) : DIRECTIONAL_HISTORY_RAY_COUNT;
#ifdef MODE_MULTISAMPLE
	// Cover every direction once, then distribute the remaining rays using the guide.
	ray_count = ray_count * params.candidate_count - uint(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT);
#endif
	if (young) {
		float world_luminance = directional_world_guide_luminance(origin_position, origin_depth, normal_world, direction_center);
		lighting = mix(world_luminance, lighting, float(directional_history_age) * 0.2);
	}
	float cosine = max(dot(normal_world, direction_center), 0.0);
	directional_importance_guide[lane] = vec2(lighting, cosine);
	barrier();

	// Scan cosine and luminance times cosine separately until the group mean is known.
	uint row = gl_LocalInvocationID.y;
	if (gl_LocalInvocationID.x == 0u) {
		vec4 sums = vec4(0.0);
		for (uint i = lane; i < lane + SCREEN_PROBE_DIRECTIONAL_TILE_SIZE; i++) {
			vec2 sample_guide = directional_importance_guide[i];
			sums.x += sample_guide.x;
			sums.y = max(sums.y, sample_guide.x);
			sums.zw += vec2(sample_guide.y, sample_guide.x * sample_guide.y);
			directional_importance_guide[i] = sums.zw;
		}
		directional_importance_rows[row] = sums;
	}
	barrier();
	float uniform_fraction = young ? 0.5 : 0.2;
	if (lane == 0u) {
		vec4 total = vec4(0.0);
		for (uint i = 0u; i < SCREEN_PROBE_DIRECTIONAL_TILE_SIZE; i++) {
			vec4 partial = directional_importance_rows[i];
			directional_importance_rows[i].zw = total.zw;
			total.x += partial.x;
			total.y = max(total.y, partial.y);
			total.zw += partial.zw;
		}
		directional_importance_average = total.x / float(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT);
		float weight_sum = mix(directional_importance_average * total.z, total.w, 0.8);
		bool high_contrast = total.y > (young ? 2.0 : 4.0) * directional_importance_average + 1e-4;
		bool use_guide = high_contrast && directional_lighting_response < 0.25 && !surface_is_dynamic(imageLoad(screen_probe_surface_input, probe_position).w);
		directional_importance_scale = use_guide && weight_sum > 1e-8 ? (1.0 - uniform_fraction) / weight_sum : 0.0;
	}
	barrier();
	if (directional_importance_scale > 0.0) {
		vec2 prefix = directional_importance_rows[row].zw + directional_importance_guide[lane];
		float uniform_pdf = uniform_fraction / float(SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT);
		float cumulative = float(lane + 1u) * uniform_pdf + mix(directional_importance_average * prefix.x, prefix.y, 0.8) * directional_importance_scale;
		uint stratum = 2u * (quadrant.x ^ quadrant.y) + quadrant.y;
		float shift = fract(directional_importance_random(block, 0x49209f47u).x + float(stratum) * 0.25);
		directional_importance_end[lane] = lane + 1u == SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT ? ray_count : min(uint(float(ray_count) * cumulative + shift), ray_count);
#ifndef MODE_MULTISAMPLE
		float pdf = uniform_pdf + mix(directional_importance_average, lighting, 0.8) * cosine * directional_importance_scale;
		directional_importance_guide[lane].x = min(float(ray_count) * pdf, 1.0);
#endif
	}
	barrier();
	if (directional_importance_scale == 0.0) {
#ifdef MODE_MULTISAMPLE
		return directional_trace_samples(direction_texel, directional_bucket_jitter(sample_key, direction_texel), params.candidate_count, origin_position, origin_depth, origin_normal);
#else
		return directional_trace_world_direction(origin_position, origin_depth, origin_normal, directional_jittered_bucket_to_world(sample_key, direction_texel));
#endif
	}
	uint first = lane > 0u ? directional_importance_end[lane - 1u] : 0u;
	uint end = directional_importance_end[lane];
	vec3 radiance = vec3(0.0);
	uint batch_count = 1u;
#ifdef MODE_MULTISAMPLE
	first += lane;
	end += lane + 1u;
	ray_count += SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT;
	batch_count = (ray_count + SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT - 1u) / SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT;
#endif
	for (uint batch_index = 0u; batch_index < batch_count; batch_index++) {
		uint batch = batch_index * SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT;
		uint batch_end = min(end, batch + SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT);
		for (uint i = max(first, batch); i < batch_end; i++) {
			directional_importance_bins[i - batch] = lane;
		}
		barrier();
		if (batch + lane < ray_count) {
			uint bin = directional_importance_bins[lane];
			uint begin_sample = bin > 0u ? directional_importance_end[bin - 1u] : 0u;
			uint count = directional_importance_end[bin] - begin_sample;
			ivec2 bucket = ivec2(bin & 7u, bin >> 3u);
#ifdef MODE_MULTISAMPLE
			uint rank = batch + lane - (begin_sample + bin);
			vec3 direction = directional_sample_direction(bucket, directional_bucket_jitter(sample_key, bucket), rank, count + 1u);
#else
			uint rank = lane - begin_sample;
			vec2 rotation = fract(directional_importance_random(block * 32u + uvec2(bucket), 0xb6e1389du) + vec2(quadrant) * 0.5);
			vec2 jitter = fract(vec2((float(rank) + 0.5) / float(count), float(bitfieldReverse(rank)) * (1.0 / 4294967296.0)) + rotation);
			vec3 direction = directional_square_to_sphere((vec2(bucket) + jitter) / float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE));
#endif
			directional_importance_samples[lane] = directional_trace_world_direction(origin_position, origin_depth, origin_normal, direction).rgb;
		}
		barrier();
		for (uint i = max(first, batch); i < batch_end; i++) {
			radiance += directional_importance_samples[i - batch];
		}
	}
#ifdef MODE_MULTISAMPLE
	return vec4(radiance / float(end - first), 0.0);
#else
	if (first == end) {
		return vec4(0.0);
	}
	// Systematic rounding visits a bin with probability min(1, N*p).
	float inclusion = directional_importance_guide[lane].x;
	return vec4(radiance / (float(end - first) * inclusion), 0.0);
#endif
}

#endif // MODE_DIRECTIONAL_TRACE_IMPORTANCE

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

	if (directional_uses_receiver_history(probe_position) && scene_data.receiver_camera_offset.w != 0.0 && gl_LocalInvocationIndex < 8u) {
		ivec2 slot = probe_position * ivec2(2, 4) + ivec2(gl_LocalInvocationIndex & 1u, gl_LocalInvocationIndex >> 1u);
		if (any(notEqual(slot, directional_receiver_slot(probe_position)))) {
			imageStore(receiver_geometry, slot, uvec4(0u));
		}
	}

	ivec2 origin_position;
	float origin_depth;
	vec3 origin_normal;
	if (!load_probe_surface(probe_position, origin_position, origin_depth, origin_normal)) {
		if (gl_LocalInvocationIndex == 0u) {
			imageStore(directional_history_age_output, probe_position, uvec4(0u));
			imageStore(directional_lighting_response_output, probe_position, vec4(0.0));
			imageStore(directional_lighting_statistics_output, probe_position, vec4(0.0));
			if (directional_uses_receiver_history(probe_position)) {
				imageStore(receiver_geometry, directional_receiver_slot(probe_position), uvec4(0u));
			}
		}
		imageStore(raw_radiance_output, atlas_position, vec4(0.0));
		return;
	}

	if (gl_LocalInvocationIndex == 0u) {
		directional_history_base = ivec2(-1);
		directional_history_weights = vec4(0.0);
		directional_history_age = 0u;
		directional_history_valid = 0u;
		directional_lighting_response = 0.0;
		uvec4 current_surface = imageLoad(screen_probe_surface_input, probe_position);
		uint previous_age;
		ivec2 previous_base;
		vec4 previous_weights;
		if (directional_select_history(probe_position, origin_position, origin_depth, origin_normal, surface_is_dynamic(current_surface.w), previous_base, previous_weights, previous_age)) {
			directional_history_base = previous_base;
			directional_history_weights = previous_weights;
			directional_history_age = min(previous_age, SCREEN_PROBE_DIRECTIONAL_HISTORY_MAX_AGE - 1u);
			directional_history_valid = 1u;
			float previous_response = 0.0;
			for (int tap = 0; tap < 4; tap++) {
				if (previous_weights[tap] > 0.0) {
					ivec2 previous_probe = previous_base + ivec2(tap & 1, tap >> 1);
					previous_response += texelFetch(sampler2D(directional_previous_lighting_response_input, linear_sampler), previous_probe, 0).r * previous_weights[tap];
				}
			}
			directional_lighting_response = max(previous_response * 0.75 - 1.0 / 255.0, 0.0);
		}
		uint current_age = directional_history_valid != 0u ? directional_history_age + 1u : 1u;
		imageStore(directional_history_age_output, probe_position, uvec4(current_age, 0u, 0u, 0u));
	}
	barrier();

	ivec2 sample_key = probe_position;
#ifdef MODE_ADAPTIVE
	if (probe_position.y >= directional_base_probe_count().y) {
		sample_key = origin_position;
	}
#endif
#ifdef MODE_DIRECTIONAL_TRACE_IMPORTANCE
	vec4 previous_sample = vec4(0.0);
	bool sample_history_valid = directional_load_previous_radiance(direction_texel, previous_sample);
	vec4 directional_sample = directional_trace_importance(probe_position, sample_key, direction_texel, origin_position, origin_depth, origin_normal, previous_sample);
#else
#ifdef MODE_MULTISAMPLE
	vec4 directional_sample = directional_trace_samples(direction_texel, directional_bucket_jitter(sample_key, direction_texel), params.candidate_count, origin_position, origin_depth, origin_normal);
#else
	vec3 direction = directional_jittered_bucket_to_world(sample_key, direction_texel);
	vec4 directional_sample = directional_trace_world_direction(origin_position, origin_depth, origin_normal, direction);
#endif
	vec4 previous_sample = vec4(0.0);
	bool sample_history_valid = directional_load_previous_radiance(direction_texel, previous_sample);
#endif
	directional_update_lighting_response(probe_position, direction_texel, normalize(mat3(scene_data.cam_transform) * origin_normal), directional_sample, previous_sample, sample_history_valid);
	directional_sample = directional_apply_stable_history(directional_sample, previous_sample, sample_history_valid);
	imageStore(raw_radiance_output, atlas_position, directional_sample);
}

#endif

#ifdef MODE_DIRECTIONAL_IRRADIANCE

shared vec3 directional_irradiance_directions[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT / 2];
shared vec3 directional_irradiance_sources[SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT];

void store_directional_irradiance(ivec2 probe, ivec2 texel, vec4 value) {
	const int last = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE - 1;
	ivec2 origin = probe * (SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + 2) + 1;
	imageStore(directional_irradiance_output, origin + texel, value);
	bvec2 edge = equal(min(texel, last - texel), ivec2(0));
	ivec2 border = mix(ivec2(-1), ivec2(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE), greaterThan(texel, ivec2(0)));
	if (edge.x) {
		imageStore(directional_irradiance_output, origin + ivec2(border.x, last - texel.y), value);
	}
	if (edge.y) {
		imageStore(directional_irradiance_output, origin + ivec2(last - texel.x, border.y), value);
	}
	if (all(edge)) {
		imageStore(directional_irradiance_output, origin + last - border, value);
	}
}

shared float directional_irradiance_neighbors[8];

ivec2 directional_neighbor_offset(int index) {
	int cell = index + int(index >= 4);
	ivec2 offset = ivec2(cell % 3, cell / 3) - 1;
	return (offset.x == 0 || offset.y == 0) ? 2 * offset : offset;
}

float directional_neighbor_weight(ivec2 probe, uvec4 surface, ivec2 offset) {
	ivec2 probe_count = (params.gi_size + params.probe_size - 1) / params.probe_size;
	ivec2 neighbor_probe = probe + offset;
	if (surface_is_dynamic(surface.w) || probe.y >= probe_count.y || any(lessThan(neighbor_probe, ivec2(0))) || any(greaterThanEqual(neighbor_probe, probe_count))) {
		return 0.0;
	}

	float age = float(imageLoad(directional_irradiance_history_age, probe).r);
	float reuse = 1.0 - smoothstep(1.0, 5.0, age);
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID) != 0u) {
		vec2 motion = texelFetch(sampler2D(directional_irradiance_velocity, directional_irradiance_nearest_sampler), ivec2(surface.xy), 0).xy;
		float travel_in_probes = length(motion * vec2(params.gi_size)) / float(params.probe_size);
		reuse = max(reuse, smoothstep(0.05, 0.55, travel_in_probes));
	}
	if (reuse == 0.0) {
		return 0.0;
	}

	uvec4 neighbor = imageLoad(directional_irradiance_surface_input, neighbor_probe);
	if (neighbor.x == 0xffffffffu || surface_is_dynamic(neighbor.w)) {
		return 0.0;
	}
	vec3 normal = unpack_surface_normal(surface.w);
	vec3 neighbor_normal = unpack_surface_normal(neighbor.w);
	vec3 position = compute_view_position(vec3((vec2(surface.xy) + 0.5) / vec2(params.screen_size), uintBitsToFloat(surface.z)));
	vec3 neighbor_position = compute_view_position(vec3((vec2(neighbor.xy) + 0.5) / vec2(params.screen_size), uintBitsToFloat(neighbor.z)));
	vec3 delta = neighbor_position - position;
	float separation = length(delta);
	float plane_distance = max(abs(dot(delta, normal)), abs(dot(delta, neighbor_normal)));
	float weight = smoothstep(0.8, 0.98, dot(normal, neighbor_normal)) * (1.0 - smoothstep(0.02, 0.12, plane_distance / max(separation, 1e-5)));
	return reuse * weight;
}

void directional_irradiance_main() {
	ivec2 probe_position = ivec2(gl_WorkGroupID.xy);
	ivec2 output_direction_texel = ivec2(gl_LocalInvocationID.xy);
	uvec4 surface = imageLoad(directional_irradiance_surface_input, probe_position);
	if (all(equal(surface.xy, uvec2(0xffffffffu)))) {
		store_directional_irradiance(probe_position, output_direction_texel, vec4(0.0));
		return;
	}

	vec3 output_normal = directional_bucket_to_world(output_direction_texel);
	const int half_size = SCREEN_PROBE_DIRECTIONAL_TILE_SIZE / 2;
	const int pairs = SCREEN_PROBE_DIRECTIONAL_SAMPLE_COUNT / 2;
	int lane = int(gl_LocalInvocationIndex);
	vec4 lane_source = texelFetch(sampler2D(directional_irradiance_source_input, directional_irradiance_nearest_sampler), ivec2(gl_GlobalInvocationID.xy), 0);
	if (lane < 8) {
		directional_irradiance_neighbors[lane] = directional_neighbor_weight(probe_position, surface, directional_neighbor_offset(lane));
	}
	barrier();
	vec3 radiance_sum_nearby = vec3(0.0);
	float neighbor_weight_sum = 0.0;
	for (int i = 0; i < 8; i++) {
		float weight = directional_irradiance_neighbors[i];
		if (weight > 0.0) {
			ivec2 texel = ivec2(gl_GlobalInvocationID.xy) + directional_neighbor_offset(i) * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE;
			radiance_sum_nearby += weight * texelFetch(sampler2D(directional_irradiance_source_input, directional_irradiance_nearest_sampler), texel, 0).rgb;
			neighbor_weight_sum += weight;
		}
	}
	// Partial neighborhoods straddle silhouettes and contact shadows.
	vec3 filtered = (lane_source.rgb + radiance_sum_nearby) / (1.0 + neighbor_weight_sum);
	lane_source.rgb = mix(lane_source.rgb, filtered, smoothstep(5.0, 8.0, neighbor_weight_sum));
	if (output_direction_texel.y < half_size) {
		directional_irradiance_directions[lane] = output_normal;
	} else {
		// Pair opposite bins of the equal-area map. Only one cosine in each pair is positive.
		ivec2 opposite = output_direction_texel.x < half_size ? ivec2(half_size * 3 - 1, half_size - 1) - output_direction_texel.yx : output_direction_texel.yx - half_size;
		lane = pairs + opposite.y * SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + opposite.x;
	}
	directional_irradiance_sources[lane] = directional_sanitize_fp16(lane_source).rgb;
	barrier();

	vec3 radiance_sum = vec3(0.0);
	float cosine_weight_sum = 0.0;
	for (int i = 0; i < pairs; i++) {
		float cosine = dot(output_normal, directional_irradiance_directions[i]);
		float weight = abs(cosine);
		cosine_weight_sum += weight;
		radiance_sum += directional_irradiance_sources[cosine < 0.0 ? i + pairs : i] * weight;
	}

	// Normalize the discrete cosine kernel so constant radiance is independent of the normal.
	vec3 diffuse_irradiance_over_pi = radiance_sum / max(cosine_weight_sum, 1e-6);
	store_directional_irradiance(probe_position, output_direction_texel, directional_sanitize_fp16(vec4(diffuse_irradiance_over_pi, SCREEN_PROBE_DIRECTIONAL_FP16_MAX)));
}

#endif

#ifdef MODE_RESOLVE

#ifdef MODE_SVGF_PREPARE

vec4 pack_svgf_normal_response(vec3 world_normal, float lighting_response) {
	float maximum_component = max(abs(world_normal.x), max(abs(world_normal.y), abs(world_normal.z)));
	world_normal /= max(maximum_component, 1e-6);
	return vec4(world_normal * 0.5 + 0.5, lighting_response);
}

void store_svgf_prepare_outputs(ivec2 resolve_position, bool valid_surface, bool dynamic_surface, float linear_depth, vec3 view_normal, float lighting_response) {
	float view_z = valid_surface ? abs(linear_depth) : 0.0;

	float normal_length_squared = dot(view_normal, view_normal);
	valid_surface = valid_surface && directional_finite(view_normal) && normal_length_squared > 1e-6;
	view_normal = valid_surface ? view_normal * inversesqrt(max(normal_length_squared, 1e-6)) : vec3(0.0, 0.0, 1.0);
	vec3 world_normal = normalize(mat3(scene_data.cam_transform) * view_normal);
	if (!directional_finite(world_normal)) {
		world_normal = vec3(0.0, 1.0, 0.0);
	}
	// Diffuse input alpha carries lighting response; history alpha stores the frame count.
	float lighting_response_alpha = valid_surface ? lighting_response : 0.0;

	imageStore(svgf_normal_roughness_output, resolve_position, pack_svgf_normal_response(world_normal, lighting_response_alpha));
	// The signed depth carries the surface class alongside native-resolution motion vectors.
	imageStore(svgf_view_z_output, resolve_position, vec4(dynamic_surface ? -view_z : view_z));
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

float directional_receiver_weight(ivec2 probe_position, vec3 pixel_view_position, vec3 pixel_normal, bool pixel_dynamic, out ivec2 probe_screen_position) {
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

float directional_detail_coverage(ivec2 pixel, vec3 position, vec3 normal, float pixel_size, vec2 pitch, float radius) {
	vec2 texel = 1.0 / vec2(params.screen_size);
	float coverage = 0.0;
	for (int axis = 0; axis < 2; axis++) {
		float near_surface = 1.0;
		float elevation = 0.0;
		for (int side = -1; side <= 1; side += 2) {
			ivec2 offset = ivec2(0);
			offset[axis] = side * max(1, int(round(0.375 * pitch[axis])));
			ivec2 tap = clamp(pixel + offset, ivec2(0), params.screen_size - 1);
			float depth = texelFetch(sampler2D(depth_buffer, surface_sampler), tap, 0).r;
			if (depth <= 0.0) {
				near_surface = 0.0;
				break;
			}
			vec3 delta = compute_view_position(vec3((vec2(tap) + 0.5) * texel, depth)) - position;
			float distance = length(delta);
			near_surface = min(near_surface, 1.0 - smoothstep(radius, 2.0 * radius, distance));
			elevation = max(elevation, (dot(delta, normal) - pixel_size) / max(distance, pixel_size));
		}
		coverage = max(coverage, near_surface * clamp(2.0 * elevation, 0.0, 1.0));
	}
	return coverage;
}

vec3 directional_resolve_detail(ivec2 pixel, vec3 position, vec3 normal, vec3 irradiance) {
	float pixel_size = 2.0 * abs(scene_data.inv_projection[params.view_index][1][1]) / float(params.screen_size.y);
	if (scene_data.projection[params.view_index][3][3] == 0.0) {
		pixel_size *= -position.z;
	}
	vec2 pitch = float(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size);
	float radius = min(2.0 * pixel_size * pitch.y, SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE);
	float coverage = directional_detail_coverage(pixel, position, normal, pixel_size, pitch, radius);
	if (coverage <= 0.0) {
		return irradiance;
	}
	uint quality = (params.flags & SCREEN_PROBE_DETAIL_TRACE_QUALITY_MASK) >> SCREEN_PROBE_DETAIL_TRACE_QUALITY_SHIFT;
	int ray_count = quality == 0u ? 2 : 1;
	int steps = quality == 2u ? 3 : 5;
	vec3 tangent = normalize(cross(normal, abs(normal.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0)));
	vec3 bitangent = cross(normal, tangent);
	vec2 phase = sample_temporally_permuted_r2_sequence(uvec2(pixel), params.frame_index, 0u);
	vec3 sum = vec3(0.0);
	for (int i = 0; i < ray_count; i++) {
		vec2 u = fract(phase + vec2(float(i) * 0.5));
		float phi = TAU * u.y;
		vec3 dir = sqrt(u.x) * (cos(phi) * tangent + sin(phi) * bitangent) + sqrt(1.0 - u.x) * normal;
		vec3 value = irradiance;
		for (int j = 0; j < steps; j++) {
			float travel = radius * exp2(float(j - 4));
			vec3 point = position + dir * travel;
			vec4 clip = scene_data.projection[params.view_index] * vec4(point, 1.0);
			if (clip.w <= 0.0) {
				break;
			}
			vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
			if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) {
				break;
			}
			ivec2 tap = ivec2(uv * vec2(params.screen_size));
			float z = texelFetch(sampler2D(depth_buffer, surface_sampler), tap, 0).r;
			if (z <= 0.0) {
				continue;
			}
			vec3 surface = compute_view_position(vec3((vec2(tap) + 0.5) / vec2(params.screen_size), z));
			if (abs(point.z - surface.z) > max(1.5 * pixel_size, travel * 0.3)) {
				continue;
			}
			vec3 hit_normal;
			if (!decode_normal(texelFetch(sampler2D(normal_roughness_buffer, surface_sampler), tap, 0).xyz, hit_normal)) {
				continue;
			}
			float separation = dot(surface - position, hit_normal);
			float cosine = dot(hit_normal, dir);
			if (abs(separation) < pixel_size * 0.75 || abs(cosine) < 0.05) {
				continue;
			}
			float hit_t = separation / cosine;
			if (hit_t <= pixel_size || abs(hit_t - travel) > travel * 0.35 + pixel_size) {
				continue;
			}
			vec3 incoming;
			if (query_endpoint_radiance(mat3(scene_data.cam_transform) * surface, mat3(scene_data.cam_transform) * faceforward(hit_normal, dir, hit_normal), incoming)) {
				value = min(incoming * hddagi.energy, vec3(SCREEN_PROBE_DIRECTIONAL_RADIANCE_MAX));
				break;
			}
		}
		sum += value;
	}
	return mix(irradiance, sum / float(ray_count), coverage);
}

vec3 directional_base_ambient(ivec2 resolve_position, ivec2 resolve_size) {
	vec2 resolve_uv = (vec2(resolve_position) + 0.5) / vec2(resolve_size);
	vec3 base_ambient = textureLod(sampler2D(directional_base_ambient_input, surface_sampler), resolve_uv, 0.0).rgb;
	return directional_finite(base_ambient) ? max(base_ambient, vec3(0.0)) : vec3(0.0);
}

vec3 directional_receiver_edge(vec2 uv, vec3 view_position, vec3 view_normal, float spacing, out vec2 screen_tangent, out float footprint) {
	ivec2 center = ivec2(uv * vec2(params.screen_size));
	ivec2 stride = max(ivec2(1), ivec2(round(0.5 * float(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size))));
	vec3 other_normal = view_normal;
	ivec2 other_pixel = center;
	float least_alignment = 1.0;
	footprint = 1.0;
	for (int axis = 0; axis < 2; axis++) {
		ivec2 offset = axis == 0 ? ivec2(stride.x, 0) : ivec2(0, stride.y);
		float best_alignment = -1.0;
		for (int side = 0; side < 2; side++) {
			ivec2 pixel = clamp(center + (side == 0 ? offset : -offset), ivec2(0), params.screen_size - 1);
			vec3 normal;
			if (!decode_normal(texelFetch(sampler2D(normal_roughness_buffer, surface_sampler), pixel, 0).xyz, normal)) {
				continue;
			}
			float alignment = dot(normal, view_normal);
			best_alignment = max(best_alignment, alignment);
			if (alignment < least_alignment) {
				least_alignment = alignment;
				other_normal = normal;
				other_pixel = pixel;
			}
		}
		// A strip narrower than the receiver lattice cannot support a reliable contrast estimate.
		footprint *= smoothstep(0.88, 0.985, best_alignment);
	}
	screen_tangent = vec2(0.0);
	vec3 tangent = cross(view_normal, other_normal);
	float length_squared = dot(tangent, tangent);
	if (footprint <= 0.0 || least_alignment >= 0.88 || length_squared <= 0.04) {
		return vec3(0.0);
	}
	float depth = texelFetch(sampler2D(depth_buffer, surface_sampler), other_pixel, 0).r;
	vec2 other_uv = (vec2(other_pixel) + 0.5) / vec2(params.screen_size);
	if (depth <= 0.0) {
		return vec3(0.0);
	}
	vec3 other_position = compute_view_position(vec3(other_uv, depth));
	if (abs(dot(other_position - view_position, other_normal)) > 1.5 * spacing) {
		return vec3(0.0);
	}
	tangent *= inversesqrt(length_squared);
	vec4 clip = scene_data.projection[params.view_index] * vec4(tangent, 0.0);
	vec2 projected = (clip.xy - (uv * 2.0 - 1.0) * clip.w) * vec2(params.gi_size);
	float projected_length_squared = dot(projected, projected);
	if (projected_length_squared <= 1e-6) {
		return vec3(0.0);
	}
	screen_tangent = projected * inversesqrt(projected_length_squared);
	return mat3(scene_data.cam_transform) * cross(view_normal, tangent);
}

float directional_receiver_detail(vec2 uv, vec3 view_position, vec3 view_normal, vec3 world_normal, float live_luminance, float lighting_response) {
	vec2 subcell = uv * vec2(params.gi_size) * (4.0 / float(params.probe_size)) - 0.5;
	vec2 lattice = vec2(subcell.x + subcell.y, subcell.x - subcell.y) * 0.5;
	vec3 position = directional_receiver_position(view_position);
	float spacing = directional_receiver_spacing(view_position);
	float inverse_spacing_squared = 1.0 / max(spacing * spacing, 1e-12);
	vec2 screen_tangent;
	float footprint;
	vec3 across_edge = directional_receiver_edge(uv, view_position, view_normal, spacing, screen_tangent, footprint);
	if (footprint <= 0.0) {
		return 1.0;
	}
	bool near_edge = any(notEqual(screen_tangent, vec2(0.0)));
	vec2 lattice_axis = 2.0 * vec2(screen_tangent.x + screen_tangent.y, screen_tangent.x - screen_tangent.y);
#ifdef MODE_SVGF_PREPARE
	float phase = (float((params.frame_index * 7u) & 15u) + 0.5) / 16.0;
	lattice += (phase * 2.0 - 1.0) * 1.75 * lattice_axis;
	const int neighborhood_count = 1;
#else
	int neighborhood_count = near_edge ? 3 : 1;
#endif
	vec2 luminance_sum = vec2(0.0);
	float support = 0.0;
	float weight_sum = 0.0;
	for (int neighborhood = 0; neighborhood < neighborhood_count; neighborhood++) {
		ivec2 lower = ivec2(floor(lattice + float(neighborhood - neighborhood_count / 2) * lattice_axis));
		for (int i = 0; i < 4; i++) {
			ivec2 corner = lower + ivec2(i & 1, i >> 1);
			ivec2 slot = ivec2((corner.x + corner.y) >> 1, corner.x - corner.y);
			if (any(lessThan(slot, ivec2(0))) || any(greaterThanEqual(slot, imageSize(receiver_geometry)))) {
				continue;
			}
			uvec4 sample_geometry = imageLoad(receiver_geometry, slot);
			uint count = sample_geometry.w & 15u;
			if (count == 0u || directional_receiver_age(sample_geometry.w) >= 8u) {
				continue;
			}
			vec3 delta = uintBitsToFloat(sample_geometry.xyz) - position;
			float alignment = dot(world_normal, directional_receiver_normal(sample_geometry.w));
			if (abs(dot(delta, world_normal)) > 0.1 * spacing || alignment < 0.8) {
				continue;
			}
			float kernel_distance = 18.0 * dot(delta, delta);
			if (near_edge) {
				float perpendicular = dot(delta, across_edge);
				kernel_distance = 0.32 * dot(delta, delta) + 75.0 * perpendicular * perpendicular;
			}
			float weight = exp2(-kernel_distance * inverse_spacing_squared) * smoothstep(0.8, 0.97, alignment);
			support += weight;
			weight *= smoothstep(1.0, 4.0, float(count));
			luminance_sum += imageLoad(receiver_luminance, slot).rg * weight;
			weight_sum += weight;
		}
	}
	if (weight_sum <= 1e-5) {
		return 1.0;
	}

	float floor_luminance = 1e-4 * weight_sum;
	float live = live_luminance * weight_sum + floor_luminance;
	float broad = luminance_sum.y + floor_luminance;
	float relight = smoothstep(0.2, 0.8, abs(log2(live / broad)));
	float detail = clamp((luminance_sum.x + floor_luminance) * mix(1.0 / live, 1.0 / broad, relight), 0.0, 4.0);
	float confidence = smoothstep(0.025, 0.35, support) * (weight_sum / support) * (1.0 - lighting_response);
	return mix(1.0, detail, 0.82 * confidence * footprint);
}

void screen_probe_resolve_main() {
	ivec2 resolve_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 resolve_size = imageSize(resolved_radiance_output);
	if (any(greaterThanEqual(resolve_position, resolve_size))) {
		return;
	}

	ivec2 screen_position = resolve_position;
	float pixel_depth;
	vec3 pixel_normal;
	bool pixel_dynamic;
	if (!load_surface(screen_position, pixel_depth, pixel_normal, pixel_dynamic)) {
		vec4 fallback_signal = vec4(0.0);
		fallback_signal.rgb = directional_base_ambient(resolve_position, resolve_size);
		imageStore(resolved_radiance_output, resolve_position, fallback_signal);
#ifdef MODE_SVGF_PREPARE
		store_svgf_prepare_outputs(resolve_position, false, false, 0.0, vec3(0.0, 0.0, 1.0), 0.0);
#endif
		return;
	}
	vec2 pixel_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec3 pixel_view_position = compute_view_position(vec3(pixel_uv, pixel_depth));
	float pixel_linear_depth = pixel_view_position.z;
	if (any(isnan(pixel_view_position)) || any(isinf(pixel_view_position))) {
		vec4 fallback_signal = vec4(0.0);
		fallback_signal.rgb = directional_base_ambient(resolve_position, resolve_size);
		imageStore(resolved_radiance_output, resolve_position, fallback_signal);
#ifdef MODE_SVGF_PREPARE
		store_svgf_prepare_outputs(resolve_position, false, false, 0.0, pixel_normal, 0.0);
#endif
		return;
	}

	vec3 pixel_world_normal = normalize(mat3(scene_data.cam_transform) * pixel_normal);
	if (!directional_finite(pixel_world_normal)) {
		vec4 fallback_signal = vec4(directional_base_ambient(resolve_position, resolve_size), 0.0);
		imageStore(resolved_radiance_output, resolve_position, fallback_signal);
#ifdef MODE_SVGF_PREPARE
		store_svgf_prepare_outputs(resolve_position, false, false, pixel_linear_depth, pixel_normal, 0.0);
#endif
		return;
	}
	vec2 direction_texel = directional_sphere_to_square(pixel_world_normal) * float(SCREEN_PROBE_DIRECTIONAL_TILE_SIZE) + 1.0;
	vec2 inv_atlas_size = 1.0 / vec2(textureSize(sampler2D(directional_irradiance_input, surface_sampler), 0));
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(pixel_uv);
	ivec2 probe_base = ivec2(floor(probe_grid_position));
	vec2 probe_blend = fract(probe_grid_position);
	ivec2 probe_count = directional_base_probe_count();
	vec3 radiance_sum = vec3(0.0);
	float bilinear_weight_sum = 0.0;
	float lighting_response_sum = 0.0;
	float weight_sum = 0.0;

	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			ivec2 probe_position = probe_base + ivec2(x, y);
			if (any(lessThan(probe_position, ivec2(0))) || any(greaterThanEqual(probe_position, probe_count))) {
				continue;
			}
			float bilinear_weight = (x == 0 ? 1.0 - probe_blend.x : probe_blend.x) * (y == 0 ? 1.0 - probe_blend.y : probe_blend.y);
			bilinear_weight_sum += bilinear_weight;
			ivec2 probe_screen_position;
			float weight = bilinear_weight * directional_receiver_weight(probe_position, pixel_view_position, pixel_normal, pixel_dynamic, probe_screen_position);
#ifdef MODE_ADAPTIVE
			if (weight < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
				uvec2 tile = texelFetch(usampler2D(directional_adaptive_tile_data_input, surface_sampler), probe_position, 0).rg;
				vec2 extent = vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size);
				for (uint i = 0u; i < tile.y; i++) {
					ivec2 extra_probe = directional_adaptive_probe_position(probe_count, tile.x + i);
					float receiver_weight = directional_receiver_weight(extra_probe, pixel_view_position, pixel_normal, pixel_dynamic, probe_screen_position);
					if (receiver_weight <= 0.0) {
						continue;
					}
					vec2 distance_in_tiles = abs(vec2(probe_screen_position - screen_position)) / extent;
					float extra_weight = receiver_weight * (1.0 - clamp(min(distance_in_tiles.x, distance_in_tiles.y), 0.0, 1.0));
					if (extra_weight > weight) {
						probe_position = extra_probe;
						weight = extra_weight;
					}
				}
			}
#endif
			if (!(weight > 0.0) || isnan(weight) || isinf(weight)) {
				continue;
			}
			vec2 texel = vec2(probe_position * (SCREEN_PROBE_DIRECTIONAL_TILE_SIZE + 2)) + direction_texel;
			radiance_sum += textureLod(sampler2D(directional_irradiance_input, surface_sampler), texel * inv_atlas_size, 0.0).rgb * weight;
			lighting_response_sum += texelFetch(sampler2D(directional_resolve_lighting_response_input, surface_sampler), probe_position, 0).r * weight;

			weight_sum += weight;
		}
	}

	vec4 resolved = vec4(0.0, 0.0, 0.0, SCREEN_PROBE_DIRECTIONAL_FP16_MAX);
	float coverage = 0.0;
	if (weight_sum > 0.0) {
		resolved.rgb = radiance_sum / weight_sum;
		float confidence = clamp(weight_sum / max(bilinear_weight_sum, 1e-4), 0.0, 1.0);
		coverage = smoothstep(0.05, 0.55, confidence);
	}
	if (any(isnan(resolved)) || any(isinf(resolved))) {
		resolved = vec4(0.0, 0.0, 0.0, SCREEN_PROBE_DIRECTIONAL_FP16_MAX);
		coverage = 0.0;
	} else {
		resolved = clamp(resolved, vec4(0.0), vec4(65504.0));
	}

	resolved.rgb = mix(directional_base_ambient(resolve_position, resolve_size), resolved.rgb, clamp(coverage, 0.0, 1.0));

	float lighting_response = lighting_response_sum / max(weight_sum, 1e-6) * coverage;
	if (!pixel_dynamic && (params.flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u) {
		resolved.rgb *= directional_receiver_detail(pixel_uv, pixel_view_position, pixel_normal, pixel_world_normal, dot(resolved.rgb, DIRECTIONAL_LUMINANCE), lighting_response);
	}
	if ((params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) != 0u) {
		resolved.rgb = directional_resolve_detail(screen_position, pixel_view_position, pixel_normal, resolved.rgb);
	}
	imageStore(resolved_radiance_output, resolve_position, resolved);
#ifdef MODE_SVGF_PREPARE
	store_svgf_prepare_outputs(resolve_position, true, pixel_dynamic, pixel_linear_depth, pixel_normal, lighting_response);
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

	vec4 resolved_radiance = texelFetch(sampler2D(resolved_radiance_input, nearest_sampler), output_position, 0);
#ifdef MODE_SVGF_APPLY
	const float radiance_scale = 512.0;
#else
	const float radiance_scale = 1.0;
#endif
	bool valid_radiance = !any(isnan(resolved_radiance)) && !any(isinf(resolved_radiance));
	vec3 diffuse_irradiance_over_pi = valid_radiance ? max(resolved_radiance.rgb, vec3(0.0)) * radiance_scale : vec3(0.0);

	if (!valid_radiance) {
		vec2 output_uv = (vec2(output_position) + 0.5) / vec2(output_size);
		diffuse_irradiance_over_pi = textureLod(sampler2D(base_ambient_input, nearest_sampler), output_uv, 0.0).rgb;
		if (any(isnan(diffuse_irradiance_over_pi)) || any(isinf(diffuse_irradiance_over_pi))) {
			diffuse_irradiance_over_pi = vec3(0.0);
		}
	}

	imageStore(ambient_output, output_position, uvec4(rgbe_encode(diffuse_irradiance_over_pi)));
}

#endif

void main() {
#ifdef MODE_SURFACE
	screen_probe_surface_main();
#elif defined(MODE_DIRECTIONAL_ADAPTIVE_MARK)
	directional_adaptive_mark_main();
#elif defined(MODE_DIRECTIONAL_ADAPTIVE_SPAWN)
	directional_adaptive_spawn_main();
#elif defined(MODE_DIRECTIONAL_TRACE)
	directional_trace_main();
#elif defined(MODE_DIRECTIONAL_IRRADIANCE)
	directional_irradiance_main();
#elif defined(MODE_RESOLVE)
	screen_probe_resolve_main();
#elif defined(MODE_APPLY)
	screen_probe_apply_main();
#endif
}
