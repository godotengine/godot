#[compute]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const uint SCREEN_PROBE_FLAG_DETAIL_TRACE = 1u << 0u;
const uint SCREEN_PROBE_FLAG_GUIDED_SAMPLING = 1u << 1u;
const uint SCREEN_PROBE_SKY_COLOR = 1u;
const uint SCREEN_PROBE_SKY_TEXTURE = 2u;
const float SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE = 2.0;

const int HDDAGI_REGION_SIZE = 8;
const int HDDAGI_HDDA_FP_BITS = 10;
const uint HDDAGI_LIGHT_CELL_VALID_BIT = 1u << 26u;
const uint HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK = HDDAGI_LIGHT_CELL_VALID_BIT - 1u;
const float TAU = 6.283185307179586;

layout(push_constant, std430) uniform Params {
	ivec2 gi_size;
	ivec2 screen_size;

	int probe_size;
	uint view_index;
	uint frame_index;
	uint flags;

	float normal_bias;
	uint candidate_count;
	uint sky_mode;
	float sky_energy;
	uint detail_trace_mip_count;
	uint padding[3];

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
};

#ifdef MODE_SURFACE

layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 2) uniform sampler nearest_sampler;
layout(rgba32ui, set = 0, binding = 3) uniform restrict writeonly uimage2D screen_probe_surface_output;

#endif

#ifdef MODE_TRACE

layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D screen_probe_surface_input;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D raw_radiance_output;
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
layout(set = 0, binding = 10) uniform texture2D detail_hiz_buffer;
layout(set = 0, binding = 11) uniform texture2D detail_normal_roughness_buffer;

#ifdef USE_RADIANCE_OCTMAP_ARRAY
layout(set = 1, binding = 0) uniform texture2DArray sky_radiance;
#else
layout(set = 1, binding = 0) uniform texture2D sky_radiance;
#endif
layout(set = 1, binding = 1) uniform sampler sky_sampler;
layout(set = 1, binding = 2) uniform texture2DArray hddagi_lightprobe_specular;

#endif

#ifdef MODE_RESOLVE

layout(rgba32ui, set = 0, binding = 0) uniform restrict readonly uimage2D screen_probe_surface_input;
layout(set = 0, binding = 1) uniform texture2D raw_radiance_input;
layout(set = 0, binding = 2) uniform texture2D depth_buffer;
layout(set = 0, binding = 3) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D resolved_radiance_output;
layout(set = 0, binding = 6, std140) uniform SceneDataBuffer {
	ScreenProbeSceneData scene_data;
};

#endif

#ifdef MODE_APPLY

layout(set = 0, binding = 0) uniform texture2D resolved_radiance_input;
layout(set = 0, binding = 1) uniform sampler nearest_sampler;
layout(r32ui, set = 0, binding = 2) uniform restrict writeonly uimage2D ambient_output;

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

#if defined(MODE_TRACE) || defined(MODE_RESOLVE)

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

uint pack_surface_normal(vec3 normal) {
	float length_squared = dot(normal, normal);
	if (!(length_squared > 1e-10) || any(isnan(normal)) || any(isinf(normal))) {
		normal = vec3(0.0, 0.0, 1.0);
	} else {
		normal *= inversesqrt(length_squared);
	}
	vec2 octahedral = clamp(vec3_to_oct(normal), vec2(0.0), vec2(1.0));
	uvec2 packed = uvec2(roundEven(octahedral * 65535.0));
	return packed.x | (packed.y << 16u);
}

vec3 unpack_surface_normal(uint packed) {
	vec2 octahedral = vec2(float(packed & 0xffffu), float(packed >> 16u)) / 65535.0;
	return oct_to_vec3(octahedral * 2.0 - 1.0);
}

#ifdef MODE_SURFACE

ivec2 gi_to_screen(ivec2 gi_position) {
	return clamp(gi_position * params.screen_size / params.gi_size, ivec2(0), params.screen_size - ivec2(1));
}

bool load_surface(ivec2 screen_position, out float r_depth, out vec3 r_normal) {
	if (any(lessThan(screen_position, ivec2(0))) || any(greaterThanEqual(screen_position, params.screen_size))) {
		return false;
	}
	r_depth = texelFetch(sampler2D(depth_buffer, nearest_sampler), screen_position, 0).r;
	if (!(r_depth > 0.0)) {
		return false;
	}
	return decode_normal(texelFetch(sampler2D(normal_roughness_buffer, nearest_sampler), screen_position, 0).xyz, r_normal);
}

void screen_probe_surface_main() {
	ivec2 probe_position = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(probe_position, imageSize(screen_probe_surface_output)))) {
		return;
	}

	ivec2 gi_begin = probe_position * params.probe_size;
	ivec2 gi_end = min(gi_begin + ivec2(params.probe_size), params.gi_size);
	ivec2 tile_extent = gi_end - gi_begin;
	ivec2 tile_center_twice = tile_extent - ivec2(1);
	ivec2 best_screen_position = ivec2(0);
	ivec2 best_gi_position = ivec2(0);
	float best_depth = 0.0;
	vec3 best_normal = vec3(0.0);
	uint best_distance_squared = 0xffffffffu;
	bool found = false;

	for (int y = 0; y < tile_extent.y; y++) {
		for (int x = 0; x < tile_extent.x; x++) {
			ivec2 gi_position = gi_begin + ivec2(x, y);
			ivec2 screen_position = gi_to_screen(gi_position);
			float depth;
			vec3 normal;
			if (!load_surface(screen_position, depth, normal)) {
				continue;
			}
			ivec2 center_delta_twice = ivec2(x, y) * 2 - tile_center_twice;
			uint distance_squared = uint(center_delta_twice.x * center_delta_twice.x + center_delta_twice.y * center_delta_twice.y);
			bool wins_tie = distance_squared == best_distance_squared && (gi_position.y < best_gi_position.y || (gi_position.y == best_gi_position.y && gi_position.x < best_gi_position.x));
			if (!found || distance_squared < best_distance_squared || wins_tie) {
				found = true;
				best_distance_squared = distance_squared;
				best_gi_position = gi_position;
				best_screen_position = screen_position;
				best_depth = depth;
				best_normal = normal;
			}
		}
	}

	if (!found) {
		imageStore(screen_probe_surface_output, probe_position, uvec4(0xffffffffu));
		return;
	}
	imageStore(screen_probe_surface_output, probe_position, uvec4(uvec2(best_screen_position), floatBitsToUint(best_depth), pack_surface_normal(best_normal)));
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

#ifdef MODE_TRACE

bool load_probe_surface(ivec2 probe_position, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal) {
	uvec4 packed = imageLoad(screen_probe_surface_input, probe_position);
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	return r_depth > 0.0 && all(greaterThanEqual(r_screen_position, ivec2(0))) && all(lessThan(r_screen_position, params.screen_size));
}

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
			block = imageLoad(hddagi_voxel_cascades, region_base + block_local).rg;
			if (block != uvec2(0)) {
				level = level_voxel;
				limits[level_voxel] = position - (position & level_masks[level_block]) + step * (level_masks[level_block] + ivec3(1));
				continue;
			}
		} else if (level == level_region) {
			ivec3 region = position >> fp_region_bits;
			region = (hddagi.cascades[cascade].region_world_offset + region) & region_offset_mask;
			region += cascade_base;
			if (imageLoad(hddagi_voxel_region_cascades, region).r > 0u) {
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
	r_metadata = imageLoad(hddagi_voxel_neighbours, texture_position).r;
	if (!bool(r_metadata & HDDAGI_LIGHT_CELL_VALID_BIT)) {
		return false;
	}

	uint disocclusion = imageLoad(hddagi_voxel_disocclusion, texture_position).r & 0x3fu;
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

bool query_endpoint_radiance(vec3 endpoint_world, vec3 endpoint_normal_world, out vec3 r_radiance) {
	r_radiance = vec3(0.0);
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

		r_radiance = radiance_sum / weight_sum * hddagi.cascades[cascade].exposure_normalization;
		if (!any(isnan(r_radiance)) && !any(isinf(r_radiance)) && !any(lessThan(r_radiance, vec3(0.0)))) {
			return true;
		}
	}
	return false;
}

vec3 sample_environment(vec3 ray_direction_view) {
	if (params.sky_mode == SCREEN_PROBE_SKY_COLOR) {
		return max(params.sky_color.rgb * params.sky_energy, vec3(0.0));
	}
	if (params.sky_mode != SCREEN_PROBE_SKY_TEXTURE) {
		return vec3(0.0);
	}

	vec3 sky_direction = scene_data.radiance_inverse_xform * ray_direction_view;
	float direction_length_squared = dot(sky_direction, sky_direction);
	if (!(direction_length_squared > 1e-8) || any(isnan(sky_direction)) || any(isinf(sky_direction))) {
		return vec3(0.0);
	}
	sky_direction *= inversesqrt(direction_length_squared);
	float border = clamp(params.sky_color.w, 0.0, 0.499);
	vec2 sky_uv = vec3_to_oct_with_border(sky_direction, vec2(border, 1.0 - border * 2.0));
#ifdef USE_RADIANCE_OCTMAP_ARRAY
	vec3 radiance = textureLod(sampler2DArray(sky_radiance, sky_sampler), vec3(sky_uv, 0.0), 0.0).rgb * params.sky_energy;
#else
	vec3 radiance = textureLod(sampler2D(sky_radiance, sky_sampler), sky_uv, 0.0).rgb * params.sky_energy;
#endif
	if (any(isnan(radiance)) || any(isinf(radiance))) {
		return vec3(0.0);
	}
	return max(radiance, vec3(0.0));
}

bool trace_hddagi_radiance(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_view, out vec3 r_radiance) {
	r_radiance = vec3(0.0);
	vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
	vec3 origin_view = compute_view_position(vec3(origin_uv, origin_depth));
	ray_direction_view = normalize(ray_direction_view);
	if (any(isnan(origin_view)) || any(isinf(origin_view)) || any(isnan(ray_direction_view)) || any(isinf(ray_direction_view))) {
		return false;
	}

	vec3 endpoint_world;
	vec3 endpoint_normal_world;
	if (trace_screen_detail(origin_view, ray_direction_view, SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE, endpoint_world, endpoint_normal_world)) {
		if (!query_endpoint_radiance(endpoint_world, endpoint_normal_world, r_radiance)) {
			r_radiance = sample_environment(ray_direction_view);
		}
		return true;
	}

	mat3 camera_basis = mat3(scene_data.cam_transform);
	vec3 ray_position = camera_basis * origin_view;
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
	bool exclude_receiver_cell = (params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) != 0u && params.normal_bias >= 0.0 && dot(ray_direction_view, origin_normal) > 1e-4;
	if (!trace_ray_hdda(ray_position, ray_direction, cascade, exclude_receiver_cell, hit_cell, hit_face, hit_cascade)) {
		return false;
	}

	bool disoccluded = false;
	if (hit_cascade == cascade && all(equal(ivec3(start_cell), hit_cell))) {
		ivec3 read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
		uint disocclusion = imageLoad(hddagi_voxel_disocclusion, read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0)).r;
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
			disocclusion = imageLoad(hddagi_voxel_disocclusion, read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0)).r;
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

	hit_cell += hit_face;
	ivec3 read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
	ivec3 texture_position = read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0);
	vec3 light = texelFetch(sampler3D(hddagi_light_cascades, linear_sampler), texture_position, 0).rgb;
	uint neighbour_bits = disoccluded ? 0u : imageLoad(hddagi_voxel_neighbours, texture_position).r & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
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
	return !any(isnan(r_radiance)) && !any(isinf(r_radiance)) && !any(lessThan(r_radiance, vec3(0.0)));
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
		if (!trace_hddagi_radiance(origin_position, origin_depth, origin_normal, ray_direction, candidate_radiance)) {
			candidate_radiance = sample_environment(ray_direction);
		}
		float cosine_pdf = max(dot(origin_normal, ray_direction), 0.0) / PI;
		radiance += candidate_radiance * hddagi.energy * (cosine_pdf / max(proposal_pdf, 1e-8));
	}
	radiance /= float(candidate_count);
	if (any(isnan(radiance)) || any(isinf(radiance))) {
		imageStore(raw_radiance_output, probe_position, vec4(0.0));
		return;
	}
	imageStore(raw_radiance_output, probe_position, vec4(clamp(radiance, vec3(0.0), vec3(65504.0)), 1.0));
}

#endif

#ifdef MODE_RESOLVE

bool load_full_resolution_surface(ivec2 screen_position, out float r_depth, out vec3 r_normal) {
	r_depth = texelFetch(sampler2D(depth_buffer, nearest_sampler), screen_position, 0).r;
	if (!(r_depth > 0.0)) {
		return false;
	}
	return decode_normal(texelFetch(sampler2D(normal_roughness_buffer, nearest_sampler), screen_position, 0).xyz, r_normal);
}

bool select_full_resolution_surface(ivec2 resolve_position, ivec2 resolve_size, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal) {
	ivec2 footprint_begin = (resolve_position * params.screen_size + resolve_size - ivec2(1)) / resolve_size;
	ivec2 footprint_end = ((resolve_position + ivec2(1)) * params.screen_size + resolve_size - ivec2(1)) / resolve_size;
	footprint_begin = clamp(footprint_begin, ivec2(0), params.screen_size - ivec2(1));
	footprint_end = clamp(footprint_end, footprint_begin + ivec2(1), params.screen_size);
	r_screen_position = clamp((footprint_begin + footprint_end - ivec2(1)) / 2, ivec2(0), params.screen_size - ivec2(1));
	r_depth = 0.0;
	r_normal = vec3(0.0, 0.0, 1.0);

	bool found = false;
	for (int y = footprint_begin.y; y < footprint_end.y; y++) {
		for (int x = footprint_begin.x; x < footprint_end.x; x++) {
			ivec2 candidate_position = ivec2(x, y);
			float candidate_depth;
			vec3 candidate_normal;
			if (!load_full_resolution_surface(candidate_position, candidate_depth, candidate_normal)) {
				continue;
			}
			if (!found || candidate_depth > r_depth) {
				found = true;
				r_screen_position = candidate_position;
				r_depth = candidate_depth;
				r_normal = candidate_normal;
			}
		}
	}
	return found;
}

bool load_resolve_probe_surface(ivec2 probe_position, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal) {
	uvec4 packed = imageLoad(screen_probe_surface_input, probe_position);
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	return r_depth > 0.0;
}

void screen_probe_resolve_main() {
	ivec2 resolve_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 resolve_size = imageSize(resolved_radiance_output);
	if (any(greaterThanEqual(resolve_position, resolve_size))) {
		return;
	}

	ivec2 screen_position;
	float pixel_depth;
	vec3 pixel_normal;
	if (!select_full_resolution_surface(resolve_position, resolve_size, screen_position, pixel_depth, pixel_normal)) {
		imageStore(resolved_radiance_output, resolve_position, vec4(0.0));
		return;
	}
	vec2 pixel_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	float pixel_linear_depth = compute_view_position(vec3(pixel_uv, pixel_depth)).z;
	if (isnan(pixel_linear_depth) || isinf(pixel_linear_depth)) {
		imageStore(resolved_radiance_output, resolve_position, vec4(0.0));
		return;
	}

	ivec2 gi_position = clamp(resolve_position * params.gi_size / resolve_size, ivec2(0), params.gi_size - ivec2(1));
	ivec2 probe_base = gi_position / max(params.probe_size, 1);
	ivec2 probe_count = textureSize(sampler2D(raw_radiance_input, nearest_sampler), 0);
	vec2 probe_screen_extent = vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size);
	float probe_extent = max((probe_screen_extent.x + probe_screen_extent.y) * 0.5, 1.0);
	vec3 radiance_sum = vec3(0.0);
	float weight_sum = 0.0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			ivec2 probe_position = probe_base + ivec2(x, y);
			if (any(lessThan(probe_position, ivec2(0))) || any(greaterThanEqual(probe_position, probe_count))) {
				continue;
			}
			ivec2 probe_screen_position;
			float probe_depth;
			vec3 probe_normal;
			if (!load_resolve_probe_surface(probe_position, probe_screen_position, probe_depth, probe_normal)) {
				continue;
			}
			vec4 raw_radiance = texelFetch(sampler2D(raw_radiance_input, nearest_sampler), probe_position, 0);
			if (!(raw_radiance.a > 0.0) || any(isnan(raw_radiance)) || any(isinf(raw_radiance))) {
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
			radiance_sum += max(raw_radiance.rgb, vec3(0.0)) * weight;
			weight_sum += weight;
		}
	}

	vec4 resolved = weight_sum > 0.0 ? vec4(radiance_sum / weight_sum, 1.0) : vec4(0.0);
	if (any(isnan(resolved)) || any(isinf(resolved))) {
		resolved = vec4(0.0);
	} else {
		resolved = clamp(resolved, vec4(0.0), vec4(65504.0));
	}
	imageStore(resolved_radiance_output, resolve_position, resolved);
}

#endif

#ifdef MODE_APPLY

void screen_probe_apply_main() {
	ivec2 gi_position = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(gi_position, imageSize(ambient_output)))) {
		return;
	}
	ivec2 radiance_size = textureSize(sampler2D(resolved_radiance_input, nearest_sampler), 0);
	ivec2 radiance_position = clamp(gi_position * radiance_size / params.gi_size, ivec2(0), radiance_size - ivec2(1));
	vec4 resolved_radiance = texelFetch(sampler2D(resolved_radiance_input, nearest_sampler), radiance_position, 0);
	if (!(resolved_radiance.a > 0.0) || any(isnan(resolved_radiance)) || any(isinf(resolved_radiance))) {
		return;
	}
	imageStore(ambient_output, gi_position, uvec4(rgbe_encode(max(resolved_radiance.rgb, vec3(0.0)))));
}

#endif

void main() {
#ifdef MODE_SURFACE
	screen_probe_surface_main();
#elif defined(MODE_TRACE)
	screen_probe_trace_main();
#elif defined(MODE_RESOLVE)
	screen_probe_resolve_main();
#elif defined(MODE_APPLY)
	screen_probe_apply_main();
#endif
}
