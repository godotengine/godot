#[compute]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"
#include "hddagi_voxel_metadata_inc.glsl"

#ifndef MOLTENVK_USED
#if defined(has_GL_KHR_shader_subgroup_ballot) && defined(has_GL_KHR_shader_subgroup_arithmetic)

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define USE_SUBGROUPS
#endif
#endif

#define REGION_SIZE 8

#define CACHE_IS_VALID 0x80000000
#define CACHE_IS_HIT 0x40000000

#define MAX_CASCADES 8

#ifdef MODE_PROCESS

layout(local_size_x = LIGHTPROBE_OCT_SIZE, local_size_y = LIGHTPROBE_OCT_SIZE, local_size_z = 1) in;

#define TRACE_SUBPIXEL

layout(rg32ui, set = 0, binding = 1) uniform restrict readonly uimage3D voxel_cascades;
layout(r8ui, set = 0, binding = 2) uniform restrict readonly uimage3D voxel_region_cascades;

layout(set = 0, binding = 3) uniform texture3D light_cascades;
layout(set = 0, binding = 4) uniform sampler linear_sampler;
layout(r32ui, set = 0, binding = 5) uniform restrict uimage2DArray lightprobe_texture_data;
layout(r32ui, set = 0, binding = 6) uniform restrict writeonly uimage2DArray lightprobe_diffuse_data;
layout(rgba16f, set = 0, binding = 7) uniform restrict writeonly image2DArray lightprobe_ambient_tex;
layout(r32ui, set = 0, binding = 8) uniform restrict uimage2DArray ray_hit_cache;
layout(r16ui, set = 0, binding = 9) uniform restrict uimage2DArray ray_hit_cache_version;
layout(r16ui, set = 0, binding = 10) uniform restrict uimage3D region_versions;
layout(r32ui, set = 0, binding = 11) uniform restrict uimage2DArray lightprobe_moving_average_history;
layout(r32ui, set = 0, binding = 12) uniform restrict uimage2DArray lightprobe_moving_average;
layout(r32ui, set = 0, binding = 14) uniform restrict uimage2DArray lightprobe_update_frames;
layout(r8, set = 0, binding = 15) uniform restrict readonly image2DArray lightprobe_geometry_proximity;
layout(r8, set = 0, binding = 16) uniform restrict readonly image2DArray lightprobe_camera_visibility;
shared uint probe_lighting_changed;

#ifdef USE_RADIANCE_OCTMAP_ARRAY
layout(set = 1, binding = 0) uniform texture2DArray sky_irradiance;
#else
layout(set = 1, binding = 0) uniform texture2D sky_irradiance;
#endif
layout(set = 1, binding = 1) uniform sampler linear_sampler_mipmaps;

#define SKY_FLAGS_MODE_COLOR 0x01
#define SKY_FLAGS_MODE_SKY 0x02
#define SKY_FLAGS_ORIENTATION_SIGN 0x04

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 region_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 13, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

// MODE_PROCESS
#endif

#ifdef MODE_FILTER

layout(local_size_x = LIGHTPROBE_OCT_SIZE, local_size_y = LIGHTPROBE_OCT_SIZE, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage2DArray lightprobe_src_diffuse_data;
layout(r32ui, set = 0, binding = 2) uniform restrict writeonly uimage2DArray lightprobe_dst_diffuse_data;
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage2DArray lightprobe_neighbours;
layout(r8, set = 0, binding = 4) uniform restrict readonly image2DArray lightprobe_geometry_proximity;
layout(r8, set = 0, binding = 5) uniform restrict readonly image2DArray lightprobe_camera_visibility;

#endif

#ifdef MODE_CAMERA_VISIBILITY

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(r8, set = 0, binding = 1) uniform restrict writeonly image2DArray lightprobe_camera_visibility;

#define MAX_CAMERA_PLANES 6
#define MAX_CAMERA_POINTS 8

layout(set = 0, binding = 2, std140) uniform CameraPlanes {
	vec4 planes[MAX_CAMERA_PLANES];
	vec4 points[MAX_CAMERA_POINTS];
}
camera;

#endif

layout(push_constant, std430) uniform Params {
	ivec3 grid_size;
	int max_cascades;

	float ray_bias;
	int cascade;
	int inactive_update_frames;
	int history_size;

	ivec3 world_offset;
	uint sky_flags;

	uint history_mode;
	float history_response;
	uint pad;
	float sky_energy;

	vec3 sky_color_or_orientation;
	float y_mult;

	ivec3 probe_axis_size;
	bool store_ambient_texture;

	vec2 sky_irradiance_border_size;
	int global_frame;
	uint motion_accum; // Motion that happened since last update (bit 0 in X, bit 1 in Y, bit 2 in Z).
}
params;

uvec3 hash3(uvec3 x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

uint hash(uint x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

vec3 octahedron_decode(vec2 f) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	f = f * 2.0 - 1.0;
	vec3 n = vec3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
	float t = clamp(-n.z, 0.0, 1.0);
	n.x += n.x >= 0 ? -t : t;
	n.y += n.y >= 0 ? -t : t;
	return normalize(n);
}

uint rgbe_encode(vec3 rgb) {
	const float rgbe_max = uintBitsToFloat(0x477F8000);
	const float rgbe_min = uintBitsToFloat(0x37800000);

	rgb = clamp(rgb, 0, rgbe_max);

	float max_channel = max(max(rgbe_min, rgb.r), max(rgb.g, rgb.b));

	float bias = uintBitsToFloat((floatBitsToUint(max_channel) + 0x07804000) & 0x7F800000);

	uvec3 urgb = floatBitsToUint(rgb + bias);
	uint e = (floatBitsToUint(bias) << 4) + 0x10000000;
	return e | (urgb.b << 18) | (urgb.g << 9) | (urgb.r & 0x1FF);
}

vec3 rgbe_decode(uint p_rgbe) {
	vec4 rgbef = vec4((uvec4(p_rgbe) >> uvec4(0, 9, 18, 27)) & uvec4(0x1FF, 0x1FF, 0x1FF, 0x1F));
	return rgbef.rgb * pow(2.0, rgbef.a - 15.0 - 9.0);
}

#define FP_BITS 14
#define FP_MAX ((1 << 22) - 1)

uvec3 rgbe_decode_fp(uint p_rgbe, int p_fp_bits) {
	uvec4 rgbe = (uvec4(p_rgbe) >> uvec4(0, 9, 18, 27)) &
			uvec4(0x1FF, 0x1FF, 0x1FF, 0x1F);
	int shift = int(rgbe.a) - 15 - 9 + p_fp_bits;
	if (shift >= 0) {
		rgbe.rgb <<= uint(shift);
	} else {
		rgbe.rgb >>= uint(-shift);
	}
	return rgbe.rgb;
}

#ifdef MODE_PROCESS

#define HDDAGI_TRACE_PROBES
#include "hddagi_trace_inc.glsl"

#if LIGHTPROBE_OCT_SIZE == 4
const uint neighbour_max_weights = 8;
#endif

#if LIGHTPROBE_OCT_SIZE == 5
const uint neighbour_max_weights = 15;
#endif

#if LIGHTPROBE_OCT_SIZE == 6
const uint neighbour_max_weights = 18;
#endif

// An SSBO avoids per-invocation lookup tables on D3D12.
layout(set = 0, binding = 17, std430) readonly restrict buffer ProbeLookupData {
	uint neighbour_weights[LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE * neighbour_max_weights];
	uint wrap_neighbours[(LIGHTPROBE_OCT_SIZE + 2) * (LIGHTPROBE_OCT_SIZE + 2)];
}
probe_lookup;

shared uvec3 neighbours_accum[LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE];
shared vec3 neighbors[LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE];
shared uvec3 ambient_accum;
shared int probe_history_index;

// MODE_PROCESS
#endif

ivec3 modi(ivec3 value, ivec3 p_y) {
	// GLSL Specification says:
	// "Results are undefined if one or both operands are negative."
	// Use nonnegative operands to define wrapping for negative coordinates.
	return mix(value % p_y, p_y - ((abs(value) - ivec3(1)) % p_y) - 1, lessThan(sign(value), ivec3(0)));
}

#define FRAME_MASK 0x0FFFFFFF
#define FORCE_UPDATE_MASK 0xF0000000
#define FORCE_UPDATE_SHIFT 28

void main() {
#ifdef MODE_PROCESS

	ivec2 pos = ivec2(gl_WorkGroupID.xy);
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy);
	uint probe_index = gl_LocalInvocationID.x + gl_LocalInvocationID.y * LIGHTPROBE_OCT_SIZE;

	// clear
	neighbours_accum[probe_index] = uvec3(0);

	ivec3 probe_cell;
	probe_cell.x = pos.x;
	probe_cell.y = pos.y % params.probe_axis_size.y;
	probe_cell.z = pos.y / params.probe_axis_size.y;

	ivec3 probe_world_pos = params.world_offset + probe_cell;

	ivec3 probe_scroll_pos = modi(probe_world_pos, params.probe_axis_size);
	ivec3 probe_texture_pos = ivec3((probe_scroll_pos.xy + ivec2(0, probe_scroll_pos.z * params.probe_axis_size.y)), params.cascade);

	if (probe_index == 0) {
		probe_lighting_changed = 0u;
		// Determine whether it should process the probe.

		bool process = false;

		// Fetch frame.
		uint frame = imageLoad(lightprobe_update_frames, probe_texture_pos).r;
		uint forced_update = frame >> FORCE_UPDATE_SHIFT;
		if (forced_update > 0) {
			// Check whether it must force the update
			process = true;
			forced_update--;
			frame = (frame & FRAME_MASK) | (forced_update << FORCE_UPDATE_SHIFT);
		}

		bool geom_proximity = imageLoad(lightprobe_geometry_proximity, probe_texture_pos).r > 0.5;
		if (geom_proximity) {
			bool camera_visible = imageLoad(lightprobe_camera_visibility, probe_texture_pos).r > 0.5;
			process = process || camera_visible;
		}

		if (!process) {
			int frame_offset = 0;
			if ((probe_world_pos.x & 1) != 0) {
				frame_offset |= 1;
			}
			if ((probe_world_pos.y & 1) != 0) {
				frame_offset |= 2;
			}
			if ((probe_world_pos.z & 1) != 0) {
				frame_offset |= 4;
			}

			if (((params.global_frame + frame_offset) % params.inactive_update_frames) == 0) {
				// Process every params.inactive_update_frames.
				process = true;
			}
		}

		if (process) {
			uint local_frame = frame & FRAME_MASK;
			probe_history_index = int(local_frame) % params.history_size;
			frame = ((local_frame + 1) & FRAME_MASK) | (frame & FORCE_UPDATE_MASK);
			// Store it back.
			imageStore(lightprobe_update_frames, probe_texture_pos, uvec4(frame));

		} else {
			probe_history_index = -1; // No processing.
		}

		ambient_accum = uvec3(0);
	}

	memoryBarrierShared();
	barrier();

	vec3 light;
	ivec3 cache_texture_pos;
	vec3 ray_dir;
	vec2 sample_ofs;
	vec3 ray_pos;
	bool hit;
	ivec3 hit_cell;
	int hit_cascade;
	bool cache_valid;
	vec3 cache_invalidated_debug;
	uint cache_entry;

	if (probe_history_index < 0) {
		return; // All threads return, so no barrier will be executed.
	}

	float probe_cell_size = float(params.grid_size.x) / float(params.probe_axis_size.x - 1) / cascades.data[params.cascade].to_cell;

	ray_pos = cascades.data[params.cascade].offset + vec3(probe_cell) * probe_cell_size;

	// Ensure a unique hash that includes the probe world position, the local octahedron pixel, and the history frame index
	uvec3 h3 = hash3(uvec3((uvec3(probe_world_pos) * LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE + uvec3(probe_index)) * uvec3(params.history_size) + uvec3(probe_history_index)));
	uint h = (h3.x ^ h3.y) ^ h3.z;
	sample_ofs = vec2(ivec2(h >> 16, h & 0xFFFF)) / vec2(0xFFFF);
	ray_dir = octahedron_decode((vec2(local_pos) + sample_ofs) / vec2(LIGHTPROBE_OCT_SIZE));

	ray_dir.y *= params.y_mult;
	ray_dir = normalize(ray_dir);

	float bias = params.ray_bias;
	vec3 abs_ray_dir = abs(ray_dir);
	ray_pos += ray_dir * 1.0 / max(abs_ray_dir.x, max(abs_ray_dir.y, abs_ray_dir.z)) * bias / cascades.data[params.cascade].to_cell;

	cache_texture_pos = ivec3(probe_texture_pos.xy * LIGHTPROBE_OCT_SIZE + local_pos, probe_texture_pos.z * params.history_size + probe_history_index);
	cache_entry = imageLoad(ray_hit_cache, cache_texture_pos).r;

	cache_valid = bool(cache_entry & CACHE_IS_VALID);

	cache_invalidated_debug = vec3(0.0);

	if (cache_valid) {
		hit = bool(cache_entry & CACHE_IS_HIT);
		uvec4 uhit = (uvec4(cache_entry) >> uvec4(0, 8, 16, 24)) & uvec4(0xFF, 0xFF, 0xFF, 0x7);
		hit_cell = ivec3(uhit.xyz);
		hit_cascade = int(uhit.w);
		uint axis = (cache_entry >> 27) & 0x3;
		if (bool((1 << axis) & params.motion_accum)) {
			// There was motion in this axis, cache is no longer valid.
			cache_valid = false;
			cache_invalidated_debug = vec3(0, 0, 4.0);
		} else if (hit) {
			// Check if the region pointed to is still valid.
			uint version = imageLoad(ray_hit_cache_version, cache_texture_pos).r;
			uint region_version = imageLoad(region_versions, (hit_cell / REGION_SIZE) + ivec3(0, hit_cascade * (params.grid_size.y / REGION_SIZE), 0)).r;

			if (region_version != version) {
				cache_valid = false;
				cache_invalidated_debug = (hit_cascade == params.cascade) ? vec3(0.0, 4.00, 0.0) : vec3(4.0, 0, 0.0);
			}
		}
	}

	if (!cache_valid) {
		ivec3 hit_face;
		hit = trace_ray_hdda(ray_pos, ray_dir, params.cascade, hit_cell, hit_face, hit_cascade);
		if (hit) {
			hit_cell += hit_face;

			ivec3 reg_cell_offset = cascades.data[hit_cascade].region_world_offset * REGION_SIZE;
			hit_cell = (hit_cell + reg_cell_offset) & (params.grid_size - 1); // Read from wrapped world coordinates
		}
	}

	if (hit) {
		ivec3 spos = hit_cell;
		spos.y += hit_cascade * params.grid_size.y;
		light = texelFetch(sampler3D(light_cascades, linear_sampler), spos, 0).rgb;
	} else if (bool(params.sky_flags & SKY_FLAGS_MODE_SKY)) {
		// Reconstruct sky orientation as a quaternion and rotate the ray before sampling.
		float sky_sign = bool(params.sky_flags & SKY_FLAGS_ORIENTATION_SIGN) ? 1.0 : -1.0;
		vec4 sky_quat = vec4(params.sky_color_or_orientation, sky_sign * sqrt(1.0 - dot(params.sky_color_or_orientation, params.sky_color_or_orientation)));
		vec3 sky_dir = cross(sky_quat.xyz, ray_dir);
		sky_dir = ray_dir + ((sky_dir * sky_quat.w) + cross(sky_quat.xyz, sky_dir)) * 2.0;
#ifdef USE_RADIANCE_OCTMAP_ARRAY
		light = textureLod(sampler2DArray(sky_irradiance, linear_sampler_mipmaps), vec3(vec3_to_oct_with_border(sky_dir, params.sky_irradiance_border_size), 0.0), 2.0).rgb; // Use second mipmap because we don't usually throw a lot of rays, so this compensates.
#else
		light = textureLod(sampler2D(sky_irradiance, linear_sampler_mipmaps), vec3_to_oct_with_border(sky_dir, params.sky_irradiance_border_size), 2.0).rgb; // Use second mipmap because we don't usually throw a lot of rays, so this compensates.
#endif
		light *= params.sky_energy;
	} else if (bool(params.sky_flags & SKY_FLAGS_MODE_COLOR)) {
		light = params.sky_color_or_orientation;
		light *= params.sky_energy;
	} else {
		light = vec3(0);
	}

	memoryBarrierShared();
	barrier();

	// Plot the light to the octahedron using bilinear filtering
#ifdef TRACE_SUBPIXEL
	sample_ofs = sample_ofs * 2.0 - 1.0;
	ivec2 bilinear_base = ivec2(1) + local_pos - mix(ivec2(0), ivec2(1), lessThan(sample_ofs, vec2(0)));
	vec2 blend = mix(sample_ofs, 1.0 + sample_ofs, lessThan(sample_ofs, vec2(0)));
	for (int i = 0; i < 2; i++) {
		float i_w = i == 0 ? 1.0 - blend.y : blend.y;
		for (int j = 0; j < 2; j++) {
			float j_w = j == 0 ? 1.0 - blend.x : blend.x;
			uint wrap_neighbour = probe_lookup.wrap_neighbours[(bilinear_base.y + i) * (LIGHTPROBE_OCT_SIZE + 2) + (bilinear_base.x + j)];
			ivec2 write_to = ivec2(wrap_neighbour & 0xFFFF, wrap_neighbour >> 16);
			int write_offset = write_to.y * LIGHTPROBE_OCT_SIZE + write_to.x;
			float write_weight = i_w * j_w;

			uvec3 lightu = uvec3(clamp((light * write_weight) * float(1 << FP_BITS), 0, float(FP_MAX)));
			atomicAdd(neighbours_accum[write_offset].r, lightu.r);
			atomicAdd(neighbours_accum[write_offset].g, lightu.g);
			atomicAdd(neighbours_accum[write_offset].b, lightu.b);
		}
	}
#else

	neighbors[probe_index] = light;
#endif

	if (!cache_valid) {
		cache_entry = CACHE_IS_VALID;
		// Determine the side of the cascade box this ray exited through, this is important for invalidation purposes.

		vec3 unit_pos = ray_pos - cascades.data[params.cascade].offset;
		unit_pos *= cascades.data[params.cascade].to_cell;

		vec3 t0 = -unit_pos / ray_dir;
		vec3 t1 = (vec3(params.grid_size) - unit_pos) / ray_dir;
		vec3 tmax = max(t0, t1);

		uint axis;
		float m;
		if (tmax.x < tmax.y) {
			axis = 0;
			m = tmax.x;
		} else {
			axis = 1;
			m = tmax.y;
		}
		if (tmax.z < m) {
			axis = 2;
		}

		cache_entry |= axis << 27; // Misses must also observe scrolling on their exit axis.
		if (hit) {
			uvec3 ucell = (uvec3(hit_cell) & uvec3(0xFF)) << uvec3(0, 8, 16);
			cache_entry |= CACHE_IS_HIT | ucell.x | ucell.y | ucell.z | (uint(min(7, hit_cascade)) << 24);

			uint region_version = imageLoad(region_versions, (hit_cell / REGION_SIZE) + ivec3(0, hit_cascade * (params.grid_size.y / REGION_SIZE), 0)).r;

			imageStore(ray_hit_cache_version, cache_texture_pos, uvec4(region_version));
		}

		imageStore(ray_hit_cache, cache_texture_pos, uvec4(cache_entry));
	}

	groupMemoryBarrier();
	barrier();

#ifdef TRACE_SUBPIXEL
	light = vec3(neighbours_accum[probe_index]) / float(1 << FP_BITS);
#else
	light = neighbors[probe_index];
#endif

	if (params.history_mode == 0u) {
		uint light_rgbe = rgbe_encode(light);

		ivec3 ma_pos = ivec3(cache_texture_pos.xy * ivec2(3, 1), params.cascade);

		uvec3 moving_average = uvec3(
				imageLoad(lightprobe_moving_average, ma_pos + ivec3(0, 0, 0)).r,
				imageLoad(lightprobe_moving_average, ma_pos + ivec3(1, 0, 0)).r,
				imageLoad(lightprobe_moving_average, ma_pos + ivec3(2, 0, 0)).r);

		uvec3 prev_val = rgbe_decode_fp(imageLoad(lightprobe_moving_average_history, cache_texture_pos).r, FP_BITS);

		moving_average -= prev_val;
		uvec3 new_val = rgbe_decode_fp(light_rgbe, FP_BITS); // Round trip to ensure integer consistency
		float lighting_response = 0.0;
		if (params.history_response > 0.0) {
			// Compare the same ray phase to detect lighting changes at a cached endpoint.
			vec3 signal_delta = abs(vec3(new_val) - vec3(prev_val));
			vec3 signal_threshold = max(max(vec3(new_val), vec3(prev_val)) * 0.2, vec3(16.0));
			vec3 relative_delta = signal_delta / signal_threshold;
			float peak_delta = max(relative_delta.x, max(relative_delta.y, relative_delta.z));
			if (peak_delta > 1.0) {
				atomicOr(probe_lighting_changed, 1u);
			}
			float current_weight = 1.0 / float(params.history_size);
			float response_limit = max(params.history_response - current_weight, 0.0) / max(1.0 - current_weight, 1e-6);
			lighting_response = response_limit * smoothstep(1.0, 3.0, peak_delta);
		}
		moving_average += new_val;

		imageStore(lightprobe_moving_average_history, cache_texture_pos, uvec4(light_rgbe));

		imageStore(lightprobe_moving_average, ma_pos + ivec3(0, 0, 0), uvec4(moving_average.r));
		imageStore(lightprobe_moving_average, ma_pos + ivec3(1, 0, 0), uvec4(moving_average.g));
		imageStore(lightprobe_moving_average, ma_pos + ivec3(2, 0, 0), uvec4(moving_average.b));

		moving_average /= params.history_size;
		if (lighting_response > 0.0) {
			moving_average = uvec3(mix(vec3(moving_average), vec3(new_val), lighting_response));
		}

		if (params.store_ambient_texture) {
			atomicAdd(ambient_accum.r, moving_average.r);
			atomicAdd(ambient_accum.g, moving_average.g);
			atomicAdd(ambient_accum.b, moving_average.b);
		}

		light = vec3(moving_average) / float(1 << FP_BITS);
	} else {
		ivec3 history_pos = ivec3(cache_texture_pos.xy, params.cascade);
		vec3 previous = rgbe_decode(imageLoad(lightprobe_moving_average_history, history_pos).r);
		light = mix(previous, light, 2.0 / float(params.history_size + 1));
		imageStore(lightprobe_moving_average_history, history_pos, uvec4(rgbe_encode(light)));
		if (params.store_ambient_texture) {
			uvec3 ambient = uvec3(clamp(light * float(1 << FP_BITS), 0.0, float(FP_MAX)));
			atomicAdd(ambient_accum.r, ambient.r);
			atomicAdd(ambient_accum.g, ambient.g);
			atomicAdd(ambient_accum.b, ambient.b);
		}
	}
	neighbors[probe_index] = light;

	groupMemoryBarrier();
	barrier();

	// Compute specular, diffuse, ambient
	if (probe_index == 0 && probe_lighting_changed != 0u) {
		uint frame = imageLoad(lightprobe_update_frames, probe_texture_pos).r;
		frame = (frame & FRAME_MASK) | (max(frame >> FORCE_UPDATE_SHIFT, 2u) << FORCE_UPDATE_SHIFT);
		imageStore(lightprobe_update_frames, probe_texture_pos, uvec4(frame));
	}

	vec3 diffuse_light = vec3(0);
	vec3 specular_light = light;

	for (uint i = 0; i < neighbour_max_weights; i++) {
		uint n = probe_lookup.neighbour_weights[probe_index * neighbour_max_weights + i];
		uint index = n >> 16;
		float weight = float(n & 0xFFFF) / float(0xFFFF);
		diffuse_light += neighbors[index] * weight;
	}

	ivec3 store_texture_pos = ivec3(probe_texture_pos.xy * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1), probe_texture_pos.z);
	ivec3 probe_read_pos = store_texture_pos + ivec3(local_pos, 0);

	//if (cache_invalidated_debug!=vec3(0.0)) {
	//	diffuse_light = cache_invalidated_debug;
	//}

	// Store in octahedral map

	ivec3 copy_to[4] = ivec3[](ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));
	copy_to[0] = probe_read_pos;

	if (local_pos == ivec2(0, 0)) {
		copy_to[1] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - 1, -1, 0);
		copy_to[2] = store_texture_pos + ivec3(-1, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, 0)) {
		copy_to[1] = store_texture_pos + ivec3(0, -1, 0);
		copy_to[2] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = store_texture_pos + ivec3(-1, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(0, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = store_texture_pos + ivec3(-1, 0, 0);
		copy_to[2] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[3] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, -1, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = store_texture_pos + ivec3(0, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[2] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, 0, 0);
		copy_to[3] = store_texture_pos + ivec3(-1, -1, 0);
	} else if (local_pos.y == 0) {
		copy_to[1] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y - 1, 0);
	} else if (local_pos.x == 0) {
		copy_to[1] = store_texture_pos + ivec3(local_pos.x - 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	} else if (local_pos.y == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = store_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y + 1, 0);
	} else if (local_pos.x == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = store_texture_pos + ivec3(local_pos.x + 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	}

	uint light_rgbe = rgbe_encode(specular_light);
	uint diffuse_rgbe = rgbe_encode(diffuse_light);

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(lightprobe_texture_data, copy_to[i], uvec4(light_rgbe));
		imageStore(lightprobe_diffuse_data, copy_to[i], uvec4(diffuse_rgbe));
		// also to diffuse
	}

	if (params.store_ambient_texture && probe_index == 0) {
		vec3 ambient_light = vec3(ambient_accum) / float(1 << FP_BITS);
		ambient_light /= LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE;
		imageStore(lightprobe_ambient_tex, ivec3(probe_texture_pos.xy, params.cascade), vec4(ambient_light, 1));
	}

#endif

#ifdef MODE_FILTER

	ivec2 pos = ivec2(gl_WorkGroupID.xy);
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy);

	ivec3 probe_cell;
	probe_cell.x = pos.x;
	probe_cell.y = pos.y % params.probe_axis_size.y;
	probe_cell.z = pos.y / params.probe_axis_size.y;

	ivec3 probe_world_pos = params.world_offset + probe_cell;

	ivec3 probe_scroll_pos = modi(probe_world_pos, params.probe_axis_size);
	ivec3 probe_base_pos = ivec3((probe_scroll_pos.xy + ivec2(0, probe_scroll_pos.z * params.probe_axis_size.y)), params.cascade);

	bool geom_proximity = imageLoad(lightprobe_geometry_proximity, probe_base_pos).r > 0.5;
	bool cam_visibility = imageLoad(lightprobe_camera_visibility, probe_base_pos).r > 0.5;

	ivec3 probe_texture_pos = ivec3(probe_base_pos.xy * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1), probe_base_pos.z);
	ivec3 probe_read_pos = probe_texture_pos + ivec3(local_pos, 0);

	vec4 light;
	light.rgb = rgbe_decode(imageLoad(lightprobe_src_diffuse_data, probe_read_pos).r);
	light.a = 1.0;

	if (geom_proximity && cam_visibility) {
		// Only filter if there is geom proximity and probe is visible by camera.

		uint neighbour_visibility = imageLoad(lightprobe_neighbours, probe_base_pos).r;

		for (int i = 0; i < 6; i++) {
			float visibility = ((neighbour_visibility >> (i * 4)) & 0xF) / float(0xF);
			if (visibility == 0.0) {
				continue; // un-neighboured
			}

			ivec3 neighbour_probe = probe_cell + hddagi_axis_direction(uint(i));
			if (any(lessThan(neighbour_probe, ivec3(0))) || any(greaterThanEqual(neighbour_probe, params.probe_axis_size))) {
				continue; // Outside range.
			}

			ivec3 probe_world_pos2 = params.world_offset + neighbour_probe;
			ivec3 probe_scroll_pos2 = modi(probe_world_pos2, params.probe_axis_size);
			ivec3 probe_base_pos2 = ivec3((probe_scroll_pos2.xy + ivec2(0, probe_scroll_pos2.z * params.probe_axis_size.y)), params.cascade);

			ivec3 probe_texture_pos2 = ivec3(probe_base_pos2.xy * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1), probe_base_pos2.z);
			ivec3 probe_read_pos2 = probe_texture_pos2 + ivec3(local_pos, 0);

			vec4 light2;
			light2.rgb = rgbe_decode(imageLoad(lightprobe_src_diffuse_data, probe_read_pos2).r);
			light2.a = 1.0;

			light += light2 * 0.7 * visibility;
		}
		light.rgb /= light.a;
	}

	ivec3 copy_to[4] = ivec3[](ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));
	copy_to[0] = probe_read_pos;

	if (local_pos == ivec2(0, 0)) {
		copy_to[1] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - 1, -1, 0);
		copy_to[2] = probe_texture_pos + ivec3(-1, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, 0)) {
		copy_to[1] = probe_texture_pos + ivec3(0, -1, 0);
		copy_to[2] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = probe_texture_pos + ivec3(-1, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(0, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = probe_texture_pos + ivec3(-1, 0, 0);
		copy_to[2] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[3] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, -1, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = probe_texture_pos + ivec3(0, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[2] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE, 0, 0);
		copy_to[3] = probe_texture_pos + ivec3(-1, -1, 0);
	} else if (local_pos.y == 0) {
		copy_to[1] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y - 1, 0);
	} else if (local_pos.x == 0) {
		copy_to[1] = probe_texture_pos + ivec3(local_pos.x - 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	} else if (local_pos.y == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = probe_texture_pos + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y + 1, 0);
	} else if (local_pos.x == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = probe_texture_pos + ivec3(local_pos.x + 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	}

	uint light_rgbe = rgbe_encode(light.rgb);

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(lightprobe_dst_diffuse_data, copy_to[i], uvec4(light_rgbe));
	}

#endif

#ifdef MODE_CAMERA_VISIBILITY

	ivec3 region = ivec3(gl_GlobalInvocationID.xyz);

	if (any(greaterThanEqual(region, params.grid_size / REGION_SIZE))) {
		return;
	}

	vec3 half_extents = vec3(REGION_SIZE * 0.5);
	vec3 ofs = vec3(region * REGION_SIZE) + half_extents;

	for (int i = 0; i < MAX_CAMERA_PLANES; i++) {
		vec4 plane = camera.planes[i];
		vec3 point = ofs + mix(+half_extents, -half_extents, greaterThan(plane.xyz, vec3(0.0)));
		if (dot(plane.xyz, point) > plane.w) {
			return;
		}
	}

	// The region's axes can separate it from a frustum even when all plane tests pass.
	ivec3 bad_points_pos = ivec3(0);
	ivec3 bad_points_neg = ivec3(0);
	vec3 test_min = ofs - half_extents;
	vec3 test_max = ofs + half_extents;
	for (int i = 0; i < MAX_CAMERA_POINTS; i++) {
		vec3 point = camera.points[i].xyz;
		bad_points_neg += mix(ivec3(0), ivec3(1), lessThan(point, test_min));
		bad_points_pos += mix(ivec3(0), ivec3(1), greaterThan(point, test_max));
	}

	if (any(equal(bad_points_pos, ivec3(MAX_CAMERA_POINTS))) || any(equal(bad_points_neg, ivec3(MAX_CAMERA_POINTS)))) {
		return;
	}

	for (int i = 0; i < 8; i++) {
		ivec3 probe = params.world_offset + region + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1));
		ivec3 probe_scroll_pos = modi(probe, params.probe_axis_size);
		ivec3 probe_pos = ivec3((probe_scroll_pos.xy + ivec2(0, probe_scroll_pos.z * params.probe_axis_size.y)), params.cascade);

		imageStore(lightprobe_camera_visibility, probe_pos, vec4(1.0));
	}

#endif
}
