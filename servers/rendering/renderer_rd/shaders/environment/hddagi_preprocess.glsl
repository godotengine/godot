#[compute]

#version 450

#VERSION_DEFINES

#if defined(MODE_LIGHT_STORE)
#define HDDAGI_DATA_BINDING 12
#define HDDAGI_PAGE_BINDING 11
#define HDDAGI_WRITE_VOXELS
#elif defined(MODE_LIGHT_SCROLL)
#define HDDAGI_DATA_BINDING 1
#define HDDAGI_PAGE_BINDING 12
#define HDDAGI_WRITE_VOXELS
#elif defined(MODE_OCCLUSION_FILTER)
#define HDDAGI_DATA_BINDING 3
#define HDDAGI_PAGE_BINDING 6
#endif
#ifdef HDDAGI_DATA_BINDING
#include "hddagi_voxel_pages_inc.glsl"
#endif

#include "hddagi_voxel_metadata_inc.glsl"

#define REGION_SIZE 8

#define square(m) ((m) * (m))

#ifdef MODE_REGION_STORE

layout(local_size_x = REGION_SIZE, local_size_y = REGION_SIZE, local_size_z = REGION_SIZE) in;

layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_normal_bits;
layout(rg32ui, set = 0, binding = 2) uniform restrict writeonly uimage3D dst_voxels;
layout(r8ui, set = 0, binding = 3) uniform restrict writeonly uimage3D dst_region_bits;
layout(r16ui, set = 0, binding = 4) uniform restrict writeonly uimage3D region_version;

shared uint region_bits[((REGION_SIZE / 4) * (REGION_SIZE / 4) * (REGION_SIZE / 4)) * 2];

#endif

#ifdef MODE_LIGHT_STORE

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(r16ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_albedo;
layout(r32ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_emission;
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_emission_aniso;
layout(r32ui, set = 0, binding = 4) uniform restrict readonly uimage3D src_normal_bits;

layout(set = 0, binding = 8) uniform texture3D occlusion[2];
layout(set = 0, binding = 9) uniform sampler linear_sampler;

shared uint normal_facings[6 * 6 * 6];

uint get_normal_facing(ivec3 p_pos) {
	p_pos += ivec3(1);
	return normal_facings[p_pos.z * 6 * 6 + p_pos.y * 6 + p_pos.x];
}

#endif

#if defined(MODE_LIGHT_STORE) || defined(MODE_LIGHT_SCROLL)

struct ProcessVoxel {
	uint position; // xyz 7 bits each, plus dynamic and static pending flags
	uint albedo_normal; // 0 - 16, 17 - 31 normal in octahedral format
	uint emission; // RGBE emission
	uint occlusion; // cached 4 bits occlusion for each 8 neighboring probes
};

#endif

#if defined(MODE_LIGHT_STORE) || defined(MODE_LIGHT_SCROLL)

#define PROCESS_STATIC_PENDING_BIT 0x80000000
#define PROCESS_DYNAMIC_PENDING_BIT 0x40000000

layout(set = 0, binding = 5, std430) restrict buffer DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;

layout(set = 0, binding = 6, std430) restrict buffer writeonly ProcessVoxels {
	ProcessVoxel data[];
}
dst_process_voxels;

layout(set = 0, binding = 13, std430) restrict writeonly buffer SourceEmission {
	uint data[];
}
dst_source_emission;

shared uint store_position_count;
shared uint store_from_index;

#endif

#if defined(MODE_LIGHT_SCROLL)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 7, std430) restrict buffer SrcDispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
src_dispatch_data;

layout(set = 0, binding = 8, std430) restrict buffer readonly SrcProcessVoxels {
	ProcessVoxel data[];
}
src_process_voxels;

layout(set = 0, binding = 14, std430) restrict readonly buffer PreviousSourceEmission {
	uint data[];
}
src_source_emission;

#endif

#ifdef MODE_OCCLUSION_FILTER
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 1) uniform texture3D occlusion[2];
layout(set = 0, binding = 2) uniform sampler linear_sampler;
layout(rgba8_snorm, set = 0, binding = 4) uniform restrict writeonly image3D dst_correction_0;
layout(rgba8_snorm, set = 0, binding = 5) uniform restrict writeonly image3D dst_correction_1;
#endif

// Some occlusion defines

#ifdef HIGH_DENSITY_PROBES
#define PROBE_CELLS 4
#else
#define PROBE_CELLS 8
#endif

#define OCC_DISTANCE_MAX 16.0

#ifdef MODE_OCCLUSION

#define OCC_PLANE (PROBE_CELLS * PROBE_CELLS)
#define OCC_VOLUME (OCC_PLANE * PROBE_CELLS)
layout(local_size_x = 4 * OCC_PLANE, local_size_y = 1, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_normal_bits;
#ifdef HDDAGI_RGBA8_OCCLUSION
layout(rgba8, set = 0, binding = 4) uniform restrict writeonly image3D dst_occlusion;
#else
layout(r16ui, set = 0, binding = 4) uniform restrict writeonly uimage3D dst_occlusion;
#endif

shared float occlusion[OCC_VOLUME * 4];
shared uint bit_normals[OCC_VOLUME];
#endif

#ifdef MODE_LIGHTPROBE_NEIGHBOURS

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 1) uniform texture3D occlusion[2];
layout(set = 0, binding = 2) uniform sampler linear_sampler;
layout(r32ui, set = 0, binding = 3) uniform restrict writeonly uimage2DArray neighbour_probe_visibility;

#endif

#ifdef MODE_LIGHTPROBE_UPDATE_FRAMES

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(r32ui, set = 0, binding = 1) uniform restrict uimage2DArray lightprobe_frames;

#endif

#ifdef MODE_LIGHTPROBE_GEOMETRY_PROXIMITY

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(r8ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_region_bits;
layout(r8, set = 0, binding = 2) uniform restrict writeonly image2DArray dst_probe_geometry_proximity;
layout(rg32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_voxel_bits;

#endif

#ifdef MODE_LIGHTPROBE_SCROLL

layout(local_size_x = LIGHTPROBE_OCT_SIZE, local_size_y = LIGHTPROBE_OCT_SIZE, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 1) uniform restrict uimage2DArray lightprobe_specular_data;
layout(r32ui, set = 0, binding = 2) uniform restrict uimage2DArray lightprobe_diffuse_data;
layout(rgba16f, set = 0, binding = 3) uniform restrict image2DArray lightprobe_ambient_tex;
layout(r32ui, set = 0, binding = 4) uniform restrict uimage2DArray ray_hit_cache;
layout(r32ui, set = 0, binding = 5) uniform restrict uimage2DArray lightprobe_moving_average_history;
layout(r32ui, set = 0, binding = 6) uniform restrict uimage2DArray lightprobe_moving_average;

layout(set = 0, binding = 7) uniform texture3D occlusion[2];
layout(set = 0, binding = 8) uniform sampler linear_sampler;

shared float occlusion_blend[8];

#endif

layout(push_constant, std430) uniform Params {
	ivec3 grid_size;
	uint region_version;

	ivec3 scroll;
	int cascade_count;

	ivec3 offset;
	int probe_update_frames;

	ivec3 limit;
	int cascade;

	ivec3 region_world_pos;
	int maximum_light_cells;

	ivec3 probe_axis_size;
	int ray_hit_cache_frames;

	ivec3 upper_region_world_pos;
	int occlusion_offset;

	uint history_mode;
	uint rigid_region;
	int upper_cascade;
	uint pad2;
}
params;

vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
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

#include "hddagi_radiance_inc.glsl"

#define FP_BITS 14
#define FP_MAX ((1 << 22) - 1)

uvec3 rgbe_decode_fp(uint p_rgbe, int p_bits) {
	uvec4 rgbe = (uvec4(p_rgbe) >> uvec4(0, 9, 18, 27)) & uvec4(0x1FF, 0x1FF, 0x1FF, 0x1F);
	int shift = int(rgbe.a) - 15 - 9 + p_bits;
	if (shift >= 0) {
		rgbe.rgb <<= uint(shift);
	} else {
		rgbe.rgb >>= uint(-shift);
	}
	return rgbe.rgb;
}

ivec2 probe_to_tex(ivec3 local_probe) {
	ivec3 cell = hddagi_wrap_probe(params.region_world_pos + local_probe, ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0, cell.z * int(params.probe_axis_size.y));
}

ivec2 probe_to_texp(ivec3 local_probe) {
	ivec3 cell = hddagi_wrap_probe(params.upper_region_world_pos + local_probe, ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0, cell.z * int(params.probe_axis_size.y));
}

#ifdef MODE_LIGHT_STORE
#include "hddagi_surface_inc.glsl"
#endif

#ifdef MODE_OCCLUSION_FILTER
float hddagi_occlusion_filter_path(vec3 midpoint, int axis, int cascade) {
	ivec3 base = ivec3(floor(midpoint - 0.5));
	vec3 fraction = fract(midpoint - 0.5);
	float weight = 0.0;
	for (int corner = 0; corner < 8; corner++) {
		ivec3 offset = (ivec3(corner) >> ivec3(0, 1, 2)) & 1;
		ivec3 cell = base + offset;
		if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, params.grid_size)) || cell[axis] < 1 || cell[axis] >= params.grid_size[axis] - 1) {
			continue;
		}
		cell = (cell + params.region_world_pos * PROBE_CELLS) & (params.grid_size - 1);
		cell.y += cascade * params.grid_size.y;
		uint paths = hddagi_read_neighbours(cell);
		vec3 w = mix(vec3(1.0) - fraction, fraction, bvec3(offset));
		weight += float((paths >> (HDDAGI_OCCLUSION_PATH_SHIFT + uint(axis))) & 1u) * w.x * w.y * w.z;
	}
	return weight;
}

void hddagi_occlusion_filter(vec3 posf, int cascade, inout vec4 occ_0, inout vec4 occ_1) {
	ivec3 posi = ivec3(floor(posf));
	vec4 accum_0 = occ_0 * 0.25;
	vec4 accum_1 = occ_1 * 0.25;
	vec4 total_0 = vec4(0.25);
	vec4 total_1 = vec4(0.25);
	vec3 local = mod(posf, float(PROBE_CELLS));
	ivec3 total_size = params.grid_size + 2;
	total_size.y *= params.cascade_count;
	for (int axis = 0; axis < 3; axis++) {
		float distance_to_seam = min(local[axis], float(PROBE_CELLS) - local[axis]);
		float strength = 1.0 - smoothstep(0.5, 2.5, distance_to_seam);
		if (strength <= 0.0) {
			continue;
		}
		int seam = int(floor(posf[axis] / float(PROBE_CELLS) + 0.5));
		// Only channels naming the same physical probe on both sides may blend.
		int parity = (seam + params.region_world_pos[axis]) & 1;
		vec4 mask_0 = vec4(equal((ivec4(0, 1, 2, 3) >> axis) & 1, ivec4(parity))) * strength;
		vec4 mask_1 = vec4(equal((ivec4(4, 5, 6, 7) >> axis) & 1, ivec4(parity))) * strength;
		vec4 samples_0[2];
		vec4 samples_1[2];
		for (int sample_index = 0; sample_index < 2; sample_index++) {
			ivec3 step = ivec3(0);
			step[axis] = sample_index * 2 - 1;
			ivec3 sample_cell = (posi + step * 2 + params.region_world_pos * PROBE_CELLS) & (params.grid_size - 1);
			sample_cell += 1;
			sample_cell.y += (params.grid_size.y + 2) * cascade;
			vec3 uv = (vec3(sample_cell) + fract(posf)) / vec3(total_size);
			samples_0[sample_index] = texture(sampler3D(occlusion[0], linear_sampler), uv.yxz);
			samples_1[sample_index] = texture(sampler3D(occlusion[1], linear_sampler), uv.yxz);
		}
		mask_0 *= smoothstep(vec4(0.25), vec4(0.75), abs(samples_0[1] - samples_0[0]));
		mask_1 *= smoothstep(vec4(0.25), vec4(0.75), abs(samples_1[1] - samples_1[0]));
		if (all(equal(mask_0, vec4(0.0))) && all(equal(mask_1, vec4(0.0)))) {
			continue;
		}
		for (int side = -1; side <= 1; side += 2) {
			ivec3 step = ivec3(0);
			step[axis] = side;
			float path_weight = hddagi_occlusion_filter_path(posf + vec3(step), axis, cascade);
			if (path_weight <= 0.0) {
				continue;
			}
			accum_0 += samples_0[(side + 1) / 2] * mask_0 * path_weight;
			accum_1 += samples_1[(side + 1) / 2] * mask_1 * path_weight;
			total_0 += mask_0 * path_weight;
			total_1 += mask_1 * path_weight;
		}
	}
	occ_0 = accum_0 / total_0;
	occ_1 = accum_1 / total_1;
}

#endif

void main() {
#ifdef MODE_OCCLUSION_FILTER
	ivec3 texel = ivec3(gl_GlobalInvocationID);
	ivec3 half_grid = params.grid_size / 2;
	ivec3 output_size = half_grid + 2;
	if (any(greaterThanEqual(texel, output_size))) {
		return;
	}
	ivec3 world_cell = (texel - 1) & (half_grid - 1);
	ivec3 local_cell = (world_cell - params.region_world_pos * (PROBE_CELLS / 2)) & (half_grid - 1);
	bool partial_update = params.rigid_region != 0u || any(notEqual(params.scroll, ivec3(0)));
	if (partial_update) {
		local_cell = ivec3(gl_GlobalInvocationID) + params.offset;
		if (any(greaterThanEqual(local_cell, params.limit))) {
			return;
		}
		local_cell &= half_grid - 1;
		world_cell = (local_cell + params.region_world_pos * (PROBE_CELLS / 2)) & (half_grid - 1);
		texel = world_cell + 1;
	}
	vec3 posf = hddagi_occlusion_cache_sample_position(local_cell);
	ivec3 occ_pos = (ivec3(posf) + params.region_world_pos * PROBE_CELLS) & (params.grid_size - 1);
	occ_pos += 1;
	occ_pos.y += (params.grid_size.y + 2) * params.cascade;
	ivec3 source_size = params.grid_size + 2;
	source_size.y *= params.cascade_count;
	vec3 uv = (vec3(occ_pos) + fract(posf)) / vec3(source_size);
	vec4 occ_0 = texture(sampler3D(occlusion[0], linear_sampler), uv.yxz);
	vec4 occ_1 = texture(sampler3D(occlusion[1], linear_sampler), uv.yxz);
	vec4 original_0 = occ_0;
	vec4 original_1 = occ_1;
	hddagi_occlusion_filter(posf, params.cascade, occ_0, occ_1);
	texel.y += output_size.y * params.cascade;
	// Cache only the correction. Downsampling absolute visibility also blurs
	// walls away from probe seams and can reintroduce light leaks.
	imageStore(dst_correction_0, texel.yxz, occ_0 - original_0);
	imageStore(dst_correction_1, texel.yxz, occ_1 - original_1);
	if (partial_update && (any(equal(world_cell, ivec3(0))) || any(equal(world_cell, half_grid - 1)))) {
		ivec3 wrap = mix(ivec3(0), half_grid, equal(world_cell, ivec3(0))) - mix(ivec3(0), half_grid, equal(world_cell, half_grid - 1));
		for (int i = 1; i < 8; i++) {
			ivec3 axes = (ivec3(i) >> ivec3(0, 1, 2)) & 1;
			if (any(notEqual(axes * ivec3(notEqual(wrap, ivec3(0))), axes))) {
				continue;
			}
			ivec3 pos = texel + wrap * axes;
			imageStore(dst_correction_0, pos.yxz, occ_0 - original_0);
			imageStore(dst_correction_1, pos.yxz, occ_1 - original_1);
		}
	}
#endif

#ifdef MODE_REGION_STORE

	ivec3 region_mask = (params.grid_size / REGION_SIZE) - 1;

	ivec3 src_pos = ivec3(gl_GlobalInvocationID.xyz) + params.offset;
	ivec3 region_pos = ivec3(gl_LocalInvocationID.xyz);
	ivec3 subregion = region_pos / 4;
	uint subregion_ofs = subregion.z * 4 + subregion.y * 2 + subregion.x;
	ivec3 subregion_pos = region_pos % 4;
	uint subregion_bit = subregion_pos.z * 16 + subregion_pos.y * 4 + subregion_pos.x;

	if (subregion_pos == ivec3(0)) {
		region_bits[subregion_ofs * 2 + 0] = 0;
		region_bits[subregion_ofs * 2 + 1] = 0;
	}

	groupMemoryBarrier();
	barrier();

	uint p = imageLoad(src_normal_bits, src_pos).r;

	if (p != 0) {
		if (subregion_bit < 32) {
			atomicOr(region_bits[subregion_ofs * 2 + 0], 1 << subregion_bit);
		} else {
			atomicOr(region_bits[subregion_ofs * 2 + 1], 1 << (subregion_bit - 32));
		}
	}

	groupMemoryBarrier();
	barrier();

	ivec3 dst_region = ((src_pos / REGION_SIZE) + params.region_world_pos) & region_mask;

	if (subregion_pos == ivec3(0)) {
		ivec3 dst_pos = (dst_region * REGION_SIZE + region_pos) / 4;
		uvec2 bits = uvec2(region_bits[subregion_ofs * 2 + 0], region_bits[subregion_ofs * 2 + 1]);
		dst_pos.y += params.cascade * (params.grid_size.y / 4);
		imageStore(dst_voxels, dst_pos, uvec4(bits, uvec2(0)));
	}

	if (region_pos == ivec3(0)) {
		dst_region.y += params.cascade * (params.grid_size.y / REGION_SIZE);
		uint block_mask = 0u;
		for (uint i = 0u; i < 8u; i++) {
			if ((region_bits[i * 2u] | region_bits[i * 2u + 1u]) != 0u) {
				block_mask |= 1u << i;
			}
		}
		imageStore(dst_region_bits, dst_region, uvec4(block_mask));

		imageStore(region_version, dst_region, uvec4(params.region_version));

		// Store region version
	}

#endif

#ifdef MODE_LIGHT_SCROLL

	int src_index = int(gl_GlobalInvocationID).x;
	int local = int(gl_LocalInvocationID).x;

	bool thread_active = true; // Early return deadlocks Intel, so code must avoid it.

	if (src_index >= src_dispatch_data.total_count) {
		thread_active = false;
	} else if (local == 0) {
		store_position_count = 0; // Base one stores as zero, the others wait
	}

	groupMemoryBarrier();
	barrier();

	bool inside_area = false;
	uint index = 0;
	ivec3 src_pos;

	if (thread_active) {
		src_pos = (ivec3(src_process_voxels.data[src_index].position) >> ivec3(0, 7, 14)) & ivec3(0x7F);
		inside_area = all(greaterThanEqual(src_pos, params.offset)) && all(lessThan(src_pos, params.limit));
		if (params.rigid_region != 0u) {
			inside_area = !inside_area;
		}

		if (!inside_area) {
			ivec3 light_pos = src_pos + params.scroll;
			light_pos = (light_pos + (params.region_world_pos * REGION_SIZE)) & (params.grid_size - 1);
			light_pos.y += params.grid_size.y * params.cascade;

			// As this will be a new area, clear the new region from the old values.
			int voxel_index = hddagi_voxel_index(light_pos);
			if (voxel_index >= 0) {
				hddagi_voxel_data.cells[voxel_index].x = 0u;
			}
		}

		if (inside_area) {
			index = atomicAdd(store_position_count, 1);
		}
	}
	groupMemoryBarrier();
	barrier();

	// global increment only once per group, to reduce pressure

	if (thread_active) {
		if (!inside_area || store_position_count == 0) {
			thread_active = false;
		} else if (index == 0) {
			store_from_index = atomicAdd(dispatch_data.total_count, store_position_count);
			if (store_from_index + store_position_count > uint(params.maximum_light_cells)) {
				atomicMin(dispatch_data.total_count, uint(params.maximum_light_cells));
			}
			uint group_count = (min(store_from_index + store_position_count, uint(params.maximum_light_cells)) + 63u) / 64u;
			atomicMax(dispatch_data.x, group_count);
		}
	}

	groupMemoryBarrier();
	barrier();

	if (thread_active && index + store_from_index < uint(params.maximum_light_cells)) {
		index += store_from_index;

		ivec3 dst_pos = src_pos + params.scroll;

		uint src_pending_bits = src_process_voxels.data[src_index].position & ~uint((1 << 21) - 1);

		dst_process_voxels.data[index].position = uint(dst_pos.x | (dst_pos.y << 7) | (dst_pos.z << 14)) | src_pending_bits;
		dst_process_voxels.data[index].albedo_normal = src_process_voxels.data[src_index].albedo_normal;
		dst_process_voxels.data[index].emission = src_process_voxels.data[src_index].emission;
		dst_source_emission.data[index] = src_source_emission.data[src_index];
		dst_process_voxels.data[index].occlusion = src_process_voxels.data[src_index].occlusion;
	}
#endif

#ifdef MODE_LIGHT_STORE

	ivec3 local = ivec3(gl_LocalInvocationID.xyz);
	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz) + params.offset;

	ivec3 group_origin = params.offset + ivec3(gl_WorkGroupID) * 4;
	for (int i = int(gl_LocalInvocationIndex); i < 216; i += 64) {
		ivec3 sample_pos = group_origin + ivec3(i % 6, (i / 6) % 6, i / 36) - 1;
		uint facing = 0u;
		if (all(greaterThanEqual(sample_pos, ivec3(0))) && all(lessThan(sample_pos, params.grid_size))) {
			facing = imageLoad(src_normal_bits, sample_pos).r;
		}
		normal_facings[i] = facing;
	}
	if (gl_LocalInvocationIndex == 0u) {
		store_position_count = 0;
	}

	bool thread_active = true; // Early return deadlocks Intel, so code must avoid it.

	if (any(greaterThanEqual(pos, params.limit))) {
		// Storing is not a multiple of the workgroup, so invalid threads can happen.
		thread_active = false;
	}

	barrier();

	vec4 albedo_accum = vec4(0.0);
	vec4 emission_accum = vec4(0.0);
	vec3 normal_accum = vec3(0.0);
	uint occlusionu = 0;
	bool voxels_found = false;
	ivec3 base_dst_pos;
	uint neighbour_voxels = 0u;

	if (thread_active) {
		uint solid = get_normal_facing(local);

		voxels_found = hddagi_build_surface(pos, local, albedo_accum, emission_accum, normal_accum, neighbour_voxels);
		base_dst_pos = (pos + params.region_world_pos * REGION_SIZE) & (params.grid_size - 1);
		ivec3 dst_pos = base_dst_pos + ivec3(0, params.grid_size.y * params.cascade, 0);
		int voxel_index = hddagi_voxel_index(dst_pos);
		if (voxel_index >= 0) {
			hddagi_voxel_data.cells[voxel_index].y = neighbour_voxels ^ HDDAGI_EMPTY_NEIGHBOURS;
			if (!voxels_found) {
				hddagi_voxel_data.cells[voxel_index].x = 0u;
			}
		}
		thread_active = solid == 0u && voxel_index >= 0;
	}

	uint index;

	if (thread_active && voxels_found) {
		index = atomicAdd(store_position_count, 1);
	}

	barrier();

	if (thread_active) {
		if (!voxels_found || store_position_count == 0) {
			thread_active = false;
		} else {
			// global increment only once per group, to reduce pressure
			if (thread_active && index == 0) {
				store_from_index = atomicAdd(dispatch_data.total_count, store_position_count);
				if (store_from_index + store_position_count > uint(params.maximum_light_cells)) {
					atomicMin(dispatch_data.total_count, uint(params.maximum_light_cells));
				}
				uint group_count = (min(store_from_index + store_position_count, uint(params.maximum_light_cells)) + 63u) / 64u;
				atomicMax(dispatch_data.x, group_count);
			}
		}
	}

	barrier();

	if (thread_active) {
		// compute occlusion

		ivec3 base_probe = params.region_world_pos + pos / PROBE_CELLS;

		ivec3 occ_pos = base_dst_pos + ivec3(1);
		occ_pos.y += (params.grid_size.y + 2) * params.cascade;

		vec4 occ_0 = texelFetch(sampler3D(occlusion[0], linear_sampler), occ_pos.yxz, 0);
		vec4 occ_1 = texelFetch(sampler3D(occlusion[1], linear_sampler), occ_pos.yxz, 0);

		if (bool(base_probe.x & 1)) {
			occ_0.xyzw = occ_0.yxwz;
			occ_1.xyzw = occ_1.yxwz;
		}

		if (bool(base_probe.y & 1)) {
			occ_0.xyzw = occ_0.zwxy;
			occ_1.xyzw = occ_1.zwxy;
		}

		if (bool(base_probe.z & 1)) {
			vec4 tmp = occ_0;
			occ_0 = occ_1;
			occ_1 = tmp;
		}

		for (int i = 0; i < 8; i++) {
			float visibility = hddagi_occlusion_weight(occ_0, occ_1, i);
			occlusionu |= uint(round(visibility * 15.0)) << (i * 4);
		}

		index += store_from_index;

		if (index < params.maximum_light_cells) {
			ivec3 metadata_pos = base_dst_pos + ivec3(0, params.grid_size.y * params.cascade, 0);
			hddagi_voxel_data.cells[hddagi_voxel_index(metadata_pos)].y = (neighbour_voxels | HDDAGI_LIGHT_CELL_VALID_BIT) ^ HDDAGI_EMPTY_NEIGHBOURS;
			emission_accum.rgb /= emission_accum.a;
			dst_process_voxels.data[index].position = uint(pos.x | (pos.y << 7) | (pos.z << 14)) | PROCESS_STATIC_PENDING_BIT | PROCESS_DYNAMIC_PENDING_BIT;
			uint albedo_norm = hddagi_pack_surface_albedo(albedo_accum, normal_accum);

			dst_process_voxels.data[index].albedo_normal = albedo_norm;
			dst_process_voxels.data[index].emission = rgbe_encode(emission_accum.rgb);
			dst_source_emission.data[index] = rgbe_encode(emission_accum.rgb);

			dst_process_voxels.data[index].occlusion = occlusionu;
		} else {
			ivec3 metadata_pos = base_dst_pos + ivec3(0, params.grid_size.y * params.cascade, 0);
			hddagi_voxel_data.cells[hddagi_voxel_index(metadata_pos)].x = 0u;
		}
	}

#endif

#ifdef MODE_LIGHTPROBE_NEIGHBOURS

	ivec3 probe_from = params.offset / REGION_SIZE;
	ivec3 probe_to = params.limit / REGION_SIZE;

	ivec3 probe = ivec3(gl_GlobalInvocationID.xyz) + probe_from;

	if (any(greaterThan(probe, probe_to))) {
		return;
	}

	uint neighbour_visibility = 0;
	ivec3 occ_bits = (params.region_world_pos + probe) & ivec3(1);
	uint occlusion_layer = 0;
	if (occ_bits.x != 0) {
		occlusion_layer |= 1;
	}
	if (occ_bits.y != 0) {
		occlusion_layer |= 2;
	}
	if (occ_bits.z != 0) {
		occlusion_layer |= 4;
	}

	ivec3 occ_tex_size = params.grid_size + ivec3(2);
	occ_tex_size.y *= params.cascade_count;
	for (int i = 0; i < 6; i++) {
		ivec3 dir = hddagi_axis_direction(uint(i));
		ivec3 probe_next = probe + dir;
		if (any(lessThan(probe_next, ivec3(0))) || any(greaterThanEqual(probe_next, params.probe_axis_size))) {
			continue; // Outside range
		}
		vec3 test_pos = vec3(probe * REGION_SIZE) + vec3(hddagi_axis_direction(uint(i))) * (REGION_SIZE - 1);

		ivec3 occ_pos = ivec3(test_pos);
		occ_pos = (occ_pos + params.region_world_pos * REGION_SIZE) & (params.grid_size - ivec3(1));
		occ_pos += ivec3(1);
		occ_pos.y += (int(params.grid_size.y) + 2) * params.cascade;
		vec3 occ_posf = vec3(occ_pos) + fract(test_pos);
		occ_posf /= vec3(occ_tex_size);
		vec4 occ_0 = texture(sampler3D(occlusion[0], linear_sampler), occ_posf.yxz);
		vec4 occ_1 = texture(sampler3D(occlusion[1], linear_sampler), occ_posf.yxz);

		float visibility = hddagi_occlusion_weight(occ_0, occ_1, int(occlusion_layer));

		neighbour_visibility |= uint(clamp(visibility * 0xF, 0, 0xF)) << (i * 4);
	}

	ivec2 probe_tex_pos = probe_to_tex(probe);
	imageStore(neighbour_probe_visibility, ivec3(probe_tex_pos, params.cascade), uvec4(neighbour_visibility));

#endif

#ifdef MODE_LIGHTPROBE_GEOMETRY_PROXIMITY

	ivec3 probe = ivec3(gl_GlobalInvocationID.xyz);
	if (params.rigid_region != 0u) {
		probe += params.offset / REGION_SIZE;
		if (any(greaterThan(probe, params.limit / REGION_SIZE))) {
			return;
		}
	}

	if (any(greaterThanEqual(probe, params.probe_axis_size))) {
		return;
	}

	ivec3 region_mask = (params.grid_size / REGION_SIZE) - 1;

	bool found_geometry = false;
	for (int i = 0; i < 8; i++) {
		// Check all 8 regions around
		ivec3 offset = ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1)) - ivec3(1);
		ivec3 region = probe + offset;
		if (any(lessThan(region, ivec3(0)))) {
			continue;
		}
		if (any(greaterThanEqual(region, params.probe_axis_size - ivec3(1)))) {
			continue;
		}

		region = (region + params.region_world_pos) & region_mask;

		region.y += params.cascade * (params.grid_size.y / REGION_SIZE);
		if (imageLoad(src_region_bits, region).r > 0) {
			found_geometry = true;
			break;
		}
	}

	// Only mark a probe buried when all eight incident voxels are solid.
	bool buried = found_geometry;
	for (int i = 0; i < 8 && buried; i++) {
		ivec3 cell = probe * REGION_SIZE + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1)) - ivec3(1);
		if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, params.grid_size))) {
			buried = false;
			break;
		}
		cell = (cell + params.region_world_pos * REGION_SIZE) & (params.grid_size - 1);
		ivec3 block = (cell >> 2) + ivec3(0, params.cascade * (params.grid_size.y >> 2), 0);
		uvec2 bits = imageLoad(src_voxel_bits, block).rg;
		ivec3 local = cell & 3;
		uint bit = uint(local.x + local.y * 4 + local.z * 16);
		buried = (bits[bit >> 5u] & (1u << (bit & 31u))) != 0u;
	}
	ivec2 tex_pos = probe_to_tex(probe);
	// 0: empty neighbourhood; 1: geometry nearby; 0.75: buried probe (still near geometry).
	imageStore(dst_probe_geometry_proximity, ivec3(tex_pos, params.cascade), vec4(buried ? 0.75 : (found_geometry ? 1.0 : 0.0)));

#endif

#ifdef MODE_LIGHTPROBE_UPDATE_FRAMES

	ivec3 probe_from = params.offset / REGION_SIZE;
	ivec3 probe_to = params.limit / REGION_SIZE;

	ivec3 probe = ivec3(gl_GlobalInvocationID.xyz) + probe_from;

	if (any(greaterThan(probe, probe_to))) {
		return;
	}

	ivec3 tex_pos = ivec3(probe_to_tex(probe), params.cascade);

	uint frame = imageLoad(lightprobe_frames, tex_pos).r;
	frame &= 0x0FFFFFFF; // Clear update frames counter
	frame |= uint(params.probe_update_frames) << 28; // Reset frames counter.

	imageStore(lightprobe_frames, tex_pos, uvec4(frame));

#endif

#ifdef MODE_OCCLUSION

	int thread = int(gl_LocalInvocationIndex);
	int corner = thread / OCC_PLANE;
	int lane = thread % OCC_PLANE;
	int x = lane % PROBE_CELLS;
	int y = lane / PROBE_CELLS;
	ivec3 workgroup = ivec3(gl_WorkGroupID);
	ivec3 region = params.region_world_pos + params.offset / REGION_SIZE + workgroup;
	ivec3 sign_bits = ((ivec3(corner + params.occlusion_offset) >> ivec3(0, 1, 2)) + region) & 1;
	ivec3 flip = (1 - sign_bits) * (PROBE_CELLS - 1);
	int mirror = flip.x + PROBE_CELLS * flip.y + OCC_PLANE * flip.z;
	int base = corner * OCC_VOLUME;
	ivec3 origin = params.offset + workgroup * PROBE_CELLS;
	for (int p = thread; p < OCC_VOLUME; p += 4 * OCC_PLANE) {
		bit_normals[p] = imageLoad(src_normal_bits, origin + ivec3(p % PROBE_CELLS, p / PROBE_CELLS % PROBE_CELLS, p / OCC_PLANE)).r;
	}
	barrier();

	// Each plane of lanes sweeps one corner in the same coordinate order.
	for (int step = 0; step <= 3 * (PROBE_CELLS - 1); step++) {
		int z = step - x - y;
		if (z >= 0 && z < PROBE_CELLS) {
			int p = lane + z * OCC_PLANE;
			float weight = 0.0;
			if (bit_normals[p ^ mirror] == 0u) {
				float sum = 0.0;
				if (x > 0) {
					sum += occlusion[base + p - 1];
				}
				if (y > 0) {
					sum += occlusion[base + p - PROBE_CELLS];
				}
				if (z > 0) {
					sum += occlusion[base + p - OCC_PLANE];
				}
				weight = step == 0 ? 1.0 : sum / float(int(x > 0) + int(y > 0) + int(z > 0));
			}
			occlusion[base + p] = weight;
		}
		barrier();
	}

	uvec3 toward = uvec3(1u, 4u, 16u) << uvec3(1 - sign_bits);
	uvec3 away = uvec3(1u, 4u, 16u) << uvec3(sign_bits);
	float result[PROBE_CELLS];
	for (int z = 0; z < PROBE_CELLS; z++) {
		int p = lane + z * OCC_PLANE;
		uint facing = bit_normals[p ^ mirror];
		int axes = int(x > 0) | (int(y > 0) << 1) | (int(z > 0) << 2);
		float value = occlusion[base + p];
		if (facing != 0u) {
			axes &= int((facing & toward.x) != 0u) | (int((facing & toward.y) != 0u) << 1) | (int((facing & toward.z) != 0u) << 2);
			float sum = 0.0;
			int count = 0;
			for (int subset = 1; subset < 8; subset++) {
				if ((axes & subset) == subset) {
					int delta = (subset & 1) + ((subset >> 1) & 1) * PROBE_CELLS + ((subset >> 2) & 1) * OCC_PLANE;
					if (bit_normals[(p - delta) ^ mirror] == 0u) {
						sum += occlusion[base + p - delta];
						count++;
					}
				}
			}
			value = sum / float(max(count, 1));
		} else {
			uint nx = x > 0 ? bit_normals[(p - 1) ^ mirror] : 0u;
			uint ny = y > 0 ? bit_normals[(p - PROBE_CELLS) ^ mirror] : 0u;
			uint nz = z > 0 ? bit_normals[(p - OCC_PLANE) ^ mirror] : 0u;
			int count = int(nx != 0u) + int(ny != 0u) + int(nz != 0u);
			int visible = int((nx & away.x) != 0u) + int((ny & away.y) != 0u) + int((nz & away.z) != 0u);
			if (count != 0) {
				value *= float(visible) / float(count);
			}
		}
		result[z] = value;
	}

	barrier();
	for (int z = 0; z < PROBE_CELLS; z++) {
		occlusion[base + ((lane + z * OCC_PLANE) ^ mirror)] = result[z];
	}
	barrier();
	int cell_from = thread * (PROBE_CELLS / 4);
	int cell_to = cell_from + PROBE_CELLS / 4;

	for (int cell_i = cell_from; cell_i < cell_to; cell_i++) {
		ivec3 cell_pos;
		cell_pos.z = cell_i / (PROBE_CELLS * PROBE_CELLS);
		cell_pos.y = (cell_i / PROBE_CELLS) % PROBE_CELLS;
		cell_pos.x = cell_i % PROBE_CELLS;

		ivec3 cascade_pos = cell_pos + params.offset + workgroup * REGION_SIZE; // pos in texture
		cascade_pos = (cascade_pos + params.region_world_pos * REGION_SIZE) & (params.grid_size - ivec3(1));
		cascade_pos += ivec3(1); // Margin

		ivec3 store_pos = cascade_pos;

		store_pos.y += (params.grid_size.y + 2) * params.cascade;

		uint enc_occlusion = 0; // 16 bits RGBA4444

		for (int i = 0; i < 4; i++) {
			float occ = occlusion[i * OCC_VOLUME + cell_i];
			enc_occlusion |= uint(clamp(occ * 0xF, 0, 0xF)) << (i * 4);
		}

#ifdef HDDAGI_RGBA8_OCCLUSION
		vec4 store_value = vec4((uvec4(enc_occlusion) >> uvec4(12, 8, 4, 0)) & 15u) / 15.0;
#else
		uvec4 store_value = uvec4(enc_occlusion);
#endif
		imageStore(dst_occlusion, store_pos.yxz, store_value);

		// Store wrap faces
		if (cascade_pos.x == 1) {
			imageStore(dst_occlusion, (store_pos + ivec3(params.grid_size.x, 0, 0)).yxz, store_value);
		}

		if (cascade_pos.y == 1) {
			imageStore(dst_occlusion, (store_pos + ivec3(0, params.grid_size.y, 0)).yxz, store_value);
		}

		if (cascade_pos.z == 1) {
			imageStore(dst_occlusion, (store_pos + ivec3(0, 0, params.grid_size.z)).yxz, store_value);
		}

		if (cascade_pos.x == params.grid_size.x) {
			imageStore(dst_occlusion, (store_pos - ivec3(params.grid_size.x, 0, 0)).yxz, store_value);
		}

		if (cascade_pos.y == params.grid_size.y) {
			imageStore(dst_occlusion, (store_pos - ivec3(0, params.grid_size.y, 0)).yxz, store_value);
		}

		if (cascade_pos.z == params.grid_size.z) {
			imageStore(dst_occlusion, (store_pos - ivec3(0, 0, params.grid_size.z)).yxz, store_value);
		}

		// Store wrap edges
		if (cascade_pos.xy == ivec2(1)) {
			imageStore(dst_occlusion, (store_pos + ivec3(params.grid_size.xy, 0)).yxz, store_value);
		}

		if (cascade_pos.xz == ivec2(1)) {
			imageStore(dst_occlusion, (store_pos + ivec3(params.grid_size.x, 0, params.grid_size.z)).yxz, store_value);
		}

		if (cascade_pos.yz == ivec2(1)) {
			imageStore(dst_occlusion, (store_pos + ivec3(0, params.grid_size.yz)).yxz, store_value);
		}

		if (cascade_pos.xy == params.grid_size.xy) {
			imageStore(dst_occlusion, (store_pos - ivec3(params.grid_size.xy, 0)).yxz, store_value);
		}

		if (cascade_pos.xz == params.grid_size.xz) {
			imageStore(dst_occlusion, (store_pos - ivec3(params.grid_size.x, 0, params.grid_size.z)).yxz, store_value);
		}

		if (cascade_pos.yz == params.grid_size.yz) {
			imageStore(dst_occlusion, (store_pos - ivec3(0, params.grid_size.yz)).yxz, store_value);
		}

		if (cascade_pos.xy == ivec2(1, params.grid_size.y)) {
			imageStore(dst_occlusion, (store_pos + ivec3(params.grid_size.x, -params.grid_size.y, 0)).yxz, store_value);
		}

		if (cascade_pos.xy == ivec2(params.grid_size.x, 1)) {
			imageStore(dst_occlusion, (store_pos + ivec3(-params.grid_size.x, params.grid_size.y, 0)).yxz, store_value);
		}

		if (cascade_pos.xz == ivec2(1, params.grid_size.z)) {
			imageStore(dst_occlusion, (store_pos + ivec3(params.grid_size.x, 0, -params.grid_size.z)).yxz, store_value);
		}

		if (cascade_pos.xz == ivec2(params.grid_size.x, 1)) {
			imageStore(dst_occlusion, (store_pos + ivec3(-params.grid_size.x, 0, params.grid_size.z)).yxz, store_value);
		}

		if (cascade_pos.yz == ivec2(1, params.grid_size.z)) {
			imageStore(dst_occlusion, (store_pos + ivec3(0, params.grid_size.y, -params.grid_size.z)).yxz, store_value);
		}

		if (cascade_pos.yz == ivec2(params.grid_size.y, 1)) {
			imageStore(dst_occlusion, (store_pos + ivec3(0, -params.grid_size.y, params.grid_size.z)).yxz, store_value);
		}

		// Store wrap vertices
		for (int i = 0; i < 8; i++) {
			bvec3 bits = bvec3((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec3 test_pos = mix(ivec3(1), params.grid_size, bits);
			ivec3 store_offset = mix(+params.grid_size, -params.grid_size, bits);
			if (cascade_pos == test_pos) {
				imageStore(dst_occlusion, (store_pos + store_offset).yxz, store_value);
			}
		}
	}

#endif

#ifdef MODE_LIGHTPROBE_SCROLL

	ivec2 local_pos = ivec2(gl_LocalInvocationID).xy;

	vec3 specular_light = vec3(0.0);
	vec3 diffuse_light = vec3(0.0);
	vec3 ambient_light = vec3(0.0);

	ivec3 probe_from = params.offset / PROBE_CELLS + ivec3(gl_WorkGroupID);
	ivec3 upper_max = params.probe_axis_size - 1;
	ivec3 upper_half_probe = probe_from + params.region_world_pos - 2 * params.upper_region_world_pos + upper_max / 2;
	if (params.upper_cascade >= 0 && all(greaterThanEqual(upper_half_probe, ivec3(0))) && all(lessThanEqual(upper_half_probe, 2 * upper_max))) {
		ivec3 base_probe = upper_half_probe / 2;
		ivec3 upper_cell = upper_half_probe * (PROBE_CELLS / 2);

		if (local_pos == ivec2(0)) {
			ivec3 occ_pos = min(upper_cell, params.grid_size - 1);
			occ_pos = (occ_pos + params.upper_region_world_pos * REGION_SIZE) & (params.grid_size - ivec3(1));
			occ_pos += ivec3(1);
			occ_pos.y += (int(params.grid_size.y) + 2) * params.upper_cascade;
			ivec3 occ_tex_size = params.grid_size + ivec3(2);
			occ_tex_size.y *= params.cascade_count;
			vec3 occ_posf = vec3(occ_pos) / vec3(occ_tex_size);
			vec4 occ_0 = texture(sampler3D(occlusion[0], linear_sampler), occ_posf.yxz);
			vec4 occ_1 = texture(sampler3D(occlusion[1], linear_sampler), occ_posf.yxz);
			float occlusion_total = 0.0;

			for (int i = 0; i < 8; i++) {
				ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));

				vec3 probe_pos = vec3(probe * PROBE_CELLS);

				vec3 probe_to_pos = vec3(upper_cell) - probe_pos;

				ivec3 probe_occ = (probe + params.upper_region_world_pos) & ivec3(1);
				uint weight_index = 0;
				if (probe_occ.x != 0) {
					weight_index |= 1;
				}
				if (probe_occ.y != 0) {
					weight_index |= 2;
				}
				if (probe_occ.z != 0) {
					weight_index |= 4;
				}

				float weight = hddagi_occlusion_weight(occ_0, occ_1, int(weight_index));

				weight = max(0.000001, weight); // make sure not zero (only trilinear can be zero)

				vec3 trilinear = vec3(1.0) - abs(probe_to_pos / float(PROBE_CELLS));

				weight *= trilinear.x * trilinear.y * trilinear.z;
				occlusion_blend[i] = weight;
				occlusion_total += weight;
			}

			for (int i = 0; i < 8; i++) {
				if (occlusion_total == 0.0) {
					occlusion_blend[i] = 0;
				} else {
					occlusion_blend[i] /= occlusion_total;
				}
			}
		}

		groupMemoryBarrier();
		barrier();

		for (int i = 0; i < 8; i++) {
			if (occlusion_blend[i] == 0.0) {
				continue;
			}
			ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec2 base_tex_pos = probe_to_texp(probe);
			ivec2 tex_pos = base_tex_pos * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1) + local_pos;
			ivec3 tex_array_pos = ivec3(tex_pos, params.upper_cascade);
			specular_light += rgbe_decode(imageLoad(lightprobe_specular_data, tex_array_pos).r) * occlusion_blend[i];
			diffuse_light += rgbe_decode(imageLoad(lightprobe_diffuse_data, tex_array_pos).r) * occlusion_blend[i];

			if (local_pos == ivec2(0)) {
				tex_array_pos = ivec3(base_tex_pos, params.upper_cascade);
				ambient_light += imageLoad(lightprobe_ambient_tex, tex_array_pos).rgb * occlusion_blend[i];
			}
		}
	}

	ivec2 probe_tex_pos = probe_to_tex(probe_from);

	ivec3 dst_tex_uv = ivec3(probe_tex_pos * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1), params.cascade);

	ivec3 copy_to[4] = ivec3[](ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));
	copy_to[0] = dst_tex_uv + ivec3(local_pos, 0);

	if (local_pos == ivec2(0, 0)) {
		copy_to[1] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - 1, -1, 0);
		copy_to[2] = dst_tex_uv + ivec3(-1, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, 0)) {
		copy_to[1] = dst_tex_uv + ivec3(0, -1, 0);
		copy_to[2] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = dst_tex_uv + ivec3(-1, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(0, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = dst_tex_uv + ivec3(-1, 0, 0);
		copy_to[2] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[3] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, -1, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = dst_tex_uv + ivec3(0, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[2] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, 0, 0);
		copy_to[3] = dst_tex_uv + ivec3(-1, -1, 0);
	} else if (local_pos.y == 0) {
		copy_to[1] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y - 1, 0);
	} else if (local_pos.x == 0) {
		copy_to[1] = dst_tex_uv + ivec3(local_pos.x - 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	} else if (local_pos.y == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y + 1, 0);
	} else if (local_pos.x == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = dst_tex_uv + ivec3(local_pos.x + 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	}

	uint specular_rgbe = rgbe_encode(specular_light);
	uint diffuse_rgbe = rgbe_encode(diffuse_light);

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(lightprobe_specular_data, copy_to[i], uvec4(specular_rgbe));
		imageStore(lightprobe_diffuse_data, copy_to[i], uvec4(diffuse_rgbe));
	}

	if (local_pos == ivec2(0)) {
		imageStore(lightprobe_ambient_tex, ivec3(probe_tex_pos, params.cascade), vec4(ambient_light, 1));
	}

	// Cache and history invalidation
	probe_tex_pos = probe_tex_pos * LIGHTPROBE_OCT_SIZE + local_pos;

	for (int i = 0; i < params.ray_hit_cache_frames; i++) {
		ivec3 history_pos = ivec3(probe_tex_pos, params.cascade * params.ray_hit_cache_frames + i);
		// Completely invalidate cache frame.
		imageStore(ray_hit_cache, history_pos, uvec4(0));
		if (params.history_mode == 0u) {
			imageStore(lightprobe_moving_average_history, history_pos, uvec4(specular_rgbe));
		}
	}

	if (params.history_mode == 0u) {
		uvec3 moving_average = rgbe_decode_fp(specular_rgbe, FP_BITS);
		moving_average *= params.ray_hit_cache_frames;

		ivec3 ma_pos = ivec3(probe_tex_pos, params.cascade);
		ma_pos.x *= 3;

		imageStore(lightprobe_moving_average, ma_pos + ivec3(0, 0, 0), uvec4(moving_average.r));
		imageStore(lightprobe_moving_average, ma_pos + ivec3(1, 0, 0), uvec4(moving_average.g));
		imageStore(lightprobe_moving_average, ma_pos + ivec3(2, 0, 0), uvec4(moving_average.b));
	} else {
		imageStore(lightprobe_moving_average_history, ivec3(probe_tex_pos, params.cascade), uvec4(specular_rgbe));
	}

#endif
}
