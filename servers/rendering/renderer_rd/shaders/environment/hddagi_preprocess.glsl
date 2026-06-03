#[compute]

#version 450

#VERSION_DEFINES

#define REGION_SIZE 8

#define square(m) ((m) * (m))

#ifdef MODE_REGION_STORE

layout(local_size_x = REGION_SIZE, local_size_y = REGION_SIZE, local_size_z = REGION_SIZE) in;

layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_normal_bits;
layout(rg32ui, set = 0, binding = 2) uniform restrict writeonly uimage3D dst_voxels;
layout(r8ui, set = 0, binding = 3) uniform restrict writeonly uimage3D dst_region_bits;
layout(r16ui, set = 0, binding = 4) uniform restrict writeonly uimage3D region_version;

shared uint solid_bit_count;
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

layout(r8ui, set = 0, binding = 10) uniform restrict writeonly uimage3D dst_disocclusion;

layout(r32ui, set = 0, binding = 11) uniform restrict writeonly uimage3D voxel_neighbours;
layout(r32ui, set = 0, binding = 12) uniform restrict uimage3D light_tex;

shared uint normal_facings[6 * 6 * 6];

uint get_normal_facing(ivec3 p_pos) {
	p_pos += ivec3(1);
	return normal_facings[p_pos.z * 6 * 6 + p_pos.y * 6 + p_pos.x];
}

void set_normal_facing(ivec3 p_pos, uint p_facing) {
	p_pos += ivec3(1);
	normal_facings[p_pos.z * 6 * 6 + p_pos.y * 6 + p_pos.x] = p_facing;
}

#endif

#if defined(MODE_LIGHT_STORE) || defined(MODE_LIGHT_SCROLL)

struct ProcessVoxel {
	uint position; // xyz 10 bit packed - then 2 extra bits for dynamic and static pending
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

shared uint store_position_count;
shared uint store_from_index;

#endif

#if defined(MODE_LIGHT_SCROLL)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 1) uniform restrict uimage3D light_tex;

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

#endif

// Some occlusion defines

#ifdef HIGH_DENSITY_PROBES
#define PROBE_CELLS 4
#else
#define PROBE_CELLS 8
#endif

#define OCC_DISTANCE_MAX 16.0

#ifdef MODE_OCCLUSION

#define REGION_THREADS (REGION_SIZE * REGION_SIZE)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_normal_bits;
layout(r16ui, set = 0, binding = 4) uniform restrict writeonly uimage3D dst_occlusion;

shared float occlusion[REGION_SIZE * REGION_SIZE * REGION_SIZE * 4]; // 4096
shared uint bit_normals[REGION_SIZE * REGION_SIZE * REGION_SIZE]; // 2048
shared uint solid_cell_list[REGION_SIZE * REGION_SIZE * REGION_SIZE]; // 2048
shared uint solid_cell_count;

void set_occlusion(ivec3 p_pos, int p_layer, float p_occlusion) {
	int occ_pos = p_pos.z * (REGION_SIZE * REGION_SIZE) + p_pos.y * REGION_SIZE + p_pos.x;
	occlusion[occ_pos * 4 + p_layer] = p_occlusion;
}

float get_occlusion(ivec3 p_pos, int p_layer) {
	int occ_pos = p_pos.z * (REGION_SIZE * REGION_SIZE) + p_pos.y * REGION_SIZE + p_pos.x;
	return occlusion[occ_pos * 4 + p_layer];
}

void set_bit_normal(ivec3 p_pos, uint p_normal) {
	int occ_pos = p_pos.z * (REGION_SIZE * REGION_SIZE) + p_pos.y * REGION_SIZE + p_pos.x;
	bit_normals[occ_pos] = p_normal;
}

uint get_bit_normal(ivec3 p_pos) {
	int occ_pos = p_pos.z * (REGION_SIZE * REGION_SIZE) + p_pos.y * REGION_SIZE + p_pos.x;
	return bit_normals[occ_pos];
}

ivec3 offset_to_pos(int p_offset) {
	return ivec3(p_offset % REGION_SIZE, (p_offset / REGION_SIZE) % REGION_SIZE, p_offset / (REGION_SIZE * REGION_SIZE));
}

const uvec2 group_size_offset[11] = uvec2[](uvec2(1, 0), uvec2(3, 1), uvec2(6, 4), uvec2(10, 10), uvec2(15, 20), uvec2(21, 35), uvec2(28, 56), uvec2(36, 84), uvec2(42, 120), uvec2(46, 162), uvec2(48, 208));
const uint group_pos[256] = uint[](0,
		65536, 256, 1,
		131072, 65792, 512, 65537, 257, 2,
		196608, 131328, 66048, 768, 131073, 65793, 513, 65538, 258, 3,
		262144, 196864, 131584, 66304, 1024, 196609, 131329, 66049, 769, 131074, 65794, 514, 65539, 259, 4,
		327680, 262400, 197120, 131840, 66560, 1280, 262145, 196865, 131585, 66305, 1025, 196610, 131330, 66050, 770, 131075, 65795, 515, 65540, 260, 5,
		393216, 327936, 262656, 197376, 132096, 66816, 1536, 327681, 262401, 197121, 131841, 66561, 1281, 262146, 196866, 131586, 66306, 1026, 196611, 131331, 66051, 771, 131076, 65796, 516, 65541, 261, 6,
		458752, 393472, 328192, 262912, 197632, 132352, 67072, 1792, 393217, 327937, 262657, 197377, 132097, 66817, 1537, 327682, 262402, 197122, 131842, 66562, 1282, 262147, 196867, 131587, 66307, 1027, 196612, 131332, 66052, 772, 131077, 65797, 517, 65542, 262, 7,
		459008, 393728, 328448, 263168, 197888, 132608, 67328, 458753, 393473, 328193, 262913, 197633, 132353, 67073, 1793, 393218, 327938, 262658, 197378, 132098, 66818, 1538, 327683, 262403, 197123, 131843, 66563, 1283, 262148, 196868, 131588, 66308, 1028, 196613, 131333, 66053, 773, 131078, 65798, 518, 65543, 263,
		459264, 393984, 328704, 263424, 198144, 132864, 459009, 393729, 328449, 263169, 197889, 132609, 67329, 458754, 393474, 328194, 262914, 197634, 132354, 67074, 1794, 393219, 327939, 262659, 197379, 132099, 66819, 1539, 327684, 262404, 197124, 131844, 66564, 1284, 262149, 196869, 131589, 66309, 1029, 196614, 131334, 66054, 774, 131079, 65799, 519,
		459520, 394240, 328960, 263680, 198400, 459265, 393985, 328705, 263425, 198145, 132865, 459010, 393730, 328450, 263170, 197890, 132610, 67330, 458755, 393475, 328195, 262915, 197635, 132355, 67075, 1795, 393220, 327940, 262660, 197380, 132100, 66820, 1540, 327685, 262405, 197125, 131845, 66565, 1285, 262150, 196870, 131590, 66310, 1030, 196615, 131335, 66055, 775);
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

ivec3 modi(ivec3 value, ivec3 p_y) {
	// GLSL Specification says:
	// "Results are undefined if one or both operands are negative."
	// So..
	return mix(value % p_y, p_y - ((abs(value) - ivec3(1)) % p_y) - 1, lessThan(sign(value), ivec3(0)));
}

ivec2 probe_to_tex(ivec3 local_probe) {
	ivec3 cell = modi(params.region_world_pos + local_probe, ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0, cell.z * int(params.probe_axis_size.y));
}

ivec2 probe_to_texp(ivec3 local_probe) {
	ivec3 cell = modi(params.upper_region_world_pos + local_probe, ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0, cell.z * int(params.probe_axis_size.y));
}

void main() {
#ifdef MODE_REGION_STORE

	ivec3 region_mask = (params.grid_size / REGION_SIZE) - 1;

	ivec3 src_pos = ivec3(gl_GlobalInvocationID.xyz) + params.offset;
	ivec3 region_pos = ivec3(gl_LocalInvocationID.xyz);
	ivec3 subregion = region_pos / 4;
	uint subregion_ofs = subregion.z * 4 + subregion.y * 2 + subregion.x;
	ivec3 subregion_pos = region_pos % 4;
	uint subregion_bit = subregion_pos.z * 16 + subregion_pos.y * 4 + subregion_pos.x;

	if (region_pos == ivec3(0)) {
		solid_bit_count = 0;
	}

	if (subregion_pos == ivec3(0)) {
		region_bits[subregion_ofs * 2 + 0] = 0;
		region_bits[subregion_ofs * 2 + 1] = 0;
	}

	groupMemoryBarrier();
	barrier();

	uint p = imageLoad(src_normal_bits, src_pos).r;

	if (p != 0) {
		atomicAdd(solid_bit_count, 1);
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
		imageStore(dst_region_bits, dst_region, uvec4(solid_bit_count > 0 ? 1 : 0));

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
	} else if (src_index >= params.maximum_light_cells) {
		thread_active = false;
	} else if (local == 0) {
		store_position_count = 0; // Base one stores as zero, the others wait
		if (src_index == 0) {
			// This lone thread clears y and z.
			dispatch_data.y = 1;
			dispatch_data.z = 1;
		}
	}

	groupMemoryBarrier();
	barrier();

	bool inside_area = false;
	uint index = 0;
	ivec3 src_pos;

	if (thread_active) {
		src_pos = (ivec3(src_process_voxels.data[src_index].position) >> ivec3(0, 7, 14)) & ivec3(0x7F);
		inside_area = all(greaterThanEqual(src_pos, params.offset)) && all(lessThan(src_pos, params.limit));

		if (!inside_area) {
			ivec3 light_pos = src_pos + params.scroll;
			light_pos = (light_pos + (params.region_world_pos * REGION_SIZE)) & (params.grid_size - 1);
			light_pos.y += params.grid_size.y * params.cascade;

			// As this will be a new area, clear the new region from the old values.
			imageStore(light_tex, light_pos, uvec4(0));
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
			uint group_count = (store_from_index + store_position_count - 1) / 64 + 1;
			atomicMax(dispatch_data.x, group_count);
		}
	}

	groupMemoryBarrier();
	barrier();

	if (thread_active) {
		index += store_from_index;

		ivec3 dst_pos = src_pos + params.scroll;

		uint src_pending_bits = src_process_voxels.data[src_index].position & ~uint((1 << 21) - 1);

		dst_process_voxels.data[index].position = uint(dst_pos.x | (dst_pos.y << 7) | (dst_pos.z << 14)) | src_pending_bits;
		dst_process_voxels.data[index].albedo_normal = src_process_voxels.data[src_index].albedo_normal;
		dst_process_voxels.data[index].emission = src_process_voxels.data[src_index].emission;
		dst_process_voxels.data[index].occlusion = src_process_voxels.data[src_index].occlusion;
	}
#endif

#ifdef MODE_LIGHT_STORE

	ivec3 local = ivec3(gl_LocalInvocationID.xyz);
	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz) + params.offset;

	{ // Fill the local normal facing pool.
		ivec3 load_from = local * 6 / 4;
		ivec3 load_to = (local + ivec3(1)) * 6 / 4;
		ivec3 base_pos = params.offset + ivec3(gl_WorkGroupID.xyz) * 4;

		for (int i = load_from.x; i < load_to.x; i++) {
			for (int j = load_from.y; j < load_to.y; j++) {
				for (int k = load_from.z; k < load_to.z; k++) {
					ivec3 i_ofs = ivec3(i, j, k) - ivec3(1);
					ivec3 load_pos = base_pos + i_ofs;
					uint solid = 0;
					if (all(greaterThanEqual(load_pos, ivec3(0))) && all(lessThan(load_pos, params.grid_size))) {
						solid = imageLoad(src_normal_bits, load_pos).r;
					}
					set_normal_facing(i_ofs, solid);
				}
			}
		}
	}

	bool thread_active = true; // Early return deadlocks Intel, so code must avoid it.

	if (any(greaterThanEqual(pos, params.limit))) {
		// Storing is not a multiple of the workgroup, so invalid threads can happen.
		thread_active = false;
	}

	groupMemoryBarrier();
	barrier();

	vec4 albedo_accum = vec4(0.0);
	vec4 emission_accum = vec4(0.0);
	vec3 normal_accum = vec3(0.0);
	uint occlusionu = 0;
	bool voxels_found = false;
	ivec3 base_dst_pos;

	if (thread_active) {
		uint solid = get_normal_facing(local);

		if (local == ivec3(0)) {
			store_position_count = 0; // Base one stores as zero, the others wait
			if (pos == params.offset) {
				// This lone thread clears y and z.
				dispatch_data.y = 1;
				dispatch_data.z = 1;
			}
		}

		//opposite to aniso dir
		const ivec3 offsets[6] = ivec3[](
				ivec3(1, 0, 0),
				ivec3(-1, 0, 0),
				ivec3(0, 1, 0),
				ivec3(0, -1, 0),
				ivec3(0, 0, 1),
				ivec3(0, 0, -1));

		const vec3 aniso_dir[6] = vec3[](
				vec3(-1, 0, 0),
				vec3(1, 0, 0),
				vec3(0, -1, 0),
				vec3(0, 1, 0),
				vec3(0, 0, -1),
				vec3(0, 0, 1));

		// aniso dir in bitform
		const uint aniso_mask[6] = uint[](
				(1 << 0),
				(1 << 1),
				(1 << 2),
				(1 << 3),
				(1 << 4),
				(1 << 5));

		const uint aniso_offset_mask[6] = uint[](
				(1 << 1),
				(1 << 0),
				(1 << 3),
				(1 << 2),
				(1 << 5),
				(1 << 4));

		uint disocclusion = 0;

		const int facing_direction_count = 26;
		const vec3 facing_directions[26] = vec3[](vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0), vec3(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, -0.7071067811865475, 0.0), vec3(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(-0.7071067811865475, 0.0, -0.7071067811865475), vec3(-0.7071067811865475, 0.0, 0.7071067811865475), vec3(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, 0.7071067811865475, 0.0), vec3(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258), vec3(0.0, -0.7071067811865475, -0.7071067811865475), vec3(0.0, -0.7071067811865475, 0.7071067811865475), vec3(0.0, 0.7071067811865475, -0.7071067811865475), vec3(0.0, 0.7071067811865475, 0.7071067811865475), vec3(0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, -0.7071067811865475, 0.0), vec3(0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(0.7071067811865475, 0.0, -0.7071067811865475), vec3(0.7071067811865475, 0.0, 0.7071067811865475), vec3(0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, 0.7071067811865475, 0.0), vec3(0.5773502691896258, 0.5773502691896258, 0.5773502691896258));

		bool use_for_filter = false;

		for (int i = 0; i < 6; i++) {
			uint n = get_normal_facing(local + offsets[i]);
			if (n == 0) {
				disocclusion |= aniso_offset_mask[i];
			} else if (solid == 0) {
				use_for_filter = true;
			}

			if (solid != 0 || !bool(n & aniso_mask[i])) {
				// Not solid, continue.
				continue;
			}

			voxels_found = true;

			for (int j = 0; j < facing_direction_count; j++) {
				if (bool(n & uint((1 << (j + 6))))) {
					normal_accum += facing_directions[j];
				}
			}

			ivec3 ofs = pos + offsets[i];
			//normal_accum += aniso_dir[i];

			ivec3 albedo_ofs = ofs >> 1;
			albedo_ofs.z *= 6;
			albedo_ofs.z += i;

			uint a = imageLoad(src_albedo, albedo_ofs).r;
			albedo_accum += vec4(vec3((ivec3(a) >> ivec3(0, 5, 11)) & ivec3(0x1f, 0x3f, 0x1f)) / vec3(31.0, 63.0, 31.0), 1.0);

			uint rgbe = imageLoad(src_emission, ofs >> 1).r;

			vec3 emission = rgbe_decode(rgbe);

			uint rgbe_aniso = imageLoad(src_emission_aniso, ofs >> 1).r;
			float strength = ((rgbe_aniso >> (i * 5)) & 0x1F) / float(0x1F);
			emission_accum += vec4(emission * strength, 1.0);
		}

		base_dst_pos = (pos + params.region_world_pos * REGION_SIZE) & (params.grid_size - 1);
		ivec3 dst_pos = base_dst_pos + params.grid_size.y * params.cascade;
		imageStore(dst_disocclusion, dst_pos, uvec4(disocclusion));

		if (solid != 0) {
			thread_active = false; // No further use for this, this is a solid voxel.
		} else if (use_for_filter) {
			uint neighbour_voxels = 0;

			for (int i = 0; i < facing_direction_count; i++) {
				ivec3 neighbor = ivec3(sign(facing_directions[i]));
				ivec3 neighbour_pos = local + neighbor;
				uint n = get_normal_facing(neighbour_pos);
				if (n == 0) {
					continue; // Nothing here
				}

				for (int j = 0; j < 6; j++) {
					//if (!bool(n&(1<<j))) {
					//	continue; // Nothing here either.
					//}
					ivec3 neighbour_neighbour = neighbour_pos + ivec3(aniso_dir[j]);
					ivec3 nn_rel = neighbour_neighbour - local;

					if (any(lessThan(nn_rel, -ivec3(1))) || any(greaterThan(nn_rel, +ivec3(1)))) {
						continue; // Too far away, ignore.
					}

					if (nn_rel == ivec3(0)) {
						continue; // Point to itself, ignore.
					}

					uint q = get_normal_facing(local + nn_rel);
					if (q != 0) {
						continue; // Points to a solid block (can happen), Ignore.
					}

					ivec3 nn_rel_abs = abs(nn_rel);

					int nn_steps = nn_rel_abs.x + nn_rel_abs.y + nn_rel_abs.z;
					if (nn_steps == 3) {
						continue;
					}
					if (nn_steps > 1) {
						// must make sure we are not occluded towards this
						ivec3 test_dirs[3] = ivec3[](ivec3(nn_rel.x, 0, 0), ivec3(0, nn_rel.y, 0), ivec3(0, 0, nn_rel.z));
						int occlusions = 0;
						for (int k = 0; k < 3; k++) {
							if (test_dirs[k] == ivec3(0)) {
								continue; // Direction not used
							}

							q = get_normal_facing(local + test_dirs[k]);
							if (q != 0) {
								occlusions++;
							}
						}

						if (occlusions >= 2) {
							continue; // Occluded from here, ignore. May be unoccluded from another neighbor.
						}
					}

					const uint reverse_map[27] = uint[](6, 14, 18, 9, 4, 21, 11, 16, 23, 7, 2, 19, 0, 0, 1, 12, 3, 24, 8, 15, 20, 10, 5, 22, 13, 17, 25);
					ivec3 abs_pos = nn_rel + ivec3(1);
					// All good, this is a valid neighbor!
					neighbour_voxels |= 1 << reverse_map[abs_pos.z * 3 * 3 + abs_pos.y * 3 + abs_pos.x];
				}
			}

			ivec3 store_pos = (pos + params.region_world_pos * REGION_SIZE) & (params.grid_size - ivec3(1));
			store_pos.y += params.grid_size.y * params.cascade;
			imageStore(voxel_neighbours, store_pos, uvec4(neighbour_voxels));
			if (!voxels_found) {
				// Light voxels won't be stored here, but still ensure this is black to avoid light leaking from outside.
				imageStore(light_tex, store_pos, uvec4(0));
			}
		}
	}

	groupMemoryBarrier();
	barrier();

	uint index;

	if (thread_active && voxels_found) {
		index = atomicAdd(store_position_count, 1);
	}

	groupMemoryBarrier();
	barrier();

	if (thread_active) {
		if (!voxels_found || store_position_count == 0) {
			thread_active = false;
		} else {
			// global increment only once per group, to reduce pressure
			if (thread_active && index == 0) {
				store_from_index = atomicAdd(dispatch_data.total_count, store_position_count);
				uint group_count = (store_from_index + store_position_count - 1) / 64 + 1;
				atomicMax(dispatch_data.x, group_count);
			}
		}
	}

	groupMemoryBarrier();
	barrier();

	if (thread_active) {
		// compute occlusion

		ivec3 base_probe = params.region_world_pos + pos / PROBE_CELLS;

		ivec3 occ_pos = base_dst_pos + ivec3(1);
		occ_pos.y += (params.grid_size.y + 2) * params.cascade;

		vec4 occ_0 = texelFetch(sampler3D(occlusion[0], linear_sampler), occ_pos, 0);
		vec4 occ_1 = texelFetch(sampler3D(occlusion[1], linear_sampler), occ_pos, 0);

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

		float total_weight = dot(occ_0, vec4(1)) + dot(occ_1, vec4(1));
		if (total_weight > 0.0) {
			occ_0 /= total_weight;
			occ_1 /= total_weight;
		}

		float weights[8] = float[](occ_0.x, occ_0.y, occ_0.z, occ_0.w, occ_1.x, occ_1.y, occ_1.z, occ_1.w);

		for (int i = 0; i < 8; i++) {
			float w;
			if (total_weight > 0.0) {
				w = weights[i];
				w *= 15.0;
			} else {
				w = 0;
			}
			occlusionu |= uint(clamp(w, 0.0, 15.0)) << (i * 4);
		}

		index += store_from_index;

		if (index < params.maximum_light_cells) {
			normal_accum = normalize(normal_accum);
			albedo_accum.rgb /= albedo_accum.a;
			emission_accum.rgb /= emission_accum.a;

			dst_process_voxels.data[index].position = uint(pos.x | (pos.y << 7) | (pos.z << 14)) | PROCESS_STATIC_PENDING_BIT | PROCESS_DYNAMIC_PENDING_BIT;

			uint albedo_norm = 0;
			albedo_norm |= clamp(uint(albedo_accum.r * 31.0), 0, 31) << 0;
			albedo_norm |= clamp(uint(albedo_accum.g * 63.0), 0, 63) << 5;
			albedo_norm |= clamp(uint(albedo_accum.b * 31.0), 0, 31) << 11;

			vec2 octa_normal = octahedron_encode(normal_accum);
			uvec2 octa_unormal = clamp(uvec2(octa_normal * 255), uvec2(0), uvec2(255));
			albedo_norm |= (octa_unormal.x << 16) | (octa_unormal.y << 24);

			dst_process_voxels.data[index].albedo_normal = albedo_norm;
			dst_process_voxels.data[index].emission = rgbe_encode(emission_accum.rgb);

			dst_process_voxels.data[index].occlusion = occlusionu;
		}
	}

	// Compute probe neighbors

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
		const vec3 aniso_dir[6] = vec3[](
				vec3(-1, 0, 0),
				vec3(1, 0, 0),
				vec3(0, -1, 0),
				vec3(0, 1, 0),
				vec3(0, 0, -1),
				vec3(0, 0, 1));

		ivec3 dir = ivec3(aniso_dir[i]);
		ivec3 probe_next = probe + dir;
		if (any(lessThan(probe_next, ivec3(0))) || any(greaterThanEqual(probe_next, params.probe_axis_size))) {
			continue; // Outside range
		}
		vec3 test_pos = vec3(probe * REGION_SIZE) + aniso_dir[i] * (REGION_SIZE - 1);

		ivec3 occ_pos = ivec3(test_pos);
		occ_pos = (occ_pos + params.region_world_pos * REGION_SIZE) & (params.grid_size - ivec3(1));
		occ_pos += ivec3(1);
		occ_pos.y += (int(params.grid_size.y) + 2) * params.cascade;
		vec3 occ_posf = vec3(occ_pos) + fract(test_pos);
		occ_posf /= vec3(occ_tex_size);
		vec4 occ_0 = texture(sampler3D(occlusion[0], linear_sampler), occ_posf);
		vec4 occ_1 = texture(sampler3D(occlusion[1], linear_sampler), occ_posf);
		float occ_weights[8] = float[](occ_0.x, occ_0.y, occ_0.z, occ_0.w, occ_1.x, occ_1.y, occ_1.z, occ_1.w);

		float visibility = occ_weights[occlusion_layer];

		neighbour_visibility |= uint(clamp(visibility * 0xF, 0, 0xF)) << (i * 4);
	}

	ivec2 probe_tex_pos = probe_to_tex(probe);
	imageStore(neighbour_probe_visibility, ivec3(probe_tex_pos, params.cascade), uvec4(neighbour_visibility));

#endif

#ifdef MODE_LIGHTPROBE_GEOMETRY_PROXIMITY

	ivec3 probe = ivec3(gl_GlobalInvocationID.xyz);

	if (any(greaterThanEqual(probe, params.probe_axis_size))) {
		return;
	}

	ivec3 region_mask = (params.grid_size / REGION_SIZE) - 1;

	bool found_geometry = false;
	for (int i = 0; i < 8; i++) {
		// Check all 8 regions around
		ivec3 offset = ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1)) * ivec3(2) - ivec3(1);
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

	ivec2 tex_pos = probe_to_tex(probe);
	imageStore(dst_probe_geometry_proximity, ivec3(tex_pos, params.cascade), vec4(found_geometry ? 1.0 : 0.0));

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

	// when x+y+z = step you do what you have to do.
	// read from x-1, y-1, z-1
	// If the main base voxel is occluded, we should try to see if it passes through?

	int invocation_idx = int(gl_LocalInvocationID.x);
	ivec3 workgroup = ivec3(gl_WorkGroupID.xyz);

	if (invocation_idx == 0) {
		solid_cell_count = 0;
	}

	groupMemoryBarrier();
	barrier();

	//ivec3 local_pos = ivec3(invocation_idx % PROBE_CELLS, (invocation_idx % (PROBE_CELLS * PROBE_CELLS)) / PROBE_CELLS, invocation_idx / (PROBE_CELLS * PROBE_CELLS));
	int cells_load_total = PROBE_CELLS * PROBE_CELLS * PROBE_CELLS;
	int group_total = 64;

	{
		int load_from = invocation_idx * cells_load_total / group_total;
		int load_to = (invocation_idx + 1) * cells_load_total / group_total;

		for (int i = load_from; i < load_to; i++) {
			ivec3 load_pos;

			load_pos.z = i / (PROBE_CELLS * PROBE_CELLS);
			load_pos.y = (i / PROBE_CELLS) % PROBE_CELLS;
			load_pos.x = i % PROBE_CELLS;

			ivec3 tex_load_pos = load_pos + ivec3(workgroup.xyz * PROBE_CELLS) + params.offset;
			uint n = imageLoad(src_normal_bits, tex_load_pos).r;
			set_bit_normal(load_pos, n);

			if (n != 0) {
				uint index = atomicAdd(solid_cell_count, 1);
				solid_cell_list[index] = uint(load_pos.x | (load_pos.y << 7) | (load_pos.z << 14));
			}
		}
	}

	groupMemoryBarrier();
	barrier();

	//process occlusion

#define OCC_STEPS (PROBE_CELLS * 3 - 2)
#define OCC_HALF_STEPS (OCC_STEPS / 2)

	int probe_offset = params.occlusion_offset;

	ivec3 base_world_region = params.region_world_pos + (params.offset / REGION_SIZE) + workgroup;

	for (int step = 0; step < OCC_STEPS; step++) {
		bool shrink = step >= OCC_HALF_STEPS;
		int occ_step = shrink ? OCC_HALF_STEPS - (step - OCC_HALF_STEPS) - 1 : step;

		if (invocation_idx < group_size_offset[occ_step].x) {
			uint pv = group_pos[group_size_offset[occ_step].y + invocation_idx];

			// Position in the occlusion volume.
			// This is an optimized version of the algorithm.
			// The idea is that you only process when x+y+z == step.
			// As such, this is what proc_abs represents (unpacks).

			ivec3 proc_abs = (ivec3(int(pv)) >> ivec3(0, 8, 16)) & ivec3(0xFF);

			if (shrink) {
				proc_abs = ivec3(PROBE_CELLS) - proc_abs - ivec3(1);
			}

			for (int i = 0; i < 4; i++) {
				// Bits to indicate where the probe starts and the direction it spreads to.
				ivec3 bits = (((ivec3(i + probe_offset) >> ivec3(0, 1, 2)) + base_world_region) & ivec3(1, 1, 1));
				ivec3 proc_sign = bits * 2 - 1;

				// Offset depends on the direction we go (probe we scan).
				ivec3 offset = ivec3(PROBE_CELLS - 1) * (ivec3(1) - bits) + proc_abs * proc_sign;

				float occ;

				uint facing = get_bit_normal(offset);

				if (facing != 0) { //solid
					occ = 0.0;
				} else if (step == 0) {
					occ = 1.0;
				} else {
					ivec3 read_dir = -proc_sign;

					float avg = 0.0;
					occ = 0.0;

					ivec3 read_x = offset + ivec3(read_dir.x, 0, 0);
					ivec3 read_y = offset + ivec3(0, read_dir.y, 0);
					ivec3 read_z = offset + ivec3(0, 0, read_dir.z);

					if (all(greaterThanEqual(read_x, ivec3(0))) && all(lessThan(read_x, ivec3(PROBE_CELLS)))) {
						uint facing_x = get_bit_normal(read_x);
						if (facing_x == 0) {
							occ += get_occlusion(read_x, i);
						}
						avg += 1.0;
					}

					if (all(greaterThanEqual(read_y, ivec3(0))) && all(lessThan(read_y, ivec3(PROBE_CELLS)))) {
						uint facing_y = get_bit_normal(read_y);
						if (facing_y == 0) {
							occ += get_occlusion(read_y, i);
						}
						avg += 1.0;
					}

					if (all(greaterThanEqual(read_z, ivec3(0))) && all(lessThan(read_z, ivec3(PROBE_CELLS)))) {
						uint facing_z = get_bit_normal(read_z);
						if (facing_z == 0) {
							occ += get_occlusion(read_z, i);
						}
						avg += 1.0;
					}

					if (avg > 0.0) {
						occ /= avg;
					}
				}

				set_occlusion(offset, i, occ);
			}
		}

		groupMemoryBarrier();
		barrier();
	}

	//bias solid voxels away

	int cell_from = invocation_idx * int(solid_cell_count) / group_total;
	int cell_to = (invocation_idx + 1) * int(solid_cell_count) / group_total;

	for (int cell_i = cell_from; cell_i < cell_to; cell_i++) {
		ivec3 offset = (ivec3(solid_cell_list[cell_i]) >> ivec3(0, 7, 14)) & (0x7F);

		uint facing = get_bit_normal(offset);

		for (int i = 0; i < 4; i++) {
			ivec3 bits = (((ivec3(i + probe_offset) >> ivec3(0, 1, 2)) + base_world_region) & ivec3(1, 1, 1));
			ivec3 proc_sign = bits * 2 - 1;
			ivec3 read_dir = -proc_sign;

			//only work on solids

			float avg = 0.0;
			float occ = 0.0;

			ivec3 read_dir_x = ivec3(read_dir.x, 0, 0);
			ivec3 read_dir_y = ivec3(0, read_dir.y, 0);
			ivec3 read_dir_z = ivec3(0, 0, read_dir.z);
			//solid

			uvec3 facing_neg = (uvec3(facing) >> uvec3(0, 2, 4)) & uvec3(1, 1, 1);
			uvec3 facing_pos = (uvec3(facing) >> uvec3(1, 3, 5)) & uvec3(1, 1, 1);

			bvec3 read_valid = bvec3(mix(facing_neg, facing_pos, greaterThan(read_dir, ivec3(0))));

			read_valid = mix(read_valid, bvec3(false), lessThan(offset + read_dir, ivec3(0)));
			read_valid = mix(read_valid, bvec3(false), greaterThanEqual(offset + read_dir, ivec3(PROBE_CELLS)));

			//sides
			if (read_valid.x) {
				ivec3 read_offset = offset + read_dir_x;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			if (read_valid.y) {
				ivec3 read_offset = offset + read_dir_y;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			if (read_valid.z) {
				ivec3 read_offset = offset + read_dir_z;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			//adjacents

			if (all(read_valid.yz)) {
				ivec3 read_offset = offset + read_dir_y + read_dir_z;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			if (all(read_valid.xz)) {
				ivec3 read_offset = offset + read_dir_x + read_dir_z;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			if (all(read_valid.xy)) {
				ivec3 read_offset = offset + read_dir_x + read_dir_y;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			//diagonal

			if (all(read_valid)) {
				ivec3 read_offset = offset + read_dir;
				uint f = get_bit_normal(read_offset);
				if (f == 0) {
					occ += get_occlusion(read_offset, i);
					avg += 1.0;
				}
			}

			if (avg > 0.0) {
				occ /= avg;
			}

			set_occlusion(offset, i, occ);
		}
	}

	groupMemoryBarrier();
	barrier();

	// Darken backfaces
	cell_from = invocation_idx * cells_load_total / group_total;
	cell_to = (invocation_idx + 1) * cells_load_total / group_total;

	for (int cell_i = cell_from; cell_i < cell_to; cell_i++) {
		ivec3 offset;
		offset.z = cell_i / (PROBE_CELLS * PROBE_CELLS);
		offset.y = (cell_i / PROBE_CELLS) % PROBE_CELLS;
		offset.x = cell_i % PROBE_CELLS;

		uint facing = get_bit_normal(offset);

		if (facing != 0) {
			continue;
		}

		for (int i = 0; i < 4; i++) {
			ivec3 bits = (((ivec3(i + probe_offset) >> ivec3(0, 1, 2)) + base_world_region) & ivec3(1, 1, 1));
			ivec3 proc_sign = bits * 2 - 1;
			ivec3 read_dir = -proc_sign;

			ivec3 read_dir_x = ivec3(read_dir.x, 0, 0);
			ivec3 read_dir_y = ivec3(0, read_dir.y, 0);
			ivec3 read_dir_z = ivec3(0, 0, read_dir.z);

			//solid, positive axis is odd bits, negative even.
			uvec3 read_mask = mix(uvec3(2, 8, 32), uvec3(1, 4, 16), greaterThan(read_dir, ivec3(0))); //match positive with negative normals

			bvec3 read_valid = bvec3(true);
			read_valid = mix(read_valid, bvec3(false), lessThan(offset + read_dir, ivec3(0)));
			read_valid = mix(read_valid, bvec3(false), greaterThanEqual(offset + read_dir, ivec3(PROBE_CELLS)));

			float visible = 0.0;
			float occlude_total = 0.0;

			if (read_valid.x) {
				uint x_mask = get_bit_normal(offset + read_dir_x);
				if (x_mask != 0) {
					occlude_total += 1.0;
					if (bool(x_mask & read_mask.x)) {
						visible += 1.0;
					}
				}
			}

			if (read_valid.y) {
				uint y_mask = get_bit_normal(offset + read_dir_y);
				if (y_mask != 0) {
					occlude_total += 1.0;
					if (bool(y_mask & read_mask.y)) {
						visible += 1.0;
					}
				}
			}

			if (read_valid.z) {
				uint z_mask = get_bit_normal(offset + read_dir_z);
				if (z_mask != 0) {
					occlude_total += 1.0;
					if (bool(z_mask & read_mask.z)) {
						visible += 1.0;
					}
				}
			}

			if (occlude_total > 0.0) {
				float occ = get_occlusion(offset, i);
				occ *= visible / occlude_total;
				set_occlusion(offset, i, occ);
			}
		}
	}

	// Store in occlusion texture

	groupMemoryBarrier();
	barrier();

	cell_from = invocation_idx * cells_load_total / group_total;
	cell_to = (invocation_idx + 1) * cells_load_total / group_total;

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
			float occ = get_occlusion(cell_pos, i);
			enc_occlusion |= uint(clamp(occ * 0xF, 0, 0xF)) << (i * 4);
		}

		imageStore(dst_occlusion, store_pos, uvec4(enc_occlusion));

		// Store wrap faces
		if (cascade_pos.x == 1) {
			imageStore(dst_occlusion, store_pos + ivec3(params.grid_size.x, 0, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.y == 1) {
			imageStore(dst_occlusion, store_pos + ivec3(0, params.grid_size.y, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.z == 1) {
			imageStore(dst_occlusion, store_pos + ivec3(0, 0, params.grid_size.z), uvec4(enc_occlusion));
		}

		if (cascade_pos.x == params.grid_size.x) {
			imageStore(dst_occlusion, store_pos - ivec3(params.grid_size.x, 0, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.y == params.grid_size.y) {
			imageStore(dst_occlusion, store_pos - ivec3(0, params.grid_size.y, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.z == params.grid_size.z) {
			imageStore(dst_occlusion, store_pos - ivec3(0, 0, params.grid_size.z), uvec4(enc_occlusion));
		}

		// Store wrap edges
		if (cascade_pos.xy == ivec2(1)) {
			imageStore(dst_occlusion, store_pos + ivec3(params.grid_size.xy, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.xz == ivec2(1)) {
			imageStore(dst_occlusion, store_pos + ivec3(params.grid_size.x, 0, params.grid_size.z), uvec4(enc_occlusion));
		}

		if (cascade_pos.yz == ivec2(1)) {
			imageStore(dst_occlusion, store_pos + ivec3(0, params.grid_size.yz), uvec4(enc_occlusion));
		}

		if (cascade_pos.xy == params.grid_size.xy) {
			imageStore(dst_occlusion, store_pos - ivec3(params.grid_size.xy, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.xz == params.grid_size.xz) {
			imageStore(dst_occlusion, store_pos - ivec3(params.grid_size.x, 0, params.grid_size.z), uvec4(enc_occlusion));
		}

		if (cascade_pos.yz == params.grid_size.yz) {
			imageStore(dst_occlusion, store_pos - ivec3(0, params.grid_size.yz), uvec4(enc_occlusion));
		}

		if (cascade_pos.xy == ivec2(1, params.grid_size.y)) {
			imageStore(dst_occlusion, store_pos + ivec3(params.grid_size.x, -params.grid_size.y, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.xy == ivec2(params.grid_size.x, 1)) {
			imageStore(dst_occlusion, store_pos + ivec3(-params.grid_size.x, params.grid_size.y, 0), uvec4(enc_occlusion));
		}

		if (cascade_pos.xz == ivec2(1, params.grid_size.z)) {
			imageStore(dst_occlusion, store_pos + ivec3(params.grid_size.x, 0, -params.grid_size.z), uvec4(enc_occlusion));
		}

		if (cascade_pos.xz == ivec2(params.grid_size.x, 1)) {
			imageStore(dst_occlusion, store_pos + ivec3(-params.grid_size.x, 0, params.grid_size.z), uvec4(enc_occlusion));
		}

		if (cascade_pos.yz == ivec2(1, params.grid_size.z)) {
			imageStore(dst_occlusion, store_pos + ivec3(0, params.grid_size.y, -params.grid_size.z), uvec4(enc_occlusion));
		}

		if (cascade_pos.yz == ivec2(params.grid_size.y, 1)) {
			imageStore(dst_occlusion, store_pos + ivec3(0, -params.grid_size.y, params.grid_size.z), uvec4(enc_occlusion));
		}

		// Store wrap vertices
		for (int i = 0; i < 8; i++) {
			bvec3 bits = bvec3((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec3 test_pos = mix(ivec3(1), params.grid_size, bits);
			ivec3 store_offset = mix(+params.grid_size, -params.grid_size, bits);
			if (cascade_pos == test_pos) {
				imageStore(dst_occlusion, store_pos + store_offset, uvec4(enc_occlusion));
			}
		}
	}

#endif

#ifdef MODE_LIGHTPROBE_SCROLL

	ivec2 local_pos = ivec2(gl_LocalInvocationID).xy;

	vec3 specular_light = vec3(0.0);
	vec3 diffuse_light = vec3(0.0);
	vec3 ambient_light = vec3(0.0);

	int upper_cascade = params.cascade + 1;

	if (upper_cascade < params.cascade_count) {
		vec3 posi = params.offset + ivec3(gl_WorkGroupID.xyz) * PROBE_CELLS;
		// Convert cell to world
		posi += params.region_world_pos * REGION_SIZE - params.grid_size / 2;
		posi -= (params.upper_region_world_pos * REGION_SIZE - params.grid_size / 2) * 2;

		vec3 posf = vec3(posi) / 2.0;

		ivec3 base_probe = ivec3(posf / PROBE_CELLS);

		if (local_pos == ivec2(0)) {
			ivec3 occ_pos = ivec3(posf);
			occ_pos = (occ_pos + params.upper_region_world_pos * REGION_SIZE) & (params.grid_size - ivec3(1));
			occ_pos += ivec3(1);
			occ_pos.y += (int(params.grid_size.y) + 2) * upper_cascade;
			vec3 occ_posf = vec3(occ_pos) + fract(posf);
			vec4 occ_0 = texture(sampler3D(occlusion[0], linear_sampler), occ_posf);
			vec4 occ_1 = texture(sampler3D(occlusion[1], linear_sampler), occ_posf);
			float occ_weights[8] = float[](occ_0.x, occ_0.y, occ_0.z, occ_0.w, occ_1.x, occ_1.y, occ_1.z, occ_1.w);
			float occlusion_total = 0.0;

			for (int i = 0; i < 8; i++) {
				ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));

				vec3 probe_pos = vec3(probe * PROBE_CELLS);

				vec3 probe_to_pos = posf - probe_pos;

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

				float weight = occ_weights[weight_index];

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
			ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec2 base_tex_pos = probe_to_texp(probe);
			ivec2 tex_pos = base_tex_pos * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1) + local_pos;
			ivec3 tex_array_pos = ivec3(tex_pos, upper_cascade);
			specular_light += rgbe_decode(imageLoad(lightprobe_specular_data, tex_array_pos).r) * occlusion_blend[i];
			diffuse_light += rgbe_decode(imageLoad(lightprobe_diffuse_data, tex_array_pos).r) * occlusion_blend[i];

			if (local_pos == ivec2(0)) {
				tex_array_pos = ivec3(base_tex_pos, upper_cascade);
				ambient_light += imageLoad(lightprobe_ambient_tex, tex_array_pos).rgb * occlusion_blend[i];
			}
		}
	}

	ivec3 probe_from = (params.offset / PROBE_CELLS) + ivec3(gl_WorkGroupID.xyz);
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
		imageStore(lightprobe_moving_average_history, history_pos, uvec4(specular_rgbe));
	}

	uvec3 moving_average = rgbe_decode_fp(specular_rgbe, FP_BITS);
	moving_average *= params.ray_hit_cache_frames;

	ivec3 ma_pos = ivec3(probe_tex_pos, params.cascade);
	ma_pos.x *= 3;

	imageStore(lightprobe_moving_average, ma_pos + ivec3(0, 0, 0), uvec4(moving_average.r));
	imageStore(lightprobe_moving_average, ma_pos + ivec3(1, 0, 0), uvec4(moving_average.g));
	imageStore(lightprobe_moving_average, ma_pos + ivec3(2, 0, 0), uvec4(moving_average.b));

#endif
}
