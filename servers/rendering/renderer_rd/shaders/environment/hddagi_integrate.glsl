#[compute]

#version 450

#VERSION_DEFINES

#ifndef MOLTENVK_USED
#if defined(has_GL_KHR_shader_subgroup_ballot) && defined(has_GL_KHR_shader_subgroup_arithmetic)

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define USE_SUBGROUPS
#endif
#endif // MOLTENVK_USED

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

#ifdef USE_CUBEMAP_ARRAY
layout(set = 1, binding = 0) uniform textureCubeArray sky_irradiance;
#else
layout(set = 1, binding = 0) uniform textureCube sky_irradiance;
#endif
layout(set = 1, binding = 1) uniform sampler linear_sampler_mipmaps;

#define SKY_MODE_DISABLED 0
#define SKY_MODE_COLOR 1
#define SKY_MODE_SKY 2

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
layout(r8ui, set = 0, binding = 3) uniform restrict readonly uimage2DArray lightprobe_neighbours;
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
	uint sky_mode;

	ivec3 scroll;
	float sky_energy;

	vec3 sky_color;
	float y_mult;

	ivec3 probe_axis_size;
	bool store_ambient_texture;

	uvec2 pad;
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

float hashf3(vec3 co) {
	return fract(sin(dot(co, vec3(12.9898, 78.233, 137.13451))) * 43758.5453);
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

bool trace_ray_hdda(vec3 ray_pos, vec3 ray_dir, int p_cascade, out ivec3 r_cell, out ivec3 r_side, out int r_cascade) {
	const int LEVEL_CASCADE = -1;
	const int LEVEL_REGION = 0;
	const int LEVEL_BLOCK = 1;
	const int LEVEL_VOXEL = 2;
	const int MAX_LEVEL = 3;

	const int fp_bits = 8;
	const int fp_block_bits = fp_bits + 2;
	const int fp_region_bits = fp_block_bits + 1;
	const int fp_cascade_bits = fp_region_bits + 4;

	bvec3 limit_dir = greaterThan(ray_dir, vec3(0.0));
	ivec3 step = mix(ivec3(0), ivec3(1), limit_dir);
	ivec3 ray_sign = ivec3(sign(ray_dir));

	ivec3 ray_dir_fp = ivec3(ray_dir * float(1 << fp_bits));

	bvec3 ray_zero = lessThan(abs(ray_dir), vec3(1.0 / 127.0));
	ivec3 inv_ray_dir_fp = ivec3(float(1 << fp_bits) / ray_dir);

	const ivec3 level_masks[MAX_LEVEL] = ivec3[](
			ivec3(1 << fp_region_bits) - ivec3(1),
			ivec3(1 << fp_block_bits) - ivec3(1),
			ivec3(1 << fp_bits) - ivec3(1));

	ivec3 region_offset_mask = (params.grid_size / REGION_SIZE) - ivec3(1);

	ivec3 limits[MAX_LEVEL];

	limits[LEVEL_REGION] = ((params.grid_size << fp_bits) - ivec3(1)) * step; // Region limit does not change, so initialize now.

	// Initialize to cascade
	int level = LEVEL_CASCADE;
	int cascade = p_cascade - 1;

	ivec3 cascade_base;
	ivec3 region_base;
	uvec2 block;
	bool hit = false;

	ivec3 pos;

	while (true) {
		// This loop is written so there is only one single main iteration.
		// This ensures that different compute threads working on different
		// levels can still run together without blocking each other.

		if (level == LEVEL_VOXEL) {
			// The first level should be (in a worst case scenario) the most used
			// so it needs to appear first. The rest of the levels go from more to least used order.

			ivec3 block_local = (pos & level_masks[LEVEL_BLOCK]) >> fp_bits;
			uint block_index = uint(block_local.z * 16 + block_local.y * 4 + block_local.x);
			if (block_index < 32) {
				// Low 32 bits.
				if (bool(block.x & uint(1 << block_index))) {
					hit = true;
					break;
				}
			} else {
				// High 32 bits.
				block_index -= 32;
				if (bool(block.y & uint(1 << block_index))) {
					hit = true;
					break;
				}
			}
		} else if (level == LEVEL_BLOCK) {
			ivec3 block_local = (pos & level_masks[LEVEL_REGION]) >> fp_block_bits;
			block = imageLoad(voxel_cascades, region_base + block_local).rg;
			if (block != uvec2(0)) {
				// Have voxels inside
				level = LEVEL_VOXEL;
				limits[LEVEL_VOXEL] = pos - (pos & level_masks[LEVEL_BLOCK]) + step * (level_masks[LEVEL_BLOCK] + ivec3(1));
				continue;
			}
		} else if (level == LEVEL_REGION) {
			ivec3 region = pos >> fp_region_bits;
			region = (cascades.data[cascade].region_world_offset + region) & region_offset_mask; // Scroll to world
			region += cascade_base;
			bool region_used = imageLoad(voxel_region_cascades, region).r > 0;

			if (region_used) {
				// The region has contents.
				region_base = (region << 1);
				level = LEVEL_BLOCK;
				limits[LEVEL_BLOCK] = pos - (pos & level_masks[LEVEL_REGION]) + step * (level_masks[LEVEL_REGION] + ivec3(1));
				continue;
			}
		} else if (level == LEVEL_CASCADE) {
			// Return to global
			if (cascade >= p_cascade) {
				ray_pos = vec3(pos) / float(1 << fp_bits);
				ray_pos /= cascades.data[cascade].to_cell;
				ray_pos += cascades.data[cascade].offset;
			}

			cascade++;
			if (cascade == params.max_cascades) {
				break;
			}

			ray_pos -= cascades.data[cascade].offset;
			ray_pos *= cascades.data[cascade].to_cell;
			pos = ivec3(ray_pos * float(1 << fp_bits));
			if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, params.grid_size << fp_bits))) {
				// Outside this cascade, go to next.
				continue;
			}

			cascade_base = ivec3(0, params.grid_size.y / REGION_SIZE * cascade, 0);
			level = LEVEL_REGION;
			continue;
		}

		// Fixed point, multi-level DDA.

		ivec3 mask = level_masks[level];
		ivec3 box = mask * step;
		ivec3 pos_diff = box - (pos & mask);
		ivec3 tv = mix((pos_diff * inv_ray_dir_fp), ivec3(0x7FFFFFFF), ray_zero) >> fp_bits;
		int t = min(tv.x, min(tv.y, tv.z));

		// The general idea here is that we _always_ need to increment to the closest next cell
		// (this is a DDA after all), so adv_box forces this increment for the minimum axis.

		ivec3 adv_box = pos_diff + ray_sign;
		ivec3 adv_t = (ray_dir_fp * t) >> fp_bits;

		pos += mix(adv_t, adv_box, equal(ivec3(t), tv));

		while (true) {
			bvec3 limit = lessThan(pos, limits[level]);
			bool inside = all(equal(limit, limit_dir));
			if (inside) {
				break;
			}
			level -= 1;
			if (level == LEVEL_CASCADE) {
				break;
			}
		}
	}

	if (hit) {
		ivec3 mask = level_masks[LEVEL_VOXEL];
		ivec3 box = mask * (step ^ ivec3(1));
		ivec3 pos_diff = box - (pos & mask);
		ivec3 tv = mix((pos_diff * -inv_ray_dir_fp), ivec3(0x7FFFFFFF), ray_zero);

		int m;
		if (tv.x < tv.y) {
			r_side = ivec3(1, 0, 0);
			m = tv.x;
		} else {
			r_side = ivec3(0, 1, 0);
			m = tv.y;
		}
		if (tv.z < m) {
			r_side = ivec3(0, 0, 1);
		}

		r_side *= -ray_sign;

		r_cell = pos >> fp_bits;

		r_cascade = cascade;
	}

	return hit;
}

#if LIGHTPROBE_OCT_SIZE == 4
const uint neighbour_max_weights = 8;
const uint neighbour_weights[128] = uint[](15544, 73563, 135085, 206971, 270171, 528301, 796795, 988221, 8569, 82130, 144347, 200892, 272100, 336249, 397500, 0, 4284, 78811, 147666, 205177, 331964, 401785, 468708, 0, 10363, 69549, 139099, 212152, 466779, 724909, 791613, 993403, 8569, 75492, 278738, 336249, 537563, 594108, 790716, 0, 73563, 135085, 270171, 343224, 403579, 528301, 600187, 660541, 69549, 139099, 338043, 408760, 466779, 595005, 665723, 724909, 141028, 205177, 401785, 475346, 659644, 734171, 987324, 0, 4284, 275419, 331964, 540882, 598393, 795001, 861924, 0, 266157, 338043, 398397, 532315, 605368, 665723, 859995, 921517, 332861, 403579, 462765, 600187, 670904, 728923, 855981, 925531, 200892, 397500, 472027, 663929, 737490, 927460, 991609, 0, 10363, 201789, 266157, 532315, 801976, 859995, 921517, 993403, 534244, 598393, 659644, 795001, 868562, 930779, 987324, 0, 594108, 663929, 730852, 790716, 865243, 934098, 991609, 0, 5181, 206971, 462765, 728923, 796795, 855981, 925531, 998584);
const uint wrap_neighbours[(LIGHTPROBE_OCT_SIZE + 2) * (LIGHTPROBE_OCT_SIZE + 2)] = uint[](196611, 3, 2, 1, 0, 196608, 196608, 0, 1, 2, 3, 196611, 131072, 65536, 65537, 65538, 65539, 131075, 65536, 131072, 131073, 131074, 131075, 65539, 0, 196608, 196609, 196610, 196611, 3, 3, 196611, 196610, 196609, 196608, 0);
#endif

#if LIGHTPROBE_OCT_SIZE == 5
const uint neighbour_max_weights = 15;
const uint neighbour_weights[375] = uint[](11139, 72624, 131886, 201671, 271258, 334768, 394335, 590836, 656174, 988103, 1319834, 1377268, 1579952, 0, 0, 6839, 76283, 139717, 205401, 267029, 334519, 400776, 461448, 527528, 590801, 657717, 984017, 1311697, 0, 0, 778, 74103, 141723, 205175, 262922, 330016, 400965, 466633, 532037, 592160, 655986, 723045, 789015, 854117, 918130, 4885, 74329, 139717, 207355, 268983, 328657, 396456, 461448, 531848, 596663, 919861, 1246161, 1573841, 0, 0, 9114, 70599, 131886, 203696, 273283, 328692, 525407, 596912, 918318, 1250247, 1317808, 1508340, 1581978, 0, 0, 6839, 72375, 133429, 197585, 263121, 338427, 400776, 664005, 723592, 991833, 1051816, 1315605, 1377233, 0, 0, 1026, 72722, 138504, 199687, 334866, 403430, 465362, 525422, 662792, 727506, 789836, 986119, 1049710, 0, 0, 68049, 138485, 199121, 399699, 468771, 530771, 657381, 727832, 794768, 858904, 919525, 1117965, 0, 0, 0, 68615, 138504, 203794, 263170, 394350, 465362, 534502, 597010, 789836, 858578, 924936, 1180782, 1248263, 0, 0, 977, 66513, 133429, 203447, 268983, 531848, 600571, 854664, 926149, 1182888, 1253977, 1508305, 1577749, 0, 0, 778, 67872, 131698, 336247, 400965, 460901, 666011, 728777, 789015, 991607, 1056325, 1116261, 1311498, 1378592, 1442418, 133093, 330193, 399699, 465688, 662773, 730915, 794768, 855821, 985553, 1055059, 1121048, 1443813, 0, 0, 0, 133468, 396510, 466974, 527582, 657756, 729118, 796314, 860190, 919900, 1051870, 1122334, 1182942, 1444188, 0, 0, 133093, 465688, 530771, 592337, 724749, 794768, 861987, 924917, 1121048, 1186131, 1247697, 1443813, 0, 0, 0, 131698, 198944, 262922, 460901, 532037, 598391, 789015, 859849, 928155, 1116261, 1187397, 1253751, 1442418, 1509664, 1573642, 4885, 66513, 336473, 396456, 664005, 723592, 993787, 1056136, 1317559, 1383095, 1444149, 1508305, 1573841, 0, 0, 330759, 394350, 662792, 727506, 789836, 990226, 1058790, 1120722, 1180782, 1311746, 1383442, 1449224, 1510407, 0, 0, 462605, 657381, 727832, 794768, 858904, 919525, 1055059, 1124131, 1186131, 1378769, 1449205, 1509841, 0, 0, 0, 525422, 592903, 789836, 858578, 924936, 1049710, 1120722, 1189862, 1252370, 1379335, 1449224, 1514514, 1573890, 0, 0, 197585, 267029, 527528, 598617, 854664, 926149, 1187208, 1255931, 1311697, 1377233, 1444149, 1514167, 1579703, 0, 0, 9114, 66548, 269232, 332743, 656174, 990128, 1049695, 1246196, 1321859, 1383344, 1442606, 1512391, 1581978, 0, 0, 977, 328657, 657717, 989879, 1056136, 1116808, 1182888, 1246161, 1317559, 1387003, 1450437, 1516121, 1577749, 0, 0, 655986, 723045, 789015, 854117, 918130, 985376, 1056325, 1121993, 1187397, 1247520, 1311498, 1384823, 1452443, 1515895, 1573642, 263121, 590801, 919861, 984017, 1051816, 1116808, 1187208, 1252023, 1315605, 1385049, 1450437, 1518075, 1579703, 0, 0, 7088, 197620, 271258, 594887, 918318, 984052, 1180767, 1252272, 1319834, 1381319, 1442606, 1514416, 1584003, 0, 0);
const uint wrap_neighbours[(LIGHTPROBE_OCT_SIZE + 2) * (LIGHTPROBE_OCT_SIZE + 2)] = uint[](262148, 4, 3, 2, 1, 0, 262144, 262144, 0, 1, 2, 3, 4, 262148, 196608, 65536, 65537, 65538, 65539, 65540, 196612, 131072, 131072, 131073, 131074, 131075, 131076, 131076, 65536, 196608, 196609, 196610, 196611, 196612, 65540, 0, 262144, 262145, 262146, 262147, 262148, 4, 4, 262148, 262147, 262146, 262145, 262144, 0);
#endif

#if LIGHTPROBE_OCT_SIZE == 6
const uint neighbour_max_weights = 18;
const uint neighbour_weights[648] = uint[](7409, 71136, 133126, 197977, 266811, 334266, 398816, 461221, 723696, 788486, 1181017, 1577531, 1902410, 1972666, 2034416, 2230090, 2299522, 0, 5357, 72623, 137015, 201808, 268219, 332144, 398278, 464109, 527325, 591849, 658038, 722921, 789403, 852980, 1180390, 1574889, 1968758, 2295545, 2062, 71772, 138508, 203472, 267601, 329055, 396334, 464939, 530524, 595281, 659485, 721675, 789292, 855086, 919566, 984415, 1049355, 0, 1375, 70993, 137936, 204044, 268380, 329742, 393995, 462877, 529745, 596060, 661547, 724014, 852747, 918879, 985102, 1051694, 1116972, 0, 4464, 71611, 136272, 202551, 269231, 333037, 395241, 461430, 526313, 592861, 660717, 725958, 1049588, 1117083, 1508070, 1902569, 1967865, 2296438, 6586, 70203, 132441, 198662, 267744, 335089, 396016, 657829, 726496, 1116166, 1508697, 1574730, 1905211, 1971842, 2033482, 2231024, 2300346, 0, 5357, 70598, 134043, 197350, 264169, 330358, 400303, 464109, 525300, 792375, 855005, 1184848, 1247209, 1578939, 1641078, 1970544, 2033641, 2295545, 2433, 71055, 137147, 200658, 264903, 398735, 466053, 529807, 592583, 792507, 857487, 919937, 1183698, 1247943, 1575623, 0, 0, 0, 68573, 137015, 201808, 264169, 394228, 464109, 531375, 595899, 658038, 789403, 857030, 922861, 987504, 1050601, 1180390, 1247209, 1313398, 1378041, 67561, 136272, 202551, 265181, 461430, 530363, 596911, 660717, 721908, 853993, 921968, 988397, 1053638, 1117083, 1312505, 1378934, 1443817, 1508070, 68295, 135122, 202683, 267663, 330113, 527047, 595343, 662661, 726415, 985473, 1054095, 1120187, 1444551, 1511378, 1903303, 0, 0, 0, 2678, 67561, 131814, 199579, 267206, 333037, 590836, 660717, 727983, 1051613, 1120055, 1443817, 1512528, 1837686, 1906619, 1967865, 2230249, 2298224, 2062, 68654, 133932, 399452, 464939, 527406, 793868, 858204, 919566, 1186512, 1250641, 1312095, 1578321, 1642525, 1704715, 1967455, 2032395, 0, 66548, 134043, 197350, 396253, 464109, 529350, 591849, 792375, 859055, 922861, 985718, 1184848, 1251259, 1315184, 1378041, 1574889, 1641078, 1705961, 133126, 197977, 461221, 529888, 594491, 788486, 857568, 924913, 989626, 1051376, 1181017, 1249851, 1317306, 1382018, 1443658, 1706736, 1771338, 0, 132441, 198662, 528955, 595424, 657829, 854768, 924090, 990449, 1054176, 1116166, 1247050, 1316482, 1382842, 1446459, 1508697, 1705802, 1772272, 0, 131814, 199579, 263156, 526313, 594886, 660717, 723933, 920182, 988397, 1055663, 1120055, 1312505, 1380720, 1447867, 1512528, 1771497, 1837686, 1902569, 199468, 265262, 329742, 592942, 661547, 727132, 985102, 1054812, 1121548, 1377631, 1447249, 1514192, 1770251, 1839133, 1906001, 2229003, 2295135, 0, 1375, 66315, 398673, 462877, 525067, 793296, 857425, 918879, 1187084, 1251420, 1312782, 1579100, 1644587, 1707054, 1968142, 2034734, 2100012, 0, 395241, 461430, 526313, 791632, 858043, 921968, 984825, 1185591, 1252271, 1316077, 1378934, 1575901, 1643757, 1708998, 1771497, 2032628, 2100123, 2163430, 527088, 591690, 787801, 856635, 924090, 988802, 1050442, 1181702, 1250784, 1318129, 1382842, 1444592, 1640869, 1709536, 1774139, 2099206, 2164057, 0, 526154, 592624, 853834, 923266, 989626, 1053243, 1115481, 1247984, 1317306, 1383665, 1447392, 1509382, 1708603, 1775072, 1837477, 2098521, 2164742, 0, 591849, 658038, 722921, 919289, 987504, 1054651, 1119312, 1313398, 1381613, 1448879, 1513271, 1705961, 1774534, 1840365, 1903581, 2097894, 2165659, 2229236, 262923, 329055, 590603, 659485, 726353, 984415, 1054033, 1120976, 1378318, 1448028, 1514764, 1772590, 1841195, 1906780, 2165548, 2231342, 2295822, 0, 4464, 67561, 329465, 399291, 461430, 791632, 853993, 1185591, 1248221, 1579951, 1643757, 1704948, 1971437, 2036678, 2100123, 2163430, 2230249, 2296438, 395975, 790482, 854727, 1185723, 1250703, 1313153, 1578383, 1645701, 1709455, 1772231, 1968513, 2037135, 2103227, 2166738, 2230983, 0, 0, 0, 787174, 853993, 920182, 984825, 1182619, 1250246, 1316077, 1380720, 1443817, 1573876, 1643757, 1711023, 1775547, 1837686, 2034653, 2103095, 2167888, 2230249, 919289, 985718, 1050601, 1114854, 1247209, 1315184, 1381613, 1446854, 1510299, 1641078, 1710011, 1776559, 1840365, 1901556, 2033641, 2102352, 2168631, 2231261, 723655, 1051335, 1118162, 1378689, 1447311, 1513403, 1706695, 1774991, 1842309, 1906063, 2034375, 2101202, 2168763, 2233743, 2296193, 0, 0, 0, 1785, 264169, 332144, 658038, 726971, 1050601, 1119312, 1444829, 1513271, 1770484, 1840365, 1907631, 1968758, 2033641, 2097894, 2165659, 2233286, 2299117, 6586, 68336, 264010, 333442, 397883, 722762, 787801, 1181702, 1578464, 1640869, 1903344, 1973489, 2037216, 2099206, 2164057, 2232891, 2300346, 0, 2678, 329465, 395241, 787174, 1182619, 1246196, 1577926, 1643757, 1706973, 1771497, 1837686, 1902569, 1971437, 2038703, 2103095, 2167888, 2234299, 2298224, 1182508, 1248302, 1312782, 1377631, 1442571, 1575982, 1644587, 1710172, 1774929, 1839133, 1901323, 1968142, 2037852, 2104588, 2169552, 2233681, 2295135, 0, 1245963, 1312095, 1378318, 1444910, 1510188, 1573643, 1642525, 1709393, 1775708, 1841195, 1903662, 1967455, 2037073, 2104016, 2170124, 2234460, 2295822, 0, 1785, 330358, 722921, 1114854, 1442804, 1510299, 1574889, 1641078, 1705961, 1772509, 1840365, 1905606, 1970544, 2037691, 2102352, 2168631, 2235311, 2299117, 5762, 67402, 264944, 334266, 395082, 725563, 1115481, 1509382, 1575664, 1837477, 1906144, 1972666, 2036283, 2098521, 2164742, 2233824, 2301169, 0);
const uint wrap_neighbours[(LIGHTPROBE_OCT_SIZE + 2) * (LIGHTPROBE_OCT_SIZE + 2)] = uint[](327685, 5, 4, 3, 2, 1, 0, 327680, 327680, 0, 1, 2, 3, 4, 5, 327685, 262144, 65536, 65537, 65538, 65539, 65540, 65541, 262149, 196608, 131072, 131073, 131074, 131075, 131076, 131077, 196613, 131072, 196608, 196609, 196610, 196611, 196612, 196613, 131077, 65536, 262144, 262145, 262146, 262147, 262148, 262149, 65541, 0, 327680, 327681, 327682, 327683, 327684, 327685, 5, 5, 327685, 327684, 327683, 327682, 327681, 327680, 0);
#endif

shared uvec3 neighbours_accum[LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE];
shared vec3 neighbours[LIGHTPROBE_OCT_SIZE * LIGHTPROBE_OCT_SIZE];
shared uvec3 ambient_accum;
shared int probe_history_index;

// MODE_PROCESS
#endif
/*
#if LIGHTPROBE_OCT_SIZE == 4
const vec3 oct_directions[16]=vec3[](vec3( (-0.408248, -0.408248, -0.816497)), vec3( (-0.316228, -0.948683, 0)), vec3( (0.316228, -0.948683, 0)), vec3( (0.408248, -0.408248, -0.816497)), vec3( (-0.948683, -0.316228, 0)), vec3( (-0.408248, -0.408248, 0.816497)), vec3( (0.408248, -0.408248, 0.816497)), vec3( (0.948683, -0.316228, 0)), vec3( (-0.948683, 0.316228, 0)), vec3( (-0.408248, 0.408248, 0.816497)), vec3( (0.408248, 0.408248, 0.816497)), vec3( (0.948683, 0.316228, 0)), vec3( (-0.408248, 0.408248, -0.816497)), vec3( (-0.316228, 0.948683, 0)), vec3( (0.316228, 0.948683, 0)), vec3( (0.408248, 0.408248, -0.816497)));
#endif

#if LIGHTPROBE_OCT_SIZE == 5
const vec3 oct_directions[25]=vec3[](vec3( (-0.301511, -0.301511, -0.904534)), vec3( (-0.301511, -0.904534, -0.301511)), vec3( (0, -0.970142, 0.242536)), vec3( (0.301511, -0.904534, -0.301511)), vec3( (0.301511, -0.301511, -0.904534)), vec3( (-0.904534, -0.301511, -0.301511)), vec3( (-0.666667, -0.666667, 0.333333)), vec3( (0, -0.5547, 0.83205)), vec3( (0.666667, -0.666667, 0.333333)), vec3( (0.904534, -0.301511, -0.301511)), vec3( (-0.970142, 0, 0.242536)), vec3( (-0.5547, 0, 0.83205)), vec3( (0, 0, 1)), vec3( (0.5547, 0, 0.83205)), vec3( (0.970142, 0, 0.242536)), vec3( (-0.904534, 0.301511, -0.301511)), vec3( (-0.666667, 0.666667, 0.333333)), vec3( (0, 0.5547, 0.83205)), vec3( (0.666667, 0.666667, 0.333333)), vec3( (0.904534, 0.301511, -0.301511)), vec3( (-0.301511, 0.301511, -0.904534)), vec3( (-0.301511, 0.904534, -0.301511)), vec3( (0, 0.970142, 0.242536)), vec3( (0.301511, 0.904534, -0.301511)), vec3( (0.301511, 0.301511, -0.904534)));
#endif

shared uvec3 neighbours[LIGHTPROBE_OCT_SIZE*LIGHTPROBE_OCT_SIZE];
*/

ivec3 modi(ivec3 value, ivec3 p_y) {
	// GLSL Specification says:
	// "Results are undefined if one or both operands are negative."
	// So..
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
			process = camera_visible;
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

	// Apply bias (by a cell)
	float bias = params.ray_bias;
	vec3 abs_ray_dir = abs(ray_dir);
	ray_pos += ray_dir * 1.0 / max(abs_ray_dir.x, max(abs_ray_dir.y, abs_ray_dir.z)) * bias / cascades.data[params.cascade].to_cell;

	cache_texture_pos = ivec3(probe_texture_pos.xy * LIGHTPROBE_OCT_SIZE + local_pos, probe_texture_pos.z * params.history_size + probe_history_index);
	cache_entry = imageLoad(ray_hit_cache, cache_texture_pos).r;

	cache_valid = bool(cache_entry & CACHE_IS_VALID);

	cache_invalidated_debug = vec3(0.0);

	if (cache_valid) {
		// Make sure the cache is really valid
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
	} else if (params.sky_mode == SKY_MODE_SKY) {
#ifdef USE_CUBEMAP_ARRAY
		light = textureLod(samplerCubeArray(sky_irradiance, linear_sampler_mipmaps), vec4(ray_dir, 0.0), 2.0).rgb; // Use second mipmap because we don't usually throw a lot of rays, so this compensates.
#else
		light = textureLod(samplerCube(sky_irradiance, linear_sampler_mipmaps), ray_dir, 2.0).rgb; // Use second mipmap because we don't usually throw a lot of rays, so this compensates.
#endif
		light *= params.sky_energy;
	} else if (params.sky_mode == SKY_MODE_COLOR) {
		light = params.sky_color;
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
			uint wrap_neighbour = wrap_neighbours[(bilinear_base.y + i) * (LIGHTPROBE_OCT_SIZE + 2) + (bilinear_base.x + j)];
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

	neighbours[probe_index] = light;
#endif

	if (!cache_valid) {
		cache_entry = CACHE_IS_VALID;
		if (hit) {
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

			uvec3 ucell = (uvec3(hit_cell) & uvec3(0xFF)) << uvec3(0, 8, 16);
			cache_entry |= CACHE_IS_HIT | ucell.x | ucell.y | ucell.z | (uint(min(7, hit_cascade)) << 24) | (axis << 27);

			uint region_version = imageLoad(region_versions, (hit_cell >> REGION_SIZE) + ivec3(0, hit_cascade * (params.grid_size.y / REGION_SIZE), 0)).r;

			imageStore(ray_hit_cache_version, cache_texture_pos, uvec4(region_version));
		}

		imageStore(ray_hit_cache, cache_texture_pos, uvec4(cache_entry));
	}

	groupMemoryBarrier();
	barrier();

	// convert back to float and do moving average

#ifdef TRACE_SUBPIXEL
	light = vec3(neighbours_accum[probe_index]) / float(1 << FP_BITS);
#else
	light = neighbours[probe_index];
#endif

	// Encode to RGBE to store in accumulator

	uint light_rgbe = rgbe_encode(light);

	ivec3 ma_pos = ivec3(cache_texture_pos.xy * ivec2(3, 1), params.cascade);

	uvec3 moving_average = uvec3(
			imageLoad(lightprobe_moving_average, ma_pos + ivec3(0, 0, 0)).r,
			imageLoad(lightprobe_moving_average, ma_pos + ivec3(1, 0, 0)).r,
			imageLoad(lightprobe_moving_average, ma_pos + ivec3(2, 0, 0)).r);

	ivec3 history_pos = ivec3(probe_texture_pos.xy * 4 + ivec2(probe_index % 4, probe_index / 4), probe_texture_pos.z * params.history_size + probe_history_index);

	uvec3 prev_val = rgbe_decode_fp(imageLoad(lightprobe_moving_average_history, cache_texture_pos).r, FP_BITS);

	moving_average -= prev_val;
	uvec3 new_val = rgbe_decode_fp(light_rgbe, FP_BITS); // Round trip to ensure integer consistency
	moving_average += new_val;

	imageStore(lightprobe_moving_average_history, cache_texture_pos, uvec4(light_rgbe));

	imageStore(lightprobe_moving_average, ma_pos + ivec3(0, 0, 0), uvec4(moving_average.r));
	imageStore(lightprobe_moving_average, ma_pos + ivec3(1, 0, 0), uvec4(moving_average.g));
	imageStore(lightprobe_moving_average, ma_pos + ivec3(2, 0, 0), uvec4(moving_average.b));

	moving_average /= params.history_size;

	if (params.store_ambient_texture) {
		atomicAdd(ambient_accum.r, moving_average.r);
		atomicAdd(ambient_accum.g, moving_average.g);
		atomicAdd(ambient_accum.b, moving_average.b);
	}

	light = vec3(moving_average) / float(1 << FP_BITS);
	neighbours[probe_index] = light;

	groupMemoryBarrier();
	barrier();

	// Compute specular, diffuse, ambient

	vec3 diffuse_light = vec3(0);
	vec3 specular_light = light;

	for (uint i = 0; i < neighbour_max_weights; i++) {
		uint n = neighbour_weights[probe_index * neighbour_max_weights + i];
		uint index = n >> 16;
		float weight = float(n & 0xFFFF) / float(0xFFFF);
		diffuse_light += neighbours[index] * weight;
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

	light_rgbe = rgbe_encode(specular_light);
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

		const vec3 aniso_dir[6] = vec3[](
				vec3(-1, 0, 0),
				vec3(1, 0, 0),
				vec3(0, -1, 0),
				vec3(0, 1, 0),
				vec3(0, 0, -1),
				vec3(0, 0, 1));

		uint neighbour_visibility = imageLoad(lightprobe_neighbours, probe_base_pos).r;

		for (int i = 0; i < 6; i++) {
			float visibility = ((neighbour_visibility >> (i * 4)) & 0xF) / float(0xF);
			if (visibility == 0.0) {
				continue; // un-neighboured
			}

			ivec3 neighbour_probe = probe_cell + ivec3(aniso_dir[i]);
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

	bool intersects = true;

	// Test planes
	for (int i = 0; i < MAX_CAMERA_PLANES; i++) {
		vec4 plane = camera.planes[i];
		vec3 point = ofs + mix(+half_extents, -half_extents, greaterThan(plane.xyz, vec3(0.0)));
		if (dot(plane.xyz, point) > plane.w) {
			return; // Does not intersect.
		}
	}

	// Test points. Most cases the above test is fast and discards entirely, but this one is needed if passes.

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
		return; // Does not intersect.
	}

	// Mark it

	for (int i = 0; i < 8; i++) {
		ivec3 probe = params.world_offset + region + ((ivec3(i) >> ivec3(1, 2, 4)) & ivec3(1));
		ivec3 probe_scroll_pos = modi(probe, params.probe_axis_size);
		ivec3 probe_pos = ivec3((probe_scroll_pos.xy + ivec2(0, probe_scroll_pos.z * params.probe_axis_size.y)), params.cascade);

		imageStore(lightprobe_camera_visibility, probe_pos, vec4(1.0));
	}

#endif
}
