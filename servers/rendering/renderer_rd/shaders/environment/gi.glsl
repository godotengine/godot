#[compute]

#version 450

#VERSION_DEFINES

#ifdef SAMPLE_VOXEL_GI_NEAREST
#extension GL_EXT_samplerless_texture_functions : enable
#endif

#include "hddagi_radiance_inc.glsl"
#include "hddagi_voxel_metadata_inc.glsl"

#define square(m) ((m) * (m))

#define GROUP_SIZE 8
#define DITHER_SIZE 2

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = 1) in;

#define M_PI 3.141592

/* Specialization Constants (Toggles) */

layout(constant_id = 0) const bool sc_half_res = false;
layout(constant_id = 1) const bool sc_use_full_projection_matrix = false;
layout(constant_id = 2) const bool sc_use_vrs = false;
layout(constant_id = 3) const bool sc_half_reflections = false;
layout(constant_id = 4) const bool sc_diffuse_only = false;

#define HDDAGI_MAX_CASCADES 8

//set 0 for HDDAGI and render buffers

layout(rg32ui, set = 0, binding = 1) uniform restrict readonly uimage3D voxel_cascades;
layout(r8ui, set = 0, binding = 2) uniform restrict readonly uimage3D voxel_region_cascades;
#define HDDAGI_DATA_BINDING 3
#define HDDAGI_PAGE_BINDING 10
#include "hddagi_voxel_pages_inc.glsl"

layout(set = 0, binding = 4) uniform texture2DArray lightprobe_specular;
layout(set = 0, binding = 5) uniform texture2DArray lightprobe_diffuse;
layout(set = 0, binding = 6) uniform texture3D occlusion[2];

layout(set = 0, binding = 7) uniform sampler linear_sampler;
layout(set = 0, binding = 8) uniform sampler linear_sampler_with_mipmaps;

struct ProbeCascadeData {
	vec3 position;
	float to_probe;

	ivec3 region_world_offset;
	float to_cell; // 1/bounds * grid_size

	vec3 blend_position;
	float exposure_normalization;

	uvec4 pad2;
};

layout(set = 0, binding = 12) uniform texture2D depth_buffer;
layout(set = 0, binding = 13) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 14) uniform utexture2D voxel_gi_buffer;

layout(set = 0, binding = 15, std140) uniform HDDAGI {
	ivec3 grid_size;
	int max_cascades;

	float normal_bias;
	float energy;
	float y_mult;
	float reflection_bias;

	ivec3 probe_axis_size;
	uint pad;

	uvec4 pad3;

	ProbeCascadeData cascades[HDDAGI_MAX_CASCADES];
}
hddagi;

#define MAX_VOXEL_GI_INSTANCES 8

struct VoxelGIData {
	mat4 xform; // 64 - 64

	vec3 bounds; // 12 - 76
	float dynamic_range; // 4 - 80

	float bias; // 4 - 84
	float normal_bias; // 4 - 88
	bool blend_ambient; // 4 - 92
	uint mipmaps; // 4 - 96

	vec3 octree_size; // 12 - 108
	float exposure_normalization; // 4 - 112
};

layout(set = 0, binding = 16, std140) uniform VoxelGIs {
	VoxelGIData data[MAX_VOXEL_GI_INSTANCES];
}
voxel_gi_instances;

layout(set = 0, binding = 17) uniform texture3D voxel_gi_textures[MAX_VOXEL_GI_INSTANCES];

layout(set = 0, binding = 18, std140) uniform SceneData {
	mat4x4 inv_projection[2];
	mat4x4 cam_transform;
	vec4 eye_offset[2];

	ivec2 screen_size;
	float pad1;
	float pad2;
}
scene_data;

#ifdef USE_VRS
layout(r8ui, set = 0, binding = 19) uniform restrict readonly uimage2D vrs_buffer;
#endif
layout(r32ui, set = 0, binding = 20) uniform restrict writeonly uimage2D ambient_buffer;
layout(r32ui, set = 0, binding = 21) uniform restrict writeonly uimage2D reflection_buffer;
layout(set = 0, binding = 23) uniform texture2DArray probe_geometry_proximity;
layout(set = 0, binding = 24) uniform texture3D occlusion_correction[2];
layout(rg8, set = 0, binding = 22) uniform restrict writeonly image2D blend_buffer;

layout(push_constant, std430) uniform Params {
	uint max_voxel_gi_instances;
	bool high_quality_vct;
	bool orthogonal;
	uint view_index;

	vec4 proj_info;

	float z_near;
	float z_far;
	uint world_gi_flags;
	uint pad;
}
params;

shared vec3 group_positions[GROUP_SIZE * GROUP_SIZE];
shared vec4 group_reflections[GROUP_SIZE * GROUP_SIZE];
shared vec3 group_normals[GROUP_SIZE * GROUP_SIZE];
shared float group_roughness[GROUP_SIZE * GROUP_SIZE];

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

vec4 blend_color(vec4 src, vec4 dst) {
	vec4 res;
	float sa = 1.0 - src.a;
	res.a = dst.a * sa + src.a;
	if (res.a == 0.0) {
		res.rgb = vec3(0);
	} else {
		res.rgb = (dst.rgb * dst.a * sa + src.rgb * src.a) / res.a;
	}
	return res;
}

vec3 reconstruct_position(ivec2 screen_pos) {
	if (sc_use_full_projection_matrix) {
		vec4 pos;
		pos.xy = (2.0 * vec2(screen_pos) / vec2(scene_data.screen_size)) - 1.0;
		pos.z = texelFetch(sampler2D(depth_buffer, linear_sampler), screen_pos, 0).r;
		pos.w = 1.0;

		pos = scene_data.inv_projection[params.view_index] * pos;

		return pos.xyz / pos.w;
	} else {
		vec3 pos;
		pos.z = texelFetch(sampler2D(depth_buffer, linear_sampler), screen_pos, 0).r;

		pos.z = pos.z * 2.0 - 1.0;
		if (params.orthogonal) {
			pos.z = -(pos.z * (params.z_far - params.z_near) - (params.z_far + params.z_near)) / 2.0;
		} else {
			pos.z = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near + pos.z * (params.z_far - params.z_near));
		}
		pos.z = -pos.z;

		pos.xy = vec2(screen_pos) * params.proj_info.xy + params.proj_info.zw;
		if (!params.orthogonal) {
			pos.xy *= pos.z;
		}

		pos.y = -pos.y;

		return pos;
	}
}

vec4 fetch_normal_and_roughness(ivec2 pos) {
	vec4 normal_roughness = texelFetch(sampler2D(normal_roughness_buffer, linear_sampler), pos, 0);
	normal_roughness.xyz = normalize(normal_roughness.xyz * 2.0 - 1.0);
	return normal_roughness;
}

#define PROBE_CELLS 8
#define OCC_DISTANCE_MAX 16.0
#define REGION_SIZE 8

ivec2 probe_to_tex(ivec3 wrapped_base, ivec3 offset) {
	ivec3 cell = wrapped_base + offset;
	cell = mix(cell, cell - hddagi.probe_axis_size, greaterThanEqual(cell, hddagi.probe_axis_size));
	return cell.xy + ivec2(0, cell.z * int(hddagi.probe_axis_size.y));
}

#define ROUGHNESS_TO_REFLECTION_TRESHOOLD 0.25
#define ROUGHNESS_TO_REFLECTION_FULL_BLUR 0.40

bool bayer_dither(float value) {
#if DITHER_SIZE == 3

	uvec2 dt = gl_GlobalInvocationID.xy % 3;
	uint index = dt.x + dt.y * 3;

	float table[9] = float[9](0.0 / 9.0, 7.0 / 9.0, 3.0 / 9.0,
			6.0 / 9.0, 5.0 / 9.0, 2.0 / 9.0,
			4.0 / 9.0, 1.0 / 9.0, 8.0 / 9.0);

	return value > table[index];
#else

	uvec2 dt = gl_GlobalInvocationID.xy & 0x1;
	uint index = dt.x + dt.y * 2;

	const float table[4] = float[](0.0, 0.5, 0.75, 0.25);

	return value > table[index];

#endif

#if 0
	uvec2 dt = gl_GlobalInvocationID.xy & 0x3;
	uint index = dt.x + dt.y * 4;

	const float table[16] = float[](0.0625, 0.5625, 0.1875, 0.6875, 0.8125, 0.3125, 0.9375, 0.4375, 0.25, 0.75, 0.125, 0.625, 1.0, 0.5, 0.875, 0.375);

	return value > table[index];
#endif
}

uint hash(uint x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

float dist_to_box(vec3 min_bound, vec3 max_bound, bvec3 step, vec3 pos, vec3 inv_dir) {
	vec3 box = (mix(min_bound, max_bound, step) - pos) * inv_dir;
	return min(box.x, min(box.y, box.z));
}

//find arbitrary tangent and bitangent, then build a matrix
mat3 create_basis_from_normal(vec3 normal) {
	vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(v0, normal));
	vec3 bitangent = normalize(cross(tangent, normal));
	return mat3(tangent, bitangent, normal);
}

vec3 vogel_hemisphere(int i, int n, float f) {
	float goldenAngle = 137.5077640500378546463487 * float(i);
	float theta = radians(goldenAngle);

	float phi = acos(1.0 - 2.0 * (float(i) + f) / float(n));

	float x = sin(phi) * cos(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(phi);

	return vec3(x, y, z);
}

#define HDDAGI_TRACE_REFLECTION
#include "hddagi_trace_inc.glsl"

void sdfvoxel_gi_process(int cascade, vec3 cascade_pos, vec3 cam_pos, vec3 cam_normal, vec3 cam_specular_normal, float roughness, bool dynamic_object, out vec3 diffuse_light, out vec3 specular_light) {
	vec3 posf = cascade_pos;

	if (!dynamic_object) {
		posf += cam_normal * hddagi.normal_bias;
	}

	ivec3 posi = ivec3(posf);
	ivec3 base_probe = posi / PROBE_CELLS;
	ivec3 world_probe = hddagi.cascades[cascade].region_world_offset + base_probe;
	ivec3 wrapped_base = hddagi_wrap_probe(world_probe, hddagi.probe_axis_size);
	ivec3 parity = world_probe & 1;
	int base_channel = parity.x | (parity.y << 1) | (parity.z << 2);

	vec3 diffuse_accum = vec3(0.0);
	vec3 specular_accum = vec3(0.0);
	float weight_accum = 0.0;

	ivec3 occ_pos = posi; // faster and numerically safer to do this computation as ints
	vec3 pos_fract = posf - vec3(posi);
	occ_pos = (occ_pos + hddagi.cascades[cascade].region_world_offset * PROBE_CELLS) & (hddagi.grid_size - 1);
	occ_pos.y += (hddagi.grid_size.y + 2) * cascade;
	occ_pos += ivec3(1);
	ivec3 occ_total_size = hddagi.grid_size + ivec3(2);
	occ_total_size.y *= hddagi.max_cascades;
	vec3 occ_posf = (vec3(occ_pos) + pos_fract) / vec3(occ_total_size);

	vec4 occ_0 = texture(sampler3D(occlusion[0], linear_sampler), occ_posf.yxz);
	vec4 occ_1 = texture(sampler3D(occlusion[1], linear_sampler), occ_posf.yxz);
	// Only the receiver weights use the seam correction. Keep the original
	// visibility field for probe feedback, tracing and surface lighting.
	vec3 local = mod(posf, float(PROBE_CELLS));
	ivec3 correction_size = hddagi.grid_size / 2 + 2;
	correction_size.y *= hddagi.max_cascades;
	vec3 correction_pos = (vec3(occ_pos) + pos_fract - vec3(1.0)) * 0.5 + vec3(1.0);
	correction_pos += hddagi_occlusion_cache_warp(local) - local * 0.5;
	correction_pos.y += float(cascade);
	vec4 correction_0 = texture(sampler3D(occlusion_correction[0], linear_sampler), (correction_pos / vec3(correction_size)).yxz);
	vec4 correction_1 = texture(sampler3D(occlusion_correction[1], linear_sampler), (correction_pos / vec3(correction_size)).yxz);
	vec4 blend_0 = vec4(0.0);
	vec4 blend_1 = vec4(0.0);
	vec4 safe_0 = vec4(1.0);
	vec4 safe_1 = vec4(1.0);
	for (int axis = 0; axis < 3; axis++) {
		float distance_to_seam = min(local[axis], float(PROBE_CELLS) - local[axis]);
		float strength = 1.0 - smoothstep(2.5, 3.5, distance_to_seam);
		int seam = int(floor(posf[axis] / float(PROBE_CELLS) + 0.5));
		int seam_parity = (seam + hddagi.cascades[cascade].region_world_offset[axis]) & 1;
		vec4 shared_0 = vec4(equal((ivec4(0, 1, 2, 3) >> axis) & 1, ivec4(seam_parity)));
		vec4 shared_1 = vec4(equal((ivec4(4, 5, 6, 7) >> axis) & 1, ivec4(seam_parity)));
		blend_0 = max(blend_0, shared_0 * strength);
		blend_1 = max(blend_1, shared_1 * strength);
		// A non-shared channel names a different probe across this face.
		// Exclude it while the half-resolution footprint crosses the face.
		float safe = smoothstep(1.0, 1.5, distance_to_seam);
		safe_0 *= mix(vec4(safe), vec4(1.0), shared_0);
		safe_1 *= mix(vec4(safe), vec4(1.0), shared_1);
	}
	occ_0 = clamp(occ_0 + correction_0 * blend_0 * safe_0, 0.0, 1.0);
	occ_1 = clamp(occ_1 + correction_1 * blend_1 * safe_1, 0.0, 1.0);

	vec2 light_probe_tex_to_uv = 1.0 / vec2((LIGHTPROBE_OCT_SIZE + 2) * hddagi.probe_axis_size.x, (LIGHTPROBE_OCT_SIZE + 2) * hddagi.probe_axis_size.y * hddagi.probe_axis_size.z);
	vec2 light_uv = octahedron_encode(vec3(cam_normal)) * float(LIGHTPROBE_OCT_SIZE);
	vec2 light_uv_spec = octahedron_encode(vec3(cam_specular_normal)) * float(LIGHTPROBE_OCT_SIZE);

	uint valid_probes = 255u;
	if ((params.world_gi_flags & 8u) != 0u) {
		valid_probes = 0u;
		for (int i = 0; i < 8; i++) {
			ivec3 offset = (ivec3(i) >> ivec3(0, 1, 2)) & 1;
			float proximity = texelFetch(sampler2DArray(probe_geometry_proximity, linear_sampler), ivec3(probe_to_tex(wrapped_base, offset), cascade), 0).r;
			if (proximity < 0.7 || proximity > 0.8) {
				valid_probes |= 1u << uint(i);
			}
		}
		// No valid interpolation neighbourhood: keep the original gather rather
		// than introducing a black hole at coarse voxel resolution.
		if (valid_probes == 0u) {
			valid_probes = 255u;
		}
	}
	for (int i = 0; i < 8; i++) {
		if ((valid_probes & (1u << uint(i))) == 0u) {
			continue;
		}
		ivec3 offset = (ivec3(i) >> ivec3(0, 1, 2)) & 1;
		ivec3 probe = base_probe + offset;

		vec3 probe_pos = vec3(probe * PROBE_CELLS);

		vec3 probe_to_pos = posf - probe_pos;
		vec3 n = normalize(probe_to_pos);

		float weight = 1.0;
		//weight *= pow(max(0.0001, (dot(-n, cam_normal) + 1.0) * 0.5),2.0) + 0.2;
		if (!dynamic_object) {
			weight *= max(0.005, (dot(-n, cam_normal)));
		}

		weight *= hddagi_occlusion_weight(occ_0, occ_1, base_channel ^ i);

		vec3 trilinear = vec3(1.0) - abs(probe_to_pos / float(PROBE_CELLS));

		weight *= trilinear.x * trilinear.y * trilinear.z;

		ivec2 tex_pos = probe_to_tex(wrapped_base, offset);
		vec2 base_tex_uv = vec2(ivec2(tex_pos * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1)));
		vec2 tex_uv = base_tex_uv + light_uv;
		tex_uv *= light_probe_tex_to_uv;

		vec3 probe_light = texture(sampler2DArray(lightprobe_diffuse, linear_sampler), vec3(tex_uv, float(cascade))).rgb;

		diffuse_accum += probe_light * weight;

		tex_uv = base_tex_uv + light_uv_spec;
		tex_uv *= light_probe_tex_to_uv;

		vec3 probe_ref_light;
		if (roughness < ROUGHNESS_TO_REFLECTION_FULL_BLUR && (params.world_gi_flags & 3u) != 2u) {
			probe_ref_light = texture(sampler2DArray(lightprobe_specular, linear_sampler), vec3(tex_uv, float(cascade))).rgb;
		} else {
			probe_ref_light = vec3(0.0);
		}

		vec3 probe_ref_full_light;
		if (roughness > ROUGHNESS_TO_REFLECTION_TRESHOOLD && (params.world_gi_flags & 3u) != 2u) {
			probe_ref_full_light = texture(sampler2DArray(lightprobe_diffuse, linear_sampler), vec3(tex_uv, float(cascade))).rgb;
		} else {
			probe_ref_full_light = vec3(0.0);
		}

		probe_ref_light = mix(probe_ref_light, probe_ref_full_light, smoothstep(ROUGHNESS_TO_REFLECTION_TRESHOOLD, ROUGHNESS_TO_REFLECTION_FULL_BLUR, roughness));

		specular_accum += probe_ref_light * weight;

		weight_accum += weight;
	}

	diffuse_light = diffuse_accum / max(weight_accum, 1e-6);
	specular_light = specular_accum / max(weight_accum, 1e-6);
}

void hddagi_process(vec3 vertex, vec3 normal, vec3 reflection, float roughness, bool dynamic_object, bool trace_reflection, out vec4 ambient_light, out vec4 reflection_light) {
	//make vertex orientation the world one, but still align to camera
	vertex.y *= hddagi.y_mult;
	normal.y *= hddagi.y_mult;
	reflection.y *= hddagi.y_mult;

	//renormalize
	normal = normalize(normal);
	reflection = normalize(reflection);

	vec3 cam_pos = vertex;
	vec3 cam_normal = normal;

	vec4 light_accum = vec4(0.0);
	float weight_accum = 0.0;

	vec4 light_blend_accum = vec4(0.0);
	float weight_blend_accum = 0.0;

	float blend = -1.0;

	// helper constants, compute once

	int cascade = 0x7FFFFFFF;
	vec3 cascade_pos;
	vec3 cascade_normal;
	float cell_size;

	for (int i = 0; i < hddagi.max_cascades; i++) {
		cascade_pos = (cam_pos - hddagi.cascades[i].position) * hddagi.cascades[i].to_cell;

		if (any(lessThan(cascade_pos, vec3(0.0))) || any(greaterThanEqual(cascade_pos, vec3(hddagi.grid_size)))) {
			continue; //skip cascade
		}

		cascade = i;
		cell_size = 1.0 / hddagi.cascades[i].to_cell;

		break;
	}

	if (cascade < HDDAGI_MAX_CASCADES) {
		ambient_light = vec4(0, 0, 0, 1);
		reflection_light = vec4(0, 0, 0, 1);

		vec3 diffuse, specular;
		float blend;
		{
			//process blend
			vec3 blend_from = ((vec3(hddagi.probe_axis_size) - 1) / 2.0);

			vec3 inner_pos = (cam_pos - hddagi.cascades[cascade].blend_position) * hddagi.cascades[cascade].to_probe;

			vec3 inner_dist = blend_from - abs(inner_pos);

			float min_d = min(inner_dist.x, min(inner_dist.y, inner_dist.z));

			blend = clamp(1.0 - smoothstep(0.5, 2.5, min_d), 0, 1);

			if (cascade < hddagi.max_cascades - 1) {
				cell_size = mix(cell_size, 1.0 / hddagi.cascades[cascade + 1].to_cell, blend);
			}

#ifndef USE_AMBIENT_BLEND
			if (cascade < hddagi.max_cascades - 1) {
				if (bayer_dither(blend)) {
					cascade++;
					cascade_pos = (cam_pos - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
				}
				blend = 0.0;
			}
#endif
		}

		sdfvoxel_gi_process(cascade, cascade_pos, cam_pos, cam_normal, reflection, roughness, dynamic_object, diffuse, specular);

#ifdef USE_AMBIENT_BLEND

		if (blend > 0.0) {
			if (cascade < hddagi.max_cascades - 1) {
				vec3 blend_cascade_pos = (cam_pos - hddagi.cascades[cascade + 1].position) * hddagi.cascades[cascade + 1].to_cell;
				vec3 diffuse2, specular2;
				sdfvoxel_gi_process(cascade + 1, blend_cascade_pos, cam_pos, cam_normal, reflection, roughness, dynamic_object, diffuse2, specular2);
				diffuse = mix(diffuse, diffuse2, blend);
				specular = mix(specular, specular2, blend);

				if (bayer_dither(blend)) {
					// Apply dither for roughness here.
					cascade++;
					cascade_pos = (cam_pos - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
				}

				blend = 0.0;
			}
		}

#endif

		ambient_light.rgb = diffuse;
		ambient_light.a = 1.0 - blend;

		if (trace_reflection && (params.world_gi_flags & 3u) == 0u && roughness < ROUGHNESS_TO_REFLECTION_TRESHOOLD) {
			ivec3 hit_cell;
			ivec3 hit_face;
			int hit_cascade;
			vec4 light = vec4(0);

			vec3 ray_pos = cam_pos;
			vec3 ray_dir = reflection;

			vec3 start_cell = (ray_pos - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;

			{ // Bias ray

				vec3 abs_cam_normal = abs(cam_normal);
				vec3 ray_bias = cam_normal * 1.0 / max(abs_cam_normal.x, max(abs_cam_normal.y, abs_cam_normal.z));

				start_cell += ray_bias * hddagi.reflection_bias; // large bias to pass through the reflector cell.
				ray_pos = start_cell / hddagi.cascades[cascade].to_cell + hddagi.cascades[cascade].position;
			}

			mat3 normal_mat = create_basis_from_normal(ray_dir);
			vec3 n = vogel_hemisphere(int((gl_GlobalInvocationID.x % DITHER_SIZE) + (gl_GlobalInvocationID.y % DITHER_SIZE) * DITHER_SIZE), DITHER_SIZE * DITHER_SIZE, 0.0);
			n = normalize(mix(vec3(0, 0, -1), n, roughness * roughness));
			n.z = -n.z;
			n = normal_mat * n;

			bool static_hit = trace_ray_hdda(ray_pos, n, cascade, hit_cell, hit_face, hit_cascade);

			if (static_hit) {
				bool valid = true;
				bool disoccluded = false;
				if (hit_cascade == cascade) {
					if (ivec3(start_cell) == hit_cell) {
						// self hit the start cell, ouch, load the disocclusion

						ivec3 read_cell = (hit_cell + (hddagi.cascades[hit_cascade].region_world_offset * REGION_SIZE)) & (ivec3(hddagi.grid_size) - 1);
						uint disocc = hddagi_unpack_disocclusion(hddagi_read_neighbours(read_cell + ivec3(0, (hddagi.grid_size.y * hit_cascade), 0)));

						if (disocc == 0) {
							// Can happen.. guess.
							vec3 abs_normal = abs(cam_normal);
							// Find closest normal to cam normal.
							int ni = 0;
							float m = abs_normal.x;
							if (abs_normal.y > m) {
								m = abs_normal.y;
								ni = 1;
							}
							if (abs_normal.z > m) {
								ni = 2;
							}

							vec3 local = fract(start_cell) - 0.5; // create local cell.

							// Find the closest axis to push.
							vec3 ax_a = vec3(ni == 2, ni == 0, ni == 1);
							vec3 ax_b = vec3(ni == 1, ni == 2, ni == 0);

							vec3 advance;
							if (abs(dot(ax_a, local)) > abs(dot(ax_b, local))) {
								advance = ax_a * sign(local);
							} else {
								advance = ax_b * sign(local);
							}

							start_cell += advance;
							hit_cell += ivec3(advance);

							read_cell = (hit_cell + (hddagi.cascades[hit_cascade].region_world_offset * REGION_SIZE)) & (ivec3(hddagi.grid_size) - 1);
							disocc = hddagi_unpack_disocclusion(hddagi_read_neighbours(read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0)));
						}

						// find best disocclusion direction.

						vec3 local = fract(start_cell) - 0.5; // create local cell.

						int best_axis = 0;
						float best_d = -20;
						for (int i = 0; i < 6; i++) {
							if (bool(disocc & (1 << i))) {
								float d = dot(local, vec3(hddagi_axis_direction(uint(i))));
								if (d > best_d) {
									best_axis = i;
									best_d = d;
								}
							}
						}

						hit_face = hddagi_axis_direction(uint(best_axis));

						/*
						if (disocc == 0) {
							light.rgb = vec3(1,0,0);
						} else {
							light.rgb = vec3(0,1,0);
						}
						light.rgb = vec3(hddagi_axis_direction(uint(best_axis))) * 0.5 + 0.5;
						light.a = 1;
						valid=false;*/
						disoccluded = true;
					}
				}
				if (valid) {
					hit_cell += hit_face;
					ivec3 read_cell = (hit_cell + (hddagi.cascades[hit_cascade].region_world_offset * REGION_SIZE)) & (ivec3(hddagi.grid_size) - 1);
					light.rgb = hddagi_read_light(read_cell + ivec3(0, (hddagi.grid_size.y * hit_cascade), 0));
					light.a = 1;

					if (!disoccluded) {
						uint neighbour_bits = hddagi_read_neighbours(read_cell + ivec3(0, (hddagi.grid_size.y * hit_cascade), 0)) & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
						vec3 cell_delta = vec3(hit_cell) + 0.5 - (ray_pos - hddagi.cascades[hit_cascade].position) * hddagi.cascades[hit_cascade].to_cell;
						vec3 direction = normalize(ray_dir);
						vec4 light_accum = vec4(light.rgb, 1.0) * hddagi_light_ray_weight(cell_delta, direction);
						while (neighbour_bits != 0u) {
							ivec3 offset = hddagi_light_neighbour_direction(findLSB(neighbour_bits));
							float w = hddagi_light_ray_weight(cell_delta + vec3(offset), direction);
							if (w > 0.0) {
								ivec3 neighbour_cell = (read_cell + offset) & (hddagi.grid_size - 1);
								vec3 neighbour_light = hddagi_read_light(neighbour_cell + ivec3(0, (hddagi.grid_size.y * hit_cascade), 0));
								light_accum += vec4(neighbour_light, 1.0) * w;
							}

							neighbour_bits &= neighbour_bits - 1u;
						}

						if (light_accum.a > 0.0) {
							light.rgb = light_accum.rgb / light_accum.a;
						}
					}

				} else {
					//light = vec4(0,0,0,1);
				}
			}

			reflection_light = mix(light, vec4(specular, 1.0 - blend), smoothstep(0, ROUGHNESS_TO_REFLECTION_TRESHOOLD, roughness));
		} else {
			reflection_light.rgb = specular;
			reflection_light.a = 1.0 - blend;
		}

		//ambient_light.rgb = cam_bias * 0.5 + 0.5;
		//reflection_light.rgb = vec3(0.0);

		ambient_light.rgb *= hddagi.energy;
		reflection_light.rgb *= hddagi.energy;
	} else {
		ambient_light = vec4(0);
		reflection_light = vec4(0);
	}
}

//standard voxel cone trace
vec4 voxel_cone_trace(texture3D probe, vec3 cell_size, vec3 pos, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {
	float dist = p_bias;
	vec4 color = vec4(0.0);

	while (dist < max_distance && color.a < 0.95) {
		float diameter = max(1.0, 2.0 * tan_half_angle * dist);
		vec3 uvw_pos = (pos + dist * direction) * cell_size;
		float half_diameter = diameter * 0.5;
		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + half_diameter * cell_size)))) {
			break;
		}
		vec4 scolor = textureLod(sampler3D(probe, linear_sampler_with_mipmaps), uvw_pos, log2(diameter));
		float a = (1.0 - color.a);
		color += a * scolor;
		dist += half_diameter;
	}

	return color;
}

vec4 voxel_cone_trace_45_degrees(texture3D probe, vec3 cell_size, vec3 pos, vec3 direction, float max_distance, float p_bias) {
	float dist = p_bias;
	vec4 color = vec4(0.0);
	float radius = max(0.5, dist);
	float lod_level = log2(radius * 2.0);

	while (dist < max_distance && color.a < 0.95) {
		vec3 uvw_pos = (pos + dist * direction) * cell_size;

		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + radius * cell_size)))) {
			break;
		}
		vec4 scolor = textureLod(sampler3D(probe, linear_sampler_with_mipmaps), uvw_pos, lod_level);
		lod_level += 1.0;

		float a = (1.0 - color.a);
		scolor *= a;
		color += scolor;
		dist += radius;
		radius = max(0.5, dist);
	}
	return color;
}

void voxel_gi_compute(uint index, vec3 position, vec3 normal, vec3 ref_vec, mat3 normal_xform, float roughness, inout vec4 out_spec, inout vec4 out_diff, inout float out_blend) {
	position = (voxel_gi_instances.data[index].xform * vec4(position, 1.0)).xyz;
	ref_vec = normalize((voxel_gi_instances.data[index].xform * vec4(ref_vec, 0.0)).xyz);
	normal = normalize((voxel_gi_instances.data[index].xform * vec4(normal, 0.0)).xyz);

	position += normal * voxel_gi_instances.data[index].normal_bias;

	//this causes corrupted pixels, i have no idea why..
	if (any(bvec2(any(lessThan(position, vec3(0.0))), any(greaterThan(position, voxel_gi_instances.data[index].bounds))))) {
		return;
	}

	mat3 dir_xform = mat3(voxel_gi_instances.data[index].xform) * normal_xform;

	vec3 blendv = abs(position / voxel_gi_instances.data[index].bounds * 2.0 - 1.0);
	float blend = clamp(1.0 - max(blendv.x, max(blendv.y, blendv.z)), 0.0, 1.0);
	//float blend=1.0;

	float max_distance = length(voxel_gi_instances.data[index].octree_size);
	vec3 cell_size = 1.0 / voxel_gi_instances.data[index].octree_size;

	//irradiance

	vec4 light = vec4(0.0);

	if (params.high_quality_vct) {
		const uint cone_dir_count = 6;
		vec3 cone_dirs[cone_dir_count] = vec3[](
				vec3(0.0, 0.0, 1.0),
				vec3(0.866025, 0.0, 0.5),
				vec3(0.267617, 0.823639, 0.5),
				vec3(-0.700629, 0.509037, 0.5),
				vec3(-0.700629, -0.509037, 0.5),
				vec3(0.267617, -0.823639, 0.5));

		float cone_weights[cone_dir_count] = float[](0.25, 0.15, 0.15, 0.15, 0.15, 0.15);
		float cone_angle_tan = 0.577;

		for (uint i = 0; i < cone_dir_count; i++) {
			vec3 dir = normalize(dir_xform * cone_dirs[i]);
			light += cone_weights[i] * voxel_cone_trace(voxel_gi_textures[index], cell_size, position, dir, cone_angle_tan, max_distance, voxel_gi_instances.data[index].bias);
		}
	} else {
		const uint cone_dir_count = 4;
		vec3 cone_dirs[cone_dir_count] = vec3[](
				vec3(0.707107, 0.0, 0.707107),
				vec3(0.0, 0.707107, 0.707107),
				vec3(-0.707107, 0.0, 0.707107),
				vec3(0.0, -0.707107, 0.707107));

		float cone_weights[cone_dir_count] = float[](0.25, 0.25, 0.25, 0.25);
		for (int i = 0; i < cone_dir_count; i++) {
			vec3 dir = normalize(dir_xform * cone_dirs[i]);
			light += cone_weights[i] * voxel_cone_trace_45_degrees(voxel_gi_textures[index], cell_size, position, dir, max_distance, voxel_gi_instances.data[index].bias);
		}
	}

	light.rgb *= voxel_gi_instances.data[index].dynamic_range * voxel_gi_instances.data[index].exposure_normalization;
	if (!voxel_gi_instances.data[index].blend_ambient) {
		light.a = 1.0;
	}

	out_diff += light * blend;

	//radiance
	vec4 irr_light = voxel_cone_trace(voxel_gi_textures[index], cell_size, position, ref_vec, tan(roughness * 0.5 * M_PI * 0.99), max_distance, voxel_gi_instances.data[index].bias);
	irr_light.rgb *= voxel_gi_instances.data[index].dynamic_range * voxel_gi_instances.data[index].exposure_normalization;
	if (!voxel_gi_instances.data[index].blend_ambient) {
		irr_light.a = 1.0;
	}

	out_spec += irr_light * blend;

	out_blend += blend;
}

void process_gi(ivec2 pos, vec3 view_vertex, vec3 view_normal, float roughness, bool dynamic_object, bool trace_reflection, inout vec4 ambient_light, inout vec4 reflection_light) {
	//valid normal, can do GI
	vec3 view = -normalize(mat3(scene_data.cam_transform) * (view_vertex - scene_data.eye_offset[gl_GlobalInvocationID.z].xyz));
	vec3 vertex = mat3(scene_data.cam_transform) * view_vertex;
	vec3 normal = normalize(mat3(scene_data.cam_transform) * view_normal);
	vec3 reflection = normalize(reflect(-view, normal));
#ifdef USE_HDDAGI
	hddagi_process(vertex, normal, reflection, roughness, dynamic_object, trace_reflection, ambient_light, reflection_light);
	if (sc_diffuse_only || (params.world_gi_flags & 3u) == 2u) {
		reflection_light = vec4(0.0);
	}
#endif

#ifdef USE_VOXEL_GI_INSTANCES
	{
#ifdef SAMPLE_VOXEL_GI_NEAREST
		uvec2 voxel_gi_tex = texelFetch(voxel_gi_buffer, pos, 0).rg;
#else
		uvec2 voxel_gi_tex = texelFetch(usampler2D(voxel_gi_buffer, linear_sampler), pos, 0).rg;
#endif
		roughness *= roughness;
		mat3 normal_mat = create_basis_from_normal(normal);

		vec4 amb_accum = vec4(0.0);
		vec4 spec_accum = vec4(0.0);
		float blend_accum = 0.0;

		for (uint i = 0; i < params.max_voxel_gi_instances; i++) {
			if (any(equal(uvec2(i), voxel_gi_tex))) {
				voxel_gi_compute(i, vertex, normal, reflection, normal_mat, roughness, spec_accum, amb_accum, blend_accum);
			}
		}
		if (blend_accum > 0.0) {
			amb_accum /= blend_accum;
			spec_accum /= blend_accum;
		}

#ifdef USE_HDDAGI
		reflection_light = blend_color(spec_accum, reflection_light);
		ambient_light = blend_color(amb_accum, ambient_light);
#else
		reflection_light = spec_accum;
		ambient_light = amb_accum;
#endif
	}
#endif
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

#if defined(USE_HDDAGI) || defined(USE_VOXEL_GI_INSTANCES)

	uint vrs_x, vrs_y;
	bool thread_active = true;
#ifdef USE_VRS
	if (sc_use_vrs) {
		ivec2 vrs_pos;

		// Currently we use a 16x16 texel, possibly some day make this configurable.
		if (sc_half_res) {
			vrs_pos = pos >> 3;
		} else {
			vrs_pos = pos >> 4;
		}

		uint vrs_texel = imageLoad(vrs_buffer, vrs_pos).r;
		// note, valid values for vrs_x and vrs_y are 1, 2 and 4.
		vrs_x = 1 << ((vrs_texel >> 2) & 3);
		vrs_y = 1 << (vrs_texel & 3);

		if (mod(pos.x, vrs_x) != 0) {
			thread_active = false;
		}

		if (mod(pos.y, vrs_y) != 0) {
			thread_active = false;
		}
	}
#endif

	if (thread_active) {
		if (sc_half_res) {
			pos <<= 1;
		}

		if (any(greaterThanEqual(pos, scene_data.screen_size))) { //too large, do nothing
			thread_active = false;
		}
	}

	vec4 ambient_light = vec4(0.0);
	vec4 reflection_light = vec4(0.0);
	float roughness = 1.0;
	vec3 vertex = vec3(0.0);
	vec3 normal = vec3(0.0);
	bool found_vertex = false;
	bool dynamic_object = false;
	bool reuse_reflection = false;
	uint group_pos = gl_LocalInvocationID.y * GROUP_SIZE + gl_LocalInvocationID.x;
	uint reflection_source = (gl_LocalInvocationID.y & ~1u) * GROUP_SIZE + (gl_LocalInvocationID.x & ~1u);
	if (thread_active) {
		vertex = reconstruct_position(pos);
		vec4 normal_roughness = fetch_normal_and_roughness(pos);
		found_vertex = length(normal_roughness.xyz) > 0.5;
		normal = normal_roughness.xyz;
		roughness = normal_roughness.w;
		dynamic_object = roughness > 0.5;
		if (dynamic_object) {
			roughness = 1.0 - roughness;
		}
		roughness /= (127.0 / 255.0);
	}
#ifdef USE_HDDAGI
	group_positions[group_pos] = vertex;
	group_normals[group_pos] = normal;
	group_roughness[group_pos] = thread_active && found_vertex ? roughness : -1.0;
	if (sc_half_reflections) {
		memoryBarrierShared();
		barrier();
		// Reuse one trace only on a continuous 2x2 surface. Mirrors, silhouettes,
		// roughness boundaries, VRS and mixed VoxelGI keep full-rate tracing.
		float source_roughness = group_roughness[reflection_source];
		reuse_reflection = source_roughness > 0.02 && source_roughness < ROUGHNESS_TO_REFLECTION_TRESHOOLD;
		for (uint y = 0u; y < 2u; y++) {
			for (uint x = 0u; x < 2u; x++) {
				uint other = reflection_source + y * GROUP_SIZE + x;
				vec3 delta = group_positions[other] - group_positions[reflection_source];
				reuse_reflection = reuse_reflection && group_roughness[other] >= 0.0 && abs(group_roughness[other] - source_roughness) < 0.015;
				reuse_reflection = reuse_reflection && dot(group_normals[other], group_normals[reflection_source]) > 0.995;
				reuse_reflection = reuse_reflection && abs(delta.z) < max(0.02, abs(group_positions[reflection_source].z) * 0.01);
			}
		}
	}
#endif
	if (thread_active && found_vertex) {
		process_gi(pos, vertex, normal, roughness, dynamic_object, !reuse_reflection || group_pos == reflection_source, ambient_light, reflection_light);
	}
#ifdef USE_HDDAGI
	group_reflections[group_pos] = reflection_light;
#endif

	if (!sc_diffuse_only) {
		memoryBarrierShared();
		barrier();
	}

#ifdef USE_HDDAGI

	if (!sc_diffuse_only && thread_active) {
		if (reuse_reflection) {
			reflection_light = group_reflections[reflection_source];
		} else if (roughness < ROUGHNESS_TO_REFLECTION_TRESHOOLD) {
			uvec2 local_group_pos_base = gl_LocalInvocationID.xy - (gl_LocalInvocationID.xy % DITHER_SIZE);
			uint local_group_pos = local_group_pos_base.y * GROUP_SIZE + local_group_pos_base.x;

			vec3 positions[DITHER_SIZE * DITHER_SIZE];
			vec3 normals[DITHER_SIZE * DITHER_SIZE];

			vec4 average = vec4(0.0);
			bool block_valid = true;
			for (int i = 0; i < DITHER_SIZE; i++) {
				for (int j = 0; j < DITHER_SIZE; j++) {
					uint src_pos = local_group_pos + i * GROUP_SIZE + j;
					block_valid = block_valid && group_roughness[src_pos] >= 0.0;
					normals[i * DITHER_SIZE + j] = group_normals[src_pos];
					positions[i * DITHER_SIZE + j] = group_positions[src_pos];
					average += group_reflections[src_pos];
				}
			}

			average /= 4.0;

			const int subgroup_count = (DITHER_SIZE - 1) * (DITHER_SIZE - 1);
			uvec4 subgroups[subgroup_count] = uvec4[](
#if DITHER_SIZE == 2
					uvec4(0, 1, 2, 3)
#elif DITHER_SIZE == 3
					uvec4(0, 1, 3, 4), uvec4(1, 2, 4, 5), uvec4(3, 4, 6, 7), uvec4(4, 5, 7, 8)
#endif
			);

			const float same_plane_threshold = 0.9659258262890683; // 15 degrees tolerance

			float weight = 1.0;
			for (int i = 0; i < subgroup_count && block_valid; i++) {
				uvec4 sg = subgroups[i];
				// Weight positions in plane.
				vec3 p[4] = vec3[](positions[sg.x], positions[sg.y], positions[sg.z], positions[sg.w]);
				vec3 n1 = normalize(cross(p[0] - p[2], p[0] - p[1]));
				vec3 n2 = normalize(cross(p[2] - p[3], p[2] - p[1]));
				weight *= max(0.0, smoothstep(same_plane_threshold, 1, dot(n1, n2)));

				// Weight normal difference.
				vec3 n[4] = vec3[](normals[sg.x], normals[sg.y], normals[sg.z], normals[sg.w]);
				weight *= max(0.0, smoothstep(same_plane_threshold, 1, length((n[0] + n[1] + n[2] + n[3]) / 4.0)));
			}

			if (block_valid) {
				reflection_light = mix(reflection_light, average, weight);
			}
		}
	}
#endif

	if (thread_active) {
		if (sc_half_res) {
			pos >>= 1;
		}

		uint ambient_rgbe = rgbe_encode(ambient_light.rgb);
		uint reflection_rgbe = rgbe_encode(reflection_light.rgb);
		uint blend = uint(clamp(reflection_light.a * 0xF, 0, 0xF)) | (uint(clamp(ambient_light.a * 0xF, 0, 0xF)) << 4);

		imageStore(ambient_buffer, pos, uvec4(ambient_rgbe));
		imageStore(reflection_buffer, pos, uvec4(reflection_rgbe));
		imageStore(blend_buffer, pos, vec4(ambient_light.a, reflection_light.a, 0, 0));

#ifdef USE_VRS
		if (sc_use_vrs) {
			if (vrs_x > 1) {
				imageStore(ambient_buffer, pos + ivec2(1, 0), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(1, 0), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(1, 0), uvec4(blend));
			}

			if (vrs_x > 2) {
				imageStore(ambient_buffer, pos + ivec2(2, 0), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(2, 0), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(2, 0), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(3, 0), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(3, 0), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(3, 0), uvec4(blend));
			}

			if (vrs_y > 1) {
				imageStore(ambient_buffer, pos + ivec2(0, 1), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(0, 1), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(0, 1), uvec4(blend));
			}

			if (vrs_y > 1 && vrs_x > 1) {
				imageStore(ambient_buffer, pos + ivec2(1, 1), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(1, 1), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(1, 1), uvec4(blend));
			}

			if (vrs_y > 1 && vrs_x > 2) {
				imageStore(ambient_buffer, pos + ivec2(2, 1), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(2, 1), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(2, 1), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(3, 1), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(3, 1), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(3, 1), uvec4(blend));
			}

			if (vrs_y > 2) {
				imageStore(ambient_buffer, pos + ivec2(0, 2), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(0, 2), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(0, 2), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(0, 3), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(0, 3), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(0, 3), uvec4(blend));
			}

			if (vrs_y > 2 && vrs_x > 1) {
				imageStore(ambient_buffer, pos + ivec2(1, 2), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(1, 2), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(1, 2), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(1, 3), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(1, 3), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(1, 3), uvec4(blend));
			}

			if (vrs_y > 2 && vrs_x > 2) {
				imageStore(ambient_buffer, pos + ivec2(2, 2), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(2, 2), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(2, 2), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(2, 3), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(2, 3), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(2, 3), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(3, 2), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(3, 2), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(3, 2), uvec4(blend));

				imageStore(ambient_buffer, pos + ivec2(3, 3), uvec4(ambient_rgbe));
				imageStore(reflection_buffer, pos + ivec2(3, 3), uvec4(reflection_rgbe));
				imageStore(blend_buffer, pos + ivec2(3, 3), uvec4(blend));
			}
		}
#endif
	}
#endif
}
