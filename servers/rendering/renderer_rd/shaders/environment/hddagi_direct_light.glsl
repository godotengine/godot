#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define MAX_CASCADES 8

layout(rg32ui, set = 0, binding = 1) uniform restrict readonly uimage3D voxel_cascades;
layout(r8ui, set = 0, binding = 2) uniform restrict readonly uimage3D voxel_region_cascades;

layout(set = 0, binding = 3) uniform sampler linear_sampler;

layout(set = 0, binding = 4, std430) restrict readonly buffer DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;

struct ProcessVoxel {
	uint position; // xyz 7 bits each, plus dynamic and static pending flags
	uint albedo_normal; // 0 - 16, 17 - 31 normal in octahedral format
	uint emission; // RGBE emission
	uint occlusion; // cached 4 bits occlusion for each 8 neighboring probes
};

#define PROCESS_STATIC_PENDING_BIT 0x80000000
#define PROCESS_DYNAMIC_PENDING_BIT 0x40000000

// Can always write, because it needs to set off dirty bits
layout(set = 0, binding = 5, std430) restrict buffer ProcessVoxels {
	ProcessVoxel data[];
}
process_voxels;

#define HDDAGI_DATA_BINDING 6
#define HDDAGI_PAGE_BINDING 14
#define HDDAGI_WRITE_VOXELS
#include "hddagi_radiance_inc.glsl"
#include "hddagi_voxel_metadata_inc.glsl"
#include "hddagi_voxel_pages_inc.glsl"

#ifdef MODE_PROCESS_STATIC
layout(set = 0, binding = 11, std430) restrict readonly buffer SourceEmission {
	uint data[];
}
source_emission;
#endif

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 region_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 7, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2
#define LIGHT_TYPE_AREA 3

#include "../area_lights_inc.glsl"
#include "hddagi_light_inc.glsl"

layout(set = 0, binding = 8, std140) buffer restrict readonly Lights {
	Light data[];
}
lights;

#ifndef MODE_PROCESS_STATIC
layout(set = 0, binding = 9) uniform texture2DArray lightprobe_texture;
#endif

#ifdef MODE_PROCESS_DYNAMIC
layout(set = 0, binding = 13, std430) restrict readonly buffer LightRegions {
	uint masks[];
}
light_regions;
#endif

layout(set = 1, binding = 0) uniform texture2D area_light_atlas;
layout(set = 1, binding = 1) uniform sampler linear_sampler_with_mipmaps;

layout(push_constant, std430) uniform Params {
	ivec3 grid_size;
	int max_cascades;

	int cascade;
	uint light_count;
	uint process_offset;
	uint process_increment;

	float bounce_feedback;
	float y_mult;
	bool use_occlusion;
	int probe_cell_size;

	ivec3 probe_axis_size;
	bool dirty_dynamic_update;
	uint use_light_regions;
	bool refresh_static;
	uint pad1;
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

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

void compute_area_light(uint index, vec3 position, vec3 normal, out float attenuation, out vec3 light_vec, out float light_distance, out vec3 texture_color) {
	vec3 area_width = lights.data[index].area_width.xyz;
	vec3 area_height = lights.data[index].area_height.xyz;
	vec3 area_direction = lights.data[index].direction;
	vec3 area_width_normalized = normalize(area_width);
	vec3 area_height_normalized = normalize(area_height);
	vec3 light_center = lights.data[index].position;
	vec3 light_to_position = position - light_center;
	vec2 position_on_light = vec2(dot(light_to_position, area_width_normalized), dot(light_to_position, area_height_normalized));
	vec2 half_size = vec2(length(area_width), length(area_height)) * 0.5;
	position_on_light = clamp(position_on_light, -half_size, half_size);

	vec3 closest_point_on_light = light_center + position_on_light.x * area_width_normalized + position_on_light.y * area_height_normalized;
	vec3 light_relative_vector = closest_point_on_light - position;
	light_distance = length(light_relative_vector);
	light_vec = light_distance > 0.0001 ? light_relative_vector / light_distance : -area_direction;

	if (light_distance * lights.data[index].inv_spot_attenuation >= 1.0) {
		attenuation = 0.0;
		return;
	}

	vec3 half_width = area_width * 0.5;
	vec3 half_height = area_height * 0.5;
	vec3 light_points[4];
	light_points[0] = light_center - half_width - half_height - position;
	light_points[1] = light_center + half_width - half_height - position;
	light_points[2] = light_center + half_width + half_height - position;
	light_points[3] = light_center - half_width + half_height - position;

	// LTC already includes inverse-square falloff.
	attenuation = get_omni_attenuation(light_distance, 1.0 / lights.data[index].radius, lights.data[index].attenuation - 2.0);
	float ltc_diffuse = 0.0;
	ltc_evaluate_diff(normal, light_points, lights.data[index].area_projector_rect, lights.data[index].cos_spot_angle, area_light_atlas, linear_sampler_with_mipmaps, ltc_diffuse, texture_color);
	attenuation *= ltc_diffuse;
}

#define REGION_SIZE 8

#define HDDAGI_TRACE_OCCLUSION
#include "hddagi_trace_inc.glsl"

void main() {
	uint voxel_index = uint(gl_GlobalInvocationID.x);
#ifndef MODE_PROCESS_STATIC
	uint phase_mask = params.process_increment - 1u;
	if (!params.dirty_dynamic_update) {
		voxel_index = voxel_index * params.process_increment + ((0u - params.process_offset) & phase_mask);
	}
#endif

	if (voxel_index >= dispatch_data.total_count) {
		return;
	}

#ifdef MODE_PROCESS_STATIC
	// Discard if not marked for static update
	if (!params.refresh_static && !bool(process_voxels.data[voxel_index].position & PROCESS_STATIC_PENDING_BIT)) {
		return;
	}
#else

	if (params.dirty_dynamic_update) {
		if (((voxel_index + params.process_offset) & phase_mask) != 0u && (process_voxels.data[voxel_index].position & PROCESS_DYNAMIC_PENDING_BIT) == 0u) {
			return;
		}
		process_voxels.data[voxel_index].position &= ~uint(PROCESS_DYNAMIC_PENDING_BIT);
	}
#endif

	// Decode ProcessVoxel

	ivec3 positioni = ivec3((uvec3(process_voxels.data[voxel_index].position) >> uvec3(0, 7, 14)) & uvec3(0x7F));

	vec3 grid_position = (vec3(positioni) + 0.5) / cascades.data[params.cascade].to_cell + cascades.data[params.cascade].offset;
	vec3 world_position = grid_position;
	world_position.y /= params.y_mult;

	uint voxel_albedo = process_voxels.data[voxel_index].albedo_normal;

	vec3 albedo = vec3((uvec3(process_voxels.data[voxel_index].albedo_normal) >> uvec3(0, 5, 11)) & uvec3(0x1F, 0x3F, 0x1F)) / vec3(0x1F, 0x3F, 0x1F);
	vec2 normal_oct = vec2((uvec2(process_voxels.data[voxel_index].albedo_normal) >> uvec2(16, 24)) & uvec2(0xFF, 0xFF)) / vec2(0xFF, 0xFF);
	vec3 normal = octahedron_decode(normal_oct);
#ifdef MODE_PROCESS_STATIC
	vec3 emission = rgbe_decode(source_emission.data[voxel_index]);
#else
	vec3 emission = rgbe_decode(process_voxels.data[voxel_index].emission);
#endif
	uint occlusionu = process_voxels.data[voxel_index].occlusion;

	int cascade = params.cascade;

	vec3 light_accum = vec3(0.0);

	// Add indirect light first, in order to save computation resources
#ifndef MODE_PROCESS_STATIC

	if (params.bounce_feedback > 0.001) {
		vec3 feedback = albedo * params.bounce_feedback;
		ivec3 base_probe = hddagi_wrap_probe(cascades.data[cascade].region_world_offset + positioni / params.probe_cell_size, params.probe_axis_size);
		ivec3 next_probe = base_probe + 1;
		next_probe = mix(next_probe, ivec3(0), equal(next_probe, params.probe_axis_size));
		vec3 probe_fraction = fract((vec3(positioni) + 0.5) / float(params.probe_cell_size));
		vec2 probe_tex_to_uv = 1.0 / vec2((LIGHTPROBE_OCT_SIZE + 2) * params.probe_axis_size.x, (LIGHTPROBE_OCT_SIZE + 2) * params.probe_axis_size.y * params.probe_axis_size.z);

		float weight_sum = 0.0;
		for (int i = 0; i < 8; i++) {
			float weight = float((occlusionu >> (i * 4)) & 0xF);
			if (weight == 0.0) {
				continue;
			}
			ivec3 corner = (ivec3(i) >> ivec3(0, 1, 2)) & 1;
			vec3 to_probe = vec3(corner) - probe_fraction;
			vec3 trilinear = 1.0 - abs(to_probe);
			to_probe.y /= params.y_mult;
			weight *= trilinear.x * trilinear.y * trilinear.z * max(0.005, dot(normal, normalize(to_probe)));
			ivec3 probe = mix(base_probe, next_probe, bvec3(corner));
			ivec2 tex_pos = probe.xy + ivec2(0, probe.z * params.probe_axis_size.y);
			vec2 tex_uv = vec2(ivec2(tex_pos * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1))) + normal_oct * float(LIGHTPROBE_OCT_SIZE);
			tex_uv *= probe_tex_to_uv;
			vec3 light = texture(sampler2DArray(lightprobe_texture, linear_sampler), vec3(tex_uv, float(params.cascade))).rgb;
			light_accum += light * weight;
			weight_sum += weight;
		}

		light_accum *= feedback / max(weight_sum, 1e-8);
	}
#endif

	// Raytrace light

#ifdef MODE_PROCESS_DYNAMIC
	ivec3 region = positioni / params.probe_cell_size;
	ivec3 region_count = params.grid_size / params.probe_cell_size;
	uint region_index = uint(region.x + region_count.x * (region.y + region_count.y * region.z));
	uint word = 0u;
	uint mask = 0u;
#endif
	for (uint light_index = 0; light_index < params.light_count; light_index++) {
		uint i = light_index;
#ifdef MODE_PROCESS_DYNAMIC
		if (params.use_light_regions != 0u) {
			while (mask == 0u && word < (params.light_count + 31u) / 32u) {
				mask = light_regions.masks[region_index * 4u + word++];
			}
			if (mask == 0u) {
				break;
			}
			i = (word - 1u) * 32u + uint(findLSB(mask));
			mask &= mask - 1u;
		}
#endif
		float attenuation = 1.0;
		vec3 direction;
		float light_distance = 1e20;
		vec3 texture_color = vec3(1.0);

		switch (lights.data[i].type) {
			case LIGHT_TYPE_DIRECTIONAL: {
				direction = -lights.data[i].direction;
				attenuation *= max(0.0, dot(normal, direction));
			} break;
			case LIGHT_TYPE_OMNI:
			case LIGHT_TYPE_SPOT: {
				vec3 to_light = lights.data[i].position - world_position;
				light_distance = length(to_light);
				direction = to_light / max(light_distance, 1e-6);
				attenuation = get_omni_attenuation(light_distance, 1.0 / lights.data[i].radius, lights.data[i].attenuation);
				attenuation *= max(0.0, dot(normal, direction));

				if (lights.data[i].type == LIGHT_TYPE_SPOT) {
					float cos_spot_angle = lights.data[i].cos_spot_angle;
					float cos_angle = dot(-direction, lights.data[i].direction);
					if (cos_angle <= cos_spot_angle) {
						continue;
					}
					float spot_rim = max(0.0001, (1.0 - cos_angle) / (1.0 - cos_spot_angle));
					attenuation *= 1.0 - pow(spot_rim, lights.data[i].inv_spot_attenuation);
				}
			} break;
			case LIGHT_TYPE_AREA: {
				vec3 area_width = lights.data[i].area_width.xyz;
				vec3 area_height = lights.data[i].area_height.xyz;
				if (dot(area_width, area_width) < 1e-7 || dot(area_height, area_height) < 1e-7) {
					continue;
				}
				if (dot(lights.data[i].direction, world_position - lights.data[i].position) <= 0.0) {
					continue;
				}
				compute_area_light(i, world_position, normal, attenuation, direction, light_distance, texture_color);
			} break;
			default: {
				continue;
			}
		}

		if (attenuation < 0.001) {
			continue;
		}

		bool hit = false;

		if (lights.data[i].has_shadow) {
			vec3 trace_direction = direction * vec3(1.0, params.y_mult, 1.0);
			float trace_scale = length(trace_direction);
			hit = trace_ray_hdda(grid_position, trace_direction / trace_scale, light_distance * trace_scale, cascade);
		}

		if (!hit) {
			light_accum += albedo * lights.data[i].color.rgb * texture_color * lights.data[i].energy * attenuation;
		}
	}

	light_accum += emission;

#ifdef MODE_PROCESS_STATIC
	process_voxels.data[voxel_index].emission = rgbe_encode(light_accum.rgb);
	process_voxels.data[voxel_index].position = (process_voxels.data[voxel_index].position & ~uint(PROCESS_STATIC_PENDING_BIT)) | uint(PROCESS_DYNAMIC_PENDING_BIT);
#else

	// Store to light texture
	positioni = (positioni + cascades.data[params.cascade].region_world_offset * REGION_SIZE) & (params.grid_size - 1);
	positioni.y += int(params.cascade) * params.grid_size.y;
	hddagi_voxel_data.cells[hddagi_voxel_index(positioni)].x = rgbe_encode(light_accum.rgb);

#endif
}
