#[compute]

#version 450

#VERSION_DEFINES

#include "hddagi_voxel_metadata_inc.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define MAX_CASCADES 8
#define REGION_SIZE 8

layout(rg32ui, set = 0, binding = 1) uniform restrict readonly uimage3D voxel_cascades;
layout(r8ui, set = 0, binding = 2) uniform restrict readonly uimage3D voxel_region_cascades;
layout(set = 0, binding = 3) uniform texture3D light_cascades;
layout(set = 0, binding = 4) uniform sampler linear_sampler;
layout(r32ui, set = 0, binding = 5) uniform restrict readonly uimage3D voxel_neighbours;

struct CascadeData {
	vec3 offset;
	float to_cell;
	ivec3 region_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 6, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(rgba16f, set = 0, binding = 7) uniform restrict writeonly image2D screen_buffer;

layout(push_constant, std430) uniform Params {
	vec3 grid_size;
	uint max_cascades;

	vec3 ray_origin;
	float y_mult;
	vec3 ray_direction;
	uint pad;
	vec3 ray_x;
	uint pad2;
	vec3 ray_y;
	uint pad3;
	vec3 origin_x;
	uint pad4;
	vec3 origin_y;
	uint pad5;
}
params;

#include "hddagi_trace_inc.glsl"

vec3 linear_to_srgb(vec3 color) {
	color = clamp(color, vec3(0.0), vec3(1.0));
	return mix(1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, 12.92 * color, lessThan(color, vec3(0.0031308)));
}

float point_to_ray_distance(vec3 point, vec3 ray_origin, vec3 ray_direction) {
	vec3 projected = ray_origin + ray_direction * dot(point - ray_origin, ray_direction);
	return length(point - projected);
}

void main() {
	ivec2 screen_pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 screen_size = imageSize(screen_buffer);
	if (any(greaterThanEqual(screen_pos, screen_size))) {
		return;
	}

	vec2 uv = (vec2(screen_pos) + 0.5) / vec2(screen_size);
	vec3 ray_pos = params.ray_origin + params.origin_x * uv.x + params.origin_y * uv.y;
	vec3 ray_dir = params.ray_direction + params.ray_x * uv.x + params.ray_y * uv.y;
	ray_pos.y *= params.y_mult;
	ray_dir.y *= params.y_mult;
	ray_dir = normalize(ray_dir);

	vec3 light = vec3(0.0);
	ivec3 hit_cell;
	ivec3 hit_face;
	int hit_cascade;
	if (trace_ray_hdda(ray_pos, ray_dir, 0, hit_cell, hit_face, hit_cascade)) {
		hit_cell += hit_face;
		ivec3 read_cell = (hit_cell + cascades.data[hit_cascade].region_world_offset * REGION_SIZE) & (ivec3(params.grid_size) - 1);
		ivec3 cascade_offset = ivec3(0, int(params.grid_size.y) * hit_cascade, 0);
		light = texelFetch(sampler3D(light_cascades, linear_sampler), read_cell + cascade_offset, 0).rgb;

		uint neighbour_bits = imageLoad(voxel_neighbours, read_cell + cascade_offset).r & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
		float to_cell = cascades.data[hit_cascade].to_cell;
		vec3 light_cell_pos = (vec3(hit_cell) + 0.5) / to_cell + cascades.data[hit_cascade].offset;
		vec4 light_accum = vec4(light, 1.0) * max(0.0, 1.0 - point_to_ray_distance(light_cell_pos, ray_pos, ray_dir) * to_cell);
		while (neighbour_bits != 0u) {
			uint index = findLSB(neighbour_bits);
			ivec3 offset = hddagi_light_neighbour_direction(index);
			vec3 neighbour_pos = light_cell_pos + vec3(offset) / to_cell;
			float weight = max(0.0, 1.0 - point_to_ray_distance(neighbour_pos, ray_pos, ray_dir) * to_cell);
			if (weight > 0.0) {
				read_cell = (hit_cell + offset + cascades.data[hit_cascade].region_world_offset * REGION_SIZE) & (ivec3(params.grid_size) - 1);
				vec3 neighbour_light = texelFetch(sampler3D(light_cascades, linear_sampler), read_cell + cascade_offset, 0).rgb;
				light_accum += vec4(neighbour_light, 1.0) * weight;
			}
			neighbour_bits &= ~(1u << index);
		}
		if (light_accum.a > 0.0) {
			light = light_accum.rgb / light_accum.a;
		}
	}

	imageStore(screen_buffer, screen_pos, vec4(linear_to_srgb(light), 1.0));
}
