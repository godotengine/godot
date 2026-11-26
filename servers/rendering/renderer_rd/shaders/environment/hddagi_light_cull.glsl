#[compute]

#version 450

#VERSION_DEFINES

#include "hddagi_light_inc.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std140) restrict readonly buffer Lights {
	Light data[];
}
lights;
layout(set = 0, binding = 1, std430) restrict writeonly buffer LightRegions {
	uint masks[];
}
regions;
layout(push_constant, std430) uniform Params {
	vec3 offset;
	float region_size;
	ivec3 region_count;
	uint light_count;
	float y_mult;
	uint pad0;
	uint pad1;
	uint pad2;
}
params;

void main() {
	uint word_count = (params.light_count + 31u) / 32u;
	uint index = gl_GlobalInvocationID.x / word_count;
	uint word = gl_GlobalInvocationID.x % word_count;
	if (index >= uint(params.region_count.x * params.region_count.y * params.region_count.z)) {
		return;
	}
	ivec3 region = ivec3(int(index) % params.region_count.x, (int(index) / params.region_count.x) % params.region_count.y, int(index) / (params.region_count.x * params.region_count.y));
	vec3 lower = params.offset + vec3(region) * params.region_size;
	vec3 upper = lower + vec3(params.region_size);
	lower.y /= params.y_mult;
	upper.y /= params.y_mult;
	uint mask = 0u;
	for (uint bit = 0u; bit < 32u && word * 32u + bit < params.light_count; bit++) {
		Light light = lights.data[word * 32u + bit];
		bool intersects = light.type == 0u;
		if (!intersects) {
			vec3 delta = max(max(lower - light.position, light.position - upper), vec3(0.0));
			float radius = light.radius;
			if (light.type == 3u) {
				radius += 0.5 * length(vec2(length(light.area_width.xyz), length(light.area_height.xyz)));
			}
			intersects = dot(delta, delta) <= radius * radius;
		}
		if (intersects) {
			mask |= 1u << bit;
		}
	}
	regions.masks[index * 4u + word] = mask;
}
