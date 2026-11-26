#ifndef HDDAGI_VOXEL_METADATA_INC
#define HDDAGI_VOXEL_METADATA_INC

// Surface neighbors exclude the eight body diagonals. The remaining fields
// mark light cells, empty axial paths and the six empty adjacent voxels.
const uint HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK = (1u << 18u) - 1u;
const uint HDDAGI_LIGHT_CELL_VALID_BIT = 1u << 18u;
const uint HDDAGI_OCCLUSION_PATH_SHIFT = 19u;
const uint HDDAGI_DISOCCLUSION_SHIFT = 22u;

// Axis pairs follow the voxel metadata order: -X, +X, -Y, +Y, -Z, +Z.
ivec3 hddagi_axis_direction(uint index) {
	uint axis = index >> 1u;
	int direction = int(index & 1u) * 2 - 1;
	return ivec3(axis == 0u ? direction : 0, axis == 1u ? direction : 0, axis == 2u ? direction : 0);
}

const uvec3 HDDAGI_SURFACE_POSITIVE = uvec3(0x3fc0002u, 0x3833808u, 0x252a520u);
const uvec3 HDDAGI_SURFACE_NEGATIVE = uvec3(0x3fc1u, 0x1cc1c4u, 0xa54a50u);

ivec3 hddagi_surface_direction(uint index) {
	uvec3 positive = (HDDAGI_SURFACE_POSITIVE >> uvec3(index)) & uvec3(1u);
	uvec3 negative = (HDDAGI_SURFACE_NEGATIVE >> uvec3(index)) & uvec3(1u);
	return ivec3(positive) - ivec3(negative);
}

vec3 hddagi_surface_normal(uint index) {
	float scale = index < 6u ? 1.0 : (((0x2942940u >> index) & 1u) != 0u ? 0.5773502691896258 : 0.7071067811865475);
	return vec3(hddagi_surface_direction(index)) * scale;
}

// Positive/negative bitplanes preserve the 18 light-neighbor metadata bits.
ivec3 hddagi_light_neighbour_direction(uint index) {
	uvec3 positive = (uvec3(0x3c002u, 0x23208u, 0x12920u) >> uvec3(index)) & uvec3(1u);
	uvec3 negative = (uvec3(0x3c1u, 0x4c44u, 0x9490u) >> uvec3(index)) & uvec3(1u);
	return ivec3(positive) - ivec3(negative);
}

float hddagi_light_ray_weight(vec3 cell_delta, vec3 ray_direction) {
	return max(0.0, 1.0 - length(cell_delta - ray_direction * dot(cell_delta, ray_direction)));
}

ivec3 hddagi_wrap_probe(ivec3 position, ivec3 size) {
	ivec3 sign_mask = position >> 31;
	ivec3 remainder = ivec3(uvec3(position ^ sign_mask) % uvec3(size));
	return mix(remainder, size - 1 - remainder, lessThan(position, ivec3(0)));
}

float hddagi_occlusion_weight(vec4 first, vec4 second, int index) {
	vec4 weights = index < 4 ? first : second;
	return (index & 2) == 0 ? ((index & 1) == 0 ? weights.x : weights.y) : ((index & 1) == 0 ? weights.z : weights.w);
}

uint hddagi_unpack_disocclusion(uint metadata) {
	return (metadata >> HDDAGI_DISOCCLUSION_SHIFT) & 0x3fu;
}

// Four samples per eight-voxel probe interval, with a sample one voxel away
// from either side of the seam.
vec3 hddagi_occlusion_cache_sample_position(ivec3 cell) {
	return vec3(cell * 2) + 1.0;
}

vec3 hddagi_occlusion_cache_warp(vec3 local) {
	vec3 distance = mix(local, local - 8.0, greaterThan(local, vec3(4.0)));
	// Match the one-voxel visibility transition at the block face.
	vec3 reconstructed = sign(distance) * min(abs(distance) * 2.0, vec3(1.0));
	vec3 shift = mix(vec3(0.0), reconstructed - distance, lessThan(abs(distance), vec3(1.0)));
	return (local + shift) * 0.5;
}

#endif
