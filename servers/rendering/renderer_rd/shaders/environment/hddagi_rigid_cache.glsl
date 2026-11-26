#[compute]

#version 450

#VERSION_DEFINES

#include "hddagi_radiance_inc.glsl"
#include "hddagi_voxel_metadata_inc.glsl"

#if defined(MODE_CAPTURE) || defined(MODE_RESTORE)
layout(r16ui, set = 0, binding = 0) uniform restrict readonly uimage3D src_albedo;
layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_emission;
layout(r32ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_emission_aniso;
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_normals;
#endif

#if defined(MODE_PROJECT) || defined(MODE_RESTORE) || defined(MODE_CLEAR)
layout(r16ui, set = 0, binding = 4) uniform restrict writeonly uimage3D dst_albedo;
layout(r32ui, set = 0, binding = 5) uniform restrict uimage3D emission_grid;
layout(r32ui, set = 0, binding = 6) uniform restrict uimage3D emission_aniso_grid;
layout(r32ui, set = 0, binding = 7) uniform restrict uimage3D dst_normals;
#endif

#if defined(MODE_CAPTURE) || defined(MODE_PROJECT)
layout(set = 0, binding = 8, std430) restrict
#ifdef MODE_PROJECT
		readonly
#endif
		buffer SurfaceCache {
	uvec4 dispatch;
	uvec4 grid;
	uint words[];
}
cache;

uint material_address(uvec3 cell, uvec3 grid) {
	uvec3 block = cell >> 1u;
	uvec3 size = grid >> 1u;
	return grid.x * grid.y * grid.z * 2u + 5u * (block.x + size.x * (block.y + size.y * block.z));
}
#endif

#ifdef MODE_CAPTURE
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main() {
	ivec3 p = ivec3(gl_GlobalInvocationID);
	if (any(greaterThanEqual(p, imageSize(src_normals)))) {
		return;
	}
	uvec3 grid = uvec3(imageSize(src_normals));
	if (all(equal(p, ivec3(0)))) {
		cache.grid = uvec4(grid, 0);
	}
	if (all(equal(p & 1, ivec3(0)))) {
		uint base = material_address(uvec3(p), grid);
		ivec3 material = p >> 1;
		cache.words[base + 3u] = imageLoad(src_emission, material).r;
		cache.words[base + 4u] = imageLoad(src_emission_aniso, material).r;
		material.z *= 6;
		for (int i = 0; i < 3; i++) {
			cache.words[base + uint(i)] = imageLoad(src_albedo, material + ivec3(0, 0, i * 2)).r | (imageLoad(src_albedo, material + ivec3(0, 0, i * 2 + 1)).r << 16u);
		}
	}
	uint facing = imageLoad(src_normals, p).r;
	if (facing != 0u) {
		uint index = atomicAdd(cache.dispatch.w, 1u);
		if ((index & 63u) == 0u) {
			atomicAdd(cache.dispatch.x, 1u);
		}
		cache.words[index * 2u] = uint(p.x | (p.y << 8) | (p.z << 16));
		cache.words[index * 2u + 1u] = facing;
	}
}
#endif

#if defined(MODE_RESTORE) || defined(MODE_CLEAR)
#ifdef MODE_RESTORE
#define LOAD_STATIC(image, pos) imageLoad(image, pos)
#else
#define LOAD_STATIC(image, pos) uvec4(0u)
#endif
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(push_constant, std430) uniform Params {
	ivec4 from;
	ivec4 offset;
	ivec3 size;
	float exposure;
}
params;

void main() {
	ivec3 pos = ivec3(gl_GlobalInvocationID);
	if (any(greaterThanEqual(pos, params.size.xyz))) {
		return;
	}
	ivec3 src = params.offset.xyz + pos;
	ivec3 dst = params.from.xyz + pos;
	imageStore(dst_normals, dst, LOAD_STATIC(src_normals, src));
	if (all(equal(dst & 1, ivec3(0)))) {
		src >>= 1;
		dst >>= 1;
		uint emission = LOAD_STATIC(src_emission, src).r;
		if (emission != 0u && params.exposure != 1.0) {
			emission = rgbe_encode(rgbe_decode(emission) * params.exposure);
		}
		imageStore(emission_grid, dst, uvec4(emission));
		imageStore(emission_aniso_grid, dst, LOAD_STATIC(src_emission_aniso, src));
		src.z *= 6;
		dst.z *= 6;
		for (int i = 0; i < 6; i++) {
			imageStore(dst_albedo, dst + ivec3(0, 0, i), LOAD_STATIC(src_albedo, src + ivec3(0, 0, i)));
		}
	}
}
#endif

#ifdef MODE_PROJECT
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(push_constant, std430) uniform Params {
	vec4 to_cell[3];
	mat3 normal;
	ivec3 from;
	uint subsamples;
	ivec3 end;
	float exposure;
}
params;

#include "hddagi_emission_inc.glsl"

vec3 hddagi_rotate_emission(vec3 lobe) {
	vec3 world = params.normal * lobe;
	return world * sqrt(dot(lobe, lobe) / max(dot(world, world), 1e-20));
}

void main() {
	uint index = gl_GlobalInvocationID.x;
	if (index >= cache.dispatch.w) {
		return;
	}
	uint pos = cache.words[index * 2u];
	uint source_facing = cache.words[index * 2u + 1u];
	uvec3 cell = (uvec3(pos) >> uvec3(0, 8, 16)) & 255u;
	uint base = material_address(cell, cache.grid.xyz);
	uint facing = source_facing & 63u;
	uint normals = source_facing >> 6u;
	uint paired = facing & (facing >> 1u) & 21u;
	uint split = paired != 0u ? uint(findLSB(paired)) / 2u : 0u;
	vec3 a = vec3(0.0);
	vec3 b = vec3(0.0);
	while (normals != 0u) {
		vec3 n = hddagi_surface_normal(uint(findLSB(normals)));
		if (paired != 0u && n[split] < 0.0) {
			b += n;
		} else {
			a += n;
		}
		normals &= normals - 1u;
	}
	uint emission = cache.words[base + 3u];
	vec3 negative = vec3(0.0);
	vec3 positive = vec3(0.0);
	if (emission != 0u) {
		uvec3 pairs = uvec3(cache.words[base + 4u]) >> HDDAGI_EMISSION_SHIFTS;
		vec3 neg = vec3(pairs & 31u) / 31.0;
		vec3 pos = vec3((pairs >> 5u) & 31u) / 31.0;
		vec3 front = pos - neg;
		vec3 back = vec3(0.0);
		if (any(greaterThan(min(pos, neg), vec3(0.0)))) {
			front = mix(-neg, pos, greaterThanEqual(a, vec3(0.0)));
			back = mix(-neg, pos, greaterThanEqual(dot(b, b) > 0.0 ? b : -a, vec3(0.0)));
		}
		front = hddagi_rotate_emission(front);
		back = hddagi_rotate_emission(back);
		negative = max(-front, -back);
		positive = max(front, back);
		emission = rgbe_encode(rgbe_decode(emission) * params.exposure);
	}
	a = params.normal * a;
	b = params.normal * b;
	a *= inversesqrt(max(dot(a, a), 1e-15));
	b *= inversesqrt(max(dot(b, b), 1e-15));
	uint world_facing = 0u;
	for (uint i = 0u; i < 26u; i++) {
		vec3 direction = hddagi_surface_normal(i);
		float strength = max(dot(a, direction), dot(b, direction));
		if (strength > 0.7) {
			world_facing |= 1u << (i + 6u);
		}
		if (i < 6u && strength > 0.001) {
			world_facing |= 1u << i;
		}
	}
	uint colors[6];
	for (uint i = 0u; i < 6u; i++) {
		if ((world_facing & (1u << i)) == 0u) {
			continue;
		}
		vec3 axis = transpose(params.normal) * vec3(hddagi_axis_direction(i));
		float best = -1e20;
		uint face = 0u;
		for (uint j = 0u; j < 6u; j++) {
			float weight = axis[j >> 1u] * (((j & 1u) == 0u) ? -1.0 : 1.0);
			if ((facing & (1u << j)) != 0u && weight > best) {
				best = weight;
				face = j;
			}
		}
		colors[i] = (cache.words[base + (face >> 1u)] >> ((face & 1u) * 16u)) & 65535u;
	}

	uint steps = params.subsamples;
	for (uint z = 0u; z < steps; z++) {
		for (uint y = 0u; y < steps; y++) {
			for (uint x = 0u; x < steps; x++) {
				vec4 sample_pos = vec4(vec3(cell) + (vec3(x, y, z) + 0.5) / float(steps), 1.0);
				ivec3 dst = ivec3(floor(vec3(dot(params.to_cell[0], sample_pos), dot(params.to_cell[1], sample_pos), dot(params.to_cell[2], sample_pos))));
				if (any(lessThan(dst, params.from)) || any(greaterThanEqual(dst, params.end))) {
					continue;
				}
				imageAtomicOr(dst_normals, dst, world_facing);
				ivec3 material = dst >> 1;
				if (emission != 0u) {
					hddagi_store_emission(material, emission, negative, positive);
				}
				material.z *= 6;
				uint sides = world_facing & 63u;
				while (sides != 0u) {
					uint i = uint(findLSB(sides));
					imageStore(dst_albedo, material + ivec3(0, 0, i), uvec4(colors[i]));
					sides &= sides - 1u;
				}
			}
		}
	}
}
#endif
