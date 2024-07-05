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
	uint position; // xyz 10 bit packed - then 2 extra bits for dynamic and static pending
	uint albedo_normal; // 0 - 16, 17 - 31 normal in octahedral format
	uint emission; // RGBE emission
	uint occlusion; // cached 4 bits occlusion for each 8 neighbouring probes
};

#define PROCESS_STATIC_PENDING_BIT 0x80000000
#define PROCESS_DYNAMIC_PENDING_BIT 0x40000000

// Can always write, because it needs to set off dirty bits
layout(set = 0, binding = 5, std430) restrict buffer ProcessVoxels {
	ProcessVoxel data[];
}
process_voxels;

layout(r32ui, set = 0, binding = 6) uniform restrict writeonly uimage3D dst_light;

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

struct Light {
	vec3 color;
	float energy;

	vec3 direction;
	bool has_shadow;

	vec3 position;
	float attenuation;

	uint type;
	float cos_spot_angle;
	float inv_spot_attenuation;
	float radius;
};

layout(set = 0, binding = 8, std140) buffer restrict readonly Lights {
	Light data[];
}
lights;

#ifndef MODE_PROCESS_STATIC
layout(set = 0, binding = 9) uniform texture2DArray lightprobe_texture;
#endif

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

#define REGION_SIZE 8

bool trace_ray_hdda(vec3 ray_pos, vec3 ray_dir, float p_distance, int p_cascade) {
	const int LEVEL_CASCADE = -1;
	const int LEVEL_REGION = 0;
	const int LEVEL_BLOCK = 1;
	const int LEVEL_VOXEL = 2;
	const int MAX_LEVEL = 3;

	const int fp_bits = 10;
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
	float distance = p_distance;
	ivec3 distance_limit;
	bool distance_limit_valid;

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
				distance /= cascades.data[cascade].to_cell;
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

			distance *= cascades.data[cascade].to_cell;

			vec3 box = (vec3(params.grid_size * step) - ray_pos) / ray_dir;
			float advance_to_bounds = min(box.x, min(box.y, box.z));

			if (distance < advance_to_bounds) {
				// Can hit the distance in this cascade?
				distance_limit = pos + ray_sign * ivec3(distance * (1 << fp_bits));
				distance_limit_valid = true;
			} else {
				// We can't so subtract the advance to the end of the cascade.
				distance -= advance_to_bounds;
				distance_limit = ray_sign * 0xFFF << fp_bits; // Unreachable limit
				distance_limit_valid = false;
			}

			cascade_base = ivec3(0, int(params.grid_size.y / REGION_SIZE) * cascade, 0);
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

		if (distance_limit_valid) { // Test against distance limit.
			bvec3 limit = lessThan(pos, distance_limit);
			bvec3 eq = equal(limit, limit_dir);
			if (!all(eq)) {
				break; // Reached limit, break.
			}
		}

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

	return hit;
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

ivec3 modi(ivec3 value, ivec3 p_y) {
	// GLSL Specification says:
	// "Results are undefined if one or both operands are negative."
	// So..
	return mix(value % p_y, p_y - ((abs(value) - ivec3(1)) % p_y) - 1, lessThan(sign(value), ivec3(0)));
}

ivec2 probe_to_tex(ivec3 local_probe) {
	ivec3 cell = modi(cascades.data[params.cascade].region_world_offset + local_probe, params.probe_axis_size);
	return cell.xy + ivec2(0, cell.z * int(params.probe_axis_size.y));
}

void main() {
	uint voxel_index = uint(gl_GlobalInvocationID.x);

	if (voxel_index >= dispatch_data.total_count) {
		return;
	}

#ifdef MODE_PROCESS_STATIC
	// Discard if not marked for static update
	if (!bool(process_voxels.data[voxel_index].position & PROCESS_STATIC_PENDING_BIT)) {
		return;
	}
#else

	//used for skipping voxels every N frames
	if (params.process_increment > 1) {
		if (((voxel_index + params.process_offset) % params.process_increment) != 0) {
			if (params.dirty_dynamic_update && bool(process_voxels.data[voxel_index].position & PROCESS_DYNAMIC_PENDING_BIT)) {
				//saved	because it still needs dynamic update
			} else {
				return;
			}
		}
	}

	if (params.dirty_dynamic_update) {
		process_voxels.data[voxel_index].position &= ~uint(PROCESS_DYNAMIC_PENDING_BIT);
	}
#endif

	// Decode ProcessVoxel

	ivec3 positioni = ivec3((uvec3(process_voxels.data[voxel_index].position) >> uvec3(0, 7, 14)) & uvec3(0x7F));

	vec3 position = vec3(positioni) + vec3(0.5);
	position /= cascades.data[params.cascade].to_cell;
	position += cascades.data[params.cascade].offset;

	uint voxel_albedo = process_voxels.data[voxel_index].albedo_normal;

	vec3 albedo = vec3((uvec3(process_voxels.data[voxel_index].albedo_normal) >> uvec3(0, 5, 11)) & uvec3(0x1F, 0x3F, 0x1F)) / vec3(0x1F, 0x3F, 0x1F);
	vec2 normal_oct = vec2((uvec2(process_voxels.data[voxel_index].albedo_normal) >> uvec2(16, 24)) & uvec2(0xFF, 0xFF)) / vec2(0xFF, 0xFF);
	vec3 normal = octahedron_decode(normal_oct);
	vec3 emission = rgbe_decode(process_voxels.data[voxel_index].emission);
	uint occlusionu = process_voxels.data[voxel_index].occlusion;

	vec3 light_accum = vec3(0.0);

	// Add indirect light first, in order to save computation resources
#ifndef MODE_PROCESS_STATIC

	if (params.bounce_feedback > 0.001) {
		vec3 feedback = albedo * params.bounce_feedback;
		ivec3 base_probe = positioni / params.probe_cell_size;
		vec2 probe_tex_to_uv = 1.0 / vec2((LIGHTPROBE_OCT_SIZE + 2) * params.probe_axis_size.x, (LIGHTPROBE_OCT_SIZE + 2) * params.probe_axis_size.y * params.probe_axis_size.z);

		for (int i = 0; i < 8; i++) {
			float weight = float((occlusionu >> (i * 4)) & 0xF) / float(0xF); //precached occlusion
			if (weight == 0.0) {
				// Do not waste time.
				continue;
			}
			ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec2 tex_pos = probe_to_tex(probe);
			vec2 tex_uv = vec2(ivec2(tex_pos * (LIGHTPROBE_OCT_SIZE + 2) + ivec2(1))) + normal_oct * float(LIGHTPROBE_OCT_SIZE);
			tex_uv *= probe_tex_to_uv;
			vec3 light = texture(sampler2DArray(lightprobe_texture, linear_sampler), vec3(tex_uv, float(params.cascade))).rgb;
			light_accum += light * weight;
		}

		light_accum *= feedback;
	}
#endif

	// Raytrace light

	for (uint i = 0; i < params.light_count; i++) {
		float attenuation = 1.0;
		vec3 direction;
		float light_distance = 1e20;

		switch (lights.data[i].type) {
			case LIGHT_TYPE_DIRECTIONAL: {
				direction = -lights.data[i].direction;
				attenuation *= max(0.0, dot(normal, direction));
			} break;
			case LIGHT_TYPE_OMNI: {
				vec3 rel_vec = lights.data[i].position - position;
				direction = normalize(rel_vec);
				light_distance = length(rel_vec);
				rel_vec.y /= params.y_mult;
				attenuation = get_omni_attenuation(light_distance, 1.0 / lights.data[i].radius, lights.data[i].attenuation);
				attenuation *= max(0.0, dot(normal, direction));

			} break;
			case LIGHT_TYPE_SPOT: {
				vec3 rel_vec = lights.data[i].position - position;
				direction = normalize(rel_vec);

				light_distance = length(rel_vec);
				rel_vec.y /= params.y_mult;
				attenuation = get_omni_attenuation(light_distance, 1.0 / lights.data[i].radius, lights.data[i].attenuation);
				attenuation *= max(0.0, dot(normal, direction));

				float cos_spot_angle = lights.data[i].cos_spot_angle;
				float cos_angle = dot(-direction, lights.data[i].direction);

				if (cos_angle < cos_spot_angle) {
					continue;
				}

				float scos = max(cos_angle, cos_spot_angle);
				float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cos_spot_angle));
				attenuation *= 1.0 - pow(spot_rim, lights.data[i].inv_spot_attenuation);
			} break;
		}

		if (attenuation < 0.001) {
			continue;
		}

		bool hit = false;

		vec3 ray_pos = position;
		vec3 ray_dir = direction;
		vec3 inv_dir = 1.0 / ray_dir;

		/* No bias needed for voxels
		float cell_size = 1.0 / cascades.data[params.cascade].to_cell;
		ray_pos += sign(direction) * cell_size * 0.48; // go almost to the box edge but remain inside
		ray_pos += ray_dir * 0.4 * cell_size; //apply a small bias from there
		*/

		if (lights.data[i].has_shadow) {
			hit = trace_ray_hdda(ray_pos, ray_dir, light_distance, params.cascade);
		}

		if (!hit) {
			light_accum += albedo * lights.data[i].color.rgb * lights.data[i].energy * attenuation;
		}
	}

	light_accum += emission;

#if 0
	vec3 an = normal;
	if (an.x < 0) {
		an.x = an.x * -0.25;
	}
	if (an.y < 0) {
		an.y = an.y * -0.25;
	}
	if (an.z < 0) {
		an.z = an.z * -0.25;
	}
	light_accum = an;
#endif

#ifdef MODE_PROCESS_STATIC
	// Add to self, since its static.
	process_voxels.data[voxel_index].emission = rgbe_encode(light_accum.rgb);
	process_voxels.data[voxel_index].position &= ~uint(PROCESS_STATIC_PENDING_BIT); // Clear process static bit.
#else

	// Store to light texture
	positioni = (positioni + cascades.data[params.cascade].region_world_offset * REGION_SIZE) & (params.grid_size - 1);
	positioni.y += int(params.cascade) * params.grid_size.y;
	imageStore(dst_light, positioni, uvec4(rgbe_encode(light_accum.rgb)));

#endif
}
