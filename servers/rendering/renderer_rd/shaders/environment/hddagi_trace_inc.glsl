vec3 hddagi_trace_position(int cascade) {
#ifdef HDDAGI_TRACE_REFLECTION
	return hddagi.cascades[cascade].position;
#else
	return cascades.data[cascade].offset;
#endif
}

float hddagi_trace_to_cell(int cascade) {
#ifdef HDDAGI_TRACE_REFLECTION
	return hddagi.cascades[cascade].to_cell;
#else
	return cascades.data[cascade].to_cell;
#endif
}

ivec3 hddagi_trace_region_offset(int cascade) {
#ifdef HDDAGI_TRACE_REFLECTION
	return hddagi.cascades[cascade].region_world_offset;
#else
	return cascades.data[cascade].region_world_offset;
#endif
}

#ifdef HDDAGI_TRACE_OCCLUSION
bool trace_ray_hdda(vec3 ray_pos, vec3 ray_dir, float p_distance, int p_cascade) {
#else
bool trace_ray_hdda(vec3 ray_pos, vec3 ray_dir, int p_cascade, out ivec3 r_cell, out ivec3 r_side, out int r_cascade) {
#endif
#ifdef HDDAGI_TRACE_REFLECTION
	ivec3 grid_size = ivec3(hddagi.grid_size);
	int max_cascades = hddagi.max_cascades;
#else
	ivec3 grid_size = ivec3(params.grid_size);
	int max_cascades = int(params.max_cascades);
#endif
	const int LEVEL_CASCADE = -1;
	const int LEVEL_REGION = 0;
	const int LEVEL_BLOCK = 1;
	const int LEVEL_VOXEL = 2;

#ifdef HDDAGI_TRACE_PROBES
	const int fp_bits = 8;
#elif defined(HQ_RAY) && !defined(HDDAGI_TRACE_OCCLUSION)
	const int fp_bits = 16;
#else
const int fp_bits = 10;
#endif
	const int fp_block_bits = fp_bits + 2;
	const int fp_region_bits = fp_block_bits + 1;

	bvec3 limit_dir = greaterThan(ray_dir, vec3(0.0));
	ivec3 step = mix(ivec3(0), ivec3(1), limit_dir);
	ivec3 ray_sign = ivec3(sign(ray_dir));

	ivec3 ray_dir_fp = ivec3(ray_dir * float(1 << fp_bits));

#if defined(HQ_RAY) && !defined(HDDAGI_TRACE_PROBES) && !defined(HDDAGI_TRACE_OCCLUSION)
	const float limit = 1.0 / 65535.0;
#else
	const float limit = 1.0 / 127.0;
#endif
	bvec3 ray_zero = lessThan(abs(ray_dir), vec3(limit));
	ivec3 inv_ray_dir_fp = ivec3(float(1 << fp_bits) / ray_dir);

	const int region_mask = (1 << fp_region_bits) - 1;
	const int block_mask = (1 << fp_block_bits) - 1;
	const int voxel_mask = (1 << fp_bits) - 1;

	ivec3 region_offset_mask = (grid_size / REGION_SIZE) - ivec3(1);

	ivec3 region_limit = ((grid_size << fp_bits) - ivec3(1)) * step;

	int level = LEVEL_CASCADE;
	int cascade = p_cascade - 1;

	ivec3 cascade_base;
	ivec3 region_base;
	uint region_blocks;
	uvec2 block;
	bool hit = false;

	ivec3 pos;
#ifdef HDDAGI_TRACE_OCCLUSION
	float distance = p_distance;
	ivec3 distance_limit;
	bool distance_limit_valid;
#elif defined(HDDAGI_TRACE_REFLECTION)
	ivec3 scroll_limit;
#endif

	while (true) {
		if (level == LEVEL_CASCADE) {
			if (cascade >= p_cascade) {
				ray_pos = vec3(pos) / float(1 << fp_bits);
				ray_pos /= hddagi_trace_to_cell(cascade);
				ray_pos += hddagi_trace_position(cascade);
#ifdef HDDAGI_TRACE_OCCLUSION
				distance /= hddagi_trace_to_cell(cascade);
#endif
			}

			cascade++;
			if (cascade == max_cascades) {
				break;
			}

			ray_pos -= hddagi_trace_position(cascade);
			ray_pos *= hddagi_trace_to_cell(cascade);
			pos = ivec3(ray_pos * float(1 << fp_bits));
			if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, grid_size << fp_bits))) {
				continue;
			}

#ifdef HDDAGI_TRACE_OCCLUSION
			distance *= hddagi_trace_to_cell(cascade);

			vec3 box = (vec3(grid_size * step) - ray_pos) / ray_dir;
			float advance_to_bounds = min(box.x, min(box.y, box.z));

			if (distance < advance_to_bounds) {
				distance_limit = pos + ray_sign * ivec3(distance * (1 << fp_bits));
				distance_limit_valid = true;
			} else {
				distance -= advance_to_bounds;
				distance_limit = ray_sign * 0xFFF << fp_bits; // Unreachable limit
				distance_limit_valid = false;
			}

#endif
			cascade_base = ivec3(0, grid_size.y / REGION_SIZE * cascade, 0);
			level = LEVEL_REGION;
#ifdef HDDAGI_TRACE_REFLECTION
			{
				vec3 box = (vec3(grid_size * step) - ray_pos) / ray_dir;
				vec3 axis = vec3(1, 0, 0);
				float m = box.x;
				if (box.y < m) {
					m = box.y;
					axis = vec3(0, 1, 0);
				}

				if (box.z < m) {
					axis = vec3(0, 0, 1);
				}

				vec3 half_size = vec3(grid_size) / 2.0;
				vec3 inner_pos = -hddagi_trace_position(cascade) * hddagi_trace_to_cell(cascade) - half_size;

				float inner_dir = dot(axis, inner_pos);
				float blend = abs(inner_dir) / float(REGION_SIZE * 0.5);

				scroll_limit = region_limit;
				if (bayer_dither(blend)) {
					scroll_limit += (ivec3(axis * sign(inner_dir)) * REGION_SIZE) << fp_bits;
				}
			}
#endif
		}
		if (level == LEVEL_REGION) {
			ivec3 region = pos >> fp_region_bits;
			region = (hddagi_trace_region_offset(cascade) + region) & region_offset_mask; // Scroll to world
			region += cascade_base;
			region_blocks = imageLoad(voxel_region_cascades, region).r;

			if (region_blocks != 0u) {
				region_base = (region << 1);
				level = LEVEL_BLOCK;
			}
		}
		if (level == LEVEL_BLOCK) {
			ivec3 block_local = (pos & ivec3(region_mask)) >> fp_block_bits;
			uint block_index = uint(block_local.z * 4 + block_local.y * 2 + block_local.x);
			if (bool(region_blocks & (1u << block_index))) {
				block = imageLoad(voxel_cascades, region_base + block_local).rg;
				level = LEVEL_VOXEL;
			}
		}
		if (level == LEVEL_VOXEL) {
			ivec3 block_local = (pos & ivec3(block_mask)) >> fp_bits;
			uint block_index = uint(block_local.z * 16 + block_local.y * 4 + block_local.x);
			uint occupied = block_index < 32u ? block.x : block.y;
			if ((occupied & (1u << (block_index & 31u))) != 0u) {
				hit = true;
				break;
			}
		}

		ivec3 mask = ivec3(level == LEVEL_REGION ? region_mask : (level == LEVEL_BLOCK ? block_mask : voxel_mask));
		ivec3 box = mask * step;
		ivec3 pos_diff = box - (pos & mask);
#if defined(HDDAGI_TRACE_PROBES) || defined(HDDAGI_TRACE_OCCLUSION)
		// Preserve the original sentinel precision for probe and shadow rays.
		ivec3 tv = mix((pos_diff * inv_ray_dir_fp), ivec3(0x7FFFFFFF), ray_zero) >> fp_bits;
#else
#if defined(HQ_RAY) && !defined(HDDAGI_TRACE_PROBES) && !defined(HDDAGI_TRACE_OCCLUSION)
		ivec3 mul_res = mul64(pos_diff, inv_ray_dir_fp, fp_bits);
#else
		ivec3 mul_res = (pos_diff * inv_ray_dir_fp) >> fp_bits;
#endif
		ivec3 tv = mix(mul_res, ivec3(0x7FFFFFFF), ray_zero);
#endif
		int t = min(tv.x, min(tv.y, tv.z));

		ivec3 adv_box = pos_diff + ray_sign;
#if defined(HQ_RAY) && !defined(HDDAGI_TRACE_PROBES) && !defined(HDDAGI_TRACE_OCCLUSION)
		ivec3 adv_t = mul64(ray_dir_fp, ivec3(t), fp_bits);
#else
		ivec3 adv_t = (ray_dir_fp * t) >> fp_bits;
#endif
		ivec3 previous_position = pos;
		pos += mix(adv_t, adv_box, equal(ivec3(t), tv));

#ifdef HDDAGI_TRACE_OCCLUSION
		if (distance_limit_valid) { // Test against distance limit.
			bvec3 limit = lessThan(pos, distance_limit);
			bvec3 eq = equal(limit, limit_dir);
			if (!all(eq)) {
				break; // Reached limit, break.
			}
		}

#elif defined(HDDAGI_TRACE_REFLECTION)
		{ // Test against scroll limit.
			bvec3 limit = lessThan(pos, scroll_limit);
			bvec3 eq = equal(limit, limit_dir);
			if (!all(eq)) {
				level = LEVEL_CASCADE;
				pos = mix(scroll_limit, pos, eq);
				continue;
			}
		}

#endif
		// The highest changed coordinate bit determines which parent cell was crossed.
		ivec3 changed = pos ^ previous_position;
		uint crossed = uint(changed.x | changed.y | changed.z);
		level = min(level, crossed > uint(region_mask) ? LEVEL_REGION : (crossed > uint(block_mask) ? LEVEL_BLOCK : LEVEL_VOXEL));
		if (level == LEVEL_REGION && any(notEqual(lessThan(pos, region_limit), limit_dir))) {
			level = LEVEL_CASCADE;
		}
	}

#ifndef HDDAGI_TRACE_OCCLUSION
	if (hit) {
		ivec3 mask = ivec3(voxel_mask);
		ivec3 box = mask * (step ^ ivec3(1));
		ivec3 pos_diff = box - (pos & mask);
#if defined(HQ_RAY) && !defined(HDDAGI_TRACE_PROBES) && !defined(HDDAGI_TRACE_OCCLUSION)
		ivec3 mul_res = mul64(pos_diff, -inv_ray_dir_fp, fp_bits);
#else
		ivec3 mul_res = (pos_diff * -inv_ray_dir_fp);
#endif

		ivec3 tv = mix(mul_res, ivec3(0x7FFFFFFF), ray_zero);

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

#endif
	return hit;
}
