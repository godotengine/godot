#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_SPECULAR_TRACE)
vec3 sample_environment(vec3 ray_direction_view) {
	if (params.sky_mode == SCREEN_PROBE_SKY_COLOR) {
		return max(params.sky_color.rgb * params.sky_energy, vec3(0.0));
	}
	if (params.sky_mode != SCREEN_PROBE_SKY_TEXTURE) {
		return vec3(0.0);
	}

	float border = clamp(params.sky_color.w, 0.0, 0.499);
	float sky_energy = params.sky_energy;
	vec3 sky_direction = scene_data.radiance_inverse_xform * ray_direction_view;
	float direction_length_squared = dot(sky_direction, sky_direction);
	if (!(direction_length_squared > 1e-8) || any(isnan(sky_direction)) || any(isinf(sky_direction))) {
		return vec3(0.0);
	}
	sky_direction *= inversesqrt(direction_length_squared);
	vec2 sky_uv = vec3_to_oct_with_border(sky_direction, vec2(border, 1.0 - border * 2.0));
#ifdef USE_RADIANCE_OCTMAP_ARRAY
	vec3 radiance = textureLod(sampler2DArray(sky_radiance, sky_sampler), vec3(sky_uv, 0.0), 0.0).rgb * sky_energy;
#else
	vec3 radiance = textureLod(sampler2D(sky_radiance, sky_sampler), sky_uv, 0.0).rgb * sky_energy;
#endif
	if (any(isnan(radiance)) || any(isinf(radiance))) {
		return vec3(0.0);
	}
	return max(radiance, vec3(0.0));
}
#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_SPECULAR_TRACE)

#ifdef MODE_SPECULAR_TRACE
bool detail_trace_geometric_normal(ivec2 pixel, vec3 center_view, vec3 shading_normal_view, bool shading_normal_valid, out vec3 r_normal_view) {
	vec3 left_view;
	vec3 right_view;
	vec3 up_view;
	vec3 down_view;
	bool left_valid = detail_trace_load_view_position(pixel + ivec2(-1, 0), left_view);
	bool right_valid = detail_trace_load_view_position(pixel + ivec2(1, 0), right_view);
	bool up_valid = detail_trace_load_view_position(pixel + ivec2(0, -1), up_view);
	bool down_valid = detail_trace_load_view_position(pixel + ivec2(0, 1), down_view);

	vec3 horizontal_derivative = vec3(0.0);
	vec3 vertical_derivative = vec3(0.0);
	bool horizontal_valid = left_valid || right_valid;
	bool vertical_valid = up_valid || down_valid;
	if (horizontal_valid) {
		bool use_left = left_valid && (!right_valid || abs(left_view.z - center_view.z) <= abs(right_view.z - center_view.z));
		horizontal_derivative = use_left ? center_view - left_view : right_view - center_view;
	}
	if (vertical_valid) {
		bool use_up = up_valid && (!down_valid || abs(up_view.z - center_view.z) <= abs(down_view.z - center_view.z));
		vertical_derivative = use_up ? center_view - up_view : down_view - center_view;
	}

	vec3 geometric_normal = cross(vertical_derivative, horizontal_derivative);
	float length_squared = dot(geometric_normal, geometric_normal);
	if (horizontal_valid && vertical_valid && length_squared > 1e-12 && !any(isnan(geometric_normal)) && !any(isinf(geometric_normal))) {
		geometric_normal *= inversesqrt(length_squared);
		if (shading_normal_valid) {
			geometric_normal *= dot(geometric_normal, shading_normal_view) < 0.0 ? -1.0 : 1.0;
		} else if (dot(geometric_normal, center_view) > 0.0) {
			geometric_normal = -geometric_normal;
		}
		r_normal_view = geometric_normal;
		return true;
	}
	if (shading_normal_valid) {
		r_normal_view = shading_normal_view;
		return true;
	}
	r_normal_view = vec3(0.0, 0.0, 1.0);
	return false;
}

#endif

bool trace_screen_detail(vec3 origin_view, vec3 ray_direction_view, float distance_limit, out vec3 r_endpoint_view, out vec3 r_endpoint_normal_view) {
	r_endpoint_view = vec3(0.0);
	r_endpoint_normal_view = vec3(0.0, 0.0, 1.0);

#ifdef MODE_SPECULAR_TRACE
	float max_distance = min(SCREEN_PROBE_SPECULAR_DETAIL_TRACE_MAX_DISTANCE, distance_limit);
#else
	float max_distance = min(SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE, distance_limit);
#endif
	if ((params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) == 0u || !(max_distance > 0.0) || params.detail_trace_mip_count == 0u || any(lessThanEqual(params.screen_size, ivec2(0)))) {
		return false;
	}
	if (any(isnan(ray_direction_view)) || any(isinf(ray_direction_view))) {
		return false;
	}
	if (ray_direction_view.z > 0.0) {
		max_distance = min(max_distance, (-0.001 - origin_view.z) / ray_direction_view.z);
	}
	if (!(max_distance > 0.02)) {
		return false;
	}

	vec3 screen_start;
	vec3 screen_end;
	if (!detail_trace_project(origin_view, screen_start) || !detail_trace_project(origin_view + ray_direction_view * max_distance, screen_end)) {
		return false;
	}
	vec3 screen_delta = screen_end - screen_start;
	if (abs(screen_delta.z) < 1e-5 || all(lessThan(abs(screen_delta.xy), vec2(1e-7)))) {
		return false;
	}

	float segment_t = abs(screen_delta.z);
	vec3 screen_direction = screen_delta / segment_t;
	bool facing_camera = screen_direction.z >= 0.0;
	vec2 direction_xy = screen_direction.xy;
	vec2 safe_direction_xy = mix(direction_xy, vec2(1e-20), lessThan(abs(direction_xy), vec2(1e-20)));
	vec2 inverse_direction_xy = 1.0 / safe_direction_xy;
	vec2 cell_step = mix(vec2(1.0), vec2(-1.0), lessThan(direction_xy, vec2(0.0)));
	vec2 positive_cell_step = clamp(cell_step, vec2(0.0), vec2(1.0));
	vec2 cell_boundary_bias = cell_step * 1e-6;
	vec2 detail_size = vec2(params.screen_size);
	vec2 inverse_detail_size = 1.0 / detail_size;

	vec2 edge_t0 = -screen_start.xy * inverse_direction_xy;
	vec2 edge_t1 = (vec2(1.0) - screen_start.xy) * inverse_direction_xy;
	vec2 positive_edge_t = max(edge_t0, edge_t1);
	float trace_t_max = min(segment_t, min(positive_edge_t.x, positive_edge_t.y));
	if (!(trace_t_max > 0.0)) {
		return false;
	}

	vec2 start_cell = floor(screen_start.xy * detail_size);
	vec2 next_cell_uv = (start_cell + positive_cell_step) * inverse_detail_size + cell_boundary_bias;
	vec2 start_cell_t = (next_cell_uv - screen_start.xy) * inverse_direction_xy;
	float trace_t = max(min(start_cell_t.x, start_cell_t.y), 0.0);
	int max_mip = int(params.detail_trace_mip_count) - 1;
	uint detail_trace_quality = (params.flags & SCREEN_PROBE_DETAIL_TRACE_QUALITY_MASK) >> SCREEN_PROBE_DETAIL_TRACE_QUALITY_SHIFT;
#ifdef MODE_SPECULAR_TRACE
	int mip_level = min(2, max_mip);
	int iterations_remaining = detail_trace_quality == 1u ? 32 : (detail_trace_quality == 2u ? 16 : 50);
	float receiver_exclusion = max(0.01, abs(origin_view.z) / detail_size.y);
	const float candidate_normal_threshold = -0.01;
#else
	int mip_level = min(1, max_mip);
	int iterations_remaining = detail_trace_quality == 1u ? 16 : (detail_trace_quality == 2u ? 8 : 24);
	float receiver_exclusion = max(0.002, abs(origin_view.z) * 1.5 / detail_size.y);
#endif

	while (mip_level >= 0 && iterations_remaining > 0 && trace_t < trace_t_max) {
		vec3 current_screen = screen_start + screen_direction * trace_t;
		ivec2 mip_size = max(params.screen_size >> mip_level, ivec2(1));
		ivec2 cell_index = clamp(ivec2(floor(current_screen.xy * vec2(mip_size))), ivec2(0), mip_size - ivec2(1));
		float cell_depth = texelFetch(sampler2D(detail_hiz_buffer, linear_sampler), cell_index, mip_level).r;

		vec2 next_mip_uv = (vec2(cell_index) + positive_cell_step) / vec2(mip_size) + cell_boundary_bias;
		vec2 mip_cell_t = (next_mip_uv - screen_start.xy) * inverse_direction_xy;
		float cell_exit_t = min(mip_cell_t.x, mip_cell_t.y);

		bool possible_hit = cell_depth > 0.0;
		float depth_t = trace_t;
		if (possible_hit) {
			depth_t = (cell_depth - screen_start.z) / screen_direction.z;
			possible_hit = facing_camera ? trace_t <= depth_t : depth_t <= cell_exit_t;
		}

		if (possible_hit && mip_level == 0) {
#ifdef MODE_SPECULAR_TRACE
			float candidate_t = facing_camera ? trace_t : max(trace_t, depth_t);
#else
			float candidate_t = clamp(depth_t, trace_t, max(trace_t, cell_exit_t - 1e-7));
#endif
			vec3 candidate_screen = screen_start + screen_direction * candidate_t;
			ivec2 candidate_pixel = ivec2(floor(candidate_screen.xy * detail_size));
			bool candidate_valid = all(greaterThanEqual(candidate_pixel, ivec2(1))) && all(lessThan(candidate_pixel, params.screen_size - ivec2(1)));
			vec3 candidate_view = vec3(0.0);
			vec3 candidate_normal_view = vec3(0.0, 0.0, 1.0);
			if (candidate_valid) {
				float candidate_depth = any(notEqual(candidate_pixel, cell_index)) ? texelFetch(sampler2D(detail_hiz_buffer, linear_sampler), candidate_pixel, 0).r : cell_depth;
				candidate_valid = candidate_depth > 0.0;
				if (candidate_valid) {
					vec2 candidate_uv = (vec2(candidate_pixel) + 0.5) * inverse_detail_size;
					candidate_view = compute_view_position(vec3(candidate_uv, candidate_depth));
					candidate_valid = !any(isnan(candidate_view)) && !any(isinf(candidate_view));
				}
				if (candidate_valid) {
					float candidate_distance = dot(candidate_view - origin_view, ray_direction_view);
#ifdef MODE_SPECULAR_TRACE
					float thickness = clamp(abs(candidate_view.z) * 0.004, 0.04, 0.25);
#else
					float thickness = clamp(abs(candidate_view.z) * 0.0015, 0.025, 0.10);
#endif
					vec3 ray_closest_view = origin_view + ray_direction_view * candidate_distance;
					vec3 ray_error = candidate_view - ray_closest_view;
					candidate_valid = candidate_distance > receiver_exclusion && candidate_distance <= max_distance && dot(ray_error, ray_error) <= thickness * thickness;
					if (candidate_valid) {
						vec3 shading_normal_view;
						bool shading_normal_valid = decode_normal(texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), candidate_pixel, 0).xyz, shading_normal_view);
#ifdef MODE_SPECULAR_TRACE
						candidate_valid = detail_trace_geometric_normal(candidate_pixel, candidate_view, shading_normal_view, shading_normal_valid, candidate_normal_view) && dot(ray_direction_view, candidate_normal_view) < candidate_normal_threshold;
#else
						candidate_normal_view = shading_normal_view;
						candidate_valid = shading_normal_valid && abs(dot(ray_direction_view, candidate_normal_view)) > 0.05 &&
								abs(dot(candidate_view - origin_view, candidate_normal_view)) > receiver_exclusion;
#endif
					}
					if (candidate_valid) {
						vec3 candidate_ray_view = compute_view_position(candidate_screen);
						float normal_thickness_error = abs(dot(candidate_view - candidate_ray_view, candidate_normal_view));
						candidate_valid = normal_thickness_error <= thickness && !any(isnan(candidate_ray_view)) && !any(isinf(candidate_ray_view));
					}
				}
			}

			if (candidate_valid) {
				r_endpoint_view = candidate_view;
				r_endpoint_normal_view = candidate_normal_view;
				return true;
			}

			trace_t = max(cell_exit_t, candidate_t + 1e-6);
			mip_level = min(2, max_mip);
			iterations_remaining--;
			continue;
		}
		if (possible_hit) {
			mip_level--;
			if (!facing_camera) {
				trace_t = max(trace_t, depth_t);
			}
		} else {
			trace_t = max(cell_exit_t, trace_t + 1e-6);
			mip_level = min(mip_level + 1, max_mip);
		}
		iterations_remaining--;
	}
	return false;
}

#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_SPECULAR_TRACE)

uvec2 load_voxel_cascade(ivec3 position) {
	return imageLoad(hddagi_voxel_cascades, position).rg;
}

uint load_voxel_region(ivec3 position) {
	return imageLoad(hddagi_voxel_region_cascades, position).r;
}

uint load_voxel_neighbours(ivec3 position) {
	return hddagi_read_neighbours(position);
}

uint load_voxel_disocclusion(ivec3 position) {
	return hddagi_unpack_disocclusion(load_voxel_neighbours(position));
}

#ifdef MODE_DIRECTIONAL_TRACE
bool load_probe_surface(ivec2 probe_position, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal) {
	uvec4 packed = imageLoad(screen_probe_surface_input, probe_position);
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	r_depth = uintBitsToFloat(packed.z);
	r_normal = unpack_surface_normal(packed.w);
	return r_depth > 0.0 && all(greaterThanEqual(r_screen_position, ivec2(0))) && all(lessThan(r_screen_position, params.screen_size));
}
#endif

bool trace_ray_hdda(vec3 ray_origin, vec3 ray_direction, int first_cascade, bool exclude_initial_cell, out ivec3 r_cell, out ivec3 r_side, out int r_cascade, out float r_time) {
	ivec3 grid_size = hddagi.grid_size;
	ivec3 forward = ivec3(greaterThan(ray_direction, vec3(0.0)));
	ivec3 ray_sign = ivec3(sign(ray_direction));
	bvec3 active_axis = greaterThan(abs(ray_direction), vec3(1e-20));
	vec3 inverse_direction = 1.0 / mix(vec3(1.0), ray_direction, active_axis);
	ivec3 world_cell = ivec3(0);
	bool crossed_cascade = false;

	for (int cascade = first_cascade; cascade < hddagi.max_cascades; cascade++) {
		vec3 origin = (ray_origin - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
		ivec3 region_offset = hddagi.cascades[cascade].region_world_offset;
		ivec3 grid_offset = region_offset * HDDAGI_REGION_SIZE - grid_size / 2;
		ivec3 cell;
		if (crossed_cascade) {
			world_cell >>= 1;
			cell = world_cell - grid_offset;
		} else {
			vec3 lower = floor(origin);
			cell = ivec3(lower) - ivec3(equal(origin, lower)) * ivec3(lessThan(ray_direction, vec3(0.0)));
		}
		if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, grid_size))) {
			continue;
		}

		ivec3 initial_cell = cell;
		bool skip_initial = exclude_initial_cell && !crossed_cascade;
		ivec3 cascade_offset = ivec3(0, grid_size.y / HDDAGI_REGION_SIZE * cascade, 0);
		ivec3 cached_region = ivec3(-1);
		ivec3 cached_block = ivec3(-1);
		uint region_bits = 0u;
		uvec2 block_bits = uvec2(0u);
		while (true) {
			ivec3 region = cell >> 3;
			ivec3 block = cell >> 2;
			if (any(notEqual(region, cached_region))) {
				region_bits = load_voxel_region(((region + region_offset) & (grid_size / 8 - 1)) + cascade_offset);
				cached_region = region;
			}

			int level = 3;
			if (region_bits != 0u) {
				level = 2;
				ivec3 local_block = block & 1;
				int block_index = local_block.x + 2 * local_block.y + 4 * local_block.z;
				if ((region_bits & (1u << block_index)) != 0u) {
					if (any(notEqual(block, cached_block))) {
						block_bits = load_voxel_cascade(((block + 2 * region_offset) & (grid_size / 4 - 1)) + 2 * cascade_offset);
						cached_block = block;
					}
					ivec3 local_cell = cell & 3;
					int index = local_cell.x + 4 * local_cell.y + 16 * local_cell.z;
					uint word = index < 32 ? block_bits.x : block_bits.y;
					level = (word & (0x00330033u << (index & 10))) == 0u ? 1 : 0;
					if ((word & (1u << (index & 31))) != 0u && (!skip_initial || any(notEqual(cell, initial_cell)))) {
						vec3 entry = mix(vec3(-3.402823e38), (vec3(cell + (1 - forward)) - origin) * inverse_direction, active_axis);
						r_side = entry.x > entry.y ? ivec3(1, 0, 0) : ivec3(0, 1, 0);
						if (entry.z > max(entry.x, entry.y)) {
							r_side = ivec3(0, 0, 1);
						}
						r_side *= -ray_sign;
						r_time = max(entry.x, max(entry.y, entry.z)) / hddagi.cascades[cascade].to_cell;
						r_cell = cell;
						r_cascade = cascade;
						return true;
					}
				}
			}

			int width = 1 << level;
			ivec3 lower = (cell >> level) << level;
			ivec3 boundary = lower + forward * width;
			vec3 exit_times = mix(vec3(3.402823e38), (vec3(boundary) - origin) * inverse_direction, active_axis);
			float t = min(exit_times.x, min(exit_times.y, exit_times.z));
			ivec3 next_cell = ivec3(floor(fma(ray_direction, vec3(t), origin)));
			cell = mix(clamp(next_cell, lower, lower + width - 1), boundary + min(ray_sign, ivec3(0)), equal(exit_times, vec3(t)));
			if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, grid_size))) {
				// Hand off the cell across the boundary without rounding the intersection a second time.
				world_cell = cell + grid_offset;
				crossed_cascade = true;
				break;
			}
		}
	}
	return false;
}

#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_RESOLVE)
ivec3 light_texture_position(ivec3 light_cell, int cascade) {
	ivec3 read_cell = (light_cell + hddagi.cascades[cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
	return read_cell + ivec3(0, hddagi.grid_size.y * cascade, 0);
}

bool query_endpoint_radiance(vec3 camera_relative_endpoint, vec3 endpoint_normal_world, out vec3 radiance) {
	camera_relative_endpoint.y *= hddagi.y_mult;
	vec3 normal = normalize(endpoint_normal_world * vec3(1.0, hddagi.y_mult, 1.0));
	uvec3 inward = uvec3(0u, 2u, 4u) + uvec3(lessThan(normal, vec3(0.0)));
	for (int cascade = 0; cascade < hddagi.max_cascades; cascade++) {
		vec3 cell = (camera_relative_endpoint - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(hddagi.grid_size)))) {
			continue;
		}
		vec3 sample_position = cell + normal * 0.75 - 0.5;
		ivec3 base = ivec3(floor(sample_position));
		vec3 blend = fract(sample_position);
		vec4 sum = vec4(0.0);
		for (int i = 0; i < 8; i++) {
			ivec3 offset = ivec3(i & 1, (i >> 1) & 1, i >> 2);
			ivec3 tap = base + offset;
			if (any(lessThan(tap, ivec3(0))) || any(greaterThanEqual(tap, hddagi.grid_size))) {
				continue;
			}
			ivec3 coord = light_texture_position(tap, cascade);
			uint metadata = hddagi_read_neighbours(coord);
			if ((metadata & HDDAGI_LIGHT_CELL_VALID_BIT) == 0u) {
				continue;
			}
			uint empty = hddagi_unpack_disocclusion(metadata);
			uvec3 faces = ((uvec3(empty) >> (inward ^ 1u)) & (uvec3(~empty) >> inward)) & 1u;
			float facing = dot(abs(normal), vec3(faces));
			vec3 weights = mix(1.0 - blend, blend, vec3(offset));
			float weight = weights.x * weights.y * weights.z * facing;
			sum += vec4(hddagi_read_light(coord), 1.0) * weight;
		}
		radiance = sum.rgb * (hddagi.cascades[cascade].exposure_normalization / max(sum.a, 1e-5));
		return true;
	}
	return false;
}
#endif

#if defined(MODE_DIRECTIONAL_TRACE) || defined(MODE_SPECULAR_TRACE)

bool trace_hddagi_radiance(ivec2 origin_position, float origin_depth, vec3 origin_normal, vec3 ray_direction_view, out vec3 r_radiance, out float r_hit_distance, out float r_hit_cell_size) {
	r_hit_distance = 0.0;
	r_hit_cell_size = 0.0;
	r_radiance = vec3(0.0);
	vec2 origin_uv = (vec2(origin_position) + 0.5) / vec2(params.screen_size);
	vec3 origin_view = compute_view_position(vec3(origin_uv, origin_depth));
	ray_direction_view = normalize(ray_direction_view);
	if (any(isnan(origin_view)) || any(isinf(origin_view)) || any(isnan(ray_direction_view)) || any(isinf(ray_direction_view))) {
		return false;
	}
	mat3 camera_basis = mat3(scene_data.cam_transform);
	vec3 ray_position = camera_basis * origin_view;
	vec3 trace_direction_world = camera_basis * ray_direction_view;
	vec3 trace_normal_world = camera_basis * origin_normal;

#ifdef MODE_DIRECTIONAL_TRACE
	vec3 endpoint_view;
	vec3 endpoint_normal_view;
	if (trace_screen_detail(origin_view, ray_direction_view, SCREEN_PROBE_DETAIL_TRACE_MAX_DISTANCE, endpoint_view, endpoint_normal_view) &&
			query_endpoint_radiance(camera_basis * endpoint_view, camera_basis * faceforward(endpoint_normal_view, ray_direction_view, endpoint_normal_view), r_radiance)) {
		return true;
	}
#endif
	vec3 ray_direction = trace_direction_world;
	vec3 hddagi_normal = trace_normal_world;
	ray_position.y *= hddagi.y_mult;
	ray_direction.y *= hddagi.y_mult;
	hddagi_normal.y *= hddagi.y_mult;
	ray_direction = normalize(ray_direction);
	hddagi_normal = normalize(hddagi_normal);
	if (any(isnan(ray_direction)) || any(isinf(ray_direction)) || any(isnan(hddagi_normal)) || any(isinf(hddagi_normal))) {
		return false;
	}

	int cascade = hddagi.max_cascades;
	for (int i = 0; i < hddagi.max_cascades; i++) {
		vec3 cascade_position = (ray_position - hddagi.cascades[i].position) * hddagi.cascades[i].to_cell;
		if (all(greaterThanEqual(cascade_position, vec3(0.0))) && all(lessThan(cascade_position, vec3(hddagi.grid_size)))) {
			cascade = i;
			break;
		}
	}
	if (cascade >= hddagi.max_cascades) {
		return false;
	}

	vec3 start_cell = (ray_position - hddagi.cascades[cascade].position) * hddagi.cascades[cascade].to_cell;
	vec3 absolute_normal = abs(hddagi_normal);
	float dominant_normal = max(absolute_normal.x, max(absolute_normal.y, absolute_normal.z));
	if (!(dominant_normal > 1e-6)) {
		return false;
	}
	vec3 ray_bias = hddagi_normal / dominant_normal;
	start_cell += ray_bias * params.normal_bias;
	ray_position = start_cell / hddagi.cascades[cascade].to_cell + hddagi.cascades[cascade].position;

	ivec3 hit_cell;
	ivec3 hit_face;
	int hit_cascade;
	bool exclude_receiver_cell = params.normal_bias >= 0.0 && dot(ray_direction_view, origin_normal) > 1e-4;
#ifdef MODE_SPECULAR_TRACE
	exclude_receiver_cell = exclude_receiver_cell && (params.flags & SCREEN_PROBE_FLAG_DETAIL_TRACE) != 0u;
#endif
	float hit_time;
	if (!trace_ray_hdda(ray_position, ray_direction, cascade, exclude_receiver_cell, hit_cell, hit_face, hit_cascade, hit_time)) {
		return false;
	}

	bool reconnectable_endpoint = !(hit_cascade == cascade && all(equal(ivec3(start_cell), hit_cell)));

	bool disoccluded = false;
	if (!reconnectable_endpoint) {
		ivec3 read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
		uint disocclusion = load_voxel_disocclusion(read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0));
		if (disocclusion == 0u) {
			vec3 local_position = fract(start_cell) - 0.5;
			vec3 absolute_origin_normal = abs(hddagi_normal);
			int closest_axis = absolute_origin_normal.y > absolute_origin_normal.x ? 1 : 0;
			if (absolute_origin_normal.z > absolute_origin_normal[closest_axis]) {
				closest_axis = 2;
			}
			vec3 axis_a = vec3(closest_axis == 2, closest_axis == 0, closest_axis == 1);
			vec3 axis_b = vec3(closest_axis == 1, closest_axis == 2, closest_axis == 0);
			vec3 advance = abs(dot(axis_a, local_position)) > abs(dot(axis_b, local_position)) ? axis_a * sign(local_position) : axis_b * sign(local_position);
			start_cell += advance;
			hit_cell += ivec3(advance);
			read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
			disocclusion = load_voxel_disocclusion(read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0));
		}
		if (disocclusion != 0u) {
			vec3 local_position = fract(start_cell) - 0.5;
			int best_axis = 0;
			float best_distance = -20.0;
			for (int i = 0; i < 6; i++) {
				if (bool(disocclusion & (1u << uint(i)))) {
					float distance = dot(local_position, vec3(hddagi_axis_direction(uint(i))));
					if (distance > best_distance) {
						best_axis = i;
						best_distance = distance;
					}
				}
			}
			hit_face = hddagi_axis_direction(uint(best_axis));
			disoccluded = true;
		}
	}

#ifdef MODE_SPECULAR_TRACE
	if (reconnectable_endpoint && hit_time > 0.0) {
		vec3 endpoint = ray_position + ray_direction * hit_time;
		endpoint.y /= hddagi.y_mult;
		r_hit_distance = min(length(endpoint - camera_basis * origin_view), 65504.0);
		r_hit_cell_size = 1.0 / hddagi.cascades[hit_cascade].to_cell;
	}
#endif

	hit_cell += hit_face;
	ivec3 read_cell = (hit_cell + hddagi.cascades[hit_cascade].region_world_offset * HDDAGI_REGION_SIZE) & (hddagi.grid_size - 1);
	ivec3 texture_position = read_cell + ivec3(0, hddagi.grid_size.y * hit_cascade, 0);
	vec3 light = hddagi_read_light(texture_position);
	uint neighbour_bits = disoccluded ? 0u : load_voxel_neighbours(texture_position) & HDDAGI_LIGHT_CELL_NEIGHBOUR_MASK;
	if (neighbour_bits != 0u) {
		vec3 cell_delta = vec3(hit_cell) + 0.5 - (ray_position - hddagi.cascades[hit_cascade].position) * hddagi.cascades[hit_cascade].to_cell;
		vec4 light_accumulator = vec4(light, 1.0) * hddagi_light_ray_weight(cell_delta, ray_direction);
		while (neighbour_bits != 0u) {
			ivec3 offset = hddagi_light_neighbour_direction(findLSB(neighbour_bits));
			float weight = hddagi_light_ray_weight(cell_delta + vec3(offset), ray_direction);
			if (weight > 0.0) {
				ivec3 coord = (read_cell + offset) & (hddagi.grid_size - 1);
				coord.y += hddagi.grid_size.y * hit_cascade;
				vec3 neighbour_light = hddagi_read_light(coord);
				light_accumulator += vec4(neighbour_light, 1.0) * weight;
			}
			neighbour_bits &= neighbour_bits - 1u;
		}
		if (light_accumulator.a > 0.0) {
			light = light_accumulator.rgb / light_accumulator.a;
		}
	}
	r_radiance = light * hddagi.cascades[hit_cascade].exposure_normalization;
	bool radiance_valid = !any(isnan(r_radiance)) && !any(isinf(r_radiance)) && !any(lessThan(r_radiance, vec3(0.0)));
	return radiance_valid;
}

#endif
