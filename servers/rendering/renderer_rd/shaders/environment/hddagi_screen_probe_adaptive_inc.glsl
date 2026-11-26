shared uint directional_adaptive_mask;
shared uint directional_adaptive_spawn_mask;
shared uint directional_adaptive_marked_candidates;
shared uint directional_adaptive_allocation_start;
shared uint directional_adaptive_allocation_count;
shared uint directional_adaptive_retained_count;
shared uvec4 directional_adaptive_candidate_surfaces[SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT];
shared uvec4 directional_adaptive_retained_surfaces[SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE];

bool directional_adaptive_load_screen_surface(ivec2 screen_position, out uvec4 r_packed, out vec3 r_view_position, out vec3 r_normal, out bool r_dynamic) {
	float depth;
	if (!load_surface(screen_position, depth, r_normal, r_dynamic)) {
		return false;
	}
	vec2 uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	r_view_position = compute_view_position(vec3(uv, depth));
	if (!directional_finite(r_view_position)) {
		return false;
	}
	r_packed = uvec4(uvec2(screen_position), floatBitsToUint(depth), pack_surface_normal(r_normal, r_dynamic));
	return true;
}

bool directional_adaptive_decode_surface_without_normal(uvec4 packed, out ivec2 r_screen_position, out vec3 r_view_position, out bool r_dynamic) {
	if (all(equal(packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}
	r_screen_position = ivec2(packed.xy);
	float depth = uintBitsToFloat(packed.z);
	r_dynamic = surface_is_dynamic(packed.w);
	if (!(depth > 0.0) || any(lessThan(r_screen_position, ivec2(0))) || any(greaterThanEqual(r_screen_position, params.screen_size))) {
		return false;
	}
	vec2 uv = (vec2(r_screen_position) + 0.5) / vec2(params.screen_size);
	r_view_position = compute_view_position(vec3(uv, depth));
	return directional_finite(r_view_position);
}

ivec2 directional_adaptive_candidate_position(ivec2 tile, uint candidate_index) {
	ivec2 screen_begin;
	ivec2 screen_end;
	screen_probe_tile_bounds(tile, screen_begin, screen_end);
	ivec2 screen_extent = screen_end - screen_begin;

	uint scramble = hash_uvec3(uvec3(uvec2(tile), 0x91e10da5u));
	uint permuted = (candidate_index + (scramble & 7u)) & 7u;
	uvec2 lattice = uvec2(permuted & 3u, permuted >> 2u);
	uint candidate_frame_index = (params.flags & SCREEN_PROBE_FLAG_UNIFORM_PLACEMENT_JITTER) != 0u ? params.frame_index & 7u : params.frame_index;
	vec2 jitter = sample_temporally_permuted_r2_sequence(uvec2(tile) * 8u + lattice, candidate_frame_index, 0x91e10da5u);
	vec2 candidate_uv = (vec2(lattice) + vec2(0.2) + jitter * 0.6) / vec2(4.0, 2.0);
	ivec2 offset = min(ivec2(floor(candidate_uv * vec2(screen_extent))), screen_extent - ivec2(1));
	return screen_begin + offset;
}

float directional_adaptive_receiver_weight_decoded(vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic, vec3 probe_view, bool probe_dynamic, float spatial_weight) {
	if (probe_dynamic != receiver_dynamic) {
		return 0.0;
	}
	vec3 receiver_delta = probe_view - receiver_view;
	float receiver_separation = length(receiver_delta);
	float receiver_plane_distance = abs(dot(receiver_delta, receiver_normal));
	float receiver_tolerance = max(0.02 + 0.02 * receiver_separation, 1e-4);
	float plane_weight = 1.0 - smoothstep(receiver_tolerance, receiver_tolerance * 2.0, receiver_plane_distance);
	return max(spatial_weight, 0.0) * plane_weight;
}

float directional_adaptive_receiver_weight(vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic, uvec4 packed_probe, float spatial_weight) {
	ivec2 probe_screen_position;
	vec3 probe_view;
	bool probe_dynamic;
	if (!directional_adaptive_decode_surface_without_normal(packed_probe, probe_screen_position, probe_view, probe_dynamic)) {
		return 0.0;
	}
	return directional_adaptive_receiver_weight_decoded(receiver_view, receiver_normal, receiver_dynamic, probe_view, probe_dynamic, spatial_weight);
}

float directional_adaptive_uniform_coverage(ivec2 screen_position, vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic) {
	ivec2 base_count = directional_base_probe_count();
	vec2 screen_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(screen_uv);
	ivec2 probe_base = ivec2(floor(probe_grid_position));
	vec2 probe_blend = fract(probe_grid_position);
	float coverage = 0.0;
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 probe_position = probe_base + ivec2(x, y);
			if (any(lessThan(probe_position, ivec2(0))) || any(greaterThanEqual(probe_position, base_count))) {
				continue;
			}
			float bilinear_weight = (x == 0 ? 1.0 - probe_blend.x : probe_blend.x) * (y == 0 ? 1.0 - probe_blend.y : probe_blend.y);
			coverage += directional_adaptive_receiver_weight(receiver_view, receiver_normal, receiver_dynamic, imageLoad(adaptive_probe_surface, probe_position), bilinear_weight);
		}
	}
	return coverage;
}

bool directional_adaptive_candidate_surface(ivec2 tile, uint candidate_index, out uvec4 r_packed, out ivec2 r_screen_position, out vec3 r_view_position, out vec3 r_normal, out bool r_dynamic) {
	r_screen_position = directional_adaptive_candidate_position(tile, candidate_index);
	return directional_adaptive_load_screen_surface(r_screen_position, r_packed, r_view_position, r_normal, r_dynamic);
}

#ifdef MODE_DIRECTIONAL_ADAPTIVE_SPAWN

ivec2 directional_adaptive_owner_tile(ivec2 screen_position) {
	return (screen_position * params.gi_size / params.screen_size) / params.probe_size;
}

ivec2 directional_adaptive_history_tile_offset(int index) {
	if (index == 0) {
		return ivec2(0);
	}
	int cell = index <= 4 ? index - 1 : index;
	return ivec2(cell % 3, cell / 3) - ivec2(1);
}

bool directional_adaptive_previous_owner_from_current_surface(uvec4 current_packed, out ivec2 r_previous_owner) {
	r_previous_owner = ivec2(-1);
	ivec2 current_screen_position;
	vec3 current_view;
	bool current_dynamic;
	if (!directional_adaptive_decode_surface_without_normal(current_packed, current_screen_position, current_view, current_dynamic) || current_dynamic) {
		return false;
	}
	vec3 current_world = (scene_data.cam_transform * vec4(current_view, 1.0)).xyz;
	vec3 previous_view = (scene_data.previous_cam_inv_transform * vec4(current_world, 1.0)).xyz;
	vec4 previous_clip = scene_data.previous_projection[params.view_index] * vec4(previous_view, 1.0);
	if (!(previous_clip.w > 1e-6) || !directional_finite(previous_clip)) {
		return false;
	}
	vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
	float previous_device_depth = previous_clip.z / previous_clip.w;
	if (any(lessThan(previous_uv, vec2(0.0))) || any(greaterThanEqual(previous_uv, vec2(1.0))) ||
			isnan(previous_device_depth) || isinf(previous_device_depth) || previous_device_depth < 0.0 || previous_device_depth > 1.0) {
		return false;
	}
	ivec2 previous_screen_position = clamp(ivec2(floor(previous_uv * vec2(params.screen_size))), ivec2(0), params.screen_size - ivec2(1));
	r_previous_owner = directional_adaptive_owner_tile(previous_screen_position);
	return true;
}

bool directional_adaptive_reproject_previous_surface(uvec4 previous_packed, ivec2 expected_current_owner, mat4 previous_cam_transform, out uvec4 r_current_packed, out ivec2 r_current_screen_position, out vec3 r_current_view_position, out vec3 r_current_normal) {
	r_current_packed = uvec4(0xffffffffu);
	r_current_screen_position = ivec2(-1);
	r_current_view_position = vec3(0.0);
	r_current_normal = vec3(0.0, 0.0, 1.0);
	if (all(equal(previous_packed.xy, uvec2(0xffffffffu)))) {
		return false;
	}

	ivec2 previous_screen_position = ivec2(previous_packed.xy);
	float previous_depth = uintBitsToFloat(previous_packed.z);
	vec3 previous_normal_view = unpack_surface_normal(previous_packed.w);
	bool previous_dynamic = surface_is_dynamic(previous_packed.w);
	if (!(previous_depth > 0.0) || previous_dynamic ||
			any(lessThan(previous_screen_position, ivec2(0))) || any(greaterThanEqual(previous_screen_position, params.screen_size)) ||
			!directional_finite(previous_normal_view)) {
		return false;
	}

	vec2 previous_uv = (vec2(previous_screen_position) + 0.5) / vec2(params.screen_size);
	vec4 previous_view_h = scene_data.previous_inv_projection[params.view_index] * vec4(previous_uv * 2.0 - 1.0, previous_depth, 1.0);
	if (!(abs(previous_view_h.w) > 1e-8) || !directional_finite(previous_view_h)) {
		return false;
	}
	vec3 previous_view = previous_view_h.xyz / previous_view_h.w;
	vec3 previous_world = (previous_cam_transform * vec4(previous_view, 1.0)).xyz;
	vec3 previous_normal_world = normalize(mat3(previous_cam_transform) * previous_normal_view);
	vec3 projected_current_view = transpose(mat3(scene_data.cam_transform)) * (previous_world - scene_data.cam_transform[3].xyz);
	vec4 current_clip = scene_data.projection[params.view_index] * vec4(projected_current_view, 1.0);
	if (!(current_clip.w > 1e-6) || !directional_finite(current_clip) || !directional_finite(previous_world) || !directional_finite(previous_normal_world)) {
		return false;
	}
	vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
	float projected_device_depth = current_clip.z / current_clip.w;
	if (any(lessThan(current_uv, vec2(0.0))) || any(greaterThanEqual(current_uv, vec2(1.0))) ||
			isnan(projected_device_depth) || isinf(projected_device_depth) || projected_device_depth < 0.0 || projected_device_depth > 1.0) {
		return false;
	}

	vec2 projected_pixel = current_uv * vec2(params.screen_size) - 0.5;
	ivec2 projected_base = ivec2(floor(projected_pixel));
	const float relative_plane_tolerance = 0.01823;
	float normal_threshold = max(params.history_normal_threshold, 0.85);
	float best_score = -1.0;
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 candidate_screen_position = projected_base + ivec2(x, y);
			if (any(lessThan(candidate_screen_position, ivec2(0))) || any(greaterThanEqual(candidate_screen_position, params.screen_size)) ||
					any(notEqual(directional_adaptive_owner_tile(candidate_screen_position), expected_current_owner))) {
				continue;
			}
			uvec4 candidate_packed;
			vec3 candidate_view;
			vec3 candidate_normal;
			bool candidate_dynamic;
			if (!directional_adaptive_load_screen_surface(candidate_screen_position, candidate_packed, candidate_view, candidate_normal, candidate_dynamic) || candidate_dynamic) {
				continue;
			}
			vec3 current_world = (scene_data.cam_transform * vec4(candidate_view, 1.0)).xyz;
			vec3 current_normal_world = normalize(mat3(scene_data.cam_transform) * candidate_normal);
			vec3 surface_delta = current_world - previous_world;
			float depth_scale = max(abs(candidate_view.z), 1e-3);
			float current_plane_distance = abs(dot(surface_delta, current_normal_world)) / depth_scale;
			float previous_plane_distance = abs(dot(surface_delta, previous_normal_world)) / depth_scale;
			float normal_similarity = dot(current_normal_world, previous_normal_world);
			if (!directional_finite(current_world) || !directional_finite(current_normal_world) ||
					isnan(current_plane_distance) || isinf(current_plane_distance) || isnan(previous_plane_distance) || isinf(previous_plane_distance) ||
					current_plane_distance > relative_plane_tolerance || previous_plane_distance > relative_plane_tolerance || normal_similarity < normal_threshold) {
				continue;
			}
			float pixel_distance = length(vec2(candidate_screen_position) - projected_pixel);
			float screen_score = 1.0 - clamp(pixel_distance * 0.70710678, 0.0, 1.0);
			float plane_score = 1.0 - clamp(max(current_plane_distance, previous_plane_distance) / relative_plane_tolerance, 0.0, 1.0);
			float normal_score = clamp((normal_similarity - normal_threshold) / max(1.0 - normal_threshold, 1e-4), 0.0, 1.0);
			float score = (0.25 + 0.75 * screen_score) * (0.25 + 0.75 * plane_score) * (0.25 + 0.75 * normal_score);
			if (score > best_score) {
				best_score = score;
				r_current_packed = candidate_packed;
				r_current_screen_position = candidate_screen_position;
				r_current_view_position = candidate_view;
				r_current_normal = candidate_normal;
			}
		}
	}
	return best_score >= 0.0;
}

float directional_adaptive_retained_coverage(ivec2 screen_position, vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic) {
	vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
	float coverage = 0.0;
	for (uint retained_index = 0u; retained_index < directional_adaptive_retained_count; retained_index++) {
		ivec2 retained_screen_position;
		vec3 retained_view;
		bool retained_dynamic;
		if (!directional_adaptive_decode_surface_without_normal(directional_adaptive_retained_surfaces[retained_index], retained_screen_position, retained_view, retained_dynamic)) {
			continue;
		}
		vec2 axis_distance = abs(vec2(retained_screen_position - screen_position)) / probe_screen_extent;
		float spatial_weight = 1.0 - clamp(max(axis_distance.x, axis_distance.y), 0.0, 1.0);
		coverage = max(coverage, directional_adaptive_receiver_weight_decoded(receiver_view, receiver_normal, receiver_dynamic, retained_view, retained_dynamic, spatial_weight));
	}
	return coverage;
}

void directional_adaptive_collect_retained(ivec2 tile) {
	directional_adaptive_retained_count = 0u;
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_HISTORY_VALID) == 0u) {
		return;
	}

	ivec2 base_count = directional_base_probe_count();
	mat4 previous_cam_transform = scene_data.previous_cam_transform;

	ivec2 previous_owners[1 + SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT];
	int previous_owner_count = 0;
	for (int anchor_index = -1; anchor_index < int(SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT); anchor_index++) {
		uvec4 anchor_packed = imageLoad(adaptive_probe_surface, tile);
		if (anchor_index >= 0) {
			ivec2 anchor_screen_position;
			vec3 anchor_view;
			vec3 anchor_normal;
			bool anchor_dynamic;
			if (!directional_adaptive_candidate_surface(tile, uint(anchor_index), anchor_packed, anchor_screen_position, anchor_view, anchor_normal, anchor_dynamic)) {
				continue;
			}
		}
		ivec2 previous_owner;
		if (!directional_adaptive_previous_owner_from_current_surface(anchor_packed, previous_owner)) {
			continue;
		}
		bool duplicate_owner = false;
		for (int previous_owner_index = 0; previous_owner_index < previous_owner_count; previous_owner_index++) {
			duplicate_owner = duplicate_owner || all(equal(previous_owner, previous_owners[previous_owner_index]));
		}
		if (!duplicate_owner) {
			previous_owners[previous_owner_count++] = previous_owner;
		}
	}

	// Compact indices are not persistent identities; reproject the previous surfaces.
	for (int previous_owner_index = 0; previous_owner_index < previous_owner_count; previous_owner_index++) {
		for (int history_tile_index = 0; history_tile_index < 9; history_tile_index++) {
			ivec2 previous_tile = previous_owners[previous_owner_index] + directional_adaptive_history_tile_offset(history_tile_index);
			if (any(lessThan(previous_tile, ivec2(0))) || any(greaterThanEqual(previous_tile, base_count))) {
				continue;
			}
			bool visited = false;
			for (int prior_owner_index = 0; prior_owner_index < previous_owner_index; prior_owner_index++) {
				visited = visited || all(lessThanEqual(abs(previous_tile - previous_owners[prior_owner_index]), ivec2(1)));
			}
			if (visited) {
				continue;
			}
			uvec2 data = imageLoad(adaptive_previous_tile_data_input, previous_tile).rg;
			for (uint i = 0u; i < data.y; i++) {
				ivec2 previous_physical_position = directional_adaptive_probe_position(base_count, data.x + i);
				uvec4 current_packed;
				ivec2 current_screen_position;
				vec3 current_view;
				vec3 current_normal;
				if (!directional_adaptive_reproject_previous_surface(imageLoad(adaptive_previous_probe_surface, previous_physical_position), tile, previous_cam_transform, current_packed, current_screen_position, current_view, current_normal) ||
						directional_adaptive_uniform_coverage(current_screen_position, current_view, current_normal, false) >= SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_RETIRE_COVERAGE ||
						directional_adaptive_retained_coverage(current_screen_position, current_view, current_normal, false) >= SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
					continue;
				}
				directional_adaptive_retained_surfaces[directional_adaptive_retained_count++] = current_packed;
				if (directional_adaptive_retained_count >= SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE) {
					return;
				}
			}
		}
	}
}

float directional_adaptive_prior_coverage(uint candidate_index, ivec2 screen_position, vec3 receiver_view, vec3 receiver_normal, bool receiver_dynamic) {
	ivec2 base_count = directional_base_probe_count();
	vec2 screen_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec2 probe_grid_position = directional_grid_uv_to_probe_texel(screen_uv);
	ivec2 search_base = ivec2(floor(probe_grid_position));
	vec2 probe_screen_extent = max(vec2(params.probe_size) * vec2(params.screen_size) / vec2(params.gi_size), vec2(1.0));
	vec4 corner_coverage = vec4(0.0);
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 prior_tile = search_base + ivec2(x, y);
			if (any(lessThan(prior_tile, ivec2(0))) || any(greaterThanEqual(prior_tile, base_count))) {
				continue;
			}
			uint prior_mask = imageLoad(adaptive_candidate_mark_input, prior_tile).r;
			for (uint prior_candidate = 0u; prior_candidate < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_CANDIDATE_COUNT; prior_candidate++) {
				if (prior_candidate >= candidate_index) {
					break;
				}
				if ((prior_mask & (1u << prior_candidate)) == 0u) {
					continue;
				}
				uvec4 prior_packed;
				ivec2 prior_screen_position;
				vec3 prior_view;
				vec3 prior_normal;
				bool prior_dynamic;
				if (!directional_adaptive_candidate_surface(prior_tile, prior_candidate, prior_packed, prior_screen_position, prior_view, prior_normal, prior_dynamic)) {
					continue;
				}
				vec2 axis_distance = abs(vec2(prior_screen_position - screen_position)) / probe_screen_extent;
				float spatial_weight = 1.0 - clamp(min(axis_distance.x, axis_distance.y), 0.0, 1.0);
				int corner_index = y * 2 + x;
				corner_coverage[corner_index] = max(corner_coverage[corner_index], directional_adaptive_receiver_weight_decoded(receiver_view, receiver_normal, receiver_dynamic, prior_view, prior_dynamic, spatial_weight));
			}
		}
	}
	return dot(corner_coverage, vec4(1.0));
}

#endif

#ifdef MODE_DIRECTIONAL_ADAPTIVE_MARK

void directional_adaptive_mark_main() {
	ivec2 tile = ivec2(gl_WorkGroupID.xy);
	ivec2 base_count = directional_base_probe_count();
	uint lane = gl_LocalInvocationIndex;
	if (lane == 0u) {
		directional_adaptive_mask = 0u;
		ivec2 extra = ivec2(tile.x, base_count.y + tile.y);
		if (extra.y < imageSize(adaptive_probe_surface).y) {
			imageStore(adaptive_probe_surface, extra, uvec4(0xffffffffu));
		}
		if (all(equal(tile, ivec2(0)))) {
			imageStore(adaptive_probe_counter, ivec2(0), uvec4(0u));
		}
	}
	barrier();
	uvec4 packed;
	ivec2 screen_position;
	vec3 view_position;
	vec3 normal;
	bool dynamic_surface;
	if (directional_adaptive_candidate_surface(tile, lane, packed, screen_position, view_position, normal, dynamic_surface) &&
			directional_adaptive_uniform_coverage(screen_position, view_position, normal, dynamic_surface) < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
		atomicOr(directional_adaptive_mask, 1u << lane);
	}
	barrier();
	if (lane == 0u) {
		imageStore(adaptive_candidate_mark_output, tile, uvec4(directional_adaptive_mask, 0u, 0u, 0u));
	}
}

#endif

#ifdef MODE_DIRECTIONAL_ADAPTIVE_SPAWN

void directional_adaptive_spawn_main() {
	ivec2 tile = ivec2(gl_WorkGroupID.xy);
	ivec2 base_count = directional_base_probe_count();
	uint lane = gl_LocalInvocationIndex;
	if (lane == 0u) {
		directional_adaptive_spawn_mask = 0u;
		directional_adaptive_marked_candidates = imageLoad(adaptive_candidate_mark_input, tile).r;
		directional_adaptive_collect_retained(tile);
	}
	barrier();

	uint marked = directional_adaptive_marked_candidates;
	if ((marked & (1u << lane)) != 0u) {
		uvec4 packed;
		ivec2 screen_position;
		vec3 view_position;
		vec3 normal;
		bool dynamic_surface;
		if (directional_adaptive_candidate_surface(tile, lane, packed, screen_position, view_position, normal, dynamic_surface)) {
			directional_adaptive_candidate_surfaces[lane] = packed;
			if (directional_adaptive_retained_coverage(screen_position, view_position, normal, dynamic_surface) < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE &&
					directional_adaptive_prior_coverage(lane, screen_position, view_position, normal, dynamic_surface) < SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_SPAWN_COVERAGE) {
				atomicOr(directional_adaptive_spawn_mask, 1u << lane);
			}
		}
	}
	barrier();

	if (lane == 0u) {
		uint requested = min(directional_adaptive_retained_count + uint(bitCount(directional_adaptive_spawn_mask)), SCREEN_PROBE_DIRECTIONAL_ADAPTIVE_MAX_PER_TILE);
		uint base_probe_total = uint(base_count.x * base_count.y);
		uint adaptive_capacity = base_probe_total / 2u;
		uint allocation_start = imageAtomicAdd(adaptive_probe_counter, ivec2(0), requested);
		uint allocation_count = allocation_start < adaptive_capacity ? min(requested, adaptive_capacity - allocation_start) : 0u;
		directional_adaptive_allocation_start = allocation_start;
		directional_adaptive_allocation_count = allocation_count;
		imageStore(adaptive_tile_data_output, tile, uvec4(allocation_start, allocation_count, 0u, 0u));
	}
	barrier();

	if (lane < min(directional_adaptive_retained_count, directional_adaptive_allocation_count)) {
		ivec2 probe = directional_adaptive_probe_position(base_count, directional_adaptive_allocation_start + lane);
		imageStore(adaptive_probe_surface, probe, directional_adaptive_retained_surfaces[lane]);
	}
	uint rank = directional_adaptive_retained_count + bitCount(directional_adaptive_spawn_mask & ((1u << lane) - 1u));
	if ((directional_adaptive_spawn_mask & (1u << lane)) != 0u && rank < directional_adaptive_allocation_count) {
		ivec2 probe = directional_adaptive_probe_position(base_count, directional_adaptive_allocation_start + rank);
		imageStore(adaptive_probe_surface, probe, directional_adaptive_candidate_surfaces[lane]);
	}
}

#endif
