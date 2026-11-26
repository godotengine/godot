float short_range_depth(ivec2 pixel) {
	if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, params.screen_size))) {
		return 0.0;
	}
	return texelFetch(sampler2D(ao_depth, nearest_sampler), pixel, 0).r;
}

float screen_probe_short_range_visibility(ivec2 pixel) {
	ivec2 corner = min(pixel, max(params.screen_size - 2, ivec2(0)));
	// Gather in raster order to estimate the receiver's device-depth plane.
	vec4 quad = textureGather(sampler2D(ao_depth, nearest_sampler), (vec2(corner) + 1.0) / vec2(params.screen_size)).wzxy;
	ivec2 lane = pixel - corner;
	int index = lane.x + 2 * lane.y;
	float depth = quad[index];
	vec2 slope = (vec2(quad[index ^ 1], quad[index ^ 2]) - depth) * vec2(1 - lane * 2);
	float depth_tolerance = max(2e-7, depth * 2e-6);
	bool planar = all(greaterThan(quad, vec4(0.0))) && abs((quad.w - quad.z) - (quad.y - quad.x)) < depth_tolerance;
	vec3 normal;
	if (depth == 0.0 || !decode_normal(texelFetch(sampler2D(ao_normal_roughness, nearest_sampler), pixel, 0).xyz, normal)) {
		return 1.0;
	}

	mat4 inv_projection = scene_data.inv_projection[params.view_index];
	mat4 projection = scene_data.projection[params.view_index];
	vec2 pixel_size = 2.0 / vec2(params.screen_size);
	vec4 center = inv_projection * vec4((vec2(pixel) + 0.5) * pixel_size - 1.0, depth, 1.0);
	vec3 position = center.xyz / center.w;
	vec3 view = projection[3][3] == 1.0 ? vec3(0.0, 0.0, 1.0) : normalize(-position);
	float normal_view = dot(normal, view);
	if (normal_view <= 0.0001) {
		return 1.0;
	}

	bool temporal = scene_data.short_range_ao.z > 0.0;
	uint frame = temporal ? params.frame_index : 0u;
	uvec2 sample_pixel = uvec2(pixel);
	vec2 jitter = sample_temporally_permuted_r2_sequence(temporal ? sample_pixel >> 3u : sample_pixel, frame, 0x414f6a09u);
	if (temporal) {
		// Share the temporal phase within each workgroup and stratify each 2x2 quad.
		uint stratum = (sample_pixel.x & 1u) + 2u * (sample_pixel.y & 1u);
		jitter = fract(jitter + vec2(stratum, (stratum * 3u) & 3u) * 0.25);
	}
	float angle = jitter.x * (TAU * 0.5);
	vec2 direction = vec2(cos(angle), sin(angle));
	float radius = scene_data.short_range_ao.x;
	vec2 screen_offset = (projection[0].xy * direction.x + projection[1].xy * direction.y) * (0.5 * radius * center.w) * vec2(params.screen_size);
	float projected_radius = length(screen_offset);
	if (projected_radius <= 1.0) {
		return 1.0;
	}
	vec2 screen_direction = screen_offset / projected_radius;
	float screen_radius = min(projected_radius, length(vec2(params.screen_size)));

	vec3 tangent = normalize(vec3(direction, 0.0) - view * dot(view.xy, direction));
	vec2 slice_normal = normalize(vec2(normal_view, dot(normal, tangent)));
	vec2 open_horizons = vec2(-slice_normal.y, slice_normal.y);
	vec2 horizons = open_horizons;
	float radius_squared = radius * radius;
	for (int step = 0; step < 3; step++) {
		float fraction = (float(step) + jitter.y) / 3.0;
		ivec2 offset = ivec2(round(screen_direction * min(1.0 + fraction * fraction * screen_radius, screen_radius)));
		vec2 depths = vec2(short_range_depth(pixel + offset), short_range_depth(pixel - offset));
		vec2 dz = depths - depth;
		// Device depth is affine on a triangle, even when its shading normal is perturbed.
		if (all(greaterThan(depths, vec2(0.0))) && abs(dz.x + dz.y) <= max(2e-7, depth * 1e-5)) {
			continue;
		}
		vec2 ndc_offset = vec2(offset) * pixel_size;
		vec4 xy_offset = inv_projection[0] * ndc_offset.x + inv_projection[1] * ndc_offset.y;
		float expected_depth_delta = dot(slope, vec2(offset));
		float sample_tolerance = depth_tolerance * (1.0 + float(abs(offset.x) + abs(offset.y)));
		for (int side = 0; side < 2; side++) {
			if (depths[side] == 0.0) {
				continue;
			}
			float side_sign = side == 0 ? 1.0 : -1.0;
			if (planar && abs(dz[side] - side_sign * expected_depth_delta) <= sample_tolerance) {
				continue;
			}
			vec4 homogeneous_delta = side_sign * xy_offset + inv_projection[2] * dz[side];
			vec3 delta = (homogeneous_delta.xyz - position * homogeneous_delta.w) / (center.w + homogeneous_delta.w);
			float distance_squared = dot(delta, delta);
			float height = dot(delta, normal);
			if (distance_squared >= radius_squared || height <= 0.0001 || height * height <= distance_squared * 0.0004) {
				continue;
			}
			float falloff = 1.0 - distance_squared / radius_squared;
			float horizon = dot(delta, view) * inversesqrt(distance_squared) - 0.02;
			horizons[side] = max(horizons[side], mix(open_horizons[side], horizon, falloff * falloff));
		}
	}
	if (horizons == open_horizons) {
		return 1.0;
	}

	float normal_angle = atan(slice_normal.y, slice_normal.x);
	vec2 angles = acos(clamp(horizons, vec2(-1.0), vec2(1.0))) * vec2(1.0, -1.0);
	vec2 arcs = vec2(slice_normal.x) + 2.0 * angles * slice_normal.y - cos(2.0 * angles - normal_angle);
	float open_arc = normal_angle * slice_normal.y + slice_normal.x;
	float visibility = clamp(0.25 * (arcs.x + arcs.y) / open_arc, 0.35, 1.0);
	return mix(1.0, visibility, scene_data.short_range_ao.y);
}
