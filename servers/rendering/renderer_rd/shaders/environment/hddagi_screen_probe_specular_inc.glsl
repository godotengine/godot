#ifdef MODE_SPECULAR_TRACE
#include "hddagi_specular_surface_inc.glsl"

vec2 sample_spatially_stratified_r2_sequence(uvec2 position, uint frame_index, uint stream) {
	uvec2 cell = position >> 1u;
	uint lane = (position.x & 1u) | ((position.y & 1u) << 1u);
	lane ^= hash_uvec3(uvec3(cell, frame_index ^ stream)) & 3u;
	vec2 stratum = vec2(lane & 1u, lane >> 1u);
	vec2 in_stratum = sample_temporally_permuted_r2_sequence(cell, frame_index, stream);
	return (stratum + in_stratum) * 0.5;
}

mat3 specular_tangent_frame(vec3 normal) {
	vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(up, normal));
	return mat3(tangent, cross(normal, tangent), normal);
}

vec3 sample_bounded_ggx_vndf_reflection(vec3 view_local, float alpha, vec2 random_sample) {
	alpha = clamp(alpha, 1e-6, 1.0);
	vec3 stretched_view = normalize(vec3(alpha * view_local.xy, max(view_local.z, 1e-6)));

	float phi = TAU * random_sample.x;
	float scale = 1.0 + length(view_local.xy);
	float alpha_squared = alpha * alpha;
	float scale_squared = scale * scale;
	float bound = (1.0 - alpha_squared) * scale_squared /
			max(scale_squared + alpha_squared * view_local.z * view_local.z, 1e-8);
	float cap = bound * stretched_view.z;
	float reflected_z_stretched = (1.0 - random_sample.y) * (1.0 + cap) - cap;
	float reflected_xy_length = sqrt(clamp(1.0 - reflected_z_stretched * reflected_z_stretched, 0.0, 1.0));
	vec3 reflected_stretched = vec3(
			reflected_xy_length * cos(phi),
			reflected_xy_length * sin(phi),
			reflected_z_stretched);

	vec3 microfacet_stretched = stretched_view + reflected_stretched;
	vec3 microfacet_normal = normalize(vec3(alpha * microfacet_stretched.xy, microfacet_stretched.z));
	return reflect(-view_local, microfacet_normal);
}

const float SPECULAR_INVALID_VIEW_Z = 65505.0;

bool specular_load_full_resolution_surface(ivec2 screen_position, bool reflective_only, out float r_depth, out vec3 r_normal, out float r_roughness, out bool r_dynamic) {
	vec4 normal_roughness = texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), screen_position, 0);
	r_dynamic = normal_roughness.w > 0.5;
	float encoded_roughness = r_dynamic ? 1.0 - normal_roughness.w : normal_roughness.w;
	r_roughness = encoded_roughness * (255.0 / 127.0);
	if (reflective_only && r_roughness >= params.specular_tuning.y) {
		return false;
	}
	r_depth = texelFetch(sampler2D(specular_depth_buffer, linear_sampler), screen_position, 0).r;
	if (r_depth <= 0.0) {
		return false;
	}
	return decode_normal(normal_roughness.xyz, r_normal);
}

bool specular_load_compatible_roughness(ivec2 screen_position, vec3 selected_view, vec3 selected_normal, bool selected_dynamic, out float r_roughness) {
	float tap_depth;
	vec3 tap_normal;
	bool tap_dynamic;
	if (!specular_load_full_resolution_surface(screen_position, false, tap_depth, tap_normal, r_roughness, tap_dynamic) || tap_dynamic != selected_dynamic ||
			r_roughness < 0.001 || r_roughness >= params.specular_tuning.y || dot(tap_normal, selected_normal) <= 0.95) {
		return false;
	}
	vec2 tap_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec3 tap_view = compute_view_position(vec3(tap_uv, tap_depth));
	if (any(isnan(tap_view)) || any(isinf(tap_view))) {
		return false;
	}
	float plane_distance = abs(dot(tap_view - selected_view, selected_normal));
	float plane_tolerance = 0.01 + 0.0025 * abs(selected_view.z);
	return !isnan(plane_distance) && !isinf(plane_distance) && plane_distance <= plane_tolerance;
}

bool specular_select_full_resolution_surface(ivec2 output_position, ivec2 output_size, out ivec2 r_screen_position, out float r_depth, out vec3 r_normal, out float r_roughness, out bool r_dynamic) {
	if (all(equal(output_size, params.screen_size))) {
		r_screen_position = output_position;
		return specular_load_full_resolution_surface(output_position, true, r_depth, r_normal, r_roughness, r_dynamic);
	}
	ivec2 footprint_begin = specular_footprint_begin(output_position, output_size, params.screen_size);
	ivec2 footprint_end = ((output_position + ivec2(1)) * params.screen_size + output_size - ivec2(1)) / output_size;
	footprint_begin = clamp(footprint_begin, ivec2(0), params.screen_size - ivec2(1));
	footprint_end = clamp(footprint_end, footprint_begin + ivec2(1), params.screen_size);
	r_screen_position = (footprint_begin + footprint_end - ivec2(1)) / 2;
	r_depth = 0.0;
	r_normal = vec3(0.0, 0.0, 1.0);
	r_roughness = 1.0;
	r_dynamic = false;
	bool found_surface = false;
	float best_stable_distance_squared = 1e30;
	vec2 stable_resolve_uv = (vec2(output_position) + 0.5) / vec2(output_size);
	for (int y = footprint_begin.y; y < footprint_end.y; y++) {
		for (int x = footprint_begin.x; x < footprint_end.x; x++) {
			ivec2 candidate_position = ivec2(x, y);
			float candidate_depth;
			vec3 candidate_normal;
			float candidate_roughness;
			bool candidate_dynamic;
			if (!specular_load_full_resolution_surface(candidate_position, false, candidate_depth, candidate_normal, candidate_roughness, candidate_dynamic)) {
				continue;
			}
			vec2 candidate_uv = (vec2(candidate_position) + 0.5) / vec2(params.screen_size);
			vec2 stable_uv = candidate_uv - scene_data.taa_jitter_uv.xy;
			vec2 stable_delta = (stable_uv - stable_resolve_uv) * vec2(output_size);
			float stable_distance_squared = dot(stable_delta, stable_delta);
			bool closer_to_stable_center = stable_distance_squared < best_stable_distance_squared - 1e-6;
			bool stable_distance_tie = abs(stable_distance_squared - best_stable_distance_squared) <= 1e-6;
			if (!found_surface || closer_to_stable_center || (stable_distance_tie && candidate_depth > r_depth)) {
				found_surface = true;
				best_stable_distance_squared = stable_distance_squared;
				r_screen_position = candidate_position;
				r_depth = candidate_depth;
				r_normal = candidate_normal;
				r_roughness = candidate_roughness;
				r_dynamic = candidate_dynamic;
			}
		}
	}
	if (found_surface && !r_dynamic && r_roughness >= 0.001 && r_roughness < params.specular_tuning.y) {
		vec2 stable_raster_uv = stable_resolve_uv + scene_data.taa_jitter_uv.xy;
		vec2 selected_uv = (vec2(r_screen_position) + 0.5) / vec2(params.screen_size);
		vec3 selected_view = compute_view_position(vec3(selected_uv, r_depth));
		if (!any(isnan(selected_view)) && !any(isinf(selected_view)) && !any(isnan(stable_raster_uv)) && !any(isinf(stable_raster_uv))) {
			vec2 stable_texel = stable_raster_uv * vec2(params.screen_size) - 0.5;
			ivec2 tap_base = ivec2(floor(stable_texel));
			vec2 tap_fraction = fract(stable_texel);
			float roughness_sum = 0.0;
			float weight_sum = 0.0;
			for (int y = 0; y < 2; y++) {
				for (int x = 0; x < 2; x++) {
					ivec2 tap_position = tap_base + ivec2(x, y);
					vec2 axis_weight = mix(vec2(1.0) - tap_fraction, tap_fraction, bvec2(x != 0, y != 0));
					float tap_weight = axis_weight.x * axis_weight.y;
					float tap_roughness = r_roughness;
					if (!(tap_weight > 0.0) || any(lessThan(tap_position, ivec2(0))) || any(greaterThanEqual(tap_position, params.screen_size))) {
						continue;
					}
					if (any(notEqual(tap_position, r_screen_position)) && !specular_load_compatible_roughness(tap_position, selected_view, r_normal, r_dynamic, tap_roughness)) {
						continue;
					}
					roughness_sum += tap_roughness * tap_weight;
					weight_sum += tap_weight;
				}
			}
			if (weight_sum > 1e-5 && !isnan(roughness_sum) && !isinf(roughness_sum)) {
				r_roughness = clamp(roughness_sum / weight_sum, 0.0, 1.0);
			}
		}
	}
	return found_surface;
}

bool specular_clip_to_uv(vec4 clip_position, out vec2 r_uv) {
	if (!(clip_position.w > 1e-6) || any(isnan(clip_position)) || any(isinf(clip_position))) {
		r_uv = vec2(0.0);
		return false;
	}
	r_uv = clip_position.xy / clip_position.w * 0.5 + 0.5;
	return !any(isnan(r_uv)) && !any(isinf(r_uv));
}

float specular_reconnect_world_hit(vec3 origin, vec3 direction, float hit_distance, float cell_size, out vec3 r_sample_position, out float r_distance, out float r_border_distance) {
	r_border_distance = 0.0;
	mat4 projection = scene_data.projection[params.view_index];
	vec3 estimate = origin + direction * hit_distance;
	vec2 uv;
	if (hit_distance <= 0.0 || !specular_clip_to_uv(projection * vec4(estimate, 1.0), uv)) {
		return 0.0;
	}
	vec2 size = vec2(params.screen_size);
	vec2 margin = min(2.0 / size, vec2(0.5));
	vec2 anchor_uv = clamp(uv, margin, 1.0 - margin);
	float border_width = 0.08 * min(size.x, size.y);
	if (length((uv - anchor_uv) * size) >= border_width) {
		return 0.0;
	}
	ivec2 pixel = ivec2(anchor_uv * size);
	float depth;
	vec3 normal;
	float roughness;
	bool dynamic;
	if (!specular_load_full_resolution_surface(pixel, false, depth, normal, roughness, dynamic)) {
		return 0.0;
	}
	vec3 surface = compute_view_position(vec3((vec2(pixel) + 0.5) / size, depth));
	float facing = -dot(normal, direction);
	float plane_offset = dot(estimate - surface, normal);
	if (facing <= 0.05 || abs(plane_offset) > 1.5 * cell_size) {
		return 0.0;
	}
	r_distance = hit_distance + plane_offset / facing;
	if (r_distance <= 0.0 || abs(r_distance - hit_distance) > 3.0 * cell_size) {
		return 0.0;
	}
	vec3 endpoint = origin + direction * r_distance;
	if (!specular_clip_to_uv(projection * vec4(endpoint, 1.0), uv)) {
		return 0.0;
	}
	anchor_uv = clamp(uv, margin, 1.0 - margin);
	pixel = ivec2(anchor_uv * size);
	depth = texelFetch(sampler2D(specular_depth_buffer, linear_sampler), pixel, 0).r;
	if (depth <= 0.0) {
		return 0.0;
	}
	vec3 visible = compute_view_position(vec3((vec2(pixel) + 0.5) / size, depth));
	if (all(equal(anchor_uv, uv))) {
		r_sample_position = endpoint;
		return float(abs(visible.z - endpoint.z) <= clamp(abs(endpoint.z) * 0.004, 0.04, 0.25));
	}
	if (dynamic || roughness < 0.6 || abs(dot(visible - endpoint, normal)) > 0.15 * cell_size) {
		return 0.0;
	}

	r_sample_position = visible;
	r_border_distance = length((anchor_uv - uv) * size);
	float offset = max(r_border_distance / border_width, length(visible - endpoint) / (4.0 * cell_size));
	return 1.0 - smoothstep(0.0, 1.0, offset);
}

float specular_sample_screen_radiance(vec3 current_view, bool reconnected, float border_distance, out vec3 r_radiance) {
	r_radiance = vec3(0.0);
	vec3 endpoint_world = (scene_data.cam_transform * vec4(current_view, 1.0)).xyz;
	vec3 previous_view = (scene_data.previous_cam_inv_transform * vec4(endpoint_world, 1.0)).xyz;
	mat4 previous_projection = scene_data.previous_projection[params.view_index];
	vec4 previous_clip = previous_projection * vec4(previous_view, 1.0);
	vec2 current_raster_uv;
	vec2 previous_raster_uv;
	if (!specular_clip_to_uv(scene_data.projection[params.view_index] * vec4(current_view, 1.0), current_raster_uv) ||
			!specular_clip_to_uv(previous_clip, previous_raster_uv)) {
		return 0.0;
	}
	float expected_previous_depth = previous_clip.z / previous_clip.w;
	if (isnan(expected_previous_depth) || isinf(expected_previous_depth) || expected_previous_depth < 0.0 || expected_previous_depth > 1.0) {
		return 0.0;
	}

	ivec2 history_size = params.screen_size;

	ivec2 endpoint_pixel = clamp(ivec2(floor(current_raster_uv * vec2(params.screen_size))), ivec2(0), params.screen_size - ivec2(1));
	float endpoint_roughness_dynamic = texelFetch(sampler2D(detail_normal_roughness_buffer, linear_sampler), endpoint_pixel, 0).a;
	bool dynamic_endpoint = endpoint_roughness_dynamic > 0.5;
	vec2 history_uv = previous_raster_uv;
	float object_motion_pixels = 0.0;
	if ((params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID) != 0u) {
		vec2 endpoint_velocity = texelFetch(sampler2D(specular_velocity_buffer, linear_sampler), endpoint_pixel, 0).xy;
		if (!any(isnan(endpoint_velocity)) && !any(isinf(endpoint_velocity))) {
			history_uv = current_raster_uv + endpoint_velocity + (scene_data.taa_jitter_uv.zw - scene_data.taa_jitter_uv.xy);
			object_motion_pixels = length((history_uv - previous_raster_uv) * vec2(history_size));
		}
	} else if (dynamic_endpoint) {
		return 0.0;
	}

	vec2 half_texel = 0.5 / vec2(history_size);
	if (any(lessThan(history_uv, half_texel)) || any(greaterThan(history_uv, vec2(1.0) - half_texel))) {
		return 0.0;
	}
	float motion_confidence = mix(1.0, 0.2, smoothstep(1.0, 24.0, object_motion_pixels));
	if (dynamic_endpoint) {
		motion_confidence *= 0.9;
	}
	if (!(motion_confidence >= 0.75)) {
		return 0.0;
	}
	ivec2 previous_depth_position = clamp(ivec2(floor(history_uv * vec2(history_size))), ivec2(0), history_size - ivec2(1));
	float actual_previous_depth = texelFetch(sampler2D(specular_previous_depth_buffer, linear_sampler), previous_depth_position, 0).r;
	// Convert the view-space history tolerance to device depth for this projection.
	float depth_per_meter = abs(previous_projection[2][2] - expected_previous_depth * previous_projection[2][3]) / previous_clip.w;
	float depth_tolerance = max(2e-7, depth_per_meter * max(0.04, abs(previous_view.z) * 0.01));
	if (actual_previous_depth <= 0.0 || !(abs(actual_previous_depth - expected_previous_depth) < depth_tolerance)) {
		return 0.0;
	}

	vec3 screen_radiance = textureLod(sampler2D(specular_screen_radiance_buffer, linear_sampler), history_uv, 0.0).rgb;
	if (any(isnan(screen_radiance)) || any(isinf(screen_radiance)) || any(lessThan(screen_radiance, vec3(0.0)))) {
		return 0.0;
	}
	float exposure_scale = clamp(params.specular_eye_offset_exposure.w, 1.0 / 16.0, 16.0);
	r_radiance = screen_radiance * exposure_scale;
	vec2 edge_texels = min(min(history_uv, 1.0 - history_uv), min(current_raster_uv, 1.0 - current_raster_uv)) * vec2(history_size);
	float edge_weight = smoothstep(0.5, max(2.0, 0.025 * float(min(history_size.x, history_size.y))), min(edge_texels.x, edge_texels.y));
	if (!reconnected || (edge_weight == 1.0 && border_distance == 0.0)) {
		return edge_weight;
	}
	if (dynamic_endpoint || endpoint_roughness_dynamic < 0.6 * (127.0 / 255.0)) {
		return border_distance > 0.0 ? 0.0 : edge_weight;
	}
	// Extend low-frequency lighting across the edge only while the visible patch agrees.
	vec2 span = sign(0.5 - history_uv) * clamp(border_distance, 8.0, 24.0) / vec2(history_size);
	vec3 x_sample = textureLod(sampler2D(specular_screen_radiance_buffer, linear_sampler), history_uv + vec2(span.x, 0.0), 0.0).rgb;
	vec3 y_sample = textureLod(sampler2D(specular_screen_radiance_buffer, linear_sampler), history_uv + vec2(0.0, span.y), 0.0).rgb;
	vec3 diagonal_sample = textureLod(sampler2D(specular_screen_radiance_buffer, linear_sampler), history_uv + span * vec2(0.5, 1.5), 0.0).rgb;
	vec3 lo = min(min(screen_radiance, x_sample), min(y_sample, diagonal_sample));
	vec3 hi = max(max(screen_radiance, x_sample), max(y_sample, diagonal_sample));
	vec3 contrast = (hi - lo) / max(hi, vec3(1e-5));
	float agreement = 1.0 - smoothstep(0.05, 0.2, max(contrast.x, max(contrast.y, contrast.z)));
	return border_distance > 0.0 ? agreement : mix(edge_weight, 1.0, agreement);
}

vec4 specular_pack_normal_roughness(vec3 world_normal, float roughness, bool screen_radiance_used) {
	float maximum_component = max(abs(world_normal.x), max(abs(world_normal.y), abs(world_normal.z)));
	world_normal /= max(maximum_component, 1e-6);
	float roughness_bin = floor(clamp(roughness, 0.0, 1.0) * 127.0 + 0.5);
	float packed_roughness_source = (roughness_bin + (screen_radiance_used ? 128.0 : 0.0)) / 255.0;
	return vec4(world_normal * 0.5 + 0.5, packed_roughness_source);
}

void specular_clear_surface(ivec2 pixel) {
	imageStore(raw_radiance_output, pixel, vec4(0.0));
	imageStore(specular_normal_roughness_output, pixel, vec4(0.0, 0.0, 0.0, 127.0 / 255.0));
	imageStore(specular_view_z_output, pixel, vec4(SPECULAR_INVALID_VIEW_Z));
}

void specular_store_guides(ivec2 output_position, ivec2 screen_position, bool dynamic_surface, vec3 view_position, vec3 view_normal, float roughness, bool screen_radiance_used) {
	vec3 world_normal = mat3(scene_data.cam_transform) * view_normal;
	bool dynamic = dynamic_surface && (params.flags & SCREEN_PROBE_FLAG_DIRECTIONAL_MOTION_VALID) != 0u;
	float view_z = specular_pack_view_z(abs(view_position.z), dynamic, output_position, imageSize(specular_view_z_output), screen_position, params.screen_size);
	imageStore(specular_normal_roughness_output, output_position, specular_pack_normal_roughness(world_normal, roughness, screen_radiance_used));
	imageStore(specular_view_z_output, output_position, vec4(view_z));
}

void specular_trace_main() {
	ivec2 output_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 output_size = imageSize(raw_radiance_output);
	if (any(greaterThanEqual(output_position, output_size))) {
		return;
	}

	ivec2 screen_position = ivec2(0);
	float origin_depth;
	vec3 origin_normal;
	float roughness;
	bool dynamic_surface;
	if (!specular_select_full_resolution_surface(output_position, output_size, screen_position, origin_depth, origin_normal, roughness, dynamic_surface) || roughness >= params.specular_tuning.y) {
		specular_clear_surface(output_position);
		return;
	}

	vec2 origin_uv = (vec2(screen_position) + 0.5) / vec2(params.screen_size);
	vec3 view_position = compute_view_position(vec3(origin_uv, origin_depth));
	vec3 eye_to_surface = view_position - params.specular_eye_offset_exposure.xyz;
	float eye_distance_squared = dot(eye_to_surface, eye_to_surface);
	vec3 view_direction = scene_data.projection[params.view_index][3][3] != 1.0 && eye_distance_squared > 1e-10 ? -eye_to_surface * inversesqrt(eye_distance_squared) : vec3(0.0, 0.0, 1.0);
	bool valid_orientation = !any(isnan(view_direction)) && !any(isinf(view_direction)) && dot(origin_normal, view_direction) > 1e-5;
	if (!valid_orientation) {
		specular_clear_surface(output_position);
		return;
	}

	vec3 ray_direction = reflect(-view_direction, origin_normal);
	if (roughness >= 0.001) {
		mat3 tangent_frame = specular_tangent_frame(origin_normal);
		vec3 view_local = transpose(tangent_frame) * view_direction;
		view_local.z = max(view_local.z, 1e-5);
		view_local = normalize(view_local);
		uint specular_stream = 0x53504543u ^ (params.view_index * 0x9e3779b9u);
		bool spatial_stratification = all(equal(output_size, params.screen_size));
		vec3 ray_local = vec3(0.0);
		bool valid_ray = false;
		float alpha = max(roughness * roughness, params.specular_tuning.w);
		for (uint attempt = 0u; attempt < 8u; attempt++) {
			uint attempt_stream = specular_stream ^ (attempt * 0x85ebca6bu);
			vec2 random_sample = spatial_stratification ? sample_spatially_stratified_r2_sequence(uvec2(output_position), params.frame_index, attempt_stream) : sample_temporally_permuted_r2_sequence(uvec2(output_position), params.frame_index, attempt_stream);
			random_sample.y *= 0.9;
			ray_local = sample_bounded_ggx_vndf_reflection(view_local, alpha, random_sample);
			valid_ray = ray_local.z > 1e-5 && !any(isnan(ray_local)) && !any(isinf(ray_local));
			if (valid_ray) {
				break;
			}
		}
		if (!valid_ray) {
			imageStore(raw_radiance_output, output_position, vec4(0.0));
			specular_store_guides(output_position, screen_position, dynamic_surface, view_position, origin_normal, roughness, false);
			return;
		}
		ray_direction = normalize(tangent_frame * ray_local);
	}

	vec3 screen_radiance = vec3(0.0);
	float screen_weight = 0.0;
	float screen_distance = 0.0;
	bool screen_history_valid = (params.flags & SCREEN_PROBE_FLAG_SPECULAR_SCREEN_RADIANCE_VALID) != 0u;
	if (screen_history_valid) {
		vec3 endpoint;
		vec3 endpoint_normal;
		if (trace_screen_detail(view_position, ray_direction, SCREEN_PROBE_SPECULAR_DETAIL_TRACE_MAX_DISTANCE, endpoint, endpoint_normal)) {
			screen_weight = specular_sample_screen_radiance(endpoint, false, 0.0, screen_radiance);
			if (screen_weight > 0.0) {
				screen_distance = length(endpoint - view_position);
			}
		}
	}

	vec3 traced_radiance = screen_radiance;
	float hit_distance = screen_distance;
	if (screen_weight < 1.0) {
		vec3 world_radiance;
		float world_distance;
		float cell_size;
		if (trace_hddagi_radiance(screen_position, origin_depth, origin_normal, ray_direction, world_radiance, world_distance, cell_size)) {
			vec3 sample_position;
			float refined_distance;
			float border_distance;
			float match = screen_history_valid ? specular_reconnect_world_hit(view_position, ray_direction, world_distance, cell_size, sample_position, refined_distance, border_distance) : 0.0;
			if (match > screen_weight && (screen_distance == 0.0 || abs(refined_distance - screen_distance) <= cell_size)) {
				vec3 recovered_radiance;
				float recovered_weight = match * specular_sample_screen_radiance(sample_position, true, border_distance, recovered_radiance);
				if (recovered_weight > screen_weight) {
					screen_weight = recovered_weight;
					screen_radiance = recovered_radiance;
					screen_distance = refined_distance;
				}
			}
		} else {
			world_radiance = sample_environment(ray_direction);
			world_distance = 65504.0;
		}
		traced_radiance = mix(world_radiance * hddagi.energy, screen_radiance, screen_weight);
		hit_distance = screen_weight >= 0.5 ? screen_distance : world_distance;
	}
	specular_store_guides(output_position, screen_position, dynamic_surface, view_position, origin_normal, roughness, screen_weight >= 0.5);

	vec4 radiance_hit_distance = vec4(traced_radiance, hit_distance);
	if (any(isnan(radiance_hit_distance)) || any(isinf(radiance_hit_distance))) {
		radiance_hit_distance = vec4(0.0);
	} else {
		radiance_hit_distance.rgb = max(radiance_hit_distance.rgb, vec3(0.0));
		float maximum_radiance_component = max(radiance_hit_distance.r, max(radiance_hit_distance.g, radiance_hit_distance.b));
		if (maximum_radiance_component > params.specular_tuning.z) {
			radiance_hit_distance.rgb *= params.specular_tuning.z / maximum_radiance_component;
		}
		radiance_hit_distance.a = clamp(radiance_hit_distance.a, 0.0, 65504.0);
	}
	imageStore(raw_radiance_output, output_position, radiance_hit_distance);
}

#endif

#ifdef MODE_SPECULAR_APPLY

void specular_apply_main() {
	ivec2 output_position = ivec2(gl_GlobalInvocationID.xy);
	ivec2 reflection_size = imageSize(reflection_output);
	if (any(greaterThanEqual(output_position, reflection_size))) {
		return;
	}

	float packed_roughness_source = floor(texelFetch(sampler2D(specular_normal_roughness_input, specular_nearest_sampler), output_position, 0).a * 255.0 + 0.5);
	float roughness = (packed_roughness_source - floor(packed_roughness_source / 128.0) * 128.0) / 127.0;
	float authority = 1.0 - smoothstep(params.specular_tuning.x, params.specular_tuning.y, roughness);
	if (authority <= 0.0) {
		return;
	}

	vec3 traced_radiance = texelFetch(sampler2D(specular_radiance_input, specular_nearest_sampler), output_position, 0).rgb;
	if (any(isnan(traced_radiance)) || any(isinf(traced_radiance))) {
		return;
	}
	vec3 fallback_radiance = rgbe_decode(imageLoad(reflection_output, output_position).r);
	vec4 blend = imageLoad(reflection_blend, output_position);
	float fallback_weight = (1.0 - authority) * blend.g;
	blend.g = fallback_weight + authority;
	vec3 reflected = (fallback_radiance * fallback_weight + max(traced_radiance, vec3(0.0)) * authority) / blend.g;
	imageStore(reflection_output, output_position, uvec4(rgbe_encode(reflected)));
	imageStore(reflection_blend, output_position, blend);
}

#endif
