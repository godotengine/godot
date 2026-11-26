const float SPECULAR_HIT_METADATA_QUANTIZATION = 127.0;
const float SPECULAR_SOURCE_METADATA_QUANTIZATION = 3.0;
const float SPECULAR_HIT_LOG_RANGE = 16.0;
const float SPECULAR_MOMENT_LUMINANCE_SCALE = 4.0;

float svgf_hit_log(float hit_distance) {
	return log2(1.0 + clamp(hit_distance, 0.0, FP16_MAX)) / SPECULAR_HIT_LOG_RANGE;
}

vec2 svgf_unpack_specular_roughness_source(float packed_value) {
	float packed = floor(clamp(packed_value, 0.0, 1.0) * 255.0 + 0.5);
	float screen_source = floor(packed / 128.0);
	float roughness_bin = packed - screen_source * 128.0;
	return vec2(roughness_bin / 127.0, screen_source);
}

// The active roughness range fits in 6 bits. Offset the payload past the FP16 subnormal range.
float svgf_pack_specular_metadata(float roughness, float hit_log, float screen_confidence) {
	uint roughness_bin = uint(roughness * 127.0 + 0.5);
	uint hit_bin = uint(hit_log * SPECULAR_HIT_METADATA_QUANTIZATION + 0.5);
	uint source_bin = uint(screen_confidence * SPECULAR_SOURCE_METADATA_QUANTIZATION + 0.5);
	return unpackHalf2x16(0x400u + ((roughness_bin << 9u) | (hit_bin << 2u) | source_bin)).x;
}

vec3 svgf_unpack_specular_metadata(float metadata) {
	uint packed = packHalf2x16(vec2(metadata, 0.0)) - 0x400u;
	return vec3(float(packed >> 9u) / 127.0, float((packed >> 2u) & 127u) / SPECULAR_HIT_METADATA_QUANTIZATION, float(packed & 3u) / SPECULAR_SOURCE_METADATA_QUANTIZATION);
}

float svgf_specular_luminance(vec3 signal) {
	return min(svgf_luminance(signal) * SPECULAR_MOMENT_LUMINANCE_SCALE, MOMENT_LUMINANCE_MAX);
}

float svgf_specular_normal_weight(float normal_cosine, float lobe_width_squared) {
	return lobe_width_squared / (lobe_width_squared + 2.0 * (1.0 - clamp(normal_cosine, 0.0, 1.0)));
}

vec2 svgf_specular_disk(int index) {
	vec2 p = (index & 4) == 0 ? ((index & 2) == 0 ? vec2(0.3500, 0.0000) : vec2(-0.4424, 0.4054)) : ((index & 2) == 0 ? vec2(0.0697, -0.7970) : vec2(0.6088, 0.7934));
	return (index & 1) == 0 ? p : -p;
}

#ifdef MODE_SPECULAR_RECONSTRUCT

layout(set = 0, binding = 0) uniform texture2D reconstruction_signal_input;
layout(set = 0, binding = 1) uniform texture2D reconstruction_normal_roughness_input;
layout(set = 0, binding = 2) uniform texture2D reconstruction_view_z_input;
layout(set = 0, binding = 3) uniform sampler reconstruction_nearest_sampler;
layout(rgba16f, set = 0, binding = 4) uniform restrict writeonly image2D reconstruction_signal_output;

void svgf_specular_reconstruct_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(reconstruction_signal_output);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	vec4 center_encoded_normal = texelFetch(sampler2D(reconstruction_normal_roughness_input, reconstruction_nearest_sampler), pixel, 0);
	vec2 center_specular_guide = svgf_unpack_specular_roughness_source(center_encoded_normal.a);
	float center_roughness = center_specular_guide.x;
	if (center_roughness >= 0.40) {
		return;
	}
	vec4 center_signal = svgf_sanitize_signal(texelFetch(sampler2D(reconstruction_signal_input, reconstruction_nearest_sampler), pixel, 0));
	if (center_roughness < 0.10) {
		imageStore(reconstruction_signal_output, pixel, center_signal);
		return;
	}

	float center_view_z = texelFetch(sampler2D(reconstruction_view_z_input, reconstruction_nearest_sampler), pixel, 0).r;
	bool center_dynamic = center_view_z < 0.0;
	center_view_z = abs(center_view_z);
	vec3 center_normal = svgf_decode_normal(center_encoded_normal);
	if (!svgf_valid_view_z(center_view_z)) {
		imageStore(reconstruction_signal_output, pixel, center_signal);
		return;
	}

	uint reconstruction_quality = min(params.control.z >> 2u, 2u);
	float quality_response = float(reconstruction_quality) * 0.5;
	float signal_resolution_scale = (params.control.x & 4u) != 0u ? 1.0 : 0.5;
	float maximum_radius = mix(4.0, 6.0, quality_response) * signal_resolution_scale;
	float filter_radius = maximum_radius * clamp(center_roughness * 8.0, 0.0, 1.0);
	int sample_count = 2 + int(reconstruction_quality) * 2;
	float broad_lobe_response = smoothstep(0.10, 0.30, center_roughness);
	float reconstruction_response = smoothstep(0.10, 0.16, center_roughness) * (1.0 - smoothstep(0.34, 0.40, center_roughness));
	float lobe_width = max(center_roughness * center_roughness, 0.02);
	bool center_far_hit = center_signal.a >= 65000.0;

	float depth_left = abs(texelFetch(sampler2D(reconstruction_view_z_input, reconstruction_nearest_sampler), max(pixel - ivec2(1, 0), ivec2(0)), 0).r);
	float depth_right = abs(texelFetch(sampler2D(reconstruction_view_z_input, reconstruction_nearest_sampler), min(pixel + ivec2(1, 0), size - 1), 0).r);
	float depth_up = abs(texelFetch(sampler2D(reconstruction_view_z_input, reconstruction_nearest_sampler), max(pixel - ivec2(0, 1), ivec2(0)), 0).r);
	float depth_down = abs(texelFetch(sampler2D(reconstruction_view_z_input, reconstruction_nearest_sampler), min(pixel + ivec2(0, 1), size - 1), 0).r);
	float gradient_left = svgf_valid_view_z(depth_left) ? center_view_z - depth_left : 0.0;
	float gradient_right = svgf_valid_view_z(depth_right) ? depth_right - center_view_z : gradient_left;
	float gradient_up = svgf_valid_view_z(depth_up) ? center_view_z - depth_up : 0.0;
	float gradient_down = svgf_valid_view_z(depth_down) ? depth_down - center_view_z : gradient_up;
	vec2 depth_gradient = vec2(
			abs(gradient_left) < abs(gradient_right) ? gradient_left : gradient_right,
			abs(gradient_up) < abs(gradient_down) ? gradient_up : gradient_down);
	float base_plane_threshold = max(0.001, params.atrous.y * (((params.control.w & 1u) != 0u) ? 1.0 : center_view_z));

	vec3 signal_sum = center_signal.rgb * 2.0;
	float weight_sum = 2.0;
	for (int sample_index = 0; sample_index < 8; sample_index++) {
		if (sample_index >= sample_count) {
			break;
		}
		vec2 disk_sample = svgf_specular_disk(sample_index);
		ivec2 offset = ivec2(round(disk_sample * filter_radius));
		if (all(equal(offset, ivec2(0)))) {
			continue;
		}
		ivec2 sample_pos = pixel + offset;
		if (any(lessThan(sample_pos, ivec2(0))) || any(greaterThanEqual(sample_pos, size))) {
			continue;
		}

		float sample_view_z = texelFetch(sampler2D(reconstruction_view_z_input, reconstruction_nearest_sampler), sample_pos, 0).r;
		if ((sample_view_z < 0.0) != center_dynamic) {
			continue;
		}
		sample_view_z = abs(sample_view_z);
		vec4 sample_encoded_normal = texelFetch(sampler2D(reconstruction_normal_roughness_input, reconstruction_nearest_sampler), sample_pos, 0);
		vec3 sample_normal = svgf_decode_normal(sample_encoded_normal);
		if (!svgf_valid_view_z(sample_view_z)) {
			continue;
		}
		float normal_cosine = max(dot(center_normal, sample_normal), 0.0);
		if (!(normal_cosine > 0.6)) {
			continue;
		}
		float expected_depth_delta = dot(depth_gradient, vec2(offset));
		float plane_threshold = base_plane_threshold * (1.0 + 0.35 * sqrt(float(dot(offset, offset))));
		if (abs((sample_view_z - center_view_z) - expected_depth_delta) > plane_threshold) {
			continue;
		}

		vec2 sample_specular_guide = svgf_unpack_specular_roughness_source(sample_encoded_normal.a);
		if (sample_specular_guide.x >= 0.40) {
			continue;
		}
		float roughness_delta = abs(sample_specular_guide.x - center_roughness);
		float lobe_roughness = min(sample_specular_guide.x, center_roughness);
		float narrow_lobe = 1.0 - smoothstep(0.05, 0.20, lobe_roughness);
		float roughness_weight = exp2(-roughness_delta * mix(8.0, 32.0, narrow_lobe));
		float route_mismatch_weight = 0.35 * smoothstep(0.16, 0.38, center_roughness);
		float source_weight = sample_specular_guide.y == center_specular_guide.y ? 1.0 : route_mismatch_weight;
		vec4 sample_signal = svgf_sanitize_signal(texelFetch(sampler2D(reconstruction_signal_input, reconstruction_nearest_sampler), sample_pos, 0));
		bool sample_far_hit = sample_signal.a >= 65000.0;
		float hit_class_weight = sample_far_hit == center_far_hit ? 1.0 : route_mismatch_weight;
		float hit_weight = hit_class_weight * mix(exp2(-abs(svgf_hit_log(sample_signal.a) - svgf_hit_log(center_signal.a)) * 12.0), 1.0, broad_lobe_response * 0.5);
		float radial_weight = max(1.0 - 0.35 * length(disk_sample), 0.5);
		float weight = radial_weight * svgf_specular_normal_weight(normal_cosine, lobe_width * lobe_width) * roughness_weight * source_weight * hit_weight;
		if (!(weight > 1e-5) || !svgf_finite_float(weight)) {
			continue;
		}
		signal_sum += sample_signal.rgb * weight;
		weight_sum += weight;
	}

	vec3 reconstructed_signal = mix(center_signal.rgb, signal_sum / weight_sum, reconstruction_response);
	imageStore(reconstruction_signal_output, pixel, vec4(reconstructed_signal, center_signal.a));
}
#endif // MODE_SPECULAR_RECONSTRUCT

#ifdef SVGF_SPECULAR

vec2 svgf_specular_parallax(vec3 current_position, vec3 previous_position, float hit_distance) {
	vec3 translation = frame_data.current_view_to_previous_view[3].xyz;
	if ((params.control.w & 3u) != 0u || dot(translation, translation) < 1e-10) {
		return vec2(0.0);
	}
	vec3 view_ray = mat3(frame_data.current_view_to_previous_view) * normalize(current_position);
	vec3 virtual_position = previous_position + view_ray * hit_distance;
	if (virtual_position.z >= -1e-5) {
		return vec2(0.0);
	}
	vec2 projection_scale = 0.5 / vec2(frame_data.previous_inv_projection[0][0], frame_data.previous_inv_projection[1][1]);
	return (virtual_position.xy / -virtual_position.z - previous_position.xy / -previous_position.z) * projection_scale;
}

void svgf_specular_temporal_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(current_signal_output);
	bool pixel_in_bounds = all(lessThan(pixel, size));
	ivec2 shared_origin = ivec2(gl_WorkGroupID.xy) * ivec2(gl_WorkGroupSize.xy) - ivec2(TEMPORAL_NEIGHBORHOOD_RADIUS);
	for (uint shared_index = gl_LocalInvocationIndex; shared_index < uint(TEMPORAL_NEIGHBORHOOD_SHARED_COUNT); shared_index += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
		ivec2 shared_position = ivec2(int(shared_index) % TEMPORAL_NEIGHBORHOOD_SHARED_SIZE, int(shared_index) / TEMPORAL_NEIGHBORHOOD_SHARED_SIZE);
		ivec2 sample_position = shared_origin + shared_position;
		vec4 packed_sample = vec4(0.0);
		if (all(greaterThanEqual(sample_position, ivec2(0))) && all(lessThan(sample_position, size))) {
			float sample_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), sample_position, 0).r;
			if (svgf_valid_view_z(abs(sample_view_z))) {
				vec4 sample_signal = texelFetch(sampler2D(current_signal_input, nearest_sampler), sample_position, 0);
				if (sample_signal.a > 0.0) {
					packed_sample = vec4(svgf_rgb_to_ycocg(sample_signal.rgb), 1.0);
				}
			}
		}
		temporal_neighborhood_ycocg[shared_position.y][shared_position.x] = packed_sample;
	}
	memoryBarrierShared();
	barrier();
	if (!pixel_in_bounds) {
		return;
	}

	vec4 current_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
	vec2 current_specular_guide = svgf_unpack_specular_roughness_source(current_encoded_normal.a);
	float current_roughness = current_specular_guide.x;
	if (current_roughness >= 0.40 || (current_roughness == 0.0 && (params.control.x & 8u) != 0u)) {
		imageStore(current_normal_roughness_output, pixel, vec4(0.0));
		return;
	}
	vec4 current_signal = svgf_sanitize_signal(texelFetch(sampler2D(current_signal_input, nearest_sampler), pixel, 0));
	float current_screen_confidence = current_specular_guide.y;
	float current_hit_log = svgf_hit_log(current_signal.a);
	float packed_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r;
	float current_view_z = abs(packed_view_z);
	vec3 current_normal = svgf_decode_normal(current_encoded_normal);
	bool current_surface_valid = svgf_valid_view_z(current_view_z);
	bool current_dynamic = packed_view_z < 0.0;
	ivec2 screen_size = textureSize(sampler2D(motion_input, nearest_sampler), 0);
	ivec2 surface_pixel = specular_surface_pixel(pixel, size, screen_size, packed_view_z);
	vec2 motion = texelFetch(sampler2D(motion_input, nearest_sampler), surface_pixel, 0).xy;
	bool motion_valid = svgf_finite_vec2(motion);

	vec3 accumulated_history = vec3(0.0);
	vec2 accumulated_moments = vec2(0.0);
	float accumulated_history_length = 0.0;
	float accumulated_history_screen_confidence = 0.0;
	float history_weight_sum = 0.0;
	float reflection_motion_pixels = length(motion * vec2(size));
	float maximum_history_tap_weight = 0.0;
	if ((params.control.x & 1u) != 0u && current_surface_valid && motion_valid) {
		vec2 current_uv = (vec2(pixel) + 0.5) / vec2(size);
		vec2 current_surface_uv = (vec2(surface_pixel) + 0.5) / vec2(screen_size);
		bool half_resolution_specular = (params.control.x & 4u) == 0u;
		vec2 previous_uv = half_resolution_specular ? current_uv + motion : current_surface_uv + motion + (frame_data.taa_jitter.zw - frame_data.taa_jitter.xy) * 0.5;
		vec3 current_view_position;
		if (svgf_reconstruct_view(frame_data.current_inv_projection, current_surface_uv, current_view_z, (params.control.w & 1u) != 0u, current_view_position)) {
			vec3 expected_previous_position = (frame_data.current_view_to_previous_view * vec4(current_view_position, 1.0)).xyz;
			vec3 current_normal_previous_view = transpose(mat3(frame_data.previous_view_to_world)) * current_normal;
			float expected_previous_view_z = -expected_previous_position.z;
			if (svgf_valid_view_z(expected_previous_view_z)) {
				vec2 parallax = svgf_specular_parallax(current_view_position, expected_previous_position, current_signal.a);
				reflection_motion_pixels = max(reflection_motion_pixels, length(parallax * vec2(size)));
				if (!current_dynamic && current_roughness < 0.12 && any(notEqual(parallax, vec2(0.0)))) {
					vec2 virtual_uv = previous_uv + parallax * (1.0 - smoothstep(0.04, 0.12, current_roughness));
					ivec2 virtual_pixel = ivec2(floor(virtual_uv * vec2(size)));
					if (all(greaterThanEqual(virtual_pixel, ivec2(0))) && all(lessThan(virtual_pixel, size))) {
						vec4 virtual_normal = texelFetch(sampler2D(previous_normal_roughness_input, nearest_sampler), virtual_pixel, 0);
						if (virtual_normal.a > 0.0 && dot(svgf_decode_normal(virtual_normal), current_normal) > 0.999) {
							previous_uv = virtual_uv;
						}
					}
				}
				vec2 previous_texel = previous_uv * vec2(size) - 0.5;
				bool previous_texel_in_range = svgf_finite_vec2(previous_texel) &&
						all(greaterThanEqual(previous_texel, vec2(-1.0))) && all(lessThanEqual(previous_texel, vec2(size)));
				if (previous_texel_in_range) {
					ivec2 previous_base = ivec2(floor(previous_texel));
					vec2 fraction = fract(previous_texel);
					for (int y = 0; y < 2; y++) {
						for (int x = 0; x < 2; x++) {
							ivec2 tap_pos = previous_base + ivec2(x, y);
							vec2 axis_weight = mix(vec2(1.0) - fraction, fraction, bvec2(x != 0, y != 0));
							float tap_weight = axis_weight.x * axis_weight.y;
							float tap_history_length;
							vec2 tap_moments;
							float tap_normal_agreement;
							if (!(tap_weight > 0.0) || !svgf_history_tap_valid(tap_pos, size, current_dynamic, current_normal, current_normal_previous_view, expected_previous_position, tap_history_length, tap_moments, tap_normal_agreement)) {
								continue;
							}
							vec4 history_signal = texelFetch(sampler2D(previous_signal_input, nearest_sampler), tap_pos, 0);
							if (!svgf_finite_vec4(history_signal)) {
								continue;
							}
							vec3 history_metadata = svgf_unpack_specular_metadata(history_signal.a);
							float mirror_response = 1.0 - smoothstep(0.03, 0.18, current_roughness);
							float hit_log_tolerance = mix(1.0, 0.03, mirror_response);
							float history_roughness_delta = abs(history_metadata.x - current_roughness);
							float lobe_roughness = min(history_metadata.x, current_roughness);
							float narrow_lobe = 1.0 - smoothstep(0.05, 0.20, lobe_roughness);
							if ((lobe_roughness < 0.10 && history_roughness_delta > 0.10) || abs(history_metadata.y - current_hit_log) > hit_log_tolerance) {
								continue;
							}
							tap_weight *= exp2(-history_roughness_delta * mix(4.0, 24.0, narrow_lobe));
							if (!(tap_weight > 1e-5)) {
								continue;
							}
							accumulated_history += max(history_signal.rgb, vec3(0.0)) * tap_weight;
							accumulated_moments += tap_moments * tap_weight;
							accumulated_history_length += tap_history_length * tap_weight;
							accumulated_history_screen_confidence += history_metadata.z * tap_weight;
							history_weight_sum += tap_weight;
							maximum_history_tap_weight = max(maximum_history_tap_weight, tap_weight);
						}
					}
				}
			}
		}
	}

	float current_moment_luminance = svgf_specular_luminance(current_signal.rgb);
	vec4 output_signal = current_signal;
	vec2 output_moments = vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance);
	float history_length = current_surface_valid ? 1.0 : 0.0;
	float output_screen_confidence = current_screen_confidence;
	bool history_footprint_valid = history_weight_sum > 1e-5 && maximum_history_tap_weight > 0.01;
	if (current_surface_valid && history_footprint_valid) {
		vec3 history_signal = accumulated_history / history_weight_sum;
		vec2 previous_moments = accumulated_moments / history_weight_sum;
		float previous_history_length = accumulated_history_length / history_weight_sum;
		float previous_screen_confidence = accumulated_history_screen_confidence / history_weight_sum;
		float reprojection_quality = !current_dynamic ? 1.0 : sqrt(clamp(history_weight_sum, 0.0, 1.0));
		float history_confidence = 1.0;
		float broad_history_response = smoothstep(0.05, 0.12, current_roughness);
		float stationary_history_length = mix(12.0, params.atrous.w, broad_history_response);
		float motion_history_response = max(current_dynamic ? 1.0 : 0.0, smoothstep(0.25, 2.0, reflection_motion_pixels));
		float moving_history_length = current_dynamic ? mix(12.0, 4.0, smoothstep(1.0, 8.0, reflection_motion_pixels)) : 12.0;
		float maximum_history_length = mix(stationary_history_length, moving_history_length, motion_history_response);
		if ((params.control.x & 4u) != 0u) {
			float temporal_roughness = max(current_roughness, 0.0316227766);
			maximum_history_length = mix(2.0, maximum_history_length, clamp(temporal_roughness / 0.05, 0.0, 1.0));
		}
		if (current_roughness == 0.0) {
			maximum_history_length = mix(maximum_history_length, 2.0, motion_history_response);
		}
		maximum_history_length = min(maximum_history_length, params.atrous.w);
		vec3 neighborhood_center;
		vec3 neighborhood_extent;
		if (svgf_temporal_neighborhood(current_signal.rgb, neighborhood_center, neighborhood_extent)) {
			vec3 history_ycocg = svgf_rgb_to_ycocg(history_signal);
			float clamp_response = motion_history_response;
			if ((params.control.x & 16u) != 0u) {
				float sample_delta = svgf_rgb_to_ycocg(current_signal.rgb).x - history_ycocg.x;
				float neighborhood_delta = neighborhood_center.x - history_ycocg.x;
				float coherent_delta = min(abs(sample_delta), sign(sample_delta) * neighborhood_delta);
				float noise_floor = 0.015 + 0.08 * history_ycocg.x + 0.2 * neighborhood_extent.x;
				float lighting_response = smoothstep(noise_floor, 2.0 * noise_floor, coherent_delta);
				maximum_history_length = mix(maximum_history_length, 1.0, lighting_response);
				clamp_response = max(clamp_response, lighting_response);
			}
			neighborhood_extent *= mix(4.0, 1.0, clamp_response);
			float history_deviation = sqrt(max(previous_moments.y - previous_moments.x * previous_moments.x, 0.0));
			float history_extent = min(history_deviation, 3.0 * previous_moments.x) / SPECULAR_MOMENT_LUMINANCE_SCALE;
			neighborhood_extent = max(neighborhood_extent, vec3(history_extent * broad_history_response * (1.0 - clamp_response)));
			vec3 clamped_history_ycocg = clamp(history_ycocg, neighborhood_center - neighborhood_extent, neighborhood_center + neighborhood_extent);
			vec3 normalized_history_error = abs(clamped_history_ycocg - history_ycocg) / max(neighborhood_extent, vec3(0.1));
			history_confidence = clamp(1.0 - length(normalized_history_error), 0.0, 1.0);
			history_confidence = history_confidence * 0.75 + 0.25;
			if (current_dynamic && reflection_motion_pixels > 0.25) {
				float mismatch = abs(history_ycocg.x - neighborhood_center.x) / max(neighborhood_extent.x, 0.03 + 0.05 * neighborhood_center.x);
				float response = smoothstep(0.35, 1.5, mismatch) * smoothstep(0.25, 2.0, reflection_motion_pixels);
				maximum_history_length = mix(maximum_history_length, min(maximum_history_length, 2.0), response);
			}
			history_signal = max(svgf_ycocg_to_rgb(clamped_history_ycocg), vec3(0.0));
		}
		float advanced_history_length = previous_history_length * history_confidence + 1.0;
		history_length = max(1.0, min(advanced_history_length, maximum_history_length) * reprojection_quality);
		float signal_alpha = 1.0 / max(history_length, 1.0);
		output_signal.rgb = mix(history_signal, current_signal.rgb, signal_alpha);
		float moment_alpha = max(signal_alpha, 0.5 * broad_history_response);
		output_moments = mix(max(previous_moments, vec2(0.0)), vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance), moment_alpha);
		output_screen_confidence = mix(previous_screen_confidence, current_screen_confidence, signal_alpha);
		float source_step = 1.0 / SPECULAR_SOURCE_METADATA_QUANTIZATION;
		float source_delta = current_screen_confidence - previous_screen_confidence;
		if (abs(source_delta) > source_step * 0.5) {
			float minimum_tracked_confidence = previous_screen_confidence + sign(source_delta) * min(abs(source_delta), source_step);
			output_screen_confidence = source_delta > 0.0 ? max(output_screen_confidence, minimum_tracked_confidence) : min(output_screen_confidence, minimum_tracked_confidence);
		}
		output_screen_confidence = clamp(output_screen_confidence, 0.0, 1.0);
	}

	output_signal = svgf_sanitize_signal(output_signal);
	if (!svgf_finite_vec2(output_moments)) {
		output_moments = vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance);
	}
	output_moments = clamp(output_moments, vec2(0.0), vec2(FP16_MAX));
	output_signal.a = svgf_pack_specular_metadata(current_roughness, current_hit_log, output_screen_confidence);
	imageStore(current_signal_output, pixel, output_signal);
	imageStore(current_moments_output, pixel, vec4(output_moments, 0.0, 0.0));
	current_encoded_normal.a = history_length / HISTORY_LENGTH_STORAGE_SCALE;
	imageStore(current_normal_roughness_output, pixel, current_encoded_normal);
}
#endif // SVGF_SPECULAR

#ifdef MODE_SPECULAR_SPATIAL

layout(set = 0, binding = 0) uniform texture2D filtered_signal_input;
layout(set = 0, binding = 1) uniform texture2D temporal_moments_input;
layout(set = 0, binding = 2) uniform texture2D current_normal_roughness_input;
layout(set = 0, binding = 3) uniform texture2D current_view_z_input;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D filtered_signal_output;
layout(set = 0, binding = 6) uniform texture2D current_raw_signal_input;
layout(set = 0, binding = 7) uniform texture2D current_roughness_input;
layout(set = 0, binding = 8) uniform texture2D current_motion_input;

uint svgf_spatial_hash(uvec2 value) {
	uint hash = value.x * 0x8da6b343u ^ value.y * 0xd8163841u;
	hash ^= hash >> 16u;
	hash *= 0x7feb352du;
	hash ^= hash >> 15u;
	hash *= 0x846ca68bu;
	return hash ^ (hash >> 16u);
}

void svgf_store_specular(ivec2 pixel, vec3 signal) {
	float hit_distance = texelFetch(sampler2D(current_raw_signal_input, nearest_sampler), pixel, 0).a;
	imageStore(filtered_signal_output, pixel, svgf_sanitize_signal(vec4(signal, hit_distance)));
}

void svgf_specular_spatial_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(filtered_signal_output);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	float center_packed_roughness_source = texelFetch(sampler2D(current_roughness_input, nearest_sampler), pixel, 0).a;
	vec2 center_specular_guide = svgf_unpack_specular_roughness_source(center_packed_roughness_source);
	float center_roughness = center_specular_guide.x;
	if (center_roughness >= 0.40) {
		imageStore(filtered_signal_output, pixel, vec4(0.0));
		return;
	}
	if (center_roughness == 0.0 && (params.control.x & 8u) != 0u) {
		// Medium and High already copied the mirror ray into this image during reconstruction.
		if ((params.control.z >> 2u) == 0u) {
			imageStore(filtered_signal_output, pixel, texelFetch(sampler2D(current_raw_signal_input, nearest_sampler), pixel, 0));
		}
		return;
	}
	vec4 center_signal = svgf_sanitize_signal(texelFetch(sampler2D(filtered_signal_input, nearest_sampler), pixel, 0));
	uint specular_quality = min(params.control.z >> 2u, 2u);
	float quality_response = float(specular_quality) * 0.5;
	float signal_resolution_scale = (params.control.x & 4u) != 0u ? 1.0 : 0.5;
	float maximum_radius = mix(4.0, 8.0, quality_response) * signal_resolution_scale;
	float filter_radius = maximum_radius * clamp(center_roughness * 8.0, 0.0, 1.0);
	if (filter_radius <= 1.0) {
		svgf_store_specular(pixel, center_signal.rgb);
		return;
	}
	float packed_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r;
	float center_view_z = abs(packed_view_z);
	vec4 center_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
	float motion_pixels = 0.0;
	ivec2 screen_size = textureSize(sampler2D(current_motion_input, nearest_sampler), 0);
	ivec2 surface_pixel = specular_surface_pixel(pixel, size, screen_size, packed_view_z);
	vec2 motion = texelFetch(sampler2D(current_motion_input, nearest_sampler), surface_pixel, 0).xy;
	if (svgf_finite_vec2(motion)) {
		motion_pixels = length(motion * vec2(size));
	}
	float motion_filter_response = smoothstep(0.25, 4.0, motion_pixels);
	vec3 center_normal = svgf_decode_normal(center_encoded_normal);
	if (!svgf_valid_view_z(center_view_z)) {
		svgf_store_specular(pixel, center_signal.rgb);
		return;
	}
	float history_length = center_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	float center_luminance = svgf_specular_luminance(center_signal.rgb);
	vec2 center_moments = texelFetch(sampler2D(temporal_moments_input, nearest_sampler), pixel, 0).xy;
	vec2 moments = center_moments;
	if (!svgf_finite_vec2(moments)) {
		moments = vec2(0.0);
	}
	moments = max(moments, vec2(0.0));
	if (history_length < 4.0) {
		vec2 moment_sum = vec2(0.0);
		float moment_weight_sum = 0.0;
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				ivec2 sample_pos = pixel + ivec2(x, y);
				if (any(lessThan(sample_pos, ivec2(0))) || any(greaterThanEqual(sample_pos, size))) {
					continue;
				}
				float sample_view_z = abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), sample_pos, 0).r);
				vec4 sample_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), sample_pos, 0);
				vec3 sample_normal = svgf_decode_normal(sample_encoded_normal);
				vec2 sample_moments = texelFetch(sampler2D(temporal_moments_input, nearest_sampler), sample_pos, 0).xy;
				if (!svgf_valid_view_z(sample_view_z) ||
						!svgf_finite_vec2(sample_moments) || sample_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE < 1.0) {
					continue;
				}
				float normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), params.atrous.x);
				float depth_scale = 0.02 + 0.02 * max(center_view_z, sample_view_z);
				float depth_weight = exp(-abs(center_view_z - sample_view_z) / depth_scale);
				float weight = float((2 - abs(x)) * (2 - abs(y))) * normal_weight * depth_weight;
				moment_sum += max(sample_moments, vec2(0.0)) * weight;
				moment_weight_sum += weight;
			}
		}
		if (moment_weight_sum > 1e-5) {
			moments = moment_sum / moment_weight_sum;
		}
	}
	float center_variance = svgf_variance_from_moments(moments, history_length);

	if (!svgf_finite_vec2(center_moments)) {
		svgf_store_specular(pixel, center_signal.rgb);
		return;
	}

	float young_history = 1.0 - smoothstep(1.0, 3.0, history_length);
	float standard_deviation = sqrt(max(center_variance, 0.0)) / SPECULAR_MOMENT_LUMINANCE_SCALE;
	float mean_luminance = max(center_moments.x / SPECULAR_MOMENT_LUMINANCE_SCALE, 0.1);
	float coefficient_of_variation = standard_deviation / mean_luminance;
	float noise_threshold = mix(0.55, 0.35, quality_response);
	bool filter_needed = young_history > 0.0 || coefficient_of_variation > noise_threshold ||
			(motion_filter_response > 0.1 && coefficient_of_variation > 0.2);
	if (!filter_needed || !svgf_finite_vec2(motion)) {
		svgf_store_specular(pixel, center_signal.rgb);
		return;
	}

	bool center_dynamic = packed_view_z < 0.0;
	vec3 center_metadata = svgf_unpack_specular_metadata(center_signal.a);
	bool center_far_hit = center_metadata.y > 1.0 - 0.5 / SPECULAR_HIT_METADATA_QUANTIZATION;
	int sample_count = min(4 + int(specular_quality) * 2 + (young_history > 0.0 ? 2 : 0), 8);
	float random_angle = float(svgf_spatial_hash(uvec2(pixel)) & 0xffffu) * (6.28318530718 / 65536.0);
	vec2 rotation = vec2(cos(random_angle), sin(random_angle));
	float roughness_response = smoothstep(0.05, 0.30, center_roughness);
	float lobe_width = max(center_roughness * center_roughness, 0.02) * mix(1.0, 1.5, young_history);
	float temporal_confidence = smoothstep(4.0, 16.0, history_length);
	float luminance_scale = (params.tuning.w * sqrt(max(center_variance, params.atrous.z)) +
									SPECULAR_MOMENT_LUMINANCE_SCALE * 1e-4) *
			mix(1.75, 1.0, temporal_confidence);
	luminance_scale *= mix(1.0, 1.6, roughness_response) * mix(1.0, 2.25, motion_filter_response);

	float depth_left = abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), max(pixel - ivec2(1, 0), ivec2(0)), 0).r);
	float depth_right = abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), min(pixel + ivec2(1, 0), size - 1), 0).r);
	float depth_up = abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), max(pixel - ivec2(0, 1), ivec2(0)), 0).r);
	float depth_down = abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), min(pixel + ivec2(0, 1), size - 1), 0).r);
	float gradient_left = svgf_valid_view_z(depth_left) ? center_view_z - depth_left : 0.0;
	float gradient_right = svgf_valid_view_z(depth_right) ? depth_right - center_view_z : gradient_left;
	float gradient_up = svgf_valid_view_z(depth_up) ? center_view_z - depth_up : 0.0;
	float gradient_down = svgf_valid_view_z(depth_down) ? depth_down - center_view_z : gradient_up;
	vec2 depth_gradient = vec2(
			abs(gradient_left) < abs(gradient_right) ? gradient_left : gradient_right,
			abs(gradient_up) < abs(gradient_down) ? gradient_up : gradient_down);
	float base_plane_threshold = max(0.001, params.atrous.y * (((params.control.w & 1u) != 0u) ? 1.0 : center_view_z));

	vec3 signal_sum = center_signal.rgb;
	float weight_sum = 1.0;
	for (int sample_index = 0; sample_index < 8; sample_index++) {
		if (sample_index >= sample_count) {
			break;
		}
		vec2 disk_sample = svgf_specular_disk(sample_index);
		vec2 rotated_sample = vec2(
				disk_sample.x * rotation.x - disk_sample.y * rotation.y,
				disk_sample.x * rotation.y + disk_sample.y * rotation.x);
		ivec2 offset = ivec2(round(rotated_sample * filter_radius));
		if (all(equal(offset, ivec2(0)))) {
			continue;
		}
		ivec2 sample_pos = pixel + offset;
		if (any(lessThan(sample_pos, ivec2(0))) || any(greaterThanEqual(sample_pos, size))) {
			continue;
		}

		float sample_packed_roughness_source = texelFetch(sampler2D(current_roughness_input, nearest_sampler), sample_pos, 0).a;
		vec2 sample_specular_guide = svgf_unpack_specular_roughness_source(sample_packed_roughness_source);
		if (sample_specular_guide.x >= 0.40) {
			continue;
		}
		float sample_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), sample_pos, 0).r;
		if ((sample_view_z < 0.0) != center_dynamic) {
			continue;
		}
		sample_view_z = abs(sample_view_z);
		vec4 sample_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), sample_pos, 0);
		vec3 sample_normal = svgf_decode_normal(sample_encoded_normal);
		if (!svgf_valid_view_z(sample_view_z) || sample_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE < 1.0) {
			continue;
		}
		float normal_cosine = max(dot(center_normal, sample_normal), 0.0);
		if (!(normal_cosine > 0.6)) {
			continue;
		}

		float roughness_delta = abs(sample_specular_guide.x - center_roughness);
		float lobe_roughness = min(sample_specular_guide.x, center_roughness);
		float narrow_lobe = 1.0 - smoothstep(0.05, 0.20, lobe_roughness);
		float roughness_weight = exp2(-roughness_delta * mix(4.0, 32.0, narrow_lobe));
		float expected_depth_delta = dot(depth_gradient, vec2(offset));
		float plane_threshold = base_plane_threshold * (1.0 + 0.35 * sqrt(float(dot(offset, offset))));
		if (abs((sample_view_z - center_view_z) - expected_depth_delta) > plane_threshold) {
			continue;
		}

		vec4 sample_signal = svgf_sanitize_signal(texelFetch(sampler2D(filtered_signal_input, nearest_sampler), sample_pos, 0));
		vec3 sample_metadata = svgf_unpack_specular_metadata(sample_signal.a);
		bool sample_far_hit = sample_metadata.y > 1.0 - 0.5 / SPECULAR_HIT_METADATA_QUANTIZATION;
		float broad_lobe_response = smoothstep(0.08, 0.25, center_roughness);
		float hit_class_weight = sample_far_hit == center_far_hit ? 1.0 : broad_lobe_response;
		if (!(hit_class_weight > 1e-5)) {
			continue;
		}
		float hit_weight = hit_class_weight * mix(exp2(-abs(sample_metadata.y - center_metadata.y) * 12.0), 1.0, broad_lobe_response);
		float source_weight = mix(exp2(-abs(sample_metadata.z - center_metadata.z) * 4.0), 1.0, roughness_response);
		float sample_luminance = svgf_specular_luminance(sample_signal.rgb);
		float luminance_weight = exp(-abs(center_luminance - sample_luminance) / max(luminance_scale, 1e-5));
		luminance_weight = mix(luminance_weight, 1.0, young_history * 0.75);
		float radial_weight = max(1.0 - 0.35 * length(disk_sample), 0.5);
		float weight = radial_weight * svgf_specular_normal_weight(normal_cosine, lobe_width * lobe_width) * roughness_weight * hit_weight * source_weight * luminance_weight;
		if (!(weight > 1e-5) || !svgf_finite_float(weight)) {
			continue;
		}

		signal_sum += sample_signal.rgb * weight;
		weight_sum += weight;
	}

	vec3 output_signal = signal_sum / max(weight_sum, 1e-5);
	svgf_store_specular(pixel, output_signal);
}
#endif // MODE_SPECULAR_SPATIAL
