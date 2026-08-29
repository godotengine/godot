#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant, std430) uniform Params {
	uvec4 control; // Mode controls: history valid / stride, history-fix frame count / A-Trous flags, current/previous orthographic bits.
	vec4 tuning; // Denoising range, temporal normal threshold, temporal relative depth, luminance phi.
	vec4 atrous; // Normal power, plane-distance threshold, variance epsilon, maximum history length.
}
params;

#if defined(MODE_TEMPORAL) || defined(MODE_HISTORY_FIX)

layout(set = 1, binding = 0, std140) uniform FrameData {
	mat4 current_inv_projection;
	mat4 previous_inv_projection;
	mat4 current_view_to_previous_view;
	mat4 current_view_to_world;
	mat4 previous_view_to_world;
	vec4 taa_jitter;
}
frame_data;

#endif

#ifdef MODE_TEMPORAL

layout(set = 0, binding = 0) uniform texture2D current_signal_input;
layout(set = 0, binding = 1) uniform texture2D current_normal_roughness_input;
layout(set = 0, binding = 2) uniform texture2D current_view_z_input;
layout(set = 0, binding = 3) uniform texture2D motion_input;
layout(set = 0, binding = 4) uniform texture2D previous_signal_input;
layout(set = 0, binding = 5) uniform texture2D previous_moments_input;
layout(set = 0, binding = 6) uniform texture2D previous_normal_roughness_input;
layout(set = 0, binding = 7) uniform texture2D previous_view_z_input;
layout(set = 0, binding = 8) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 9) uniform restrict writeonly image2D current_signal_output;
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image2D current_moments_output;
layout(rgba8, set = 0, binding = 11) uniform restrict writeonly image2D current_normal_roughness_output;
layout(r32f, set = 0, binding = 12) uniform restrict writeonly image2D current_view_z_output;

const int SPECULAR_NEIGHBORHOOD_RADIUS = 2;
const int SPECULAR_NEIGHBORHOOD_SHARED_SIZE = 12;
const int SPECULAR_NEIGHBORHOOD_SHARED_COUNT = SPECULAR_NEIGHBORHOOD_SHARED_SIZE * SPECULAR_NEIGHBORHOOD_SHARED_SIZE;
shared vec4 specular_neighborhood_ycocg[SPECULAR_NEIGHBORHOOD_SHARED_SIZE][SPECULAR_NEIGHBORHOOD_SHARED_SIZE];

#endif

#ifdef MODE_HISTORY_FIX

layout(set = 0, binding = 0) uniform texture2D history_fix_signal_input;
layout(set = 0, binding = 1) uniform texture2D history_fix_moments_input;
layout(set = 0, binding = 2) uniform texture2D history_fix_normal_input;
layout(set = 0, binding = 3) uniform texture2D history_fix_view_z_input;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D history_fix_signal_output;
layout(rgba16f, set = 0, binding = 6) uniform restrict writeonly image2D history_fix_moments_output;

#endif

#ifdef MODE_ATROUS

layout(set = 0, binding = 0) uniform texture2D filtered_signal_input;
layout(set = 0, binding = 1) uniform texture2D temporal_moments_input;
layout(set = 0, binding = 2) uniform texture2D current_normal_roughness_input;
layout(set = 0, binding = 3) uniform texture2D current_view_z_input;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D filtered_signal_output;
layout(set = 0, binding = 6) uniform texture2D current_raw_signal_input;
layout(set = 0, binding = 7) uniform texture2D current_roughness_input;
layout(set = 0, binding = 8) uniform texture2D current_motion_input;

#endif

const vec3 LUMINANCE_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);
const float FP16_MAX = 65504.0;
const float HISTORY_LENGTH_STORAGE_SCALE = 255.0;
const float MOMENT_LUMINANCE_SCALE = 64.0;
const float MOMENT_LUMINANCE_MAX = 255.0;
const float SPECULAR_METADATA_QUANTIZATION = 31.0;
const float SPECULAR_HIT_METADATA_QUANTIZATION = 7.0;
const float SPECULAR_SOURCE_METADATA_QUANTIZATION = 3.0;
const float SPECULAR_HIT_LOG_RANGE = 16.0;
const float SPECULAR_MOMENT_LUMINANCE_SCALE = 16.0;
const float SPECULAR_DENOISER_TONEMAP_RANGE = 10.0;
const float SPECULAR_DENOISER_INV_TONEMAP_RANGE = 1.0 / SPECULAR_DENOISER_TONEMAP_RANGE;
const float SPECULAR_DENOISER_MAX_LUMINANCE = 8.0;

bool svgf_finite_float(float value) {
	return !isnan(value) && !isinf(value);
}

bool svgf_finite_vec2(vec2 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

bool svgf_finite_vec3(vec3 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

bool svgf_finite_vec4(vec4 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

float svgf_luminance(vec3 value) {
	return max(dot(max(value, vec3(0.0)), LUMINANCE_WEIGHTS), 0.0);
}

vec3 svgf_specular_project_denoiser_space(vec3 value) {
	value = max(value, vec3(0.0));
	float signal_luminance = svgf_luminance(value);
	if (signal_luminance > SPECULAR_DENOISER_MAX_LUMINANCE) {
		value *= SPECULAR_DENOISER_MAX_LUMINANCE / signal_luminance;
	}
	return value;
}

vec3 svgf_specular_to_denoiser_space(vec3 value) {
	value = max(value, vec3(0.0));
	value /= 1.0 + svgf_luminance(value) * SPECULAR_DENOISER_INV_TONEMAP_RANGE;
	return svgf_specular_project_denoiser_space(value);
}

vec3 svgf_specular_from_denoiser_space(vec3 value) {
	value = svgf_specular_project_denoiser_space(value);
	float inverse_weight = 1.0 / max(1.0 - svgf_luminance(value) * SPECULAR_DENOISER_INV_TONEMAP_RANGE, 1e-3);
	return value * inverse_weight;
}

vec3 svgf_rgb_to_ycocg(vec3 value) {
	return vec3(
			dot(value, vec3(0.25, 0.5, 0.25)),
			dot(value, vec3(0.5, 0.0, -0.5)),
			dot(value, vec3(-0.25, 0.5, -0.25)));
}

vec3 svgf_ycocg_to_rgb(vec3 value) {
	return vec3(value.x + value.y - value.z, value.x + value.z, value.x - value.y - value.z);
}

float svgf_filter_luminance(vec3 value, bool specular_mode) {
	float luminance = svgf_luminance(value);
	return specular_mode ? min(luminance * SPECULAR_MOMENT_LUMINANCE_SCALE, MOMENT_LUMINANCE_MAX) : luminance * MOMENT_LUMINANCE_SCALE;
}

float svgf_hit_log(float hit_distance) {
	return log2(1.0 + clamp(hit_distance, 0.0, FP16_MAX)) / SPECULAR_HIT_LOG_RANGE;
}

vec4 svgf_sanitize_signal(vec4 value) {
	if (!svgf_finite_vec4(value)) {
		return vec4(0.0);
	}
	return clamp(value, vec4(0.0), vec4(FP16_MAX));
}

vec2 svgf_unpack_specular_roughness_source(float packed_value) {
	float packed = floor(clamp(packed_value, 0.0, 1.0) * 255.0 + 0.5);
	float screen_source = floor(packed / 128.0);
	float roughness_bin = packed - screen_source * 128.0;
	return vec2(roughness_bin / 127.0, screen_source);
}

float svgf_pack_specular_metadata(float roughness, float hit_distance, float screen_confidence) {
	float roughness_bin = floor(clamp(roughness, 0.0, 1.0) * SPECULAR_METADATA_QUANTIZATION + 0.5);
	float hit_bin = floor(clamp(svgf_hit_log(hit_distance), 0.0, 1.0) * SPECULAR_HIT_METADATA_QUANTIZATION + 0.5);
	float source_bin = floor(clamp(screen_confidence, 0.0, 1.0) * SPECULAR_SOURCE_METADATA_QUANTIZATION + 0.5);
	return (roughness_bin * 32.0 + hit_bin * 4.0 + source_bin) / 1023.0;
}

vec3 svgf_unpack_specular_metadata(float metadata) {
	float packed = floor(clamp(metadata, 0.0, 1.0) * 1023.0 + 0.5);
	float roughness_bin = floor(packed / 32.0);
	float remainder = packed - roughness_bin * 32.0;
	float hit_bin = floor(remainder / 4.0);
	float source_bin = remainder - hit_bin * 4.0;
	return vec3(
			roughness_bin / SPECULAR_METADATA_QUANTIZATION,
			hit_bin / SPECULAR_HIT_METADATA_QUANTIZATION,
			source_bin / SPECULAR_SOURCE_METADATA_QUANTIZATION);
}

bool svgf_decode_normal(vec4 encoded, out vec3 r_normal) {
	vec3 unpacked = encoded.xyz * 2.0 - 1.0;
	float length_squared = dot(unpacked, unpacked);
	if (!svgf_finite_vec3(unpacked) || !(length_squared > 1e-6)) {
		r_normal = vec3(0.0, 0.0, 1.0);
		return false;
	}
	r_normal = unpacked * inversesqrt(length_squared);
	return true;
}

bool svgf_valid_view_z(float view_z) {
	return svgf_finite_float(view_z) && view_z > 0.0 && view_z <= params.tuning.x;
}

#if defined(MODE_TEMPORAL) || defined(MODE_HISTORY_FIX)

bool svgf_reconstruct_view(mat4 inv_projection, vec2 uv, float view_z, bool orthographic, out vec3 r_view_position) {
	vec2 ndc = uv * 2.0 - 1.0;
	if (!orthographic) {
		// A homogeneous point defines the ray for perspective and asymmetric frusta;
		// its W division cancels when scaled to View Z.
		vec3 point_h = (inv_projection * vec4(ndc, 0.5, 1.0)).xyz;
		if (!svgf_finite_vec3(point_h) || abs(point_h.z) <= 1e-7) {
			r_view_position = vec3(0.0);
			return false;
		}
		r_view_position = point_h * (-view_z / point_h.z);
		return svgf_finite_vec3(r_view_position);
	}

	vec4 point_a_h = inv_projection * vec4(ndc, 0.25, 1.0);
	vec4 point_b_h = inv_projection * vec4(ndc, 0.75, 1.0);
	if (!svgf_finite_vec4(point_a_h) || !svgf_finite_vec4(point_b_h) || abs(point_a_h.w) <= 1e-7 || abs(point_b_h.w) <= 1e-7) {
		r_view_position = vec3(0.0);
		return false;
	}
	vec3 point_a = point_a_h.xyz / point_a_h.w;
	vec3 point_b = point_b_h.xyz / point_b_h.w;
	vec3 ray = point_b - point_a;
	if (!svgf_finite_vec3(point_a) || !svgf_finite_vec3(point_b) || abs(ray.z) <= 1e-7) {
		r_view_position = vec3(0.0);
		return false;
	}
	r_view_position = point_a + ray * ((-view_z - point_a.z) / ray.z);
	return svgf_finite_vec3(r_view_position);
}

bool svgf_decode_surface_metadata(vec2 packed_metadata, out vec2 r_surface_uv_offset, out bool r_dynamic) {
	r_dynamic = packed_metadata.y > 1.0;
	r_surface_uv_offset = vec2(packed_metadata.x, r_dynamic ? packed_metadata.y - 2.0 : packed_metadata.y);
	return svgf_finite_vec2(r_surface_uv_offset) && all(lessThanEqual(abs(r_surface_uv_offset), vec2(1.0)));
}

#endif

#ifdef MODE_TEMPORAL

bool svgf_history_tap_valid(ivec2 tap_pos, ivec2 size, bool current_dynamic, vec3 current_normal, vec3 current_normal_previous_view, vec3 expected_previous_position, out float r_history_length, out vec2 r_moments) {
	r_history_length = 0.0;
	r_moments = vec2(0.0);
	if (!svgf_finite_vec3(expected_previous_position) || expected_previous_position.z >= -1e-5) {
		return false;
	}
	if (any(lessThan(tap_pos, ivec2(0))) || any(greaterThanEqual(tap_pos, size))) {
		return false;
	}
	float previous_view_z = texelFetch(sampler2D(previous_view_z_input, nearest_sampler), tap_pos, 0).r;
	vec4 previous_encoded_normal = texelFetch(sampler2D(previous_normal_roughness_input, nearest_sampler), tap_pos, 0);
	vec3 previous_normal;
	bool previous_normal_valid = svgf_decode_normal(previous_encoded_normal, previous_normal);
	if (!previous_normal_valid || !svgf_valid_view_z(previous_view_z)) {
		return false;
	}
	vec4 previous_moments_surface_offset = texelFetch(sampler2D(previous_moments_input, nearest_sampler), tap_pos, 0);
	if (!svgf_finite_vec4(previous_moments_surface_offset)) {
		return false;
	}
	vec2 previous_surface_uv_offset;
	bool previous_dynamic;
	if (!svgf_decode_surface_metadata(previous_moments_surface_offset.zw, previous_surface_uv_offset, previous_dynamic) || previous_dynamic != current_dynamic) {
		return false;
	}
	float expected_previous_view_z = -expected_previous_position.z;
	float relative_depth_tolerance = current_dynamic ? max(params.tuning.z * 6.0, 0.08) : max(params.tuning.z * 6.0, 0.02);
	float depth_tolerance = 0.05 + relative_depth_tolerance * max(expected_previous_view_z, previous_view_z);
	float normal_threshold = current_dynamic ? min(params.tuning.y, 0.7) : params.tuning.y;
	r_history_length = previous_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	r_moments = max(previous_moments_surface_offset.xy, vec2(0.0));
	if (!svgf_finite_float(r_history_length) || r_history_length < 1.0 ||
			abs(previous_view_z - expected_previous_view_z) > depth_tolerance ||
			dot(current_normal, previous_normal) < normal_threshold) {
		return false;
	}
	if (current_dynamic) {
		return true;
	}
	vec2 tap_uv = (vec2(tap_pos) + 0.5) / vec2(size) + previous_surface_uv_offset;
	vec3 previous_position;
	if (!svgf_reconstruct_view(frame_data.previous_inv_projection, tap_uv, previous_view_z, (params.control.w & 2u) != 0u, previous_position)) {
		return false;
	}
	float plane_tolerance = 0.01 + params.tuning.z * expected_previous_view_z;
	return abs(dot(previous_position - expected_previous_position, current_normal_previous_view)) <= plane_tolerance;
}

bool svgf_specular_neighborhood(vec3 center_signal, out vec3 r_center, out vec3 r_extent) {
	vec3 center_ycocg = svgf_rgb_to_ycocg(center_signal);
	vec3 sample_sum = center_ycocg;
	vec3 sample_square_sum = center_ycocg * center_ycocg;
	float sample_count = 1.0;
	ivec2 shared_center = ivec2(gl_LocalInvocationID.xy) + ivec2(SPECULAR_NEIGHBORHOOD_RADIUS);
	for (int y = -SPECULAR_NEIGHBORHOOD_RADIUS; y <= SPECULAR_NEIGHBORHOOD_RADIUS; y++) {
		for (int x = -SPECULAR_NEIGHBORHOOD_RADIUS; x <= SPECULAR_NEIGHBORHOOD_RADIUS; x++) {
			if ((x == 0 && y == 0) || (abs(x) == SPECULAR_NEIGHBORHOOD_RADIUS && abs(y) == SPECULAR_NEIGHBORHOOD_RADIUS)) {
				continue;
			}
			vec4 packed_sample = specular_neighborhood_ycocg[shared_center.y + y][shared_center.x + x];
			if (packed_sample.w < 0.5) {
				continue;
			}
			vec3 sample_ycocg = packed_sample.xyz;
			sample_sum += sample_ycocg;
			sample_square_sum += sample_ycocg * sample_ycocg;
			sample_count += 1.0;
		}
	}
	if (sample_count < 2.0) {
		r_center = center_ycocg;
		r_extent = vec3(0.0);
		return false;
	}
	r_center = sample_sum / sample_count;
	vec3 variance = max(sample_square_sum / sample_count - r_center * r_center, vec3(0.0));
	r_extent = sqrt(variance);
	return svgf_finite_vec3(r_center) && svgf_finite_vec3(r_extent);
}

void svgf_temporal_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(current_signal_output);
	bool specular_mode = (params.control.x & 2u) != 0u;
	bool pixel_in_bounds = all(lessThan(pixel, size));
	if (specular_mode) {
		ivec2 shared_origin = ivec2(gl_WorkGroupID.xy) * ivec2(gl_WorkGroupSize.xy) - ivec2(SPECULAR_NEIGHBORHOOD_RADIUS);
		for (uint shared_index = gl_LocalInvocationIndex; shared_index < uint(SPECULAR_NEIGHBORHOOD_SHARED_COUNT); shared_index += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
			ivec2 shared_position = ivec2(int(shared_index) % SPECULAR_NEIGHBORHOOD_SHARED_SIZE, int(shared_index) / SPECULAR_NEIGHBORHOOD_SHARED_SIZE);
			ivec2 sample_position = shared_origin + shared_position;
			vec4 packed_sample = vec4(0.0);
			if (all(greaterThanEqual(sample_position, ivec2(0))) && all(lessThan(sample_position, size))) {
				float sample_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), sample_position, 0).r;
				vec4 sample_signal = texelFetch(sampler2D(current_signal_input, nearest_sampler), sample_position, 0);
				if (svgf_valid_view_z(sample_view_z) && svgf_finite_vec4(sample_signal) && sample_signal.a > 0.0) {
					packed_sample = vec4(svgf_rgb_to_ycocg(svgf_specular_to_denoiser_space(sample_signal.rgb)), 1.0);
				}
			}
			specular_neighborhood_ycocg[shared_position.y][shared_position.x] = packed_sample;
		}
		memoryBarrierShared();
		barrier();
	}
	if (!pixel_in_bounds) {
		return;
	}

	vec4 current_signal = svgf_sanitize_signal(texelFetch(sampler2D(current_signal_input, nearest_sampler), pixel, 0));
	if (specular_mode) {
		current_signal.rgb = svgf_specular_to_denoiser_space(current_signal.rgb);
	}
	vec4 current_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
	vec2 current_specular_guide = specular_mode ? svgf_unpack_specular_roughness_source(current_encoded_normal.a) : vec2(clamp(current_encoded_normal.a, 0.0, 1.0), 0.0);
	float current_roughness = current_specular_guide.x;
	float current_screen_confidence = current_specular_guide.y;
	float current_hit_log = svgf_hit_log(current_signal.a);
	float current_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r;
	vec3 current_normal;
	bool current_surface_valid = svgf_valid_view_z(current_view_z) && svgf_decode_normal(current_encoded_normal, current_normal);
	vec4 motion_sample = texelFetch(sampler2D(motion_input, nearest_sampler), pixel, 0);
	vec2 current_surface_uv_offset;
	bool current_dynamic;
	bool motion_valid = svgf_finite_vec4(motion_sample) && svgf_decode_surface_metadata(motion_sample.zw, current_surface_uv_offset, current_dynamic);
	if (!motion_valid) {
		motion_sample = vec4(0.0);
		current_surface_uv_offset = vec2(0.0);
		current_dynamic = false;
	}

	vec3 accumulated_history = vec3(0.0);
	vec2 accumulated_moments = vec2(0.0);
	float accumulated_history_length = 0.0;
	float accumulated_history_screen_confidence = 0.0;
	float history_weight_sum = 0.0;
	if ((params.control.x & 1u) != 0u && current_surface_valid && motion_valid) {
		vec2 motion = motion_sample.xy;
		vec2 current_uv = (vec2(pixel) + 0.5) / vec2(size);
		vec2 current_surface_uv = current_uv + current_surface_uv_offset;
		vec2 previous_uv = current_surface_uv + motion + (frame_data.taa_jitter.zw - frame_data.taa_jitter.xy) * 0.5;
		vec3 current_view_position;
		if (svgf_finite_vec4(motion_sample) && svgf_finite_vec2(previous_uv) && svgf_finite_vec2(current_surface_uv) &&
				svgf_reconstruct_view(frame_data.current_inv_projection, current_surface_uv, current_view_z, (params.control.w & 1u) != 0u, current_view_position)) {
			vec3 expected_previous_position = (frame_data.current_view_to_previous_view * vec4(current_view_position, 1.0)).xyz;
			vec3 current_normal_previous_view = transpose(mat3(frame_data.previous_view_to_world)) * current_normal;
			float expected_previous_view_z = -expected_previous_position.z;
			if (svgf_valid_view_z(expected_previous_view_z)) {
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
							if (!(tap_weight > 0.0) || !svgf_history_tap_valid(tap_pos, size, current_dynamic, current_normal, current_normal_previous_view, expected_previous_position, tap_history_length, tap_moments)) {
								continue;
							}
							vec4 history_signal = texelFetch(sampler2D(previous_signal_input, nearest_sampler), tap_pos, 0);
							if (!svgf_finite_vec4(history_signal)) {
								continue;
							}
							float history_screen_confidence = 0.0;
							if (specular_mode) {
								vec3 history_metadata = svgf_unpack_specular_metadata(history_signal.a);
								float mirror_response = 1.0 - smoothstep(0.03, 0.18, current_roughness);
								float hit_log_tolerance = mix(1.0, 1.0 / SPECULAR_HIT_METADATA_QUANTIZATION, mirror_response);
								if (abs(history_metadata.x - current_roughness) > 0.08 || abs(history_metadata.y - current_hit_log) > hit_log_tolerance) {
									continue;
								}
								history_screen_confidence = history_metadata.z;
							}
							accumulated_history += max(history_signal.rgb, vec3(0.0)) * tap_weight;
							accumulated_moments += tap_moments * tap_weight;
							accumulated_history_length += tap_history_length * tap_weight;
							accumulated_history_screen_confidence += history_screen_confidence * tap_weight;
							history_weight_sum += tap_weight;
						}
					}
				}
			}
		}
	}

	float current_moment_luminance = min(svgf_filter_luminance(current_signal.rgb, specular_mode), MOMENT_LUMINANCE_MAX);
	vec4 output_signal = current_signal;
	vec2 output_moments = vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance);
	float history_length = current_surface_valid ? 1.0 : 0.0;
	float output_screen_confidence = current_screen_confidence;
	if (current_surface_valid && history_weight_sum > 1e-5) {
		vec3 history_signal = accumulated_history / history_weight_sum;
		vec2 previous_moments = accumulated_moments / history_weight_sum;
		float previous_history_length = accumulated_history_length / history_weight_sum;
		float previous_screen_confidence = accumulated_history_screen_confidence / history_weight_sum;
		float reprojection_quality = sqrt(clamp(history_weight_sum, 0.0, 1.0));
		float maximum_history_length = params.atrous.w;
		float history_confidence = 1.0;
		if (specular_mode) {
			if ((params.control.x & 4u) != 0u) {
				maximum_history_length = mix(2.0, maximum_history_length, clamp(current_roughness / 0.05, 0.0, 1.0));
			}
			vec3 neighborhood_center;
			vec3 neighborhood_extent;
			if (svgf_specular_neighborhood(current_signal.rgb, neighborhood_center, neighborhood_extent)) {
				vec3 history_ycocg = svgf_rgb_to_ycocg(history_signal);
				vec3 clamped_history_ycocg = clamp(history_ycocg, neighborhood_center - neighborhood_extent, neighborhood_center + neighborhood_extent);
				vec3 normalized_history_error = abs(clamped_history_ycocg - history_ycocg) / max(neighborhood_extent, vec3(0.1));
				history_confidence = clamp(1.0 - length(normalized_history_error), 0.0, 1.0);
				history_confidence = history_confidence * 0.75 + 0.25;
				history_signal = svgf_specular_project_denoiser_space(svgf_ycocg_to_rgb(clamped_history_ycocg));
			}
		}
		float advanced_history_length = specular_mode ? previous_history_length * history_confidence + 1.0 : previous_history_length + 1.0;
		history_length = max(1.0, min(advanced_history_length, maximum_history_length) * reprojection_quality);
		float signal_alpha = 1.0 / max(history_length, 1.0);
		output_signal.rgb = mix(history_signal, current_signal.rgb, signal_alpha);
		output_moments = mix(max(previous_moments, vec2(0.0)), vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance), signal_alpha);
		output_screen_confidence = mix(previous_screen_confidence, current_screen_confidence, signal_alpha);
		if (specular_mode) {
			float source_step = 1.0 / SPECULAR_SOURCE_METADATA_QUANTIZATION;
			float source_delta = current_screen_confidence - previous_screen_confidence;
			if (abs(source_delta) > source_step * 0.5) {
				float minimum_tracked_confidence = previous_screen_confidence + sign(source_delta) * min(abs(source_delta), source_step);
				output_screen_confidence = source_delta > 0.0 ? max(output_screen_confidence, minimum_tracked_confidence) : min(output_screen_confidence, minimum_tracked_confidence);
			}
			output_screen_confidence = clamp(output_screen_confidence, 0.0, 1.0);
		}
	}

	output_signal = svgf_sanitize_signal(output_signal);
	if (!svgf_finite_vec2(output_moments)) {
		output_moments = vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance);
	}
	output_moments = clamp(output_moments, vec2(0.0), vec2(FP16_MAX));
	output_signal.a = specular_mode ? svgf_pack_specular_metadata(current_roughness, current_signal.a, output_screen_confidence) : current_signal.a;
	imageStore(current_signal_output, pixel, output_signal);
	imageStore(current_moments_output, pixel, vec4(output_moments, current_surface_valid && motion_valid ? motion_sample.zw : vec2(0.0)));
	current_encoded_normal.a = history_length / HISTORY_LENGTH_STORAGE_SCALE;
	imageStore(current_normal_roughness_output, pixel, current_encoded_normal);
	imageStore(current_view_z_output, pixel, vec4(current_view_z, 0.0, 0.0, 0.0));
}

#endif

#ifdef MODE_HISTORY_FIX

int svgf_mirror_coordinate(int position, int size) {
	if (size <= 1) {
		return 0;
	}
	int period = size * 2;
	int folded = position % period;
	if (folded < 0) {
		folded += period;
	}
	return folded < size ? folded : period - 1 - folded;
}

ivec2 svgf_mirror_pixel(ivec2 pixel, ivec2 size) {
	return ivec2(
			svgf_mirror_coordinate(pixel.x, size.x),
			svgf_mirror_coordinate(pixel.y, size.y));
}

void svgf_store_history_fix(ivec2 pixel, vec4 signal, vec4 moments) {
	imageStore(history_fix_signal_output, pixel, svgf_sanitize_signal(signal));
	if (!svgf_finite_vec4(moments)) {
		moments = vec4(0.0);
	}
	moments.xy = clamp(moments.xy, vec2(0.0), vec2(FP16_MAX));
	imageStore(history_fix_moments_output, pixel, moments);
}

void svgf_history_fix_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(history_fix_signal_output);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	vec4 center_signal = svgf_sanitize_signal(texelFetch(sampler2D(history_fix_signal_input, nearest_sampler), pixel, 0));
	vec4 center_moments = texelFetch(sampler2D(history_fix_moments_input, nearest_sampler), pixel, 0);
	vec4 center_encoded_normal = texelFetch(sampler2D(history_fix_normal_input, nearest_sampler), pixel, 0);
	float center_view_z = texelFetch(sampler2D(history_fix_view_z_input, nearest_sampler), pixel, 0).r;
	vec3 center_normal_world;
	vec2 center_surface_uv_offset;
	bool center_dynamic;
	float history_length = center_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	float history_fix_frame_count = float(params.control.z);
	if (!svgf_finite_vec4(center_moments) || !svgf_valid_view_z(center_view_z) ||
			!svgf_decode_normal(center_encoded_normal, center_normal_world) ||
			!svgf_decode_surface_metadata(center_moments.zw, center_surface_uv_offset, center_dynamic) ||
			!svgf_finite_float(history_length) || history_length < 1.0 ||
			history_fix_frame_count < 1.0 || history_length > history_fix_frame_count) {
		svgf_store_history_fix(pixel, center_signal, center_moments);
		return;
	}

	vec2 center_uv = (vec2(pixel) + 0.5) / vec2(size) + center_surface_uv_offset;
	vec3 center_view_position;
	bool orthographic = (params.control.w & 1u) != 0u;
	if (!svgf_finite_vec2(center_uv) || !svgf_reconstruct_view(frame_data.current_inv_projection, center_uv, center_view_z, orthographic, center_view_position)) {
		svgf_store_history_fix(pixel, center_signal, center_moments);
		return;
	}
	vec3 center_normal_view = transpose(mat3(frame_data.current_view_to_world)) * center_normal_world;
	float center_normal_view_length_squared = dot(center_normal_view, center_normal_view);
	if (!svgf_finite_vec3(center_normal_view) || !(center_normal_view_length_squared > 1e-6)) {
		svgf_store_history_fix(pixel, center_signal, center_moments);
		return;
	}
	center_normal_view *= inversesqrt(center_normal_view_length_squared);

	// Use a contracting sparse footprint to reconstruct short history without recurrent filtering.
	int base_stride = max(int(params.control.y), 1);
	int stride = max(int(round(float(base_stride) / (1.0 + history_length))), 1);
	float plane_threshold = max(0.001, params.atrous.y * (orthographic ? 1.0 : center_view_z));
	float normal_power = max(params.atrous.x, 0.01);
	vec3 signal_sum = center_signal.rgb;
	vec2 moments_sum = max(center_moments.xy, vec2(0.0));
	float weight_sum = 1.0;

	for (int y = -2; y <= 2; y++) {
		for (int x = -2; x <= 2; x++) {
			if (x == 0 && y == 0) {
				continue;
			}
			ivec2 sample_pos = svgf_mirror_pixel(pixel + ivec2(x, y) * stride, size);
			float sample_view_z = texelFetch(sampler2D(history_fix_view_z_input, nearest_sampler), sample_pos, 0).r;
			vec4 sample_encoded_normal = texelFetch(sampler2D(history_fix_normal_input, nearest_sampler), sample_pos, 0);
			vec4 sample_moments = texelFetch(sampler2D(history_fix_moments_input, nearest_sampler), sample_pos, 0);
			vec4 sample_signal = texelFetch(sampler2D(history_fix_signal_input, nearest_sampler), sample_pos, 0);
			vec3 sample_normal_world;
			vec2 sample_surface_uv_offset;
			bool sample_dynamic;
			if (!svgf_valid_view_z(sample_view_z) || !svgf_decode_normal(sample_encoded_normal, sample_normal_world) ||
					!svgf_finite_vec4(sample_moments) || !svgf_finite_vec4(sample_signal) ||
					!svgf_decode_surface_metadata(sample_moments.zw, sample_surface_uv_offset, sample_dynamic) || sample_dynamic != center_dynamic) {
				continue;
			}

			vec2 sample_uv = (vec2(sample_pos) + 0.5) / vec2(size) + sample_surface_uv_offset;
			vec3 sample_view_position;
			if (!svgf_finite_vec2(sample_uv) || !svgf_reconstruct_view(frame_data.current_inv_projection, sample_uv, sample_view_z, orthographic, sample_view_position)) {
				continue;
			}
			float plane_distance = abs(dot(sample_view_position - center_view_position, center_normal_view));
			if (!center_dynamic && (!svgf_finite_float(plane_distance) || plane_distance >= plane_threshold)) {
				continue;
			}
			float normal_weight = pow(max(dot(center_normal_world, sample_normal_world), 0.0), normal_power);
			if (!svgf_finite_float(normal_weight) || normal_weight <= 1e-4) {
				continue;
			}

			signal_sum += max(sample_signal.rgb, vec3(0.0)) * normal_weight;
			moments_sum += max(sample_moments.xy, vec2(0.0)) * normal_weight;
			weight_sum += normal_weight;
		}
	}

	vec4 output_signal = center_signal;
	vec4 output_moments = center_moments;
	if (svgf_finite_float(weight_sum) && weight_sum > 1e-5) {
		output_signal.rgb = signal_sum / weight_sum;
		output_moments.xy = moments_sum / weight_sum;
	}
	// Reconstruct radiance and moments while preserving center-pixel metadata.
	output_signal.a = center_signal.a;
	output_moments.zw = center_moments.zw;
	svgf_store_history_fix(pixel, output_signal, output_moments);
}

#endif

#ifdef MODE_ATROUS

float svgf_variance_from_moments(vec2 moments, float history_length) {
	if (!svgf_finite_vec2(moments)) {
		return 0.0;
	}
	moments = max(moments, vec2(0.0));
	float variance = max(moments.y - moments.x * moments.x, 0.0);
	variance *= max(1.0, 4.0 / (history_length + 1.0));
	return svgf_finite_float(variance) ? clamp(variance, 0.0, FP16_MAX) : 0.0;
}

void svgf_store_atrous(ivec2 pixel, vec3 signal, float variance) {
	float packed_value = clamp(variance, 0.0, FP16_MAX);
	if ((params.control.z & 2u) != 0u) {
		if ((params.control.x & 2u) != 0u) {
			signal = svgf_specular_from_denoiser_space(signal);
		}
		float hit_distance = texelFetch(sampler2D(current_raw_signal_input, nearest_sampler), pixel, 0).a;
		packed_value = svgf_finite_float(hit_distance) ? clamp(hit_distance, 0.0, FP16_MAX) : 0.0;
	}
	imageStore(filtered_signal_output, pixel, svgf_sanitize_signal(vec4(signal, packed_value)));
}

void svgf_atrous_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(filtered_signal_output);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	bool first_iteration = (params.control.z & 1u) != 0u;
	vec4 center_signal = svgf_sanitize_signal(texelFetch(sampler2D(filtered_signal_input, nearest_sampler), pixel, 0));
	float center_variance = first_iteration ? 0.0 : center_signal.a;
	float center_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r;
	vec4 center_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
	bool specular_mode = (params.control.x & 2u) != 0u;
	float center_packed_roughness_source = texelFetch(sampler2D(current_roughness_input, nearest_sampler), pixel, 0).a;
	vec2 center_specular_guide = specular_mode ? svgf_unpack_specular_roughness_source(center_packed_roughness_source) : vec2(center_packed_roughness_source, 0.0);
	float center_roughness = center_specular_guide.x;
	float motion_pixels = 0.0;
	if (specular_mode) {
		vec2 motion = texelFetch(sampler2D(current_motion_input, nearest_sampler), pixel, 0).xy;
		if (svgf_finite_vec2(motion)) {
			motion_pixels = length(motion * vec2(size));
		}
	}
	float motion_filter_response = smoothstep(0.25, 4.0, motion_pixels);
	vec3 center_normal;
	if (!svgf_valid_view_z(center_view_z) || !svgf_decode_normal(center_encoded_normal, center_normal)) {
		svgf_store_atrous(pixel, center_signal.rgb, center_variance);
		return;
	}
	float history_length = center_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	int step_width = max(int(params.control.y), 1);
	float center_luminance = svgf_filter_luminance(center_signal.rgb, specular_mode);
	vec2 center_moments = texelFetch(sampler2D(temporal_moments_input, nearest_sampler), pixel, 0).xy;
	if (first_iteration) {
		vec2 moments = center_moments;
		if (!svgf_finite_vec2(moments)) {
			moments = vec2(0.0);
		}
		moments = max(moments, vec2(0.0));
		if (history_length < 4.0) {
			vec2 moment_sum = vec2(0.0);
			float moment_weight_sum = 0.0;
			const float moment_kernel[3] = float[](1.0, 2.0, 1.0);
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					ivec2 sample_pos = pixel + ivec2(x, y);
					if (any(lessThan(sample_pos, ivec2(0))) || any(greaterThanEqual(sample_pos, size))) {
						continue;
					}
					float sample_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), sample_pos, 0).r;
					vec4 sample_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), sample_pos, 0);
					vec3 sample_normal;
					vec2 sample_moments = texelFetch(sampler2D(temporal_moments_input, nearest_sampler), sample_pos, 0).xy;
					if (!svgf_valid_view_z(sample_view_z) || !svgf_decode_normal(sample_encoded_normal, sample_normal) ||
							!svgf_finite_vec2(sample_moments) || sample_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE < 1.0) {
						continue;
					}
					float normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), params.atrous.x);
					float depth_scale = 0.02 + 0.02 * max(center_view_z, sample_view_z);
					float depth_weight = exp(-abs(center_view_z - sample_view_z) / depth_scale);
					float weight = moment_kernel[x + 1] * moment_kernel[y + 1] * normal_weight * depth_weight;
					moment_sum += max(sample_moments, vec2(0.0)) * weight;
					moment_weight_sum += weight;
				}
			}
			if (moment_weight_sum > 1e-5) {
				moments = moment_sum / moment_weight_sum;
			}
		}
		center_variance = svgf_variance_from_moments(moments, history_length);
	}

	if (!svgf_finite_vec2(center_moments)) {
		svgf_store_atrous(pixel, center_signal.rgb, center_variance);
		return;
	}

	// A sparse 3x3 A-Trous kernel limits guide and signal fetches.
	const float kernel[2] = float[](0.44198, 0.27901);
	float temporal_confidence = smoothstep(4.0, min(16.0, params.atrous.w), history_length);
	// Relax luminance rejection while moments are young, then restore the base threshold.
	float history_fix_relaxation = mix(1.75, 1.0, temporal_confidence);
	if (specular_mode) {
		history_fix_relaxation *= mix(1.0, 1.35, motion_filter_response);
	}
	float luminance_domain_scale = specular_mode ? SPECULAR_MOMENT_LUMINANCE_SCALE : MOMENT_LUMINANCE_SCALE;
	float luminance_scale = (params.tuning.w * sqrt(max(center_variance, params.atrous.z)) + luminance_domain_scale * 1e-4) * history_fix_relaxation;
	if (specular_mode) {
		float roughness_filter_response = smoothstep(0.08, 0.50, center_roughness);
		luminance_scale *= mix(1.0, 1.6, roughness_filter_response) * mix(1.0, 2.25, motion_filter_response);
	}
	float normal_power = params.atrous.x * sqrt(float(step_width)) * mix(0.5, 1.0, temporal_confidence);
	if (specular_mode) {
		normal_power *= mix(1.0, 0.8, motion_filter_response);
	}
	float center_weight = kernel[0] * kernel[0];
	vec3 signal_sum = center_signal.rgb * center_weight;
	float variance_sum = center_variance * center_weight * center_weight;
	float weight_sum = center_weight;

	// Use the lower-magnitude one-sided depth derivative to preserve slopes without crossing discontinuities.
	float depth_left = texelFetch(sampler2D(current_view_z_input, nearest_sampler), max(pixel - ivec2(1, 0), ivec2(0)), 0).r;
	float depth_right = texelFetch(sampler2D(current_view_z_input, nearest_sampler), min(pixel + ivec2(1, 0), size - 1), 0).r;
	float depth_up = texelFetch(sampler2D(current_view_z_input, nearest_sampler), max(pixel - ivec2(0, 1), ivec2(0)), 0).r;
	float depth_down = texelFetch(sampler2D(current_view_z_input, nearest_sampler), min(pixel + ivec2(0, 1), size - 1), 0).r;
	float gradient_left = svgf_valid_view_z(depth_left) ? center_view_z - depth_left : 0.0;
	float gradient_right = svgf_valid_view_z(depth_right) ? depth_right - center_view_z : gradient_left;
	float gradient_up = svgf_valid_view_z(depth_up) ? center_view_z - depth_up : 0.0;
	float gradient_down = svgf_valid_view_z(depth_down) ? depth_down - center_view_z : gradient_up;
	vec2 depth_gradient = vec2(
			abs(gradient_left) < abs(gradient_right) ? gradient_left : gradient_right,
			abs(gradient_up) < abs(gradient_down) ? gradient_up : gradient_down);
	float plane_threshold = max(0.001, params.atrous.y * (((params.control.w & 1u) != 0u) ? 1.0 : center_view_z));
	plane_threshold *= 1.0 + 0.5 * sqrt(float(step_width));

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			if (x == 0 && y == 0) {
				continue;
			}
			ivec2 offset = ivec2(x, y) * step_width;
			ivec2 sample_pos = pixel + offset;
			if (any(lessThan(sample_pos, ivec2(0))) || any(greaterThanEqual(sample_pos, size))) {
				continue;
			}
			float sample_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), sample_pos, 0).r;
			vec4 sample_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), sample_pos, 0);
			vec3 sample_normal;
			if (!svgf_valid_view_z(sample_view_z) || !svgf_decode_normal(sample_encoded_normal, sample_normal)) {
				continue;
			}
			float normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), normal_power);
			if (specular_mode) {
				float sample_packed_roughness_source = texelFetch(sampler2D(current_roughness_input, nearest_sampler), sample_pos, 0).a;
				float sample_roughness = svgf_unpack_specular_roughness_source(sample_packed_roughness_source).x;
				normal_weight *= exp2(-abs(sample_roughness - center_roughness) * 32.0);
			}
			if (normal_weight <= 1e-4) {
				continue;
			}
			float expected_depth_delta = dot(depth_gradient, vec2(offset));
			if (abs((sample_view_z - center_view_z) - expected_depth_delta) > plane_threshold) {
				continue;
			}
			vec4 sample_signal = svgf_sanitize_signal(texelFetch(sampler2D(filtered_signal_input, nearest_sampler), sample_pos, 0));
			float sample_variance = sample_signal.a;
			if (first_iteration) {
				vec2 sample_moments = texelFetch(sampler2D(temporal_moments_input, nearest_sampler), sample_pos, 0).xy;
				if (!svgf_finite_vec2(sample_moments)) {
					continue;
				}
				float sample_history_length = sample_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
				sample_variance = svgf_variance_from_moments(sample_moments, sample_history_length);
				if (sample_history_length < 4.0) {
					sample_variance = max(sample_variance, center_variance);
				}
			}

			float kernel_weight = kernel[abs(x)] * kernel[abs(y)];
			float sample_luminance = svgf_filter_luminance(sample_signal.rgb, specular_mode);
			float luminance_delta = abs(center_luminance - sample_luminance) / luminance_scale;
			float luminance_weight = exp(-luminance_delta);
			float weight = kernel_weight * normal_weight * luminance_weight;
			signal_sum += sample_signal.rgb * weight;
			variance_sum += sample_variance * weight * weight;
			weight_sum += weight;
		}
	}

	vec3 output_signal = center_signal.rgb;
	float output_variance = center_variance;
	if (weight_sum > 1e-5) {
		output_signal = signal_sum / weight_sum;
		output_variance = variance_sum / (weight_sum * weight_sum);
	}
	if (!svgf_finite_float(output_variance)) {
		output_variance = center_variance;
	}
	svgf_store_atrous(pixel, output_signal, output_variance);
}

#endif

void main() {
#ifdef MODE_TEMPORAL
	svgf_temporal_main();
#elif defined(MODE_HISTORY_FIX)
	svgf_history_fix_main();
#elif defined(MODE_ATROUS)
	svgf_atrous_main();
#endif
}
