#[compute]

#version 450

#VERSION_DEFINES

#if defined(SVGF_SPECULAR) || defined(MODE_SPECULAR_SPATIAL)
#include "hddagi_specular_surface_inc.glsl"
#endif

#ifdef ATROUS_STEP_SIZE
#if ATROUS_STEP_SIZE <= 2
#define ATROUS_SHARED
#endif
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
#else
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#endif

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
#ifdef SVGF_SPECULAR
layout(rg16f, set = 0, binding = 10) uniform restrict writeonly image2D current_moments_output;
#else
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image2D current_moments_output;
#endif
layout(rgba8, set = 0, binding = 11) uniform restrict writeonly image2D current_normal_roughness_output;

const int TEMPORAL_NEIGHBORHOOD_RADIUS = 2;
#ifdef SVGF_SPECULAR
const int TEMPORAL_NEIGHBORHOOD_SHARED_SIZE = 12;
const int TEMPORAL_NEIGHBORHOOD_SHARED_COUNT = TEMPORAL_NEIGHBORHOOD_SHARED_SIZE * TEMPORAL_NEIGHBORHOOD_SHARED_SIZE;
shared vec4 temporal_neighborhood_ycocg[TEMPORAL_NEIGHBORHOOD_SHARED_SIZE][TEMPORAL_NEIGHBORHOOD_SHARED_SIZE];
#endif

#endif

#ifdef MODE_HISTORY_FIX

layout(set = 0, binding = 0) uniform texture2D history_fix_signal_input;
layout(set = 0, binding = 1) uniform texture2D history_fix_moments_input;
layout(set = 0, binding = 2) uniform texture2D history_fix_normal_input;
layout(set = 0, binding = 3) uniform texture2D history_fix_view_z_input;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D history_fix_signal_output;
layout(rg16f, set = 0, binding = 6) uniform restrict writeonly image2D history_fix_moments_output;

#endif

#ifdef MODE_ATROUS

layout(set = 0, binding = 0) uniform texture2D filtered_signal_input;
layout(set = 0, binding = 1) uniform texture2D temporal_moments_input;
layout(set = 0, binding = 2) uniform texture2D current_normal_roughness_input;
layout(set = 0, binding = 3) uniform texture2D current_view_z_input;
layout(set = 0, binding = 4) uniform sampler nearest_sampler;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D filtered_signal_output;

#endif

const vec3 LUMINANCE_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);
const float FP16_MAX = 65504.0;
const float HISTORY_LENGTH_STORAGE_SCALE = 255.0;
const float MOMENT_LUMINANCE_SCALE = 64.0;
const float DIFFUSE_SIGNAL_SCALE = 1.0 / 512.0;
const float MOMENT_LUMINANCE_MAX = 255.0;

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

vec3 svgf_rgb_to_ycocg(vec3 value) {
	return vec3(
			dot(value, vec3(0.25, 0.5, 0.25)),
			dot(value, vec3(0.5, 0.0, -0.5)),
			dot(value, vec3(-0.25, 0.5, -0.25)));
}

vec3 svgf_ycocg_to_rgb(vec3 value) {
	return vec3(value.x + value.y - value.z, value.x + value.z, value.x - value.y - value.z);
}

float svgf_filter_luminance(vec3 value) {
	float luminance = svgf_luminance(value);
	return luminance * MOMENT_LUMINANCE_SCALE;
}

vec4 svgf_sanitize_signal(vec4 value) {
	if (!svgf_finite_vec4(value)) {
		return vec4(0.0);
	}
	return clamp(value, vec4(0.0), vec4(FP16_MAX));
}

vec3 svgf_decode_normal(vec4 encoded) {
	return normalize(encoded.xyz * 2.0 - 1.0);
}

bool svgf_valid_view_z(float view_z) {
	return view_z > 0.0 && view_z <= params.tuning.x;
}

#if defined(MODE_TEMPORAL) || defined(MODE_HISTORY_FIX)

bool svgf_reconstruct_view(mat4 inv_projection, vec2 uv, float view_z, bool orthographic, out vec3 r_view_position) {
	vec2 ndc = uv * 2.0 - 1.0;
	if (!orthographic) {
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

#endif

#ifdef MODE_TEMPORAL

bool svgf_history_tap_valid(ivec2 tap_pos, ivec2 size, bool current_dynamic, vec3 current_normal, vec3 current_normal_previous_view, vec3 expected_previous_position, out float r_history_length, out vec2 r_moments, out float r_normal_agreement) {
	r_history_length = 0.0;
	r_moments = vec2(0.0);
	if (!svgf_finite_vec3(expected_previous_position) || expected_previous_position.z >= -1e-5) {
		return false;
	}
	if (any(lessThan(tap_pos, ivec2(0))) || any(greaterThanEqual(tap_pos, size))) {
		return false;
	}
	float packed_view_z = texelFetch(sampler2D(previous_view_z_input, nearest_sampler), tap_pos, 0).r;
	float previous_view_z = abs(packed_view_z);
	vec4 previous_encoded_normal = texelFetch(sampler2D(previous_normal_roughness_input, nearest_sampler), tap_pos, 0);
	vec3 previous_normal = svgf_decode_normal(previous_encoded_normal);
	r_normal_agreement = dot(current_normal, previous_normal);
	if (!svgf_valid_view_z(previous_view_z)) {
		return false;
	}
	float expected_previous_view_z = -expected_previous_position.z;
	float relative_depth_tolerance = current_dynamic ? max(params.tuning.z * 6.0, 0.08) : max(params.tuning.z * 6.0, 0.02);
	float depth_tolerance = 0.05 + relative_depth_tolerance * max(expected_previous_view_z, previous_view_z);
	float normal_threshold = current_dynamic ? min(params.tuning.y, 0.7) : params.tuning.y;
	r_history_length = previous_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	if (!svgf_finite_float(r_history_length) || r_history_length < 1.0 || abs(previous_view_z - expected_previous_view_z) > depth_tolerance || r_normal_agreement < normal_threshold) {
		return false;
	}
	vec4 previous_moments = texelFetch(sampler2D(previous_moments_input, nearest_sampler), tap_pos, 0);
	r_moments = previous_moments.xy;
#ifdef SVGF_SPECULAR
	if (current_dynamic && packed_view_z < 0.0) {
#else
	if (current_dynamic) {
#endif
		return true;
	}
	vec2 tap_uv = (vec2(tap_pos) + 0.5) / vec2(size);
#ifdef SVGF_SPECULAR
	if ((params.control.x & 4u) == 0u) {
		ivec2 screen_size = textureSize(sampler2D(motion_input, nearest_sampler), 0);
		tap_uv = (vec2(specular_surface_pixel(tap_pos, size, screen_size, packed_view_z)) + 0.5) / vec2(screen_size);
	}
#endif
	vec3 previous_position;
	if (!svgf_reconstruct_view(frame_data.previous_inv_projection, tap_uv, previous_view_z, (params.control.w & 2u) != 0u, previous_position)) {
		return false;
	}
	float plane_tolerance = 0.01 + params.tuning.z * expected_previous_view_z;
	return abs(dot(previous_position - expected_previous_position, current_normal_previous_view)) <= plane_tolerance;
}

bool svgf_temporal_neighborhood(vec3 center_signal, out vec3 r_center, out vec3 r_extent) {
	vec3 center_ycocg = svgf_rgb_to_ycocg(center_signal);
	vec3 sample_sum = center_ycocg;
	vec3 sample_square_sum = center_ycocg * center_ycocg;
	float sample_count = 1.0;
#ifdef SVGF_SPECULAR
	ivec2 shared_center = ivec2(gl_LocalInvocationID.xy) + ivec2(TEMPORAL_NEIGHBORHOOD_RADIUS);
#else
	ivec2 size = textureSize(sampler2D(current_signal_input, nearest_sampler), 0);
#endif
	for (int y = -TEMPORAL_NEIGHBORHOOD_RADIUS; y <= TEMPORAL_NEIGHBORHOOD_RADIUS; y++) {
		for (int x = -TEMPORAL_NEIGHBORHOOD_RADIUS; x <= TEMPORAL_NEIGHBORHOOD_RADIUS; x++) {
			if ((x == 0 && y == 0) || (abs(x) == TEMPORAL_NEIGHBORHOOD_RADIUS && abs(y) == TEMPORAL_NEIGHBORHOOD_RADIUS)) {
				continue;
			}
#ifdef SVGF_SPECULAR
			vec4 packed_sample = temporal_neighborhood_ycocg[shared_center.y + y][shared_center.x + x];
#else
			ivec2 tap = ivec2(gl_GlobalInvocationID.xy) + ivec2(x, y);
			if (any(lessThan(tap, ivec2(0))) || any(greaterThanEqual(tap, size))) {
				continue;
			}
			float view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), tap, 0).r;
			vec4 signal = texelFetch(sampler2D(current_signal_input, nearest_sampler), tap, 0);
			if (!svgf_valid_view_z(abs(view_z)) || !svgf_finite_vec4(signal)) {
				continue;
			}
			vec4 packed_sample = vec4(svgf_rgb_to_ycocg(signal.rgb * DIFFUSE_SIGNAL_SCALE), 1.0);
#endif
			sample_sum += packed_sample.xyz;
			sample_square_sum += packed_sample.xyz * packed_sample.xyz;
			sample_count += packed_sample.w;
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

#ifndef SVGF_SPECULAR
void svgf_temporal_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(current_signal_output);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	vec4 current_signal = svgf_sanitize_signal(texelFetch(sampler2D(current_signal_input, nearest_sampler), pixel, 0));
	current_signal.rgb *= DIFFUSE_SIGNAL_SCALE;

	vec4 current_encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
	float current_view_z = texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r;
	bool current_dynamic = current_view_z < 0.0;
	current_view_z = abs(current_view_z);
	vec3 current_normal = svgf_decode_normal(current_encoded_normal);
	bool current_surface_valid = svgf_valid_view_z(current_view_z);
	vec2 motion = texelFetch(sampler2D(motion_input, nearest_sampler), pixel, 0).xy;
	if (!svgf_finite_vec2(motion)) {
		motion = vec2(0.0);
	}

	vec3 accumulated_history = vec3(0.0);
	vec2 accumulated_moments = vec2(0.0);
	float accumulated_history_length = 0.0;
	float accumulated_normal_agreement = 0.0;
	float history_weight_sum = 0.0;
	if ((params.control.x & 1u) != 0u && current_surface_valid) {
		vec2 current_uv = (vec2(pixel) + 0.5) / vec2(size);
		vec2 previous_uv = current_uv + motion + (frame_data.taa_jitter.zw - frame_data.taa_jitter.xy) * 0.5;
		vec3 current_view_position;
		if (svgf_finite_vec2(previous_uv) && svgf_reconstruct_view(frame_data.current_inv_projection, current_uv, current_view_z, (params.control.w & 1u) != 0u, current_view_position)) {
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
							float tap_normal_agreement;
							if (!(tap_weight > 0.0) || !svgf_history_tap_valid(tap_pos, size, current_dynamic, current_normal, current_normal_previous_view, expected_previous_position, tap_history_length, tap_moments, tap_normal_agreement)) {
								continue;
							}
							vec4 history_signal = texelFetch(sampler2D(previous_signal_input, nearest_sampler), tap_pos, 0);
							if (!svgf_finite_vec4(history_signal)) {
								continue;
							}

							accumulated_history += max(history_signal.rgb, vec3(0.0)) * tap_weight;
							accumulated_moments += tap_moments * tap_weight;
							accumulated_history_length += tap_history_length * tap_weight;
							accumulated_normal_agreement += tap_normal_agreement * tap_weight;
							history_weight_sum += tap_weight;
						}
					}
				}
			}
		}
	}

	float current_moment_luminance = min(svgf_filter_luminance(current_signal.rgb), MOMENT_LUMINANCE_MAX);
	vec4 output_signal = current_signal;
	vec2 output_moments = vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance);
	float history_length = current_surface_valid ? 1.0 : 0.0;
	if (current_surface_valid && history_weight_sum > 1e-5) {
		vec3 history_signal = accumulated_history / history_weight_sum;
		vec2 previous_moments = accumulated_moments / history_weight_sum;
		float previous_history_length = accumulated_history_length / history_weight_sum;
		float reprojection_quality = sqrt(clamp(history_weight_sum, 0.0, 1.0));
		float base_alpha = 1.0 / params.atrous.w;

		float probe_lighting_response = smoothstep(0.1, 0.6, current_encoded_normal.a);
		float original_history_luminance = svgf_luminance(history_signal);
		bool rotating = current_dynamic && accumulated_normal_agreement < 0.99995 * history_weight_sum;
		float maximum_lighting_alpha = rotating ? 0.25 : 0.125;

		// Interpolate update rates so lighting changes can shorten a mature history.
		float maximum_history_length = 1.0 / mix(base_alpha, max(base_alpha, maximum_lighting_alpha), probe_lighting_response);

		float lighting_response = 0.0;
		vec3 neighborhood_center;
		vec3 neighborhood_extent;
		if ((probe_lighting_response > 0.0 || (rotating && abs(svgf_luminance(current_signal.rgb) - original_history_luminance) > 0.15 * original_history_luminance)) && svgf_temporal_neighborhood(current_signal.rgb, neighborhood_center, neighborhood_extent)) {
			float neighborhood_luminance = max(svgf_luminance(svgf_ycocg_to_rgb(neighborhood_center)), 0.0);
			float spatial_noise = dot(neighborhood_extent, vec3(1.0, 0.1404, 0.4304));
			float temporal_noise = sqrt(max(previous_moments.y - previous_moments.x * previous_moments.x, 0.0)) / MOMENT_LUMINANCE_SCALE;

			// Compare filtered means using the variance of the temporal mean.
			temporal_noise /= sqrt(max(previous_history_length, 1.0));

			float luminance_threshold = max(3.0 * max(spatial_noise, temporal_noise), max(neighborhood_luminance, original_history_luminance) * 0.15 + 1e-6);
			float disagreement = abs(neighborhood_luminance - original_history_luminance) / luminance_threshold;
			lighting_response = maximum_lighting_alpha * smoothstep(1.0, 2.0, disagreement);
			if (!current_dynamic) {
				lighting_response *= probe_lighting_response;
			}

			maximum_history_length = min(maximum_history_length, 1.0 / max(lighting_response, 1.0 / maximum_history_length));
		}
		float advanced_history_length = previous_history_length + 1.0;
		history_length = max(1.0, min(advanced_history_length, maximum_history_length) * reprojection_quality);
		float signal_alpha = max(1.0 / max(history_length, 1.0), lighting_response);
		output_signal.rgb = mix(history_signal, current_signal.rgb, signal_alpha);
		output_moments = mix(max(previous_moments, vec2(0.0)), vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance), signal_alpha);
	}

	output_signal = svgf_sanitize_signal(output_signal);
	if (!svgf_finite_vec2(output_moments)) {
		output_moments = vec2(current_moment_luminance, current_moment_luminance * current_moment_luminance);
	}
	output_moments = clamp(output_moments, vec2(0.0), vec2(FP16_MAX));
	output_signal.a = current_signal.a;
	imageStore(current_signal_output, pixel, output_signal);
	imageStore(current_moments_output, pixel, vec4(output_moments, 0.0, 0.0));
	current_encoded_normal.a = history_length / HISTORY_LENGTH_STORAGE_SCALE;
	imageStore(current_normal_roughness_output, pixel, current_encoded_normal);
}

#endif // !SVGF_SPECULAR
#endif // MODE_TEMPORAL

#ifdef MODE_HISTORY_FIX

ivec2 svgf_mirror_pixel(ivec2 pixel, ivec2 size) {
	ivec2 folded = max(pixel, -1 - pixel);
	if (any(greaterThanEqual(folded, size * 2))) {
		folded %= size * 2;
	}
	return min(folded, size * 2 - 1 - folded);
}

void svgf_fix_short_history(ivec2 pixel, ivec2 size, float history_length, vec3 center_normal_world, inout vec3 signal, inout vec2 moments) {
	float signed_view_z = texelFetch(sampler2D(history_fix_view_z_input, nearest_sampler), pixel, 0).r;
	float center_view_z = abs(signed_view_z);
	bool center_dynamic = signed_view_z < 0.0;
	bool orthographic = (params.control.w & 1u) != 0u;
	vec2 center_uv = (vec2(pixel) + 0.5) / vec2(size);
	vec3 center_view_position;
	if (!svgf_reconstruct_view(frame_data.current_inv_projection, center_uv, center_view_z, orthographic, center_view_position)) {
		return;
	}
	vec3 center_normal_view = normalize(transpose(mat3(frame_data.current_view_to_world)) * center_normal_world);
	int stride = max(int(round(float(params.control.y) / (1.0 + history_length))), 1);
	float plane_threshold = max(0.001, params.atrous.y * (orthographic ? 1.0 : center_view_z));
	float weight_sum = 1.0;
	for (int y = -2; y <= 2; y++) {
		for (int x = -2; x <= 2; x++) {
			if (x == 0 && y == 0) {
				continue;
			}
			ivec2 tap = svgf_mirror_pixel(pixel + ivec2(x, y) * stride, size);
			float sample_z = texelFetch(sampler2D(history_fix_view_z_input, nearest_sampler), tap, 0).r;
			if ((sample_z < 0.0) != center_dynamic || !svgf_valid_view_z(abs(sample_z))) {
				continue;
			}
			if (!center_dynamic) {
				vec2 uv = (vec2(tap) + 0.5) / vec2(size);
				vec3 view_position;
				if (!svgf_reconstruct_view(frame_data.current_inv_projection, uv, sample_z, orthographic, view_position) || abs(dot(view_position - center_view_position, center_normal_view)) >= plane_threshold) {
					continue;
				}
			}
			vec3 encoded = texelFetch(sampler2D(history_fix_normal_input, nearest_sampler), tap, 0).xyz;
			vec3 normal = normalize(encoded * 2.0 - 1.0);
			float weight = pow(max(dot(center_normal_world, normal), 0.0), params.atrous.x);
			if (weight > 1e-4) {
				signal += texelFetch(sampler2D(history_fix_signal_input, nearest_sampler), tap, 0).rgb * weight;
				moments += texelFetch(sampler2D(history_fix_moments_input, nearest_sampler), tap, 0).xy * weight;
				weight_sum += weight;
			}
		}
	}
	signal /= weight_sum;
	moments /= weight_sum;
}

void svgf_history_fix_main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(history_fix_signal_output);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	vec4 signal = texelFetch(sampler2D(history_fix_signal_input, nearest_sampler), pixel, 0);
	vec2 moments = texelFetch(sampler2D(history_fix_moments_input, nearest_sampler), pixel, 0).xy;
	vec4 center_encoded_normal = texelFetch(sampler2D(history_fix_normal_input, nearest_sampler), pixel, 0);
	float history_length = center_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	if (history_length >= 1.0 && history_length <= float(params.control.z)) {
		svgf_fix_short_history(pixel, size, history_length, svgf_decode_normal(center_encoded_normal), signal.rgb, moments);
	}
	imageStore(history_fix_signal_output, pixel, signal);
	imageStore(history_fix_moments_output, pixel, vec4(moments, 0.0, 0.0));
}

#endif

#if defined(MODE_ATROUS) || defined(MODE_SPECULAR_SPATIAL)
float svgf_variance_from_moments(vec2 moments, float history_length) {
	if (!svgf_finite_vec2(moments)) {
		return 0.0;
	}
	moments = max(moments, vec2(0.0));
	float variance = max(moments.y - moments.x * moments.x, 0.0);
	variance *= max(1.0, 4.0 / (history_length + 1.0));
	return svgf_finite_float(variance) ? clamp(variance, 0.0, FP16_MAX) : 0.0;
}
#endif

#ifdef MODE_ATROUS

#ifdef ATROUS_SHARED
// Tiled remainder-based execution reordering and shared neighborhoods, adapted from:
// Pikulsky, "Optimizing SVGF for Modern GPU Architectures", JCGT 15(1), 2026.
// https://jcgt.org/published/0015/01/02/
const int ATROUS_SHARED_WIDTH = 18;
const int ATROUS_SHARED_HEIGHT = 10;
shared vec4 atrous_signal[ATROUS_SHARED_HEIGHT][ATROUS_SHARED_WIDTH];
shared float atrous_luminance[ATROUS_SHARED_HEIGHT][ATROUS_SHARED_WIDTH];
shared vec4 atrous_normal[ATROUS_SHARED_HEIGHT][ATROUS_SHARED_WIDTH];
shared float atrous_view_z[ATROUS_SHARED_HEIGHT][ATROUS_SHARED_WIDTH];
#if ATROUS_STEP_SIZE == 1
shared vec2 atrous_moments[ATROUS_SHARED_HEIGHT][ATROUS_SHARED_WIDTH];
shared float atrous_variance[ATROUS_SHARED_HEIGHT][ATROUS_SHARED_WIDTH];
#endif

ivec2 svgf_atrous_group_origin() {
	uvec2 phase = gl_WorkGroupID.xy % uint(ATROUS_STEP_SIZE);
	return ivec2((gl_WorkGroupID.xy / uint(ATROUS_STEP_SIZE)) * gl_WorkGroupSize.xy) * ATROUS_STEP_SIZE + ivec2(phase);
}

void svgf_atrous_load_tile(ivec2 size) {
	ivec2 origin = svgf_atrous_group_origin() - ATROUS_STEP_SIZE;
	for (uint index = gl_LocalInvocationIndex; index < uint(ATROUS_SHARED_WIDTH * ATROUS_SHARED_HEIGHT); index += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
		ivec2 local_pos = ivec2(int(index) % ATROUS_SHARED_WIDTH, int(index) / ATROUS_SHARED_WIDTH);
		ivec2 pixel = clamp(origin + local_pos * ATROUS_STEP_SIZE, ivec2(0), size - 1);
		vec4 signal = texelFetch(sampler2D(filtered_signal_input, nearest_sampler), pixel, 0);
		atrous_signal[local_pos.y][local_pos.x] = signal;
		atrous_luminance[local_pos.y][local_pos.x] = svgf_filter_luminance(signal.rgb);
		vec4 encoded_normal = texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
		atrous_normal[local_pos.y][local_pos.x] = vec4(svgf_decode_normal(encoded_normal), encoded_normal.a);
		atrous_view_z[local_pos.y][local_pos.x] = abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r);
#if ATROUS_STEP_SIZE == 1
		vec2 moments = texelFetch(sampler2D(temporal_moments_input, nearest_sampler), pixel, 0).xy;
		atrous_moments[local_pos.y][local_pos.x] = moments;
		atrous_variance[local_pos.y][local_pos.x] = svgf_variance_from_moments(moments, encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE);
#endif
	}
	// Partial workgroups must populate the halo and reach this barrier before exiting.
	barrier();
}
#endif

vec4 svgf_atrous_load_signal(ivec2 pixel, ivec2 offset) {
#ifdef ATROUS_SHARED
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + 1 + offset;
	return atrous_signal[local_pos.y][local_pos.x];
#else
	return texelFetch(sampler2D(filtered_signal_input, nearest_sampler), pixel, 0);
#endif
}

vec4 svgf_atrous_load_normal(ivec2 pixel, ivec2 offset) {
#ifdef ATROUS_SHARED
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + 1 + offset;
	return atrous_normal[local_pos.y][local_pos.x];
#else
	return texelFetch(sampler2D(current_normal_roughness_input, nearest_sampler), pixel, 0);
#endif
}

float svgf_atrous_load_view_z(ivec2 pixel, ivec2 offset) {
#ifdef ATROUS_SHARED
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + 1 + offset;
	return atrous_view_z[local_pos.y][local_pos.x];
#else
	return abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), pixel, 0).r);
#endif
}

float svgf_atrous_gradient_depth(ivec2 pixel, ivec2 offset, ivec2 size) {
#if ATROUS_STEP_SIZE == 1
	return svgf_atrous_load_view_z(pixel, offset);
#else
	return abs(texelFetch(sampler2D(current_view_z_input, nearest_sampler), clamp(pixel + offset, ivec2(0), size - 1), 0).r);
#endif
}

vec2 svgf_atrous_load_moments(ivec2 pixel, ivec2 offset) {
#if (defined(ATROUS_SHARED)) && (ATROUS_STEP_SIZE == 1)
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + 1 + offset;
	return atrous_moments[local_pos.y][local_pos.x];
#else
	return texelFetch(sampler2D(temporal_moments_input, nearest_sampler), pixel, 0).xy;
#endif
}

float svgf_atrous_load_luminance(vec3 signal, ivec2 offset) {
#ifdef ATROUS_SHARED
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + 1 + offset;
	return atrous_luminance[local_pos.y][local_pos.x];
#else
	return svgf_filter_luminance(signal);
#endif
}

float svgf_atrous_load_variance(vec2 moments, float history_length, ivec2 offset) {
#if (defined(ATROUS_SHARED)) && (ATROUS_STEP_SIZE == 1)
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + 1 + offset;
	return atrous_variance[local_pos.y][local_pos.x];
#else
	return svgf_variance_from_moments(moments, history_length);
#endif
}

vec3 svgf_atrous_decode_normal(vec4 encoded) {
#ifdef ATROUS_SHARED
	return encoded.xyz;
#else
	return svgf_decode_normal(encoded);
#endif
}

void svgf_store_atrous(ivec2 pixel, vec3 signal, float variance) {
	float packed_value = clamp(variance, 0.0, FP16_MAX);
	if ((params.control.z & 2u) != 0u) {
		packed_value = FP16_MAX;
	}
	imageStore(filtered_signal_output, pixel, svgf_sanitize_signal(vec4(signal, packed_value)));
}

void svgf_atrous_main() {
#ifdef ATROUS_SHARED
	ivec2 pixel = svgf_atrous_group_origin() + ivec2(gl_LocalInvocationID.xy) * ATROUS_STEP_SIZE;
#else
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
#endif
	ivec2 size = imageSize(filtered_signal_output);
#ifdef ATROUS_SHARED
	svgf_atrous_load_tile(size);
#endif
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}

	const bool first_iteration = ATROUS_STEP_SIZE == 1;
	const int step_width = ATROUS_STEP_SIZE;
	vec4 center_signal = svgf_atrous_load_signal(pixel, ivec2(0));
	float center_variance = first_iteration ? 0.0 : center_signal.a;
	float center_view_z = svgf_atrous_load_view_z(pixel, ivec2(0));
	vec4 center_encoded_normal = svgf_atrous_load_normal(pixel, ivec2(0));

	vec3 center_normal = svgf_atrous_decode_normal(center_encoded_normal);
	if (!svgf_valid_view_z(center_view_z)) {
		svgf_store_atrous(pixel, center_signal.rgb, center_variance);
		return;
	}
	float history_length = center_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
	float center_luminance = svgf_atrous_load_luminance(center_signal.rgb, ivec2(0));
	if (first_iteration) {
		vec2 moments = svgf_atrous_load_moments(pixel, ivec2(0));
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
					float sample_view_z = svgf_atrous_load_view_z(sample_pos, ivec2(x, y));
					vec4 sample_encoded_normal = svgf_atrous_load_normal(sample_pos, ivec2(x, y));
					vec3 sample_normal = svgf_atrous_decode_normal(sample_encoded_normal);
					vec2 sample_moments = svgf_atrous_load_moments(sample_pos, ivec2(x, y));
					if (!svgf_valid_view_z(sample_view_z) || sample_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE < 1.0) {
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

	const float kernel[2] = float[](0.44198, 0.27901);
	float temporal_confidence = smoothstep(4.0, 16.0, history_length);
	float history_fix_relaxation = mix(1.75, 1.0, temporal_confidence);

	float luminance_domain_scale = MOMENT_LUMINANCE_SCALE;
	float luminance_scale = (params.tuning.w * sqrt(max(center_variance, params.atrous.z)) + luminance_domain_scale * 1e-4) * history_fix_relaxation;

#if ATROUS_STEP_SIZE >= 8
	// Use the measured variance to keep coarse taps from blurring dark contact edges.
	float relative_luminance_scale = params.tuning.w * sqrt(max(center_variance, 0.0)) + 0.35 * max(center_luminance, 1e-4);
	luminance_scale = mix(luminance_scale, min(luminance_scale, relative_luminance_scale), temporal_confidence);
#endif

	float normal_power = params.atrous.x * sqrt(float(step_width)) * mix(0.5, 1.0, temporal_confidence);

	float center_weight = kernel[0] * kernel[0];
	vec3 signal_sum = center_signal.rgb * center_weight;
	float variance_sum = center_variance * center_weight * center_weight;
	float weight_sum = center_weight;

	// Use the lower-magnitude one-sided depth derivative to preserve slopes without crossing discontinuities.
	float depth_left = svgf_atrous_gradient_depth(pixel, ivec2(-1, 0), size);
	float depth_right = svgf_atrous_gradient_depth(pixel, ivec2(1, 0), size);
	float depth_up = svgf_atrous_gradient_depth(pixel, ivec2(0, -1), size);
	float depth_down = svgf_atrous_gradient_depth(pixel, ivec2(0, 1), size);
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
			float sample_view_z = svgf_atrous_load_view_z(sample_pos, ivec2(x, y));
			vec4 sample_encoded_normal = svgf_atrous_load_normal(sample_pos, ivec2(x, y));
			vec3 sample_normal = svgf_atrous_decode_normal(sample_encoded_normal);
			if (!svgf_valid_view_z(sample_view_z)) {
				continue;
			}
			float normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), normal_power);

			if (normal_weight <= 1e-4) {
				continue;
			}
			float expected_depth_delta = dot(depth_gradient, vec2(offset));
			if (abs((sample_view_z - center_view_z) - expected_depth_delta) > plane_threshold) {
				continue;
			}
			vec4 sample_signal = svgf_atrous_load_signal(sample_pos, ivec2(x, y));
			float sample_variance = sample_signal.a;
			if (first_iteration) {
				vec2 sample_moments = svgf_atrous_load_moments(sample_pos, ivec2(x, y));
				float sample_history_length = sample_encoded_normal.a * HISTORY_LENGTH_STORAGE_SCALE;
				sample_variance = svgf_atrous_load_variance(sample_moments, sample_history_length, ivec2(x, y));
				if (sample_history_length < 4.0) {
					sample_variance = max(sample_variance, center_variance);
				}
			}

			float kernel_weight = kernel[abs(x)] * kernel[abs(y)];
			float sample_luminance = svgf_atrous_load_luminance(sample_signal.rgb, ivec2(x, y));
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

#if defined(SVGF_SPECULAR) || defined(MODE_SPECULAR_RECONSTRUCT) || defined(MODE_SPECULAR_SPATIAL)
#include "hddagi_screen_probe_svgf_specular_inc.glsl"
#endif

void main() {
#ifdef SVGF_SPECULAR
	svgf_specular_temporal_main();
#elif defined(MODE_SPECULAR_RECONSTRUCT)
	svgf_specular_reconstruct_main();
#elif defined(MODE_SPECULAR_SPATIAL)
	svgf_specular_spatial_main();
#elif defined(MODE_TEMPORAL)
	svgf_temporal_main();
#elif defined(MODE_HISTORY_FIX)
	svgf_history_fix_main();
#elif defined(MODE_ATROUS)
	svgf_atrous_main();
#endif
}
