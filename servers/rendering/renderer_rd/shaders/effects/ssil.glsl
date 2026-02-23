///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2016-09-07: filip.strugar@intel.com: first commit
// 2020-12-05: clayjohn: convert to Vulkan and Godot
// 2021-05-27: clayjohn: convert SSAO to SSIL
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

#define PI 3.14159265359

const int num_samples[5] = { 4, 8, 16, 32, 48 };

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16, set = 0, binding = 0) uniform restrict writeonly image2D dest_image;

// Buffers
layout(set = 1, binding = 0) uniform sampler2D depth_buffer;
layout(rgba8, set = 1, binding = 1) uniform restrict readonly image2D normal_buffer;

layout(set = 2, binding = 0) uniform sampler2D last_frame;
layout(set = 2, binding = 1) uniform Matrices {
	mat4 last_frame_reproj;
}
matrices;

// Push constant
layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int quality;
	int pad;

	float z_near;
	float z_far;
	float radius;
	float thickness;

	float intensity;
	float normal_rejection;
	bool backface_rejection;
	bool is_orthogonal;

	vec2 NDC_to_view_mul;
	ivec2 full_screen_size;
}
params;

// Projection conversions
vec3 viewspace_to_screenspace(vec3 p_vpos) {
	vec2 half_ndc;
	vec2 tex_coord;

	if (params.is_orthogonal) {
		half_ndc = p_vpos.xy / (params.NDC_to_view_mul);
		tex_coord = vec2(0.5) - half_ndc;
	} else {
		half_ndc = p_vpos.xy / (params.NDC_to_view_mul * p_vpos.z);
		tex_coord = vec2(0.5) - half_ndc;
	}

	return (vec3(tex_coord * vec2(params.screen_size), -p_vpos.z));
}

vec3 clipspace_to_viewspace(vec2 p_tex_coord, float p_linear_depth) {
	vec2 half_ndc_position = vec2(0.5) - p_tex_coord;
	vec3 view_space_position;

	//-p_linear_depth since cam points at -z
	if (params.is_orthogonal) {
		view_space_position = vec3(half_ndc_position * params.NDC_to_view_mul, -p_linear_depth);
	} else {
		view_space_position = vec3(half_ndc_position * params.NDC_to_view_mul * -p_linear_depth, -p_linear_depth);
	}

	return (view_space_position);
}

// Quaternion utils
vec4 get_quaternion(vec3 p_to) {
	//vec3 from = vec3(0.0, 0.0,-1.0);

	vec3 xyz = vec3(p_to.y, -p_to.x, 0.0); // cross(from, p_to);
	float s = -p_to.z; // dot(from, p_to);

	float u = inversesqrt(max(0.0, s * 0.5 + 0.5)); // rcp(cosine half-angle formula)

	s = 1.0 / u;
	xyz *= u * 0.5;

	return vec4(xyz, s);
}

// transform p_v.xy0 by unit quaternion q.xy0s
vec3 transform_vz0qz0(vec2 p_v, vec4 p_q) {
	float o = p_q.x * p_v.y;
	float c = p_q.y * p_v.x;

	vec3 b = vec3(o - c,
			-o + c,
			o - c);

	return vec3(p_v, 0.0) + 2.0 * (b * p_q.yxw);
}

// Helper functions
vec3 load_normal(ivec2 p_pos) {
	vec3 encoded_normal = normalize(imageLoad(normal_buffer, p_pos).xyz * 2.0 - 1.0);
	encoded_normal.z = -encoded_normal.z;
	return encoded_normal;
}

// noise
float ign(ivec2 p_pixel, uint p_n) {
	float offset = float(p_n);

	float x = float(p_pixel.x) + 5.588238f * offset;
	float y = float(p_pixel.y) + 5.588238f * offset;

	float rnd01 = mod(52.9829189 * mod(0.06711056 * x + 0.00583715 * y, 1.0), 1.0);

	return rnd01;
}

vec2 ign_01x4(ivec2 p_pixel, uint p_n) {
	return vec2(ign(p_pixel, p_n), ign(p_pixel, p_n + 1u));
}

// rnd01.x/rnd01.xy -> used to sample a slice direction (exact importance sampling needs 2 rnd numbers)
// rnd01.z -> used to jitter sample positions along ray marching direction
// rnd01.w -> used to jitter sample positions radially around slice normal
vec4 rnd_01x4(ivec2 p_pixel, uint p_n) {
	vec4 rnd01 = vec4(0.0);

	rnd01.x = ign(p_pixel, p_n);
	rnd01.zw = ign_01x4(p_pixel, p_n + 1u);

	return rnd01;
}

vec4 ssilvb(vec2 p_pos, const int p_quality, float p_linear_depth) {
	ivec2 uvi = ivec2(p_pos * vec2(params.screen_size));
	ivec2 full_res_uvi = ivec2(p_pos * vec2(params.full_screen_size));

	uint count = uint(num_samples[p_quality]);

	float near_radius = params.radius * 20.0;
	float far_radius = params.radius * 40.0;
	float radius_distance_factor = clamp(p_linear_depth / (params.z_far * 0.1), 0.0, 1.0);
	float adaptive_radius = mix(near_radius, far_radius, radius_distance_factor);

	const float s = pow(adaptive_radius, 1.0 / float(count));
	uint OxFFFFFFFFu = 0xFFFFFFFFu;

	vec3 vs_normal = load_normal(full_res_uvi);

	vec3 vs_pos = clipspace_to_viewspace(p_pos, p_linear_depth);

	// Move center pixel slightly towards camera to avoid imprecision artifacts due to using of 16bit depth buffer.
	vs_pos *= 0.99;

	vec3 v = params.is_orthogonal ? vec3(0.0, 0.0, -1.0) : -normalize(vs_pos);
	vec4 q_to_v = get_quaternion(v);

	// Micro optimization by taking this out of the inner loop to avoid doing this multiply more than necessary.
	vec3 v_mul_thickness = v * params.thickness;

	vec2 ray_start = viewspace_to_screenspace(vs_pos).xy;
	vec3 ray_start_vc3 = vec3(ray_start, p_linear_depth);

	vec3 gi = vec3(0.0);

	// Hardcoded, since its only intended to change if there is temporal accumulation.
	uint frame = 0u;

	// Also hardcoded to 1u, as that is the best tradeoff between quality and performance.
	for (uint i = 0u; i < 1u; ++i) {
		uint n = frame * 1u + i;
		vec4 rnd01 = rnd_01x4(uvi, n);

		vec3 sample_dir_vs;
		vec2 dir;

		dir = vec2(cos(rnd01.x * PI), sin(rnd01.x * PI));
		sample_dir_vs = vec3(dir, 0.0);

		if (!params.is_orthogonal) {
			sample_dir_vs = transform_vz0qz0(dir, q_to_v);

			vec3 ray_end = viewspace_to_screenspace(vs_pos + sample_dir_vs * (params.z_near * 0.5));

			vec3 ray_dir = ray_end - ray_start_vc3;
			ray_dir /= length(ray_dir.xy);

			dir = ray_dir.xy;
		}

		// Slice construction
		vec3 slice_n = cross(v, sample_dir_vs);
		vec3 proj_n = vs_normal - slice_n * dot(vs_normal, slice_n);

		float proj_n_sqr_len = dot(proj_n, proj_n);
		if (proj_n_sqr_len == 0.0) {
			return vec4(0.0, 0.0, 0.0, 1.0);
		}

		vec3 t = cross(slice_n, proj_n);

		float proj_nr_cp_len = inversesqrt(proj_n_sqr_len);
		float cos_n = dot(proj_n, v) * proj_nr_cp_len;
		float sin_n = dot(t, v) * proj_nr_cp_len;

		vec3 gi0 = vec3(0.0);
		uint occ_bits = 0u;

		for (float d = -1.0; d <= 1.0; d += 2.0) {
			vec2 ray_dir0 = dir * d;

			float t1 = pow(s, rnd01.z);
			rnd01.z = 1.0 - rnd01.z;

			float d05 = d * 0.5;

			for (int i = 0; i < int(count); ++i) {
				vec2 sample_pos = ray_start + ray_dir0 * t1;

				t1 *= s;

				// handle out of bounds samples
				if (sample_pos.x < 0.0 || sample_pos.x >= float(params.screen_size.x) ||
						sample_pos.y < 0.0 || sample_pos.y >= float(params.screen_size.y)) {
					break;
				}

				vec2 sample_uv = sample_pos / vec2(params.screen_size);

				float sample_depth = textureLod(depth_buffer, sample_uv, 0.0).r;

				if (sample_depth <= 0.001) {
					continue;
				}

				// Get view-space position
				vec3 sample_pos_vs = clipspace_to_viewspace(sample_uv, sample_depth);

				vec3 delta_pos_front = sample_pos_vs - vs_pos;
				vec3 delta_pos_back = delta_pos_front - v_mul_thickness;

				// Normalize to get horizon angles
				vec2 hor_cos = vec2(
						dot(normalize(delta_pos_front), v),
						dot(normalize(delta_pos_back), v));

				hor_cos = d >= 0.0 ? hor_cos.xy : hor_cos.yx;

				vec2 hor01 = ((0.5 + 0.5 * sin_n) + d05) - d05 * hor_cos;
				hor01 = clamp(hor01 + rnd01.w * (1.0 / 32.0), 0.0, 1.0);

				uvec2 hor_int = uvec2(floor(hor01 * 32.0));

				uint m_x = hor_int.x < 32u ? OxFFFFFFFFu << hor_int.x : 0u;
				uint m_y = hor_int.y != 0u ? OxFFFFFFFFu >> (32u - hor_int.y) : 0u;

				uint occ_bits0 = m_x & m_y;
				uint vis_bits0 = occ_bits0 & (~occ_bits);

				// compute GI contribution
				if (vis_bits0 != 0u) {
					if (params.backface_rejection) {
						vec3 n0 = load_normal(ivec2(sample_uv * vec2(params.full_screen_size)));

						vec3 proj_n0 = n0 - slice_n * dot(n0, slice_n);
						float proj_n0_sqr_len = dot(proj_n0, proj_n0);

						if (proj_n0_sqr_len != 0.0) {
							float proj_n0r_cp_len = inversesqrt(proj_n0_sqr_len);

							float n_1 = proj_nr_cp_len * proj_n0r_cp_len;

							float sin_phi = dot(proj_n, proj_n0) * n_1;
							float cos_phi = dot(t, proj_n0) * n_1;

							bool flip_t = cos_phi < 0.0;

							sin_phi = !flip_t ? -sin_phi : sin_phi;

							bool c = sin_phi > sin_n;

							float m0 = c ? 1.0 : 0.0;
							float m1 = c ? -0.5 : 0.5;

							float hor_01 = m0 + m1 * (cos_n * abs(cos_phi) + sin_n * sin_phi) + (0.5 * sin_n);
							float rejection_flip = flip_t ? 0.0 : 1.0; // Need this because vis_bits_n can get inverted by flip_t, so we need to account for flip_t's value
							hor_01 = mix(rejection_flip, hor_01, params.normal_rejection); // blend between the rejection output and no rejection
							hor_01 = clamp(hor_01 + rnd01.w * (1.0 / 32.0), 0.0, 1.0);

							uint hor_int_0 = uint(floor(hor_01 * 32.0));
							uint vis_bits_n = hor_int_0 < 32u ? 0xFFFFFFFFu << hor_int_0 : 0u;

							vis_bits_n = !flip_t ? ~vis_bits_n : vis_bits_n;
							vis_bits0 = vis_bits0 & vis_bits_n;
						}
					}

					if (vis_bits0 != 0u) {
						vec4 reprojected_sample_pos = matrices.last_frame_reproj * vec4(sample_uv * 2.0 - 1.0, (sample_depth - params.z_near) / (params.z_far - params.z_near) * 2.0 - 1.0, 1.0);
						vec2 reprojected_sample_uv = (reprojected_sample_pos.xy / reprojected_sample_pos.w) * 0.5 + 0.5;

						vec3 sample_color = textureLod(last_frame, reprojected_sample_uv, 5.0).rgb;

						// Reduce impact of fireflies by tonemapping before averaging: http://graphicrants.blogspot.com/2013/12/tone-mapping.html
						sample_color /= (1.0 + dot(sample_color, vec3(0.299, 0.587, 0.114)));

						float vis0 = float(bitCount(vis_bits0)) * (1.0 / 32.0);
						gi0 += sample_color * vis0;
					}
				}

				occ_bits = occ_bits | occ_bits0;
			}
		}

		gi += gi0;
	}

	// inverse tonemap
	gi /= 1.0 - dot(gi, vec3(0.299, 0.587, 0.114));
	gi *= params.intensity;
	return vec4(gi, 1.0);
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 uv = ((vec2(ssC) + 0.5) / vec2(params.screen_size));

	vec4 lighting;
	float depth = textureLod(depth_buffer, uv, 0.0).r;

	if (depth <= 0.001) {
		imageStore(dest_image, ssC, vec4(0.0, 0.0, 0.0, 1.0));
		return;
	}

	lighting = ssilvb(uv, params.quality, depth);
	//lighting.rgb *= params.intensity;

	imageStore(dest_image, ssC, lighting);
}
