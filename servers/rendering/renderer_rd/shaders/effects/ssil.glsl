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
#define SSIL_DEPTH_MIPS_GLOBAL_OFFSET (-4.3)

// Sample count is num_slices * num_samples.
const int num_samples[5] = { 4, 4, 4, 4, 4 };
const int num_slices[5] = { 1, 2, 4, 6, 8 };

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16, set = 0, binding = 0) uniform restrict writeonly image2D dest_image;

// Buffers
layout(set = 1, binding = 0) uniform sampler2D depth_buffer;
layout(rgba8, set = 1, binding = 1) uniform restrict readonly image2D normal_buffer;

layout(r8, set = 2, binding = 0) uniform restrict writeonly image2D edge_weights_image;
layout(set = 2, binding = 1) uniform sampler2D last_frame;
layout(set = 2, binding = 2) uniform Matrices {
	mat4 last_frame_reproj;
}
matrices;

// Push constant
layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int quality;
	uint frame_index;

	float z_near;
	float z_far;
	float radius;
	float thickness;

	float intensity;
	float normal_rejection;
	float pad;
	bool is_orthogonal;

	vec2 NDC_to_view_mul;
	ivec2 full_screen_size;
}
params;

float pack_edges(vec4 p_edgesLRTB) {
	p_edgesLRTB = round(clamp(p_edgesLRTB, 0.0, 1.0) * 3.05);
	return dot(p_edgesLRTB, vec4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

vec4 calculate_edges(const float p_center_z, const float p_left_z, const float p_right_z, const float p_top_z, const float p_bottom_z) {
	// slope-sensitive depth-based edge detection
	vec4 edgesLRTB = vec4(p_left_z, p_right_z, p_top_z, p_bottom_z) - p_center_z;
	vec4 edgesLRTB_slope_adjusted = edgesLRTB + edgesLRTB.yxwz;
	edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTB_slope_adjusted));
	return clamp((1.3 - edgesLRTB / (p_center_z * 0.040)), 0.0, 1.0);
}

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

// https://jcgt.org/published/0006/01/01/
void get_orthonormal_basis(vec3 n, out vec3 b1, out vec3 b2) {
	float sign_ = n.z >= 0.0 ? 1.0 : -1.0;
	float a = -1.0 / (sign_ + n.z);
	float b = n.x * n.y * a;
	b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
	b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

// Helper functions
vec3 load_normal(ivec2 p_pos) {
	vec3 encoded_normal = normalize(imageLoad(normal_buffer, p_pos).xyz * 2.0 - 1.0);
	encoded_normal.z = -encoded_normal.z;
	return encoded_normal;
}

// https://graphics.stanford.edu/%7Eseander/bithacks.html#CountBitsSetParallel
uint CountBits(uint v) {
	v = v - ((v >> 1u) & 0x55555555u);
	v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
	return ((v + (v >> 4u) & 0xF0F0F0Fu) * 0x1010101u) >> 24u;
}

// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/
float ign(vec2 p_uv, uint p_n) {
	p_uv += 5.588238 * float(p_n);

	return mod(52.9829189 * mod(0.06711056 * p_uv.x + 0.00583715 * p_uv.y, 1.0), 1.0);
}

vec2 hash23(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

vec2 r2_modified(float idx, vec2 seed)
{
    return fract(seed + float(idx) * vec2(0.245122333753, 0.430159709002));
}

float GetBayerFromCoordLevel(vec2 pixelpos)
{
    ivec2 ppos = ivec2(pixelpos);
    int sum = 0;
    for(int i = 0; i<4; i++)
    {
         ivec2 t = ppos & 1;
         sum = sum * 4 | (t.x ^ t.y) * 2 | t.x;
         ppos /= 2;
    }    
    return float(sum) / float(1 << (2 * 4));
}

float ReshapeUniformToTriangle(float v) {
    v = v * 2.0 - 1.0;
    v = sign(v) * (1.0 - sqrt(max(0.0, 1.0 - abs(v)))); // [-1, 1], max prevents NaNs
    return v + 0.5; // [-0.5, 1.5]
}

float PhiNoise(uvec3 uvw)
{
    // flip every other tile to reduce anisotropy
    if(((uvw.x ^ uvw.y ^ uvw.z) & 4u) == 0u) uvw = uvw.yzx;
    
    // constants of 3d Roberts sequence rounded to nearest primes
    const uint r0 = 3518319149u;// prime[(2^32-1) / phi_3  ]
    const uint r1 = 2882110339u;// prime[(2^32-1) / phi_3^2]
    const uint r2 = 2360945581u;// prime[(2^32-1) / phi_3^3]
    
    // h = high-freq dither noise
    uint h = (uvw.x * r0) + (uvw.y * r1) + (uvw.z * r2);
    
    // l = low-freq white noise
    uvw = uvw >> 2u;// 3u works equally well (I think)
    uint l = ((uvw.x * r0) ^ (uvw.y * r1) ^ (uvw.z * r2)) * r1;
    
    // combine low and high
    return float(l + h) * (1.0 / 4294967296.0);
}

void ssilvb(out vec4 r_color, out vec4 r_edges, vec2 p_pos, const int p_quality) {
	ivec2 uvi = ivec2(p_pos * vec2(params.screen_size));
	ivec2 full_res_uvi = ivec2(p_pos * vec2(params.full_screen_size));
	vec2 pos_rounded = vec2(uvi);

	float pix_z, pix_left_z, pix_top_z, pix_right_z, pix_bottom_z;

	vec4 valuesUL = textureGather(depth_buffer, pos_rounded * (1.0 / params.screen_size));
	vec4 valuesBR = textureGather(depth_buffer, (pos_rounded + vec2(1.0)) * (1.0 / params.screen_size));

	// get this pixel's viewspace depth
	pix_z = valuesUL.y;

	// get left right top bottom neighboring pixels for edge detection (gets compiled out on quality_level == 0)
	pix_left_z = valuesUL.x;
	pix_top_z = valuesUL.z;
	pix_right_z = valuesBR.z;
	pix_bottom_z = valuesBR.x;

	// edge mask for between this and left/right/top/bottom neighbor pixels - not used in quality level 0 so initialize to "no edge" (1 is no edge, 0 is edge)
	vec4 edgesLRTB = vec4(1.0, 1.0, 1.0, 1.0);
	edgesLRTB = calculate_edges(pix_z, pix_left_z, pix_right_z, pix_top_z, pix_bottom_z);

	uint count = uint(num_samples[p_quality]);

	vec3 vs_normal = load_normal(full_res_uvi);

	vec3 vs_pos = clipspace_to_viewspace(p_pos, pix_z);
	const vec2 pixel_size_at_center = clipspace_to_viewspace(p_pos + (1.0 / vec2(params.screen_size)), pix_z).xy - vs_pos.xy;
	const float s = pow(params.radius / pixel_size_at_center.x, 1.0 / float(count));

	// Move center pixel slightly towards camera to avoid imprecision artifacts due to using of 16bit depth buffer.
	vs_pos *= 0.99;

	vec3 v = params.is_orthogonal ? vec3(0.0, 0.0, -1.0) : -normalize(vs_pos);
	vec3 v_tangent;
	vec3 v_bitangent;
	get_orthonormal_basis(v, v_tangent, v_bitangent);

	// Micro optimization by taking this out of the inner loop to avoid doing this multiply more than necessary.
	vec3 v_mul_thickness = v * params.thickness;

	vec2 ray_start = viewspace_to_screenspace(vs_pos).xy;
	vec3 ray_start_vc3 = vec3(ray_start, pix_z);

	float ao = 0.0;
	vec3 gi = vec3(0.0);

	uint frame = params.frame_index;
	uint dir_count = uint(num_slices[p_quality]);

	for (uint i = 0u; i < dir_count; ++i) {
		uint n = frame * dir_count + i;
		float rnd01 = ign(floor(p_pos * vec2(params.screen_size)), n);
		//rnd01 = ReshapeUniformToTriangle(rnd01);

		vec3 sample_dir_vs;
		vec2 dir;

		dir = vec2(cos(rnd01 * PI), sin(rnd01 * PI));
		sample_dir_vs = vec3(dir, 0.0);

		if (!params.is_orthogonal) {
			sample_dir_vs = dir.x * v_tangent + dir.y * v_bitangent;

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
			r_color = vec4(0.0, 0.0, 0.0, 1.0);
			r_edges = edgesLRTB;
			return;
		}

		vec3 t = cross(slice_n, proj_n);

		float proj_nr_cp_len = inversesqrt(proj_n_sqr_len);
		float cos_n = dot(proj_n, v) * proj_nr_cp_len;
		float sin_n = dot(t, v) * proj_nr_cp_len;

		vec3 gi0 = vec3(0.0);
		uint occ_bits = 0u;

		const float global_mip_offset = SSIL_DEPTH_MIPS_GLOBAL_OFFSET;
		float mip_offset = (log2(s) + global_mip_offset);

		//vec2 b_rnd01_vc2 = vec2(GetBayerFromCoordLevel(p_pos * vec2(params.screen_size)), fract(GetBayerFromCoordLevel(p_pos * vec2(params.screen_size))) + 0.6180339887);
		//vec2 rnd01_vc2 = vec2(ReshapeUniformToTriangle(b_rnd01_vc2.x), ReshapeUniformToTriangle(b_rnd01_vc2.y));
		vec2 rnd01_vc2 = vec2(PhiNoise(uvec3(p_pos * vec2(params.screen_size), n)), GetBayerFromCoordLevel(p_pos * vec2(params.screen_size)));

		for (float d = -1.0; d <= 1.0; d += 2.0) {
			vec2 ray_dir0 = dir * d;

			float t1 = pow(s, rnd01_vc2.x);
			rnd01_vc2.x = 1.0 - rnd01_vc2.x;

			float d05 = d * 0.5;

			for (int i = 0; i < int(count); ++i) {
				vec2 sample_pos = ray_start + ray_dir0 * t1;
				vec2 sample_uv = sample_pos / vec2(params.screen_size);

				t1 *= s;

				// handle out of bounds samples
				if (sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
					sample_uv.y < 0.0 || sample_uv.y > 1.0) {
					break;
				}

				float sample_depth = textureLod(depth_buffer, sample_uv, mip_offset).r;

				// Get view-space position
				vec3 sample_pos_vs = clipspace_to_viewspace(sample_uv, sample_depth);

				vec3 delta_pos_front = sample_pos_vs - vs_pos;
				vec3 delta_pos_back = delta_pos_front - v_mul_thickness;

				if (!params.is_orthogonal) {
					delta_pos_back = delta_pos_front + normalize(sample_pos_vs) * params.thickness;
				}

				// Normalize to get horizon angles
				vec2 hor_cos = vec2(
						dot(normalize(delta_pos_front), v),
						dot(normalize(delta_pos_back), v));

				hor_cos = d >= 0.0 ? hor_cos.xy : hor_cos.yx;

				vec2 hor01 = ((0.5 + 0.5 * sin_n) + d05) - d05 * hor_cos;
				hor01 = clamp(hor01 + rnd01_vc2.y * (1.0 / 32.0), 0.0, 1.0);

				uvec2 hor_int = uvec2(floor(hor01 * 32.0));

				uint m_x = hor_int.x < 32u ? 0xFFFFFFFFu << hor_int.x : 0u;
				uint m_y = hor_int.y != 0u ? 0xFFFFFFFFu >> (32u - hor_int.y) : 0u;

				uint occ_bits0 = m_x & m_y;
				uint vis_bits0 = occ_bits0 & (~occ_bits);

				// compute GI contribution
				if (vis_bits0 != 0u) {
					float vis0 = float(CountBits(vis_bits0)) * (1.0 / 32.0);

					if (params.normal_rejection > 0.01) {
						vec3 n0 = load_normal(ivec2(sample_uv * vec2(params.full_screen_size)));

						vec3 proj_n0 = n0 - slice_n * dot(n0, slice_n);
						float proj_n0_sqr_len = dot(proj_n0, proj_n0);

						if (proj_n0_sqr_len != 0.0) {
							float proj_n0r_cp_len = inversesqrt(proj_n0_sqr_len);

							float u = dot(proj_n, proj_n0);
							u *= proj_nr_cp_len;
							u *= proj_n0r_cp_len;

							float v = u * -0.5 + 0.5;

							float rejection = clamp(v * 4.0 + 0.0, 0.0, 1.0);
							vis0 *= mix(1.0, rejection, params.normal_rejection);
						}
					}

					vec4 reprojected_sample_pos = matrices.last_frame_reproj * vec4(sample_uv * 2.0 - 1.0, (sample_depth - params.z_near) / (params.z_far - params.z_near) * 2.0 - 1.0, 1.0);
					vec2 reprojected_sample_uv = (reprojected_sample_pos.xy / reprojected_sample_pos.w) * 0.5 + 0.5;

					vec3 sample_color = textureLod(last_frame, reprojected_sample_uv, 5.0).rgb;

					// Reduce impact of fireflies by tonemapping before averaging: http://graphicrants.blogspot.com/2013/12/tone-mapping.html
					sample_color /= (1.0 + dot(sample_color, vec3(0.299, 0.587, 0.114)));

					gi0 += sample_color * vis0;
				}

				occ_bits = occ_bits | occ_bits0;
			}

			if (occ_bits == 0xFFFFFFFFu) {
				break;
			}
		}
		float occ0 = float(CountBits(occ_bits) * (1.0 / 32.0));

		ao += 1.0 - occ0;
		gi += gi0;
	}

	float norm = (1.0 / float(dir_count));

	// inverse tonemap
	gi *= norm;
	gi /= 1.0 - dot(gi, vec3(0.299, 0.587, 0.114));
	gi = params.intensity * gi;

	ao *= norm;

	r_color = vec4(gi, ao);
	r_edges = edgesLRTB;
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec4 out_color;
	vec4 out_edges;

	vec2 uv = ((vec2(ssC) + 0.5) / vec2(params.screen_size));
	ssilvb(out_color, out_edges, uv, params.quality);

	imageStore(dest_image, ssC, out_color);
	imageStore(edge_weights_image, ssC, vec4(pack_edges(out_edges)));
}
