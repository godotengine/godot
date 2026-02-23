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

// pre-calculated weights and offsets to speed up rendering
const float WEIGHTS[5] = { 0.20454469416555826,
	0.23471987829919136,
	0.24126966719678053,
	0.222149652256188,
	0.09731610808228183 };
const float OFFSETS[5] = { -3.4757694081446777,
	-1.489609431487625,
	0.4965349085037341,
	2.4826862657413393,
	4 };

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssil;

layout(rgba16, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

#ifdef SSIL_BLUR_ACCURATE
layout(set = 2, binding = 0) uniform sampler2D depth_buffer;
#endif
layout(rgba8, set = 2, binding = 1) uniform restrict readonly image2D normal_buffer;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	float edge_threshold;
	int quality;

	float z_near;
	float z_far;
	float blur_intensity;
	float depth_difference_threshold;

	vec2 blur_offset;
	ivec2 full_screen_size;
}
params;

vec3 load_normal(ivec2 p_pos) {
	vec3 encoded_normal = normalize(imageLoad(normal_buffer, p_pos).xyz * 2.0 - 1.0);
	encoded_normal.z = -encoded_normal.z;
	return encoded_normal;
}

vec4 bilateral_blur(vec2 p_uv, int p_quality) {
	vec3 center_nor = load_normal(ivec2((p_uv * vec2(params.full_screen_size))));
	vec2 step = (params.blur_offset * params.blur_intensity) / vec2(params.screen_size);

#ifdef SSIL_BLUR_ACCURATE
	float center_depth = textureLod(depth_buffer, p_uv, 0.0).r;
#endif

	vec2 size = vec2(params.screen_size);

	float gaussian_weight_total = 0.0;
	vec4 result = vec4(0.0);

	for (int i = 0; i < 5; ++i) {
		vec2 sample_uv = p_uv + (OFFSETS[i] * step);
		ivec2 sample_uvi = ivec2((sample_uv * vec2(params.full_screen_size)));

		vec3 sample_nor = load_normal(sample_uvi);

		if (dot(sample_nor, center_nor) <= params.edge_threshold) {
			continue;
		}

#ifdef SSIL_BLUR_ACCURATE
		float sample_depth = textureLod(depth_buffer, sample_uv, 0.0).r;
		if (abs(sample_depth - center_depth) >= params.depth_difference_threshold) {
			continue;
		}
#endif
		// Despite the fact we aren't generating a mip5 for the source_ssil, choosing mip5 gives a significant speed increase. May need outside testing to confirm
		vec4 sample_color = textureLod(source_ssil, sample_uv, 5.0);
		float gaussian_weight = WEIGHTS[i];
		gaussian_weight_total += gaussian_weight;
		result += gaussian_weight * sample_color;
	}

	return gaussian_weight_total > 0.0 ? result / gaussian_weight_total : textureLod(source_ssil, p_uv, 5.0);
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 uv = (vec2(ssC) + 0.5) / vec2(params.screen_size);

	vec4 blurred_ssilvb = bilateral_blur(uv, params.quality);

	imageStore(dest_image, ssC, blurred_ssilvb);
}
