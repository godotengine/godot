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
#ifdef SSIL_BLUR_FAST
const int SAMPLE_COUNT = 16;

const float OFFSETS[16] = float[16](
		-14.400867399097198,
		-12.414249746021257,
		-10.427760570612987,
		-8.441379804986935,
		-6.4550869992703435,
		-4.468862236297167,
		-2.4826862657413393,
		-0.49653490850373416,
		1.4896094314876247,
		3.475769408144678,
		5.461967313484028,
		7.448223675960468,
		9.434557915309014,
		11.420990335232771,
		13.407541291721701,
		15);

const float WEIGHTS[16] = float[16](
		0.007404318560869809,
		0.015567980037277579,
		0.02931957619414378,
		0.04946106610551326,
		0.07474014650384328,
		0.10116569566918296,
		0.12265954763625181,
		0.13321663093389685,
		0.12960017628212572,
		0.11293900036723548,
		0.0881596518143952,
		0.06164321939087579,
		0.03860893326330322,
		0.021660706903709255,
		0.010885200413793057,
		0.002968149923582982);
#endif
#ifdef SSIL_BLUR_ACCURATE
const int SAMPLE_COUNT = 8;

const float OFFSETS[8] = float[8](
		-6.400317149797591,
		-4.4305055426526785,
		-2.4612181104350137,
		-0.49222828273139496,
		1.4767017588568079,
		3.4458098836553415,
		5.415332322090894,
		7);

const float WEIGHTS[8] = float[8](
		0.057675741328188125,
		0.1130965670295305,
		0.17359677602551388,
		0.20858861531815848,
		0.19620401492479278,
		0.14447384508442518,
		0.08327585190842782,
		0.023088588380963393);
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssil;

layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

layout(set = 2, binding = 0) uniform sampler2D depth_buffer;
layout(rgba8, set = 2, binding = 1) uniform restrict readonly image2D normal_buffer;

layout(push_constant, std430) uniform Params {
	float edge_sharpness;
	int blur_dir;
	vec2 half_screen_pixel_size;

	ivec2 full_screen_size;
	float pad1;
	float pad2;
}
params;

vec4 bilateral_blur(ivec2 p_pos, vec2 p_uv) {
	vec2 blur_offset = params.blur_dir == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

	float weight_total = 0.0;
	vec4 result = vec4(0.0);

	float center_depth = textureLod(depth_buffer, p_uv, 0.0).r;
	vec3 center_normal = imageLoad(normal_buffer, p_pos).rgb;
	float clamped_sharpness = clamp(0.02 + params.edge_sharpness, 0.0, 1.0);

	for (int i = 0; i < SAMPLE_COUNT; ++i) {
		vec2 sample_uv = p_uv + ((blur_offset * OFFSETS[i]) * params.half_screen_pixel_size);
		ivec2 sample_pos = ivec2(sample_uv * vec2(params.full_screen_size));

		float sample_depth = textureLod(depth_buffer, sample_uv, 0.0).r;
		vec3 sample_normal = imageLoad(normal_buffer, sample_pos).rgb;
		float depth_diff = abs(sample_depth - center_depth) / max(center_depth, 0.001);
		float normal_diff = dot(center_normal, sample_normal);

		if (depth_diff <= clamped_sharpness && normal_diff >= clamped_sharpness) {
			vec4 sample_color = textureLod(source_ssil, sample_uv, 0.0);
			float weight = (WEIGHTS[i]);
			weight_total += weight;
			result += sample_color * weight;
		}
	}

	return weight_total > 0.0 ? result / weight_total : textureLod(source_ssil, p_uv, 0.0);
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	vec2 uv = (vec2(ssC) + 0.5) * params.half_screen_pixel_size;

	vec4 sampled = bilateral_blur(ssC, uv);

	imageStore(dest_image, ssC, sampled);
}
