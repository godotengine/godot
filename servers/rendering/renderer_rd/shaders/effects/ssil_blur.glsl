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

#ifdef SSIL_BLUR_FAST
// pre-calculated weights and offsets to speed up rendering
const int SAMPLE_COUNT = 8;

const float OFFSETS[8] = float[8](
    -6.168999895680605,
    -4.2493201357512165,
    -2.351564403533789,
    -0.46943377969837197,
    1.4091998770852123,
    3.2979348079914823,
    5.206290077682596,
    7
);

const float WEIGHTS[8] = float[8](
    0.0028918856621288915,
    0.03708585635306172,
    0.18660117722359726,
    0.3721448146065306,
    0.29569619257017404,
    0.09344193718198306,
    0.011649408027304327,
    0.00048872837522002
);
#endif
#ifdef SSIL_BLUR_ACCURATE
const int SAMPLE_COUNT = 8;

const float OFFSETS[8] = float[8](
    -6.328357272092126,
    -4.378621204796657,
    -2.431625915613778,
    -0.4862426846689484,
    1.4588111840004858,
    3.4048471718931532,
    5.353083811756559,
    7
);

const float WEIGHTS[8] = float[8](
    0.027508406306604068,
    0.08940648616079577,
    0.18921490087565024,
    0.26088633929947086,
    0.2343989200518563,
    0.13722534949218246,
    0.052327012559001844,
    0.009032585254438357
);
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssil;

layout(rgba16f, set = 1, binding = 0) uniform image2D dest_image;

layout(set = 2, binding = 0) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform Params {
    float edge_sharpness;
    int blur_dir;
    vec2 half_screen_pixel_size;
}
params;

vec4 bilateral_blur(ivec2 p_pos, vec2 p_uv) {
    vec2 blur_offset = params.blur_dir == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

	float weight_total = 0.0;
	vec4 result = vec4(0.0);

	float center_depth = textureLod(depth_buffer, p_uv, 0.0).r;

	for (int i = 0; i < SAMPLE_COUNT; ++i) {
		vec2 sample_uv = p_uv + ((blur_offset * OFFSETS[i]) * params.half_screen_pixel_size);

        float sample_depth = textureLod(depth_buffer, sample_uv, 0.0).r;
        float depth_diff = abs(sample_depth - center_depth) / max(center_depth, 0.001);

		if(depth_diff <= clamp(0.02 + params.edge_sharpness, 0.0, 1.0)) {
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