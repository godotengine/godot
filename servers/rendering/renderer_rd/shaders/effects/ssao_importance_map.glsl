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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef GENERATE_MAP
layout(set = 0, binding = 0) uniform sampler2DArray source_texture;
#else
layout(set = 0, binding = 0) uniform sampler2D source_importance;
#endif
layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

#ifdef PROCESS_MAPB
layout(set = 2, binding = 0, std430) buffer Counter {
	uint sum;
}
counter;
#endif

layout(push_constant, std430) uniform Params {
	vec2 half_screen_pixel_size;
	float intensity;
	float power;
}
params;

void main() {
	// Pixel being shaded
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

#ifdef GENERATE_MAP
	// importance map stuff
	uvec2 base_position = ssC * 2;

	vec2 base_uv = (vec2(base_position) + vec2(0.5f, 0.5f)) * params.half_screen_pixel_size;

	float minV = 1.0;
	float maxV = 0.0;
	for (int i = 0; i < 4; i++) {
		vec4 vals = textureGather(source_texture, vec3(base_uv, i));

		// apply the same modifications that would have been applied in the main shader
		vals = params.intensity * vals;

		vals = 1 - vals;

		vals = pow(clamp(vals, 0.0, 1.0), vec4(params.power));

		maxV = max(maxV, max(max(vals.x, vals.y), max(vals.z, vals.w)));
		minV = min(minV, min(min(vals.x, vals.y), min(vals.z, vals.w)));
	}

	float min_max_diff = maxV - minV;

	imageStore(dest_image, ssC, vec4(pow(clamp(min_max_diff * 2.0, 0.0, 1.0), 0.8)));
#endif

#ifdef PROCESS_MAPA
	vec2 uv = (vec2(ssC) + 0.5f) * params.half_screen_pixel_size * 2.0;

	float center = textureLod(source_importance, uv, 0.0).x;

	vec2 half_pixel = params.half_screen_pixel_size;

	vec4 vals;
	vals.x = textureLod(source_importance, uv + vec2(-half_pixel.x * 3, -half_pixel.y), 0.0).x;
	vals.y = textureLod(source_importance, uv + vec2(+half_pixel.x, -half_pixel.y * 3), 0.0).x;
	vals.z = textureLod(source_importance, uv + vec2(+half_pixel.x * 3, +half_pixel.y), 0.0).x;
	vals.w = textureLod(source_importance, uv + vec2(-half_pixel.x, +half_pixel.y * 3), 0.0).x;

	float avg = dot(vals, vec4(0.25, 0.25, 0.25, 0.25));

	imageStore(dest_image, ssC, vec4(avg));
#endif

#ifdef PROCESS_MAPB
	vec2 uv = (vec2(ssC) + 0.5f) * params.half_screen_pixel_size * 2.0;

	float center = textureLod(source_importance, uv, 0.0).x;

	vec2 half_pixel = params.half_screen_pixel_size;

	vec4 vals;
	vals.x = textureLod(source_importance, uv + vec2(-half_pixel.x, -half_pixel.y * 3), 0.0).x;
	vals.y = textureLod(source_importance, uv + vec2(+half_pixel.x * 3, -half_pixel.y), 0.0).x;
	vals.z = textureLod(source_importance, uv + vec2(+half_pixel.x, +half_pixel.y * 3), 0.0).x;
	vals.w = textureLod(source_importance, uv + vec2(-half_pixel.x * 3, +half_pixel.y), 0.0).x;

	float avg = dot(vals, vec4(0.25, 0.25, 0.25, 0.25));

	imageStore(dest_image, ssC, vec4(avg));

	// sum the average; to avoid overflowing we assume max AO resolution is not bigger than 16384x16384; so quarter res (used here) will be 4096x4096, which leaves us with 8 bits per pixel
	uint sum = uint(clamp(avg, 0.0, 1.0) * 255.0 + 0.5);

	// save every 9th to avoid InterlockedAdd congestion - since we're blurring, this is good enough; compensated by multiplying load_counter_avg_div by 9
	if (((ssC.x % 3) + (ssC.y % 3)) == 0) {
		atomicAdd(counter.sum, sum);
	}
#endif
}
