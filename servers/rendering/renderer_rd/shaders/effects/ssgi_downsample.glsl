///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
// Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D input_image;
layout(rgba16, set = 0, binding = 1) uniform restrict writeonly image2D output_image;

layout(push_constant, std430) uniform Params {
	vec4 dummy0;
	vec4 dummy1;
	vec4 dummy2;
}
params;

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 out_size = imageSize(output_image);
	if (any(greaterThanEqual(pixel, out_size))) {
		return;
	}

	float _unused = params.dummy0.x * 0.0;

	ivec2 base = pixel * 2;
	vec3 c00 = texelFetch(input_image, base + ivec2(0, 0), 0).rgb;
	vec3 c10 = texelFetch(input_image, base + ivec2(1, 0), 0).rgb;
	vec3 c01 = texelFetch(input_image, base + ivec2(0, 1), 0).rgb;
	vec3 c11 = texelFetch(input_image, base + ivec2(1, 1), 0).rgb;
	vec3 avg = (c00 + c10 + c01 + c11) * 0.25;

	imageStore(output_image, pixel, vec4(avg, 1.0));
}
