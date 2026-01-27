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

layout(set = 0, binding = 0) uniform sampler2D input_color;
layout(set = 0, binding = 1) uniform sampler2D input_depth;
layout(set = 0, binding = 2) uniform sampler2D input_normal;
layout(set = 0, binding = 3) uniform sampler2D input_velocity;
layout(rgba16, set = 0, binding = 4) uniform restrict writeonly image2D output_image;

layout(push_constant, std430) uniform Params {
	vec4 pixel_sizes; // xy: full pixel size, zw: half pixel size
	vec4 thresholds; // x: depth_threshold
	vec4 camera; // x: z_near, y: z_far, z: orthogonal
}
params;

vec3 decode_normal(vec3 p_encoded) {
	vec3 n = normalize(p_encoded * 2.0 - 1.0);
	n.z = -n.z;
	return n;
}

float linearize_depth(float p_depth) {
	float ndc = p_depth * 2.0 - 1.0;
	if (params.camera.z > 0.5) {
		return -(ndc * (params.camera.y - params.camera.x) - (params.camera.y + params.camera.x)) * 0.5;
	}
	return 2.0 * params.camera.x * params.camera.y / (params.camera.y + params.camera.x + ndc * (params.camera.y - params.camera.x));
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 half_size = ivec2(1.0 / params.pixel_sizes.zw);
	if (any(greaterThanEqual(pixel, half_size))) {
		return;
	}

	vec2 uv = (vec2(pixel) * 2.0 + vec2(1.0)) * params.pixel_sizes.xy;
	vec2 velocity = textureLod(input_velocity, uv, 0.0).xy;
	vec2 prev_uv = uv + velocity;

	vec3 current = textureLod(input_color, uv, 0.0).rgb;
	vec3 reprojected = current;

	if (all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)))) {
		reprojected = textureLod(input_color, prev_uv, 0.0).rgb;
	}

	float center_depth = linearize_depth(textureLod(input_depth, uv, 0.0).r);
	float prev_depth = center_depth;
	if (all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)))) {
		prev_depth = linearize_depth(textureLod(input_depth, prev_uv, 0.0).r);
	}

	float depth_weight = clamp(1.0 - abs(prev_depth - center_depth) * params.thresholds.x, 0.0, 1.0);
	vec3 color = mix(current, reprojected, depth_weight);

	vec3 normal = decode_normal(textureLod(input_normal, uv, 0.0).xyz);
	float normal_weight = max(normal.z, 0.0);
	color *= mix(0.75, 1.0, normal_weight);

	float luma = dot(color, vec3(0.299, 0.587, 0.114));
	float light_weight = smoothstep(0.05, 0.2, luma);
	color *= light_weight;

	color = max(color, vec3(0.0));
	imageStore(output_image, pixel, vec4(color, 1.0));
}
