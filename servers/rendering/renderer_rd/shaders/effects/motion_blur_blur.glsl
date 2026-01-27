///////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025 sphynx-owner

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2025-01-11: sphynx: first commit
// 2026-01-16: HydrogenC: make tile size specification constant and simplify push constant
// 2026-01-18: AR-DEV-1: add missing t in overlapn
///////////////////////////////////////////////////////////////////////////////////
// Original file link: https://github.com/sphynx-owner/godot-motion-blur-addon-simplified/blob/master/addons/sphynx_motion_blur_toolkit/guertin/shader_stages/shader_files/guertin_sphynx_blur.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define M_PI 3.1415926535897932384626433832795

layout(set = 0, binding = 0) uniform sampler2D color_sampler;
layout(set = 0, binding = 1) uniform sampler2D velocity_sampler;
layout(set = 0, binding = 2) uniform sampler2D neighbor_max;
layout(rgba16f, set = 0, binding = 3) uniform writeonly image2D output_color;

layout(push_constant, std430) uniform Params {
	float motion_blur_intensity;
	int sample_count;
	int frame;
	int clamp_velocities_to_tile;
	int transparent_bg;
	int pad1;
	int pad2;
	int pad3;
}
params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Guertin's functions https://research.nvidia.com/sites/default/files/pubs/2013-11_A-Fast-and/Guertin2013MotionBlur-small.pdf
// ----------------------------------------------------------
float soft_compare(float a, float b, float sze) {
	return clamp(sze * (a - b), 0, 1);
}
// ----------------------------------------------------------

// from https://www.shadertoy.com/view/ftKfzc
// ----------------------------------------------------------
float interleaved_gradient_noise(vec2 uv) {
	uv += float(params.frame) * (vec2(47, 17) * 0.695);

	vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);

	return fract(magic.z * fract(dot(uv, magic.xy)));
}
// ----------------------------------------------------------

// from https://github.com/bradparks/KinoMotion__unity_motion_blur/tree/master
// ----------------------------------------------------------
vec2 safenorm(vec2 v) {
	float l = max(length(v), 1e-6);
	return v / l * int(l >= 0.5);
}

vec2 jitter_tile(vec2 uvi) {
	float rx, ry;
	float angle = interleaved_gradient_noise(uvi + vec2(2, 0)) * M_PI * 2;
	rx = cos(angle);
	ry = sin(angle);
	return vec2(rx, ry) / textureSize(neighbor_max, 0) / 4;
}
// ----------------------------------------------------------

vec4 sample_velocity(sampler2D velocity_texture, vec2 uv) {
	return textureLod(velocity_texture, uv, 0.0) * vec4(vec2(params.motion_blur_intensity), 1, 1);
}

vec4 sample_x_velocity(vec2 x, float t, vec2 vx, float z, float zx, ivec2 render_size, out float x_weight) {
	vec2 yx = x + t * vx / vec2(render_size);

	vec4 vyzwx = sample_velocity(velocity_sampler, yx);

	float zyx = vyzwx.w;

	x_weight = 1 - soft_compare(z + zx * t, zyx, -10);

	return textureLod(color_sampler, yx, 0.0);
}

vec4 sample_y_velocity(vec2 x, float t, vec2 vn, vec2 wn, float z, ivec2 render_size, out float y_weight) {
	vec2 yn = x + t * vn / vec2(render_size);

	vec4 vyzwn = sample_velocity(velocity_sampler, yn);

	vec2 vyn = vyzwn.xy;

	float zyn = vyzwn.w;

	float overlapn = 1 - soft_compare(zyn - vyzwn.z * t, z, -10);

	vec2 wyn = safenorm(vyn);

	float Tn = abs(t * length(vn));

	float vyn_length = max(0.5, length(vyn));

	if (params.clamp_velocities_to_tile == 1) {
		float clamp_ratio = max(vyn_length / TILE_SIZE, 1.0);
		vyn /= clamp_ratio;
		vyn_length /= clamp_ratio;
	}

	float projected = abs(dot(wyn, wn));

	y_weight = step(Tn, vyn_length * projected) * overlapn;

	return textureLod(color_sampler, yn, 0.0);
}

void blend_blur(
		vec4 base_color,
		vec4 x_sample,
		float x_weight,
		vec4 neg_x_sample,
		float neg_x_weight,
		vec4 y_sample,
		float y_weight,
		float weight_modifier,
		inout vec4 color_sum,
		inout float color_weight,
		inout float alpha_weight) {
	float current_weight_x = max(x_weight, neg_x_weight);

	vec4 x_color_sample = mix(neg_x_sample, x_sample, clamp(x_weight / neg_x_weight, 0, 1));

	vec4 current_color = mix(mix(base_color, x_color_sample, current_weight_x), y_sample, y_weight);

	float current_color_weight = weight_modifier * max(current_color.a, 1 - params.transparent_bg);

	float current_alpha_weight = weight_modifier;

	color_sum += vec4(current_color.xyz * current_color_weight, current_color.a * current_alpha_weight);

	color_weight += current_color_weight;

	alpha_weight += current_alpha_weight;
}

void main() {
	// The size of the output texture
	ivec2 render_size = ivec2(textureSize(color_sampler, 0));

	// The pixel we are running the shader for.
	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);

	// If the pixel we are in is outside the target render's size, we
	// exit early
	if ((uvi.x >= render_size.x) || (uvi.y >= render_size.y)) {
		return;
	}

	// We convert the pixel position into a texturing sampling position
	// we add 0.5 to offset the sampling to be in the "middle" of the pixel
	// and avoid artifacts caused by bilinear interpolation.
	vec2 x = (vec2(uvi) + vec2(0.5)) / vec2(render_size);

	// We get the neighbor-max velocity for the tile we are in, with some jitter
	// between tiles to hide seams between them.
	vec4 vnzw = sample_velocity(neighbor_max, x + jitter_tile(uvi));

	vec2 vn = vnzw.xy;

	float vn_length = length(vn);

	vec4 base_color = textureLod(color_sampler, x, 0.0);

	// We get the true velocity at the current pixel
	vec4 vxzw = sample_velocity(velocity_sampler, x);

	vec2 vx = vxzw.xy;

	float vx_length = length(vx);

	if (params.clamp_velocities_to_tile == 1) {
		float clamp_ratio = max(vn_length / TILE_SIZE, 1.0);
		vn /= clamp_ratio;
		vn_length /= clamp_ratio;

		clamp_ratio = max(vx_length / TILE_SIZE, 1.0);
		vx /= clamp_ratio;
		vx_length /= clamp_ratio;
	}

	// We must account for cases where the dominant velocity is 0 even though
	// The current velocity is not. This is only the case for the skybox, which
	// Will never overlap geometry so it can safely be ignored when calculating neighbor_max
	if (vn_length < 0.5) {
		imageStore(output_color, uvi, base_color);
		return;
	}

	// We normalize neighbor-max velocity
	vec2 wn = safenorm(vn);

	// Get the depth at current pixel
	float zx = vxzw.w;

	// We get some random value for the current pixel between 0 and 1. This will be used to
	// jitter the blur sampling, and achieve smoother looking blur gradient
	// with a fraction of the sample count.
	float j = interleaved_gradient_noise(uvi);

	float color_weight = 1e-6;

	float alpha_weight = 1e-6;

	// Create an initial color sum
	vec4 sum = vec4(base_color.xyx * base_color.a * color_weight, base_color.a * alpha_weight);

	for (int i = 0; i < params.sample_count; i++) {
		float ti = (i + j) / params.sample_count;

		// A point in time along the blur interval, used to scale velocity vectors to sample for color.
		float t = mix(-0.5, 0, ti);
		float neg_t = -t;
		float current_total_weight = 1;

		float x_weight;
		vec4 x_sample = sample_x_velocity(x, t, vx, zx, vxzw.z, render_size, x_weight);
		float neg_x_weight;
		vec4 neg_x_sample = sample_x_velocity(x, neg_t, vx, zx, vxzw.z, render_size, neg_x_weight);

		float y_weight;
		vec4 y_sample = sample_y_velocity(x, t, vn, wn, zx, render_size, y_weight);
		float neg_y_weight;
		vec4 neg_y_sample = sample_y_velocity(x, -t, vn, wn, zx, render_size, neg_y_weight);
		blend_blur(base_color, x_sample, x_weight, neg_x_sample, neg_x_weight, y_sample, y_weight, current_total_weight, sum, color_weight, alpha_weight);
		blend_blur(base_color, neg_x_sample, neg_x_weight, x_sample, x_weight, neg_y_sample, neg_y_weight, current_total_weight, sum, color_weight, alpha_weight);
	}

	sum.xyz /= color_weight;
	sum.a /= alpha_weight;

	imageStore(output_color, uvi, sum);
}
