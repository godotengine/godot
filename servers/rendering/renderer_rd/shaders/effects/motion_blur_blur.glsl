///////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025 sphynx-owner
//
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
// Original file link: https://github.com/sphynx-owner/Godot-Motion-Blur-Addon/blob/main/addons/godot-motion-blur/guertin/shader_stages/guertin_sphynx_blur.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define M_PI 3.1415926535897932384626433832795
#define EPSILON 1e-6
#define SMALL_EPSILON 1e-9
#define PIXEL_RADIUS 0.5
#define PIXEL_RADIUS_SQUARED 0.25

// At depth difference of 1 / SOFT_DEPTH_SENSITIVITY the velocity weights are saturated.
// Arrived at via experimentation, larger values mean stronger sensitivity to depth,
// and potential double-blurring of the same object yielding unwanted and obvious harsh colors.
// Smaller values means smoother but weaker blur between close geometry.
#define SOFT_DEPTH_SENSITIVITY 10

layout(set = 0, binding = 0) uniform sampler2D color_sampler;
layout(set = 0, binding = 1) uniform sampler2D velocity_sampler;
layout(set = 0, binding = 2) uniform isampler2D neighbor_max;
layout(rgba16f, set = 0, binding = 3) uniform writeonly image2D output_color;

layout(push_constant, std430) uniform Params {
	int tile_size;
	int sample_count;
	int frame;
	int transparent_bg;
}
params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Guertin's functions https://research.nvidia.com/sites/default/files/pubs/2013-11_A-Fast-and/Guertin2013MotionBlur-small.pdf
// ----------------------------------------------------------
float soft_compare(float a, float b, float sze) {
	return clamp(sze * (a - b), 0, 1);
}

float cone(float a, float b, float sze) {
	return clamp(1 - sze * abs(a - b), 0, 1);
}
// ----------------------------------------------------------

// from https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
// and https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/ (section: Derivation Of IGN And Extensions) for animation of the noise.
// ----------------------------------------------------------
float interleaved_gradient_noise(vec2 uv) {
	uv += float(params.frame) * 5.588238;

	vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);

	return fract(magic.z * fract(dot(uv, magic.xy)));
}
// ----------------------------------------------------------

// from https://github.com/keijiro/KinoMotion
// ----------------------------------------------------------
ivec2 jitter_tile(ivec2 uvi) {
	float rx, ry;
	// HACK @sphynx-owner: multiplying the input uvi seems to help reducing large emergent
	// patchiness in the blurred results along the jittered seams between tiles.
	// TODO @sphynx-owner: find a better jitter setup, there are visible striping along the seams between tiles I would like to eliminate.
	float angle = interleaved_gradient_noise(uvi * 4) * M_PI * 2;
	rx = cos(angle);
	ry = sin(angle);
	return ivec2(vec2(rx, ry) * params.tile_size / 4);
}
// ----------------------------------------------------------

vec4 sample_x_velocity(
		ivec2 x,
		float t,
		vec2 vx,
		float vx_length,
		vec2 wvx,
		float zx,
		float vzx,
		float soft_depth_sensitivity,
		ivec2 render_size,
		out float x_weight,
		out float x_back_weight) {
	// The sample position along the current velocity.
	ivec2 yx = x + ivec2(t * vx);

	// If the position is outside the texture bounds, exit early.
	if (yx.x < 0 || yx.x > render_size.x || yx.y < 0 || yx.y > render_size.y) {
		x_weight = 0;

		x_back_weight = 0;

		return vec4(0);
	}

	// We sample velocity and depth data at the position.
	vec4 syx = texelFetch(velocity_sampler, yx, 0);

	// The UV velocity at the sample position.
	vec2 vyx = syx.xy;

	// Get the distance to the sampled pixel
	float tx = abs(t * vx_length);

	// Whether the found velocity reaches the current pixel
	float reaches_weight = step(tx, abs(dot(vyx * 0.5, wvx)));

	// Get the depth at the sampled pixel
	float zyx = syx.w;

	// get a z-velocity-aware depth estimate of the current pixel
	float x_depth = zx + vzx * t;

	// derive the midground weight (smear the current object's color). It's defined by
	// how similar in depth the found pixel is, and if it's velocity would reach the current pixel.
	x_weight = cone(x_depth, zyx, soft_depth_sensitivity) * reaches_weight;

	// whether the sampled pixel is behind the current pixel
	float overlap_x = soft_compare(x_depth, zyx, soft_depth_sensitivity);

	// derive the background weight (fake transparency). It's defined by
	// whether the sampled pixel is behind the current pixel (part of the background relatively)
	x_back_weight = soft_compare(x_depth, zyx, soft_depth_sensitivity);

	return texelFetch(color_sampler, yx, 0);
}

vec4 sample_y_velocity(
		ivec2 x,
		float t,
		vec2 vn,
		float vn_length,
		vec2 wvn,
		float zx,
		float vzx,
		float soft_depth_sensitivity,
		ivec2 render_size,
		out float y_weight) {
	// The sample position along the neighbor_max velocity.
	ivec2 yn = x + ivec2(t * vn);

	// If the position is outside the texture bounds, exit early.
	if (yn.x < 0 || yn.x > render_size.x || yn.y < 0 || yn.y > render_size.y) {
		y_weight = 0;

		return vec4(0);
	}

	// We sample velocity and depth data at the position.
	vec4 syn = texelFetch(velocity_sampler, yn, 0);

	// The UV velocity at the sample position.
	vec2 vyn = syn.xy;

	// Get the length of the found velocity
	float vyn_length = length(vyn);

	// The depth at the sample position.
	float zyn = syn.w;

	// The z velocity at the sample position.
	float vzyn = syn.z;

	// get a z-velocity-aware depth estimate of the sampled pixel
	float y_depth = zyn - vzyn * t;

	// Get whether the sampled pixel is in front of the current pixel.
	float y_in_front = soft_compare(y_depth, zx, soft_depth_sensitivity);

	// If the found velocity is smaller than a pixel's radius, exit early.
	if (vyn_length < PIXEL_RADIUS || y_in_front <= EPSILON) {
		y_weight = 0;

		return vec4(0);
	}

	// get the distance to the sampled pixel.
	float vn_distance = abs(t * vn_length);

	// derive the foreground weight (foreground  dominant-velocity object's blur onto us). It's determined by:
	// 1. If the found velocity reach over to this pixel.
	// 2. If the depth at the sampled pixel in front of the current one (foreground relatively).
	// 3. An additional bias that handles when the neighbor_max velocity is larger than the found velocity
	// to counter-act the opacity dilution resulting from fewer samples.
	y_weight = step(vn_distance, abs(dot(vyn * 0.5, wvn))) * y_in_front * max(1.05, pow(vn_length / vyn_length, 0.5));

	return texelFetch(color_sampler, yn, 0);
}

void blend_blur(
		vec4 base_color,
		vec4 x_sample,
		float x_weight,
		float x_back_weight,
		vec4 neg_x_sample,
		float neg_x_weight,
		vec4 y_sample,
		float y_weight,
		inout vec4 color_sum,
		inout float color_weight,
		inout float alpha_weight) {
	// We get an optimistic midground weight.
	float current_weight_x = max(x_weight, neg_x_weight);

	// We get an optimistic midground color value
	// TODO @sphynx-owner: see if there's a better heuristic to choosing this value.
	vec4 x_color_sample = mix(neg_x_sample, x_sample, clamp(x_weight / neg_x_weight, 0, 1));

	// We compose the midground, background, and foreground samples based on their weights. Midground (object color smear) is the baseline, on top of it is applied the background (faked transparency), at the top is the foreground (dominant-velocity object's blur over us).
	vec4 current_color = mix(mix(mix(base_color, x_color_sample, current_weight_x), x_sample, x_back_weight), y_sample, y_weight);

	// current_color_weight enables custom support for transparent background. This is relevant for SubViewports.
	float current_color_weight = max(current_color.a, 1 - params.transparent_bg);

	// accumulate into the color sum
	color_sum += vec4(current_color.rgb * current_color_weight, current_color.a);

	// accumulate into the color weight
	color_weight += current_color_weight;

	// color_weight would have been += 1 too but we want
	// to support transparent background so only the alpha weight is agnostic.
	alpha_weight += 1;
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

	// x is the pixel we will start sampling from.
	ivec2 x = uvi;

	// We get velocity and depth data at the current pixel
	vec4 sx = texelFetch(velocity_sampler, x, 0);

	// UV velocity data
	vec2 vx = sx.xy;

	// get the target neighbor_max tile (pixel) to sample from, add jitter between tiles
	// to hide the seams.
	ivec2 neighbor_max_uvi = (x + jitter_tile(x)) / params.tile_size;

	// We get the neighbor-max velocity.
	vec2 vn = texelFetch(neighbor_max, neighbor_max_uvi, 0).xy;

	// color at the current pixel
	vec4 base_color = texelFetch(color_sampler, x, 0);

	// We must account for cases where the dominant velocity is 0 even though
	// The current velocity is not. This is only the case for the skybox, which
	// Will never overlap geometry so it can safely be ignored when calculating neighbor_max
	// NOTE @sphynx-owner: using PIXEL_RADIUS_SQUARED because we compare against the squared length.
	if (dot(vn, vn) < PIXEL_RADIUS_SQUARED && dot(vx, vx) < PIXEL_RADIUS_SQUARED) {
		imageStore(output_color, uvi, base_color);
		return;
	}

	// Length of neighbor_max velocity
	float vn_length = length(vn);

	// normalized neighbor-max velocity
	vec2 wvn = vn / vn_length;

	// Length of current pixel's velocity
	float vx_length = length(vx);

	// We normalize the current pixel's velocity
	vec2 wvx = vx / vx_length;

	// Get the depth at current pixel
	float zx = sx.w;

	// Get z velocity at current pixel
	float vzx = sx.z;

	// We determine a depth sensitivity dynamically based on the depth of the current pixel.
	// The further it is away, the closer to 0 the depth value would be, and thus the greater the sensitivity.
	float soft_depth_sensitivity = SOFT_DEPTH_SENSITIVITY / max(SMALL_EPSILON, zx);

	// Get a jitter value
	float j = interleaved_gradient_noise(x);

	float color_weight = EPSILON;

	float alpha_weight = EPSILON;

	// Create an initial color sum to avoid division-by-0 errors.
	vec4 sum = vec4(base_color.rgb * base_color.a * color_weight, base_color.a * alpha_weight);

	// Slight optimization to not divide every iteration
	float inv_sample_count = 1.0 / params.sample_count;

	for (int i = 0; i < params.sample_count; i++) {
		// time offset
		float t = mix(0, 0.5, float(i + j) * inv_sample_count);

		// opposing time offset
		float neg_t = mix(0, -0.5, float(i + 1 - j) * inv_sample_count);

		float x_weight;

		float x_back_weight;

		// get the midground and background weights (color smearing and fake transparency, respectively)
		vec4 x_sample = sample_x_velocity(x, t, vx, vx_length, wvx, zx, vzx, soft_depth_sensitivity, render_size, x_weight, x_back_weight);

		float neg_x_weight;

		float neg_x_back_weight;

		// get the midground and background weights in the opposing direction
		vec4 neg_x_sample = sample_x_velocity(x, neg_t, vx, vx_length, wvx, zx, vzx, soft_depth_sensitivity, render_size, neg_x_weight, neg_x_back_weight);

		float y_weight;

		// get the foreground weight (dominant-velocity object to blur over us)
		vec4 y_sample = sample_y_velocity(x, t, vn, vn_length, wvn, zx, vzx, soft_depth_sensitivity, render_size, y_weight);

		float neg_y_weight;

		// get the foreground weight in the opposing direction
		vec4 neg_y_sample = sample_y_velocity(x, neg_t, vn, vn_length, wvn, zx, vzx, soft_depth_sensitivity, render_size, neg_y_weight);

		// blend blur given current-direction weights, and opposing-direction midground weights for optimistic smearing.
		blend_blur(base_color, x_sample, x_weight, x_back_weight, neg_x_sample, neg_x_weight, y_sample, y_weight, sum, color_weight, alpha_weight);

		// blend blur given opposing-direction weights, and current-direction midground weights for optimistic smearing.
		blend_blur(base_color, neg_x_sample, neg_x_weight, neg_x_back_weight, x_sample, x_weight, neg_y_sample, neg_y_weight, sum, color_weight, alpha_weight);
	}

	sum.rgb /= color_weight;
	sum.a /= alpha_weight;

	imageStore(output_color, uvi, sum);
}
