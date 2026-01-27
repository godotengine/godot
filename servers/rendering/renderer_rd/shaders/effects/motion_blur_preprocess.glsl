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
///////////////////////////////////////////////////////////////////////////////////
// Original file link: https://github.com/sphynx-owner/Godot-Motion-Blur-Addon/blob/main/addons/godot-motion-blur/pre_blur_processing/shader_stages/pre_blur_processor.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define PIXEL_RADIUS_SQUARED 0.25

// Arrived at via experimentation
#define OBJECT_UV_CHANGE_EPSILON 0.00001

#define MAX_VIEWS 2

#include "../scene_data_inc.glsl"

layout(set = 0, binding = 0) uniform sampler2D depth_sampler;
layout(set = 0, binding = 1) uniform sampler2D vector_sampler;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D vector_output;

#define view_mat3x4_to_mat4(matrix) transpose(mat4(matrix[0], matrix[1], matrix[2], vec4(0.0, 0.0, 0.0, 1.0)))

#define sharp_step(lower, upper, x) clamp((x - lower) / (upper - lower), 0, 1)

#define clamp_length(vec, length_vec, max_length) vec *= max_length / max(max_length, length(length_vec))

#define uv_to_ndc(uv) uv * 2.0 - 1.0

#define ndc_to_uv(ndc) ndc * 0.5 + 0.5

layout(set = 0, binding = 3, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene;

layout(push_constant, std430) uniform Params {
	float rotation_velocity_multiplier;
	float movement_velocity_multiplier;
	float object_velocity_multiplier;
	float velocity_lower_threshold;

	float velocity_upper_threshold;
	float support_fsr2;
	float motion_blur_intensity;
	float tile_size;
}
params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
	ivec2 render_size = ivec2(textureSize(vector_sampler, 0));

	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);

	if ((uvi.x >= render_size.x) || (uvi.y >= render_size.y)) {
		return;
	}

	SceneData scene_data = scene.data;

	SceneData previous_scene_data = scene.prev_data;

	vec2 uvn = vec2(uvi + vec2(0.5) + vec2(0, -0.1)) / render_size;

	// We get the view-space position at the pixel
	// ---------------------------------------------------
	float depth = texelFetch(depth_sampler, uvi, 0).x;

	vec4 view_position = scene_data.inv_projection_matrix * vec4(uv_to_ndc(uvn), depth, 1.0);

	view_position.xyz /= view_position.w;
	// ---------------------------------------------------

	// We derive a current_uv which we can compare against our manually extracted UVs.
	vec3 current_uv = vec3(uvn, depth);

	mat4 inv_view_matrix = view_mat3x4_to_mat4(scene_data.inv_view_matrix);

	// We take the view position, transform it to a world position, and then back to a view position using the past view matrix, resulting in an estimation of where the pixel
	// was last frame. This estimation only works for static environment. It breaks for moving objects.
	// ---------------------------------------------------
	vec4 world_position = inv_view_matrix * vec4(view_position.xyz, 1.0);

	mat4 prev_view_matrix = view_mat3x4_to_mat4(previous_scene_data.view_matrix);

	vec4 view_past_position = prev_view_matrix * vec4(world_position.xyz, 1.0);
	// ---------------------------------------------------

	// We extract a UV and depth change
	// ---------------------------------------------------
	vec4 view_past_ndc = previous_scene_data.projection_matrix * view_past_position;

	view_past_ndc.xyz /= view_past_ndc.w;

	vec3 past_uv = vec3(ndc_to_uv(view_past_ndc.xy), view_past_ndc.z);

	vec4 view_past_ndc_cache = view_past_ndc;

	vec3 camera_uv_change = past_uv - current_uv;
	// ---------------------------------------------------

	// We do a similar process, but this time only using the rotation part of the view matrices,
	// resulting in the part of the UV change that was caused by the rotation between frames.
	// ---------------------------------------------------
	world_position = mat4(mat3(inv_view_matrix)) * vec4(view_position.xyz, 1.0);

	view_past_position = mat4(mat3(prev_view_matrix)) * vec4(world_position.xyz, 1.0);

	view_past_ndc = previous_scene_data.projection_matrix * view_past_position;

	view_past_ndc.xyz /= view_past_ndc.w;

	past_uv = vec3(ndc_to_uv(view_past_ndc.xy), view_past_ndc.z);

	vec3 camera_rotation_uv_change = past_uv - current_uv;
	// ---------------------------------------------------

	// By subtracting the rotation part of the UV change from the total UV change, we can arrive
	// at the UV change that was cause by the camera's movement.
	vec3 camera_movement_uv_change = camera_uv_change - camera_rotation_uv_change;

	// Get a velocity sample
	vec2 sampled_velocity = texelFetch(vector_sampler, uvi, 0).xy;

	// FSR2 alters the velocity buffer in a very specific way:
	// 1. Static geometry has its velocity replaced with a vec2(-1).
	// 2. Around the edges of moving geometry there are some pixels that have their velocities *divided by 2* and then added a vec2(-0.5).
	// The following code attempts to account for that, but it would
	// fail if valid velocities happen to land on these looked-for edge cases.
	if (params.support_fsr2 > 0.5) {
		if (sampled_velocity == vec2(-1)) {
			sampled_velocity = camera_uv_change.xy;
		}

		vec2 potential_replacement = (sampled_velocity + 0.5) * 2.0;

		if (dot(potential_replacement, potential_replacement) < dot(sampled_velocity, sampled_velocity)) {
			sampled_velocity = potential_replacement;
		}
	}

	/**
	In Godot, background and skyboxes do not write to the velocity buffer. However, our manually-extracted UV change uses the view-matrices and the depth buffer to generate equivalent velocities, and it works even when the depth is 0 (infinity/background). Assuming the skybox is always static (does not move on its own), the value we extracted can serve as the ground truth. We set the base velocity to that of the manually extracted vectors, and keep it if the depth is 0 (background depth). It's not currently possible, but in the future you may be able to write to the veolcity buffer without writing to the depth buffer, so I'm checking for non-zero velocity as well just to be safe.
	**/
	// ---------------------------------------------------
	vec3 base_velocity = camera_uv_change;

	if (dot(sampled_velocity * render_size, sampled_velocity * render_size) > PIXEL_RADIUS_SQUARED || depth > 0) {
		base_velocity.xy = sampled_velocity;
	}
	// ---------------------------------------------------

	// By subtracting the "original" UV change stored on base_velocity from the manuall-derived
	// camera UV change, we end up with the UV change that was caused by the object's motion
	vec3 object_uv_change = base_velocity - camera_uv_change;

	// Now that we have the 3 components that make the original motion vectors isolated, we
	// can put them back together after tuning them however we like.
	// We assume that component magnitudes are between 0 and 1. This must be enforced on the editor interface level.
	vec3 total_velocity = camera_rotation_uv_change * params.rotation_velocity_multiplier + camera_movement_uv_change * params.movement_velocity_multiplier + object_uv_change * params.object_velocity_multiplier;

	// If depth == 0 (skybox), or the objcet is not static (has some object uv change), clear z velocity.
	// The z velocity was manually extracted using view matrices and thus can only be safely assumed for static environment.
	// In the case of background pixels, it does not make much sense for them to have "depth velocity". In addition, the depth velocity
	// of the background is very saturated since it's a point at infinity that covers large distances easily, and I worry
	// about noise it might introduce.
	if (depth == 0 || dot(object_uv_change.xy, object_uv_change.xy) > OBJECT_UV_CHANGE_EPSILON) {
		total_velocity.z = 0;
		base_velocity.z = 0;
	}

	// This is a heuristic I came up with. Simply scaling down individual components of the original velocity
	// can yield unintuivite results if those components are large but cancel out. For example, if a camera is following
	// a speeding car, that car appears stationary in the camera's view, and so it's original velocity is small or zero.
	// However under the hood that velocity is comprized of a very large object movement component on the car, cancelled out
	// by the movement component of the camera that follows it. In that scenario, turning off just the object movement component would
	// uncover that hidden camera movement component, and we would see the car start blurring more instead of less.
	// The solution I stumbled across when trying to solve this issue, has proven to be more robust than expected.
	// The rule of thumb is that users that configure these velocity multipliers expect to REDUCE one or more aspects
	// that otherwise trigger motion blur. So intuitively, the final velocity that decides the motion blur amount should
	// be reduced or kept the same as the original velocity. Now, if all multipliers are set to lower than 1, we can
	// adjust our expectations and say that we expect the final velocity to be no larger than the largest configured multiplier multiplied
	// by the original velocity. So if we have 0.2 object movement, 0.4 camera movement, and 0.1 camera rotation, we should not
	// see any velocity that's larger than 0.4 of the original velocity.
	// ---------------------------------------------------
	float max_component_multiplier = max(params.rotation_velocity_multiplier, max(params.movement_velocity_multiplier, params.object_velocity_multiplier));

	vec3 fallback_velocity = base_velocity * max_component_multiplier;

	if (length(total_velocity.xy) > length(fallback_velocity.xy)) {
		total_velocity = fallback_velocity;
	}
	// ---------------------------------------------------

	// Here is where we apply the velocity thresholds, customized by the user.
	total_velocity *= sharp_step(
			params.velocity_lower_threshold,
			params.velocity_upper_threshold,
			length(total_velocity.xy));

	// If the previous position is happening behind the camera's near clip plane, which can happen when the camera moves backwards at high speed,
	// the w component of the projected vector would be negative, and the velocity vector would be flipped.
	// This happens with Godot's native motion vectors as well. We can detect this and flip them back, avoiding
	// crazy artifacts.
	total_velocity.xy = total_velocity.xy * render_size * (view_past_ndc_cache.w < 0 ? -1 : 1);

	// Now we clamp the velocity magnitudes to the tile size. This is a pretty important step that greatly
	// improves stability and robustness. We multiply the tile size by 2 here, because we blur the velocity
	// symmetrically forwards and backwards, so it's radius is half its magnitude.
	// NOTE @sphynx-owner: this clamp also handles the asymptotical behavior of near-clip-plane previous positions' velocities.
	// ---------------------------------------------------
	float clamp_size = params.tile_size * 2;

	clamp_length(total_velocity, total_velocity.xy, clamp_size);
	// ---------------------------------------------------

	// Here is where the intensity parameter is applied, customized by the user.
	total_velocity *= params.motion_blur_intensity;

	// total_velocity up to this point was backwards, because it was derived using UV differences, which were vectors
	// pointing to the previous UV, meaning the velocity of the pixel is in the other direction.
	vec4 final_output = vec4(-total_velocity, depth);

	imageStore(vector_output, uvi, final_output);
}
