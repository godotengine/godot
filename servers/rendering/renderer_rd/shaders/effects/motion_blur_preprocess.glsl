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
// Original file link: https://github.com/sphynx-owner/godot-motion-blur-addon-simplified/blob/master/addons/sphynx_motion_blur_toolkit/pre_blur_processing/shader_stages/shaders/pre_blur_processor.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

#define MAX_VIEWS 2

#include "../scene_data_inc.glsl"

layout(set = 0, binding = 0) uniform sampler2D depth_sampler;
layout(set = 0, binding = 1) uniform sampler2D vector_sampler;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D vector_output;
// layout(set = 0, binding = 4) uniform sampler2D stencil_texture;

layout(set = 0, binding = 5, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene;

layout(push_constant, std430) uniform Params {
	float rotation_velocity_multiplier;
	float movement_velocity_multiplier;
	float object_velocity_multiplier;
	float rotation_velocity_lower_threshold;

	float movement_velocity_lower_threshold;
	float object_velocity_lower_threshold;
	float rotation_velocity_upper_threshold;
	float movement_velocity_upper_threshold;

	float object_velocity_upper_threshold;
	float support_fsr2;
	float motion_blur_intensity;
	float pad;
}
params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

float sharp_step(float lower, float upper, float x) {
	return clamp((x - lower) / (upper - lower), 0, 1);
}

float get_view_depth(float depth) {
	return 0.;
}

void main() {
	ivec2 render_size = ivec2(textureSize(vector_sampler, 0));
	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);
	if ((uvi.x >= render_size.x) || (uvi.y >= render_size.y)) {
		return;
	}

	// must be on pixel center for whole values (tested)
	vec2 uvn = vec2(uvi + vec2(0.5)) / render_size;

	SceneData scene_data = scene.data;

	SceneData previous_scene_data = scene.prev_data;

	float depth = textureLod(depth_sampler, uvn, 0.0).x;

	vec4 view_position = inverse(scene_data.projection_matrix) * vec4(uvn * 2.0 - 1.0, depth, 1.0);

	view_position.xyz /= view_position.w;

	mat4 read_view_matrix = transpose(mat4(scene_data.view_matrix[0],
			scene_data.view_matrix[1],
			scene_data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

	// get full change
	vec4 world_local_position = inverse(read_view_matrix) * vec4(view_position.xyz, 1.0);

	mat4 read_prev_view_matrix = transpose(mat4(previous_scene_data.view_matrix[0],
			previous_scene_data.view_matrix[1],
			previous_scene_data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

	vec4 view_past_position = read_prev_view_matrix * vec4(world_local_position.xyz, 1.0);

	vec4 view_past_ndc = previous_scene_data.projection_matrix * view_past_position;

	view_past_ndc.xyz /= view_past_ndc.w;

	vec3 past_uv = vec3(view_past_ndc.xy * 0.5 + 0.5, view_past_position.z);

	vec4 view_past_ndc_cache = view_past_ndc;

	vec3 camera_uv_change = past_uv - vec3(uvn, view_position.z);

	// get just rotation change
	world_local_position = mat4(mat3(inverse(read_view_matrix))) * vec4(view_position.xyz, 1.0);

	view_past_position = mat4(mat3(read_prev_view_matrix)) * vec4(world_local_position.xyz, 1.0);

	view_past_ndc = previous_scene_data.projection_matrix * view_past_position;

	view_past_ndc.xyz /= view_past_ndc.w;

	past_uv = vec3(view_past_ndc.xy * 0.5 + 0.5, view_past_position.z);

	vec3 camera_rotation_uv_change = past_uv - vec3(uvn, view_position.z);

	// get just movement change
	vec3 camera_movement_uv_change = camera_uv_change - camera_rotation_uv_change;

	// fill in gaps in base velocity (skybox, z velocity)
	vec3 base_velocity = vec3(
			textureLod(vector_sampler, uvn, 0.0).xy +
					mix(vec2(0), camera_uv_change.xy, step(depth, 0.)),
			depth == 0 ? 0 : camera_uv_change.z);

	// fsr just makes it so values are larger than 1, I assume its the only case when it happens
	if (params.support_fsr2 > 0.5 && dot(base_velocity.xy, base_velocity.xy) >= 1) {
		base_velocity = camera_uv_change;
	}

	// get object velocity
	vec3 object_uv_change = base_velocity - camera_uv_change.xyz;

	// construct final velocity with user defined weights
	vec3 total_velocity =

			camera_rotation_uv_change * params.rotation_velocity_multiplier *
					sharp_step(params.rotation_velocity_lower_threshold, params.rotation_velocity_upper_threshold,
							length(camera_rotation_uv_change.xy) * params.rotation_velocity_multiplier * params.motion_blur_intensity)

			+ camera_movement_uv_change * params.movement_velocity_multiplier *
					sharp_step(params.movement_velocity_lower_threshold, params.movement_velocity_upper_threshold,
							length(camera_movement_uv_change.xy) * params.movement_velocity_multiplier * params.motion_blur_intensity)

			+ object_uv_change * params.object_velocity_multiplier *
					sharp_step(params.object_velocity_lower_threshold, params.object_velocity_upper_threshold,
							length(object_uv_change.xy) * params.object_velocity_multiplier * params.motion_blur_intensity);

	// if objects move, clear z direction, (velocity z can only be assumed for static environment)
	if (dot(object_uv_change.xy, object_uv_change.xy) > 0.000001) {
		total_velocity.z = 0;
		base_velocity.z = 0;
	}

	// choose the smaller option out of the two based on magnitude, seems to work well
	if (dot(total_velocity.xy * 99, total_velocity.xy * 100) >= dot(base_velocity.xy * 100, base_velocity.xy * 100)) {
		total_velocity = base_velocity;
	}

	float total_velocity_length = max(FLT_MIN, length(total_velocity.xy));
	total_velocity.xy /= max(total_velocity_length, 1);

	float enable_velocity = 1; //step(textureLod(stencil_texture, uvn, 0.0).x, 0.5);

	// If the previous position is happening behind the camera, the w component of the projected vector would be negative,
	// and the velocity vector would be flipped. (I am not 100% sure this is the whole story but this handles velocities
	// that are extracted from the environment when the camera moves backwards rapidly, avoiding crazy artifacts)
	// If degth == 0 (skybox), we use an arithmetic operation to generate a negative infinity float.
	imageStore(vector_output, uvi, vec4(enable_velocity * total_velocity.xy / scene_data.screen_pixel_size * (view_past_ndc_cache.w < 0 ? -1 : 1), enable_velocity * total_velocity.z, depth == 0 ? (-1.0 / 0.0) : view_position.z));
}
