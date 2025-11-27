///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// See `gtao.glsl` for details
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#[compute]

#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_ARB_shader_group_vote : enable

#VERSION_DEFINES


// This push_constant is full - 128 bytes - if you need to add more data, consider adding to the uniform buffer instead
layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int quality;
	bool is_orthogonal;

	vec2 viewport_pixel_size;
	vec2 pad;

	int size_multiplier;
	float fov_scale;
	float depth_linearize_mul;
	float depth_linearize_add;

	vec2 NDC_to_view_mul;
	vec2 NDC_to_view_add;

	float radius;
	float intensity;
	float shadow_power;
	float shadow_clamp;

	float fade_out_mul;
	float fade_out_add;
	float inv_radius_near_limit;
	float thickness_heuristic;
}
params;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_depth;

layout(rgba8, set = 0, binding = 1) uniform restrict readonly image2D source_normal;

layout(r8, set = 0, binding = 2) uniform restrict writeonly image2D dest_working_term;

layout(r8, set = 0, binding = 3) uniform restrict writeonly image2D dest_edges;


shared float scratch_depths[8][8];

// ------------------------------------------------------------------
// Helper: clamp depth to valid float range
// ------------------------------------------------------------------
float clamp_depth(float depth)
{
	// Full float range (same as #else branch in original)
	return clamp(depth, 0.0, 3.402823466e+38);
}

// ------------------------------------------------------------------
// Optional half-precision version (uncomment if using r16f textures)
// float clamp_depth(float depth)
// {
//     return clamp(depth, 0.0, 65504.0);
// }

// ------------------------------------------------------------------
// Weighted average depth filter (used for all MIP levels)
// ------------------------------------------------------------------
float depth_mip_filter(float d0, float d1, float d2, float d3)
{
	float max_depth = max(max(d0, d1), max(d2, d3));

	const float depth_range_scale_factor = 0.75;

	#if 1 // Use runtime constants (equivalent to XE_GTAO_USE_DEFAULT_CONSTANTS == 0)
	float effect_radius  = depth_range_scale_factor * consts.effect_radius * consts.radius_multiplier;
	float falloff_range  = consts.effect_falloff_range * effect_radius;
	#else // Hardcoded defaults (original default path)
	const float EFFECT_RADIUS_MULTIPLIER = 1.457; // XE_GTAO_DEFAULT_RADIUS_MULTIPLIER
	const float EFFECT_FALLOFF_RANGE    = 0.615; // XE_GTAO_DEFAULT_FALLOFF_RANGE
	float effect_radius  = depth_range_scale_factor * consts.effect_radius * EFFECT_RADIUS_MULTIPLIER;
	float falloff_range  = EFFECT_FALLOFF_RANGE * effect_radius;
	#endif

	float falloff_from = effect_radius * (1.0 - consts.effect_falloff_range);
	float falloff_mul  = -1.0 / falloff_range;
	float falloff_add  = falloff_from / falloff_range + 1.0;

	float w0 = clamp((max_depth - d0) * falloff_mul + falloff_add, 0.0, 1.0);
	float w1 = clamp((max_depth - d1) * falloff_mul + falloff_add, 0.0, 1.0);
	float w2 = clamp((max_depth - d2) * falloff_mul + falloff_add, 0.0, 1.0);
	float w3 = clamp((max_depth - d3) * falloff_mul + falloff_add, 0.0, 1.0);

	float weight_sum = w0 + w1 + w2 + w3 + 1e-6; // avoid div0
	return (w0*d0 + w1*d1 + w2*d2 + w3*d3) / weight_sum;
}

// ------------------------------------------------------------------
// Main compute shader
// ------------------------------------------------------------------
void main()
{
	uvec2 dispatch_thread_id = gl_GlobalInvocationID.xy;
	uvec2 group_thread_id    = gl_LocalInvocationID.xy;

	// ------------------------------------------------------------------
	// MIP 0 – sample 2×2 block from source depth
	// ------------------------------------------------------------------
	uvec2 pix_coord = dispatch_thread_id * 2u;

	// Manual GatherRed equivalent (GLSL has no built-in Gather for float textures)
	vec2 uv = (vec2(pix_coord) + vec2(0.5)) * consts.viewport_pixel_size;

	float d00 = textureLod(source_ndc_depth, uv + vec2(+0.5, +0.5) * consts.viewport_pixel_size, 0.0).r;
	float d10 = textureLod(source_ndc_depth, uv + vec2(-0.5, +0.5) * consts.viewport_pixel_size, 0.0).r;
	float d01 = textureLod(source_ndc_depth, uv + vec2(+0.5, -0.5) * consts.viewport_pixel_size, 0.0).r;
	float d11 = textureLod(source_ndc_depth, uv + vec2(-0.5, -0.5) * consts.viewport_pixel_size, 0.0).r;

	// NOTE: You must implement this function or inline linear depth conversion
	// float screen_space_to_view_space_depth(float ndc_depth, GtaoConstants c);
	float depth0 = clamp_depth(screen_space_to_view_space_depth(d00, consts));
	float depth1 = clamp_depth(screen_space_to_view_space_depth(d10, consts));
	float depth2 = clamp_depth(screen_space_to_view_space_depth(d01, consts));
	float depth3 = clamp_depth(screen_space_to_view_space_depth(d11, consts));

	// Write full-res depth (MIP 0)
	if (pix_coord.x < imageSize(out_depth_0).x && pix_coord.y < imageSize(out_depth_0).y)
	{
		imageStore(out_depth_0, ivec2(pix_coord + uvec2(0,0)), vec4(depth0));
		imageStore(out_depth_0, ivec2(pix_coord + uvec2(1,0)), vec4(depth1));
		imageStore(out_depth_0, ivec2(pix_coord + uvec2(0,1)), vec4(depth2));
		imageStore(out_depth_0, ivec2(pix_coord + uvec2(1,1)), vec4(depth3));
	}

	// ------------------------------------------------------------------
	// MIP 1 – 2×2 → 1 pixel
	// ------------------------------------------------------------------
	float dm1 = depth_mip_filter(depth0, depth1, depth2, depth3);
	imageStore(out_depth_1, ivec2(dispatch_thread_id), vec4(dm1));

	scratch_depths[group_thread_id.x][group_thread_id.y] = dm1;

	groupMemoryBarrier();
	barrier();

	// ------------------------------------------------------------------
	// MIP 2 – 4×4 → 1 pixel
	// ------------------------------------------------------------------
	if ((group_thread_id.x & 1u) == 0u && (group_thread_id.y & 1u) == 0u)
	{
		float in_tl = scratch_depths[group_thread_id.x + 0u][group_thread_id.y + 0u];
		float in_tr = scratch_depths[group_thread_id.x + 1u][group_thread_id.y + 0u];
		float in_bl = scratch_depths[group_thread_id.x + 0u][group_thread_id.y + 1u];
		float in_br = scratch_depths[group_thread_id.x + 1u][group_thread_id.y + 1u];

		float dm2 = depth_mip_filter(in_tl, in_tr, in_bl, in_br);
		imageStore(out_depth_2, ivec2(dispatch_thread_id / 2u), vec4(dm2));

		scratch_depths[group_thread_id.x][group_thread_id.y] = dm2;
	}

	groupMemoryBarrier();
	barrier();

	// ------------------------------------------------------------------
	// MIP 3 – 8×8 → 1 pixel
	// ------------------------------------------------------------------
	if ((group_thread_id.x & 3u) == 0u && (group_thread_id.y & 3u) == 0u)
	{
		float in_tl = scratch_depths[group_thread_id.x + 0u][group_thread_id.y + 0u];
		float in_tr = scratch_depths[group_thread_id.x + 2u][group_thread_id.y + 0u];
		float in_bl = scratch_depths[group_thread_id.x + 0u][group_thread_id.y + 2u];
		float in_br = scratch_depths[group_thread_id.x + 2u][group_thread_id.y + 2u];

		float dm3 = depth_mip_filter(in_tl, in_tr, in_bl, in_br);
		imageStore(out_depth_3, ivec2(dispatch_thread_id / 4u), vec4(dm3));

		scratch_depths[group_thread_id.x][group_thread_id.y] = dm3;
	}

	groupMemoryBarrier();
	barrier();

	// ------------------------------------------------------------------
	// MIP 4 – 16×16 → 1 pixel
	// ------------------------------------------------------------------
	if ((group_thread_id.x & 7u) == 0u && (group_thread_id.y & 7u) == 0u)
	{
		float in_tl = scratch_depths[group_thread_id.x + 0u][group_thread_id.y + 0u];
		float in_tr = scratch_depths[group_thread_id.x + 4u][group_thread_id.y + 0u];
		float in_bl = scratch_depths[group_thread_id.x + 0u][group_thread_id.y + 4u];
		float in_br = scratch_depths[group_thread_id.x + 4u][group_thread_id.y + 4u];

		float dm4 = depth_mip_filter(in_tl, in_tr, in_bl, in_br);
		imageStore(out_depth_4, ivec2(dispatch_thread_id / 8u), vec4(dm4));
	}
}
