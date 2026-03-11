#[compute]

#version 450

#VERSION_DEFINES

#extension GL_KHR_shader_subgroup_ballot : enable

#include "../oct_inc.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_BLUR_SECTION (1 << 1)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 2)
#define FLAG_DOF_NEAR_FIRST_TAP (1 << 3)
#define FLAG_GLOW_FIRST_PASS (1 << 4)
#define FLAG_FLIP_Y (1 << 5)
#define FLAG_FORCE_LUMINANCE (1 << 6)
#define FLAG_COPY_ALL_SOURCE (1 << 7)
#define FLAG_ALPHA_TO_ONE (1 << 8)
#define FLAG_SANITIZE_INF_NAN (1 << 9)

layout(push_constant, std430) uniform Params {
	ivec4 section;
	ivec2 target;
	uint flags;
	float luminance_multiplier;
	// Glow.
	float glow_strength;
	float glow_bloom;
	float glow_hdr_threshold;
	float glow_hdr_scale;

	float glow_exposure;
	float glow_white;
	float glow_luminance_cap;
	float glow_auto_exposure_scale;
	// DOF.
	float camera_z_far;
	float camera_z_near;
	// Octmap.
	vec2 octmap_border_size;

	vec4 set_color;
}
params;

#ifdef MODE_OCTMAP_ARRAY_TO_PANORAMA
layout(set = 0, binding = 0) uniform sampler2DArray source_color;
#elif defined(MODE_OCTMAP_TO_PANORAMA)
layout(set = 0, binding = 0) uniform sampler2D source_color;
#elif !defined(MODE_SET_COLOR)
layout(set = 0, binding = 0) uniform sampler2D source_color;
#endif

#ifdef GLOW_USE_AUTO_EXPOSURE
layout(set = 1, binding = 0) uniform sampler2D source_auto_exposure;
#endif

#if defined(MODE_LINEARIZE_DEPTH_COPY) || defined(MODE_SIMPLE_COPY_DEPTH)
layout(r32f, set = 3, binding = 0) uniform restrict writeonly image2D dest_buffer;
#elif defined(DST_IMAGE_8BIT)
layout(rgba8, set = 3, binding = 0) uniform restrict writeonly image2D dest_buffer;
#else
layout(rgba16f, set = 3, binding = 0) uniform restrict writeonly image2D dest_buffer;
#endif

#ifdef MODE_GAUSSIAN_BLUR
shared vec4 local_cache[256];
shared vec4 temp_cache[128];
#endif

void main() {
	// Pixel being shaded
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

#ifndef MODE_GAUSSIAN_BLUR // Gaussian blur needs the extra threads
	if (any(greaterThanEqual(pos, params.section.zw))) { //too large, do nothing
		return;
	}
#endif

#ifdef MODE_MIPMAP

	ivec2 base_pos = (pos + params.section.xy) << 1;
	vec4 color = texelFetch(source_color, base_pos, 0);
	color += texelFetch(source_color, base_pos + ivec2(0, 1), 0);
	color += texelFetch(source_color, base_pos + ivec2(1, 0), 0);
	color += texelFetch(source_color, base_pos + ivec2(1, 1), 0);
	color /= 4.0;
	color = mix(color, vec4(100.0, 100.0, 100.0, 1.0), isinf(color));
	color = mix(color, vec4(100.0, 100.0, 100.0, 1.0), isnan(color));

	imageStore(dest_buffer, pos + params.target, color);
#endif

#ifdef MODE_GAUSSIAN_BLUR

#ifdef MODE_GLOW
	const vec3 tonemap_col = vec3(0.299, 0.587, 0.114) / max(params.glow_luminance_cap, 6.0);
#endif

	const uint num_subgroups = gl_NumSubgroups;
	const uint subgroup_size = (gl_WorkGroupSize.x * gl_WorkGroupSize.y) / num_subgroups;

	// First pass copy texture into 16x16 local memory for every 8x8 thread block

	// To avoid bank conflicts, linear index "i" in the 16x16 grid will be placed at
	// i_write according to the equation:
	// i_write = i ^ ((i & shuffle_mask) >> 1)

	// Compute optimal shuffle mask for the number of subgroups
	const uint shuffle_mask = (0x70u / num_subgroups) & 0x70u;

	const uvec2 group_top_left = gl_WorkGroupID.xy * gl_WorkGroupSize.xy;
	const uint linear_write_offset = gl_SubgroupInvocationID + gl_SubgroupID * ((16u * 16u) / num_subgroups);

// Each subgroup fetches contiguous memory in the 16x16 block
#pragma unroll 4u
	for (uint b = 0u; b < 4u; b++) {
		// Compute the linear offset of the work item
		const uint linear_index = linear_write_offset + (b * subgroup_size);
		// Extract (x,y) coordinate of sub block
		const uint xi = linear_index & 0xfu;
		const uint yi = linear_index >> 4u;
		// Fetch pixel value
		const vec2 fetch_uv = clamp(
				vec2(params.section.xy + group_top_left + vec2(xi, yi) - 3.5) / params.section.zw,
				vec2(0.5 / params.section.zw), vec2(1.0 - 0.5 / params.section.zw));

		// Shuffle write index to avoid bank conflicts during horizontal blur pass
		const uint store_index = linear_index ^ ((linear_index & shuffle_mask) >> 1u);
		vec4 color = textureLod(source_color, fetch_uv, 0.);

#ifdef MODE_GLOW
		// Tonemap initial samples to reduce weight of fireflies: https://graphicrants.blogspot.com/2013/12/tone-mapping.html
		color = bool(params.flags & FLAG_GLOW_FIRST_PASS) ? color / (1.0 + dot(color.rgb, tonemap_col)) : color;
#endif // MODE_GLOW

		// Store in shuffled index
		local_cache[store_index] = color;
	}

#ifdef MODE_GLOW
#define KERNEL_LENGTH 9u
	const uint kernel_offset = 0u;
	const float kernel[9] = { 0.0285, 0.0672, 0.1240, 0.1790, 0.2024, 0.1790, 0.1240, 0.0672, 0.0285 };
#else
// Simpler blur uses SIGMA2 for the gaussian kernel for a stronger effect.
#define KERNEL_LENGTH 7u
	const uint kernel_offset = 1u;
	const float kernel[7] = { 0.071303, 0.131514, 0.189879, 0.214607, 0.189879, 0.131514, 0.071303 };
#endif

	// Only need to wait on horizontal pass if subgroups fetch less than 2 rows
	if (subgroup_size < 8u) {
		barrier();
	} else {
		subgroupBarrier();
	}

	// Linear index of first computed element in output 16x8 temp_cache (all kernels start on "left")
	const uint linear_start_0 = gl_SubgroupInvocationID + gl_SubgroupID * (2u * subgroup_size);

	vec4 color_0 = vec4(0.);
	// Compute corresponding 16x8 position in the 16x16 local_cache by promoting index at 8-bit
	const uint start_0 = ((linear_start_0 & 0xf8u) << 1u) + (linear_start_0 & 0x7u) + kernel_offset;

#pragma unroll KERNEL_LENGTH
	for (uint k = 0u; k < KERNEL_LENGTH; k++) {
		const uint linear_index = start_0 + k;
		// Shuffle linear index to get stored location
		const uint read_index = linear_index ^ ((linear_index & shuffle_mask) >> 1u);
		// Accumulate horizontal pass
		color_0 += local_cache[read_index] * kernel[k];
	}

	// Stride by subgroup size
	const uint linear_start_1 = linear_start_0 + subgroup_size;
	vec4 color_1 = vec4(0.);
	// Promote 8-bit for second pass
	const uint start_1 = ((linear_start_1 & 0xf8u) << 1u) + (linear_start_1 & 0x7u) + kernel_offset;

#pragma unroll KERNEL_LENGTH
	for (uint k = 0u; k < KERNEL_LENGTH; k++) {
		const uint linear_index = start_1 + k;
		// Shuffle linear index to get stored location
		const uint read_index = linear_index ^ ((linear_index & shuffle_mask) >> 1u);
		// Accumulate second horizontal pass
		color_1 += local_cache[read_index] * kernel[k];
	}

	// Store values at linear 16x8 position
	// Memory is stored and fetched contiguously within subgroups, no risk of bank conflicts
	temp_cache[linear_start_0] = color_0;
	temp_cache[linear_start_1] = color_1;

	// Only need to wait on vertical pass if more than 1 subgroup is present
	if (num_subgroups > 1u) {
		barrier();
	} else {
		subgroupBarrier();
	}

	// If destination outside of texture, can stop doing work now
	if (any(greaterThanEqual(pos, params.section.zw))) {
		return;
	}

	// Vertical pass memory is already contiguous
	const uint result_start_index = gl_LocalInvocationID.x + (gl_LocalInvocationID.y + kernel_offset) * 8u;
	vec4 color = vec4(0.0);

// Compute the vertical pass for the 16x8 elements
#pragma unroll KERNEL_LENGTH
	for (uint k = 0; k < KERNEL_LENGTH; k++) {
		color += temp_cache[result_start_index + 8u * k] * kernel[k];
	}

#ifdef MODE_GLOW
	if (bool(params.flags & FLAG_GLOW_FIRST_PASS)) {
		// Undo tonemap to restore range: https://graphicrants.blogspot.com/2013/12/tone-mapping.html
		color /= 1.0 - dot(color.rgb, tonemap_col);
	}

	color *= params.glow_strength;

	if (bool(params.flags & FLAG_GLOW_FIRST_PASS)) {
#ifdef GLOW_USE_AUTO_EXPOSURE

		color /= texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / params.glow_auto_exposure_scale;
#endif
		color *= params.glow_exposure;

		float luminance = max(color.r, max(color.g, color.b));
		float feedback = max(smoothstep(params.glow_hdr_threshold, params.glow_hdr_threshold + params.glow_hdr_scale, luminance), params.glow_bloom);

		color = min(color * feedback, vec4(params.glow_luminance_cap));
	}
#endif // MODE_GLOW
	imageStore(dest_buffer, pos + params.target, color);

#endif // MODE_GAUSSIAN_BLUR

#ifdef MODE_SIMPLE_COPY

	vec4 color;
	if (bool(params.flags & FLAG_COPY_ALL_SOURCE)) {
		vec2 uv = vec2(pos) / vec2(params.section.zw);
		if (bool(params.flags & FLAG_FLIP_Y)) {
			uv.y = 1.0 - uv.y;
		}
		color = textureLod(source_color, uv, 0.0);

	} else {
		color = texelFetch(source_color, pos + params.section.xy, 0);

		if (bool(params.flags & FLAG_FLIP_Y)) {
			pos.y = params.section.w - pos.y - 1;
		}
	}

	if (bool(params.flags & FLAG_FORCE_LUMINANCE)) {
		color.rgb = vec3(max(max(color.r, color.g), color.b));
	}

	if (bool(params.flags & FLAG_ALPHA_TO_ONE)) {
		color.a = 1.0;
	}

	if (bool(params.flags & FLAG_SANITIZE_INF_NAN)) {
		color = mix(color, vec4(100.0, 100.0, 100.0, 1.0), isinf(color));
		color = mix(color, vec4(100.0, 100.0, 100.0, 1.0), isnan(color));
	}

	imageStore(dest_buffer, pos + params.target, color);

#endif // MODE_SIMPLE_COPY

#ifdef MODE_SIMPLE_COPY_DEPTH

	vec4 color = texelFetch(source_color, pos + params.section.xy, 0);

	if (bool(params.flags & FLAG_FLIP_Y)) {
		pos.y = params.section.w - pos.y - 1;
	}

	imageStore(dest_buffer, pos + params.target, vec4(color.r));

#endif // MODE_SIMPLE_COPY_DEPTH

#ifdef MODE_LINEARIZE_DEPTH_COPY

	float depth = texelFetch(source_color, pos + params.section.xy, 0).r;
	depth = depth * 2.0 - 1.0;
	depth = 2.0 * params.camera_z_near * params.camera_z_far / (params.camera_z_far + params.camera_z_near - depth * (params.camera_z_far - params.camera_z_near));
	vec4 color = vec4(depth / params.camera_z_far);

	if (bool(params.flags & FLAG_FLIP_Y)) {
		pos.y = params.section.w - pos.y - 1;
	}

	imageStore(dest_buffer, pos + params.target, color);
#endif // MODE_LINEARIZE_DEPTH_COPY

#if defined(MODE_OCTMAP_TO_PANORAMA) || defined(MODE_OCTMAP_ARRAY_TO_PANORAMA)

	const float PI = 3.14159265359;
	vec2 uv = vec2(pos) / vec2(params.section.zw);
	if (bool(params.flags & FLAG_FLIP_Y)) {
		uv.y = 1.0 - uv.y;
	}
	float phi = uv.x * 2.0 * PI;
	float theta = uv.y * PI;

	vec3 normal;
	normal.x = sin(phi) * sin(theta) * -1.0;
	normal.y = cos(theta);
	normal.z = cos(phi) * sin(theta) * -1.0;

#ifdef MODE_OCTMAP_TO_PANORAMA
	vec4 color = textureLod(source_color, vec3_to_oct_with_border(normal, params.octmap_border_size), params.camera_z_far); //the biggest the lod the least the acne
#else
	vec4 color = textureLod(source_color, vec3(vec3_to_oct_with_border(normal, params.octmap_border_size), params.camera_z_far), 0.0); //the biggest the lod the least the acne
#endif
	imageStore(dest_buffer, pos + params.target, color * params.luminance_multiplier);
#endif // defined(MODE_OCTMAP_TO_PANORAMA) || defined(MODE_OCTMAP_ARRAY_TO_PANORAMA)

#ifdef MODE_SET_COLOR
	imageStore(dest_buffer, pos + params.target, params.set_color);
#endif
}
