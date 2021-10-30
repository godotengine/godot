#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_BLUR_SECTION (1 << 1)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 2)
#define FLAG_DOF_NEAR_FIRST_TAP (1 << 3)
#define FLAG_GLOW_FIRST_PASS (1 << 4)
#define FLAG_FLIP_Y (1 << 5)
#define FLAG_FORCE_LUMINANCE (1 << 6)
#define FLAG_COPY_ALL_SOURCE (1 << 7)
#define FLAG_HIGH_QUALITY_GLOW (1 << 8)
#define FLAG_ALPHA_TO_ONE (1 << 9)

layout(push_constant, binding = 1, std430) uniform Params {
	ivec4 section;
	ivec2 target;
	uint flags;
	uint pad;
	// Glow.
	float glow_strength;
	float glow_bloom;
	float glow_hdr_threshold;
	float glow_hdr_scale;

	float glow_exposure;
	float glow_white;
	float glow_luminance_cap;
	float glow_auto_exposure_grey;
	// DOF.
	float camera_z_far;
	float camera_z_near;
	uint pad2[2];

	vec4 set_color;
}
params;

#ifdef MODE_CUBEMAP_ARRAY_TO_PANORAMA
layout(set = 0, binding = 0) uniform samplerCubeArray source_color;
#elif defined(MODE_CUBEMAP_TO_PANORAMA)
layout(set = 0, binding = 0) uniform samplerCube source_color;
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
layout(rgba32f, set = 3, binding = 0) uniform restrict writeonly image2D dest_buffer;
#endif

#ifdef MODE_GAUSSIAN_GLOW
shared vec4 local_cache[256];
shared vec4 temp_cache[128];
#endif

void main() {
	// Pixel being shaded
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

#ifndef MODE_GAUSSIAN_GLOW // Glow needs the extra threads
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

	imageStore(dest_buffer, pos + params.target, color);
#endif

#ifdef MODE_GAUSSIAN_BLUR

	//Simpler blur uses SIGMA2 for the gaussian kernel for a stronger effect

	if (bool(params.flags & FLAG_HORIZONTAL)) {
		ivec2 base_pos = (pos + params.section.xy) << 1;
		vec4 color = texelFetch(source_color, base_pos + ivec2(0, 0), 0) * 0.214607;
		color += texelFetch(source_color, base_pos + ivec2(1, 0), 0) * 0.189879;
		color += texelFetch(source_color, base_pos + ivec2(2, 0), 0) * 0.131514;
		color += texelFetch(source_color, base_pos + ivec2(3, 0), 0) * 0.071303;
		color += texelFetch(source_color, base_pos + ivec2(-1, 0), 0) * 0.189879;
		color += texelFetch(source_color, base_pos + ivec2(-2, 0), 0) * 0.131514;
		color += texelFetch(source_color, base_pos + ivec2(-3, 0), 0) * 0.071303;
		imageStore(dest_buffer, pos + params.target, color);
	} else {
		ivec2 base_pos = (pos + params.section.xy);
		vec4 color = texelFetch(source_color, base_pos + ivec2(0, 0), 0) * 0.38774;
		color += texelFetch(source_color, base_pos + ivec2(0, 1), 0) * 0.24477;
		color += texelFetch(source_color, base_pos + ivec2(0, 2), 0) * 0.06136;
		color += texelFetch(source_color, base_pos + ivec2(0, -1), 0) * 0.24477;
		color += texelFetch(source_color, base_pos + ivec2(0, -2), 0) * 0.06136;
		imageStore(dest_buffer, pos + params.target, color);
	}
#endif

#ifdef MODE_GAUSSIAN_GLOW

	// First pass copy texture into 16x16 local memory for every 8x8 thread block
	vec2 quad_center_uv = clamp(vec2(gl_GlobalInvocationID.xy + gl_LocalInvocationID.xy - 3.5) / params.section.zw, vec2(0.5 / params.section.zw), vec2(1.0 - 1.5 / params.section.zw));
	uint dest_index = gl_LocalInvocationID.x * 2 + gl_LocalInvocationID.y * 2 * 16;

	if (bool(params.flags & FLAG_HIGH_QUALITY_GLOW)) {
		vec2 quad_offset_uv = clamp((vec2(gl_GlobalInvocationID.xy + gl_LocalInvocationID.xy - 3.0)) / params.section.zw, vec2(0.5 / params.section.zw), vec2(1.0 - 1.5 / params.section.zw));

		local_cache[dest_index] = (textureLod(source_color, quad_center_uv, 0) + textureLod(source_color, quad_offset_uv, 0)) * 0.5;
		local_cache[dest_index + 1] = (textureLod(source_color, quad_center_uv + vec2(1.0 / params.section.z, 0.0), 0) + textureLod(source_color, quad_offset_uv + vec2(1.0 / params.section.z, 0.0), 0)) * 0.5;
		local_cache[dest_index + 16] = (textureLod(source_color, quad_center_uv + vec2(0.0, 1.0 / params.section.w), 0) + textureLod(source_color, quad_offset_uv + vec2(0.0, 1.0 / params.section.w), 0)) * 0.5;
		local_cache[dest_index + 16 + 1] = (textureLod(source_color, quad_center_uv + vec2(1.0 / params.section.zw), 0) + textureLod(source_color, quad_offset_uv + vec2(1.0 / params.section.zw), 0)) * 0.5;
	} else {
		local_cache[dest_index] = textureLod(source_color, quad_center_uv, 0);
		local_cache[dest_index + 1] = textureLod(source_color, quad_center_uv + vec2(1.0 / params.section.z, 0.0), 0);
		local_cache[dest_index + 16] = textureLod(source_color, quad_center_uv + vec2(0.0, 1.0 / params.section.w), 0);
		local_cache[dest_index + 16 + 1] = textureLod(source_color, quad_center_uv + vec2(1.0 / params.section.zw), 0);
	}
	if (bool(params.flags & FLAG_GLOW_FIRST_PASS)) {
		// Tonemap initial samples to reduce weight of fireflies: https://graphicrants.blogspot.com/2013/12/tone-mapping.html
		local_cache[dest_index] /= 1.0 + dot(local_cache[dest_index].rgb, vec3(0.299, 0.587, 0.114));
		local_cache[dest_index + 1] /= 1.0 + dot(local_cache[dest_index + 1].rgb, vec3(0.299, 0.587, 0.114));
		local_cache[dest_index + 16] /= 1.0 + dot(local_cache[dest_index + 16].rgb, vec3(0.299, 0.587, 0.114));
		local_cache[dest_index + 16 + 1] /= 1.0 + dot(local_cache[dest_index + 16 + 1].rgb, vec3(0.299, 0.587, 0.114));
	}

	memoryBarrierShared();
	barrier();

	// Horizontal pass. Needs to copy into 8x16 chunk of local memory so vertical pass has full resolution
	uint read_index = gl_LocalInvocationID.x + gl_LocalInvocationID.y * 32 + 4;
	vec4 color_top = vec4(0.0);
	color_top += local_cache[read_index] * 0.174938;
	color_top += local_cache[read_index + 1] * 0.165569;
	color_top += local_cache[read_index + 2] * 0.140367;
	color_top += local_cache[read_index + 3] * 0.106595;
	color_top += local_cache[read_index - 1] * 0.165569;
	color_top += local_cache[read_index - 2] * 0.140367;
	color_top += local_cache[read_index - 3] * 0.106595;

	vec4 color_bottom = vec4(0.0);
	color_bottom += local_cache[read_index + 16] * 0.174938;
	color_bottom += local_cache[read_index + 1 + 16] * 0.165569;
	color_bottom += local_cache[read_index + 2 + 16] * 0.140367;
	color_bottom += local_cache[read_index + 3 + 16] * 0.106595;
	color_bottom += local_cache[read_index - 1 + 16] * 0.165569;
	color_bottom += local_cache[read_index - 2 + 16] * 0.140367;
	color_bottom += local_cache[read_index - 3 + 16] * 0.106595;

	// rotate samples to take advantage of cache coherency
	uint write_index = gl_LocalInvocationID.y * 2 + gl_LocalInvocationID.x * 16;

	temp_cache[write_index] = color_top;
	temp_cache[write_index + 1] = color_bottom;

	memoryBarrierShared();
	barrier();

	// Vertical pass
	uint index = gl_LocalInvocationID.y + gl_LocalInvocationID.x * 16 + 4;
	vec4 color = vec4(0.0);

	color += temp_cache[index] * 0.174938;
	color += temp_cache[index + 1] * 0.165569;
	color += temp_cache[index + 2] * 0.140367;
	color += temp_cache[index + 3] * 0.106595;
	color += temp_cache[index - 1] * 0.165569;
	color += temp_cache[index - 2] * 0.140367;
	color += temp_cache[index - 3] * 0.106595;

	if (bool(params.flags & FLAG_GLOW_FIRST_PASS)) {
		// Undo tonemap to restore range: https://graphicrants.blogspot.com/2013/12/tone-mapping.html
		color /= 1.0 - dot(color.rgb, vec3(0.299, 0.587, 0.114));
	}

	color *= params.glow_strength;

	if (bool(params.flags & FLAG_GLOW_FIRST_PASS)) {
#ifdef GLOW_USE_AUTO_EXPOSURE

		color /= texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / params.glow_auto_exposure_grey;
#endif
		color *= params.glow_exposure;

		float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
		float feedback = max(smoothstep(params.glow_hdr_threshold, params.glow_hdr_threshold + params.glow_hdr_scale, luminance), params.glow_bloom);

		color = min(color * feedback, vec4(params.glow_luminance_cap));
	}

	imageStore(dest_buffer, pos + params.target, color);

#endif

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

	imageStore(dest_buffer, pos + params.target, color);

#endif

#ifdef MODE_SIMPLE_COPY_DEPTH

	vec4 color = texelFetch(source_color, pos + params.section.xy, 0);

	if (bool(params.flags & FLAG_FLIP_Y)) {
		pos.y = params.section.w - pos.y - 1;
	}

	imageStore(dest_buffer, pos + params.target, vec4(color.r));

#endif

#ifdef MODE_LINEARIZE_DEPTH_COPY

	float depth = texelFetch(source_color, pos + params.section.xy, 0).r;
	depth = depth * 2.0 - 1.0;
	depth = 2.0 * params.camera_z_near * params.camera_z_far / (params.camera_z_far + params.camera_z_near - depth * (params.camera_z_far - params.camera_z_near));
	vec4 color = vec4(depth / params.camera_z_far);

	if (bool(params.flags & FLAG_FLIP_Y)) {
		pos.y = params.section.w - pos.y - 1;
	}

	imageStore(dest_buffer, pos + params.target, color);
#endif

#if defined(MODE_CUBEMAP_TO_PANORAMA) || defined(MODE_CUBEMAP_ARRAY_TO_PANORAMA)

	const float PI = 3.14159265359;
	vec2 uv = vec2(pos) / vec2(params.section.zw);
	uv.y = 1.0 - uv.y;
	float phi = uv.x * 2.0 * PI;
	float theta = uv.y * PI;

	vec3 normal;
	normal.x = sin(phi) * sin(theta) * -1.0;
	normal.y = cos(theta);
	normal.z = cos(phi) * sin(theta) * -1.0;

#ifdef MODE_CUBEMAP_TO_PANORAMA
	vec4 color = textureLod(source_color, normal, params.camera_z_far); //the biggest the lod the least the acne
#else
	vec4 color = textureLod(source_color, vec4(normal, params.camera_z_far), 0.0); //the biggest the lod the least the acne
#endif
	imageStore(dest_buffer, pos + params.target, color);
#endif

#ifdef MODE_SET_COLOR
	imageStore(dest_buffer, pos + params.target, params.set_color);
#endif
}
