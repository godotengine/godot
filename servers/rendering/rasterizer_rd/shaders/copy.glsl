/* clang-format off */
[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
/* clang-format on */

#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_BLUR_SECTION (1 << 1)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 2)
#define FLAG_DOF_NEAR_FIRST_TAP (1 << 3)
#define FLAG_GLOW_FIRST_PASS (1 << 4)
#define FLAG_FLIP_Y (1 << 5)
#define FLAG_FORCE_LUMINANCE (1 << 6)
#define FLAG_COPY_ALL_SOURCE (1 << 7)

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
}
params;

layout(set = 0, binding = 0) uniform sampler2D source_color;

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

void main() {

	// Pixel being shaded
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThan(pos, params.section.zw))) { //too large, do nothing
		return;
	}

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

	//Glow uses larger sigma 1 for a more rounded blur effect

#define GLOW_ADD(m_ofs, m_mult)                                                             \
	{                                                                                       \
		ivec2 ofs = base_pos + m_ofs;                                                       \
		if (all(greaterThanEqual(ofs, section_begin)) && all(lessThan(ofs, section_end))) { \
			color += texelFetch(source_color, ofs, 0) * m_mult;                             \
		}                                                                                   \
	}

	vec4 color = vec4(0.0);

	if (bool(params.flags & FLAG_HORIZONTAL)) {

		ivec2 base_pos = (pos + params.section.xy) << 1;
		ivec2 section_begin = params.section.xy << 1;
		ivec2 section_end = section_begin + (params.section.zw << 1);

		GLOW_ADD(ivec2(0, 0), 0.174938);
		GLOW_ADD(ivec2(1, 0), 0.165569);
		GLOW_ADD(ivec2(2, 0), 0.140367);
		GLOW_ADD(ivec2(3, 0), 0.106595);
		GLOW_ADD(ivec2(-1, 0), 0.165569);
		GLOW_ADD(ivec2(-2, 0), 0.140367);
		GLOW_ADD(ivec2(-3, 0), 0.106595);
		color *= params.glow_strength;
	} else {

		ivec2 base_pos = pos + params.section.xy;
		ivec2 section_begin = params.section.xy;
		ivec2 section_end = section_begin + params.section.zw;

		GLOW_ADD(ivec2(0, 0), 0.288713);
		GLOW_ADD(ivec2(0, 1), 0.233062);
		GLOW_ADD(ivec2(0, 2), 0.122581);
		GLOW_ADD(ivec2(0, -1), 0.233062);
		GLOW_ADD(ivec2(0, -2), 0.122581);
		color *= params.glow_strength;
	}

#undef GLOW_ADD

	if (bool(params.flags & FLAG_GLOW_FIRST_PASS)) {
#ifdef GLOW_USE_AUTO_EXPOSURE

		color /= texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / params.glow_auto_exposure_grey;
#endif
		color *= params.glow_exposure;

		float luminance = max(color.r, max(color.g, color.b));
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

		if (bool(params.flags & FLAG_FORCE_LUMINANCE)) {
			color.rgb = vec3(max(max(color.r, color.g), color.b));
		}
		imageStore(dest_buffer, pos + params.target, color);

	} else {
		color = texelFetch(source_color, pos + params.section.xy, 0);

		if (bool(params.flags & FLAG_FORCE_LUMINANCE)) {
			color.rgb = vec3(max(max(color.r, color.g), color.b));
		}

		if (bool(params.flags & FLAG_FLIP_Y)) {
			pos.y = params.section.w - pos.y - 1;
		}

		imageStore(dest_buffer, pos + params.target, color);
	}

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
}
