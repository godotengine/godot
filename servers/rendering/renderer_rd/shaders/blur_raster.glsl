/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

#include "blur_raster_inc.glsl"

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];

	gl_Position = vec4(uv_interp * 2.0 - 1.0, 0.0, 1.0);
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

#include "blur_raster_inc.glsl"

layout(location = 0) in vec2 uv_interp;
/* clang-format on */

layout(set = 0, binding = 0) uniform sampler2D source_color;

#ifdef GLOW_USE_AUTO_EXPOSURE
layout(set = 1, binding = 0) uniform sampler2D source_auto_exposure;
#endif

layout(location = 0) out vec4 frag_color;

void main() {
	// We do not apply our color scale for our mobile renderer here, we'll leave our colors at half brightness and apply scale in the tonemap raster.

#ifdef MODE_MIPMAP

	vec2 pix_size = blur.pixel_size;
	vec4 color = texture(source_color, uv_interp + vec2(-0.5, -0.5) * pix_size);
	color += texture(source_color, uv_interp + vec2(0.5, -0.5) * pix_size);
	color += texture(source_color, uv_interp + vec2(0.5, 0.5) * pix_size);
	color += texture(source_color, uv_interp + vec2(-0.5, 0.5) * pix_size);
	frag_color = color / 4.0;

#endif

#ifdef MODE_GAUSSIAN_BLUR

	//Simpler blur uses SIGMA2 for the gaussian kernel for a stronger effect

	if (bool(blur.flags & FLAG_HORIZONTAL)) {
		vec2 pix_size = blur.pixel_size;
		pix_size *= 0.5; //reading from larger buffer, so use more samples
		vec4 color = texture(source_color, uv_interp + vec2(0.0, 0.0) * pix_size) * 0.214607;
		color += texture(source_color, uv_interp + vec2(1.0, 0.0) * pix_size) * 0.189879;
		color += texture(source_color, uv_interp + vec2(2.0, 0.0) * pix_size) * 0.131514;
		color += texture(source_color, uv_interp + vec2(3.0, 0.0) * pix_size) * 0.071303;
		color += texture(source_color, uv_interp + vec2(-1.0, 0.0) * pix_size) * 0.189879;
		color += texture(source_color, uv_interp + vec2(-2.0, 0.0) * pix_size) * 0.131514;
		color += texture(source_color, uv_interp + vec2(-3.0, 0.0) * pix_size) * 0.071303;
		frag_color = color;
	} else {
		vec2 pix_size = blur.pixel_size;
		vec4 color = texture(source_color, uv_interp + vec2(0.0, 0.0) * pix_size) * 0.38774;
		color += texture(source_color, uv_interp + vec2(0.0, 1.0) * pix_size) * 0.24477;
		color += texture(source_color, uv_interp + vec2(0.0, 2.0) * pix_size) * 0.06136;
		color += texture(source_color, uv_interp + vec2(0.0, -1.0) * pix_size) * 0.24477;
		color += texture(source_color, uv_interp + vec2(0.0, -2.0) * pix_size) * 0.06136;
		frag_color = color;
	}
#endif

#ifdef MODE_GAUSSIAN_GLOW

	//Glow uses larger sigma 1 for a more rounded blur effect

#define GLOW_ADD(m_ofs, m_mult)                                                  \
	{                                                                            \
		vec2 ofs = uv_interp + m_ofs * pix_size;                                 \
		vec4 c = texture(source_color, ofs) * m_mult;                            \
		if (any(lessThan(ofs, vec2(0.0))) || any(greaterThan(ofs, vec2(1.0)))) { \
			c *= 0.0;                                                            \
		}                                                                        \
		color += c;                                                              \
	}

	if (bool(blur.flags & FLAG_HORIZONTAL)) {
		vec2 pix_size = blur.pixel_size;
		pix_size *= 0.5; //reading from larger buffer, so use more samples
		vec4 color = texture(source_color, uv_interp + vec2(0.0, 0.0) * pix_size) * 0.174938;
		GLOW_ADD(vec2(1.0, 0.0), 0.165569);
		GLOW_ADD(vec2(2.0, 0.0), 0.140367);
		GLOW_ADD(vec2(3.0, 0.0), 0.106595);
		GLOW_ADD(vec2(-1.0, 0.0), 0.165569);
		GLOW_ADD(vec2(-2.0, 0.0), 0.140367);
		GLOW_ADD(vec2(-3.0, 0.0), 0.106595);
		color *= blur.glow_strength;
		frag_color = color;
	} else {
		vec2 pix_size = blur.pixel_size;
		vec4 color = texture(source_color, uv_interp + vec2(0.0, 0.0) * pix_size) * 0.288713;
		GLOW_ADD(vec2(0.0, 1.0), 0.233062);
		GLOW_ADD(vec2(0.0, 2.0), 0.122581);
		GLOW_ADD(vec2(0.0, -1.0), 0.233062);
		GLOW_ADD(vec2(0.0, -2.0), 0.122581);
		color *= blur.glow_strength;
		frag_color = color;
	}

#undef GLOW_ADD

	if (bool(blur.flags & FLAG_GLOW_FIRST_PASS)) {
#ifdef GLOW_USE_AUTO_EXPOSURE

		frag_color /= texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / blur.glow_auto_exposure_grey;
#endif
		frag_color *= blur.glow_exposure;

		float luminance = max(frag_color.r, max(frag_color.g, frag_color.b));
		float feedback = max(smoothstep(blur.glow_hdr_threshold, blur.glow_hdr_threshold + blur.glow_hdr_scale, luminance), blur.glow_bloom);

		frag_color = min(frag_color * feedback, vec4(blur.glow_luminance_cap));
	}

#endif

#ifdef MODE_COPY
	vec4 color = textureLod(source_color, uv_interp, 0.0);
	frag_color = color;
#endif
}
