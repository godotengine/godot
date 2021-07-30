/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */
layout(location = 4) in vec2 uv_in;

out vec2 uv_interp;

#ifdef USE_BLUR_SECTION

uniform vec4 blur_section;

#endif

void main() {
	uv_interp = uv_in;
	gl_Position = vertex_attrib;
#ifdef USE_BLUR_SECTION

	uv_interp = blur_section.xy + uv_interp * blur_section.zw;
	gl_Position.xy = (blur_section.xy + (gl_Position.xy * 0.5 + 0.5) * blur_section.zw) * 2.0 - 1.0;
#endif
}

/* clang-format off */
[fragment]

#if !defined(GLES_OVER_GL)
precision mediump float;
#endif
/* clang-format on */

in vec2 uv_interp;
uniform sampler2D source_color; //texunit:0

#ifdef SSAO_MERGE
uniform sampler2D source_ssao; //texunit:1
#endif

uniform float lod;
uniform vec2 pixel_size;
uniform float camera_z_far;
uniform float camera_z_near;

layout(location = 0) out vec4 frag_color;

#ifdef SSAO_MERGE

uniform vec4 ssao_color;

#endif

#if defined(GLOW_GAUSSIAN_HORIZONTAL) || defined(GLOW_GAUSSIAN_VERTICAL)

uniform float glow_strength;

#endif

#if defined(DOF_FAR_BLUR) || defined(DOF_NEAR_BLUR)

// Bokeh single pass implementation based on http://tuxedolabs.blogspot.com/2018/05/bokeh-depth-of-field-in-single-pass.html
const float GOLDEN_ANGLE = 2.39996323;

uniform sampler2D dof_source_depth; //texunit:1
uniform float dof_far_begin;
uniform float dof_far_end;
uniform float dof_near_begin;
uniform float dof_near_end;
uniform float dof_radius;
uniform float dof_scale;

float get_blur_depth(vec2 uv) {
	float depth = textureLod(dof_source_depth, uv, 0.0).r;
	depth = depth * 2.0 - 1.0;
#ifdef USE_ORTHOGONAL_PROJECTION
	depth = ((depth + (camera_z_far + camera_z_near) / (camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near)) / 2.0;
#else
	depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth * (camera_z_far - camera_z_near));
#endif

	return depth;
}

float get_blur_size(float depth) {
	// Combine near and far, our depth is unlikely to be in both ranges
	float amount = 1.0;
#ifdef DOF_FAR_BLUR
	amount *= 1.0 - smoothstep(dof_far_begin, dof_far_end, depth);
#endif
#ifdef DOF_NEAR_BLUR
	amount *= smoothstep(dof_near_end, dof_near_begin, depth);
#endif
	amount = 1.0 - amount;

	return amount * dof_radius;
}

#endif

#ifdef GLOW_FIRST_PASS

uniform float exposure;
uniform float white;
uniform highp float luminance_cap;

#ifdef GLOW_USE_AUTO_EXPOSURE

uniform highp sampler2D source_auto_exposure; //texunit:1
uniform highp float auto_exposure_grey;

#endif

uniform float glow_bloom;
uniform float glow_hdr_threshold;
uniform float glow_hdr_scale;

#endif

void main() {
#ifdef GAUSSIAN_HORIZONTAL
	vec2 pix_size = pixel_size;
	pix_size *= 0.5; //reading from larger buffer, so use more samples
	// sigma 2
	vec4 color = textureLod(source_color, uv_interp + vec2(0.0, 0.0) * pix_size, lod) * 0.214607;
	color += textureLod(source_color, uv_interp + vec2(1.0, 0.0) * pix_size, lod) * 0.189879;
	color += textureLod(source_color, uv_interp + vec2(2.0, 0.0) * pix_size, lod) * 0.131514;
	color += textureLod(source_color, uv_interp + vec2(3.0, 0.0) * pix_size, lod) * 0.071303;
	color += textureLod(source_color, uv_interp + vec2(-1.0, 0.0) * pix_size, lod) * 0.189879;
	color += textureLod(source_color, uv_interp + vec2(-2.0, 0.0) * pix_size, lod) * 0.131514;
	color += textureLod(source_color, uv_interp + vec2(-3.0, 0.0) * pix_size, lod) * 0.071303;
	frag_color = color;
#endif

#ifdef GAUSSIAN_VERTICAL
	vec4 color = textureLod(source_color, uv_interp + vec2(0.0, 0.0) * pixel_size, lod) * 0.38774;
	color += textureLod(source_color, uv_interp + vec2(0.0, 1.0) * pixel_size, lod) * 0.24477;
	color += textureLod(source_color, uv_interp + vec2(0.0, 2.0) * pixel_size, lod) * 0.06136;
	color += textureLod(source_color, uv_interp + vec2(0.0, -1.0) * pixel_size, lod) * 0.24477;
	color += textureLod(source_color, uv_interp + vec2(0.0, -2.0) * pixel_size, lod) * 0.06136;
	frag_color = color;
#endif

	//glow uses larger sigma for a more rounded blur effect

#ifdef GLOW_GAUSSIAN_HORIZONTAL
	vec2 pix_size = pixel_size;
	pix_size *= 0.5; //reading from larger buffer, so use more samples
	vec4 color = textureLod(source_color, uv_interp + vec2(0.0, 0.0) * pix_size, lod) * 0.174938;
	color += textureLod(source_color, uv_interp + vec2(1.0, 0.0) * pix_size, lod) * 0.165569;
	color += textureLod(source_color, uv_interp + vec2(2.0, 0.0) * pix_size, lod) * 0.140367;
	color += textureLod(source_color, uv_interp + vec2(3.0, 0.0) * pix_size, lod) * 0.106595;
	color += textureLod(source_color, uv_interp + vec2(-1.0, 0.0) * pix_size, lod) * 0.165569;
	color += textureLod(source_color, uv_interp + vec2(-2.0, 0.0) * pix_size, lod) * 0.140367;
	color += textureLod(source_color, uv_interp + vec2(-3.0, 0.0) * pix_size, lod) * 0.106595;
	color *= glow_strength;
	frag_color = color;
#endif

#ifdef GLOW_GAUSSIAN_VERTICAL
	vec4 color = textureLod(source_color, uv_interp + vec2(0.0, 0.0) * pixel_size, lod) * 0.288713;
	color += textureLod(source_color, uv_interp + vec2(0.0, 1.0) * pixel_size, lod) * 0.233062;
	color += textureLod(source_color, uv_interp + vec2(0.0, 2.0) * pixel_size, lod) * 0.122581;
	color += textureLod(source_color, uv_interp + vec2(0.0, -1.0) * pixel_size, lod) * 0.233062;
	color += textureLod(source_color, uv_interp + vec2(0.0, -2.0) * pixel_size, lod) * 0.122581;
	color *= glow_strength;
	frag_color = color;
#endif

#if defined(DOF_FAR_BLUR) || defined(DOF_NEAR_BLUR)

	vec4 center_color = textureLod(source_color, uv_interp, 0.0);
	vec3 color_accum = center_color.rgb;
	float accum = 1.0;
	float center_depth = get_blur_depth(uv_interp);
	float center_size = get_blur_size(center_depth);

	float radius = dof_scale;
	for (float ang = 0.0; radius < dof_radius; ang += GOLDEN_ANGLE) {
		vec2 uv_adj = uv_interp + vec2(cos(ang), sin(ang)) * pixel_size * radius;

		vec3 sample_color = textureLod(source_color, uv_adj, 0.0).rgb;
		float sample_depth = get_blur_depth(uv_adj);
		float sample_size = get_blur_size(sample_depth);
		if (sample_depth > center_depth) {
			// sample is behind our center fragment, limit contribution
			sample_size = clamp(sample_size, 0.0, center_size * 2.0);
		}

		float m = smoothstep(radius - 0.5, radius + 0.5, sample_size);
		color_accum += mix(color_accum / accum, sample_color, m);
		accum += 1.0;

		radius += dof_scale / radius;
	}

	frag_color.rgb = color_accum / accum;
	frag_color.a = center_color.a;
#endif

#ifdef GLOW_FIRST_PASS

#ifdef GLOW_USE_AUTO_EXPOSURE

	frag_color /= texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / auto_exposure_grey;
#endif
	frag_color *= exposure;

	float luminance = max(frag_color.r, max(frag_color.g, frag_color.b));
	float feedback = max(smoothstep(glow_hdr_threshold, glow_hdr_threshold + glow_hdr_scale, luminance), glow_bloom);

	frag_color = min(frag_color * feedback, vec4(luminance_cap));

#endif

#ifdef SIMPLE_COPY
	vec4 color = textureLod(source_color, uv_interp, 0.0);
	frag_color = color;
#endif

#ifdef SSAO_MERGE

	vec4 color = textureLod(source_color, uv_interp, 0.0);
	float ssao = textureLod(source_ssao, uv_interp, 0.0).r;

	frag_color = vec4(mix(color.rgb, color.rgb * mix(ssao_color.rgb, vec3(1.0), ssao), color.a), 1.0);

#endif
}
