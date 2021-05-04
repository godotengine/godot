/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */
layout(location = 4) in vec2 uv_in;

out vec2 uv_interp;

void main() {
	uv_interp = uv_in;
	gl_Position = vertex_attrib;
}

/* clang-format off */
[fragment]

#if !defined(GLES_OVER_GL)
precision mediump float;
#endif
/* clang-format on */

in vec2 uv_interp;
uniform sampler2D source_specular; // texunit:0
uniform sampler2D source_ssr; // texunit:1

uniform vec2 pixel_size;

in vec2 uv2_interp;

layout(location = 0) out vec4 frag_color;

void main() {
	vec4 specular = texture(source_specular, uv_interp);

#ifdef USE_SSR
	vec4 ssr = textureLod(source_ssr, uv_interp, 0.0);
	specular.rgb = mix(specular.rgb, ssr.rgb * specular.a, ssr.a);
#endif

	frag_color = vec4(specular.rgb, 1.0);
}
