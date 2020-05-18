#[vertex]

#version 450

VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];

	gl_Position = vec4(uv_interp * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform sampler2D specular;

#ifdef MODE_SSR

layout(set = 1, binding = 0) uniform sampler2D ssr;

#endif

#ifdef MODE_MERGE

layout(set = 2, binding = 0) uniform sampler2D diffuse;

#endif

layout(location = 0) out vec4 frag_color;

void main() {
	frag_color.rgb = texture(specular, uv_interp).rgb;
	frag_color.a = 0.0;
#ifdef MODE_SSR

	vec4 ssr_color = texture(ssr, uv_interp);
	frag_color.rgb = mix(frag_color.rgb, ssr_color.rgb, ssr_color.a);
#endif

#ifdef MODE_MERGE
	frag_color += texture(diffuse, uv_interp);
#endif
	//added using additive blend
}
