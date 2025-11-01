#[vertex]

#version 450

#VERSION_DEFINES

#if defined(USE_MULTIVIEW)
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#endif // USE_MULTIVIEW

#ifdef USE_MULTIVIEW
layout(location = 0) out vec3 uv_interp;
#else // USE_MULTIVIEW
layout(location = 0) out vec2 uv_interp;
#endif //USE_MULTIVIEW

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp.xy = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
#ifdef USE_MULTIVIEW
	uv_interp.z = ViewIndex;
#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

#ifdef USE_MULTIVIEW
layout(location = 0) in vec3 uv_interp;
#else // USE_MULTIVIEW
layout(location = 0) in vec2 uv_interp;
#endif //USE_MULTIVIEW

#ifdef USE_MULTIVIEW
layout(set = 0, binding = 0) uniform sampler2DArray specular;
#else // USE_MULTIVIEW
layout(set = 0, binding = 0) uniform sampler2D specular;
#endif //USE_MULTIVIEW

#ifdef MODE_SSR

#ifdef USE_MULTIVIEW
layout(set = 1, binding = 0) uniform sampler2DArray ssr;
#else // USE_MULTIVIEW
layout(set = 1, binding = 0) uniform sampler2D ssr;
#endif //USE_MULTIVIEW

#endif

#ifdef MODE_MERGE

#ifdef USE_MULTIVIEW
layout(set = 2, binding = 0) uniform sampler2DArray diffuse;
#else // USE_MULTIVIEW
layout(set = 2, binding = 0) uniform sampler2D diffuse;
#endif //USE_MULTIVIEW

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
