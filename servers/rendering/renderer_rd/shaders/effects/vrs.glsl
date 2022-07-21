#[vertex]

#version 450

#VERSION_DEFINES

#ifdef MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#endif //MULTIVIEW

#ifdef MULTIVIEW
layout(location = 0) out vec3 uv_interp;
#else
layout(location = 0) out vec2 uv_interp;
#endif

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp.xy = base_arr[gl_VertexIndex];
#ifdef MULTIVIEW
	uv_interp.z = ViewIndex;
#endif

	gl_Position = vec4(uv_interp.xy * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#ifdef MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#endif //MULTIVIEW

#ifdef MULTIVIEW
layout(location = 0) in vec3 uv_interp;
layout(set = 0, binding = 0) uniform sampler2DArray source_color;
#else /* MULTIVIEW */
layout(location = 0) in vec2 uv_interp;
layout(set = 0, binding = 0) uniform sampler2D source_color;
#endif /* MULTIVIEW */

layout(location = 0) out uint frag_color;

void main() {
#ifdef MULTIVIEW
	vec3 uv = uv_interp;
#else
	vec2 uv = uv_interp;
#endif

#ifdef MULTIVIEW
	vec4 color = textureLod(source_color, uv, 0.0);
#else /* MULTIVIEW */
	vec4 color = textureLod(source_color, uv, 0.0);
#endif /* MULTIVIEW */

	// See if we can change the sampler to one that returns int...
	frag_color = uint(color.r * 256.0);
}
