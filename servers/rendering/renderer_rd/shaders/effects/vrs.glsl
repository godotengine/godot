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
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp.xy = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
#ifdef MULTIVIEW
	uv_interp.z = ViewIndex;
#endif
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
	frag_color = uint(color.r * 255.0);
#else /* MULTIVIEW */
	vec4 color = textureLod(source_color, uv, 0.0);

	// for user supplied VRS map we do a color mapping
	color.r *= 3.0;
	frag_color = int(color.r) << 2;

	color.g *= 3.0;
	frag_color += int(color.g);

	// note 1x4, 4x1, 1x8, 8x1, 2x8 and 8x2 are not supported
	// 4x8, 8x4 and 8x8 are only available on some GPUs
#endif /* MULTIVIEW */
}
