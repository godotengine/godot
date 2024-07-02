// effects shader
// 'base_arr[gl_VertexIndex]' does not work on Android Mali-GXXx GPUs and Vulkan API 1.3.xxx
// ==========================================================================================
// previous code:
// vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
// gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
// uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
#define SET_VERTEX_BASE_TRIANGLE()      \
	vec2 vertex_base;                   \
	if (gl_VertexIndex == 0) {          \
		vertex_base = vec2(-1.0, -1.0); \
	} else if (gl_VertexIndex == 1) {   \
		vertex_base = vec2(-1.0, 3.0);  \
	} else {                            \
		vertex_base = vec2(3.0, -1.0);  \
	}

#define SET_POSITION_AND_UV()                  \
	SET_VERTEX_BASE_TRIANGLE();                \
	gl_Position = vec4(vertex_base, 0.0, 1.0); \
	uv_interp = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0

// sky.glsl
// 'array[gl_VertexIndex]' maybe buggy on some devices or driver versions
// vec2 base_arr[3] = vec2[](vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
#define SET_VERTEX_BASE_SKY_TRIANGLE()  \
	vec2 vertex_base;                   \
	if (gl_VertexIndex == 0) {          \
		vertex_base = vec2(-1.0, -3.0); \
	} else if (gl_VertexIndex == 1) {   \
		vertex_base = vec2(-1.0, 1.0);  \
	} else {                            \
		vertex_base = vec2(3.0, 1.0);   \
	}

// blit.glsl, canvas.glsl, copy_to_fb.glsl, cube_to_dp.glsl
// 'array[gl_VertexIndex]' maybe buggy on some devices or driver versions
// vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
#define SET_VERTEX_BASE_QUAD()        \
	vec2 vertex_base;                 \
	if (gl_VertexIndex == 0) {        \
		vertex_base = vec2(0.0, 0.0); \
	} else if (gl_VertexIndex == 1) { \
		vertex_base = vec2(0.0, 1.0); \
	} else if (gl_VertexIndex == 2) { \
		vertex_base = vec2(1.0, 1.0); \
	} else {                          \
		vertex_base = vec2(1.0, 0.0); \
	}
