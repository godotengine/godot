#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp.xy = base_arr[gl_VertexIndex];
	uv_interp.y = 1.0 - uv_interp.y; // invert Y
	gl_Position = vec4(uv_interp.xy * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(set = 0, binding = 0) uniform sampler2D static_shadow_map;
layout(set = 1, binding = 0) uniform sampler2D dynamic_shadow_map;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

void main() {
	float static_depth = textureLod(static_shadow_map, uv_interp, 0.0).r;
	float dynamic_depth = textureLod(dynamic_shadow_map, uv_interp, 0.0).r;
	if (static_depth <= dynamic_depth) {
		vec3 static_color = vec3(static_depth);
		frag_color = vec4(static_color, 1.0);
	} else {
		vec3 dynamic_color = vec3(0.25, 0.25, 0.5 + 0.5 * (1.0 - dynamic_depth));
		frag_color = vec4(dynamic_color, 1.0);
	}
}
