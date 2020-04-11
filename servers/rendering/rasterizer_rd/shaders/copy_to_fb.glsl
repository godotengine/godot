/* clang-format off */
[vertex]

#version 450

VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

layout(push_constant, binding = 1, std430) uniform Params {
	vec4 section;
	vec2 pixel_size;
	bool flip_y;
	bool use_section;
	bool force_luminance;
	uint pad[3];
}
params;

void main() {

	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];

	if (params.use_section) {
		uv_interp = params.section.xy + uv_interp * params.section.zw;
	}

	gl_Position = vec4(uv_interp * 2.0 - 1.0, 0.0, 1.0);

	if (params.flip_y) {
		uv_interp.y = 1.0 - uv_interp.y;
	}
}

/* clang-format off */
[fragment]

#version 450

VERSION_DEFINES

layout(push_constant, binding = 1, std430) uniform Params {
	vec4 section;
	vec2 pixel_size;
	bool flip_y;
	bool use_section;
	bool force_luminance;
	uint pad[3];
} params;


layout(location = 0) in vec2 uv_interp;
/* clang-format on */

layout(set = 0, binding = 0) uniform sampler2D source_color;

layout(location = 0) out vec4 frag_color;

void main() {

	vec4 color = texture(source_color, uv_interp, 0.0);
	if (params.force_luminance) {
		color.rgb = vec3(max(max(color.r, color.g), color.b));
	}
	frag_color = color;
}
