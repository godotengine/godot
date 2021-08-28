#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

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

	vec2 vpos = uv_interp;
	if (params.use_section) {
		vpos = params.section.xy + vpos * params.section.zw;
	}

	gl_Position = vec4(vpos * 2.0 - 1.0, 0.0, 1.0);

	if (params.flip_y) {
		uv_interp.y = 1.0 - uv_interp.y;
	}
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(push_constant, binding = 1, std430) uniform Params {
	vec4 section;
	vec2 pixel_size;
	bool flip_y;
	bool use_section;

	bool force_luminance;
	bool alpha_to_zero;
	bool srgb;
	uint pad;
}
params;

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform sampler2D source_color;
#ifdef MODE_TWO_SOURCES
layout(set = 1, binding = 0) uniform sampler2D source_color2;
#endif
layout(location = 0) out vec4 frag_color;

vec3 linear_to_srgb(vec3 color) {
	//if going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

void main() {
	vec2 uv = uv_interp;

#ifdef MODE_PANORAMA_TO_DP

	//obtain normal from dual paraboloid uv
#define M_PI 3.14159265359

	float side;
	uv.y = modf(uv.y * 2.0, side);
	side = side * 2.0 - 1.0;
	vec3 normal = vec3(uv * 2.0 - 1.0, 0.0);
	normal.z = 0.5 - 0.5 * ((normal.x * normal.x) + (normal.y * normal.y));
	normal *= -side;
	normal = normalize(normal);

	//now convert normal to panorama uv

	vec2 st = vec2(atan(normal.x, normal.z), acos(normal.y));

	if (st.x < 0.0) {
		st.x += M_PI * 2.0;
	}

	uv = st / vec2(M_PI * 2.0, M_PI);

	if (side < 0.0) {
		//uv.y = 1.0 - uv.y;
		uv = 1.0 - uv;
	}
#endif
	vec4 color = textureLod(source_color, uv, 0.0);
#ifdef MODE_TWO_SOURCES
	color += textureLod(source_color2, uv, 0.0);
#endif
	if (params.force_luminance) {
		color.rgb = vec3(max(max(color.r, color.g), color.b));
	}
	if (params.alpha_to_zero) {
		color.rgb *= color.a;
	}
	if (params.srgb) {
		color.rgb = linear_to_srgb(color.rgb);
	}
	frag_color = color;
}
