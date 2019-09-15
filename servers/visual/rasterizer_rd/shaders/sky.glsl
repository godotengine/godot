/* clang-format off */
[vertex]

#version 450

VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

layout(push_constant, binding = 1, std430) uniform Params {
	mat3 orientation;
	vec4 proj;
	float multiplier;
	float alpha;
	float depth;
	float pad;
}
params;

void main() {

	vec2 base_arr[4] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0), vec2(1.0, -1.0));
	uv_interp = base_arr[gl_VertexIndex];
	gl_Position = vec4(uv_interp, params.depth, 1.0);
}

/* clang-format off */
[fragment]

#version 450

VERSION_DEFINES

#define M_PI 3.14159265359

layout(location = 0) in vec2 uv_interp;
/* clang-format on */

layout(set = 0, binding = 0) uniform sampler2D source_panorama;

layout(push_constant, binding = 1, std430) uniform Params {
	mat3 orientation;
	vec4 proj;
	float multiplier;
	float alpha;
	float depth;
	float pad;
}
params;

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 3, binding = 0, std140) uniform MaterialUniforms{
	/* clang-format off */

MATERIAL_UNIFORMS

	/* clang-format on */
} material;
#endif

/* clang-format off */

FRAGMENT_SHADER_GLOBALS

/* clang-format on */

vec4 texturePanorama(sampler2D pano, vec3 normal) {

	vec2 st = vec2(
			atan(normal.x, normal.z),
			acos(normal.y));

	if (st.x < 0.0)
		st.x += M_PI * 2.0;

	st /= vec2(M_PI * 2.0, M_PI);

	return texture(pano, st);
}

layout(location = 0) out vec4 frag_color;

void main() {

	vec3 cube_normal;
	cube_normal.z = -1.0;
	cube_normal.x = (cube_normal.z * (-uv_interp.x - params.proj.x)) / params.proj.y;
	cube_normal.y = -(cube_normal.z * (-uv_interp.y - params.proj.z)) / params.proj.w;
	cube_normal = mat3(params.orientation) * cube_normal;
	cube_normal.z = -cube_normal.z;

	vec3 color = vec3(0.0, 0.0, 0.0);

	// unused, just here to make our compiler happy, make sure we don't execute any light code the user adds in..
#ifndef REALLYINCLUDETHIS
	{
		/* clang-format off */

LIGHT_SHADER_CODE

		/* clang-format on */
	}
#endif

	// color = texturePanorama(source_panorama, normalize(cube_normal.xyz)).rgb;
	{
		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */
	}

	frag_color.rgb = color;
	frag_color.a = params.alpha;
}
