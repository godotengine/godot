/* clang-format off */
[vertex]

#version 450

VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

layout(push_constant, binding = 1, std430) uniform Params {
	mat3 orientation;
	vec4 proj;
	vec4 position_multiplier;
	float time;
}
params;

void main() {

	vec2 base_arr[4] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0), vec2(1.0, -1.0));
	uv_interp = base_arr[gl_VertexIndex];
	gl_Position = vec4(uv_interp, 1.0, 1.0);
}

/* clang-format off */
[fragment]

#version 450

VERSION_DEFINES

#define M_PI 3.14159265359

layout(location = 0) in vec2 uv_interp;
/* clang-format on */

layout(push_constant, binding = 1, std430) uniform Params {
	mat3 orientation;
	vec4 proj;
	vec4 position_multiplier;
	float time; //TODO consider adding vec2 screen res, and float radiance size
}
params;

layout(set = 0, binding = 0) uniform sampler material_samplers[12];

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 1, binding = 0, std140) uniform MaterialUniforms{
	/* clang-format off */

MATERIAL_UNIFORMS

	/* clang-format on */
} material;
#endif

layout(set = 2, binding = 0) uniform textureCube radiance;
layout(set = 2, binding = 1) uniform texture2D half_res;
layout(set = 2, binding = 2) uniform texture2D quarter_res;

struct DirectionalLightData {
	vec3 direction;
	float energy;
	vec3 color;
	bool enabled;
};

layout(set = 3, binding = 0, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

/* clang-format off */

FRAGMENT_SHADER_GLOBALS

/* clang-format on */

layout(location = 0) out vec4 frag_color;

void main() {

	vec3 cube_normal;
	cube_normal.z = -1.0;
	cube_normal.x = (cube_normal.z * (-uv_interp.x - params.proj.x)) / params.proj.y;
	cube_normal.y = -(cube_normal.z * (-uv_interp.y - params.proj.z)) / params.proj.w;
	cube_normal = mat3(params.orientation) * cube_normal;
	cube_normal.z = -cube_normal.z;
	cube_normal = normalize(cube_normal);

	vec2 uv = uv_interp * 0.5 + 0.5;

	vec2 panorama_coords = vec2(atan(cube_normal.x, cube_normal.z), acos(cube_normal.y));

	if (panorama_coords.x < 0.0) {
		panorama_coords.x += M_PI * 2.0;
	}

	panorama_coords /= vec2(M_PI * 2.0, M_PI);

	vec3 color = vec3(0.0, 0.0, 0.0);
	float alpha = 1.0; // Only available to subpasses

// unused, just here to make our compiler happy, make sure we don't execute any light code the user adds in..
#ifndef REALLYINCLUDETHIS
	{
		/* clang-format off */

LIGHT_SHADER_CODE

		/* clang-format on */
	}
#endif
	{
		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */
	}

	frag_color.rgb = color * params.position_multiplier.w;
	frag_color.a = alpha;
}
