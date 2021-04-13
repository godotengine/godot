#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

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

#[fragment]

#version 450

#VERSION_DEFINES

#define M_PI 3.14159265359

layout(location = 0) in vec2 uv_interp;

layout(push_constant, binding = 1, std430) uniform Params {
	mat3 orientation;
	vec4 proj;
	vec4 position_multiplier;
	float time; //TODO consider adding vec2 screen res, and float radiance size
}
params;

#define SAMPLER_NEAREST_CLAMP 0
#define SAMPLER_LINEAR_CLAMP 1
#define SAMPLER_NEAREST_WITH_MIPMAPS_CLAMP 2
#define SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP 3
#define SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_CLAMP 4
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP 5
#define SAMPLER_NEAREST_REPEAT 6
#define SAMPLER_LINEAR_REPEAT 7
#define SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT 8
#define SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT 9
#define SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_REPEAT 10
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT 11

layout(set = 0, binding = 0) uniform sampler material_samplers[12];

layout(set = 0, binding = 1, std430) restrict readonly buffer GlobalVariableData {
	vec4 data[];
}
global_variables;

layout(set = 0, binding = 2, std140) uniform SceneData {
	bool volumetric_fog_enabled;
	float volumetric_fog_inv_length;
	float volumetric_fog_detail_spread;

	float fog_aerial_perspective;

	vec3 fog_light_color;
	float fog_sun_scatter;

	bool fog_enabled;
	float fog_density;

	float z_far;
	uint directional_light_count;
}
scene_data;

struct DirectionalLightData {
	vec4 direction_energy;
	vec4 color_size;
	bool enabled;
};

layout(set = 0, binding = 3, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}

directional_lights;

#ifdef MATERIAL_UNIFORMS_USED
layout(set = 1, binding = 0, std140) uniform MaterialUniforms{
#MATERIAL_UNIFORMS
} material;
#endif

layout(set = 2, binding = 0) uniform textureCube radiance;
#ifdef USE_CUBEMAP_PASS
layout(set = 2, binding = 1) uniform textureCube half_res;
layout(set = 2, binding = 2) uniform textureCube quarter_res;
#else
layout(set = 2, binding = 1) uniform texture2D half_res;
layout(set = 2, binding = 2) uniform texture2D quarter_res;
#endif

layout(set = 3, binding = 0) uniform texture3D volumetric_fog_texture;

#ifdef USE_CUBEMAP_PASS
#define AT_CUBEMAP_PASS true
#else
#define AT_CUBEMAP_PASS false
#endif

#ifdef USE_HALF_RES_PASS
#define AT_HALF_RES_PASS true
#else
#define AT_HALF_RES_PASS false
#endif

#ifdef USE_QUARTER_RES_PASS
#define AT_QUARTER_RES_PASS true
#else
#define AT_QUARTER_RES_PASS false
#endif

#GLOBALS

layout(location = 0) out vec4 frag_color;

vec4 volumetric_fog_process(vec2 screen_uv) {
	vec3 fog_pos = vec3(screen_uv, 1.0);

	return texture(sampler3D(volumetric_fog_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), fog_pos);
}

vec4 fog_process(vec3 view, vec3 sky_color) {
	vec3 fog_color = mix(scene_data.fog_light_color, sky_color, scene_data.fog_aerial_perspective);

	if (scene_data.fog_sun_scatter > 0.001) {
		vec4 sun_scatter = vec4(0.0);
		float sun_total = 0.0;
		for (uint i = 0; i < scene_data.directional_light_count; i++) {
			vec3 light_color = directional_lights.data[i].color_size.xyz * directional_lights.data[i].direction_energy.w;
			float light_amount = pow(max(dot(view, directional_lights.data[i].direction_energy.xyz), 0.0), 8.0);
			fog_color += light_color * light_amount * scene_data.fog_sun_scatter;
		}
	}

	float fog_amount = clamp(1.0 - exp(-scene_data.z_far * scene_data.fog_density), 0.0, 1.0);

	return vec4(fog_color, fog_amount);
}

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
	vec4 half_res_color = vec4(1.0);
	vec4 quarter_res_color = vec4(1.0);
	vec4 custom_fog = vec4(0.0);

#ifdef USE_CUBEMAP_PASS
	vec3 inverted_cube_normal = cube_normal;
	inverted_cube_normal.z *= -1.0;
#ifdef USES_HALF_RES_COLOR
	half_res_color = texture(samplerCube(half_res, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), inverted_cube_normal);
#endif
#ifdef USES_QUARTER_RES_COLOR
	quarter_res_color = texture(samplerCube(quarter_res, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), inverted_cube_normal);
#endif
#else
#ifdef USES_HALF_RES_COLOR
	half_res_color = textureLod(sampler2D(half_res, material_samplers[SAMPLER_LINEAR_CLAMP]), uv, 0.0);
#endif
#ifdef USES_QUARTER_RES_COLOR
	quarter_res_color = textureLod(sampler2D(quarter_res, material_samplers[SAMPLER_LINEAR_CLAMP]), uv, 0.0);
#endif
#endif

	{

#CODE : SKY

	}

	frag_color.rgb = color * params.position_multiplier.w;
	frag_color.a = alpha;

#if !defined(DISABLE_FOG) && !defined(USE_CUBEMAP_PASS)

	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.
	if (scene_data.fog_enabled) {
		vec4 fog = fog_process(cube_normal, frag_color.rgb);
		frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a);
	}

	if (scene_data.volumetric_fog_enabled) {
		vec4 fog = volumetric_fog_process(uv);
		frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a);
	}

	if (custom_fog.a > 0.0) {
		frag_color.rgb = mix(frag_color.rgb, custom_fog.rgb, custom_fog.a);
	}

#endif // DISABLE_FOG

	// Blending is disabled for Sky, so alpha doesn't blend
	// alpha is used for subsurface scattering so make sure it doesn't get applied to Sky
	if (!AT_CUBEMAP_PASS && !AT_HALF_RES_PASS && !AT_QUARTER_RES_PASS) {
		frag_color.a = 0.0;
	}
}
