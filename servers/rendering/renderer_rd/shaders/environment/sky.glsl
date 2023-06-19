#[vertex]

#version 450

#VERSION_DEFINES

#include "sky_inc.glsl"

layout(location = 0) in vec3 vertex_attrib;
layout(location = 1) in vec2 uv_attrib;

layout(location = 0) out vec2 uv_interp;
layout(location = 1) out vec3 cube_normal;
layout(location = 2) out vec2 panorama_coords;

void main() {
	// We project a sphere around the camera location.
	// This removes the need for some expensive calculations in the fragment shader,
	// and enables prefetch of panoramic textures on mobile devices.

	panorama_coords = uv_attrib;
	cube_normal = vertex_attrib;

	// On mono the radius of the sphere doesn't matter.
	// On stereo we encorporate an eye offset and it does,
	// we need a sufficiently high value.
	float radius = 100000.0;

#ifdef USE_CUBEMAP_PASS
	// Adjust our orientation
	vec3 vertex = params.cubemap_view_orientation * cube_normal;

	// Project our cube normal.
	vec4 position = sky_scene_data.cubemap_projection * vec4(vertex * radius, 1.0);
#else
	// Adjust our orientation
	vec3 vertex = sky_scene_data.view_orientation * cube_normal;

	// Project our cube normal.
	vec4 position = sky_scene_data.view_projections[ViewIndex] * vec4(vertex * radius, 1.0);
#endif

	// Cap our Z so we're projecting on our far plane
	if (position.w > 0.0 && position.z > position.w) {
		position.z = position.w;
	}

	gl_Position = position;

	// uv_interp should match our screen coords.
	uv_interp = position.xy / position.w;
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "sky_inc.glsl"

layout(location = 0) in vec2 uv_interp;
layout(location = 1) in vec3 cube_normal;
layout(location = 2) in vec2 panorama_coords;

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

layout(set = 0, binding = 1, std430) restrict readonly buffer GlobalShaderUniformData {
	vec4 data[];
}
global_shader_uniforms;

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
#elif defined(USE_MULTIVIEW)
layout(set = 2, binding = 1) uniform texture2DArray half_res;
layout(set = 2, binding = 2) uniform texture2DArray quarter_res;
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

#ifdef USE_DEBANDING
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
vec3 interleaved_gradient_noise(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	float res = fract(magic.z * fract(dot(pos, magic.xy))) * 2.0 - 1.0;
	return vec3(res, -res, res) / 255.0;
}
#endif

vec4 volumetric_fog_process(vec2 screen_uv) {
	vec3 fog_pos = vec3(screen_uv, 1.0);

	return texture(sampler3D(volumetric_fog_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), fog_pos);
}

vec4 fog_process(vec3 view, vec3 sky_color) {
	vec3 fog_color = mix(sky_scene_data.fog_light_color, sky_color, sky_scene_data.fog_aerial_perspective);

	if (sky_scene_data.fog_sun_scatter > 0.001) {
		vec4 sun_scatter = vec4(0.0);
		float sun_total = 0.0;
		for (uint i = 0; i < sky_scene_data.directional_light_count; i++) {
			vec3 light_color = directional_lights.data[i].color_size.xyz * directional_lights.data[i].direction_energy.w;
			float light_amount = pow(max(dot(view, directional_lights.data[i].direction_energy.xyz), 0.0), 8.0);
			fog_color += light_color * light_amount * sky_scene_data.fog_sun_scatter;
		}
	}

	return vec4(fog_color, 1.0);
}

void main() {
	// Change this, uv_interp will now be perspective corrected which is NOT what we want!!
	// Also this calculation prevents the ability to prefetch...
	vec2 uv = uv_interp * 0.5 + 0.5;

	vec3 eye_dir = normalize(cube_normal);

	vec3 color = vec3(0.0, 0.0, 0.0);
	float alpha = 1.0; // Only available to subpasses
	vec4 half_res_color = vec4(1.0);
	vec4 quarter_res_color = vec4(1.0);
	vec4 custom_fog = vec4(0.0);

#ifdef USE_CUBEMAP_PASS

#ifdef USES_HALF_RES_COLOR
	half_res_color = texture(samplerCube(half_res, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), cube_normal) / params.luminance_multiplier;
#endif
#ifdef USES_QUARTER_RES_COLOR
	quarter_res_color = texture(samplerCube(quarter_res, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), cube_normal) / params.luminance_multiplier;
#endif

#else

#ifdef USES_HALF_RES_COLOR
#ifdef USE_MULTIVIEW
	half_res_color = textureLod(sampler2DArray(half_res, material_samplers[SAMPLER_LINEAR_CLAMP]), vec3(uv, ViewIndex), 0.0) / params.luminance_multiplier;
#else
	half_res_color = textureLod(sampler2D(half_res, material_samplers[SAMPLER_LINEAR_CLAMP]), uv, 0.0) / params.luminance_multiplier;
#endif // USE_MULTIVIEW
#endif // USES_HALF_RES_COLOR

#ifdef USES_QUARTER_RES_COLOR
#ifdef USE_MULTIVIEW
	quarter_res_color = textureLod(sampler2DArray(quarter_res, material_samplers[SAMPLER_LINEAR_CLAMP]), vec3(uv, ViewIndex), 0.0) / params.luminance_multiplier;
#else
	quarter_res_color = textureLod(sampler2D(quarter_res, material_samplers[SAMPLER_LINEAR_CLAMP]), uv, 0.0) / params.luminance_multiplier;
#endif // USE_MULTIVIEW
#endif // USES_QUARTER_RES_COLOR

#endif //USE_CUBEMAP_PASS

	{

#CODE : SKY

	}

	frag_color.rgb = color;
	frag_color.a = alpha;

	// For mobile renderer we're multiplying by 0.5 as we're using a UNORM buffer.
	// For both mobile and clustered, we also bake in the exposure value for the environment and camera.
	frag_color.rgb = frag_color.rgb * params.luminance_multiplier;

#if !defined(DISABLE_FOG) && !defined(USE_CUBEMAP_PASS)

	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.
	if (sky_scene_data.fog_enabled) {
		vec4 fog = fog_process(cube_normal, frag_color.rgb);
		frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a * sky_scene_data.fog_sky_affect);
	}

	if (sky_scene_data.volumetric_fog_enabled) {
		vec4 fog = volumetric_fog_process(uv);
		frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a * sky_scene_data.volumetric_fog_sky_affect);
	}

	if (custom_fog.a > 0.0) {
		frag_color.rgb = mix(frag_color.rgb, custom_fog.rgb, custom_fog.a);
	}

#endif // DISABLE_FOG

	// Blending is disabled for Sky, so alpha doesn't blend.
	// Alpha is used for subsurface scattering so make sure it doesn't get applied to Sky.
	if (!AT_CUBEMAP_PASS && !AT_HALF_RES_PASS && !AT_QUARTER_RES_PASS) {
		frag_color.a = 0.0;
	}

#ifdef USE_DEBANDING
	frag_color.rgb += interleaved_gradient_noise(gl_FragCoord.xy) * params.luminance_multiplier;
#endif
}
