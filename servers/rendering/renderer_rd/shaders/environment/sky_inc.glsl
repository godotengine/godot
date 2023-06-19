// Include file for sky shader so the same definations are included in vertex and fragment shader

#define M_PI 3.14159265359
#define MAX_VIEWS 2
#define EPSILON 0.0001

#ifdef USE_MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
// !BAS! This needs to become an input once we implement our fallback!
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#else // USE_MULTIVIEW
// Set to zero, not supported in non stereo
#define ViewIndex 0
#endif //USE_MULTIVIEW

layout(push_constant, std430) uniform Params {
	mat3 cubemap_view_orientation;
	vec3 pad;
	float luminance_multiplier;
}
params;

layout(set = 0, binding = 2, std140) uniform SkySceneData {
	mat4 view_projections[2];
	mat4 cubemap_projection;

	mat3 view_orientation;
	vec3 position;
	float time;

	bool volumetric_fog_enabled; // 4 - 4
	float volumetric_fog_inv_length; // 4 - 8
	float volumetric_fog_detail_spread; // 4 - 12
	float volumetric_fog_sky_affect; // 4 - 16

	bool fog_enabled; // 4 - 20
	float fog_sky_affect; // 4 - 24
	float fog_density; // 4 - 28
	float fog_sun_scatter; // 4 - 32

	vec3 fog_light_color; // 12 - 44
	float fog_aerial_perspective; // 4 - 48

	float z_far; // 4 - 52
	uint directional_light_count; // 4 - 56
	uint pad1; // 4 - 60
	uint pad2; // 4 - 60
}
sky_scene_data;
