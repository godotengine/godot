#ifndef GS_LIGHTING_BRIDGE_GLSL
#define GS_LIGHTING_BRIDGE_GLSL

#ifndef MAX_VIEWS
#define MAX_VIEWS 2
#endif

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#ifndef MAX_ROUGHNESS_LOD
#define MAX_ROUGHNESS_LOD 5.0
#endif

#ifndef MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS
#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS 1
#endif

#include "../../../../servers/rendering/renderer_rd/shaders/half_inc.glsl"
#include "../../../../servers/rendering/renderer_rd/shaders/scene_data_inc.glsl"
#include "../../../../servers/rendering/renderer_rd/shaders/light_data_inc.glsl"

// ============================================================================
// Compute shader compatibility shim
// ============================================================================
// The Godot scene_forward_lights_inc.glsl uses gl_FragCoord for shadow sampling
// randomization. In compute shaders, gl_FragCoord doesn't exist, so we provide
// a substitute. Since we disable soft shadows (sc_*_shadow_samples() return 0),
// the code paths using gl_FragCoord are never executed, but GLSL still needs
// to be able to parse them.
//
// Compute shaders should define GS_COMPUTE_SHADER before including this file
// and can optionally set gs_frag_coord_substitute to a meaningful value if
// soft shadows are ever enabled in the future.
// ============================================================================
#ifdef GS_COMPUTE_SHADER
// Provide a substitute for gl_FragCoord in compute shaders.
// This is used by shadow sampling for randomization; since soft shadows are
// disabled, the actual value doesn't matter - it just needs to compile.
vec4 gs_frag_coord_substitute = vec4(0.0);
#define gl_FragCoord gs_frag_coord_substitute
#endif // GS_COMPUTE_SHADER

bool sc_use_light_projector() {
	return false;
}

bool sc_use_light_soft_shadows() {
	return false;
}

bool sc_projector_use_mipmaps() {
	return false;
}

uint sc_soft_shadow_samples() {
	return 0u;
}

uint sc_penumbra_shadow_samples() {
	return 0u;
}

uint sc_directional_soft_shadow_samples() {
	return 0u;
}

uint sc_directional_penumbra_shadow_samples() {
	return 0u;
}

float sc_luminance_multiplier() {
	return 1.0;
}

layout(set = 2, binding = 0, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene_data_block;

layout(set = 2, binding = 1, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

layout(set = 2, binding = 2, std430) restrict readonly buffer OmniLights {
	LightData data[];
}
omni_lights;

layout(set = 2, binding = 3, std430) restrict readonly buffer SpotLights {
	LightData data[];
}
spot_lights;

layout(set = 2, binding = 4, std430) restrict readonly buffer ReflectionProbeData {
	ReflectionData data[];
}
reflections;

layout(set = 2, binding = 9, std430) restrict readonly buffer ClusterBuffer {
	uint data[];
}
cluster_buffer;

layout(set = 2, binding = 5) uniform texture2D decal_atlas_srgb;
layout(set = 2, binding = 6) uniform textureCubeArray reflection_atlas;
layout(set = 2, binding = 7) uniform sampler light_projector_sampler;
layout(set = 2, binding = 8) uniform sampler DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP;
layout(set = 2, binding = 10) uniform sampler shadow_sampler;
layout(set = 2, binding = 11) uniform texture2D shadow_atlas;
layout(set = 2, binding = 12) uniform texture2D directional_shadow_atlas;
layout(set = 2, binding = 13) uniform sampler SAMPLER_LINEAR_CLAMP;

#include "../../../../servers/rendering/renderer_rd/shaders/scene_forward_lights_inc.glsl"

#endif // GS_LIGHTING_BRIDGE_GLSL
