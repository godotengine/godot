
#define MAX_LIGHTS_PER_ITEM 16

#define M_PI 3.14159265359

#define FLAGS_INSTANCING_STRIDE_MASK 0xF
#define FLAGS_INSTANCING_ENABLED (1 << 4)
#define FLAGS_INSTANCING_HAS_COLORS (1 << 5)
#define FLAGS_INSTANCING_COLOR_8BIT (1 << 6)
#define FLAGS_INSTANCING_HAS_CUSTOM_DATA (1 << 7)
#define FLAGS_INSTANCING_CUSTOM_DATA_8_BIT (1 << 8)

#define FLAGS_CLIP_RECT_UV (1 << 9)
#define FLAGS_TRANSPOSE_RECT (1 << 10)
#define FLAGS_USING_LIGHT_MASK (1 << 11)
#define FLAGS_NINEPACH_DRAW_CENTER (1 << 12)
#define FLAGS_USING_PARTICLES (1 << 13)

#define FLAGS_NINEPATCH_H_MODE_SHIFT 16
#define FLAGS_NINEPATCH_V_MODE_SHIFT 18

#define FLAGS_LIGHT_COUNT_SHIFT 20

#define FLAGS_DEFAULT_NORMAL_MAP_USED (1 << 26)
#define FLAGS_DEFAULT_SPECULAR_MAP_USED (1 << 27)

// Push Constant

layout(push_constant, binding = 0, std430) uniform DrawData {
	vec2 world_x;
	vec2 world_y;
	vec2 world_ofs;
	uint flags;
	uint specular_shininess;
#ifdef USE_PRIMITIVE
	vec2 points[3];
	vec2 uvs[3];
	uint colors[6];
#else
	vec4 modulation;
	vec4 ninepatch_margins;
	vec4 dst_rect; //for built-in rect and UV
	vec4 src_rect;
	vec2 pad;

#endif
	vec2 color_texture_pixel_size;
	uint lights[4];
}
draw_data;

// In vulkan, sets should always be ordered using the following logic:
// Lower Sets: Sets that change format and layout less often
// Higher sets: Sets that change format and layout very often
// This is because changing a set for another with a different layout or format,
// invalidates all the upper ones (as likely internal base offset changes)

/* SET0: Globals */

// The values passed per draw primitives are cached within it

layout(set = 0, binding = 1, std140) uniform CanvasData {
	mat4 canvas_transform;
	mat4 screen_transform;
	mat4 canvas_normal_transform;
	vec4 canvas_modulation;
	vec2 screen_pixel_size;
	float time;
	bool use_pixel_snap;

	uint directional_light_count;
	uint pad0;
	uint pad1;
	uint pad2;
}
canvas_data;

#define LIGHT_FLAGS_BLEND_MASK (3 << 16)
#define LIGHT_FLAGS_BLEND_MODE_ADD (0 << 16)
#define LIGHT_FLAGS_BLEND_MODE_SUB (1 << 16)
#define LIGHT_FLAGS_BLEND_MODE_MIX (2 << 16)
#define LIGHT_FLAGS_BLEND_MODE_MASK (3 << 16)
#define LIGHT_FLAGS_HAS_SHADOW (1 << 20)
#define LIGHT_FLAGS_FILTER_SHIFT 22
#define LIGHT_FLAGS_FILTER_MASK (3 << 22)
#define LIGHT_FLAGS_SHADOW_NEAREST (0 << 22)
#define LIGHT_FLAGS_SHADOW_PCF5 (1 << 22)
#define LIGHT_FLAGS_SHADOW_PCF13 (2 << 22)

struct Light {
	mat2x4 texture_matrix; //light to texture coordinate matrix (transposed)
	mat2x4 shadow_matrix; //light to shadow coordinate matrix (transposed)
	vec4 color;

	uint shadow_color; // packed
	uint flags; //index to light texture
	float shadow_pixel_size;
	float height;

	vec2 position;
	float shadow_zfar_inv;
	float shadow_y_ofs;

	vec4 atlas_rect;
};

layout(set = 0, binding = 2, std140) uniform LightData {
	Light data[MAX_LIGHTS];
}
light_array;

layout(set = 0, binding = 3) uniform texture2D atlas_texture;
layout(set = 0, binding = 4) uniform texture2D shadow_atlas_texture;

layout(set = 0, binding = 5) uniform sampler shadow_sampler;

layout(set = 0, binding = 6) uniform texture2D screen_texture;

layout(set = 0, binding = 7) uniform sampler material_samplers[12];

layout(set = 0, binding = 8, std430) restrict readonly buffer GlobalVariableData {
	vec4 data[];
}
global_variables;

/* SET1: Is reserved for the material */

//

/* SET2: Instancing and Skeleton */

layout(set = 2, binding = 0, std430) restrict readonly buffer Transforms {
	vec4 data[];
}
transforms;

/* SET3: Texture */

layout(set = 3, binding = 0) uniform texture2D color_texture;
layout(set = 3, binding = 1) uniform texture2D normal_texture;
layout(set = 3, binding = 2) uniform texture2D specular_texture;
layout(set = 3, binding = 3) uniform sampler texture_sampler;
