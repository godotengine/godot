
#define MAX_LIGHTS_PER_ITEM uint(16)

#define M_PI 3.14159265359

#define SDF_MAX_LENGTH 16384.0

//1 means enabled, 2+ means trails in use
#define FLAGS_INSTANCING_MASK uint(0x7F)
#define FLAGS_INSTANCING_HAS_COLORS uint(1 << 7)
#define FLAGS_INSTANCING_HAS_CUSTOM_DATA uint(1 << 8)

#define FLAGS_CLIP_RECT_UV uint(1 << 9)
#define FLAGS_TRANSPOSE_RECT uint(1 << 10)
// (1 << 11) is for FLAGS_CONVERT_ATTRIBUTES_TO_LINEAR in RD backends, unused here.
#define FLAGS_NINEPACH_DRAW_CENTER uint(1 << 12)

#define FLAGS_NINEPATCH_H_MODE_SHIFT 16
#define FLAGS_NINEPATCH_V_MODE_SHIFT 18

#define FLAGS_LIGHT_COUNT_SHIFT 20

#define FLAGS_DEFAULT_NORMAL_MAP_USED uint(1 << 26)
#define FLAGS_DEFAULT_SPECULAR_MAP_USED uint(1 << 27)

#define FLAGS_USE_MSDF uint(1 << 28)
#define FLAGS_USE_LCD uint(1 << 29)

#define FLAGS_FLIP_H uint(1 << 30)
#define FLAGS_FLIP_V uint(1 << 31)

layout(std140) uniform GlobalShaderUniformData { //ubo:1
	vec4 global_shader_uniforms[MAX_GLOBAL_SHADER_UNIFORMS];
};

layout(std140) uniform CanvasData { //ubo:0
	mat4 canvas_transform;
	mat4 screen_transform;
	mat4 canvas_normal_transform;
	vec4 canvas_modulation;
	vec2 screen_pixel_size;
	float time;
	bool use_pixel_snap;

	vec4 sdf_to_tex;
	vec2 screen_to_sdf;
	vec2 sdf_to_screen;

	uint directional_light_count;
	float tex_to_sdf;
	uint pad1;
	uint pad2;
};

#ifndef DISABLE_LIGHTING
#define LIGHT_FLAGS_BLEND_MASK uint(3 << 16)
#define LIGHT_FLAGS_BLEND_MODE_ADD uint(0 << 16)
#define LIGHT_FLAGS_BLEND_MODE_SUB uint(1 << 16)
#define LIGHT_FLAGS_BLEND_MODE_MIX uint(2 << 16)
#define LIGHT_FLAGS_BLEND_MODE_MASK uint(3 << 16)
#define LIGHT_FLAGS_HAS_SHADOW uint(1 << 20)
#define LIGHT_FLAGS_FILTER_SHIFT 22
#define LIGHT_FLAGS_FILTER_MASK uint(3 << 22)
#define LIGHT_FLAGS_SHADOW_NEAREST uint(0 << 22)
#define LIGHT_FLAGS_SHADOW_PCF5 uint(1 << 22)
#define LIGHT_FLAGS_SHADOW_PCF13 uint(2 << 22)

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

layout(std140) uniform LightData { //ubo:2
	Light light_array[MAX_LIGHTS];
};
#endif // DISABLE_LIGHTING
