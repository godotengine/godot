
/* SET0: Per draw primitive settings */


#define MAX_LIGHTS 128

#define FLAGS_INSTANCING_STRIDE_MASK 0xF
#define FLAGS_INSTANCING_ENABLED (1<<4)
#define FLAGS_INSTANCING_HAS_COLORS (1 << 5)
#define FLAGS_INSTANCING_COLOR_8BIT (1 << 6)
#define FLAGS_INSTANCING_HAS_CUSTOM_DATA (1 << 7)
#define FLAGS_INSTANCING_CUSTOM_DATA_8_BIT (1 << 8)

#define FLAGS_CLIP_RECT_UV (1 << 9)
#define FLAGS_TRANSPOSE_RECT (1 << 10)
#define FLAGS_NINEPACH_DRAW_CENTER (1 << 12)
#define FLAGS_USING_PARTICLES (1 << 13)
#define FLAGS_USE_PIXEL_SNAP (1 << 14)

#define FLAGS_USE_SKELETON (1 << 16)

layout(push_constant, binding = 0, std140) uniform DrawData {
	mat2x4 world;
	vec4 modulation;
	vec4 ninepatch_margins;
	vec4 dst_rect; //for built-in rect and UV
	vec4 src_rect;
	uint flags;
	uint ninepatch_repeat;
	vec2 color_texture_pixel_size;
} draw_data;

// The values passed per draw primitives are cached within it

layout(set = 0, binding = 1) uniform texture2D color_texture;
layout(set = 0, binding = 2) uniform texture2D normal_texture;
layout(set = 0, binding = 3) uniform texture2D specular_texture;
layout(set = 0, binding = 4) uniform sampler texture_sampler;

layout(set = 0, binding = 5) uniform textureBuffer instancing_buffer;

/* SET1: Is reserved for the material */

//

/* SET2: Per Canvas Item Settings */

layout(set = 1, binding = 1) uniform textureBuffer skeleton_buffer;

layout(set = 1, binding = 2, std140) uniform SkeletonData {
	mat4 skeleton_transform;
	mat4 skeleton_transform_inverse;
} skeleton_data;

// this set (set 2) is also used for instance specific uniforms

/* SET3: Per Scene settings */

layout(set = 3, binding = 0, std140) uniform CanvasData {
	mat4 canvas_transform;
	mat4 screen_transform;
	//uint light_count;
} canvas_data;

struct Light {
	// light matrices
	mat4 light_matrix;
	mat4 light_local_matrix;
	mat4 shadow_matrix;
	vec4 light_color;
	vec4 light_shadow_color;
	vec2 light_pos;
	float shadowpixel_size;
	float shadow_gradient;
	float light_height;
	float light_outside_alpha;
	float shadow_distance_mult;
};

layout(set = 3, binding = 1, std140) uniform LightData {
	Light lights[MAX_LIGHTS];
} light_data;

layout(set = 3, binding = 2) uniform texture2D light_textures[MAX_LIGHTS];
