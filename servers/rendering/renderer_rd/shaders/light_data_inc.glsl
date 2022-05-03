#define LIGHT_BAKE_DISABLED 0
#define LIGHT_BAKE_STATIC 1
#define LIGHT_BAKE_DYNAMIC 2

struct LightData { //this structure needs to be as packed as possible
	highp vec3 position;
	highp float inv_radius;

	mediump vec3 direction;
	highp float size;

	mediump vec3 color;
	mediump float attenuation;

	mediump float cone_attenuation;
	mediump float cone_angle;
	mediump float specular_amount;
	bool shadow_enabled;

	highp vec4 atlas_rect; // rect in the shadow atlas
	highp mat4 shadow_matrix;
	highp float shadow_bias;
	highp float shadow_normal_bias;
	highp float transmittance_bias;
	highp float soft_shadow_size; // for spot, it's the size in uv coordinates of the light, for omni it's the span angle
	highp float soft_shadow_scale; // scales the shadow kernel for blurrier shadows
	uint mask;
	mediump float shadow_volumetric_fog_fade;
	uint bake_mode;
	highp vec4 projector_rect; //projector rect in srgb decal atlas
};

#define REFLECTION_AMBIENT_DISABLED 0
#define REFLECTION_AMBIENT_ENVIRONMENT 1
#define REFLECTION_AMBIENT_COLOR 2

struct ReflectionData {
	highp vec3 box_extents;
	mediump float index;
	highp vec3 box_offset;
	uint mask;
	mediump vec3 ambient; // ambient color
	mediump float intensity;
	bool exterior;
	bool box_project;
	uint ambient_mode;
	uint pad;
	//0-8 is intensity,8-9 is ambient, mode
	highp mat4 local_matrix; // up to here for spot and omni, rest is for directional
	// notes: for ambientblend, use distance to edge to blend between already existing global environment
};

uv_scale1
