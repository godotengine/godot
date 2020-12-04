
#define CLUSTER_COUNTER_SHIFT 20
#define CLUSTER_POINTER_MASK ((1 << CLUSTER_COUNTER_SHIFT) - 1)
#define CLUSTER_COUNTER_MASK 0xfff

struct LightData { //this structure needs to be as packed as possible
	vec3 position;
	float inv_radius;
	vec3 direction;
	float size;
	uint attenuation_energy; //attenuation
	uint color_specular; //rgb color, a specular (8 bit unorm)
	uint cone_attenuation_angle; // attenuation and angle, (16bit float)
	uint shadow_color_enabled; //shadow rgb color, a>0.5 enabled (8bit unorm)
	vec4 atlas_rect; // rect in the shadow atlas
	mat4 shadow_matrix;
	float shadow_bias;
	float shadow_normal_bias;
	float transmittance_bias;
	float soft_shadow_size; // for spot, it's the size in uv coordinates of the light, for omni it's the span angle
	float soft_shadow_scale; // scales the shadow kernel for blurrier shadows
	uint mask;
	float shadow_volumetric_fog_fade;
	uint pad;
	vec4 projector_rect; //projector rect in srgb decal atlas
};

#define REFLECTION_AMBIENT_DISABLED 0
#define REFLECTION_AMBIENT_ENVIRONMENT 1
#define REFLECTION_AMBIENT_COLOR 2

struct ReflectionData {
	vec3 box_extents;
	float index;
	vec3 box_offset;
	uint mask;
	vec4 params; // intensity, 0, interior , boxproject
	vec3 ambient; // ambient color
	uint ambient_mode;
	mat4 local_matrix; // up to here for spot and omni, rest is for directional
	// notes: for ambientblend, use distance to edge to blend between already existing global environment
};

struct DirectionalLightData {
	vec3 direction;
	float energy;
	vec3 color;
	float size;
	float specular;
	uint mask;
	float softshadow_angle;
	float soft_shadow_scale;
	bool blend_splits;
	bool shadow_enabled;
	float fade_from;
	float fade_to;
	uvec3 pad;
	float shadow_volumetric_fog_fade;
	vec4 shadow_bias;
	vec4 shadow_normal_bias;
	vec4 shadow_transmittance_bias;
	vec4 shadow_z_range;
	vec4 shadow_range_begin;
	vec4 shadow_split_offsets;
	mat4 shadow_matrix1;
	mat4 shadow_matrix2;
	mat4 shadow_matrix3;
	mat4 shadow_matrix4;
	vec4 shadow_color1;
	vec4 shadow_color2;
	vec4 shadow_color3;
	vec4 shadow_color4;
	vec2 uv_scale1;
	vec2 uv_scale2;
	vec2 uv_scale3;
	vec2 uv_scale4;
};

struct DecalData {
	mat4 xform; //to decal transform
	vec3 inv_extents;
	float albedo_mix;
	vec4 albedo_rect;
	vec4 normal_rect;
	vec4 orm_rect;
	vec4 emission_rect;
	vec4 modulate;
	float emission_energy;
	uint mask;
	float upper_fade;
	float lower_fade;
	mat3x4 normal_xform;
	vec3 normal;
	float normal_fade;
};
