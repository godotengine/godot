[vertex]

#define M_PI 3.14159265359

/*
from VisualServer:

ARRAY_VERTEX=0,
ARRAY_NORMAL=1,
ARRAY_TANGENT=2,
ARRAY_COLOR=3,
ARRAY_TEX_UV=4,
ARRAY_TEX_UV2=5,
ARRAY_BONES=6,
ARRAY_WEIGHTS=7,
ARRAY_INDEX=8,
*/

//hack to use uv if no uv present so it works with lightmap


/* INPUT ATTRIBS */

layout(location=0) in highp vec4 vertex_attrib;
layout(location=1) in vec3 normal_attrib;
#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
layout(location=2) in vec4 tangent_attrib;
#endif

#if defined(ENABLE_COLOR_INTERP)
layout(location=3) in vec4 color_attrib;
#endif

#if defined(ENABLE_UV_INTERP)
layout(location=4) in vec2 uv_attrib;
#endif

#if defined(ENABLE_UV2_INTERP)
layout(location=5) in vec2 uv2_attrib;
#endif

uniform float normal_mult;

#ifdef USE_SKELETON
layout(location=6) in ivec4 bone_indices; // attrib:6
layout(location=7) in vec4 bone_weights; // attrib:7
#endif

#ifdef USE_INSTANCING

layout(location=8) in highp vec4 instance_xform0;
layout(location=9) in highp vec4 instance_xform1;
layout(location=10) in highp vec4 instance_xform2;
layout(location=11) in lowp vec4 instance_color;

#if defined(ENABLE_INSTANCE_CUSTOM)
layout(location=12) in highp vec4 instance_custom_data;
#endif

#endif

layout(std140) uniform SceneData { //ubo:0

	highp mat4 projection_matrix;
	highp mat4 inv_projection_matrix;
	highp mat4 camera_inverse_matrix;
	highp mat4 camera_matrix;

	mediump vec4 ambient_light_color;
	mediump vec4 bg_color;

	mediump vec4 fog_color_enabled;
	mediump vec4 fog_sun_color_amount;

	mediump float ambient_energy;
	mediump float bg_energy;

	mediump float z_offset;
	mediump float z_slope_scale;
	highp float shadow_dual_paraboloid_render_zfar;
	highp float shadow_dual_paraboloid_render_side;

	highp vec2 viewport_size;
	highp vec2 screen_pixel_size;
	highp vec2 shadow_atlas_pixel_size;
	highp vec2 directional_shadow_pixel_size;

	highp float time;
	highp float z_far;
	mediump float reflection_multiplier;
	mediump float subsurface_scatter_width;
	mediump float ambient_occlusion_affect_light;

	bool fog_depth_enabled;
	highp float fog_depth_begin;
	highp float fog_depth_curve;
	bool fog_transmit_enabled;
	highp float fog_transmit_curve;
	bool fog_height_enabled;
	highp float fog_height_min;
	highp float fog_height_max;
	highp float fog_height_curve;

};

uniform highp mat4 world_transform;


#ifdef USE_LIGHT_DIRECTIONAL

layout(std140) uniform DirectionalLightData { //ubo:3

	highp vec4 light_pos_inv_radius;
	mediump vec4 light_direction_attenuation;
	mediump vec4 light_color_energy;
	mediump vec4 light_params; //cone attenuation, angle, specular, shadow enabled,
	mediump vec4 light_clamp;
	mediump vec4 shadow_color_contact;
	highp mat4 shadow_matrix1;
	highp mat4 shadow_matrix2;
	highp mat4 shadow_matrix3;
	highp mat4 shadow_matrix4;
	mediump vec4 shadow_split_offsets;
};

#endif

#ifdef USE_VERTEX_LIGHTING
//omni and spot

struct LightData {

	highp vec4 light_pos_inv_radius;
	mediump vec4 light_direction_attenuation;
	mediump vec4 light_color_energy;
	mediump vec4 light_params; //cone attenuation, angle, specular, shadow enabled,
	mediump vec4 light_clamp;
	mediump vec4 shadow_color_contact;
	highp mat4 shadow_matrix;

};


layout(std140) uniform OmniLightData { //ubo:4

	LightData omni_lights[MAX_LIGHT_DATA_STRUCTS];
};

layout(std140) uniform SpotLightData { //ubo:5

	LightData spot_lights[MAX_LIGHT_DATA_STRUCTS];
};

#ifdef USE_FORWARD_LIGHTING


uniform int omni_light_indices[MAX_FORWARD_LIGHTS];
uniform int omni_light_count;

uniform int spot_light_indices[MAX_FORWARD_LIGHTS];
uniform int spot_light_count;

#endif

out vec4 diffuse_light_interp;
out vec4 specular_light_interp;

void light_compute(vec3 N, vec3 L,vec3 V, vec3 light_color, float roughness, inout vec3 diffuse, inout vec3 specular) {

	float dotNL = max(dot(N,L), 0.0 );
	diffuse += dotNL * light_color / M_PI;

	if (roughness > 0.0) {

		vec3 H = normalize(V + L);
		float dotNH = max(dot(N,H), 0.0 );
		float intensity = pow( dotNH, (1.0-roughness) * 256.0);
		specular += light_color * intensity;

	}
}

void light_process_omni(int idx, vec3 vertex, vec3 eye_vec,vec3 normal, float roughness,inout vec3 diffuse, inout vec3 specular) {

	vec3 light_rel_vec = omni_lights[idx].light_pos_inv_radius.xyz-vertex;
	float light_length = length( light_rel_vec );
	float normalized_distance = light_length*omni_lights[idx].light_pos_inv_radius.w;
	vec3 light_attenuation = vec3(pow( max(1.0 - normalized_distance, 0.0), omni_lights[idx].light_direction_attenuation.w ));

	light_compute(normal,normalize(light_rel_vec),eye_vec,omni_lights[idx].light_color_energy.rgb * light_attenuation,roughness,diffuse,specular);

}

void light_process_spot(int idx, vec3 vertex, vec3 eye_vec, vec3 normal, float roughness, inout vec3 diffuse, inout vec3 specular) {

	vec3 light_rel_vec = spot_lights[idx].light_pos_inv_radius.xyz-vertex;
	float light_length = length( light_rel_vec );
	float normalized_distance = light_length*spot_lights[idx].light_pos_inv_radius.w;
	vec3 light_attenuation = vec3(pow( max(1.0 - normalized_distance, 0.001), spot_lights[idx].light_direction_attenuation.w ));
	vec3 spot_dir = spot_lights[idx].light_direction_attenuation.xyz;
	float spot_cutoff=spot_lights[idx].light_params.y;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir),spot_cutoff);
	float spot_rim = (1.0 - scos) / (1.0 - spot_cutoff);
	light_attenuation *= 1.0 - pow( max(spot_rim,0.001), spot_lights[idx].light_params.x);


	light_compute(normal,normalize(light_rel_vec),eye_vec,spot_lights[idx].light_color_energy.rgb*light_attenuation,roughness,diffuse,specular);
}


#endif

/* Varyings */

out highp vec3 vertex_interp;
out vec3 normal_interp;

#if defined(ENABLE_COLOR_INTERP)
out vec4 color_interp;
#endif

#if defined(ENABLE_UV_INTERP)
out vec2 uv_interp;
#endif

#if defined(ENABLE_UV2_INTERP)
out vec2 uv2_interp;
#endif


#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif





#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:1

MATERIAL_UNIFORMS

};

#endif

VERTEX_SHADER_GLOBALS

#ifdef RENDER_DEPTH_DUAL_PARABOLOID

out highp float dp_clip;

#endif

#define SKELETON_TEXTURE_WIDTH 256

#ifdef USE_SKELETON
uniform highp sampler2D skeleton_texture; //texunit:-1
#endif

out highp vec4 position_interp;

void main() {

	highp vec4 vertex = vertex_attrib; // vec4(vertex_attrib.xyz * data_attrib.x,1.0);

	mat4 world_matrix = world_transform;


#ifdef USE_INSTANCING

	{
		highp mat4 m=mat4(instance_xform0,instance_xform1,instance_xform2,vec4(0.0,0.0,0.0,1.0));
		world_matrix = world_matrix * transpose(m);
	}
#endif

	vec3 normal = normal_attrib * normal_mult;


#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
	vec3 tangent = tangent_attrib.xyz;
	tangent*=normal_mult;
	float binormalf = tangent_attrib.a;
#endif

#if defined(ENABLE_COLOR_INTERP)
	color_interp = color_attrib;
#if defined(USE_INSTANCING)
	color_interp *= instance_color;
#endif

#endif

#ifdef USE_SKELETON
	{
		//skeleton transform
		ivec2 tex_ofs = ivec2( bone_indices.x%256, (bone_indices.x/256)*3 );
		highp mat3x4 m = mat3x4(
			texelFetch(skeleton_texture,tex_ofs,0),
			texelFetch(skeleton_texture,tex_ofs+ivec2(0,1),0),
			texelFetch(skeleton_texture,tex_ofs+ivec2(0,2),0)
		) * bone_weights.x;

		tex_ofs = ivec2( bone_indices.y%256, (bone_indices.y/256)*3 );

		m+= mat3x4(
					texelFetch(skeleton_texture,tex_ofs,0),
					texelFetch(skeleton_texture,tex_ofs+ivec2(0,1),0),
					texelFetch(skeleton_texture,tex_ofs+ivec2(0,2),0)
				) * bone_weights.y;

		tex_ofs = ivec2( bone_indices.z%256, (bone_indices.z/256)*3 );

		m+= mat3x4(
					texelFetch(skeleton_texture,tex_ofs,0),
					texelFetch(skeleton_texture,tex_ofs+ivec2(0,1),0),
					texelFetch(skeleton_texture,tex_ofs+ivec2(0,2),0)
				) * bone_weights.z;


		tex_ofs = ivec2( bone_indices.w%256, (bone_indices.w/256)*3 );

		m+= mat3x4(
					texelFetch(skeleton_texture,tex_ofs,0),
					texelFetch(skeleton_texture,tex_ofs+ivec2(0,1),0),
					texelFetch(skeleton_texture,tex_ofs+ivec2(0,2),0)
				) * bone_weights.w;


		vertex.xyz = vertex * m;

		normal = vec4(normal,0.0) * m;
#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
		tangent.xyz = vec4(tangent.xyz,0.0) * m;
#endif
	}
#endif


#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)

	vec3 binormal = normalize( cross(normal,tangent) * binormalf );
#endif

#if defined(ENABLE_UV_INTERP)
	uv_interp = uv_attrib;
#endif

#if defined(ENABLE_UV2_INTERP)
	uv2_interp = uv2_attrib;
#endif

#if defined(USE_INSTANCING) && defined(ENABLE_INSTANCE_CUSTOM)
	vec4 instance_custom = instance_custom_data;
#else
	vec4 instance_custom = vec4(0.0);
#endif

	highp mat4 modelview = camera_inverse_matrix * world_matrix;
	highp mat4 local_projection = projection_matrix;

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = world_matrix * vertex;
	normal = normalize((world_matrix * vec4(normal,0.0)).xyz);

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)

	tangent = normalize((world_matrix * vec4(tangent,0.0)).xyz);
	binormal = normalize((world_matrix * vec4(binormal,0.0)).xyz);
#endif
#endif

	float roughness=0.0;

//defines that make writing custom shaders easier
#define projection_matrix local_projection
#define world_transform world_matrix
{

VERTEX_SHADER_CODE

}



//using local coordinates (default)
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

	vertex = modelview * vertex;
	normal = normalize((modelview * vec4(normal,0.0)).xyz);

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)

	tangent = normalize((modelview * vec4(tangent,0.0)).xyz);
	binormal = normalize((modelview * vec4(binormal,0.0)).xyz);
#endif
#endif

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = camera_inverse_matrix * vertex;
	normal = normalize((camera_inverse_matrix * vec4(normal,0.0)).xyz);

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)

	tangent = normalize((camera_inverse_matrix * vec4(tangent,0.0)).xyz);
	binormal = normalize((camera_inverse_matrix * vec4(binormal,0.0)).xyz);
#endif
#endif

	vertex_interp = vertex.xyz;
	normal_interp = normal;


#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif


#ifdef RENDER_DEPTH


#ifdef RENDER_DEPTH_DUAL_PARABOLOID

	vertex_interp.z*= shadow_dual_paraboloid_render_side;
	normal_interp.z*= shadow_dual_paraboloid_render_side;

	dp_clip=vertex_interp.z; //this attempts to avoid noise caused by objects sent to the other parabolloid side due to bias

	//for dual paraboloid shadow mapping, this is the fastest but least correct way, as it curves straight edges

	highp vec3 vtx = vertex_interp+normalize(vertex_interp)*z_offset;
	highp float distance = length(vtx);
	vtx = normalize(vtx);
	vtx.xy/=1.0-vtx.z;
	vtx.z=(distance/shadow_dual_paraboloid_render_zfar);
	vtx.z=vtx.z * 2.0 - 1.0;

	vertex.xyz=vtx;
	vertex.w=1.0;


#else

	float z_ofs = z_offset;
	z_ofs += (1.0-abs(normal_interp.z))*z_slope_scale;
	vertex_interp.z-=z_ofs;

#endif //RENDER_DEPTH_DUAL_PARABOLOID

#endif //RENDER_DEPTH


#if !defined(SKIP_TRANSFORM_USED) && !defined(RENDER_DEPTH_DUAL_PARABOLOID)
	gl_Position = projection_matrix * vec4(vertex_interp,1.0);
#else
	gl_Position = vertex;
#endif

	position_interp=gl_Position;

#ifdef USE_VERTEX_LIGHTING

	diffuse_light_interp=vec4(0.0);
	specular_light_interp=vec4(0.0);

#ifdef USE_FORWARD_LIGHTING

	for(int i=0;i<omni_light_count;i++) {
		light_process_omni(omni_light_indices[i],vertex_interp,-normalize( vertex_interp ),normal_interp,roughness,diffuse_light_interp.rgb,specular_light_interp.rgb);
	}

	for(int i=0;i<spot_light_count;i++) {
		light_process_spot(spot_light_indices[i],vertex_interp,-normalize( vertex_interp ),normal_interp,roughness,diffuse_light_interp.rgb,specular_light_interp.rgb);
	}
#endif

#ifdef USE_LIGHT_DIRECTIONAL

	vec3 directional_diffuse = vec3(0.0);
	vec3 directional_specular = vec3(0.0);
	light_compute(normal_interp,-light_direction_attenuation.xyz,-normalize( vertex_interp ),light_color_energy.rgb,roughness,directional_diffuse,directional_specular);

	float diff_avg = dot(diffuse_light_interp.rgb,vec3(0.33333));
	float diff_dir_avg = dot(directional_diffuse,vec3(0.33333));
	if (diff_avg>0.0) {
		diffuse_light_interp.a=diff_dir_avg/(diff_avg+diff_dir_avg);
	} else {
		diffuse_light_interp.a=1.0;
	}

	diffuse_light_interp.rgb+=directional_diffuse;

	float spec_avg = dot(specular_light_interp.rgb,vec3(0.33333));
	float spec_dir_avg = dot(directional_specular,vec3(0.33333));
	if (spec_avg>0.0) {
		specular_light_interp.a=spec_dir_avg/(spec_avg+spec_dir_avg);
	} else {
		specular_light_interp.a=1.0;
	}

	specular_light_interp.rgb+=directional_specular;

#endif //USE_LIGHT_DIRECTIONAL


#endif // USE_VERTEX_LIGHTING

}


[fragment]

/* texture unit usage, N is max_texture_unity-N

1-skeleton
2-radiance
3-reflection_atlas
4-directional_shadow
5-shadow_atlas
6-decal_atlas
7-screen
8-depth
9-probe1
10-probe2

*/

uniform highp mat4 world_transform;

#define M_PI 3.14159265359

/* Varyings */

#if defined(ENABLE_COLOR_INTERP)
in vec4 color_interp;
#endif

#if defined(ENABLE_UV_INTERP)
in vec2 uv_interp;
#endif

#if defined(ENABLE_UV2_INTERP)
in vec2 uv2_interp;
#endif

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
in vec3 tangent_interp;
in vec3 binormal_interp;
#endif

in highp vec3 vertex_interp;
in vec3 normal_interp;


/* PBR CHANNELS */

//used on forward mainly
uniform bool no_ambient_light;



#ifdef USE_RADIANCE_MAP



layout(std140) uniform Radiance { //ubo:2

	mat4 radiance_inverse_xform;
	float radiance_ambient_contribution;

};

#define RADIANCE_MAX_LOD 5.0

#ifdef USE_RADIANCE_MAP_ARRAY

uniform sampler2DArray radiance_map; //texunit:-2

vec3 textureDualParaboloid(sampler2DArray p_tex, vec3 p_vec,float p_roughness) {

	vec3 norm = normalize(p_vec);
	norm.xy/=1.0+abs(norm.z);
	norm.xy=norm.xy * vec2(0.5,0.25) + vec2(0.5,0.25);

	// we need to lie the derivatives (normg) and assume that DP side is always the same
	// to get proper texture filtering
	vec2 normg=norm.xy;
	if (norm.z>0.0) {
		norm.y=0.5-norm.y+0.5;
	}

	// thanks to OpenGL spec using floor(layer + 0.5) for texture arrays,
	// it's easy to have precision errors using fract() to interpolate layers
	// as such, using fixed point to ensure it works.

	float index = p_roughness * RADIANCE_MAX_LOD;
	int indexi = int(index * 256.0);
	vec3 base = textureGrad(p_tex, vec3(norm.xy, float(indexi/256)),dFdx(normg),dFdy(normg)).xyz;
	vec3 next = textureGrad(p_tex, vec3(norm.xy, float(indexi/256+1)),dFdx(normg),dFdy(normg)).xyz;
	return mix(base,next,float(indexi%256)/256.0);
}

#else

uniform sampler2D radiance_map; //texunit:-2

vec3 textureDualParaboloid(sampler2D p_tex, vec3 p_vec,float p_roughness) {

	vec3 norm = normalize(p_vec);
	norm.xy/=1.0+abs(norm.z);
	norm.xy=norm.xy * vec2(0.5,0.25) + vec2(0.5,0.25);
	if (norm.z>0.0) {
		norm.y=0.5-norm.y+0.5;
	}
	return textureLod(p_tex, norm.xy, p_roughness * RADIANCE_MAX_LOD).xyz;
}

#endif

#endif

/* Material Uniforms */



#if defined(USE_MATERIAL)

layout(std140) uniform UniformData {

MATERIAL_UNIFORMS

};

#endif

FRAGMENT_SHADER_GLOBALS

layout(std140) uniform SceneData {

	highp mat4 projection_matrix;
	highp mat4 inv_projection_matrix;
	highp mat4 camera_inverse_matrix;
	highp mat4 camera_matrix;

	mediump vec4 ambient_light_color;
	mediump vec4 bg_color;

	mediump vec4 fog_color_enabled;
	mediump vec4 fog_sun_color_amount;

	mediump float ambient_energy;
	mediump float bg_energy;

	mediump float z_offset;
	mediump float z_slope_scale;
	highp float shadow_dual_paraboloid_render_zfar;
	highp float shadow_dual_paraboloid_render_side;

	highp vec2 viewport_size;
	highp vec2 screen_pixel_size;
	highp vec2 shadow_atlas_pixel_size;
	highp vec2 directional_shadow_pixel_size;

	highp float time;
	highp float z_far;
	mediump float reflection_multiplier;
	mediump float subsurface_scatter_width;
	mediump float ambient_occlusion_affect_light;

	bool fog_depth_enabled;
	highp float fog_depth_begin;
	highp float fog_depth_curve;
	bool fog_transmit_enabled;
	highp float fog_transmit_curve;
	bool fog_height_enabled;
	highp float fog_height_min;
	highp float fog_height_max;
	highp float fog_height_curve;
};

//directional light data

#ifdef USE_LIGHT_DIRECTIONAL

layout(std140) uniform DirectionalLightData {

	highp vec4 light_pos_inv_radius;
	mediump vec4 light_direction_attenuation;
	mediump vec4 light_color_energy;
	mediump vec4 light_params; //cone attenuation, angle, specular, shadow enabled,
	mediump vec4 light_clamp;
	mediump vec4 shadow_color_contact;
	highp mat4 shadow_matrix1;
	highp mat4 shadow_matrix2;
	highp mat4 shadow_matrix3;
	highp mat4 shadow_matrix4;
	mediump vec4 shadow_split_offsets;
};


uniform highp sampler2DShadow directional_shadow; //texunit:-4

#endif

#ifdef USE_VERTEX_LIGHTING
in vec4 diffuse_light_interp;
in vec4 specular_light_interp;
#endif
//omni and spot

struct LightData {

	highp vec4 light_pos_inv_radius;
	mediump vec4 light_direction_attenuation;
	mediump vec4 light_color_energy;
	mediump vec4 light_params; //cone attenuation, angle, specular, shadow enabled,
	mediump vec4 light_clamp;
	mediump vec4 shadow_color_contact;
	highp mat4 shadow_matrix;

};


layout(std140) uniform OmniLightData { //ubo:4

	LightData omni_lights[MAX_LIGHT_DATA_STRUCTS];
};

layout(std140) uniform SpotLightData { //ubo:5

	LightData spot_lights[MAX_LIGHT_DATA_STRUCTS];
};


uniform highp sampler2DShadow shadow_atlas; //texunit:-5


struct ReflectionData {

	mediump vec4 box_extents;
	mediump vec4 box_offset;
	mediump vec4 params; // intensity, 0, interior , boxproject
	mediump vec4 ambient; //ambient color, energy
	mediump vec4 atlas_clamp;
	highp mat4 local_matrix; //up to here for spot and omni, rest is for directional
	//notes: for ambientblend, use distance to edge to blend between already existing global environment
};

layout(std140) uniform ReflectionProbeData { //ubo:6

	ReflectionData reflections[MAX_REFLECTION_DATA_STRUCTS];
};
uniform mediump sampler2D reflection_atlas; //texunit:-3


#ifdef USE_FORWARD_LIGHTING

uniform int omni_light_indices[MAX_FORWARD_LIGHTS];
uniform int omni_light_count;

uniform int spot_light_indices[MAX_FORWARD_LIGHTS];
uniform int spot_light_count;

uniform int reflection_indices[MAX_FORWARD_LIGHTS];
uniform int reflection_count;

#endif


#if defined(SCREEN_TEXTURE_USED)

uniform highp sampler2D screen_texture; //texunit:-7

#endif

#ifdef USE_MULTIPLE_RENDER_TARGETS

layout(location=0) out vec4 diffuse_buffer;
layout(location=1) out vec4 specular_buffer;
layout(location=2) out vec4 normal_mr_buffer;
#if defined(ENABLE_SSS)
layout(location=3) out float sss_buffer;
#endif

#else

layout(location=0) out vec4 frag_color;

#endif

in highp vec4 position_interp;
uniform highp sampler2D depth_buffer; //texunit:-8

#ifdef USE_CONTACT_SHADOWS

float contact_shadow_compute(vec3 pos, vec3 dir, float max_distance) {

	if (abs(dir.z)>0.99)
		return 1.0;

	vec3 endpoint = pos+dir*max_distance;
	vec4 source = position_interp;
	vec4 dest = projection_matrix * vec4(endpoint, 1.0);

	vec2 from_screen = (source.xy / source.w) * 0.5 + 0.5;
	vec2 to_screen = (dest.xy / dest.w) * 0.5 + 0.5;

	vec2 screen_rel = to_screen - from_screen;

	if (length(screen_rel)<0.00001)
		return 1.0; //too small, don't do anything

	/*float pixel_size; //approximate pixel size

	if (screen_rel.x > screen_rel.y) {

		pixel_size = abs((pos.x-endpoint.x)/(screen_rel.x/screen_pixel_size.x));
	} else {
		pixel_size = abs((pos.y-endpoint.y)/(screen_rel.y/screen_pixel_size.y));

	}*/
	vec4 bias = projection_matrix * vec4(pos+vec3(0.0,0.0,0.04), 1.0); //todo un-harcode the 0.04



	vec2 pixel_incr = normalize(screen_rel)*screen_pixel_size;


	float steps = length(screen_rel) / length(pixel_incr);
	steps = min(2000.0,steps); //put a limit to avoid freezing in some strange situation
	//steps=10.0;

	vec4 incr = (dest - source)/steps;
	float ratio=0.0;
	float ratio_incr = 1.0/steps;

	while(steps>0.0) {
		source += incr*2.0;
		bias+=incr*2.0;

		vec3 uv_depth = (source.xyz / source.w) * 0.5 + 0.5;
		float depth = texture(depth_buffer,uv_depth.xy).r;

		if (depth < uv_depth.z) {
			if (depth > (bias.z/bias.w) * 0.5 + 0.5) {
				return min(pow(ratio,4.0),1.0);
			} else {
				return 1.0;
			}
		}


		ratio+=ratio_incr;
		steps-=1.0;
	}

	return 1.0;
}

#endif


// This returns the G_GGX function divided by 2 cos_theta_m, where in practice cos_theta_m is either N.L or N.V.
// We're dividing this factor off because the overall term we'll end up looks like
// (see, for example, the first unnumbered equation in B. Burley, "Physically Based Shading at Disney", SIGGRAPH 2012):
//
//   F(L.V) D(N.H) G(N.L) G(N.V) / (4 N.L N.V)
//
// We're basically regouping this as
//
//   F(L.V) D(N.H) [G(N.L)/(2 N.L)] [G(N.V) / (2 N.V)]
//
// and thus, this function implements the [G(N.m)/(2 N.m)] part with m = L or V.
//
// The contents of the D and G (G1) functions (GGX) are taken from
// E. Heitz, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", J. Comp. Graph. Tech. 3 (2) (2014).
// Eqns 71-72 and 85-86 (see also Eqns 43 and 80).

float G_GGX_2cos(float cos_theta_m, float alpha) {
	// Schlick's approximation
	// C. Schlick, "An Inexpensive BRDF Model for Physically-based Rendering", Computer Graphics Forum. 13 (3): 233 (1994)
	// Eq. (19), although see Heitz (2014) the about the problems with his derivation.
	// It nevertheless approximates GGX well with k = alpha/2.
	float k = 0.5*alpha;
	return 0.5 / (cos_theta_m * (1.0 - k) + k);

	// float cos2 = cos_theta_m*cos_theta_m;
	// float sin2 = (1.0-cos2);
	// return 1.0 /( cos_theta_m + sqrt(cos2 + alpha*alpha*sin2) );
}

float D_GGX(float cos_theta_m, float alpha) {
	float alpha2 = alpha*alpha;
	float d = 1.0 + (alpha2-1.0)*cos_theta_m*cos_theta_m;
	return alpha2/(M_PI * d * d);
}

float G_GGX_anisotropic_2cos(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float cos2 = cos_theta_m * cos_theta_m;
	float sin2 = (1.0-cos2);
	float s_x = alpha_x * cos_phi;
	float s_y = alpha_y * sin_phi;
	return 1.0  / (cos_theta_m + sqrt(cos2 + (s_x*s_x + s_y*s_y)*sin2 ));
}

float D_GGX_anisotropic(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float cos2 = cos_theta_m * cos_theta_m;
	float sin2 = (1.0-cos2);
	float r_x = cos_phi/alpha_x;
	float r_y = sin_phi/alpha_y;
	float d = cos2 + sin2*(r_x * r_x + r_y * r_y);
	return 1.0 / (M_PI * alpha_x * alpha_y * d * d );
}


float SchlickFresnel(float u)
{
    float m = 1.0-u;
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
    if (a >= 1.0) return 1.0/M_PI;
    float a2 = a*a;
    float t = 1.0 + (a2-1.0)*NdotH*NdotH;
    return (a2-1.0) / (M_PI*log(a2)*t);
}

vec3 metallic_to_specular_color(float metallic, float specular, vec3 albedo) {
	float dielectric = (0.034 * 2.0) * specular;
	// energy conservation
	return mix(vec3(dielectric), albedo, metallic); // TODO: reference?
}

void light_compute(vec3 N, vec3 L, vec3 V, vec3 B, vec3 T, vec3 light_color, vec3 attenuation, vec3 diffuse_color, vec3 transmission, float specular_blob_intensity, float roughness, float metallic, float rim, float rim_tint, float clearcoat, float clearcoat_gloss, float anisotropy, inout vec3 diffuse_light, inout vec3 specular_light) {

#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader

	vec3 normal = N;
	vec3 albedo = diffuse_color;
	vec3 light = L;
	vec3 view = V;

LIGHT_SHADER_CODE


#else
	float NdotL = dot(N,L);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

	if (metallic < 1.0) {
#if defined(DIFFUSE_OREN_NAYAR)
		vec3 diffuse_brdf_NL;
#else
		float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance
#endif


#if defined(DIFFUSE_LAMBERT_WRAP)
		//energy conserving lambert wrap shader
		diffuse_brdf_NL = max(0.0,(NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness)));

#elif defined(DIFFUSE_OREN_NAYAR)

		{
			// see http://mimosa-pudica.net/improved-oren-nayar.html
			float LdotV = dot(L, V);


			float s = LdotV - NdotL * NdotV;
			float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));

			float sigma2 = roughness * roughness; // TODO: this needs checking
			vec3 A = 1.0 + sigma2 * (- 0.5 / (sigma2 + 0.33) + 0.17*diffuse_color / (sigma2 + 0.13) );
			float B = 0.45 * sigma2 / (sigma2 + 0.09);

			diffuse_brdf_NL = cNdotL * (A + vec3(B) * s / t) * (1.0 / M_PI);
		}

#elif defined(DIFFUSE_TOON)

		diffuse_brdf_NL = smoothstep(-roughness,max(roughness,0.01),NdotL);

#elif defined(DIFFUSE_BURLEY)

		{


			vec3 H = normalize(V + L);
			float cLdotH = max(0.0,dot(L, H));

			float FD90 = 0.5 + 2.0 * cLdotH * cLdotH * roughness;
			float FdV = 1.0 + (FD90 - 1.0) * SchlickFresnel(cNdotV);
			float FdL = 1.0 + (FD90 - 1.0) * SchlickFresnel(cNdotL);
			diffuse_brdf_NL = (1.0 / M_PI) * FdV * FdL * cNdotL;
	/*
			float energyBias = mix(roughness, 0.0, 0.5);
			float energyFactor = mix(roughness, 1.0, 1.0 / 1.51);
			float fd90 = energyBias + 2.0 * VoH * VoH * roughness;
			float f0 = 1.0;
			float lightScatter = f0 + (fd90 - f0) * pow(1.0 - cNdotL, 5.0);
			float viewScatter = f0 + (fd90 - f0) * pow(1.0 - cNdotV, 5.0);

			diffuse_brdf_NL = lightScatter * viewScatter * energyFactor;*/
		}
#else
		//lambert
		diffuse_brdf_NL = cNdotL * (1.0 / M_PI);
#endif

#if defined(TRANSMISSION_USED)
		diffuse_light += light_color * diffuse_color * mix(vec3(diffuse_brdf_NL), vec3(M_PI), transmission) * attenuation;
#else
		diffuse_light += light_color * diffuse_color * diffuse_brdf_NL * attenuation;
#endif



#if defined(LIGHT_USE_RIM)
		float rim_light = pow(1.0-cNdotV, (1.0-roughness)*16.0);
		diffuse_light += rim_light * rim * mix(vec3(1.0),diffuse_color,rim_tint) * light_color;
#endif
	}


	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely


		// D

#if defined(SPECULAR_BLINN)

		vec3 H = normalize(V + L);
		float cNdotH = max(dot(N,H), 0.0 );
		float intensity = pow( cNdotH, (1.0-roughness) * 256.0);
		specular_light += light_color * intensity * specular_blob_intensity * attenuation;

#elif defined(SPECULAR_PHONG)

		 vec3 R = normalize(-reflect(L,N));
		 float cRdotV = max(0.0,dot(R,V));
		 float intensity = pow( cRdotV, (1.0-roughness) * 256.0);
		 specular_light += light_color * intensity * specular_blob_intensity * attenuation;

#elif defined(SPECULAR_TOON)

		vec3 R = normalize(-reflect(L,N));
		float RdotV = dot(R,V);
		float mid = 1.0-roughness;
		mid*=mid;
		float intensity = smoothstep(mid-roughness*0.5, mid+roughness*0.5, RdotV) * mid;
		diffuse_light += light_color * intensity * specular_blob_intensity * attenuation; // write to diffuse_light, as in toon shading you generally want no reflection

#elif defined(SPECULAR_DISABLED)
		//none..

#elif defined(SPECULAR_SCHLICK_GGX)
		// shlick+ggx as default

		vec3 H = normalize(V + L);

		float cNdotH = max(dot(N,H), 0.0);
		float cLdotH = max(dot(L,H), 0.0);

# if defined(LIGHT_USE_ANISOTROPY)

		float aspect = sqrt(1.0-anisotropy*0.9);
		float rx = roughness/aspect;
		float ry = roughness*aspect;
		float ax = rx*rx;
		float ay = ry*ry;
		float XdotH = dot( T, H );
		float YdotH = dot( B, H );
		float D = D_GGX_anisotropic(cNdotH, ax, ay, XdotH, YdotH);
		float G = G_GGX_anisotropic_2cos(cNdotL, ax, ay, XdotH, YdotH) * G_GGX_anisotropic_2cos(cNdotV, ax, ay, XdotH, YdotH);

# else
		float alpha = roughness * roughness;
		float D = D_GGX(cNdotH, alpha);
		float G = G_GGX_2cos(cNdotL, alpha) * G_GGX_2cos(cNdotV, alpha);
# endif
		// F
		float F0 = 1.0; // FIXME
		float cLdotH5 = SchlickFresnel(cLdotH);
		float F = mix(cLdotH5, 1.0, F0);

		float specular_brdf_NL = cNdotL * D * F * G;

		specular_light += specular_brdf_NL * light_color * specular_blob_intensity * attenuation;
#endif

#if defined(LIGHT_USE_CLEARCOAT)
		if (clearcoat_gloss > 0.0) {
# if !defined(SPECULAR_SCHLICK_GGX) && !defined(SPECULAR_BLINN)
			vec3 H = normalize(V + L);
# endif
# if !defined(SPECULAR_SCHLICK_GGX)
			float cNdotH = max(dot(N,H), 0.0);
			float cLdotH = max(dot(L,H), 0.0);
			float cLdotH5 = SchlickFresnel(cLdotH);
#endif
			float Dr = GTR1(cNdotH, mix(.1, .001, clearcoat_gloss));
			float Fr = mix(.04, 1.0, cLdotH5);
			float Gr = G_GGX_2cos(cNdotL, .25) * G_GGX_2cos(cNdotV, .25);


			float specular_brdf_NL = 0.25 * clearcoat * Gr * Fr * Dr * cNdotL;

			specular_light += specular_brdf_NL * light_color * specular_blob_intensity * attenuation;
		}
#endif
	}


#endif //defined(USE_LIGHT_SHADER_CODE)
}


float sample_shadow(highp sampler2DShadow shadow, vec2 shadow_pixel_size, vec2 pos, float depth, vec4 clamp_rect) {

#ifdef SHADOW_MODE_PCF_13

	float avg=textureProj(shadow,vec4(pos,depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(shadow_pixel_size.x,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(-shadow_pixel_size.x,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,-shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(shadow_pixel_size.x,shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(-shadow_pixel_size.x,shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(shadow_pixel_size.x,-shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(-shadow_pixel_size.x,-shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(shadow_pixel_size.x*2.0,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(-shadow_pixel_size.x*2.0,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,shadow_pixel_size.y*2.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,-shadow_pixel_size.y*2.0),depth,1.0));
	return avg*(1.0/13.0);

#elif defined(SHADOW_MODE_PCF_5)

	float avg=textureProj(shadow,vec4(pos,depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(shadow_pixel_size.x,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(-shadow_pixel_size.x,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,-shadow_pixel_size.y),depth,1.0));
	return avg*(1.0/5.0);

#else

	return textureProj(shadow,vec4(pos,depth,1.0));

#endif

}

#ifdef RENDER_DEPTH_DUAL_PARABOLOID

in highp float dp_clip;

#endif



#if 0
//need to save texture depth for this

vec3 light_transmittance(float translucency,vec3 light_vec, vec3 normal, vec3 pos, float distance) {

	float scale = 8.25 * (1.0 - translucency) / subsurface_scatter_width;
	float d = scale * distance;

    /**
     * Armed with the thickness, we can now calculate the color by means of the
     * precalculated transmittance profile.
     * (It can be precomputed into a texture, for maximum performance):
     */
	float dd = -d * d;
	vec3 profile = vec3(0.233, 0.455, 0.649) * exp(dd / 0.0064) +
		     vec3(0.1,   0.336, 0.344) * exp(dd / 0.0484) +
		     vec3(0.118, 0.198, 0.0)   * exp(dd / 0.187)  +
		     vec3(0.113, 0.007, 0.007) * exp(dd / 0.567)  +
		     vec3(0.358, 0.004, 0.0)   * exp(dd / 1.99)   +
		     vec3(0.078, 0.0,   0.0)   * exp(dd / 7.41);

    /**
     * Using the profile, we finally approximate the transmitted lighting from
     * the back of the object:
     */
    return profile * clamp(0.3 + dot(light_vec, normal),0.0,1.0);
}
#endif

void light_process_omni(int idx, vec3 vertex, vec3 eye_vec,vec3 normal,vec3 binormal, vec3 tangent, vec3 albedo, vec3 transmission, float roughness, float metallic, float rim, float rim_tint, float clearcoat, float clearcoat_gloss, float anisotropy, float p_blob_intensity, inout vec3 diffuse_light, inout vec3 specular_light) {

	vec3 light_rel_vec = omni_lights[idx].light_pos_inv_radius.xyz-vertex;
	float light_length = length( light_rel_vec );
	float normalized_distance = light_length*omni_lights[idx].light_pos_inv_radius.w;
	vec3 light_attenuation = vec3(pow( max(1.0 - normalized_distance, 0.0), omni_lights[idx].light_direction_attenuation.w ));

	if (omni_lights[idx].light_params.w>0.5) {
		//there is a shadowmap

		highp vec3 splane=(omni_lights[idx].shadow_matrix * vec4(vertex,1.0)).xyz;
		float shadow_len=length(splane);
		splane=normalize(splane);
		vec4 clamp_rect=omni_lights[idx].light_clamp;

		if (splane.z>=0.0) {

			splane.z+=1.0;

			clamp_rect.y+=clamp_rect.w;

		} else {

			splane.z=1.0 - splane.z;

			/*
			if (clamp_rect.z<clamp_rect.w) {
				clamp_rect.x+=clamp_rect.z;
			} else {
				clamp_rect.y+=clamp_rect.w;
			}
			*/

		}

		splane.xy/=splane.z;
		splane.xy=splane.xy * 0.5 + 0.5;
		splane.z = shadow_len * omni_lights[idx].light_pos_inv_radius.w;

		splane.xy = clamp_rect.xy+splane.xy*clamp_rect.zw;
		float shadow = sample_shadow(shadow_atlas,shadow_atlas_pixel_size,splane.xy,splane.z,clamp_rect);

#ifdef USE_CONTACT_SHADOWS

		if (shadow>0.01 && omni_lights[idx].shadow_color_contact.a>0.0) {

			float contact_shadow = contact_shadow_compute(vertex,normalize(light_rel_vec),min(light_length,omni_lights[idx].shadow_color_contact.a));
			shadow=min(shadow,contact_shadow);

		}
#endif
		light_attenuation*=mix(omni_lights[idx].shadow_color_contact.rgb,vec3(1.0),shadow);
	}

	light_compute(normal,normalize(light_rel_vec),eye_vec,binormal,tangent,omni_lights[idx].light_color_energy.rgb,light_attenuation,albedo,transmission,omni_lights[idx].light_params.z*p_blob_intensity,roughness,metallic,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);

}

void light_process_spot(int idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 binormal, vec3 tangent,vec3 albedo, vec3 transmission,float roughness, float metallic, float rim, float rim_tint, float clearcoat, float clearcoat_gloss,float anisotropy,float p_blob_intensity, inout vec3 diffuse_light, inout vec3 specular_light) {

	vec3 light_rel_vec = spot_lights[idx].light_pos_inv_radius.xyz-vertex;
	float light_length = length( light_rel_vec );
	float normalized_distance = light_length*spot_lights[idx].light_pos_inv_radius.w;
	vec3 light_attenuation = vec3(pow( max(1.0 - normalized_distance, 0.001), spot_lights[idx].light_direction_attenuation.w ));
	vec3 spot_dir = spot_lights[idx].light_direction_attenuation.xyz;
	float spot_cutoff=spot_lights[idx].light_params.y;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir),spot_cutoff);
	float spot_rim = (1.0 - scos) / (1.0 - spot_cutoff);
	light_attenuation *= 1.0 - pow( max(spot_rim,0.001), spot_lights[idx].light_params.x);

	if (spot_lights[idx].light_params.w>0.5) {
		//there is a shadowmap
		highp vec4 splane=(spot_lights[idx].shadow_matrix * vec4(vertex,1.0));
		splane.xyz/=splane.w;

		float shadow = sample_shadow(shadow_atlas,shadow_atlas_pixel_size,splane.xy,splane.z,spot_lights[idx].light_clamp);

#ifdef USE_CONTACT_SHADOWS
		if (shadow>0.01 && spot_lights[idx].shadow_color_contact.a>0.0) {

			float contact_shadow = contact_shadow_compute(vertex,normalize(light_rel_vec),min(light_length,spot_lights[idx].shadow_color_contact.a));
			shadow=min(shadow,contact_shadow);

		}
#endif
		light_attenuation*=mix(spot_lights[idx].shadow_color_contact.rgb,vec3(1.0),shadow);
	}

	light_compute(normal,normalize(light_rel_vec),eye_vec,binormal,tangent,spot_lights[idx].light_color_energy.rgb,light_attenuation,albedo,transmission,spot_lights[idx].light_params.z*p_blob_intensity,roughness,metallic,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);

}

void reflection_process(int idx, vec3 vertex, vec3 normal,vec3 binormal, vec3 tangent,float roughness,float anisotropy,vec3 ambient,vec3 skybox, inout highp vec4 reflection_accum,inout highp vec4 ambient_accum) {

	vec3 ref_vec = normalize(reflect(vertex,normal));
	vec3 local_pos = (reflections[idx].local_matrix * vec4(vertex,1.0)).xyz;
	vec3 box_extents = reflections[idx].box_extents.xyz;

	if (any(greaterThan(abs(local_pos),box_extents))) { //out of the reflection box
		return;
	}

	vec3 inner_pos = abs(local_pos / box_extents);
	float blend = max(inner_pos.x,max(inner_pos.y,inner_pos.z));
	//make blend more rounded
	blend=mix(length(inner_pos),blend,blend);
	blend*=blend;
	blend=1.001-blend;

	if (reflections[idx].params.x>0.0){// compute reflection

		vec3 local_ref_vec = (reflections[idx].local_matrix * vec4(ref_vec,0.0)).xyz;

		if (reflections[idx].params.w > 0.5) { //box project

			vec3 nrdir = normalize(local_ref_vec);
			vec3 rbmax = (box_extents - local_pos)/nrdir;
			vec3 rbmin = (-box_extents - local_pos)/nrdir;


			vec3 rbminmax = mix(rbmin,rbmax,greaterThan(nrdir,vec3(0.0,0.0,0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_ref_vec = posonbox - reflections[idx].box_offset.xyz;
		}


		vec4 clamp_rect=reflections[idx].atlas_clamp;
		vec3 norm = normalize(local_ref_vec);
		norm.xy/=1.0+abs(norm.z);
		norm.xy=norm.xy * vec2(0.5,0.25) + vec2(0.5,0.25);
		if (norm.z>0.0) {
			norm.y=0.5-norm.y+0.5;
		}

		vec2 atlas_uv =  norm.xy * clamp_rect.zw + clamp_rect.xy;
		atlas_uv = clamp(atlas_uv,clamp_rect.xy,clamp_rect.xy+clamp_rect.zw);

		highp vec4 reflection;
		reflection.rgb = textureLod(reflection_atlas,atlas_uv,roughness*5.0).rgb;

		if (reflections[idx].params.z < 0.5) {
			reflection.rgb = mix(skybox,reflection.rgb,blend);
		}
		reflection.rgb*=reflections[idx].params.x;
		reflection.a = blend;
		reflection.rgb*=reflection.a;

		reflection_accum+=reflection;
	}

	if (reflections[idx].ambient.a>0.0) { //compute ambient using skybox


		vec3 local_amb_vec = (reflections[idx].local_matrix * vec4(normal,0.0)).xyz;

		vec3 splane=normalize(local_amb_vec);
		vec4 clamp_rect=reflections[idx].atlas_clamp;

		splane.z*=-1.0;
		if (splane.z>=0.0) {
			splane.z+=1.0;
			clamp_rect.y+=clamp_rect.w;
		} else {
			splane.z=1.0 - splane.z;
			splane.y=-splane.y;
		}

		splane.xy/=splane.z;
		splane.xy=splane.xy * 0.5 + 0.5;

		splane.xy = splane.xy * clamp_rect.zw + clamp_rect.xy;
		splane.xy = clamp(splane.xy,clamp_rect.xy,clamp_rect.xy+clamp_rect.zw);

		highp vec4 ambient_out;
		ambient_out.a=blend;
		ambient_out.rgb = textureLod(reflection_atlas,splane.xy,5.0).rgb;
		ambient_out.rgb=mix(reflections[idx].ambient.rgb,ambient_out.rgb,reflections[idx].ambient.a);
		if (reflections[idx].params.z < 0.5) {
			ambient_out.rgb = mix(ambient,ambient_out.rgb,blend);
		}

		ambient_out.rgb *= ambient_out.a;
		ambient_accum+=ambient_out;
	} else {

		highp vec4 ambient_out;
		ambient_out.a=blend;
		ambient_out.rgb=reflections[idx].ambient.rgb;
		if (reflections[idx].params.z < 0.5) {
			ambient_out.rgb = mix(ambient,ambient_out.rgb,blend);
		}
		ambient_out.rgb *= ambient_out.a;
		ambient_accum+=ambient_out;

	}
}

#ifdef USE_GI_PROBES

uniform mediump sampler3D gi_probe1; //texunit:-9
uniform highp mat4 gi_probe_xform1;
uniform highp vec3 gi_probe_bounds1;
uniform highp vec3 gi_probe_cell_size1;
uniform highp float gi_probe_multiplier1;
uniform highp float gi_probe_bias1;
uniform highp float gi_probe_normal_bias1;
uniform bool gi_probe_blend_ambient1;

uniform mediump sampler3D gi_probe2; //texunit:-10
uniform highp mat4 gi_probe_xform2;
uniform highp vec3 gi_probe_bounds2;
uniform highp vec3 gi_probe_cell_size2;
uniform highp float gi_probe_multiplier2;
uniform highp float gi_probe_bias2;
uniform highp float gi_probe_normal_bias2;
uniform bool gi_probe2_enabled;
uniform bool gi_probe_blend_ambient2;

vec3 voxel_cone_trace(sampler3D probe, vec3 cell_size, vec3 pos, vec3 ambient, bool blend_ambient, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {

	float dist = p_bias;//1.0; //dot(direction,mix(vec3(-1.0),vec3(1.0),greaterThan(direction,vec3(0.0))))*2.0;
	float alpha=0.0;
	vec3 color = vec3(0.0);

	while(dist < max_distance && alpha < 0.95) {
		float diameter = max(1.0, 2.0 * tan_half_angle * dist);
		vec4 scolor = textureLod(probe, (pos + dist * direction) * cell_size, log2(diameter) );
		float a = (1.0 - alpha);
		color += scolor.rgb * a;
		alpha += a * scolor.a;
		dist += diameter * 0.5;
	}

	if (blend_ambient) {
		color.rgb = mix(ambient,color.rgb,min(1.0,alpha/0.95));
	}

	return color;
}

void gi_probe_compute(sampler3D probe, mat4 probe_xform, vec3 bounds,vec3 cell_size,vec3 pos, vec3 ambient, vec3 environment, bool blend_ambient,float multiplier, mat3 normal_mtx,vec3 ref_vec, float roughness,float p_bias,float p_normal_bias, inout vec4 out_spec, inout vec4 out_diff) {



	vec3 probe_pos = (probe_xform * vec4(pos,1.0)).xyz;
	vec3 ref_pos = (probe_xform * vec4(pos+ref_vec,1.0)).xyz;
	ref_vec = normalize(ref_pos - probe_pos);

	probe_pos+=(probe_xform * vec4(normal_mtx[2],0.0)).xyz*p_normal_bias;

/*	out_diff.rgb = voxel_cone_trace(probe,cell_size,probe_pos,normalize((probe_xform * vec4(ref_vec,0.0)).xyz),0.0 ,100.0);
	out_diff.a = 1.0;
	return;*/
	//out_diff = vec4(textureLod(probe,probe_pos*cell_size,3.0).rgb,1.0);
	//return;

	//this causes corrupted pixels, i have no idea why..
	if (any(bvec2(any(lessThan(probe_pos,vec3(0.0))),any(greaterThan(probe_pos,bounds))))) {
		return;
	}

	//vec3 blendv = probe_pos/bounds * 2.0 - 1.0;
	//float blend = 1.001-max(blendv.x,max(blendv.y,blendv.z));
	float blend=1.0;

	float max_distance = length(bounds);

	//radiance
#ifdef VCT_QUALITY_HIGH

#define MAX_CONE_DIRS 6
	vec3 cone_dirs[MAX_CONE_DIRS] = vec3[] (
		vec3(0, 0, 1),
		vec3(0.866025, 0, 0.5),
		vec3(0.267617, 0.823639, 0.5),
		vec3(-0.700629, 0.509037, 0.5),
		vec3(-0.700629, -0.509037, 0.5),
		vec3(0.267617, -0.823639, 0.5)
	);

	float cone_weights[MAX_CONE_DIRS] = float[](0.25, 0.15, 0.15, 0.15, 0.15, 0.15);
	float cone_angle_tan = 0.577;
	float min_ref_tan = 0.0;
#else

#define MAX_CONE_DIRS 4

	vec3 cone_dirs[MAX_CONE_DIRS] = vec3[] (
			vec3(0.707107, 0, 0.707107),
			vec3(0, 0.707107, 0.707107),
			vec3(-0.707107, 0, 0.707107),
			vec3(0, -0.707107, 0.707107)
	);

	float cone_weights[MAX_CONE_DIRS] = float[](0.25, 0.25, 0.25, 0.25);
	float cone_angle_tan = 0.98269;
	max_distance*=0.5;
	float min_ref_tan = 0.2;

#endif
	vec3 light=vec3(0.0);
	for(int i=0;i<MAX_CONE_DIRS;i++) {

		vec3 dir = normalize( (probe_xform * vec4(pos + normal_mtx * cone_dirs[i],1.0)).xyz - probe_pos);
		light+=cone_weights[i] * voxel_cone_trace(probe,cell_size,probe_pos,ambient,blend_ambient,dir,cone_angle_tan,max_distance,p_bias);

	}

	light*=multiplier;

	out_diff += vec4(light*blend,blend);

	//irradiance

	vec3 irr_light =  voxel_cone_trace(probe,cell_size,probe_pos,environment,blend_ambient,ref_vec,max(min_ref_tan,tan(roughness * 0.5 * M_PI)) ,max_distance,p_bias);

	irr_light *= multiplier;
	//irr_light=vec3(0.0);

	out_spec += vec4(irr_light*blend,blend);

}


void gi_probes_compute(vec3 pos, vec3 normal, float roughness, inout vec3 out_specular, inout vec3 out_ambient) {

	roughness = roughness * roughness;

	vec3 ref_vec = normalize(reflect(normalize(pos),normal));

	//find arbitrary tangent and bitangent, then build a matrix
	vec3 v0 = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0);
	vec3 tangent = normalize(cross(v0, normal));
	vec3 bitangent = normalize(cross(tangent, normal));
	mat3 normal_mat = mat3(tangent,bitangent,normal);

	vec4 diff_accum = vec4(0.0);
	vec4 spec_accum = vec4(0.0);

	vec3 ambient = out_ambient;
	out_ambient = vec3(0.0);

	vec3 environment = out_specular;

	out_specular = vec3(0.0);

	gi_probe_compute(gi_probe1,gi_probe_xform1,gi_probe_bounds1,gi_probe_cell_size1,pos,ambient,environment,gi_probe_blend_ambient1,gi_probe_multiplier1,normal_mat,ref_vec,roughness,gi_probe_bias1,gi_probe_normal_bias1,spec_accum,diff_accum);

	if (gi_probe2_enabled) {

		gi_probe_compute(gi_probe2,gi_probe_xform2,gi_probe_bounds2,gi_probe_cell_size2,pos,ambient,environment,gi_probe_blend_ambient2,gi_probe_multiplier2,normal_mat,ref_vec,roughness,gi_probe_bias2,gi_probe_normal_bias2,spec_accum,diff_accum);
	}

	if (diff_accum.a>0.0) {
		diff_accum.rgb/=diff_accum.a;
	}

	if (spec_accum.a>0.0) {
		spec_accum.rgb/=spec_accum.a;
	}

	out_specular+=spec_accum.rgb;
	out_ambient+=diff_accum.rgb;

}

#endif



void main() {

#ifdef RENDER_DEPTH_DUAL_PARABOLOID

	if (dp_clip>0.0)
		discard;
#endif

	//lay out everything, whathever is unused is optimized away anyway
	highp vec3 vertex = vertex_interp;
	vec3 albedo = vec3(0.8,0.8,0.8);
	vec3 transmission = vec3(0.0);
	float metallic = 0.0;
	float specular = 0.5;
	vec3 emission = vec3(0.0,0.0,0.0);
	float roughness = 1.0;
	float rim = 0.0;
	float rim_tint = 0.0;
	float clearcoat=0.0;
	float clearcoat_gloss=0.0;
	float anisotropy = 1.0;
	vec2 anisotropy_flow = vec2(1.0,0.0);

#if defined(ENABLE_AO)
	float ao=1.0;
	float ao_light_affect=0.0;
#endif

	float alpha = 1.0;

#ifdef METERIAL_DOUBLESIDED
	float side=float(gl_FrontFacing)*2.0-1.0;
#else
	float side=1.0;
#endif


#if defined(ALPHA_SCISSOR_USED)
	float alpha_scissor = 0.5;
#endif

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
	vec3 binormal = normalize(binormal_interp)*side;
	vec3 tangent = normalize(tangent_interp)*side;
#else
	vec3 binormal = vec3(0.0);
	vec3 tangent = vec3(0.0);
#endif
	vec3 normal = normalize(normal_interp)*side;

#if defined(ENABLE_UV_INTERP)
	vec2 uv = uv_interp;
#endif

#if defined(ENABLE_UV2_INTERP)
	vec2 uv2 = uv2_interp;
#endif

#if defined(ENABLE_COLOR_INTERP)
	vec4 color = color_interp;
#endif

#if defined(ENABLE_NORMALMAP)

	vec3 normalmap = vec3(0.0);
#endif

	float normaldepth=1.0;

#if defined(SCREEN_UV_USED)
	vec2 screen_uv = gl_FragCoord.xy*screen_pixel_size;
#endif

#if defined (ENABLE_SSS)
	float sss_strength=0.0;
#endif

{


FRAGMENT_SHADER_CODE

}


#if defined(ALPHA_SCISSOR_USED)
	if (alpha<alpha_scissor) {
		discard;
	}
#endif

#ifdef USE_OPAQUE_PREPASS

	if (alpha<0.99) {
		discard;
	}
#endif

#if defined(ENABLE_NORMALMAP)

	normalmap.xy=normalmap.xy*2.0-1.0;
	normalmap.z=sqrt(1.0-dot(normalmap.xy,normalmap.xy)); //always ignore Z, as it can be RG packed, Z may be pos/neg, etc.

	normal = normalize( mix(normal_interp,tangent * normalmap.x + binormal * normalmap.y + normal * normalmap.z,normaldepth) ) * side;

#endif

#if defined(LIGHT_USE_ANISOTROPY)

	if (anisotropy>0.01) {
		//rotation matrix
		mat3 rot = mat3( tangent, binormal, normal );
		//make local to space
		tangent = normalize(rot * vec3(anisotropy_flow.x,anisotropy_flow.y,0.0));
		binormal = normalize(rot * vec3(-anisotropy_flow.y,anisotropy_flow.x,0.0));
	}

#endif

#ifdef ENABLE_CLIP_ALPHA
	if (albedo.a<0.99) {
		//used for doublepass and shadowmapping
		discard;
	}
#endif

/////////////////////// LIGHTING //////////////////////////////

	//apply energy conservation

#ifdef USE_VERTEX_LIGHTING

	vec3 specular_light = specular_light_interp.rgb;
	vec3 diffuse_light = diffuse_light_interp.rgb;
#else

	vec3 specular_light = vec3(0.0,0.0,0.0);
	vec3 diffuse_light = vec3(0.0,0.0,0.0);

#endif

	vec3 ambient_light;
	vec3 env_reflection_light = vec3(0.0,0.0,0.0);

	vec3 eye_vec = -normalize( vertex_interp );



#ifdef USE_RADIANCE_MAP

	if (no_ambient_light) {
		ambient_light=vec3(0.0,0.0,0.0);
	} else {
		{

			{ //read radiance from dual paraboloid

				vec3 ref_vec = reflect(-eye_vec,normal); //2.0 * ndotv * normal - view; // reflect(v, n);
				ref_vec=normalize((radiance_inverse_xform * vec4(ref_vec,0.0)).xyz);
				vec3 radiance = textureDualParaboloid(radiance_map,ref_vec,roughness) * bg_energy;
				env_reflection_light = radiance;

			}
			//no longer a cubemap
			//vec3 radiance = textureLod(radiance_cube, r, lod).xyz * ( brdf.x + brdf.y);

		}

		{

			vec3 ambient_dir=normalize((radiance_inverse_xform * vec4(normal,0.0)).xyz);
			vec3 env_ambient=textureDualParaboloid(radiance_map,ambient_dir,1.0) * bg_energy;

			ambient_light=mix(ambient_light_color.rgb,env_ambient,radiance_ambient_contribution);
			//ambient_light=vec3(0.0,0.0,0.0);
		}
	}

#else

	if (no_ambient_light){
		ambient_light=vec3(0.0,0.0,0.0);
	} else {
		ambient_light=ambient_light_color.rgb;
	}
#endif

	ambient_light*=ambient_energy;

	float specular_blob_intensity=1.0;
#if defined(SPECULAR_TOON)
	specular_blob_intensity*=specular * 2.0;
#endif

#if defined(USE_LIGHT_DIRECTIONAL)

	vec3 light_attenuation=vec3(1.0);

	float depth_z = -vertex.z;
#ifdef LIGHT_DIRECTIONAL_SHADOW

#ifdef LIGHT_USE_PSSM4
	if (depth_z < shadow_split_offsets.w) {
#elif defined(LIGHT_USE_PSSM2)
	if (depth_z < shadow_split_offsets.y) {
#else
	if (depth_z < shadow_split_offsets.x) {
#endif //LIGHT_USE_PSSM4

	vec3 pssm_coord;
	float pssm_fade=0.0;

#ifdef LIGHT_USE_PSSM_BLEND
	float pssm_blend;
	vec3 pssm_coord2;
	bool use_blend=true;
#endif


#ifdef LIGHT_USE_PSSM4


	if (depth_z < shadow_split_offsets.y) {

		if (depth_z < shadow_split_offsets.x) {

			highp vec4 splane=(shadow_matrix1 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;


#if defined(LIGHT_USE_PSSM_BLEND)

			splane=(shadow_matrix2 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(0.0,shadow_split_offsets.x,depth_z);
#endif

		} else {

			highp vec4 splane=(shadow_matrix2 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;

#if defined(LIGHT_USE_PSSM_BLEND)
			splane=(shadow_matrix3 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(shadow_split_offsets.x,shadow_split_offsets.y,depth_z);
#endif

		}
	} else {


		if (depth_z < shadow_split_offsets.z) {

			highp vec4 splane=(shadow_matrix3 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;

#if defined(LIGHT_USE_PSSM_BLEND)
			splane=(shadow_matrix4 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(shadow_split_offsets.y,shadow_split_offsets.z,depth_z);
#endif

		} else {

			highp vec4 splane=(shadow_matrix4 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;
			pssm_fade = smoothstep(shadow_split_offsets.z,shadow_split_offsets.w,depth_z);

#if defined(LIGHT_USE_PSSM_BLEND)
			use_blend=false;

#endif

		}
	}



#endif //LIGHT_USE_PSSM4

#ifdef LIGHT_USE_PSSM2

	if (depth_z < shadow_split_offsets.x) {

		highp vec4 splane=(shadow_matrix1 * vec4(vertex,1.0));
		pssm_coord=splane.xyz/splane.w;


#if defined(LIGHT_USE_PSSM_BLEND)

		splane=(shadow_matrix2 * vec4(vertex,1.0));
		pssm_coord2=splane.xyz/splane.w;
		pssm_blend=smoothstep(0.0,shadow_split_offsets.x,depth_z);
#endif

	} else {
		highp vec4 splane=(shadow_matrix2 * vec4(vertex,1.0));
		pssm_coord=splane.xyz/splane.w;
		pssm_fade = smoothstep(shadow_split_offsets.x,shadow_split_offsets.y,depth_z);
#if defined(LIGHT_USE_PSSM_BLEND)
		use_blend=false;

#endif

	}

#endif //LIGHT_USE_PSSM2

#if !defined(LIGHT_USE_PSSM4) && !defined(LIGHT_USE_PSSM2)
	{ //regular orthogonal
		highp vec4 splane=(shadow_matrix1 * vec4(vertex,1.0));
		pssm_coord=splane.xyz/splane.w;
	}
#endif


	//one one sample

	float shadow = sample_shadow(directional_shadow,directional_shadow_pixel_size,pssm_coord.xy,pssm_coord.z,light_clamp);

#if defined(LIGHT_USE_PSSM_BLEND)

	if (use_blend) {
		shadow=mix(shadow, sample_shadow(directional_shadow,directional_shadow_pixel_size,pssm_coord2.xy,pssm_coord2.z,light_clamp),pssm_blend);
	}
#endif

#ifdef USE_CONTACT_SHADOWS
	if (shadow>0.01 && shadow_color_contact.a>0.0) {

		float contact_shadow = contact_shadow_compute(vertex,-light_direction_attenuation.xyz,shadow_color_contact.a);
		shadow=min(shadow,contact_shadow);

	}
#endif
	light_attenuation=mix(mix(shadow_color_contact.rgb,vec3(1.0),shadow),vec3(1.0),pssm_fade);


	}


#endif //LIGHT_DIRECTIONAL_SHADOW

#ifdef USE_VERTEX_LIGHTING
	diffuse_light*=mix(vec3(1.0),light_attenuation,diffuse_light_interp.a);
	specular_light*=mix(vec3(1.0),light_attenuation,specular_light_interp.a);

#else
	light_compute(normal,-light_direction_attenuation.xyz,eye_vec,binormal,tangent,light_color_energy.rgb,light_attenuation,albedo,transmission,light_params.z*specular_blob_intensity,roughness,metallic,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);
#endif


#endif //#USE_LIGHT_DIRECTIONAL

#ifdef USE_GI_PROBES
	gi_probes_compute(vertex,normal,roughness,env_reflection_light,ambient_light);

#endif

#ifdef USE_FORWARD_LIGHTING


	highp vec4 reflection_accum = vec4(0.0,0.0,0.0,0.0);
	highp vec4 ambient_accum = vec4(0.0,0.0,0.0,0.0);
	for(int i=0;i<reflection_count;i++) {
		reflection_process(reflection_indices[i],vertex,normal,binormal,tangent,roughness,anisotropy,ambient_light,env_reflection_light,reflection_accum,ambient_accum);
	}

	if (reflection_accum.a>0.0) {
		specular_light+=reflection_accum.rgb/reflection_accum.a;
	} else {
		specular_light+=env_reflection_light;
	}

	if (ambient_accum.a>0.0) {
		ambient_light+=ambient_accum.rgb/ambient_accum.a;
	}



#ifdef USE_VERTEX_LIGHTING

	diffuse_light*=albedo;
#else

	for(int i=0;i<omni_light_count;i++) {
		light_process_omni(omni_light_indices[i],vertex,eye_vec,normal,binormal,tangent,albedo,transmission,roughness,metallic,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,specular_blob_intensity,diffuse_light,specular_light);
	}

	for(int i=0;i<spot_light_count;i++) {
		light_process_spot(spot_light_indices[i],vertex,eye_vec,normal,binormal,tangent,albedo,transmission,roughness,metallic,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,specular_blob_intensity,diffuse_light,specular_light);
	}

#endif //USE_VERTEX_LIGHTING

#endif




#ifdef RENDER_DEPTH
//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else

	specular_light*=reflection_multiplier;
	ambient_light*=albedo; //ambient must be multiplied by albedo at the end

#if defined(ENABLE_AO)
	ambient_light*=ao;
	ao_light_affect = mix(1.0,ao,ao_light_affect);
	specular_light*=ao_light_affect;
	diffuse_light*=ao_light_affect;
#endif



	//energy conservation
	diffuse_light *= 1.0-metallic; // TODO: avoid all diffuse and ambient light calculations when metallic == 1 up to this point
	ambient_light *= 1.0-metallic;


	{

#if defined(DIFFUSE_TOON)
		//simplify for toon, as
		specular_light *= specular * metallic * albedo * 2.0;
#else
		// Environment brdf approximation (Lazarov 2013)
		// see https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
		const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
		const vec4 c1 = vec4( 1.0, 0.0425, 1.04, -0.04);
		vec4 r = roughness * c0 + c1;
		float ndotv = clamp(dot(normal,eye_vec),0.0,1.0);
		float a004 = min( r.x * r.x, exp2( -9.28 * ndotv ) ) * r.x + r.y;
		vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;

		vec3 specular_color = metallic_to_specular_color(metallic, specular, albedo);
		specular_light *= AB.x * specular_color + AB.y;
#endif

	}

	if (fog_color_enabled.a > 0.5) {

		float fog_amount=0.0;



#ifdef USE_LIGHT_DIRECTIONAL

		vec3 fog_color = mix( fog_color_enabled.rgb, fog_sun_color_amount.rgb,fog_sun_color_amount.a * pow(max( dot(normalize(vertex),-light_direction_attenuation.xyz), 0.0),8.0) );
#else

		vec3 fog_color = fog_color_enabled.rgb;
#endif

		//apply fog

		if (fog_depth_enabled) {

			float fog_z = smoothstep(fog_depth_begin,z_far,length(vertex));

			fog_amount = pow(fog_z,fog_depth_curve);
			if (fog_transmit_enabled) {
				vec3 total_light = emission + ambient_light + specular_light + diffuse_light;
				float transmit = pow(fog_z,fog_transmit_curve);
				fog_color = mix(max(total_light,fog_color),fog_color,transmit);
			}
		}

		if (fog_height_enabled) {
			float y = (camera_matrix * vec4(vertex,1.0)).y;
			fog_amount = max(fog_amount,pow(1.0-smoothstep(fog_height_min,fog_height_max,y),fog_height_curve));
		}

		float rev_amount = 1.0 - fog_amount;


		emission = emission * rev_amount + fog_color * fog_amount;
		ambient_light*=rev_amount;
		specular_light*rev_amount;
		diffuse_light*=rev_amount;

	}

#ifdef USE_MULTIPLE_RENDER_TARGETS


#ifdef SHADELESS
	diffuse_buffer=vec4(albedo.rgb,0.0);
	specular_buffer=vec4(0.0);

#else

#if defined(ENABLE_AO)

	float ambient_scale=0.0; // AO is supplied by material
#else
	//approximate ambient scale for SSAO, since we will lack full ambient
	float max_emission=max(emission.r,max(emission.g,emission.b));
	float max_ambient=max(ambient_light.r,max(ambient_light.g,ambient_light.b));
	float max_diffuse=max(diffuse_light.r,max(diffuse_light.g,diffuse_light.b));
	float total_ambient = max_ambient+max_diffuse+max_emission;
	float ambient_scale = (total_ambient>0.0) ? (max_ambient+ambient_occlusion_affect_light*max_diffuse)/total_ambient : 0.0;
#endif //ENABLE_AO

	diffuse_buffer=vec4(emission+diffuse_light+ambient_light,ambient_scale);
	specular_buffer=vec4(specular_light,metallic);

#endif //SHADELESS

	normal_mr_buffer=vec4(normalize(normal)*0.5+0.5,roughness);

#if defined (ENABLE_SSS)
	sss_buffer = sss_strength;
#endif


#else //USE_MULTIPLE_RENDER_TARGETS


#ifdef SHADELESS
	frag_color=vec4(albedo,alpha);
#else
	frag_color=vec4(emission+ambient_light+diffuse_light+specular_light,alpha);
#endif //SHADELESS


#endif //USE_MULTIPLE_RENDER_TARGETS



#endif //RENDER_DEPTH


}
