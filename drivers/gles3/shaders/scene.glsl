[vertex]


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
	highp mat4 camera_inverse_matrix;
	highp mat4 camera_matrix;
	highp vec4 time;

	highp vec4 ambient_light_color;
	highp vec4 bg_color;
	float ambient_energy;
	float bg_energy;

	float shadow_z_offset;
	float shadow_z_slope_scale;
	float shadow_dual_paraboloid_render_zfar;
	float shadow_dual_paraboloid_render_side;

	highp vec2 screen_pixel_size;
	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;

	float reflection_multiplier;
	float subsurface_scatter_width;
	float ambient_occlusion_affect_light;

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


VERTEX_SHADER_GLOBALS


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:1

MATERIAL_UNIFORMS

};

#endif

#ifdef RENDER_DEPTH_DUAL_PARABOLOID

out highp float dp_clip;

#endif

#define SKELETON_TEXTURE_WIDTH 256

#ifdef USE_SKELETON
uniform highp sampler2D skeleton_texture; //texunit:-6
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
		tangent.xyz = vec4(tangent.xyz,0.0) * mn;
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

//defines that make writing custom shaders easier
#define projection_matrix local_projection
#define world_transform world_matrix
{

VERTEX_SHADER_CODE

}




#if !defined(SKIP_TRANSFORM_USED)

	vertex = modelview * vertex;
	normal = normalize((modelview * vec4(normal,0.0)).xyz);
#endif


	vertex_interp = vertex.xyz;
	normal_interp = normal;

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)

#if !defined(SKIP_TRANSFORM_USED)

	tangent = normalize((modelview * vec4(tangent,0.0)).xyz);
	binormal = normalize((modelview * vec4(binormal,0.0)).xyz);

#endif
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

#ifdef RENDER_DEPTH


#ifdef RENDER_DEPTH_DUAL_PARABOLOID

	vertex_interp.z*= shadow_dual_paraboloid_render_side;
	normal_interp.z*= shadow_dual_paraboloid_render_side;

	dp_clip=vertex_interp.z; //this attempts to avoid noise caused by objects sent to the other parabolloid side due to bias

	//for dual paraboloid shadow mapping, this is the fastest but least correct way, as it curves straight edges

	highp vec3 vtx = vertex_interp+normalize(vertex_interp)*shadow_z_offset;
	highp float distance = length(vtx);
	vtx = normalize(vtx);
	vtx.xy/=1.0-vtx.z;
	vtx.z=(distance/shadow_dual_paraboloid_render_zfar);
	vtx.z=vtx.z * 2.0 - 1.0;

	vertex.xyz=vtx;
	vertex.w=1.0;


#else

	float z_ofs = shadow_z_offset;
	z_ofs += (1.0-abs(normal_interp.z))*shadow_z_slope_scale;
	vertex_interp.z-=z_ofs;

#endif //RENDER_DEPTH_DUAL_PARABOLOID

#endif //RENDER_DEPTH


#if !defined(SKIP_TRANSFORM_USED) && !defined(RENDER_DEPTH_DUAL_PARABOLOID)
	gl_Position = projection_matrix * vec4(vertex_interp,1.0);
#else
	gl_Position = vertex;
#endif

	position_interp=gl_Position;
}


[fragment]



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

uniform sampler2D brdf_texture; //texunit:-1

#ifdef USE_RADIANCE_MAP

uniform sampler2D radiance_map; //texunit:-2


layout(std140) uniform Radiance { //ubo:2

	mat4 radiance_inverse_xform;
	vec3 radiance_box_min;
	vec3 radiance_box_max;
	float radiance_ambient_contribution;

};

#endif

/* Material Uniforms */


FRAGMENT_SHADER_GLOBALS


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData {

MATERIAL_UNIFORMS

};

#endif


layout(std140) uniform SceneData {

	highp mat4 projection_matrix;
	highp mat4 camera_inverse_matrix;
	highp mat4 camera_matrix;
	highp vec4 time;

	highp vec4 ambient_light_color;
	highp vec4 bg_color;


	float ambient_energy;
	float bg_energy;

	float shadow_z_offset;
	float shadow_z_slope_scale;
	float shadow_dual_paraboloid_render_zfar;
	float shadow_dual_paraboloid_render_side;

	highp vec2 screen_pixel_size;
	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;

	float reflection_multiplier;
	float subsurface_scatter_width;
	float ambient_occlusion_affect_light;

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


uniform highp sampler2DShadow shadow_atlas; //texunit:-3


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
uniform mediump sampler2D reflection_atlas; //texunit:-5


#ifdef USE_FORWARD_LIGHTING

uniform int omni_light_indices[MAX_FORWARD_LIGHTS];
uniform int omni_light_count;

uniform int spot_light_indices[MAX_FORWARD_LIGHTS];
uniform int spot_light_count;

uniform int reflection_indices[MAX_FORWARD_LIGHTS];
uniform int reflection_count;

#endif



#ifdef USE_MULTIPLE_RENDER_TARGETS

layout(location=0) out vec4 diffuse_buffer;
layout(location=1) out vec4 specular_buffer;
layout(location=2) out vec4 normal_mr_buffer;
#if defined (ENABLE_SSS_MOTION)
layout(location=3) out vec4 motion_ssr_buffer;
#endif

#else

layout(location=0) out vec4 frag_color;

#endif

in highp vec4 position_interp;
uniform highp sampler2D depth_buffer; //texunit:-9

float contact_shadow_compute(vec3 pos, vec3 dir, float max_distance) {

	if (abs(dir.z)>0.99)
		return 1.0;

	vec3 endpoint = pos+dir*max_distance;
	vec4 source = position_interp;
	vec4 dest = projection_matrix * vec4(endpoint, 1.0);

	vec2 from_screen = (source.xy / source.w) * 0.5 + 0.5;
	vec2 to_screen = (dest.xy / dest.w) * 0.5 + 0.5;

	vec2 screen_rel = to_screen - from_screen;

	/*float pixel_size; //approximate pixel size

	if (screen_rel.x > screen_rel.y) {

		pixel_size = abs((pos.x-endpoint.x)/(screen_rel.x/screen_pixel_size.x));
	} else {
		pixel_size = abs((pos.y-endpoint.y)/(screen_rel.y/screen_pixel_size.y));

	}*/
	vec4 bias = projection_matrix * vec4(pos+vec3(0.0,0.0,0.04), 1.0); //todo un-harcode the 0.04



	vec2 pixel_incr = normalize(screen_rel)*screen_pixel_size;

	float steps = length(screen_rel) / length(pixel_incr);

	//steps=10.0;

	vec4 incr = (dest - source)/steps;
	float ratio=0.0;
	float ratio_incr = 1.0/steps;

	do {
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
	} while (steps>0.0);

	return 1.0;
}


// GGX Specular
// Source: http://www.filmicworlds.com/images/ggx-opt/optimized-ggx.hlsl
float G1V(float dotNV, float k)
{
    return 1.0 / (dotNV * (1.0 - k) + k);
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



void light_compute(vec3 N, vec3 L,vec3 V,vec3 B, vec3 T,vec3 light_color,vec3 diffuse_color, vec3 specular_color, float specular_blob_intensity, float roughness, float rim,float rim_tint, float clearcoat, float clearcoat_gloss,float anisotropy,inout vec3 diffuse, inout vec3 specular) {

	float dotNL = max(dot(N,L), 0.0 );
	float dotNV = max(dot(N,V), 0.0 );

#if defined(LIGHT_USE_RIM)
	float rim_light = pow(1.0-dotNV,(1.0-roughness)*16.0);
	diffuse += rim_light * rim * mix(vec3(1.0),diffuse_color,rim_tint) * light_color;
#endif

	diffuse += dotNL * light_color * diffuse_color;

	if (roughness > 0.0) {

		float alpha = roughness * roughness;

		vec3 H = normalize(V + L);

		float dotNH = max(dot(N,H), 0.0 );
		float dotLH = max(dot(L,H), 0.0 );

		// D
#if defined(LIGHT_USE_ANISOTROPY)

		float aspect = sqrt(1.0-anisotropy*0.9);
		float rx = roughness/aspect;
		float ry = roughness*aspect;
		float ax = rx*rx;
		float ay = ry*ry;
		float dotXH = dot( T, H );
		float dotYH = dot( B, H );
		float pi = M_PI;
		float denom = dotXH*dotXH / (ax*ax) + dotYH*dotYH / (ay*ay) + dotNH*dotNH;
		float D = 1.0 / ( pi * ax*ay * denom*denom );

#else
		float alphaSqr = alpha * alpha;
		float pi = M_PI;
		float denom = dotNH * dotNH * (alphaSqr - 1.0) + 1.0;
		float D = alphaSqr / (pi * denom * denom);
#endif
		// F
		float F0 = 1.0;
		float dotLH5 = SchlickFresnel( dotLH );
		float F = F0 + (1.0 - F0) * (dotLH5);

		// V
		float k = alpha / 2.0f;
		float vis = G1V(dotNL, k) * G1V(dotNV, k);

		float speci = dotNL * D * F * vis;

		specular += speci * light_color /* specular_color*/ * specular_blob_intensity;

#if defined(LIGHT_USE_CLEARCOAT)
		float Dr = GTR1(dotNH, mix(.1,.001,clearcoat_gloss));
		float Fr = mix(.04, 1.0, dotLH5);
		float Gr = G1V(dotNL, .25) * G1V(dotNV, .25);

		specular += .25*clearcoat*Gr*Fr*Dr;
#endif
	}


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

#endif

#ifdef SHADOW_MODE_PCF_5

	float avg=textureProj(shadow,vec4(pos,depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(shadow_pixel_size.x,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(-shadow_pixel_size.x,0.0),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,shadow_pixel_size.y),depth,1.0));
	avg+=textureProj(shadow,vec4(pos+vec2(0.0,-shadow_pixel_size.y),depth,1.0));
	return avg*(1.0/5.0);
#endif

#if !defined(SHADOW_MODE_PCF_5) && !defined(SHADOW_MODE_PCF_13)

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

void light_process_omni(int idx, vec3 vertex, vec3 eye_vec,vec3 normal,vec3 binormal, vec3 tangent, vec3 albedo, vec3 specular, float roughness, float rim, float rim_tint, float clearcoat, float clearcoat_gloss,float anisotropy,inout vec3 diffuse_light, inout vec3 specular_light) {

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
		if (shadow>0.01 && omni_lights[idx].shadow_color_contact.a>0.0) {

			float contact_shadow = contact_shadow_compute(vertex,normalize(light_rel_vec),min(light_length,omni_lights[idx].shadow_color_contact.a));
			shadow=min(shadow,contact_shadow);


		}
		light_attenuation*=mix(omni_lights[idx].shadow_color_contact.rgb,vec3(1.0),shadow);
	}

	light_compute(normal,normalize(light_rel_vec),eye_vec,binormal,tangent,omni_lights[idx].light_color_energy.rgb*light_attenuation,albedo,specular,omni_lights[idx].light_params.z,roughness,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);

}

void light_process_spot(int idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 binormal, vec3 tangent,vec3 albedo, vec3 specular, float roughness, float rim,float rim_tint, float clearcoat, float clearcoat_gloss,float anisotropy, inout vec3 diffuse_light, inout vec3 specular_light) {

	vec3 light_rel_vec = spot_lights[idx].light_pos_inv_radius.xyz-vertex;
	float light_length = length( light_rel_vec );
	float normalized_distance = light_length*spot_lights[idx].light_pos_inv_radius.w;
	vec3 light_attenuation = vec3(pow( max(1.0 - normalized_distance, 0.0), spot_lights[idx].light_direction_attenuation.w ));
	vec3 spot_dir = spot_lights[idx].light_direction_attenuation.xyz;
	float spot_cutoff=spot_lights[idx].light_params.y;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir),spot_cutoff);
	float spot_rim = (1.0 - scos) / (1.0 - spot_cutoff);
	light_attenuation *= 1.0 - pow( spot_rim, spot_lights[idx].light_params.x);

	if (spot_lights[idx].light_params.w>0.5) {
		//there is a shadowmap
		highp vec4 splane=(spot_lights[idx].shadow_matrix * vec4(vertex,1.0));
		splane.xyz/=splane.w;

		float shadow = sample_shadow(shadow_atlas,shadow_atlas_pixel_size,splane.xy,splane.z,spot_lights[idx].light_clamp);

		if (shadow>0.01 && spot_lights[idx].shadow_color_contact.a>0.0) {

			float contact_shadow = contact_shadow_compute(vertex,normalize(light_rel_vec),min(light_length,spot_lights[idx].shadow_color_contact.a));
			shadow=min(shadow,contact_shadow);

		}

		light_attenuation*=mix(spot_lights[idx].shadow_color_contact.rgb,vec3(1.0),shadow);
	}

	light_compute(normal,normalize(light_rel_vec),eye_vec,binormal,tangent,spot_lights[idx].light_color_energy.rgb*light_attenuation,albedo,specular,spot_lights[idx].light_params.z,roughness,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);

}

void reflection_process(int idx, vec3 vertex, vec3 normal,vec3 binormal, vec3 tangent,float roughness,float anisotropy,vec3 ambient,vec3 skybox,vec2 brdf, inout highp vec4 reflection_accum,inout highp vec4 ambient_accum) {

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



		vec3 splane=normalize(local_ref_vec);
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

		highp vec4 reflection;
		reflection.rgb = textureLod(reflection_atlas,splane.xy,roughness*5.0).rgb *  brdf.x + brdf.y;

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

uniform mediump sampler3D gi_probe1; //texunit:-11
uniform highp mat4 gi_probe_xform1;
uniform highp vec3 gi_probe_bounds1;
uniform highp vec3 gi_probe_cell_size1;
uniform highp float gi_probe_multiplier1;
uniform highp float gi_probe_bias1;
uniform bool gi_probe_blend_ambient1;

uniform mediump sampler3D gi_probe2; //texunit:-10
uniform highp mat4 gi_probe_xform2;
uniform highp vec3 gi_probe_bounds2;
uniform highp vec3 gi_probe_cell_size2;
uniform highp float gi_probe_multiplier2;
uniform highp float gi_probe_bias2;
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

void gi_probe_compute(sampler3D probe, mat4 probe_xform, vec3 bounds,vec3 cell_size,vec3 pos, vec3 ambient, vec3 environment, bool blend_ambient,float multiplier, mat3 normal_mtx,vec3 ref_vec, float roughness,float p_bias, out vec4 out_spec, out vec4 out_diff) {



	vec3 probe_pos = (probe_xform * vec4(pos,1.0)).xyz;
	vec3 ref_pos = (probe_xform * vec4(pos+ref_vec,1.0)).xyz;

	ref_vec = normalize(ref_pos - probe_pos);

/*	out_diff.rgb = voxel_cone_trace(probe,cell_size,probe_pos,normalize((probe_xform * vec4(ref_vec,0.0)).xyz),0.0 ,100.0);
	out_diff.a = 1.0;
	return;*/
	//out_diff = vec4(textureLod(probe,probe_pos*cell_size,3.0).rgb,1.0);
	//return;

	if (any(bvec2(any(lessThan(probe_pos,vec3(0.0))),any(greaterThan(probe_pos,bounds)))))
		return;

	vec3 blendv = probe_pos/bounds * 2.0 - 1.0;
	float blend = 1.001-max(blendv.x,max(blendv.y,blendv.z));
	blend=1.0;

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

	out_diff = vec4(light*blend,blend);

	//irradiance

	vec3 irr_light =  voxel_cone_trace(probe,cell_size,probe_pos,environment,blend_ambient,ref_vec,max(min_ref_tan,tan(roughness * 0.5 * M_PI)) ,max_distance,p_bias);

	irr_light *= multiplier;
	//irr_light=vec3(0.0);

	out_spec = vec4(irr_light*blend,blend);
}


void gi_probes_compute(vec3 pos, vec3 normal, float roughness, vec3 specular, inout vec3 out_specular, inout vec3 out_ambient) {

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

	gi_probe_compute(gi_probe1,gi_probe_xform1,gi_probe_bounds1,gi_probe_cell_size1,pos,ambient,environment,gi_probe_blend_ambient1,gi_probe_multiplier1,normal_mat,ref_vec,roughness,gi_probe_bias1,spec_accum,diff_accum);

	if (gi_probe2_enabled) {

		gi_probe_compute(gi_probe2,gi_probe_xform2,gi_probe_bounds2,gi_probe_cell_size2,pos,ambient,environment,gi_probe_blend_ambient2,gi_probe_multiplier2,normal_mat,ref_vec,roughness,gi_probe_bias2,spec_accum,diff_accum);
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
	vec3 specular = vec3(0.2,0.2,0.2);
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
#endif

	float alpha = 1.0;

#ifdef METERIAL_DOUBLESIDED
	float side=float(gl_FrontFacing)*2.0-1.0;
#else
	float side=1.0;
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



#if defined(ENABLE_DISCARD)
	bool discard_=false;
#endif

#if defined (ENABLE_SSS_MOTION)
	float sss_strength=0.0;
#endif

{


FRAGMENT_SHADER_CODE

}



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

#if defined(ENABLE_DISCARD)
	if (discard_) {
	//easy to eliminate dead code
		discard;
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

	vec3 specular_light = vec3(0.0,0.0,0.0);
	vec3 ambient_light;
	vec3 diffuse_light = vec3(0.0,0.0,0.0);

	vec3 eye_vec = -normalize( vertex_interp );

#ifndef RENDER_DEPTH
	float ndotv = clamp(dot(normal,eye_vec),0.0,1.0);

	vec2 brdf = texture(brdf_texture, vec2(roughness, ndotv)).xy;
#endif

#ifdef USE_RADIANCE_MAP

	if (no_ambient_light) {
		ambient_light=vec3(0.0,0.0,0.0);
	} else {
		{



			float lod = roughness * 5.0;

			{ //read radiance from dual paraboloid

				vec3 ref_vec = reflect(-eye_vec,normal); //2.0 * ndotv * normal - view; // reflect(v, n);
				ref_vec=normalize((radiance_inverse_xform * vec4(ref_vec,0.0)).xyz);

				vec3 norm = normalize(ref_vec);
				float y_ofs=0.0;
				if (norm.z>=0.0) {

					norm.z+=1.0;
					y_ofs+=0.5;
				} else {
					norm.z=1.0 - norm.z;
					norm.y=-norm.y;
				}

				norm.xy/=norm.z;
				norm.xy=norm.xy * vec2(0.5,0.25) + vec2(0.5,0.25+y_ofs);
				specular_light = textureLod(radiance_map, norm.xy, lod).xyz * brdf.x + brdf.y;

			}
			//no longer a cubemap
			//vec3 radiance = textureLod(radiance_cube, r, lod).xyz * ( brdf.x + brdf.y);

		}

		{

			/*vec3 ambient_dir=normalize((radiance_inverse_xform * vec4(normal,0.0)).xyz);
			vec3 env_ambient=textureLod(radiance_cube, ambient_dir, 5.0).xyz;

			ambient_light=mix(ambient_light_color.rgb,env_ambient,radiance_ambient_contribution);*/
			ambient_light=vec3(0.0,0.0,0.0);
		}
	}

#else

	if (no_ambient_light){
		ambient_light=vec3(0.0,0.0,0.0);
	} else {
		ambient_light=ambient_light_color.rgb;
	}
#endif


#ifdef USE_LIGHT_DIRECTIONAL

	vec3 light_attenuation=vec3(1.0);

#ifdef LIGHT_DIRECTIONAL_SHADOW

	if (gl_FragCoord.w > shadow_split_offsets.w) {

	vec3 pssm_coord;

#ifdef LIGHT_USE_PSSM_BLEND
	float pssm_blend;
	vec3 pssm_coord2;
	bool use_blend=true;
	vec3 light_pssm_split_inv = 1.0/shadow_split_offsets.xyz;
	float w_inv = 1.0/gl_FragCoord.w;
#endif


#ifdef LIGHT_USE_PSSM4


	if (gl_FragCoord.w > shadow_split_offsets.y) {

		if (gl_FragCoord.w > shadow_split_offsets.x) {

			highp vec4 splane=(shadow_matrix1 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;


#if defined(LIGHT_USE_PSSM_BLEND)

			splane=(shadow_matrix2 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(0.0,light_pssm_split_inv.x,w_inv);
#endif

		} else {

			highp vec4 splane=(shadow_matrix2 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;

#if defined(LIGHT_USE_PSSM_BLEND)
			splane=(shadow_matrix3 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(light_pssm_split_inv.x,light_pssm_split_inv.y,w_inv);
#endif

		}
	} else {


		if (gl_FragCoord.w > shadow_split_offsets.z) {

			highp vec4 splane=(shadow_matrix3 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;

#if defined(LIGHT_USE_PSSM_BLEND)
			splane=(shadow_matrix4 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(light_pssm_split_inv.y,light_pssm_split_inv.z,w_inv);
#endif

		} else {
			highp vec4 splane=(shadow_matrix4 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;

#if defined(LIGHT_USE_PSSM_BLEND)
			use_blend=false;

#endif

		}
	}



#endif //LIGHT_USE_PSSM4

#ifdef LIGHT_USE_PSSM2

	if (gl_FragCoord.w > shadow_split_offsets.x) {

		highp vec4 splane=(shadow_matrix1 * vec4(vertex,1.0));
		pssm_coord=splane.xyz/splane.w;


#if defined(LIGHT_USE_PSSM_BLEND)

		splane=(shadow_matrix2 * vec4(vertex,1.0));
		pssm_coord2=splane.xyz/splane.w;
		pssm_blend=smoothstep(0.0,light_pssm_split_inv.x,w_inv);
#endif

	} else {
		highp vec4 splane=(shadow_matrix2 * vec4(vertex,1.0));
		pssm_coord=splane.xyz/splane.w;
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
		shadow=mix(shadow, sample_shadow(directional_shadow,directional_shadow_pixel_size,pssm_coord2.xy,pssm_coord2.z,light_clamp));
	}
#endif

	if (shadow>0.01 && shadow_color_contact.a>0.0) {

		float contact_shadow = contact_shadow_compute(vertex,-light_direction_attenuation.xyz,shadow_color_contact.a);
		shadow=min(shadow,contact_shadow);

	}

	light_attenuation=mix(shadow_color_contact.rgb,vec3(1.0),shadow);


	}

#endif //LIGHT_DIRECTIONAL_SHADOW

	light_compute(normal,-light_direction_attenuation.xyz,eye_vec,binormal,tangent,light_color_energy.rgb*light_attenuation,albedo,specular,light_params.z,roughness,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);


#endif //#USE_LIGHT_DIRECTIONAL

#ifdef USE_GI_PROBES
	gi_probes_compute(vertex,normal,roughness,specular,specular_light,ambient_light);
#endif


#ifdef USE_FORWARD_LIGHTING

	highp vec4 reflection_accum = vec4(0.0,0.0,0.0,0.0);
	highp vec4 ambient_accum = vec4(0.0,0.0,0.0,0.0);



	for(int i=0;i<reflection_count;i++) {
		reflection_process(reflection_indices[i],vertex,normal,binormal,tangent,roughness,anisotropy,ambient_light,specular_light,brdf,reflection_accum,ambient_accum);
	}

	if (reflection_accum.a>0.0) {
		specular_light+=reflection_accum.rgb/reflection_accum.a;
	}
	if (ambient_accum.a>0.0) {
		ambient_light+=ambient_accum.rgb/ambient_accum.a;
	}

	for(int i=0;i<omni_light_count;i++) {
		light_process_omni(omni_light_indices[i],vertex,eye_vec,normal,binormal,tangent,albedo,specular,roughness,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);
	}

	for(int i=0;i<spot_light_count;i++) {
		light_process_spot(spot_light_indices[i],vertex,eye_vec,normal,binormal,tangent,albedo,specular,roughness,rim,rim_tint,clearcoat,clearcoat_gloss,anisotropy,diffuse_light,specular_light);
	}



#endif




#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader
{

LIGHT_SHADER_CODE

}
#endif

#ifdef RENDER_DEPTH
//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else

	specular_light*=reflection_multiplier;
	ambient_light*=albedo; //ambient must be multiplied by albedo at the end

#if defined(ENABLE_AO)
	ambient_light*=ao;
#endif

	//energy conservation
	diffuse_light=mix(diffuse_light,vec3(0.0),specular);
	ambient_light=mix(ambient_light,vec3(0.0),specular);
	specular_light *= max(vec3(0.04),specular);

#ifdef USE_MULTIPLE_RENDER_TARGETS

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
	specular_buffer=vec4(specular_light,max(specular.r,max(specular.g,specular.b)));


	normal_mr_buffer=vec4(normalize(normal)*0.5+0.5,roughness);

#if defined (ENABLE_SSS_MOTION)
	motion_ssr_buffer = vec4(vec3(0.0),sss_strength);
#endif

#else


#ifdef SHADELESS
	frag_color=vec4(albedo,alpha);
#else
	frag_color=vec4(emission+ambient_light+diffuse_light+specular_light,alpha);
#endif //SHADELESS


#endif //USE_MULTIPLE_RENDER_TARGETS



#endif //RENDER_DEPTH


}


