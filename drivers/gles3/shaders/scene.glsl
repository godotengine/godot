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
layout(location=2) in vec4 tangent_attrib;
layout(location=3) in vec4 color_attrib;
layout(location=4) in vec2 uv_attrib;
layout(location=5) in vec2 uv2_attrib;

uniform float normal_mult;

#ifdef USE_SKELETON
layout(location=6) mediump ivec4 bone_indices; // attrib:6
layout(location=7) mediump vec4 bone_weights; // attrib:7
uniform highp sampler2D skeleton_matrices;
#endif

#ifdef USE_ATTRIBUTE_INSTANCING

layout(location=8) in highp vec4 instance_xform0;
layout(location=9) in highp vec4 instance_xform1;
layout(location=10) in highp vec4 instance_xform2;
layout(location=11) in lowp vec4 instance_color;

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

	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;

};

uniform highp mat4 world_transform;

#ifdef USE_FORWARD_LIGHTING

layout(std140) uniform LightData { //ubo:3

	highp vec4 light_pos_inv_radius;
	mediump vec4 light_direction_attenuation;
	mediump vec4 light_color_energy;
	mediump vec4 light_params; //cone attenuation, specular, shadow darkening,
	mediump vec4 light_clamp;
	mediump vec4 shadow_split_offsets;
	highp mat4 shadow_matrix1;
	highp mat4 shadow_matrix2;
	highp mat4 shadow_matrix3;
	highp mat4 shadow_matrix4;
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


#if defined(ENABLE_TANGENT_INTERP)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif


#if !defined(USE_DEPTH_SHADOWS) && defined(USE_SHADOW_PASS)

varying vec4 position_interp;

#endif


VERTEX_SHADER_GLOBALS


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:1

MATERIAL_UNIFORMS

};

#endif

#ifdef RENDER_SHADOW_DUAL_PARABOLOID

out highp float dp_clip;

#endif

void main() {

	highp vec4 vertex = vertex_attrib; // vec4(vertex_attrib.xyz * data_attrib.x,1.0);
	highp mat4 modelview = camera_inverse_matrix * world_transform;
	vec3 normal = normal_attrib * normal_mult;

#if defined(ENABLE_TANGENT_INTERP)
	vec3 tangent = tangent_attrib.xyz;
	tangent*=normal_mult;
	float binormalf = tangent_attrib.a;
#endif

#ifdef USE_SKELETON

	{
		//skeleton transform
		highp mat4 m=mat4(texture2D(skeleton_matrices,vec2((bone_indices.x*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.x*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.x*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.x;
		m+=mat4(texture2D(skeleton_matrices,vec2((bone_indices.y*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.y*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.y*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.y;
		m+=mat4(texture2D(skeleton_matrices,vec2((bone_indices.z*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.z*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.z*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.z;
		m+=mat4(texture2D(skeleton_matrices,vec2((bone_indices.w*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.w*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.w*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.w;

		vertex = vertex_in * m;
		normal = (vec4(normal,0.0) * m).xyz;
#if defined(ENABLE_TANGENT_INTERP)
		tangent = (vec4(tangent,0.0) * m).xyz;
#endif
	}

#endif

#if !defined(SKIP_TRANSFORM_USED)

	vertex = modelview * vertex;
	normal = normalize((modelview * vec4(normal,0.0)).xyz);
#endif

#if defined(ENABLE_TANGENT_INTERP)
# if !defined(SKIP_TRANSFORM_USED)

	tangent=normalize((modelview * vec4(tangent,0.0)).xyz);
# endif
	vec3 binormal = normalize( cross(normal,tangent) * binormalf );
#endif



#if defined(ENABLE_COLOR_INTERP)
	color_interp = color_attrib;
#endif

#if defined(ENABLE_UV_INTERP)
	uv_interp = uv_attrib;
#endif

#if defined(ENABLE_UV2_INTERP)
	uv2_interp = uv2_attrib;
#endif

{

VERTEX_SHADER_CODE

}

	vertex_interp = vertex.xyz;
	normal_interp = normal;

#if defined(ENABLE_TANGENT_INTERP)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

#ifdef RENDER_SHADOW


#ifdef RENDER_SHADOW_DUAL_PARABOLOID

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

#endif //RENDER_SHADOW_DUAL_PARABOLOID

#endif //RENDER_SHADOW


#if !defined(SKIP_TRANSFORM_USED) && !defined(RENDER_SHADOW_DUAL_PARABOLOID)
	gl_Position = projection_matrix * vec4(vertex_interp,1.0);
#else
	gl_Position = vertex;
#endif


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

#if defined(ENABLE_TANGENT_INTERP)
in vec3 tangent_interp;
in vec3 binormal_interp;
#endif

in highp vec3 vertex_interp;
in vec3 normal_interp;


/* PBR CHANNELS */

//used on forward mainly
uniform bool no_ambient_light;


#ifdef USE_RADIANCE_CUBEMAP

uniform sampler2D brdf_texture; //texunit:-1
uniform samplerCube radiance_cube; //texunit:-2


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

	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;

};


#ifdef USE_FORWARD_LIGHTING

layout(std140) uniform LightData {

	highp vec4 light_pos_inv_radius;
	mediump vec4 light_direction_attenuation;
	mediump vec4 light_color_energy;
	mediump vec4 light_params; //cone attenuation, specular, shadow darkening, shadow enabled
	mediump vec4 light_clamp;
	mediump vec4 shadow_split_offsets;
	highp mat4 shadow_matrix1;
	highp mat4 shadow_matrix2;
	highp mat4 shadow_matrix3;
	highp mat4 shadow_matrix4;
};

#endif


uniform highp sampler2DShadow directional_shadow; //texunit:-4
uniform highp sampler2DShadow shadow_atlas; //texunit:-3


#ifdef USE_MULTIPLE_RENDER_TARGETS

layout(location=0) out vec4 diffuse_buffer;
layout(location=1) out vec4 specular_buffer;
layout(location=2) out vec4 normal_mr_buffer;

#else

layout(location=0) out vec4 frag_color;

#endif


// GGX Specular
// Source: http://www.filmicworlds.com/images/ggx-opt/optimized-ggx.hlsl
float G1V(float dotNV, float k)
{
    return 1.0 / (dotNV * (1.0 - k) + k);
}

float specularGGX(vec3 N, vec3 V, vec3 L, float roughness, float F0)
{
    float alpha = roughness * roughness;

    vec3 H = normalize(V + L);

    float dotNL = max(dot(N,L), 0.0 );
    float dotNV = max(dot(N,V), 0.0 );
    float dotNH = max(dot(N,H), 0.0 );
    float dotLH = max(dot(L,H), 0.0 );

    // D
    float alphaSqr = alpha * alpha;
    float pi = M_PI;
    float denom = dotNH * dotNH * (alphaSqr - 1.0) + 1.0;
    float D = alphaSqr / (pi * denom * denom);

    // F
    float dotLH5 = pow(1.0 - dotLH, 5.0);
    float F = F0 + (1.0 - F0) * (dotLH5);

    // V
    float k = alpha / 2.0f;
    float vis = G1V(dotNL, k) * G1V(dotNV, k);

    return dotNL * D * F * vis;
}

void light_compute(vec3 normal, vec3 light_vec,vec3 eye_vec,vec3 diffuse_color, vec3 specular_color, float roughness, float attenuation, inout vec3 diffuse, inout vec3 specular) {

	diffuse += max(0.0,dot(normal,light_vec)) * diffuse_color * attenuation;
	//specular += specular_ggx( roughness, max(0.0,dot(normal,eye_vec)) ) * specular_color * attenuation;
	float s = roughness > 0.0 ? specularGGX(normal,eye_vec,light_vec,roughness,1.0) : 0.0;
	specular += s * specular_color * attenuation;
}


float sample_shadow(highp sampler2DShadow shadow, vec2 shadow_pixel_size, vec2 pos, float depth, vec4 clamp_rect) {

	return textureProj(shadow,vec4(pos,depth,1.0));
}

#ifdef RENDER_SHADOW_DUAL_PARABOLOID

in highp float dp_clip;

#endif

void main() {

#ifdef RENDER_SHADOW_DUAL_PARABOLOID

	if (dp_clip>0.0)
		discard;
#endif

	//lay out everything, whathever is unused is optimized away anyway
	highp vec3 vertex = vertex_interp;
	vec3 albedo = vec3(0.8,0.8,0.8);
	vec3 specular = vec3(0.2,0.2,0.2);
	float roughness = 1.0;
	float alpha = 1.0;

#ifdef METERIAL_DOUBLESIDED
	float side=float(gl_FrontFacing)*2.0-1.0;
#else
	float side=1.0;
#endif


#if defined(ENABLE_TANGENT_INTERP)
	vec3 binormal = normalize(binormal_interp)*side;
	vec3 tangent = normalize(tangent_interp)*side;
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

{


FRAGMENT_SHADER_CODE

}

#if defined(ENABLE_NORMALMAP)

	normal = normalize( mix(normal_interp,tangent_interp * normalmap.x + binormal_interp * normalmap.y + normal_interp * normalmap.z,normaldepth) ) * side;

#endif

#if defined(ENABLE_DISCARD)
	if (discard_) {
	//easy to eliminate dead code
		discard;
	}
#endif

#ifdef ENABLE_CLIP_ALPHA
	if (diffuse.a<0.99) {
		//used for doublepass and shadowmapping
		discard;
	}
#endif

/////////////////////// LIGHTING //////////////////////////////

	vec3 specular_light = vec3(0.0,0.0,0.0);
	vec3 ambient_light;
	vec3 diffuse_light = vec3(0.0,0.0,0.0);

	vec3 eye_vec = -normalize( vertex_interp );

#ifdef USE_RADIANCE_CUBEMAP

	if (no_ambient_light) {
		ambient_light=vec3(0.0,0.0,0.0);
	} else {
		{

			float ndotv = clamp(dot(normal,eye_vec),0.0,1.0);
			vec2 brdf = texture(brdf_texture, vec2(roughness, ndotv)).xy;

			float lod = roughness * 5.0;
			vec3 r = reflect(-eye_vec,normal); //2.0 * ndotv * normal - view; // reflect(v, n);
			r=normalize((radiance_inverse_xform * vec4(r,0.0)).xyz);
			vec3 radiance = textureLod(radiance_cube, r, lod).xyz * ( brdf.x + brdf.y);

			specular_light=mix(albedo,radiance,specular);

		}

		{

			vec3 ambient_dir=normalize((radiance_inverse_xform * vec4(normal,0.0)).xyz);
			vec3 env_ambient=textureLod(radiance_cube, ambient_dir, 5.0).xyz;

			ambient_light=mix(ambient_light_color.rgb,env_ambient,radiance_ambient_contribution);
		}
	}

#else

	if (no_ambient_light){
		ambient_light=vec3(0.0,0.0,0.0);
	} else {
		ambient_light=ambient_light_color.rgb;
	}
#endif


#ifdef USE_FORWARD_DIRECTIONAL

	float light_attenuation=1.0;

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
			ambient_light=vec3(1.0,0.4,0.4);


#if defined(LIGHT_USE_PSSM_BLEND)

			splane=(shadow_matrix2 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(0.0,light_pssm_split_inv.x,w_inv);
#endif

		} else {

			highp vec4 splane=(shadow_matrix2 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;
			ambient_light=vec3(0.4,1.0,0.4);

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
			ambient_light=vec3(0.4,0.4,1.0);

#if defined(LIGHT_USE_PSSM_BLEND)
			splane=(shadow_matrix4 * vec4(vertex,1.0));
			pssm_coord2=splane.xyz/splane.w;
			pssm_blend=smoothstep(light_pssm_split_inv.y,light_pssm_split_inv.z,w_inv);
#endif

		} else {
			highp vec4 splane=(shadow_matrix4 * vec4(vertex,1.0));
			pssm_coord=splane.xyz/splane.w;
			diffuse_light*=vec3(1.0,0.4,1.0);

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
	light_attenuation=sample_shadow(directional_shadow,directional_shadow_pixel_size,pssm_coord.xy,pssm_coord.z,light_clamp);


#if defined(LIGHT_USE_PSSM_BLEND)
	if (use_blend) {
		float light_attenuation2=sample_shadow(directional_shadow,directional_shadow_pixel_size,pssm_coord2.xy,pssm_coord2.z,light_clamp);
		light_attenuation=mix(light_attenuation,light_attenuation2,pssm_blend);
	}
#endif

	}

#endif //LIGHT_DIRECTIONAL_SHADOW

	light_compute(normal,-light_direction_attenuation.xyz,eye_vec,albedo,specular,roughness,light_attenuation,diffuse_light,specular_light);


#endif //USE_FORWARD_DIRECTIONAL


#ifdef USE_FORWARD_OMNI

	vec3 light_rel_vec = light_pos_inv_radius.xyz-vertex;
	float normalized_distance = length( light_rel_vec )*light_pos_inv_radius.w;
	float light_attenuation = pow( max(1.0 - normalized_distance, 0.0), light_direction_attenuation.w );

	if (light_params.w>0.5) {
		//there is a shadowmap

		highp vec3 splane=(shadow_matrix1 * vec4(vertex,1.0)).xyz;
		float shadow_len=length(splane);
		splane=normalize(splane);
		vec4 clamp_rect=light_clamp;

		if (splane.z>=0.0) {

			splane.z+=1.0;

			clamp_rect.y+=clamp_rect.w;

		} else {

			splane.z=1.0 - splane.z;

			//if (clamp_rect.z<clamp_rect.w) {
			//	clamp_rect.x+=clamp_rect.z;
			//} else {
			//	clamp_rect.y+=clamp_rect.w;
			//}

		}

		splane.xy/=splane.z;
		splane.xy=splane.xy * 0.5 + 0.5;
		splane.z = shadow_len * light_pos_inv_radius.w;

		splane.xy = clamp_rect.xy+splane.xy*clamp_rect.zw;

		light_attenuation*=sample_shadow(shadow_atlas,shadow_atlas_pixel_size,splane.xy,splane.z,clamp_rect);
	}

	light_compute(normal,normalize(light_rel_vec),eye_vec,albedo,specular,roughness,light_attenuation,diffuse_light,specular_light);


#endif //USE_FORWARD_OMNI

#ifdef USE_FORWARD_SPOT

	vec3 light_rel_vec = light_pos_inv_radius.xyz-vertex;
	float normalized_distance = length( light_rel_vec )*light_pos_inv_radius.w;
	float light_attenuation = pow( max(1.0 - normalized_distance, 0.0), light_direction_attenuation.w );
	vec3 spot_dir = light_direction_attenuation.xyz;
	float spot_cutoff=light_params.y;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir),spot_cutoff);
	float rim = (1.0 - scos) / (1.0 - spot_cutoff);
	light_attenuation *= 1.0 - pow( rim, light_params.x);

	if (light_params.w>0.5) {
		//there is a shadowmap

		highp vec4 splane=(shadow_matrix1 * vec4(vertex,1.0));
		splane.xyz/=splane.w;
	//	splane.xy=splane.xy*0.5+0.5;

		//splane.xy=light_clamp.xy+splane.xy*light_clamp.zw;
		light_attenuation*=sample_shadow(shadow_atlas,shadow_atlas_pixel_size,splane.xy,splane.z,light_clamp);

	}

	light_compute(normal,normalize(light_rel_vec),eye_vec,albedo,specular,roughness,light_attenuation,diffuse_light,specular_light);

#endif //USE_FORWARD_SPOT



#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader
{

LIGHT_SHADER_CODE

}
#endif

#ifdef RENDER_SHADOW
//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else


#ifdef USE_MULTIPLE_RENDER_TARGETS

	//approximate ambient scale for SSAO, since we will lack full ambient
	float max_ambient=max(ambient_light.r,max(ambient_light.g,ambient_light.b));
	float max_diffuse=max(diffuse_light.r,max(diffuse_light.g,diffuse_light.b));
	float total_ambient = max_ambient+max_diffuse;
	float ambient_scale = (total_ambient>0.0) ? max_ambient/total_ambient : 0.0;

	diffuse_buffer=vec4(diffuse_light+ambient_light,ambient_scale);
	specular_buffer=vec4(specular_light,0.0);
	normal_mr_buffer=vec4(normal.x,normal.y,max(specular.r,max(specular.g,specular.b)),roughness);

#else


#ifdef SHADELESS
	frag_color=vec4(albedo,alpha);
#else
	frag_color=vec4(ambient_light+diffuse_light+specular_light,alpha);
#endif //SHADELESS

#endif //USE_MULTIPLE_RENDER_TARGETS

#endif //RENDER_SHADOW


}


