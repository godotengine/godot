[vertex]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

#include "stdlib.glsl"



//
// attributes
//

attribute highp vec4 vertex_attrib; // attrib:0
attribute vec3 normal_attrib; // attrib:1

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
attribute vec4 tangent_attrib; // attrib:2
#endif

#ifdef ENABLE_COLOR_INTERP
attribute vec4 color_attrib; // attrib:3
#endif

#ifdef ENABLE_UV_INTERP
attribute vec2 uv_attrib; // attrib:4
#endif

#ifdef ENABLE_UV2_INTERP
attribute vec2 uv2_attrib; // attrib:5
#endif

#ifdef USE_SKELETON

#ifdef USE_SKELETON_SOFTWARE

attribute highp vec4 bone_transform_row_0; // attrib:9
attribute highp vec4 bone_transform_row_1; // attrib:10
attribute highp vec4 bone_transform_row_2; // attrib:11

#else

attribute vec4 bone_ids; // attrib:6
attribute highp vec4 bone_weights; // attrib:7

uniform highp sampler2D bone_transforms; // texunit:4
uniform ivec2 skeleton_texture_size;

#endif

#endif

#ifdef USE_INSTANCING

attribute highp vec4 instance_xform_row_0; // attrib:12
attribute highp vec4 instance_xform_row_1; // attrib:13
attribute highp vec4 instance_xform_row_2; // attrib:14

attribute highp vec4 instance_color; // attrib:15
attribute highp vec4 instance_custom_data; // attrib:8

#endif



//
// uniforms
//

uniform mat4 camera_matrix;
uniform mat4 camera_inverse_matrix;
uniform mat4 projection_matrix;
uniform mat4 projection_inverse_matrix;

uniform mat4 world_transform;

uniform highp float time;

uniform float normal_mult;

#ifdef RENDER_DEPTH
uniform float light_bias;
uniform float light_normal_bias;
#endif


//
// varyings
//

varying highp vec3 vertex_interp;
varying vec3 normal_interp;

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
varying vec3 tangent_interp;
varying vec3 binormal_interp;
#endif

#ifdef ENABLE_COLOR_INTERP
varying vec4 color_interp;
#endif

#ifdef ENABLE_UV_INTERP
varying vec2 uv_interp;
#endif

#ifdef ENABLE_UV2_INTERP
varying vec2 uv2_interp;
#endif


VERTEX_SHADER_GLOBALS

void main() {

	highp vec4 vertex = vertex_attrib;

	mat4 world_matrix = world_transform;

#ifdef USE_INSTANCING
	{
		highp mat4 m = mat4(instance_xform_row_0,
		                    instance_xform_row_1,
		                    instance_xform_row_2,
		                    vec4(0.0, 0.0, 0.0, 1.0));
		world_matrix = world_matrix * transpose(m);
	}
#endif

	vec3 normal = normal_attrib * normal_mult;

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
	vec3 tangent = tangent_attrib.xyz;
	tangent *= normal_mult;
	float binormalf = tangent_attrib.a;
	vec3 binormal = normalize(cross(normal, tangent) * binormalf);
#endif

#ifdef ENABLE_COLOR_INTERP
	color_interp = color_attrib;
#ifdef USE_INSTANCING
	color_interp *= instance_color;
#endif
#endif

#ifdef ENABLE_UV_INTERP
	uv_interp = uv_attrib;
#endif

#ifdef ENABLE_UV2_INTERP
	uv2_interp = uv2_attrib;
#endif

#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)
	vertex = world_matrix * vertex;
	normal = normalize((world_matrix * vec4(normal, 0.0)).xyz);
#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)

	tangent = normalize((world_matrix * vec4(tangent, 0.0)),xyz);
	binormal = normalize((world_matrix * vec4(binormal, 0.0)).xyz);
#endif
#endif

#ifdef USE_SKELETON

	highp mat4 bone_transform = mat4(1.0);

#ifdef USE_SKELETON_SOFTWARE
	// passing the transform as attributes

	bone_transform[0] = vec4(bone_transform_row_0.x, bone_transform_row_1.x, bone_transform_row_2.x, 0.0);
	bone_transform[1] = vec4(bone_transform_row_0.y, bone_transform_row_1.y, bone_transform_row_2.y, 0.0);
	bone_transform[2] = vec4(bone_transform_row_0.z, bone_transform_row_1.z, bone_transform_row_2.z, 0.0);
	bone_transform[3] = vec4(bone_transform_row_0.w, bone_transform_row_1.w, bone_transform_row_2.w, 1.0);

#else
	// look up transform from the "pose texture"
	{
		
		for (int i = 0; i < 4; i++) {
			ivec2 tex_ofs = ivec2(int(bone_ids[i]) * 3, 0);

			highp mat4 b = mat4(texel2DFetch(bone_transforms, skeleton_texture_size, tex_ofs + ivec2(0, 0)),
			              texel2DFetch(bone_transforms, skeleton_texture_size, tex_ofs + ivec2(1, 0)),
			              texel2DFetch(bone_transforms, skeleton_texture_size, tex_ofs + ivec2(2, 0)),
			              vec4(0.0, 0.0, 0.0, 1.0));

			bone_transform += transpose(b) * bone_weights[i];
		}
	}

#endif

	world_matrix = bone_transform * world_matrix;
#endif


#ifdef USE_INSTANCING
	vec4 instance_custom = instance_custom_data;
#else
	vec4 instance_custom = vec4(0.0);

#endif


	mat4 modelview = camera_matrix * world_matrix;

#define world_transform world_matrix

{

VERTEX_SHADER_CODE

}

	vec4 outvec = vertex;

	// use local coordinates
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)
	vertex = modelview * vertex;
	normal = normalize((modelview * vec4(normal, 0.0)).xyz);

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
	tangent = normalize((modelview * vec4(tangent, 0.0)).xyz);
	binormal = normalize((modelview * vec4(binormal, 0.0)).xyz);
#endif
#endif

#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)
	vertex = camera_matrix * vertex;
	normal = normalize((camera_matrix * vec4(normal, 0.0)).xyz);
#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
	tangent = normalize((camera_matrix * vec4(tangent, 0.0)).xyz);
	binormal = normalize((camera_matrix * vec4(binormal, 0.0)).xyz);
#endif
#endif

	vertex_interp = vertex.xyz;
	normal_interp = normal;

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

#ifdef RENDER_DEPTH

	float z_ofs = light_bias;
	z_ofs += (1.0 - abs(normal_interp.z)) * light_normal_bias;
	
	vertex_interp.z -= z_ofs;

#endif

	gl_Position = projection_matrix * vec4(vertex_interp, 1.0);

}

[fragment]
#extension GL_ARB_shader_texture_lod : require

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

#include "stdlib.glsl"

#define M_PI 3.14159265359

//
// uniforms
//

uniform mat4 camera_matrix;
uniform mat4 camera_inverse_matrix;
uniform mat4 projection_matrix;
uniform mat4 projection_inverse_matrix;

uniform mat4 world_transform;

uniform highp float time;


#ifdef SCREEN_UV_USED
uniform vec2 screen_pixel_size;
#endif

uniform highp sampler2D depth_buffer; //texunit:1

#if defined(SCREEN_TEXTURE_USED)
uniform highp sampler2D screen_texture; //texunit:2
#endif

#ifdef USE_RADIANCE_MAP

#define RADIANCE_MAX_LOD 6.0

uniform samplerCube radiance_map; // texunit:0

uniform mat4 radiance_inverse_xform;

#endif

uniform float bg_energy;

uniform float ambient_sky_contribution;
uniform vec4 ambient_color;
uniform float ambient_energy;

#ifdef LIGHT_PASS

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2

// general for all lights
uniform int light_type;

uniform float light_energy;
uniform vec4 light_color;
uniform float light_specular;

// directional
uniform vec3 light_direction;

// omni
uniform vec3 light_position;

uniform float light_range;
uniform vec4 light_attenuation;

// spot
uniform float light_spot_attenuation;
uniform float light_spot_range;
uniform float light_spot_angle;


// shadows
uniform highp sampler2D light_shadow_atlas; //texunit:3
uniform float light_has_shadow;

uniform mat4 light_shadow_matrix;
uniform vec4 light_clamp;

// directional shadow

uniform highp sampler2D light_directional_shadow; // texunit:3
uniform vec4 light_split_offsets;

uniform mat4 light_shadow_matrix1;
uniform mat4 light_shadow_matrix2;
uniform mat4 light_shadow_matrix3;
uniform mat4 light_shadow_matrix4;
#endif


//
// varyings
//

varying highp vec3 vertex_interp;
varying vec3 normal_interp;

#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
varying vec3 tangent_interp;
varying vec3 binormal_interp;
#endif

#ifdef ENABLE_COLOR_INTERP
varying vec4 color_interp;
#endif

#ifdef ENABLE_UV_INTERP
varying vec2 uv_interp;
#endif

#ifdef ENABLE_UV2_INTERP
varying vec2 uv2_interp;
#endif

varying vec3 view_interp;

vec3 metallic_to_specular_color(float metallic, float specular, vec3 albedo) {
	float dielectric = (0.034 * 2.0) * specular;
	// energy conservation
	return mix(vec3(dielectric), albedo, metallic); // TODO: reference?
}

FRAGMENT_SHADER_GLOBALS


#ifdef LIGHT_PASS
void light_compute(vec3 N,
                   vec3 L,
                   vec3 V,
                   vec3 B,
                   vec3 T,
                   vec3 light_color,
                   vec3 attenuation,
                   vec3 diffuse_color,
                   vec3 transmission,
                   float specular_blob_intensity,
                   float roughness,
                   float metallic,
                   float rim,
                   float rim_tint,
                   float clearcoat,
                   float clearcoat_gloss,
                   float anisotropy,
                   inout vec3 diffuse_light,
                   inout vec3 specular_light) {

	float NdotL = dot(N, L);
	float cNdotL = max(NdotL, 0.0);
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

	{
		// calculate diffuse reflection

		// TODO hardcode Oren Nayar for now
		float diffuse_brdf_NL;

		diffuse_brdf_NL = max(0.0,(NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness)));
		// diffuse_brdf_NL = cNdotL * (1.0 / M_PI);

		diffuse_light += light_color * diffuse_color * diffuse_brdf_NL * attenuation;
	}

	{
		// calculate specular reflection

		 vec3 R = normalize(-reflect(L,N));
		 float cRdotV = max(dot(R, V), 0.0);
		 float blob_intensity = pow(cRdotV, (1.0 - roughness) * 256.0);
		 specular_light += light_color * attenuation * blob_intensity * specular_blob_intensity;


	}
}




// shadows

float sample_shadow(highp sampler2D shadow,
                    vec2 shadow_pixel_size,
                    vec2 pos,
                    float depth,
                    vec4 clamp_rect)
{
	// vec4 depth_value = texture2D(shadow, pos);
	
	// return depth_value.z;
	return texture2DProj(shadow, vec4(pos, depth, 1.0)).r;
	// return (depth_value.x + depth_value.y + depth_value.z + depth_value.w) / 4.0;
}


#endif

void main() 
{

	highp vec3 vertex = vertex_interp;
	vec3 albedo = vec3(0.8, 0.8, 0.8);
	vec3 transmission = vec3(0.0);
	float metallic = 0.0;
	float specular = 0.5;
	vec3 emission = vec3(0.0, 0.0, 0.0);
	float roughness = 1.0;
	float rim = 0.0;
	float rim_tint = 0.0;
	float clearcoat = 0.0;
	float clearcoat_gloss = 0.0;
	float anisotropy = 1.0;
	vec2 anisotropy_flow = vec2(1.0,0.0);

	float alpha = 1.0;
	float side = 1.0;

#if defined(ENABLE_AO)
	float ao = 1.0;
	float ao_light_affect = 0.0;
#endif


#if defined(ENABLE_TANGENT_INTERP) || defined(ENABLE_NORMALMAP)
	vec3 binormal = normalize(binormal_interp) * side;
	vec3 tangent = normalize(tangent_interp) * side;
#else
	vec3 binormal = vec3(0.0);
	vec3 tangent = vec3(0.0);
#endif
	vec3 normal = normalize(normal_interp) * side;

#if defined(ENABLE_NORMALMAP)
	vec3 normalmap = vec3(0.5);
#endif
	float normaldepth = 1.0;


#ifdef ALPHA_SCISSOR_USED
	float alpha_scissor = 0.5;
#endif

#ifdef SCREEN_UV_USED
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif

{

FRAGMENT_SHADER_CODE


}

#if defined(ENABLE_NORMALMAP)
	normalmap.xy = normalmap.xy * 2.0 - 1.0;
	normalmap.z = sqrt(1.0 - dot(normalmap.xy, normalmap.xy));

	// normal = normalize(mix(normal_interp, tangent * normalmap.x + binormal * normalmap.y + normal * normalmap.z, normaldepth)) * side;
	normal = normalmap;
#endif

	normal = normalize(normal);

	vec3 N = normal;
	
	vec3 specular_light = vec3(0.0, 0.0, 0.0);
	vec3 diffuse_light = vec3(0.0, 0.0, 0.0);

	vec3 ambient_light = vec3(0.0, 0.0, 0.0);

	vec3 env_reflection_light = vec3(0.0, 0.0, 0.0);

	vec3 eye_position = -normalize(vertex_interp);

#ifdef ALPHA_SCISSOR_USED
	if (alpha < alpha_scissor) {
		discard;
	}
#endif
	
//
// Lighting
//
#ifdef LIGHT_PASS

	if (light_type == LIGHT_TYPE_OMNI) {
		vec3 light_vec = light_position - vertex;
		float light_length = length(light_vec);

		float normalized_distance = light_length / light_range;

		float omni_attenuation = pow(1.0 - normalized_distance, light_attenuation.w);

		vec3 attenuation = vec3(omni_attenuation);

		if (light_has_shadow > 0.5) {
			highp vec3 splane =  (light_shadow_matrix * vec4(vertex, 1.0)).xyz;
			float shadow_len = length(splane);

			splane = normalize(splane);

			vec4 clamp_rect = light_clamp;

			if (splane.z >= 0.0) {
				splane.z += 1.0;

				clamp_rect.y += clamp_rect.w;
			} else {
				splane.z = 1.0 - splane.z;
			}

			splane.xy /= splane.z;
			splane.xy = splane.xy * 0.5 + 0.5;
			splane.z = shadow_len / light_range;

			splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;

			float shadow = sample_shadow(light_shadow_atlas, vec2(0.0), splane.xy, splane.z, clamp_rect);

			if (shadow > splane.z) {
			} else {
				attenuation = vec3(0.0);
			}
		}

		light_compute(normal,
		              normalize(light_vec),
		              eye_position,
		              binormal,
		              tangent,
		              light_color.xyz * light_energy,
		              attenuation,
		              albedo,
		              transmission,
		              specular * light_specular,
		              roughness,
		              metallic,
		              rim,
		              rim_tint,
		              clearcoat,
		              clearcoat_gloss,
		              anisotropy,
		              diffuse_light,
		              specular_light);

	} else if (light_type == LIGHT_TYPE_DIRECTIONAL) {

		vec3 light_vec = -light_direction;
		vec3 attenuation = vec3(1.0, 1.0, 1.0);
		
		float depth_z = -vertex.z;
		
		if (light_has_shadow > 0.5) {
		
#ifdef LIGHT_USE_PSSM4
			if (depth_z < light_split_offsets.w) {
#elif defined(LIGHT_USE_PSSM2)
			if (depth_z < light_split_offsets.y) {
#else
			if (depth_z < light_split_offsets.x) {
#endif
		
			vec3 pssm_coord;
			float pssm_fade = 0.0;
			
#ifdef LIGHT_USE_PSSM_BLEND
			float pssm_blend;
			vec3 pssm_coord2;
			bool use_blend = true;
#endif
			
#ifdef LIGHT_USE_PSSM4
			if (depth_z < light_split_offsets.y) {
				if (depth_z < light_split_offsets.x) {
					highp vec4 splane = (light_shadow_matrix1 * vec4(vertex, 1.0));
					pssm_coord = splane.xyz / splane.w;
					
#ifdef LIGHT_USE_PSSM_BLEND
					splane = (light_shadow_matrix2 * vec4(vertex, 1.0));
					pssm_coord2 = splane.xyz / splane.w;
					
					pssm_blend = smoothstep(0.0, light_split_offsets.x, depth_z);
#endif
				} else {
					highp vec4 splane = (light_shadow_matrix2 * vec4(vertex, 1.0));
					pssm_coord = splane.xyz / splane.w;
					
#ifdef LIGHT_USE_PSSM_BLEND
					splane = (light_shadow_matrix3 * vec4(vertex, 1.0));
					pssm_coord2 = splane.xyz / splane.w;
					
					pssm_blend = smoothstep(light_split_offsets.x, light_split_offsets.y, depth_z);
#endif
				}
			} else {
				if (depth_z < light_split_offsets.z) {

					highp vec4 splane = (light_shadow_matrix3 * vec4(vertex, 1.0));
					pssm_coord = splane.xyz / splane.w;

#if defined(LIGHT_USE_PSSM_BLEND)
					splane = (light_shadow_matrix4 * vec4(vertex, 1.0));
					pssm_coord2 = splane.xyz / splane.w;
					pssm_blend = smoothstep(light_split_offsets.y, light_split_offsets.z, depth_z);
#endif

				} else {

					highp vec4 splane = (light_shadow_matrix4 * vec4(vertex, 1.0));
					pssm_coord = splane.xyz / splane.w;
					pssm_fade = smoothstep(light_split_offsets.z, light_split_offsets.w, depth_z);

#if defined(LIGHT_USE_PSSM_BLEND)
					use_blend = false;
#endif
				}
			}
			
#endif // LIGHT_USE_PSSM4
			
#ifdef LIGHT_USE_PSSM2
			if (depth_z < light_split_offsets.x) {
				
				highp vec4 splane = (light_shadow_matrix1 * vec4(vertex, 1.0));
				pssm_coord = splane.xyz / splane.w;
				
#ifdef LIGHT_USE_PSSM_BLEND
				splane = (light_shadow_matrix2 * vec4(vertex, 1.0));
				pssm_coord2 = splane.xyz / splane.w;
				pssm_blend = smoothstep(0.0, light_split_offsets.x, depth_z);
#endif
			} else {
				highp vec4 splane = (light_shadow_matrix2 * vec4(vertex, 1.0));
				pssm_coord = splane.xyz / splane.w;
				pssm_fade = smoothstep(light_split_offsets.x, light_split_offsets.y, depth_z);
#ifdef LIGHT_USE_PSSM_BLEND
				use_blend = false;
#endif
			}
			
#endif // LIGHT_USE_PSSM2
			
#if !defined(LIGHT_USE_PSSM4) && !defined(LIGHT_USE_PSSM2)
			{
				highp vec4 splane = (light_shadow_matrix1 * vec4(vertex, 1.0));
				pssm_coord = splane.xyz / splane.w;
			}
#endif
			
			float shadow = sample_shadow(light_shadow_atlas, vec2(0.0), pssm_coord.xy, pssm_coord.z, light_clamp);
			
#ifdef LIGHT_USE_PSSM_BLEND
			if (use_blend) {
				shadow = mix(shadow, sample_shadow(light_shadow_atlas, vec2(0.0), pssm_coord2.xy, pssm_coord2.z, light_clamp), pssm_blend);
			}
#endif
			
			attenuation *= shadow;
			
			
		}
			
		}

		light_compute(normal,
		              normalize(light_vec),
		              eye_position,
		              binormal,
		              tangent,
		              light_color.xyz * light_energy,
		              attenuation,
		              albedo,
		              transmission,
		              specular * light_specular,
		              roughness,
		              metallic,
		              rim,
		              rim_tint,
		              clearcoat,
		              clearcoat_gloss,
		              anisotropy,
		              diffuse_light,
		              specular_light);
	} else if (light_type == LIGHT_TYPE_SPOT) {

		vec3 light_att = vec3(1.0);
		
		if (light_has_shadow > 0.5) {
			highp vec4 splane =  (light_shadow_matrix * vec4(vertex, 1.0));
			splane.xyz /= splane.w;
			
			float shadow = sample_shadow(light_shadow_atlas, vec2(0.0), splane.xy, splane.z, light_clamp);
			
			if (shadow > splane.z) {
			} else {
				light_att = vec3(0.0);
			}
			
			
		}

		vec3 light_rel_vec = light_position - vertex;
		float light_length = length(light_rel_vec);
		float normalized_distance = light_length / light_range;

		float spot_attenuation = pow(1.0 - normalized_distance, light_attenuation.w);
		vec3 spot_dir = light_direction;

		float spot_cutoff = light_spot_angle;

		float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_cutoff);
		float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_cutoff));

		spot_attenuation *= 1.0 - pow(spot_rim, light_spot_attenuation);

		light_att *= vec3(spot_attenuation);
		
		light_compute(normal,
		              normalize(light_rel_vec),
		              eye_position,
		              binormal,
		              tangent,
		              light_color.xyz * light_energy,
		              light_att,
		              albedo,
		              transmission,
		              specular * light_specular,
		              roughness,
		              metallic,
		              rim,
		              rim_tint,
		              clearcoat,
		              clearcoat_gloss,
		              anisotropy,
		              diffuse_light,
		              specular_light);

	}

	gl_FragColor = vec4(ambient_light + diffuse_light + specular_light, alpha);
#else

#ifdef RENDER_DEPTH

#else

#ifdef USE_RADIANCE_MAP


	vec3 ref_vec = reflect(-eye_position, N);
	ref_vec = normalize((radiance_inverse_xform * vec4(ref_vec, 0.0)).xyz);

	ref_vec.z *= -1.0;

	env_reflection_light = textureCubeLod(radiance_map, ref_vec, roughness * RADIANCE_MAX_LOD).xyz * bg_energy;

	{
		vec3 ambient_dir = normalize((radiance_inverse_xform * vec4(normal, 0.0)).xyz);
		vec3 env_ambient = textureCubeLod(radiance_map, ambient_dir, RADIANCE_MAX_LOD).xyz * bg_energy;

		ambient_light = mix(ambient_color.rgb, env_ambient, ambient_sky_contribution);

	}

	ambient_light *= ambient_energy;
	
	specular_light += env_reflection_light;
	
	ambient_light *= albedo;

#if defined(ENABLE_AO)
	ambient_light *= ao;
	ao_light_affect = mix(1.0, ao, ao_light_affect);
	specular_light *= ao_light_affect;
	diffuse_light *= ao_light_affect;
#endif
	
	diffuse_light *= 1.0 - metallic;
	ambient_light *= 1.0 - metallic;
	
	// environment BRDF approximation
	
	// TODO shadeless
	{
		const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
		const vec4 c1 = vec4( 1.0, 0.0425, 1.04, -0.04);
		vec4 r = roughness * c0 + c1;
		float ndotv = clamp(dot(normal,eye_position),0.0,1.0);
		float a004 = min( r.x * r.x, exp2( -9.28 * ndotv ) ) * r.x + r.y;
		vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;

		vec3 specular_color = metallic_to_specular_color(metallic, specular, albedo);
		specular_light *= AB.x * specular_color + AB.y;
	}


	gl_FragColor = vec4(ambient_light + diffuse_light + specular_light, alpha);
	// gl_FragColor = vec4(normal, 1.0);


#else
	gl_FragColor = vec4(albedo, alpha);
#endif
#endif // RENDER_DEPTH


#endif // lighting


}
