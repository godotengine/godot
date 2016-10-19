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

	highp vec4 ambient_light;
};

uniform highp mat4 world_transform;

/* Varyings */

out vec3 vertex_interp;
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

#if defined(ENABLE_VAR1_INTERP)
out vec4 var1_interp;
#endif

#if defined(ENABLE_VAR2_INTERP)
out vec4 var2_interp;
#endif

#if defined(ENABLE_TANGENT_INTERP)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif


#if !defined(USE_DEPTH_SHADOWS) && defined(USE_SHADOW_PASS)

varying vec4 position_interp;

#endif

#ifdef USE_SHADOW_PASS

uniform highp float shadow_z_offset;
uniform highp float shadow_z_slope_scale;

#endif


VERTEX_SHADER_GLOBALS


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:1

MATERIAL_UNIFORMS

};

#endif


void main() {

	highp vec4 vertex_in = vertex_attrib; // vec4(vertex_attrib.xyz * data_attrib.x,1.0);
	highp mat4 modelview = camera_inverse_matrix * world_transform;
	vec3 normal_in = normal_attrib;
	normal_in*=normal_mult;
#if defined(ENABLE_TANGENT_INTERP)
	vec3 tangent_in = tangent_attrib.xyz;
	tangent_in*=normal_mult;
	float binormalf = tangent_attrib.a;
#endif

#ifdef USE_SKELETON

	{
		//skeleton transform
		highp mat4 m=mat4(texture2D(skeleton_matrices,vec2((bone_indices.x*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.x*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.x*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.x;
		m+=mat4(texture2D(skeleton_matrices,vec2((bone_indices.y*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.y*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.y*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.y;
		m+=mat4(texture2D(skeleton_matrices,vec2((bone_indices.z*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.z*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.z*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.z;
		m+=mat4(texture2D(skeleton_matrices,vec2((bone_indices.w*3.0+0.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.w*3.0+1.0)*skeltex_pixel_size,0.0)),texture2D(skeleton_matrices,vec2((bone_indices.w*3.0+2.0)*skeltex_pixel_size,0.0)),vec4(0.0,0.0,0.0,1.0))*bone_weights.w;

		vertex_in = vertex_in * m;
		normal_in = (vec4(normal_in,0.0) * m).xyz;
#if defined(ENABLE_TANGENT_INTERP)
		tangent_in = (vec4(tangent_in,0.0) * m).xyz;
#endif
	}

#endif

	vertex_interp = (modelview * vertex_in).xyz;
	normal_interp = normalize((modelview * vec4(normal_in,0.0)).xyz);

#if defined(ENABLE_TANGENT_INTERP)
	tangent_interp=normalize((modelview * vec4(tangent_in,0.0)).xyz);
	binormal_interp = normalize( cross(normal_interp,tangent_interp) * binormalf );
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


VERTEX_SHADER_CODE


#ifdef USE_SHADOW_PASS

	float z_ofs = shadow_z_offset;
	z_ofs += (1.0-abs(normal_interp.z))*shadow_z_slope_scale;
	vertex_interp.z-=z_ofs;
#endif


#ifdef USE_FOG

	fog_interp.a = pow( clamp( (length(vertex_interp)-fog_params.x)/(fog_params.y-fog_params.x), 0.0, 1.0 ), fog_params.z );
	fog_interp.rgb = mix( fog_color_begin, fog_color_end, fog_interp.a );
#endif

#ifndef VERTEX_SHADER_WRITE_POSITION
//vertex shader might write a position
	gl_Position = projection_matrix * vec4(vertex_interp,1.0);
#endif




}


[fragment]


//hack to use uv if no uv present so it works with lightmap


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

#if defined(ENABLE_VAR1_INTERP)
in vec4 var1_interp;
#endif

#if defined(ENABLE_VAR2_INTERP)
in vec4 var2_interp;
#endif

in vec3 vertex_interp;
in vec3 normal_interp;


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

	highp vec4 ambient_light;
};

layout(location=0) out vec4 frag_color;

void main() {

	//lay out everything, whathever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
	vec3 albedo = vec3(0.9,0.9,0.9);
	vec3 metal = vec3(0.0,0.0,0.0);
	float rough = 0.0;
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



#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader
{

LIGHT_SHADER_CODE

}
#endif

	frag_color=vec4(albedo,alpha);
}


