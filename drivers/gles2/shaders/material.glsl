[vertex]


#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#define roundfix( m_val ) floor( (m_val) + 0.5 )
#else
precision mediump float;
precision mediump int;
#endif



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
#ifdef ENABLE_AMBIENT_LIGHTMAP

#ifdef USE_LIGHTMAP_ON_UV2

#ifndef ENABLE_UV2_INTERP
#define ENABLE_UV2_INTERP
#endif

#else

#ifndef ENABLE_UV_INTERP
#define ENABLE_UV_INTERP
#endif

#endif

#endif


/* INPUT ATTRIBS */

attribute highp vec4 vertex_attrib; // attrib:0
attribute vec3 normal_attrib; // attrib:1
attribute vec4 tangent_attrib; // attrib:2
attribute vec4 color_attrib; // attrib:3
attribute vec2 uv_attrib; // attrib:4
attribute vec2 uv2_attrib; // attrib:5

uniform float normal_mult;

#ifdef USE_SKELETON
attribute vec4 bone_indices; // attrib:6
attribute vec4 bone_weights; // attrib:7
uniform highp sampler2D skeleton_matrices;
uniform highp float skeltex_pixel_size;
#endif

#ifdef USE_ATTRIBUTE_INSTANCING

attribute highp vec4 instance_row0; // attrib:8
attribute highp vec4 instance_row1; // attrib:9
attribute highp vec4 instance_row2; // attrib:10
attribute highp vec4 instance_row3; // attrib:11

#endif

#ifdef USE_TEXTURE_INSTANCING

attribute highp vec3 instance_uv; // attrib:6
uniform highp sampler2D instance_matrices;

#endif

uniform highp mat4 world_transform;
uniform highp mat4 camera_inverse_transform;
uniform highp mat4 projection_transform;

#ifdef USE_UNIFORM_INSTANCING
//shittiest form of instancing (but most compatible)
uniform highp mat4 instance_transform;
#endif

/* Varyings */

varying vec3 vertex_interp;
varying vec3 normal_interp;

#if defined(ENABLE_COLOR_INTERP)
varying vec4 color_interp;
#endif

#if defined(ENABLE_UV_INTERP)
varying vec2 uv_interp;
#endif

#if defined(ENABLE_UV2_INTERP)
varying vec2 uv2_interp;
#endif

#if defined(ENABLE_VAR1_INTERP)
varying vec4 var1_interp;
#endif

#if defined(ENABLE_VAR2_INTERP)
varying vec4 var2_interp;
#endif

#if defined(ENABLE_TANGENT_INTERP)
varying vec3 tangent_interp;
varying vec3 binormal_interp;
#endif

#ifdef ENABLE_AMBIENT_OCTREE

uniform highp mat4 ambient_octree_inverse_transform;
varying highp vec3 ambient_octree_coords;

#endif

#ifdef USE_FOG

varying vec4 fog_interp;
uniform highp vec3 fog_params;
uniform vec3 fog_color_begin;
uniform vec3 fog_color_end;

#endif

#ifdef USE_VERTEX_LIGHTING

uniform vec3 light_pos;
uniform vec3 light_direction;
uniform vec3 light_attenuation;
uniform vec3 light_spot_attenuation;
uniform vec3 light_diffuse;
uniform vec3 light_specular;



#endif

varying vec4 diffuse_interp;
varying vec3 specular_interp;
//intended for static branching
//pretty much all meaningful platforms support
//static branching

uniform float time;
uniform float instance_id;

uniform vec3 ambient_light;

#if !defined(USE_DEPTH_SHADOWS) && defined(USE_SHADOW_PASS)

varying vec4 position_interp;

#endif

#ifdef LIGHT_USE_SHADOW

uniform highp mat4 shadow_matrix;
varying highp vec4 shadow_coord;
#ifdef LIGHT_USE_PSSM
uniform highp mat4 shadow_matrix2;
varying highp vec4 shadow_coord2;
#endif
#ifdef LIGHT_USE_PSSM4
uniform highp mat4 shadow_matrix3;
varying highp vec4 shadow_coord3;
uniform highp mat4 shadow_matrix4;
varying highp vec4 shadow_coord4;
#endif


#endif

#ifdef USE_SHADOW_PASS

uniform highp float shadow_z_offset;
uniform highp float shadow_z_slope_scale;

#endif

#ifdef USE_DUAL_PARABOLOID
uniform highp vec2 dual_paraboloid;
varying float dp_clip;
#endif



VERTEX_SHADER_GLOBALS




void main() {
#ifdef USE_UNIFORM_INSTANCING

	highp mat4 modelview = (camera_inverse_transform * (world_transform * instance_transform));
#ifdef ENABLE_AMBIENT_OCTREE
	highp mat4 ambient_octree_transform = (ambient_octree_inverse_transform * (world_transform * instance_transform));
#endif

#else

#ifdef USE_ATTRIBUTE_INSTANCING

	highp mat4 minst=mat4(instance_row0,instance_row1,instance_row2,instance_row3);
	highp mat4 modelview = (camera_inverse_transform * (world_transform * minst));
#ifdef ENABLE_AMBIENT_OCTREE
	highp mat4 ambient_octree_transform = (ambient_octree_inverse_transform * (world_transform * minst));
#endif

#else

#ifdef USE_TEXTURE_INSTANCING

	highp vec2 ins_ofs=vec2(instance_uv.z,0.0);

	highp mat4 minst=mat4(
		texture2D(instance_matrices,instance_uv.xy),
		texture2D(instance_matrices,instance_uv.xy+ins_ofs),
		texture2D(instance_matrices,instance_uv.xy+ins_ofs*2.0),
		texture2D(instance_matrices,instance_uv.xy+ins_ofs*3.0)
	);

	/*highp mat4 minst=mat4(
		vec4(1.0,0.0,0.0,0.0),
		vec4(0.0,1.0,0.0,0.0),
		vec4(0.0,0.0,1.0,0.0),
		vec4(0.0,0.0,0.0,1.0)
	);*/

	highp mat4 modelview = (camera_inverse_transform * (world_transform * minst));
#ifdef ENABLE_AMBIENT_OCTREE
	highp mat4 ambient_octree_transform = (ambient_octree_inverse_transform * (world_transform * minst));
#endif

#else
	highp mat4 modelview = (camera_inverse_transform * world_transform);
#ifdef ENABLE_AMBIENT_OCTREE
	highp mat4 ambient_octree_transform = (ambient_octree_inverse_transform * world_transform);
#endif

#endif

#endif

#endif
	highp vec4 vertex_in = vertex_attrib; // vec4(vertex_attrib.xyz * data_attrib.x,1.0);
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

#ifdef ENABLE_AMBIENT_OCTREE

	ambient_octree_coords = (ambient_octree_transform * vertex_in).xyz;
#endif

	vertex_interp = (modelview * vertex_in).xyz;
	normal_interp = normalize((modelview * vec4(normal_in,0.0)).xyz);

#if defined(ENABLE_COLOR_INTERP)
#ifdef USE_COLOR_ATTRIB_SRGB_TO_LINEAR

	color_interp = vec4(
		color_attrib.r<0.04045 ? color_attrib.r * (1.0 / 12.92) : pow((color_attrib.r + 0.055) * (1.0 / (1 + 0.055)), 2.4),
		color_attrib.g<0.04045 ? color_attrib.g * (1.0 / 12.92) : pow((color_attrib.g + 0.055) * (1.0 / (1 + 0.055)), 2.4),
		color_attrib.b<0.04045 ? color_attrib.b * (1.0 / 12.92) : pow((color_attrib.b + 0.055) * (1.0 / (1 + 0.055)), 2.4),
		color_attrib.a
	);
#else
	color_interp = color_attrib;
#endif
#endif

#if defined(ENABLE_TANGENT_INTERP)
	tangent_interp=normalize((modelview * vec4(tangent_in,0.0)).xyz);
	binormal_interp = normalize( cross(normal_interp,tangent_interp) * binormalf );
#endif

#if defined(ENABLE_UV_INTERP)
	uv_interp = uv_attrib;
#endif
#if defined(ENABLE_UV2_INTERP)
	uv2_interp = uv2_attrib;
#endif

	float vertex_specular_exp = 40.0; //material_specular.a;



VERTEX_SHADER_CODE


#ifdef USE_DUAL_PARABOLOID
//for dual paraboloid shadow mapping
        highp vec3 vtx = vertex_interp;
        vtx.z*=dual_paraboloid.y; //side to affect
        vtx.z+=0.01;
        dp_clip=vtx.z;
        highp float len=length( vtx );
        vtx=normalize(vtx);
        vtx.xy/=1.0+vtx.z;
        vtx.z = len*dual_paraboloid.x; // it's a reciprocal(len - z_near) / (z_far - z_near);
        vtx+=normalize(vtx)*0.025;
        vtx.z = vtx.z * 2.0 - 1.0; // fit to clipspace
        vertex_interp=vtx;

        //vertex_interp.w = z_clip;

#endif

#ifdef USE_SHADOW_PASS

	float z_ofs = shadow_z_offset;
	z_ofs += (1.0-abs(normal_interp.z))*shadow_z_slope_scale;
	vertex_interp.z-=z_ofs;
#endif

#ifdef LIGHT_USE_SHADOW

        shadow_coord = shadow_matrix * vec4(vertex_interp,1.0);
	shadow_coord.xyz/=shadow_coord.w;

#ifdef LIGHT_USE_PSSM
	shadow_coord.y*=0.5;
	shadow_coord.y+=0.5;
	shadow_coord2 = shadow_matrix2 * vec4(vertex_interp,1.0);
	shadow_coord2.xyz/=shadow_coord2.w;
	shadow_coord2.y*=0.5;
#endif
#ifdef LIGHT_USE_PSSM4
	shadow_coord.x*=0.5;
	shadow_coord2.x*=0.5;

	shadow_coord3 = shadow_matrix3 * vec4(vertex_interp,1.0);
	shadow_coord3.xyz/=shadow_coord3.w;
	shadow_coord3.xy*=vec2(0.5);
	shadow_coord3.xy+=vec2(0.5);

	shadow_coord4 = shadow_matrix4 * vec4(vertex_interp,1.0);
	shadow_coord4.xyz/=shadow_coord4.w;
	shadow_coord4.xy*=vec2(0.5);
	shadow_coord4.x+=0.5;

#endif

#endif

#ifdef USE_FOG

	fog_interp.a = pow( clamp( (length(vertex_interp)-fog_params.x)/(fog_params.y-fog_params.x), 0.0, 1.0 ), fog_params.z );
	fog_interp.rgb = mix( fog_color_begin, fog_color_end, fog_interp.a );
#endif

#ifndef VERTEX_SHADER_WRITE_POSITION
//vertex shader might write a position
	gl_Position = projection_transform * vec4(vertex_interp,1.0);
#endif



#if !defined(USE_DEPTH_SHADOWS) && defined(USE_SHADOW_PASS)

    position_interp=gl_Position;

#endif


#ifdef USE_VERTEX_LIGHTING

	vec3 eye_vec = -normalize(vertex_interp);

#ifdef LIGHT_TYPE_DIRECTIONAL

	vec3 light_dir = -light_direction;
	float attenuation = light_attenuation.r;


#endif

#ifdef LIGHT_TYPE_OMNI
	vec3 light_dir = light_pos-vertex_interp;
	float radius = light_attenuation.g;
	float dist = min(length(light_dir),radius);
	light_dir=normalize(light_dir);
	float attenuation = pow( max(1.0 - dist/radius, 0.0), light_attenuation.b ) * light_attenuation.r;

#endif

#ifdef LIGHT_TYPE_SPOT

	vec3 light_dir = light_pos-vertex_interp;
	float radius = light_attenuation.g;
	float dist = min(length(light_dir),radius);
	light_dir=normalize(light_dir);
	float attenuation = pow(  max(1.0 - dist/radius, 0.0), light_attenuation.b ) * light_attenuation.r;
	vec3 spot_dir = light_direction;
	float spot_cutoff=light_spot_attenuation.r;
	float scos = max(dot(-light_dir, spot_dir),spot_cutoff);
	float rim = (1.0 - scos) / (1.0 - spot_cutoff);
	attenuation *= 1.0 - pow( rim, light_spot_attenuation.g);


#endif

#if defined(LIGHT_TYPE_DIRECTIONAL) || defined(LIGHT_TYPE_OMNI) || defined(LIGHT_TYPE_SPOT)

	//process_shade(normal_interp,light_dir,eye_vec,vertex_specular_exp,attenuation,diffuse_interp,specular_interp);
	{
		float NdotL = max(0.0,dot( normal_interp, light_dir ));
		vec3 half_vec = normalize(light_dir + eye_vec);
		float eye_light = max(dot(normal_interp, half_vec),0.0);
		diffuse_interp.rgb=light_diffuse * NdotL * attenuation;
		diffuse_interp.a=attenuation;
		if (NdotL > 0.0) {
			specular_interp=light_specular * pow( eye_light, vertex_specular_exp ) * attenuation;
		} else {
			specular_interp=vec3(0.0);
		}
	}

#else

#ifdef SHADELESS

	diffuse_interp=vec4(vec3(1.0),0.0);
	specular_interp=vec3(0.0);
# else

	diffuse_interp=vec4(0.0);
	specular_interp=vec3(0.0);
# endif

#endif




#endif


}


[fragment]


#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#define roundfix( m_val ) floor( (m_val) + 0.5 )
#else

precision mediump float;
precision mediump int;

#endif


//hack to use uv if no uv present so it works with lightmap
#ifdef ENABLE_AMBIENT_LIGHTMAP

#ifdef USE_LIGHTMAP_ON_UV2

#ifndef ENABLE_UV2_INTERP
#define ENABLE_UV2_INTERP
#endif

#else

#ifndef ENABLE_UV_INTERP
#define ENABLE_UV_INTERP
#endif

#endif

#endif


/* Varyings */

#if defined(ENABLE_COLOR_INTERP)
varying vec4 color_interp;
#endif

#if defined(ENABLE_UV_INTERP)
varying vec2 uv_interp;
#endif

#if defined(ENABLE_UV2_INTERP)
varying vec2 uv2_interp;
#endif

#if defined(ENABLE_TANGENT_INTERP)
varying vec3 tangent_interp;
varying vec3 binormal_interp;
#endif

#if defined(ENABLE_VAR1_INTERP)
varying vec4 var1_interp;
#endif

#if defined(ENABLE_VAR2_INTERP)
varying vec4 var2_interp;
#endif

#ifdef LIGHT_USE_PSSM
uniform vec3 light_pssm_split;
#endif

varying vec3 vertex_interp;
varying vec3 normal_interp;

#ifdef USE_FOG

varying vec4 fog_interp;

#endif

/* Material Uniforms */

#ifdef USE_VERTEX_LIGHTING

varying vec4 diffuse_interp;
varying vec3 specular_interp;

#endif

#if !defined(USE_DEPTH_SHADOWS) && defined(USE_SHADOW_PASS)

varying vec4 position_interp;

#endif



uniform vec3 light_pos;
uniform vec3 light_direction;
uniform vec3 light_attenuation;
uniform vec3 light_spot_attenuation;
uniform vec3 light_diffuse;
uniform vec3 light_specular;

uniform vec3 ambient_light;


#ifdef USE_FRAGMENT_LIGHTING



# ifdef USE_DEPTH_SHADOWS
# else
# endif

#endif

uniform float const_light_mult;
uniform float time;

#ifdef ENABLE_AMBIENT_OCTREE

varying highp vec3 ambient_octree_coords;
uniform highp float ambient_octree_lattice_size;
uniform highp vec2 ambient_octree_pix_size;
uniform highp vec2 ambient_octree_light_pix_size;
uniform highp float ambient_octree_lattice_divide;
uniform highp sampler2D ambient_octree_tex;
uniform highp sampler2D ambient_octree_light_tex;
uniform float ambient_octree_multiplier;
uniform int ambient_octree_steps;

#endif

#ifdef ENABLE_AMBIENT_LIGHTMAP

uniform highp sampler2D ambient_lightmap;
uniform float ambient_lightmap_multiplier;

#endif

#ifdef ENABLE_AMBIENT_DP_SAMPLER

uniform highp sampler2D ambient_dp_sampler;
uniform float ambient_dp_sampler_multiplier;

#endif

#ifdef ENABLE_AMBIENT_COLOR

uniform vec3 ambient_color;

#endif

FRAGMENT_SHADER_GLOBALS



#ifdef LIGHT_USE_SHADOW

varying highp vec4 shadow_coord;
#ifdef LIGHT_USE_PSSM
varying highp vec4 shadow_coord2;
#endif
#ifdef LIGHT_USE_PSSM4
varying highp vec4 shadow_coord3;
varying highp vec4 shadow_coord4;
#endif

uniform highp sampler2D shadow_texture;
uniform highp vec2 shadow_texel_size;

uniform float shadow_darkening;

#ifdef USE_DEPTH_SHADOWS

#define SHADOW_DEPTH(m_tex,m_uv) (texture2D((m_tex),(m_uv)).z)

#else

//#define SHADOW_DEPTH(m_tex,m_uv) dot(texture2D((m_tex),(m_uv)),highp vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  )
#define SHADOW_DEPTH(m_tex,m_uv) dot(texture2D((m_tex),(m_uv)),vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  )

#endif

#ifdef USE_SHADOW_PCF


#ifdef USE_SHADOW_PCF_HQ


float SAMPLE_SHADOW_TEX( highp vec2 coord, highp float refdepth) {

	float avg=(SHADOW_DEPTH(shadow_texture,coord) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(shadow_texel_size.x,0.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(-shadow_texel_size.x,0.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(0.0,shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(0.0,-shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(shadow_texel_size.x,shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(-shadow_texel_size.x,shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(shadow_texel_size.x,-shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(-shadow_texel_size.x,-shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(shadow_texel_size.x*2.0,0.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(-shadow_texel_size.x*2.0,0.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(0.0,shadow_texel_size.y*2.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(0.0,-shadow_texel_size.y*2.0)) < refdepth ?  0.0 : 1.0);
	return avg*(1.0/13.0);
}

#else

float SAMPLE_SHADOW_TEX( highp vec2 coord, highp float refdepth) {

	float avg=(SHADOW_DEPTH(shadow_texture,coord) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(shadow_texel_size.x,0.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(-shadow_texel_size.x,0.0)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(0.0,shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	avg+=(SHADOW_DEPTH(shadow_texture,coord+vec2(0.0,-shadow_texel_size.y)) < refdepth ?  0.0 : 1.0);
	return avg*0.2;
}

#endif




/*
	16x averaging
float SAMPLE_SHADOW_TEX( highp vec2 coord, highp float refdepth) {

	vec2 offset = vec2(
		lessThan(vec2(0.25),fract(gl_FragCoord.xy * 0.5))
		);
	offset.y += offset.x;  // y ^= x in floating point

	if (offset.y > 1.1)
		offset.y = 0.0;
	float avg = step( refdepth, SHADOW_DEPTH(shadow_texture, coord+ (offset + vec2(-1.5, 0.5))*shadow_texel_size) );
	avg+=step(refdepth, SHADOW_DEPTH(shadow_texture, coord+ (offset + vec2(0.5, 0.5))*shadow_texel_size) );
	avg+=step(refdepth, SHADOW_DEPTH(shadow_texture, coord+ (offset + vec2(-1.5, -1.5))*shadow_texel_size) );
	avg+=step(refdepth, SHADOW_DEPTH(shadow_texture, coord+ (offset + vec2(0.5, -1.5))*shadow_texel_size) );
	return avg * 0.25;
}
*/

/*
float SAMPLE_SHADOW_TEX( highp vec2 coord, highp float refdepth) {

	vec2 offset = vec2(
		lessThan(vec2(0.25),fract(gl_FragCoord.xy * 0.5))
		);
	offset.y += offset.x;  // y ^= x in floating point

	if (offset.y > 1.1)
		offset.y = 0.0;
	return step( refdepth, SHADOW_DEPTH(shadow_texture, coord+ offset*shadow_texel_size) );

}

*/
/* simple pcf4 */
//#define SAMPLE_SHADOW_TEX(m_coord,m_depth) ((step(m_depth,SHADOW_DEPTH(shadow_texture,m_coord))+step(m_depth,SHADOW_DEPTH(shadow_texture,m_coord+vec2(0.0,shadow_texel_size.y)))+step(m_depth,SHADOW_DEPTH(shadow_texture,m_coord+vec2(shadow_texel_size.x,0.0)))+step(m_depth,SHADOW_DEPTH(shadow_texture,m_coord+shadow_texel_size)))/4.0)

#endif

#ifdef USE_SHADOW_ESM

uniform float esm_multiplier;

float SAMPLE_SHADOW_TEX(vec2 p_uv,float p_depth) {

#if defined (USE_DEPTH_SHADOWS)
	//these only are used if interpolation exists
	highp float occluder = SHADOW_DEPTH(shadow_texture, p_uv);
#else
	vec2 unnormalized = p_uv/shadow_texel_size;
	vec2 fractional = fract(unnormalized);
	unnormalized = floor(unnormalized);

	vec4 exponent;
	exponent.x = SHADOW_DEPTH(shadow_texture, (unnormalized + vec2( -0.5, 0.5 )) * shadow_texel_size );
	exponent.y = SHADOW_DEPTH(shadow_texture, (unnormalized + vec2( 0.5, 0.5 )) * shadow_texel_size );
	exponent.z = SHADOW_DEPTH(shadow_texture, (unnormalized + vec2( 0.5, -0.5 )) * shadow_texel_size );
	exponent.w = SHADOW_DEPTH(shadow_texture, (unnormalized + vec2( -0.5, -0.5 )) * shadow_texel_size );

	highp float occluder = (exponent.w + (exponent.x - exponent.w) * fractional.y);
	occluder = occluder + ((exponent.z + (exponent.y - exponent.z) * fractional.y) - occluder)*fractional.x;
#endif
	return clamp(exp(esm_multiplier* ( occluder - p_depth )),0.0,1.0);

}


#endif

#if !defined(USE_SHADOW_PCF) && !defined(USE_SHADOW_ESM)

#define SAMPLE_SHADOW_TEX(m_coord,m_depth) (SHADOW_DEPTH(shadow_texture,m_coord) < m_depth ?  0.0 : 1.0)

#endif


#endif

#ifdef USE_DUAL_PARABOLOID

varying float dp_clip;

#endif

uniform highp mat4 camera_inverse_transform;

#if defined(ENABLE_TEXSCREEN)

uniform vec2 texscreen_screen_mult;
uniform vec4 texscreen_screen_clamp;
uniform sampler2D texscreen_tex;

#endif

#if defined(ENABLE_SCREEN_UV)

uniform vec2 screen_uv_mult;

#endif

void main() {

#ifdef USE_DUAL_PARABOLOID
        if (dp_clip<0.0)
            discard;
#endif

	//lay out everything, whathever is unused is optimized away anyway
        vec3 vertex = vertex_interp;
	vec4 diffuse = vec4(0.9,0.9,0.9,1.0);
	vec3 specular = vec3(0.0,0.0,0.0);
	vec3 emission = vec3(0.0,0.0,0.0);
	float specular_exp=1.0;
	float glow=0.0;
	float shade_param=0.0;
#ifdef DISABLE_FRONT_FACING
	float side=float(1)*2.0-1.0;
#else
	float side=float(gl_FrontFacing)*2.0-1.0;
#endif
#if defined(ENABLE_TANGENT_INTERP)
	vec3 binormal = normalize(binormal_interp)*side;
	vec3 tangent = normalize(tangent_interp)*side;
#endif
//	vec3 normal = abs(normalize(normal_interp))*side;
	vec3 normal = normalize(normal_interp)*side;
#if defined(ENABLE_SCREEN_UV)
	vec2 screen_uv = gl_FragCoord.xy*screen_uv_mult;
#endif

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

	float shadow_attenuation = 1.0;

#ifdef ENABLE_AMBIENT_LIGHTMAP

	vec3 ambientmap_color = vec3(0.0,0.0,0.0);
	vec2 ambientmap_uv = vec2(0.0,0.0);

#ifdef USE_LIGHTMAP_ON_UV2

	ambientmap_uv = uv2_interp;

#else

	ambientmap_uv = uv_interp;

#endif

	vec4 amcol = texture2D(ambient_lightmap,ambientmap_uv);
	shadow_attenuation=amcol.a;
	ambientmap_color = amcol.rgb;
	ambientmap_color*=ambient_lightmap_multiplier;
	ambientmap_color*=diffuse.rgb;



#endif


#ifdef ENABLE_AMBIENT_OCTREE

	vec3 ambientmap_color = vec3(0.0,0.0,0.0);


	{

		//read position from initial lattice grid
		highp vec3 lattice_pos = floor(ambient_octree_coords*ambient_octree_lattice_size);
		highp vec2 octant_uv = highp vec2(lattice_pos.x+ambient_octree_lattice_size*lattice_pos.z,lattice_pos.y);
		octant_uv=(octant_uv*highp vec2(2.0,4.0)+highp vec2(0.0,4.0));
		highp float ld = 1.0/ambient_octree_lattice_size;


		//go down the octree

		for(int i=0;i<ambient_octree_steps;i++) {


			highp vec3 sub=mod(ambient_octree_coords,ld);
			ld*=0.5;
			highp vec3 s = step(ld,sub);
			octant_uv+=s.xy;
			octant_uv.y+=s.z*2.0;
			octant_uv=(octant_uv+0.5)*ambient_octree_pix_size;
			highp vec4 new_uv = texture2D(ambient_octree_tex,octant_uv);
			octant_uv=floor(highp vec2( dot(new_uv.xy,highp vec2(65280.0,255.0)),  dot(new_uv.zw,highp vec2(65280.0,255.0)) )+0.5);//+ambient_octree_pix_size*0.5;

		}

		//sample color
		octant_uv=(octant_uv+0.5)*ambient_octree_light_pix_size;
		highp vec3 sub=(mod(ambient_octree_coords,ld)/ld);
		octant_uv.xy+=sub.xy*ambient_octree_light_pix_size.xy;
		vec3 col_up=texture2D(ambient_octree_light_tex,octant_uv).rgb;
		octant_uv.y+=ambient_octree_light_pix_size.y*2.0;
		vec3 col_down=texture2D(ambient_octree_light_tex,octant_uv).rgb;
		ambientmap_color=mix(col_up,col_down,sub.z)*ambient_octree_multiplier;

		ambientmap_color*=diffuse.rgb;

	}

#endif



#ifdef ENABLE_AMBIENT_DP_SAMPLER

	vec3 ambientmap_color = vec3(0.0,0.0,0.0);

	{

		vec3 dp_normal = normalize((vec4(normal,0) * camera_inverse_transform).xyz);
		vec2 ambient_uv = (dp_normal.xy / (1.0+abs(dp_normal.z)))*0.5+0.5; //dual paraboloid
		ambient_uv.y*=0.5;
		if (dp_normal.z<0) {

			ambient_uv.y=(0.5-ambient_uv.y)+0.5;

		}

		ambientmap_color = texture2D(ambient_dp_sampler,ambient_uv ).rgb * ambient_dp_sampler_multiplier;
		ambientmap_color*=diffuse.rgb;
	}

#endif




#ifdef LIGHT_USE_SHADOW
#ifdef LIGHT_TYPE_DIRECTIONAL

	float shadow_fade_exponent=5.0;  //hardcoded for now
	float shadow_fade=pow(length(vertex_interp)/light_attenuation.g,shadow_fade_exponent);

// optimization - skip shadows outside visible range
	if(shadow_fade<1.0){

#ifdef LIGHT_USE_PSSM


//	if (vertex_interp.z > light_pssm_split) {
#if 0
	highp vec3 splane = vec3(0.0,0.0,0.0);

	if (gl_FragCoord.w > light_pssm_split.x) {

		splane = shadow_coord.xyz;
		splane.y+=1.0;
	} else {
		splane = shadow_coord2.xyz;
	}
	splane.y*=0.5;
	shadow_attenuation=SAMPLE_SHADOW_TEX(splane.xy,splane.z);

#else
/*
	float sa_a = SAMPLE_SHADOW_TEX(shadow_coord.xy,shadow_coord.z);
	float sa_b = SAMPLE_SHADOW_TEX(shadow_coord2.xy,shadow_coord2.z);
	if (gl_FragCoord.w > light_pssm_split.x) {
		shadow_attenuation=sa_a;
	} else {
		shadow_attenuation=sa_b;
	}
*/

	vec2 pssm_coord;
	float pssm_z;

#if defined(LIGHT_USE_PSSM) && defined(USE_SHADOW_ESM)
#define USE_PSSM_BLEND
	float pssm_blend;
	vec2 pssm_coord_2;
	float pssm_z_2;
	vec3 light_pssm_split_inv = 1.0/light_pssm_split;
	float w_inv = 1.0/gl_FragCoord.w;
#endif

#ifdef LIGHT_USE_PSSM4


	if (gl_FragCoord.w > light_pssm_split.y) {

		if (gl_FragCoord.w > light_pssm_split.x) {
			pssm_coord=shadow_coord.xy;
			pssm_z=shadow_coord.z;
#if defined(USE_PSSM_BLEND)
			pssm_coord_2=shadow_coord2.xy;
			pssm_z_2=shadow_coord2.z;
			pssm_blend=smoothstep(0.0,light_pssm_split_inv.x,w_inv);
#endif

		} else {
			pssm_coord=shadow_coord2.xy;
			pssm_z=shadow_coord2.z;
#if defined(USE_PSSM_BLEND)
			pssm_coord_2=shadow_coord3.xy;
			pssm_z_2=shadow_coord3.z;
			pssm_blend=smoothstep(light_pssm_split_inv.x,light_pssm_split_inv.y,w_inv);
#endif

		}
	} else {


		if (gl_FragCoord.w > light_pssm_split.z) {
			pssm_coord=shadow_coord3.xy;
			pssm_z=shadow_coord3.z;
#if defined(USE_PSSM_BLEND)
			pssm_coord_2=shadow_coord4.xy;
			pssm_z_2=shadow_coord4.z;
			pssm_blend=smoothstep(light_pssm_split_inv.y,light_pssm_split_inv.z,w_inv);
#endif

		} else {
			pssm_coord=shadow_coord4.xy;
			pssm_z=shadow_coord4.z;
#if defined(USE_PSSM_BLEND)
			pssm_coord_2=shadow_coord4.xy;
			pssm_z_2=shadow_coord4.z;
			pssm_blend=0.0;
#endif

		}
	}

#else

	if (gl_FragCoord.w > light_pssm_split.x) {
		pssm_coord=shadow_coord.xy;
		pssm_z=shadow_coord.z;
#if defined(USE_PSSM_BLEND)
		pssm_coord_2=shadow_coord2.xy;
		pssm_z_2=shadow_coord2.z;
		pssm_blend=smoothstep(0.0,light_pssm_split_inv.x,w_inv);
#endif

	} else {
		pssm_coord=shadow_coord2.xy;
		pssm_z=shadow_coord2.z;
#if defined(USE_PSSM_BLEND)
		pssm_coord_2=shadow_coord2.xy;
		pssm_z_2=shadow_coord2.z;
		pssm_blend=0.0;
#endif

	}

#endif

	//one one sample
	shadow_attenuation=SAMPLE_SHADOW_TEX(pssm_coord,pssm_z);
#if defined(USE_PSSM_BLEND)
	shadow_attenuation=mix(shadow_attenuation,SAMPLE_SHADOW_TEX(pssm_coord_2,pssm_z_2),pssm_blend);
#endif


#endif

#else

	shadow_attenuation=SAMPLE_SHADOW_TEX(shadow_coord.xy,shadow_coord.z);
#endif

	shadow_attenuation=mix(shadow_attenuation,1.0,shadow_fade);
	}else{
	shadow_attenuation=1.0;
	};

#endif

#ifdef LIGHT_TYPE_OMNI

        vec3 splane=shadow_coord.xyz;///shadow_coord.w;
        float shadow_len=length(splane);
        splane=normalize(splane);
        float vofs=0.0;

        if (splane.z>=0.0) {

                splane.z+=1.0;
        } else {

                splane.z=1.0 - splane.z;
                vofs=0.5;
        }
        splane.xy/=splane.z;
        splane.xy=splane.xy * 0.5 + 0.5;
	float lradius = light_attenuation.g;
        splane.z = shadow_len / lradius;
        splane.y=clamp(splane.y,0.0,1.0)*0.5+vofs;

        shadow_attenuation=SAMPLE_SHADOW_TEX(splane.xy,splane.z);
#endif

#ifdef LIGHT_TYPE_SPOT

	shadow_attenuation=SAMPLE_SHADOW_TEX(shadow_coord.xy,shadow_coord.z);
#endif

	shadow_attenuation=mix(shadow_attenuation,1.0,shadow_darkening);
#endif


#ifdef USE_FRAGMENT_LIGHTING

	vec3 eye_vec = -normalize(vertex);

#ifdef LIGHT_TYPE_DIRECTIONAL

	vec3 light_dir = -light_direction;
	float attenuation = light_attenuation.r;


#endif

#ifdef LIGHT_TYPE_OMNI

	vec3 light_dir = light_pos-vertex;
	float radius = light_attenuation.g;
	float dist = min(length(light_dir),radius);
	light_dir=normalize(light_dir);
	float attenuation = pow( max(1.0 - dist/radius, 0.0), light_attenuation.b ) * light_attenuation.r;

#endif


#ifdef LIGHT_TYPE_SPOT

	vec3 light_dir = light_pos-vertex;
	float radius = light_attenuation.g;
	float dist = min(length(light_dir),radius);
	light_dir=normalize(light_dir);
	float attenuation = pow(  max(1.0 - dist/radius, 0.0), light_attenuation.b ) * light_attenuation.r;
	vec3 spot_dir = light_direction;
	float spot_cutoff=light_spot_attenuation.r;
	float scos = max(dot(-light_dir, spot_dir),spot_cutoff);
	float rim = (1.0 - scos) / (1.0 - spot_cutoff);
	attenuation *= 1.0 - pow( rim, light_spot_attenuation.g);

#endif

# if defined(LIGHT_TYPE_DIRECTIONAL) || defined(LIGHT_TYPE_OMNI) || defined (LIGHT_TYPE_SPOT)

	{

		vec3 mdiffuse = diffuse.rgb;
		vec3 light;

#if defined(USE_OUTPUT_SHADOW_COLOR)
		vec3 shadow_color=vec3(0.0,0.0,0.0);
#endif

#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader
{

LIGHT_SHADER_CODE

}
#else
//traditional lambert + blinn
		float NdotL = max(0.0,dot( normal, light_dir ));
		vec3 half_vec = normalize(light_dir + eye_vec);
		float eye_light = max(dot(normal, half_vec),0.0);

		light = light_diffuse * mdiffuse * NdotL;
		if (NdotL > 0.0) {
			light+=specular * light_specular * pow( eye_light, specular_exp );
		}
#endif
		diffuse.rgb = const_light_mult * ambient_light *diffuse.rgb + light * attenuation * shadow_attenuation;

#if defined(USE_OUTPUT_SHADOW_COLOR)
		diffuse.rgb += light * shadow_color * attenuation * (1.0 - shadow_attenuation);
#endif

#ifdef USE_FOG

		diffuse.rgb = mix(diffuse.rgb,fog_interp.rgb,fog_interp.a);

# if defined(LIGHT_TYPE_OMNI) || defined (LIGHT_TYPE_SPOT)
		diffuse.rgb = mix(mix(vec3(0.0),diffuse.rgb,attenuation),diffuse.rgb,const_light_mult);
# endif


#endif


	}


# endif

# if !defined(LIGHT_TYPE_DIRECTIONAL) && !defined(LIGHT_TYPE_OMNI) && !defined (LIGHT_TYPE_SPOT)
//none
#ifndef SHADELESS
	diffuse.rgb=ambient_light *diffuse.rgb;
#endif

# endif

	diffuse.rgb+=const_light_mult*emission;

#endif




#ifdef USE_VERTEX_LIGHTING

	vec3 ambient = const_light_mult*ambient_light*diffuse.rgb;
# if defined(LIGHT_TYPE_OMNI) || defined (LIGHT_TYPE_SPOT)
//	ambient*=diffuse_interp.a; //attenuation affects ambient too

# endif

//	diffuse.rgb=(diffuse.rgb * diffuse_interp.rgb + specular * specular_interp)*shadow_attenuation + ambient;
//	diffuse.rgb+=emission * const_light_mult;
	diffuse.rgb=(diffuse.rgb * diffuse_interp.rgb + specular * specular_interp)*shadow_attenuation + ambient;
	diffuse.rgb+=emission * const_light_mult;

#ifdef USE_FOG

	diffuse.rgb = mix(diffuse.rgb,fog_interp.rgb,fog_interp.a);

# if defined(LIGHT_TYPE_OMNI) || defined (LIGHT_TYPE_SPOT)
	diffuse.rgb = mix(mix(vec3(0.0),diffuse.rgb,diffuse_interp.a),diffuse.rgb,const_light_mult);
# endif

#endif

#endif


#if defined(ENABLE_AMBIENT_OCTREE) || defined(ENABLE_AMBIENT_LIGHTMAP) || defined(ENABLE_AMBIENT_DP_SAMPLER)
#if defined(ENABLE_AMBIENT_COLOR)
	ambientmap_color*=ambient_color;
#endif
	diffuse.rgb+=ambientmap_color;
#endif


#ifdef USE_SHADOW_PASS

#ifdef USE_DEPTH_SHADOWS

        //do nothing, depth is just written
#else
        // pack depth to rgba
        //highp float bias = 0.0005;
	highp float depth = ((position_interp.z / position_interp.w) + 1.0) * 0.5 + 0.0;//bias;
        highp vec4 comp = fract(depth * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
        comp -= comp.xxyz * vec4(0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
        gl_FragColor = comp;

#endif

#else



#ifdef USE_GLOW

	diffuse.a=glow;
#endif

#ifdef USE_8BIT_HDR
	diffuse.rgb*=0.25;
#endif

	gl_FragColor = diffuse;
#endif
}


