[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;

#ifdef USE_BLUR_SECTION

uniform vec4 blur_section;

#endif

void main() {

	uv_interp = uv_in;
	gl_Position = vertex_attrib;
#ifdef USE_BLUR_SECTION

	uv_interp = blur_section.xy + uv_interp * blur_section.zw;
	gl_Position.xy = (blur_section.xy + (gl_Position.xy * 0.5 + 0.5) * blur_section.zw) * 2.0 - 1.0;
#endif
}

[fragment]

#if !defined(GLES_OVER_GL)
precision mediump float;
#endif

in vec2 uv_interp;
uniform sampler2D source_color; //texunit:0

#ifdef SSAO_MERGE
uniform sampler2D source_ssao; //texunit:1
#endif

uniform float lod;
uniform vec2 pixel_size;


layout(location = 0) out vec4 frag_color;

#ifdef SSAO_MERGE

uniform vec4 ssao_color;

#endif

#if defined (GLOW_GAUSSIAN_HORIZONTAL) || defined(GLOW_GAUSSIAN_VERTICAL)

uniform float glow_strength;

#endif

#if defined(DOF_FAR_BLUR) || defined (DOF_NEAR_BLUR)

#ifdef DOF_QUALITY_LOW
const int dof_kernel_size=5;
const int dof_kernel_from=2;
const float dof_kernel[5] = float[] (0.153388,0.221461,0.250301,0.221461,0.153388);
#endif

#ifdef DOF_QUALITY_MEDIUM
const int dof_kernel_size=11;
const int dof_kernel_from=5;
const float dof_kernel[11] = float[] (0.055037,0.072806,0.090506,0.105726,0.116061,0.119726,0.116061,0.105726,0.090506,0.072806,0.055037);

#endif

#ifdef DOF_QUALITY_HIGH
const int dof_kernel_size=21;
const int dof_kernel_from=10;
const float dof_kernel[21] = float[] (0.028174,0.032676,0.037311,0.041944,0.046421,0.050582,0.054261,0.057307,0.059587,0.060998,0.061476,0.060998,0.059587,0.057307,0.054261,0.050582,0.046421,0.041944,0.037311,0.032676,0.028174);
#endif

uniform sampler2D dof_source_depth; //texunit:1
uniform float dof_begin;
uniform float dof_end;
uniform vec2 dof_dir;
uniform float dof_radius;

#ifdef DOF_NEAR_BLUR_MERGE

uniform sampler2D source_dof_original; //texunit:2
#endif

#endif


#ifdef GLOW_FIRST_PASS

uniform float exposure;
uniform float white;

#ifdef GLOW_USE_AUTO_EXPOSURE

uniform highp sampler2D source_auto_exposure; //texunit:1
uniform highp float auto_exposure_grey;

#endif

uniform float glow_bloom;
uniform float glow_hdr_threshold;
uniform float glow_hdr_scale;

#endif

uniform float camera_z_far;
uniform float camera_z_near;

void main() {



#ifdef GAUSSIAN_HORIZONTAL
	vec2 pix_size = pixel_size;
	pix_size*=0.5; //reading from larger buffer, so use more samples
	vec4 color =textureLod( source_color,  uv_interp+vec2( 0.0, 0.0)*pix_size,lod )*0.214607;
	color+=textureLod( source_color,  uv_interp+vec2( 1.0, 0.0)*pix_size,lod )*0.189879;
	color+=textureLod( source_color,  uv_interp+vec2( 2.0, 0.0)*pix_size,lod )*0.157305;
	color+=textureLod( source_color,  uv_interp+vec2( 3.0, 0.0)*pix_size,lod )*0.071303;
	color+=textureLod( source_color,  uv_interp+vec2(-1.0, 0.0)*pix_size,lod )*0.189879;
	color+=textureLod( source_color,  uv_interp+vec2(-2.0, 0.0)*pix_size,lod )*0.157305;
	color+=textureLod( source_color,  uv_interp+vec2(-3.0, 0.0)*pix_size,lod )*0.071303;
	frag_color = color;
#endif

#ifdef GAUSSIAN_VERTICAL
	vec4 color =textureLod( source_color,  uv_interp+vec2( 0.0, 0.0)*pixel_size,lod )*0.38774;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0, 1.0)*pixel_size,lod )*0.24477;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0, 2.0)*pixel_size,lod )*0.06136;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0,-1.0)*pixel_size,lod )*0.24477;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0,-2.0)*pixel_size,lod )*0.06136;
	frag_color = color;
#endif

//glow uses larger sigma for a more rounded blur effect

#ifdef GLOW_GAUSSIAN_HORIZONTAL
	vec2 pix_size = pixel_size;
	pix_size*=0.5; //reading from larger buffer, so use more samples
	vec4 color =textureLod( source_color,  uv_interp+vec2( 0.0, 0.0)*pix_size,lod )*0.174938;
	color+=textureLod( source_color,  uv_interp+vec2( 1.0, 0.0)*pix_size,lod )*0.165569;
	color+=textureLod( source_color,  uv_interp+vec2( 2.0, 0.0)*pix_size,lod )*0.140367;
	color+=textureLod( source_color,  uv_interp+vec2( 3.0, 0.0)*pix_size,lod )*0.106595;
	color+=textureLod( source_color,  uv_interp+vec2(-1.0, 0.0)*pix_size,lod )*0.165569;
	color+=textureLod( source_color,  uv_interp+vec2(-2.0, 0.0)*pix_size,lod )*0.140367;
	color+=textureLod( source_color,  uv_interp+vec2(-3.0, 0.0)*pix_size,lod )*0.106595;
	color*=glow_strength;
	frag_color = color;
#endif

#ifdef GLOW_GAUSSIAN_VERTICAL
	vec4 color =textureLod( source_color,  uv_interp+vec2(0.0, 0.0)*pixel_size,lod )*0.288713;
	color+=textureLod( source_color,  uv_interp+vec2(0.0, 1.0)*pixel_size,lod )*0.233062;
	color+=textureLod( source_color,  uv_interp+vec2(0.0, 2.0)*pixel_size,lod )*0.122581;
	color+=textureLod( source_color,  uv_interp+vec2(0.0,-1.0)*pixel_size,lod )*0.233062;
	color+=textureLod( source_color,  uv_interp+vec2(0.0,-2.0)*pixel_size,lod )*0.122581;
	color*=glow_strength;
	frag_color = color;
#endif

#ifdef DOF_FAR_BLUR

	vec4 color_accum = vec4(0.0);

	float depth = textureLod( dof_source_depth, uv_interp, 0.0).r;
	depth = depth * 2.0 - 1.0;
#ifdef USE_ORTHOGONAL_PROJECTION
	depth = ((depth + (camera_z_far + camera_z_near)/(camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near))/2.0;
#else
	depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth * (camera_z_far - camera_z_near));
#endif

	float amount = smoothstep(dof_begin,dof_end,depth);
	float k_accum=0.0;

	for(int i=0;i<dof_kernel_size;i++) {

		int int_ofs = i-dof_kernel_from;
		vec2 tap_uv = uv_interp + dof_dir * float(int_ofs) * amount * dof_radius;

		float tap_k = dof_kernel[i];

		float tap_depth = texture( dof_source_depth, tap_uv, 0.0).r;
		tap_depth = tap_depth * 2.0 - 1.0;
#ifdef USE_ORTHOGONAL_PROJECTION
		tap_depth = ((tap_depth + (camera_z_far + camera_z_near)/(camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near))/2.0;
#else
		tap_depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - tap_depth * (camera_z_far - camera_z_near));
#endif
		float tap_amount = mix(smoothstep(dof_begin,dof_end,tap_depth),1.0,int_ofs==0);
		tap_amount*=tap_amount*tap_amount; //prevent undesired glow effect

		vec4 tap_color = textureLod( source_color, tap_uv, 0.0) * tap_k;

		k_accum+=tap_k*tap_amount;
		color_accum+=tap_color*tap_amount;


	}

	if (k_accum>0.0) {
		color_accum/=k_accum;
	}

	frag_color = color_accum;///k_accum;

#endif

#ifdef DOF_NEAR_BLUR

	vec4 color_accum = vec4(0.0);

	float max_accum=0;

	for(int i=0;i<dof_kernel_size;i++) {

		int int_ofs = i-dof_kernel_from;
		vec2 tap_uv = uv_interp + dof_dir * float(int_ofs) * dof_radius;
		float ofs_influence = max(0.0,1.0-float(abs(int_ofs))/float(dof_kernel_from));

		float tap_k = dof_kernel[i];

		vec4 tap_color = textureLod( source_color, tap_uv, 0.0);

		float tap_depth = texture( dof_source_depth, tap_uv, 0.0).r;
		tap_depth = tap_depth * 2.0 - 1.0;
#ifdef USE_ORTHOGONAL_PROJECTION	
		tap_depth = ((tap_depth + (camera_z_far + camera_z_near)/(camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near))/2.0;
#else
		tap_depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - tap_depth * (camera_z_far - camera_z_near));
#endif
		float tap_amount = 1.0-smoothstep(dof_end,dof_begin,tap_depth);
		tap_amount*=tap_amount*tap_amount; //prevent undesired glow effect

#ifdef DOF_NEAR_FIRST_TAP

		tap_color.a= 1.0-smoothstep(dof_end,dof_begin,tap_depth);

#endif

		max_accum=max(max_accum,tap_amount*ofs_influence);

		color_accum+=tap_color*tap_k;

	}

	color_accum.a=max(color_accum.a,sqrt(max_accum));


#ifdef DOF_NEAR_BLUR_MERGE

	vec4 original = textureLod( source_dof_original, uv_interp, 0.0);
	color_accum = mix(original,color_accum,color_accum.a);

#endif

#ifndef DOF_NEAR_FIRST_TAP
	//color_accum=vec4(vec3(color_accum.a),1.0);
#endif
	frag_color = color_accum;

#endif



#ifdef GLOW_FIRST_PASS

#ifdef GLOW_USE_AUTO_EXPOSURE

	frag_color/=texelFetch(source_auto_exposure,ivec2(0,0),0).r/auto_exposure_grey;
#endif
	frag_color*=exposure;

	float luminance = max(frag_color.r,max(frag_color.g,frag_color.b));
	float feedback = max( smoothstep(glow_hdr_threshold,glow_hdr_threshold+glow_hdr_scale,luminance), glow_bloom );

	frag_color *= feedback;

#endif


#ifdef SIMPLE_COPY
	vec4 color =textureLod( source_color,  uv_interp,0.0);
	frag_color = color;
#endif

#ifdef SSAO_MERGE

	vec4 color =textureLod( source_color,  uv_interp,0.0);
	float ssao =textureLod( source_ssao,  uv_interp,0.0).r;

	frag_color = vec4( mix(color.rgb,color.rgb*mix(ssao_color.rgb,vec3(1.0),ssao),color.a), 1.0 );

#endif


}
