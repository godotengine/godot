[vertex]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

attribute highp vec4 vertex_attrib; // attrib:0
#ifdef USE_CUBEMAP
attribute vec3 cube_in; // attrib:4
#else
attribute vec2 uv_in; // attrib:4
#endif
attribute vec2 uv2_in; // attrib:5

#ifdef USE_CUBEMAP
varying vec3 cube_interp;
#else
varying vec2 uv_interp;
#endif

varying vec2 uv2_interp;

void main() {

#ifdef USE_CUBEMAP
	cube_interp = cube_in;
#else
	uv_interp = uv_in;
#endif
	uv2_interp = uv2_in;
	gl_Position = vertex_attrib;
}

[fragment]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif


#define LUM_RANGE 4.0


#ifdef USE_CUBEMAP
varying vec3 cube_interp;
uniform samplerCube source_cube;
#else
varying vec2 uv_interp;
uniform sampler2D source;
#endif
varying vec2 uv2_interp;

#ifdef USE_GLOW

uniform sampler2D glow_source;

#endif


#if defined(USE_HDR) && defined(USE_GLOW_COPY)
uniform highp float hdr_glow_treshold;
uniform highp float hdr_glow_scale;
#endif

#ifdef USE_HDR

uniform sampler2D hdr_source;
uniform highp float tonemap_exposure;

#endif

#ifdef USE_BCS

uniform vec3 bcs;

#endif

#ifdef USE_GAMMA

uniform float gamma;

#endif

#ifdef USE_GLOW_COPY

uniform float bloom;
uniform float bloom_treshold;

#endif

#if defined(BLUR_V_PASS) || defined(BLUR_H_PASS) || defined(USE_HDR_REDUCE)

uniform vec2 pixel_size;

#ifdef USE_HDR_STORE

uniform highp float hdr_time_delta;
uniform highp float hdr_exp_adj_speed;
uniform highp float min_luminance;
uniform highp float max_luminance;
uniform sampler2D source_vd_lum;

#endif

#endif

#ifdef USE_ENERGY

uniform highp float energy;

#endif


void main() {

	//vec4 color = color_interp;
#ifdef USE_CUBEMAP
	vec4 color = textureCube( source_cube,  normalize(cube_interp) );

#else
	vec4 color = texture2D( source,  uv_interp );
#endif

	//color.rg=uv_interp;

#ifdef USE_BCS

	color.rgb = mix(vec3(0.0),color.rgb,bcs.x);
	color.rgb = mix(vec3(0.5),color.rgb,bcs.y);
	color.rgb = mix(vec3(dot(vec3(1.0),color.rgb)*0.33333),color.rgb,bcs.z);

#endif

#ifdef BLUR_V_PASS

	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-3.0));
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-2.0));
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-1.0));
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*1.0));
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*2.0));
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*3.0));

	color*=(1.0/7.0);

#endif

#ifdef BLUR_H_PASS


	color+=texture2D(source,uv_interp+vec2(pixel_size.x*-3.0,0.0));
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*-2.0,0.0));
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*-1.0,0.0));
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*1.0,0.0));
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*2.0,0.0));
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*3.0,0.0));

	color*=(1.0/7.0);

#endif

#ifdef USE_HDR

	highp vec4 _mult = vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1);
	highp float hdr_lum = dot(texture2D( hdr_source, vec2(0.0) ), _mult  );
	color.rgb*=LUM_RANGE;
	hdr_lum*=LUM_RANGE; //restore to full range
	highp float tone_scale = tonemap_exposure / hdr_lum; //only linear supported
	color.rgb*=tone_scale;

#endif

#ifdef USE_GLOW_COPY

	highp vec3 glowcol = color.rgb*color.a+step(bloom_treshold,dot(vec3(0.3333,0.3333,0.3333),color.rgb))*bloom*color.rgb;

#ifdef USE_HDR
	highp float collum = max(color.r,max(color.g,color.b));
	glowcol+=color.rgb*max(collum-hdr_glow_treshold,0.0)*hdr_glow_scale;
#endif
	color.rgb=glowcol;
	color.a=0.0;

#endif


#ifdef USE_GLOW

	vec4 glow = texture2D( glow_source,  uv2_interp );

	color.rgb+=glow.rgb;

#endif

#ifdef USE_GAMMA

	color.rgb = pow(color.rgb,gamma);

#endif


#ifdef USE_HDR_COPY

	//highp float lum = dot(color.rgb,highp vec3(1.0/3.0,1.0/3.0,1.0/3.0));
	highp float lum = max(color.r,max(color.g,color.b));
	highp vec4 comp = fract(lum * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	color=comp;
#endif

#ifdef USE_HDR_REDUCE

	highp vec4 _multcv = vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0, 1.0);
	highp float lum_accum = dot(color,_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(-pixel_size.x,-pixel_size.y) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(0.0,-pixel_size.y) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(pixel_size.x,-pixel_size.y) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(-pixel_size.x,0.0) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(pixel_size.x,0.0) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(-pixel_size.x,pixel_size.y) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(0.0,pixel_size.y) ),_multcv  );
	lum_accum += dot(texture2D( source,  uv_interp+vec2(pixel_size.x,pixel_size.y) ),_multcv  );
	lum_accum/=9.0;

#ifdef USE_HDR_STORE

	highp float vd_lum = dot(texture2D( source_vd_lum, vec2(0.0) ), _multcv  );
	lum_accum = clamp( vd_lum + (lum_accum-vd_lum)*hdr_time_delta*hdr_exp_adj_speed,min_luminance*(1.0/LUM_RANGE),max_luminance*(1.0/LUM_RANGE));
#endif

	highp vec4 comp = fract(lum_accum * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	color=comp;
#endif

#ifdef USE_RGBE

	color.rgb = pow(color.rgb,color.a*255.0-(8.0+128.0));
#endif

#ifdef USE_ENERGY
	color.rgb*=energy;
#endif

#ifdef USE_NO_ALPHA
        color.a=1.0;
#endif

        gl_FragColor = color;
}

