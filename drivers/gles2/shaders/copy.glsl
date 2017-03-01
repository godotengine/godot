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



float sRGB_gamma_correct(float c){
    float a = 0.055;
    if(c < 0.0031308)
	return 12.92*c;
    else
	return (1.0+a)*pow(c, 1.0/2.4) - a;
}

#define LUM_RANGE 4.0


#ifdef USE_CUBEMAP
varying vec3 cube_interp;
uniform samplerCube source_cube;
#else
varying vec2 uv_interp;
#ifdef HIGHP_SOURCE
uniform highp sampler2D source;
#else
uniform sampler2D source;
#endif
#endif
varying vec2 uv2_interp;


#ifdef USE_DEPTH
uniform highp sampler2D source_depth; //texunit:1
#endif

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
uniform highp float tonemap_white;
#endif

#ifdef USE_BCS

uniform vec3 bcs;

#endif

#ifdef USE_GLOW_COPY

uniform float bloom;
uniform float bloom_treshold;

#endif

#if defined(SHADOW_BLUR_V_PASS) || defined(SHADOW_BLUR_H_PASS) || defined(BLUR_V_PASS) || defined(BLUR_H_PASS) || defined(USE_HDR_REDUCE)

uniform vec2 pixel_size;
uniform float pixel_scale;
uniform float blur_magnitude;

#ifdef USE_HDR_STORE

uniform highp float hdr_time_delta;
uniform highp float hdr_exp_adj_speed;
uniform highp float min_luminance;
uniform highp float max_luminance;
uniform sampler2D source_vd_lum;

#endif

//endif
#elif defined(USE_FXAA)

uniform vec2 pixel_size;

#endif

#ifdef USE_ENERGY

uniform highp float energy;

#endif

#ifdef USE_CUSTOM_ALPHA
uniform float custom_alpha;
#endif


void main() {

	//vec4 color = color_interp;
#ifdef USE_HIGHP_SOURCE

#ifdef USE_CUBEMAP
	highp vec4 color = textureCube( source_cube,  normalize(cube_interp) );

#else
	highp vec4 color = texture2D( source,  uv_interp );
#endif

#else

#ifdef USE_CUBEMAP
	vec4 color = textureCube( source_cube,  normalize(cube_interp) );

#else
	vec4 color = texture2D( source,  uv_interp );
#endif


#endif

#ifdef USE_FXAA

#define FXAA_REDUCE_MIN   (1.0/ 128.0)
#define FXAA_REDUCE_MUL   (1.0 / 8.0)
#define FXAA_SPAN_MAX     8.0

	{
		vec3 rgbNW = texture2D(source, uv_interp + vec2(-1.0, -1.0) * pixel_size).xyz;
		vec3 rgbNE = texture2D(source, uv_interp + vec2(1.0, -1.0) * pixel_size).xyz;
		vec3 rgbSW = texture2D(source, uv_interp + vec2(-1.0, 1.0) * pixel_size).xyz;
		vec3 rgbSE = texture2D(source, uv_interp + vec2(1.0, 1.0) * pixel_size).xyz;
		vec3 rgbM  = color.rgb;
		vec3 luma = vec3(0.299, 0.587, 0.114);
		float lumaNW = dot(rgbNW, luma);
		float lumaNE = dot(rgbNE, luma);
		float lumaSW = dot(rgbSW, luma);
		float lumaSE = dot(rgbSE, luma);
		float lumaM  = dot(rgbM,  luma);
		float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
			float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

		 vec2 dir;
		 dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
		 dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

		 float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
				       (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

		 float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
		 dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
			   max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
			   dir * rcpDirMin)) * pixel_size;

		 vec3 rgbA = 0.5 * (
		     texture2D(source, uv_interp + dir * (1.0 / 3.0 - 0.5)).xyz +
		     texture2D(source, uv_interp + dir * (2.0 / 3.0 - 0.5)).xyz);
		 vec3 rgbB = rgbA * 0.5 + 0.25 * (
		     texture2D(source, uv_interp + dir * -0.5).xyz +
		     texture2D(source, uv_interp + dir * 0.5).xyz);

		 float lumaB = dot(rgbB, luma);
		 if ((lumaB < lumaMin) || (lumaB > lumaMax))
		     color.rgb = rgbA;
		 else
		     color.rgb = rgbB;
	}

#endif
	//color.rg=uv_interp;

#ifdef USE_BCS

	color.rgb = mix(vec3(0.0),color.rgb,bcs.x);
	color.rgb = mix(vec3(0.5),color.rgb,bcs.y);
	color.rgb = mix(vec3(dot(vec3(1.0),color.rgb)*0.33333),color.rgb,bcs.z);

#endif

#ifdef BLUR_V_PASS

	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-3.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-2.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-1.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*1.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*2.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*3.0)*pixel_scale);

	color*=(1.0/7.0)*blur_magnitude;

#endif

#ifdef BLUR_H_PASS


	color+=texture2D(source,uv_interp+vec2(pixel_size.x*-3.0,0.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*-2.0,0.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*-1.0,0.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*1.0,0.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*2.0,0.0)*pixel_scale);
	color+=texture2D(source,uv_interp+vec2(pixel_size.x*3.0,0.0)*pixel_scale);

	color*=(1.0/7.0)*blur_magnitude;

#endif

#ifdef SHADOW_BLUR_V_PASS

#ifdef USE_RGBA_DEPTH

#define VEC42DEPTH(m_vec4) dot(m_vec4,vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1))

	highp float depth = VEC42DEPTH(color)*0.383;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-3.0)*pixel_scale))*0.006;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-2.0)*pixel_scale))*0.061;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-1.0)*pixel_scale))*0.242;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(0.0,pixel_size.y*1.0)*pixel_scale))*0.242;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(0.0,pixel_size.y*2.0)*pixel_scale))*0.061;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(0.0,pixel_size.y*3.0)*pixel_scale))*0.006;
	highp vec4 comp = fract(depth * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	color=comp;

#else

	highp float depth = color.r*0.383;
	depth+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-3.0)*pixel_scale).r*0.006;
	depth+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-2.0)*pixel_scale).r*0.061;
	depth+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*-1.0)*pixel_scale).r*0.242;
	depth+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*1.0)*pixel_scale).r*0.242;
	depth+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*2.0)*pixel_scale).r*0.061;
	depth+=texture2D(source,uv_interp+vec2(0.0,pixel_size.y*3.0)*pixel_scale).r*0.006;

#ifdef USE_GLES_OVER_GL
	gl_FragDepth = depth;

#else
	gl_FragDepthEXT = depth;
#endif

	return;
#endif

#endif

#ifdef SHADOW_BLUR_H_PASS


#ifdef USE_RGBA_DEPTH

#define VEC42DEPTH(m_vec4) dot(m_vec4,vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1))

	highp float depth = VEC42DEPTH(color)*0.383;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(pixel_size.x*-3.0,0.0)*pixel_scale))*0.006;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(pixel_size.x*-2.0,0.0)*pixel_scale))*0.061;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(pixel_size.x*-1.0,0.0)*pixel_scale))*0.242;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(pixel_size.x*1.0,0.0)*pixel_scale))*0.242;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(pixel_size.x*2.0,0.0)*pixel_scale))*0.061;
	depth+=VEC42DEPTH(texture2D(source,uv_interp+vec2(pixel_size.x*3.0,0.0)*pixel_scale))*0.006;
	highp vec4 comp = fract(depth * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	color=comp;
#else


	highp float depth = color.r*0.383;
	depth+=texture2D(source,uv_interp+vec2(pixel_size.x*-3.0,0.0)*pixel_scale).r*0.006;
	depth+=texture2D(source,uv_interp+vec2(pixel_size.x*-2.0,0.0)*pixel_scale).r*0.061;
	depth+=texture2D(source,uv_interp+vec2(pixel_size.x*-1.0,0.0)*pixel_scale).r*0.242;
	depth+=texture2D(source,uv_interp+vec2(pixel_size.x*1.0,0.0)*pixel_scale).r*0.242;
	depth+=texture2D(source,uv_interp+vec2(pixel_size.x*2.0,0.0)*pixel_scale).r*0.061;
	depth+=texture2D(source,uv_interp+vec2(pixel_size.x*3.0,0.0)*pixel_scale).r*0.006;

#ifdef USE_GLES_OVER_GL
	gl_FragDepth = depth;
#else
	gl_FragDepthEXT = depth;
#endif

	return;

#endif

#endif

#ifdef USE_HDR

	highp float white_mult = 1.0;

#ifdef USE_8BIT_HDR
	highp vec4 _mult = vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1);
	highp float hdr_lum = dot(texture2D( hdr_source, vec2(0.0) ), _mult  );
	color.rgb*=LUM_RANGE;
	hdr_lum*=LUM_RANGE; //restore to full range
#else

	highp vec2 lv = texture2D( hdr_source, vec2(0.0) ).rg;
	highp float hdr_lum = lv.r;
#ifdef USE_AUTOWHITE
	white_mult=lv.g;
#endif

#endif

#ifdef USE_REINHARDT_TONEMAPPER
	float src_lum = dot(color.rgb,vec3(0.3, 0.58, 0.12));
	float lp = tonemap_exposure/hdr_lum*src_lum;
	float white = tonemap_white;
#ifdef USE_AUTOWHITE
	white_mult = (white_mult + 1.0 * white_mult);
	white_mult*=white_mult;
	white*=white_mult;
#endif
	lp = ( lp * ( 1.0 + ( lp / ( white) ) ) ) / ( 1.0 + lp );
	color.rgb*=lp;

#else

#ifdef USE_LOG_TONEMAPPER
	color.rgb = tonemap_exposure * log(color.rgb+1.0)/log(hdr_lum+1.0);
#else
	highp float tone_scale = tonemap_exposure / hdr_lum; //only linear supported
	color.rgb*=tone_scale;
#endif

#endif

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

#ifdef USE_GLOW_SCREEN

	color.rgb = clamp((color.rgb + glow.rgb) - (color.rgb * glow.rgb), 0.0, 1.0);

#endif

#ifdef USE_GLOW_SOFTLIGHT

	{

		glow.rgb = (glow.rgb * 0.5) + 0.5;
		color.r =  (glow.r <= 0.5) ? (color.r - (1.0 - 2.0 * glow.r) * color.r * (1.0 - color.r)) : (((glow.r > 0.5) && (color.r <= 0.25)) ? (color.r + (2.0 * glow.r - 1.0) * (4.0 * color.r * (4.0 * color.r + 1.0) * (color.r - 1.0) + 7.0 * color.r)) : (color.r + (2.0 * glow.r - 1.0) * (sqrt(color.r) - color.r)));
		color.g =  (glow.g <= 0.5) ? (color.g - (1.0 - 2.0 * glow.g) * color.g * (1.0 - color.g)) : (((glow.g > 0.5) && (color.g <= 0.25)) ? (color.g + (2.0 * glow.g - 1.0) * (4.0 * color.g * (4.0 * color.g + 1.0) * (color.g - 1.0) + 7.0 * color.g)) : (color.g + (2.0 * glow.g - 1.0) * (sqrt(color.g) - color.g)));
		color.b =  (glow.b <= 0.5) ? (color.b - (1.0 - 2.0 * glow.b) * color.b * (1.0 - color.b)) : (((glow.b > 0.5) && (color.b <= 0.25)) ? (color.b + (2.0 * glow.b - 1.0) * (4.0 * color.b * (4.0 * color.b + 1.0) * (color.b - 1.0) + 7.0 * color.b)) : (color.b + (2.0 * glow.b - 1.0) * (sqrt(color.b) - color.b)));
	}

#endif

#if !defined(USE_GLOW_SCREEN) && !defined(USE_GLOW_SOFTLIGHT)
	color.rgb+=glow.rgb;
#endif



#endif

#ifdef USE_SRGB

#if 0
	//this was fast, but was commented out because it looked kind of shitty, might it be fixable?

	{ //i have my doubts about how fast this is

		color.rgb = min(color.rgb,vec3(1.0)); //clamp just in case
		vec3 S1 = sqrt(color.rgb);
		vec3 S2 = sqrt(S1);
		vec3 S3 = sqrt(S2);
		color.rgb = 0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.225411470 * color.rgb;
	}
#else

	color.r=sRGB_gamma_correct(color.r);
	color.g=sRGB_gamma_correct(color.g);
	color.b=sRGB_gamma_correct(color.b);

#endif

#endif

#ifdef USE_HDR_COPY

	//highp float lum = dot(color.rgb,highp vec3(1.0/3.0,1.0/3.0,1.0/3.0));
	//highp float lum = max(color.r,max(color.g,color.b));
	highp float lum = dot(color.rgb,vec3(0.3, 0.58, 0.12));

	//lum=log(lum+0.0001); //everyone does it

#ifdef USE_8BIT_HDR
	highp vec4 comp = fract(lum * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	color=comp;
#else
	color.rgb=vec3(lum);
#endif


#endif

#ifdef USE_HDR_REDUCE

#ifdef USE_8BIT_HDR
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
#else

	highp float lum_accum = color.r;
	highp float lum_max = color.g;

#define LUM_REDUCE(m_uv) \
	{\
		vec2 val = texture2D( source,  uv_interp+m_uv ).rg;\
		lum_accum+=val.x;\
		lum_max=max(val.y,lum_max);\
	}

	LUM_REDUCE( vec2(-pixel_size.x,-pixel_size.y) );
	LUM_REDUCE( vec2(0.0,-pixel_size.y) );
	LUM_REDUCE( vec2(pixel_size.x,-pixel_size.y) );
	LUM_REDUCE( vec2(-pixel_size.x,0.0) );
	LUM_REDUCE( vec2(pixel_size.x,0.0) );
	LUM_REDUCE( vec2(-pixel_size.x,pixel_size.y) );
	LUM_REDUCE( vec2(0.0,pixel_size.y) );
	LUM_REDUCE( vec2(pixel_size.x,pixel_size.y) );
	lum_accum/=9.0;

#endif

#ifdef USE_HDR_STORE

	//lum_accum=exp(lum_accum);

#ifdef USE_8BIT_HDR	

	highp float vd_lum = dot(texture2D( source_vd_lum, vec2(0.0) ), _multcv  );
	lum_accum = clamp( vd_lum + (lum_accum-vd_lum)*hdr_time_delta*hdr_exp_adj_speed,min_luminance*(1.0/LUM_RANGE),max_luminance*(1.0/LUM_RANGE));
#else
	highp float vd_lum=texture2D( source_vd_lum, vec2(0.0) ).r;
	lum_accum = clamp( vd_lum + (lum_accum-vd_lum)*hdr_time_delta*hdr_exp_adj_speed,min_luminance,max_luminance);
#endif

#endif

#ifdef USE_8BIT_HDR
	highp vec4 comp = fract(lum_accum * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	color=comp;
#else
#ifdef USE_AUTOWHITE
	color.r=lum_accum;
	color.g=lum_max;
#else
	color.rgb=vec3(lum_accum);
#endif


#endif

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

#ifdef USE_CUSTOM_ALPHA
	color.a=custom_alpha;
#endif


        gl_FragColor = color;

#ifdef USE_DEPTH
	gl_FragDepth = texture(source_depth,uv_interp).r;
#endif

}

