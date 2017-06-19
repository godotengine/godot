[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;

void main() {

	gl_Position = vertex_attrib;
	uv_interp = uv_in;
#ifdef V_FLIP
	uv_interp.y = 1.0-uv_interp.y;
#endif

}

[fragment]


in vec2 uv_interp;

uniform highp sampler2D source; //texunit:0

uniform float exposure;
uniform float white;

#ifdef USE_AUTO_EXPOSURE

uniform highp sampler2D source_auto_exposure; //texunit:1
uniform highp float auto_exposure_grey;

#endif

#if defined(USE_GLOW_LEVEL1) || defined(USE_GLOW_LEVEL2) || defined(USE_GLOW_LEVEL3) || defined(USE_GLOW_LEVEL4) || defined(USE_GLOW_LEVEL5) || defined(USE_GLOW_LEVEL6) || defined(USE_GLOW_LEVEL7)

uniform highp sampler2D source_glow; //texunit:2
uniform highp float glow_intensity;

#endif

#ifdef USE_BCS

uniform vec3 bcs;

#endif

#ifdef USE_COLOR_CORRECTION

uniform sampler2D color_correction; //texunit:3

#endif


layout(location = 0) out vec4 frag_color;

#ifdef USE_GLOW_FILTER_BICUBIC

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float w0(float a)
{
    return (1.0/6.0)*(a*(a*(-a + 3.0) - 3.0) + 1.0);
}

float w1(float a)
{
    return (1.0/6.0)*(a*a*(3.0*a - 6.0) + 4.0);
}

float w2(float a)
{
    return (1.0/6.0)*(a*(a*(-3.0*a + 3.0) + 3.0) + 1.0);
}

float w3(float a)
{
    return (1.0/6.0)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
float h0(float a)
{
    return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a)
{
    return 1.0 + w3(a) / (w2(a) + w3(a));
}

uniform ivec2 glow_texture_size;

vec4 texture2D_bicubic(sampler2D tex, vec2 uv,int p_lod)
{
	float lod=float(p_lod);
	vec2 tex_size = vec2(glow_texture_size >> p_lod);
	vec2 pixel_size =1.0/tex_size;
	uv = uv*tex_size + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * pixel_size;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * pixel_size;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * pixel_size;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * pixel_size;

	return g0(fuv.y) * (g0x * textureLod(tex, p0,lod)  +
			    g1x * textureLod(tex, p1,lod)) +
			g1(fuv.y) * (g0x * textureLod(tex, p2,lod)  +
				     g1x * textureLod(tex, p3,lod));
}



#define GLOW_TEXTURE_SAMPLE(m_tex,m_uv,m_lod) texture2D_bicubic(m_tex,m_uv,m_lod)

#else

#define GLOW_TEXTURE_SAMPLE(m_tex,m_uv,m_lod) textureLod(m_tex,m_uv,float(m_lod))

#endif


void main() {

	ivec2 coord = ivec2(gl_FragCoord.xy);
	vec3 color = texelFetch(source,coord,0).rgb;


#ifdef USE_AUTO_EXPOSURE

	color/=texelFetch(source_auto_exposure,ivec2(0,0),0).r/auto_exposure_grey;
#endif

	color*=exposure;


#if defined(USE_GLOW_LEVEL1) || defined(USE_GLOW_LEVEL2) || defined(USE_GLOW_LEVEL3) || defined(USE_GLOW_LEVEL4) || defined(USE_GLOW_LEVEL5) || defined(USE_GLOW_LEVEL6) || defined(USE_GLOW_LEVEL7)
	vec3 glow = vec3(0.0);

#ifdef USE_GLOW_LEVEL1

	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,1).rgb;
#endif

#ifdef USE_GLOW_LEVEL2
	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,2).rgb;
#endif

#ifdef USE_GLOW_LEVEL3
	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,3).rgb;
#endif

#ifdef USE_GLOW_LEVEL4
	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,4).rgb;
#endif

#ifdef USE_GLOW_LEVEL5
	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,5).rgb;
#endif

#ifdef USE_GLOW_LEVEL6
	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,6).rgb;
#endif

#ifdef USE_GLOW_LEVEL7
	glow+=GLOW_TEXTURE_SAMPLE(source_glow,uv_interp,7).rgb;
#endif


	glow *= glow_intensity;



#ifdef USE_GLOW_REPLACE

	color.rgb = glow;

#endif

#ifdef USE_GLOW_SCREEN

	color.rgb = clamp((color.rgb + glow) - (color.rgb * glow), 0.0, 1.0);

#endif

#ifdef USE_GLOW_SOFTLIGHT

	{

		glow = (glow * 0.5) + 0.5;
		color.r =  (glow.r <= 0.5) ? (color.r - (1.0 - 2.0 * glow.r) * color.r * (1.0 - color.r)) : (((glow.r > 0.5) && (color.r <= 0.25)) ? (color.r + (2.0 * glow.r - 1.0) * (4.0 * color.r * (4.0 * color.r + 1.0) * (color.r - 1.0) + 7.0 * color.r)) : (color.r + (2.0 * glow.r - 1.0) * (sqrt(color.r) - color.r)));
		color.g =  (glow.g <= 0.5) ? (color.g - (1.0 - 2.0 * glow.g) * color.g * (1.0 - color.g)) : (((glow.g > 0.5) && (color.g <= 0.25)) ? (color.g + (2.0 * glow.g - 1.0) * (4.0 * color.g * (4.0 * color.g + 1.0) * (color.g - 1.0) + 7.0 * color.g)) : (color.g + (2.0 * glow.g - 1.0) * (sqrt(color.g) - color.g)));
		color.b =  (glow.b <= 0.5) ? (color.b - (1.0 - 2.0 * glow.b) * color.b * (1.0 - color.b)) : (((glow.b > 0.5) && (color.b <= 0.25)) ? (color.b + (2.0 * glow.b - 1.0) * (4.0 * color.b * (4.0 * color.b + 1.0) * (color.b - 1.0) + 7.0 * color.b)) : (color.b + (2.0 * glow.b - 1.0) * (sqrt(color.b) - color.b)));
	}

#endif

#if !defined(USE_GLOW_SCREEN) && !defined(USE_GLOW_SOFTLIGHT) && !defined(USE_GLOW_REPLACE)
	color.rgb+=glow;
#endif


#endif


#ifdef USE_REINDHART_TONEMAPPER

	{
		color.rgb = ( color.rgb * ( 1.0 + ( color.rgb / ( white) ) ) ) / ( 1.0 + color.rgb );

	}
#endif

#ifdef USE_FILMIC_TONEMAPPER

	{

		float A = 0.15;
		float B = 0.50;
		float C = 0.10;
		float D = 0.20;
		float E = 0.02;
		float F = 0.30;
		float W = 11.2;

		vec3 coltn = ((color.rgb*(A*color.rgb+C*B)+D*E)/(color.rgb*(A*color.rgb+B)+D*F))-E/F;
		float whitetn = ((white*(A*white+C*B)+D*E)/(white*(A*white+B)+D*F))-E/F;

		color.rgb=coltn/whitetn;

	}
#endif

#ifdef USE_ACES_TONEMAPPER

	{
		float a = 2.51f;
		float b = 0.03f;
		float c = 2.43f;
		float d = 0.59f;
		float e = 0.14f;
		color.rgb = clamp((color.rgb*(a*color.rgb+b))/(color.rgb*(c*color.rgb+d)+e),vec3(0.0),vec3(1.0));
	}

#endif

	//regular Linear -> SRGB conversion
	vec3 a = vec3(0.055);
	color.rgb = mix( (vec3(1.0)+a)*pow(color.rgb,vec3(1.0/2.4))-a , 12.92*color.rgb , lessThan(color.rgb,vec3(0.0031308)));


#ifdef USE_BCS

	color.rgb = mix(vec3(0.0),color.rgb,bcs.x);
	color.rgb = mix(vec3(0.5),color.rgb,bcs.y);
	color.rgb = mix(vec3(dot(vec3(1.0),color.rgb)*0.33333),color.rgb,bcs.z);

#endif

#ifdef USE_COLOR_CORRECTION

	color.r = texture(color_correction,vec2(color.r,0.0)).r;
	color.g = texture(color_correction,vec2(color.g,0.0)).g;
	color.b = texture(color_correction,vec2(color.b,0.0)).b;
#endif


	frag_color=vec4(color.rgb,1.0);
}


