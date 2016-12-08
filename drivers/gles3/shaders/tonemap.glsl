[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;



void main() {

	gl_Position = vertex_attrib;
	uv_interp = uv_in;

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

layout(location = 0) out vec4 frag_color;


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
	glow+=textureLod(source_glow,uv_interp,1.0).rgb;
#endif

#ifdef USE_GLOW_LEVEL2
	glow+=textureLod(source_glow,uv_interp,2.0).rgb;
#endif

#ifdef USE_GLOW_LEVEL3
	glow+=textureLod(source_glow,uv_interp,3.0).rgb;
#endif

#ifdef USE_GLOW_LEVEL4
	glow+=textureLod(source_glow,uv_interp,4.0).rgb;
#endif

#ifdef USE_GLOW_LEVEL5
	glow+=textureLod(source_glow,uv_interp,5.0).rgb;
#endif

#ifdef USE_GLOW_LEVEL6
	glow+=textureLod(source_glow,uv_interp,6.0).rgb;
#endif

#ifdef USE_GLOW_LEVEL7
	glow+=textureLod(source_glow,uv_interp,7.0).rgb;
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




	frag_color=vec4(color.rgb,1.0);
}


