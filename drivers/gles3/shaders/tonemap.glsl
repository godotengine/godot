[vertex]


layout(location=0) in highp vec4 vertex_attrib;


void main() {

	gl_Position = vertex_attrib;

}

[fragment]


uniform highp sampler2D source; //texunit:0

uniform float exposure;
uniform float white;

#ifdef USE_AUTO_EXPOSURE

uniform highp sampler2D source_auto_exposure; //texunit:1
uniform highp float auto_exposure_grey;

#endif


layout(location = 0) out vec4 frag_color;


void main() {

	ivec2 coord = ivec2(gl_FragCoord.xy);
	vec3 color = texelFetch(source,coord,0).rgb;


#ifdef USE_AUTO_EXPOSURE

	color/=texelFetch(source_auto_exposure,ivec2(0,0),0).r/auto_exposure_grey;
#endif

	color*=exposure;


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


