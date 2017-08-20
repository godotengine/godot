[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;

void main() {

	uv_interp = uv_in;
#ifdef H_FLIP
	uv_interp.x = 1.0 - uv_interp.x;
#endif
#ifdef V_FLIP
	uv_interp.y = 1.0 - uv_interp.y;
#endif

	gl_Position = vertex_attrib;

}

[fragment]

in vec2 uv_interp;

// seems without atleast one settable uniform we can't compile our shader because our enum won't be setup...
uniform float i_seem_to_need_one_of_these;

uniform sampler2D color_y; //texunit:0
uniform sampler2D color_cbcr; //texunit:1

layout(location = 0) out vec4 frag_color;

void main() {
	vec3 yuv;
	vec3 rgb;

	yuv.x = texture(color_y, uv_interp).r;
	yuv.yz = texture(color_cbcr, uv_interp).rg - vec2(0.5, 0.5);

	// BT.601, which is the standard for SDTV is provided as a reference
	/*
	rgb = mat3(
		vec3( 1.00000, 1.00000, 1.00000),
		vec3( 0.00000,-0.34413, 1.77200),
		vec3( 1.40200,-0.71414, 0.00000)
	) * yuv;
	*/

	// Using BT.709 which is the standard for HDTV
	rgb = mat3(
		vec3( 1.00000, 1.00000, 1.00000),
		vec3( 0.00000,-0.18732, 1.85560),
		vec3( 1.57481,-0.46813, 0.00000)
	) * yuv;

	frag_color = vec4(rgb, 1.0);
}
