#version 130
uniform sampler2D texSampler2D;
varying vec4 color;
varying float alpha;

varying vec4 gl_TexCoord[6];

varying vec4 foo[3];

void main()
{
	vec4 texColor = texture2D(texSampler2D, vec2(gl_TexCoord[4] + gl_TexCoord[5]));

	texColor += color;

	texColor.a = alpha;

    gl_FragColor = foo[1] + gl_TexCoord[0] + gl_TexCoord[4] + texColor;
}
