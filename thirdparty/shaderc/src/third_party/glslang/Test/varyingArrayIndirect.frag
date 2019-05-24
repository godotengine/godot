#version 130
uniform sampler2D texSampler2D;
varying vec4 color;
varying float alpha;

varying vec4 gl_TexCoord[6];

varying  vec4 userIn[2];

uniform int a, b;

void main()
{
	vec4 texColor = texture2D(texSampler2D, vec2(userIn[b] + gl_TexCoord[a] + gl_TexCoord[5]));

	texColor += color;

	texColor.a = alpha;

    gl_FragColor = gl_TexCoord[0] + gl_TexCoord[b] + texColor + userIn[a];
}
