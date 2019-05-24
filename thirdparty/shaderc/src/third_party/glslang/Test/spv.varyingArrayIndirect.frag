#version 140
uniform sampler2D texSampler2D;
in vec4 color;
in float alpha;

in vec4 TexCoord[6];

in  vec4 userIn[2];

flat in int a, b;

void main()
{
	vec4 texColor = texture(texSampler2D, vec2(userIn[b] + TexCoord[a] + TexCoord[5]));

	texColor += color;

	texColor.a = alpha;

    gl_FragColor = TexCoord[0] + TexCoord[b] + texColor + userIn[a];
}
