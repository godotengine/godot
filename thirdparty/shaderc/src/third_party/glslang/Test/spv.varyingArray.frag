#version 140
uniform sampler2D texSampler2D;
in  vec4 color;
in  float alpha;

in  vec4 TexCoord[6];

in  vec4 foo[3];

void main()
{
	vec4 texColor = texture(texSampler2D, vec2(TexCoord[4] + TexCoord[5]));

	texColor += color;

	texColor.a = alpha;

    gl_FragColor = foo[1] + TexCoord[0] + TexCoord[4] + texColor;
}
