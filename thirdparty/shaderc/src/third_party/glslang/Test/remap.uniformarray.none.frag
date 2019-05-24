#version 140

uniform sampler2D texSampler2D;
in vec3 inColor;
in vec4 color[6];
in float alpha[16];

void main()
{
	vec4 texColor = color[1] + color[1];

	texColor.xyz += inColor;

	texColor.a += alpha[12];

    gl_FragColor = texColor;
}
