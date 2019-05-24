#version 130
uniform sampler2D texSampler2D;
uniform vec3 inColor;
uniform vec4 color[6];
uniform float alpha[16];

void main()
{
	vec4 texColor = color[1] + color[1];

	texColor.xyz += inColor;

	texColor.a += alpha[12];

    gl_FragColor = texColor;
}
