#version 140
uniform sampler2D samp2D;
in vec2 coord;

struct lunarStruct1 {
    int i;
	float f[4];
	vec4 color[5];
};

struct lunarStruct2 {
    int i[5];
    float f;
	lunarStruct1 s1_1[7];
};

lunarStruct1 foo;
lunarStruct2 foo2[5];

void main()
{
	float scale = 0.0;

	if (foo2[3].i[4] > 0)
		scale = foo2[3].s1_1[2].color[3].x;
	else
		scale = foo2[3].s1_1[2].f[3];

	gl_FragColor =  scale * texture(samp2D, coord);
}

