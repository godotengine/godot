#version 140

precision mediump int;

uniform sampler2D samp2D;
in mediump vec2 coord;

struct lunarStruct1 {
    int i;
    float f;
};

struct lunarStruct2 {
    int i;
    float f;
	lunarStruct1 s1_1;
};

struct lunarStruct3 {
	lunarStruct2 s2_1;
    int i;
    float f;
	lunarStruct1 s1_1;
};


lunarStruct1 foo;
lunarStruct2 foo2;
lunarStruct3 foo3;

void main()
{
	lunarStruct2 locals2;

	if (foo3.s2_1.i > 0)
		locals2 = foo3.s2_1;
	else
		locals2 = foo2;

	gl_FragColor =  locals2.s1_1.f * texture(samp2D, coord);
}
