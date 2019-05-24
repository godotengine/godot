#version 130
uniform sampler2D sampler;
varying vec2 coord;

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
    lunarStruct2 s2_1[3];
    int i;
    float f;
    lunarStruct1 s1_1;
};


uniform lunarStruct1 foo;
uniform lunarStruct2 foo2[5];
uniform lunarStruct3 foo3;
uniform int Count;

void main()
{
    float scale;
    int iLocal = Count;

    if (foo3.s2_1[1].i > 0)
        scale = foo2[foo3.s2_1[foo.i].i + 2 + ++iLocal].s1_1.f;
    else
        scale = foo3.s2_1[0].s1_1.f;

    //for (int i = 0; i < iLocal; ++i) {
    //	scale += foo2[i].f;
    //}

    gl_FragColor =  scale * texture2D(sampler, coord);

    vec2[3] constructed = vec2[3](coord, vec2(scale), vec2(1.0, 2.0));
    gl_FragColor += vec4(constructed[foo.i], constructed[foo.i]);
}

