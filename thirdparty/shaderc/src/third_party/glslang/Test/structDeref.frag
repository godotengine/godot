#version 130

uniform sampler2D sampler;
varying vec2 coord;

struct s0 {
    int i;
};

struct s00 {
    s0 s0_0;
};

struct s1 {
    int i;
    float f;
    s0 s0_1;
};

struct s2 {
    int i;
    float f;
    s1 s1_1;
};

struct s3 {
    s2[12] s2_1;
    int i;
    float f;
    s1 s1_1;
};


uniform s0 foo0;
uniform s1 foo1;
uniform s2 foo2;
uniform s3 foo3;

uniform s00 foo00;

void main()
{
    s0 locals0;
    s2 locals2;
    s00 locals00;

    float[6] fArray;

    s1[10] locals1Array;

    if (foo3.s2_1[9].i > 0) {
        locals2.f = 1.0;
        locals2.s1_1 = s1(0, 1.0, s0(0));
        fArray = float[6]( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        locals1Array[6] = foo1;
        locals0 = s0(0);
        locals00 = s00(s0(0));
    } else {
        locals2.f = coord.x;
        locals2.s1_1 = s1(1, coord.y, foo0);
        fArray = float[6]( 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        locals1Array[6] = locals2.s1_1;
        locals0 = foo1.s0_1;
        locals00 = foo00;
    }

    if (locals0.i > 5)
        locals0 = locals00.s0_0;

    gl_FragColor = (float(locals0.i) + locals1Array[6].f + fArray[3] + locals2.s1_1.f) * texture2D(sampler, coord);
}
