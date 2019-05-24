#version 450

uniform sampler2D samp2D;
in mediump vec2 coord;

in vec4 u, w;
out vec4 color;

struct s1 {
    int i;
    float f;
};

struct s2 {
    int i;
    float f;
	s1 s1_1;
};

layout(std140) uniform ub1 { s2 foo2a; } uName1;
layout(std430) buffer  ub2 { s2 foo2b; } uName2;

void main()
{
    vec4 v;
    s1 a[3], b[3];
    a = s1[3](s1(int(u.x), u.y), s1(int(u.z), u.w), s1(14, 14.0));
    b = s1[3](s1(17, 17.0), s1(int(w.x), w.y), s1(int(w.z), w.w));

    if (uName1.foo2a == uName2.foo2b)
        v = texture(samp2D, coord);
    else
        v = texture(samp2D, 2.0*coord);

    if (u == v)
        v *= 3.0;

    if (u != v)
        v *= 4.0;

    if (coord == v.yw)
        v *= 5.0;

    if (a == b)
        v *= 6.0;

    if (a != b)
        v *= 7.0;

	color =  v;
}
