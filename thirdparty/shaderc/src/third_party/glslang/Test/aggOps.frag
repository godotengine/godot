#version 130

uniform sampler2D sampler;
varying mediump vec2 coord;

varying vec4 u, w;

struct s1 {
    int i;
    float f;
};

struct s2 {
    int i;
    float f;
	s1 s1_1;
};

uniform s1 foo1;
uniform s2 foo2a;
uniform s2 foo2b;

void main()
{
    vec4 v;
    s1 a[3], b[3];
    a = s1[3](s1(int(u.x), u.y), s1(int(u.z), u.w), s1(14, 14.0));
    b = s1[3](s1(17, 17.0), s1(int(w.x), w.y), s1(int(w.z), w.w));

    if (foo2a == foo2b)
        v = texture2D(sampler, coord);
    else
        v = texture2D(sampler, 2.0*coord);

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

	gl_FragColor =  v;
}
