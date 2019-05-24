#version 450

struct S {
    int a;
};

uniform ubuf {
    S s;
};

uniform sampler2D s2d;

layout(location = 0) in vec4 inv;
layout(location = 0) out vec4 outv;

vec4 foo(S s)
{
    vec4 r = s.a * inv;
    ++r;
    if (r.x > 3.0)
        --r;
    else
        r *= 2;

    return r;
}

void main()
{
    outv = foo(s);
    outv += texture(s2d, vec2(0.5));

    switch (s.a) {
    case 10:
        ++outv;
        break;
    case 20:
        outv = 2 * outv;
        ++outv;
        break;
    default:
        --outv;
        break;
    }

    for (int i = 0; i < 10; ++i)
        outv *= 3.0;

    outv.x < 10.0 ?
        outv = sin(outv) :
        outv = cos(outv);
}