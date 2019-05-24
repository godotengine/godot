#version 310 es

precision mediump float;

in vec4 pos;
in vec3 color;

layout(location = 7) out vec3 c;
layout(LocatioN = 3) out vec4 p[2];

struct S {
    vec3 c;
    float f;
};

in S s;

void main()
{
    c = color + s.c;
    p[1] = pos * s.f;
}
