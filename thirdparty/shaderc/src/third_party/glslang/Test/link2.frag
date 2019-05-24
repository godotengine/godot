#version 130

uniform vec4 uv4;
uniform vec2 glass;

const int ci = 8;

vec4 d = ci * uv4;

in vec3 iv3;
flat in vec4 cup;

vec4 e = ci * d;

ivec2 foo()
{
    return ivec2(2);
}

vec4 f = e * e;

const vec3 cv3 = vec3(43.0, 0.34, 9.9);
const vec3 cv3e = vec3(43.0, 0.34, 2.9);
uniform mat2 um2 = mat2(4.0);
uniform mat2 um2n;
uniform mat2 um2e = mat2(3.0);
struct S {
    int a;
    float b;
};
uniform S s = S(82, 3.9);
uniform S sn = S(82, 3.9);
uniform S se = S(81, 3.9);

#extension GL_OES_texture_3D : enable
#extension GL_OES_standard_derivatives : enable
