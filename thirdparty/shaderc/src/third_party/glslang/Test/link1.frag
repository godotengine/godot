#version 130

uniform vec4 uv4;
uniform vec3 glass;

const int ci = 8;

vec4 a = ci * uv4;

in vec3 iv3;
in vec4 cup;

void main()
{
}

vec4 b = ci * a;

ivec2 foo(mat2 m)
{
    return ivec2(m[0]);
}

vec4 c = b * b;

const vec3 cv3 = vec3(43.0, 0.34, 9.9);
const vec3 cv3n = vec3(43.0, 0.34, 9.9);
const vec3 cv3e = vec3(43.0, 0.34, 9.9);
uniform mat2 um2 = mat2(4.0);
uniform mat2 um2n = mat2(4.0);
uniform mat2 um2e = mat2(4.0);
struct S {
    int a;
    float b;
};
uniform S s = S(82, 3.9);
uniform S sn;
uniform S se = S(82, 3.9);
