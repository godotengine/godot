#version 330

in vec4 inVar;
out vec4 outVar;

const int constInt = 3;

uniform int uniformInt;

void main()
{
    const int a1 = 2;          // okay
    const int a2 = constInt;   // okay
    const int a3 = uniformInt; // error

    vec4 c[constInt];              // okay
    vec4 d[uniformInt];            // error
    vec4 e[constInt + uniformInt]; // error
    vec4 f[uniformInt + constInt]; // error

    vec4 g[int(sin(0.3)) + 1];     // okay
}

const struct S {
    vec3 v3;
    ivec2 iv2;
} s = S(vec3(3.0), ivec2(3, constInt + uniformInt));  // ERROR, non-const y componenent

const struct S2 {
    vec3 v3;
    ivec2 iv2;
    mat2x4 m;
} s2 = S2(vec3(3.0), ivec2(3, constInt), mat2x4(1.0, 2.0, 3.0, inVar.x, 5.0, 6.0, 7.0, 8.0));  // ERROR, non-constant matrix

const float f = 3; // okay, type conversion
