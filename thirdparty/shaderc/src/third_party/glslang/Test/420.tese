#version 420 core

const mat2x2 a = mat2( vec2( 1.0, 0.0 ), vec2( 0.0, 1.0 ) );
mat2x2 b = { vec2( 1.0, 0.0 ), vec2( 0.0, 1.0 ) };
const mat2x2 c = { { 1.0, 0.0, }, { 0.0, 1.0 } };

float a2[2] = { 3.4, 4.2, 5.0 }; // illegal
vec2 b2 = { 1.0, 2.0, 3.0 }; // illegal
mat3x3 c2 = { vec3(0.0), vec3(1.0), vec3(2.0), vec3(3.0) }; // illegal
mat2x2 d = { 1.0, 0.0, 0.0, 1.0 }; // illegal, can't flatten nesting

struct {
    float a;
    int b;
} e = { 1.2, 2, };

struct {
    float a;
    int b;
} e2 = { 1, 3 }; // legal, first initializer is converted

struct {
    float a;
    int b;
} e3 = { 1.2, 2, 3 }; // illegal

int a3 = true; // illegal
vec4 b3[2] = { vec4(0.0), 1.0 }; // illegal
vec4 b4[2] = vec4[2](vec4(0.0), mat2x2(1.0)); // illegal
mat4x2 c3 = { vec3(0.0), vec3(1.0) }; // illegal

struct S1 {
    vec4 a;
    vec4 b;
};

struct {
    float s;
    float t;
} d2[] = { S1(vec4(0.0), vec4(1.1)) }; // illegal

float b5[] = { 3.4, 4.2, 5.0, 5.2, 1.1 };

struct S3 {
    float f;
    mat2x3 m23;
};

struct S4 {
    uvec2 uv2;
    S3 s[2];
};

struct Single1 { int f; };
Single1 single1 = { 10 };

struct Single2 { uvec2 v; };
Single2 single2 = { { 1, 2 } };

struct Single3 { Single1 s1; };
Single3 single3 = { { 3 } };

struct Single4 { Single2 s1; };
Single4 single4 = { { { 4u, 5u } } };

const S4 constructed = S4(uvec2(1, 2), 
                          S3[2](S3(3.0, mat2x3(4.0)), 
                                S3(5.0, mat2x3(6.0))));

const S4 curlybad1 = { {1, 2},
                       { {3,   {4.0, 0, 0.0}, {0.0, 4.0, 0.0 } },       // ERROR, the mat2x3 isn't isolated
                         {5.0, {6, 0.0, 0.0}, {0.0, 6.0, 0.0 } } } }; 

const S4 curlyInit = { {1, 2},
                       { {3,   { {4.0, 0, 0.0}, {0.0, 4.0, 0.0 } } },
                         {5.0, { {6, 0.0, 0.0}, {0.0, 6.0, 0.0 } } } } }; 

float vc1, vc2, vc3;
vec3 av3 = vec3(vc1, vc2, vc3);
vec3 bv3 = { vc1, vc2, vc3 };

void main()
{
    memoryBarrier();

    if (constructed == curlybad1)
        ;
    if (constructed == curlyInit)
        ;
}
