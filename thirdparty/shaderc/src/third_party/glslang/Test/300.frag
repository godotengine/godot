#version 300 es
void nodef1(float f); // ERROR, no default precision
uniform sampler2D s2D;
uniform lowp sampler3D s3D;
uniform samplerCube sCube;
uniform lowp samplerCubeShadow sCubeShadow;
uniform lowp sampler2DShadow s2DShadow;
uniform lowp sampler2DArray s2DArray;
uniform lowp sampler2DArrayShadow s2DArrayShadow;

uniform lowp isampler2D is2D;
uniform lowp isampler3D is3D;
uniform lowp isamplerCube isCube;
uniform lowp isampler2DArray is2DArray;

uniform lowp usampler2D us2D;
uniform lowp usampler3D us3D;
uniform lowp usamplerCube usCube;
uniform lowp usampler2DArray us2DArray;
precision lowp float;
in float c1D;
in vec2  c2D;
in vec3  c3D;
smooth vec4  c4D;

flat in int   ic1D;
flat in ivec2 ic2D;
flat in ivec3 ic3D;
flat in ivec4 ic4D;
noperspective in vec4 badv; // ERROR
in sampler2D bads;          // ERROR
precision lowp uint;        // ERROR

struct s {
    int i;
    sampler2D s;
};

in s badout;               // ERROR, can't contain a sampler
                           // ERROR, can't have int in struct without flat
struct S2 {
    vec3 c;
    float f;
};

in S2 s2;

out vec3 sc;
out float sf;

uniform sampler2D arrayedSampler[5];

void main()
{
    float f;
    vec4 v;
    v = texture(s2D, c2D);
    v = textureProj(s3D, c4D);
    v = textureLod(s2DArray, c3D, 1.2);
    f = textureOffset(s2DShadow, c3D, ic2D, c1D);  // ERROR, offset argument not constant
    v = texelFetch(s3D, ic3D, ic1D);
    v = texelFetchOffset(arrayedSampler[2], ic2D, 4, ic2D);   // ERROR, offset argument not constant
    f = textureLodOffset(s2DShadow, c3D, c1D, ic2D);
    v = textureProjLodOffset(s2D, c3D, c1D, ic2D);
    v = textureGrad(sCube, c3D, c3D, c3D);
    f = textureGradOffset(s2DArrayShadow, c4D, c2D, c2D, ic2D);
    v = textureProjGrad(s3D, c4D, c3D, c3D);
    v = textureProjGradOffset(s2D, c3D, c2D, c2D, ic2D);
    v = texture(arrayedSampler[ic1D], c2D);                 // ERROR

    ivec4 iv;
    iv = texture(is2D, c2D);
    iv = textureProjOffset(is2D, c4D, ic2D);
    iv = textureProjLod(is2D, c3D, c1D);
    iv = textureProjGrad(is2D, c3D, c2D, c2D);
    iv = texture(is3D, c3D, 4.2);
    iv = textureLod(isCube, c3D, c1D);
    iv = texelFetch(is2DArray, ic3D, ic1D);

    iv.xy = textureSize(sCubeShadow, 2);

    float precise;
    double boo;       // ERROR
    dvec2 boo2;       // ERROR
    dvec3 boo3;       // ERROR
    dvec4 boo4;       // ERROR

    f += gl_FragCoord.y;
    gl_FragDepth = f;

    sc = s2.c;
    sf = s2.f;

    sinh(c1D) +
    cosh(c1D) * tanh(c2D);
    asinh(c4D) + acosh(c4D);
    atanh(c3D);
}

uniform multi {
    int[2] a[3];      // ERROR
    int[2][3] b;      // ERROR
    int c[2][3];      // ERROR
} multiInst[2][3];    // ERROR

out vec4 colors[4];

void foo()
{
    colors[2] = c4D;
    colors[ic1D] = c4D;  // ERROR
}

uniform s st1;
uniform s st2;

void foo13(s inSt2)
{
    if (st1 == st2);  // ERROR
    if (st1 != st2);  // ERROR
    st1.s == st2.s;   // ERROR
    inSt2 = st1;      // ERROR
    inSt2 == st1;     // ERROR
}

void foo23()
{
    textureOffset(s2DShadow, c3D, ivec2(-8, 7), c1D);
    textureOffset(s2DShadow, c3D, ivec2(-9, 8), c1D);
}

void foo324(void)
{
    float p = pow(3.2, 4.6);
    p += sin(0.4);
    p += distance(vec2(10.0, 11.0), vec2(13.0, 15.0)); // 5
    p += dot(vec3(2,3,5), vec3(-2,-1,4));              // 13
    vec3 c3 = cross(vec3(3,-3,1), vec3(4,9,2));        // (-15, -2, 39)
    c3 += faceforward(vec3(1,2,3), vec3(2,3,5), vec3(-2,-1,4));     // (-1,-2,-3)
    c3 += faceforward(vec3(1,2,3), vec3(-2,-3,-5), vec3(-2,-1,4));  // (1,2,3)
    vec2 c2 = reflect(vec2(1,3), vec2(0,1));           // (1,-3)
    c2 += refract(vec2(1,3), vec2(0,1), 1.0);          // (1,-3)
    c2 += refract(vec2(1,3), vec2(0,1), 3.0);
    c2 += refract(vec2(1,0.1), vec2(0,1), 5.0);        // (0,0)
    mat3x2 m32 = outerProduct(vec2(2,3), vec3(5,7,11));// rows: (10, 14, 22), (15, 21, 33)
}

uniform mediump;       // ERROR

layout(early_fragment_tests) in;  // ERROR

#ifndef GL_FRAGMENT_PRECISION_HIGH
#error missing GL_FRAGMENT_PRECISION_HIGH
#endif

invariant in;                // ERROR
invariant in vec4;           // ERROR
invariant in vec4 fooinv;    // ERROR

float imageBuffer;    // ERROR, reserved
float uimage2DRect;   // ERROR, reserved
