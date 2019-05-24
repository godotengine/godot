#version 300 es

uniform mat4x3 m43;
uniform mat3x3 m33;
uniform mat4x4 m44;

in vec3 v3;
varying vec2 v2;               // ERROR, varying reserved
in vec4 bad[10];               // ERROR, no arrayed inputs
highp in vec4 badorder;        // ERROR, incorrect qualifier order
out invariant vec4 badorder2;  // ERROR, incorrect qualifier order
in centroid vec4 badorder4;    // ERROR, incorrect qualifier order
out flat vec4 badorder3;       // ERROR, incorrect qualifier order
void bar(in const float a);    // ERROR, incorrect qualifier order
void bar2(highp in float b);   // ERROR, incorrect qualifier order
smooth flat out vec4 rep;      // ERROR, replicating interpolation qualification
centroid sample out vec4 rep2; // ERROR, replicating auxiliary qualification
in uniform vec4 rep3;          // ERROR, replicating storage qualification

struct S {
    vec3 c;
    float f;
};

out S s;

void main()
{
    int id = gl_VertexID + gl_InstanceID;

    int c0 = gl_MaxVertexAttribs;
    int c1 = gl_MaxVertexUniformVectors;
    int c2 = gl_MaxVertexOutputVectors;
    int c3 = gl_MaxFragmentInputVectors;
    int c4 = gl_MaxVertexTextureImageUnits;
    int c5 = gl_MaxCombinedTextureImageUnits;
    int c6 = gl_MaxTextureImageUnits;
    int c7 = gl_MaxFragmentUniformVectors;
    int c8 = gl_MaxDrawBuffers;
    int c9 = gl_MinProgramTexelOffset;
    int c10 = gl_MaxProgramTexelOffset;

    mat3x4 tm = transpose(m43);
    highp float dm = determinant(m44);
    mat3x3 im = inverse(m33);

    mat3x2 op = outerProduct(v2, v3);

    gl_Position = m44[2];
    gl_PointSize = v2.y;

     s.c = v3;
     s.f = dm;

#ifdef GL_ES
#error GL_ES is set
#else
#error GL_ES is not set
#endif
}

float badsize[];    // ERROR
float[] badsize2;   // ERROR
uniform ub {
    int a[];        // ERROR
} ubInst[];         // ERROR
void foo(int a[]);  // ERROR
float okayA[] = float[](3.0f, 4.0F);  // Okay

out vec3 newV;
void newVFun()
{
    newV = v3;
}

invariant newV;  // ERROR, variable already used
in vec4 invIn;
invariant invIn; // ERROR, in v300
out S s2;
invariant s2;
invariant out S s3;
flat out int;

uniform ub2 {
    float f;
} a;

uniform ub2 {  // ERROR redeclaration of block name (same instance name)
    float g;
} a;

uniform ub2 {  // ERROR redeclaration of block name (different instance name)
    float f;
} c;

uniform ub2 {  // ERROR redeclaration of block name (no instance name)
    float f123;
};

uniform ub3 {
    bool b23;
};

uniform ub3 {  // ERROR redeclaration of block name (no instance name in first or declared)
    bool b234;
};

precision lowp sampler3D;
precision lowp sampler2DShadow;
precision lowp sampler2DArrayShadow;

uniform sampler2D s2D;
uniform sampler3D s3D;
uniform sampler2DShadow s2DS;
uniform sampler2DArrayShadow s2DAS;
in vec2 c2D;

void foo23()
{
    ivec2 x1 = textureSize(s2D, 2);
    textureSize(s2D);        // ERROR, no lod
    ivec3 x3 = textureSize(s2DAS, -1);
    textureSize(s2DAS);      // ERROR, no lod
    vec4 x4 = texture(s2D, c2D);
    texture(s2D, c2D, 0.2);  // ERROR, bias
    vec4 x5 = textureProjOffset(s3D, vec4(0.2), ivec3(1));
    textureProjOffset(s3D, vec4(0.2), ivec3(1), .03);  // ERROR, bias
    float x6 = textureProjGradOffset(s2DS, invIn, vec2(4.2), vec2(5.3), ivec2(1));
}

int fgfg(float f, mediump int i);
int fgfg(float f, highp int i);   // ERROR, precision qualifier difference

int fgfgh(float f, const in mediump int i);
int fgfgh(float f, in mediump int i);   // ERROR, precision qualifier difference

void foo2349()
{
    float[] x = float[] (1.0, 2.0, 3.0);
	float[] y = x;
    float[3] z = x;
    float[3] w;
    w = y;
}

int[] foo213234();        // ERROR
int foo234234(float[]);   // ERROR
int foo234235(vec2[] v);  // ERROR
precision highp float[2]; // ERROR

int fffg(float f);
int fffg(float f);

int gggf(float f);
int gggf(float f) { return 2; }
int gggf(float f);

int agggf(float f) { return 2; }
int agggf(float f);

out struct Ssss { float f; } ssss;

uniform Bblock {
   int a;
} Binst;
int Bfoo;

layout(std140) Binst;    // ERROR
layout(std140) Bblock;   // ERROR
layout(std140) Bfoo;     // ERROR

layout(std430) uniform B430 { int a; } B430i;     // ERROR

struct SNA {
    int a[];             // ERROR
};

void fooDeeparray()
{
    float[] x = float[] (1.0, 2.0, 3.0),
            y = float[] (1.0, 2.0, 3.0, 4.0);
    float xp[3], yp[4];
    xp = x;
    yp = y;
    xp = y; // ERROR, wrong size
    yp = x; // ERROR, wrong size
}

layout(num_views = 2) in; // ERROR, no extension

void mwErr()
{
    gl_ViewID_OVR;   // ERROR, no extension
}

#extension GL_OVR_multiview : enable

layout(num_views = 2) uniform float mwUniform; // ERROR, must be global
layout(num_views = 2) in; // OK

void mwOk()
{
    gl_ViewID_OVR;
}
