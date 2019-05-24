#version 420 core
#version 420 core
varying vec2 v2;               // ERROR, varying reserved
in vec4 bad[10];
highp in vec4 badorder;
out invariant vec4 badorder2;
in centroid vec4 badorder4;    // ERROR, no centroid input to vertex stage
out flat vec4 badorder3;
void bar(in const float a);
void bar2(highp in float b);
smooth flat out vec4 rep;      // ERROR, replicating interpolation qualification
centroid sample out vec4 rep2; // ERROR, replicating auxiliary qualification
in uniform vec4 rep3;          // ERROR, replicating storage qualification

int anonconst;
const int aconst = 5;
const int a = aconst;
const int b = anonconst;       // ERROR at global scope

const int foo()                // ERROR, no const functions
{
    const int a = aconst;
    const int b = anonconst;
    const int c = a;          // still compile-time const
    const int d = b;          // not a compile-time const
    float x[c];               // okay
    float y[d];               // ERROR

    return b;
}

void main()
{
    int i;
    if (i == 3)
        int j = i;
    else
        int k = j;              // ERROR, j is undeclared
    int m = k;                  // ERROR, k is undeclared
    int n = j;                  // ERROR, j is undeclared

    while (true)
        int jj;
    int kk = jj;                // ERROR, jj is undeclared
}

const float cx = 4.20;
const float dx = 4.20;

void bar(in highp volatile vec4 v)
{
    int s;
    s.x;       // okay
    s.y;       // ERROR
    if (bad[0].x == cx.x)
        ;
    if (cx.x == dx.x)
        badorder3 = bad[0];

    float f;
    vec3 smeared = f.xxx;
    f.xxxxx;   // ERROR
    f.xxy;     // ERROR
}

layout(binding = 3) uniform;  // ERROR
layout(binding = 3) uniform boundblock { int aoeu; } boundInst;
layout(binding = 7) uniform anonblock { int aoeu; } ;
layout(location = 1) in;      // ERROR
layout(binding = 1) in inblock { int aoeua; };       // ERROR
layout(binding = 100000) uniform anonblock2 { int aooeu; } ;
layout(binding = 4) uniform sampler2D sampb1;
layout(binding = 5) uniform sampler2D sampb2[10];
layout(binding = 80) uniform sampler2D sampb3; // ERROR, binding too big
layout(binding = 31) uniform sampler2D sampb4;
layout(binding = 79) uniform sampler2D sampb5[2]; // ERROR, binding too big

int fgfg(float f, mediump int i);
int fgfg(float f, highp int i);

out gl_PerVertex {
    float gl_ClipDistance[4];
};

patch in vec4 patchIn;              // ERROR
patch out vec4 patchOut;            // ERROR

void bar23444()
{
    mat4x3 m43;  \
    float a1 = m43[3].y;
    vec3 v3;
    int a2 = m43.length();
    a2 += m43[1].length();
    a2 += v3.length();
    const float b = 2 * a1;
    int a = gl_MinProgramTexelOffset + gl_MaxProgramTexelOffset;
}

const int comma0 = (2, 3);  // ERROR
int comma1[(2, 3)];   // ERROR

layout(r32i) uniform iimage2D iimg2D;
layout(rgba32i) uniform iimage2D iimg2Drgba;
layout(rgba32f) uniform image2D img2Drgba;
layout(r32ui) uniform uimage2D uimg2D;
uniform image2DMS img2DMS; // ERROR image variables not declared writeonly must have format layout qualifier
uniform writeonly image2DMS img2DMSWO;
void qux()
{
    int i = aoeu;
    imageAtomicCompSwap(iimg2D, ivec2(i,i), i, i);
    imageAtomicAdd(uimg2D, ivec2(i,i), uint(i));
    imageAtomicMin(iimg2Drgba, ivec2(i,i), i); // ERROR iimg2Drgba does not have r32i layout
    imageAtomicMax(img2Drgba, ivec2(i,i), i);  // ERROR img2Drgba is not integer image
    ivec4 pos = imageLoad(iimg2D, ivec2(i,i));
    vec4 col = imageLoad(img2DMS, ivec2(i,i), i);
    imageStore(img2DMSWO, ivec2(i,i), i, vec4(0));
    imageLoad(img2DMSWO, ivec2(i,i), i);       // ERROR, drops writeonly
}

volatile float vol; // ERROR, not an image
readonly int vol2;  // ERROR, not an image

void passr(coherent readonly iimage2D image)
{
}

layout(r32i) coherent readonly uniform iimage2D qualim1;
layout(r32i) coherent volatile readonly uniform iimage2D qualim2;

void passrc()
{
    passr(qualim1);
    passr(qualim2);   // ERROR, drops volatile
    passr(iimg2D);
}

layout(rg8i) uniform uimage2D i1bad;     // ERROR, type mismatch
layout(rgba32i) uniform image2D i2bad;   // ERROR, type mismatch
layout(rgba32f) uniform uimage2D i3bad;  // ERROR, type mismatch
layout(r8_snorm) uniform iimage2D i4bad; // ERROR, type mismatch
layout(rgba32ui) uniform iimage2D i5bad; // ERROR, type mismatch
layout(r8ui) uniform iimage2D i6bad;     // ERROR, type mismatch

uniform offcheck {
    layout(offset = 16) int foo;   // ERROR
} offcheckI;

uniform sampler1D samp1D;
uniform sampler1DShadow samp1Ds;

void qlod()
{
    int levels;

    levels = textureQueryLevels(samp1D);   // ERROR, not until 430
    levels = textureQueryLevels(samp1Ds);  // ERROR, not until 430
}

layout(binding=0) writeonly uniform image1D badArray[];
