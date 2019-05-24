#version 120

in vec4 i;                // ERROR
out vec4 o;               // ERROR

attribute vec2 attv2;
attribute vec4 attv4;
uniform sampler2D s2D;
invariant varying vec2 centTexCoord;
invariant gl_Position;
centroid gl_Position;     // ERROR
centroid centroid foo;    // ERROR
invariant gl_Position, gl_PointSize;

void main()
{
    centTexCoord = attv2; 
    gl_Position = attv4;

    gl_ClipVertex = attv4;
    gl_ClipDistance[1] = 0.2;  // ERROR

    vec3[12] a;
    vec4[a.length()] b;
    gl_Position = b[b.length()-1];

    float f[];
    int a1 = f.length();  // ERROR
    float f[7];
    int aa = f.length();
    int a2 = f.length;    // ERROR
    int a3 = f.length(a); // ERROR
    int a4 = f.flizbit;   // ERROR
    int a4 = f.flizbit(); // ERROR
    float md[2][4];       // ERROR
    float[2] md2[4];      // ERROR
    float[2][4] md3;      // ERROR
    float md5, md6[2][3]; // ERROR
    float[2] md4, md7[4]; // ERROR
    float md9[2][3] = float[2][3](1, 2, 3, 4, 5, 6);  // ERROR
    float md10, md11[2][3] = float[2][3](1, 2, 3, 4, 5, 6);  // ERROR

    gl_PointSize = 3.8;
}

uniform float initted = 3.4;   // okay

const float concall = sin(0.3);

int[2][3] foo(                 // ERROR
              float[2][3] a,   // ERROR
              float[2] b[3],   // ERROR
              float c[2][3]);  // ERROR

int overloadA(in float f);
int overloadA(out float f);        // ERROR, different qualifiers
float overloadA(float);            // ERROR, different return value for same signature
float overloadA(out float f, int);
float overloadA(int i);

void overloadB(float, const in float) { }

vec2 overloadC(int, int);
vec2 overloadC(const in int, float);
vec2 overloadC(float, int);
vec2 overloadC(vec2, vec2);

vec3 overloadD(int, float);
vec3 overloadD(float, in int);

vec3 overloadE(float[2]);
vec3 overloadE(mat2 m);
vec3 overloadE(vec2 v);

vec3 overloadF(int);
vec3 overloadF(float);

void foo()
{
    float f;
    int i;

    overloadB(f, f);
    overloadB(f, 2);
    overloadB(1, i);

    overloadC(1);    // ERROR
    overloadC(1, i);
    overloadC(vec2(1), vec2(2));
    overloadC(f, 3.0);           // ERROR, no way
    overloadC(ivec2(1), vec2(2));

    overloadD(i, f);
    overloadD(f, i);
    overloadD(i, i);   // ERROR, ambiguous

    int overloadB;     // hiding
    overloadB(1, i);   // ERROR

    sin(1);
    texture2D(s2D, ivec2(0));
    clamp(attv4, 0, 1);
    clamp(ivec4(attv4), 0, 1);

    int a[2];
    overloadC(a, 3); // ERROR
    overloadE(a);    // ERROR
    overloadE(3.3);  // ERROR
    overloadE(vec2(3.3));
    overloadE(mat2(0.5));
    overloadE(ivec4(1)); // ERROR
    overloadE(ivec2(1));

    float b[2];
    overloadE(b);
    
    overloadF(1, 1); // ERROR
    overloadF(1);
}

varying vec4 gl_TexCoord[35]; // ERROR, size too big

// tests for output conversions
void outFun(in float, out ivec2, in int, out float);
int outFunRet(in float, out int, const in int, out ivec4);
ivec2 outFunRet(in float, out ivec4, in int, out ivec4);

void foo2()
{
    vec2 v2;
    vec4 v4;
    float f;
    int i;

    outFun(i, v2, i, f);
    outFunRet(i, f, i, v4);
    float ret = outFunRet(i, f, i, v4);
    vec2 ret2 = outFunRet(i, v4, i, v4);
    bool b = any(lessThan(v4, attv4));  // tests aggregate arg to unary built-in 
}

void noise()
{
    float f1 = noise1(1.0);
    vec2 f2 = noise2(vec2(1.0));
    vec3 f3 = noise3(vec3(1.0));
    vec4 f4 = noise4(vec4(1.0));
}

// version 130 features

uniform int c;

attribute ivec2 x;
attribute vec2 v2a;
attribute float c1D;
attribute vec2  c2D;
attribute vec3  c3D;

uniform vec4 v4;

void foo213()
{
    float f = 3;
    switch (c) {         // ERRORs...
    case 1:              
        f = sin(f);
        break;
    case 2:
        f = f * f;
    default:
        f = 3.0;
    }

    int i;          
    i << 3 | 0x8A >> 1 & 0xFF;      // ERRORs...

    vec3 modfOut, modfIn;
    vec3 v11 = modf(modfIn, modfOut); // ERRORS...
    float t = trunc(f);
    vec2 v12 = round(v2a);
    vec2 v13 = roundEven(v2a);
    bvec2 b10 = isnan(v2a);
    bvec4 b11 = isinf(v4);

    sinh(c1D) +                      // ERRORS...
    cosh(c1D) * tanh(c2D);
    asinh(c4D) + acosh(c4D);
    atanh(c3D);

    int id = gl_VertexID;            // ERROR
    gl_ClipDistance[1] = 0.3;        // ERROR
}

int gl_ModelViewMatrix[] = 0;

// token pasting (ERRORS...)

#define mac abc##def
int mac;

#define macr(A,B) A ## B
int macr(qrs,tuv);
