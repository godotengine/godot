#version 400 core

void main()
{
    EmitStreamVertex(1);
    EndStreamPrimitive(0);
    EmitVertex();
    EndPrimitive();
    int id = gl_InvocationID;
}

layout(invocations = 4) in outbn { int a; } bn[]; // ERROR, not on a block
layout(max_vertices = 127) out;
layout(invocations = 4) in;

#extension GL_ARB_separate_shader_objects : enable

in gl_PerVertex {      // testing input arrays with a block redeclaration, see 420.geom for without
    vec4 gl_Position;
    layout(std140, location = 3) patch float gl_PointSize; // ERRORs...
} gl_in[];

void foo()
{
    gl_in.length();  // ERROR
    gl_in[1].gl_Position;
}

in vec4 color[];
in vec4 color2[];
in vec4 colorS[3];
in vec4 colorBad[4];

void foo2()
{
    color.length(); // ERROR
    colorS.length();
}

layout(triangles) in;  // give ERROR just for colorBad

in vec4 color[3];
in vec4 color2[3];
in vec4 colorbad2[2];  // ERROR

void foo3()
{
    gl_in.length();
    color.length();
    color2.length();
    colorS.length();
}

layout(location = 4) in vec4 cva[3];
layout(location = 5) in vec4 cvb[3];
layout(location = 2) in mat3 cmc[3];  // ERROR, collision

patch in vec4 patchIn[];            // ERROR
patch out vec4 patchOut;            // ERROR

in float scalar;  // ERROR, no array

layout(max_vertices = 127, invocations = 4) out;      // ERROR
layout(invocations = 4, max_vertices = 127) in;       // ERROR
layout(max_vertices = 127, invocations = 4) uniform;  // 2 ERRORs

in inblockscalar {
    int a;
} inbls;  // ERROR, not an array

in inblocka {
    int a;
} inbla[17];  // ERROR, wrong array size

void bits()
{
    uvec2 u2;
    u2 = uaddCarry(u2, u2, u2);
    uint u1;
    u1 = usubBorrow(u1, u1, u1);
    uvec4 u4;
    umulExtended(u4, u4, u4, u4);
    ivec4 i4;
    imulExtended(i4, i4, i4, i4);
    int i1;
    i1 = bitfieldExtract(i1, 4, 5);
    uvec3 u3;
    u3 = bitfieldExtract(u3, 4, 5);
    ivec3 i3;
    i3 = bitfieldInsert(i3, i3, 4, 5);
    u1 = bitfieldInsert(u1, u1, 4, 5);
    ivec2 i2;
    i2 = bitfieldReverse(i2);
    u4 = bitfieldReverse(u4);
    i1 = bitCount(i1);
    i3 = bitCount(u3);
    i2 = findLSB(i2);
    i4 = findLSB(u4);
    i1 = findMSB(i1);
    i2 = findMSB(u2);
}

layout(location = 7, index = 1) out vec4 indexedOut;

uniform sampler1D samp1D;
uniform sampler2DShadow samp2Ds;

void qlod()
{
    vec2 lod;
    float pf;
    vec2 pf2;
    vec3 pf3;

    lod = textureQueryLod(samp1D, pf);      // ERROR, only in fragment
    lod = textureQueryLod(samp2Ds, pf2);    // ERROR, only in fragment
}

void doubles()
{
    double doublev;
    dvec2 dvec2v;
    dvec3 dvec3v;
    dvec4 dvec4v;

    bool boolv;
    bvec2 bvec2v;
    bvec3 bvec3v;
    bvec4 bvec4v;

    doublev = sqrt(2.9);
    dvec2v  = sqrt(dvec2(2.7));
    dvec3v  = sqrt(dvec3(2.0));
    dvec4v  = sqrt(dvec4(2.1));

    doublev += inversesqrt(doublev);
    dvec2v  += inversesqrt(dvec2v);
    dvec3v  += inversesqrt(dvec3v);
    dvec4v  += inversesqrt(dvec4v);

    doublev += abs(doublev);
    dvec2v  += abs(dvec2v);
    dvec3v  += abs(dvec3v);
    dvec4v  += abs(dvec4v);

    doublev += sign(doublev);
    dvec2v  += sign(dvec2v);
    dvec3v  += sign(dvec3v);
    dvec4v  += sign(dvec4v);

    doublev += floor(doublev);
    dvec2v  += floor(dvec2v);
    dvec3v  += floor(dvec3v);
    dvec4v  += floor(dvec4v);

    doublev += trunc(doublev);
    dvec2v  += trunc(dvec2v);
    dvec3v  += trunc(dvec3v);
    dvec4v  += trunc(dvec4v);

    doublev += round(doublev);
    dvec2v  += round(dvec2v);
    dvec3v  += round(dvec3v);
    dvec4v  += round(dvec4v);

    doublev += roundEven(doublev);
    dvec2v  += roundEven(dvec2v);
    dvec3v  += roundEven(dvec3v);
    dvec4v  += roundEven(dvec4v);

    doublev += ceil(doublev);
    dvec2v  += ceil(dvec2v);
    dvec3v  += ceil(dvec3v);
    dvec4v  += ceil(dvec4v);

    doublev += fract(doublev);
    dvec2v  += fract(dvec2v);
    dvec3v  += fract(dvec3v);
    dvec4v  += fract(dvec4v);

    doublev += mod(doublev, doublev);
    dvec2v  += mod(dvec2v, doublev);
    dvec3v  += mod(dvec3v, doublev);
    dvec4v  += mod(dvec4v, doublev);
    dvec2v  += mod(dvec2v, dvec2v);
    dvec3v  += mod(dvec3v, dvec3v);
    dvec4v  += mod(dvec4v, dvec4v);

    doublev += modf(doublev, doublev);
    dvec2v  += modf(dvec2v,  dvec2v);
    dvec3v  += modf(dvec3v,  dvec3v);
    dvec4v  += modf(dvec4v,  dvec4v);

    doublev += min(doublev, doublev);
    dvec2v  += min(dvec2v, doublev);
    dvec3v  += min(dvec3v, doublev);
    dvec4v  += min(dvec4v, doublev);
    dvec2v  += min(dvec2v, dvec2v);
    dvec3v  += min(dvec3v, dvec3v);
    dvec4v  += min(dvec4v, dvec4v);

    doublev += max(doublev, doublev);
    dvec2v  += max(dvec2v, doublev);
    dvec3v  += max(dvec3v, doublev);
    dvec4v  += max(dvec4v, doublev);
    dvec2v  += max(dvec2v, dvec2v);
    dvec3v  += max(dvec3v, dvec3v);
    dvec4v  += max(dvec4v, dvec4v);

    doublev += clamp(doublev, doublev, doublev);
    dvec2v  += clamp(dvec2v, doublev, doublev);
    dvec3v  += clamp(dvec3v, doublev, doublev);
    dvec4v  += clamp(dvec4v, doublev, doublev);
    dvec2v  += clamp(dvec2v, dvec2v, dvec2v);
    dvec3v  += clamp(dvec3v, dvec3v, dvec3v);
    dvec4v  += clamp(dvec4v, dvec4v, dvec4v);

    doublev += mix(doublev, doublev, doublev);
    dvec2v  += mix(dvec2v, dvec2v, doublev);
    dvec3v  += mix(dvec3v, dvec3v, doublev);
    dvec4v  += mix(dvec4v, dvec4v, doublev);
    dvec2v  += mix(dvec2v, dvec2v, dvec2v);
    dvec3v  += mix(dvec3v, dvec3v, dvec3v);
    dvec4v  += mix(dvec4v, dvec4v, dvec4v);
    doublev += mix(doublev, doublev, boolv);
    dvec2v  += mix(dvec2v, dvec2v, bvec2v);
    dvec3v  += mix(dvec3v, dvec3v, bvec3v);
    dvec4v  += mix(dvec4v, dvec4v, bvec4v);

    doublev += step(doublev, doublev);
    dvec2v  += step(dvec2v, dvec2v);
    dvec3v  += step(dvec3v, dvec3v);
    dvec4v  += step(dvec4v, dvec4v);
    dvec2v  += step(doublev, dvec2v);
    dvec3v  += step(doublev, dvec3v);
    dvec4v  += step(doublev, dvec4v);

    doublev += smoothstep(doublev, doublev, doublev);
    dvec2v  += smoothstep(dvec2v, dvec2v, dvec2v);
    dvec3v  += smoothstep(dvec3v, dvec3v, dvec3v);
    dvec4v  += smoothstep(dvec4v, dvec4v, dvec4v);
    dvec2v  += smoothstep(doublev, doublev, dvec2v);
    dvec3v  += smoothstep(doublev, doublev, dvec3v);
    dvec4v  += smoothstep(doublev, doublev, dvec4v);

    boolv  = isnan(doublev);
    bvec2v = isnan(dvec2v);
    bvec3v = isnan(dvec3v);
    bvec4v = isnan(dvec4v);

    boolv  = boolv ? isinf(doublev) : false;
    bvec2v = boolv ? isinf(dvec2v)  : bvec2(false);
    bvec3v = boolv ? isinf(dvec3v)  : bvec3(false);
    bvec4v = boolv ? isinf(dvec4v)  : bvec4(false);

    doublev += length(doublev);
    doublev += length(dvec2v);
    doublev += length(dvec3v);
    doublev += length(dvec4v);

    doublev += distance(doublev, doublev);
    doublev += distance(dvec2v, dvec2v);
    doublev += distance(dvec3v, dvec3v);
    doublev += distance(dvec4v, dvec4v);

    doublev += dot(doublev, doublev);
    doublev += dot(dvec2v, dvec2v);
    doublev += dot(dvec3v, dvec3v);
    doublev += dot(dvec4v, dvec4v);

    dvec3v += cross(dvec3v, dvec3v);

    doublev += normalize(doublev);
    dvec2v  += normalize(dvec2v);
    dvec3v  += normalize(dvec3v);
    dvec4v  += normalize(dvec4v);

    doublev += faceforward(doublev, doublev, doublev);
    dvec2v  += faceforward(dvec2v, dvec2v, dvec2v);
    dvec3v  += faceforward(dvec3v, dvec3v, dvec3v);
    dvec4v  += faceforward(dvec4v, dvec4v, dvec4v);

    doublev += reflect(doublev, doublev);
    dvec2v  += reflect(dvec2v, dvec2v);
    dvec3v  += reflect(dvec3v, dvec3v);
    dvec4v  += reflect(dvec4v, dvec4v);

    doublev += refract(doublev, doublev, doublev);
    dvec2v  += refract(dvec2v, dvec2v, doublev);
    dvec3v  += refract(dvec3v, dvec3v, doublev);
    dvec4v  += refract(dvec4v, dvec4v, doublev);

    dmat2   dmat2v   = outerProduct(dvec2v, dvec2v);
    dmat3   dmat3v   = outerProduct(dvec3v, dvec3v);
    dmat4   dmat4v   = outerProduct(dvec4v, dvec4v);
    dmat2x3 dmat2x3v = outerProduct(dvec3v, dvec2v);
    dmat3x2 dmat3x2v = outerProduct(dvec2v, dvec3v);
    dmat2x4 dmat2x4v = outerProduct(dvec4v, dvec2v);
    dmat4x2 dmat4x2v = outerProduct(dvec2v, dvec4v);
    dmat3x4 dmat3x4v = outerProduct(dvec4v, dvec3v);
    dmat4x3 dmat4x3v = outerProduct(dvec3v, dvec4v);

    dmat2v *= matrixCompMult(dmat2v, dmat2v);
    dmat3v *= matrixCompMult(dmat3v, dmat3v);
    dmat4v *= matrixCompMult(dmat4v, dmat4v);
    dmat2x3v = matrixCompMult(dmat2x3v, dmat2x3v);
    dmat2x4v = matrixCompMult(dmat2x4v, dmat2x4v);
    dmat3x2v = matrixCompMult(dmat3x2v, dmat3x2v);
    dmat3x4v = matrixCompMult(dmat3x4v, dmat3x4v);
    dmat4x2v = matrixCompMult(dmat4x2v, dmat4x2v);
    dmat4x3v = matrixCompMult(dmat4x3v, dmat4x3v);

    dmat2v   *= transpose(dmat2v);
    dmat3v   *= transpose(dmat3v);
    dmat4v   *= transpose(dmat4v);
    dmat2x3v  = transpose(dmat3x2v);
    dmat3x2v  = transpose(dmat2x3v);
    dmat2x4v  = transpose(dmat4x2v);
    dmat4x2v  = transpose(dmat2x4v);
    dmat3x4v  = transpose(dmat4x3v);
    dmat4x3v  = transpose(dmat3x4v);

    doublev += determinant(dmat2v);
    doublev += determinant(dmat3v);
    doublev += determinant(dmat4v);

    dmat2v *= inverse(dmat2v);
    dmat3v *= inverse(dmat3v);
    dmat4v *= inverse(dmat4v);
}
