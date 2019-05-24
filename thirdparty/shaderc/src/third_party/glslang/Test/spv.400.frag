#version 400 core

in vec2 c2D;
flat in int i;
flat in uint u;
out uint uo;
out vec4 outp;
out ivec4 ioutp;
out uvec4 uoutp;
uniform sampler2D arrayedSampler[5];
uniform usampler2DRect samp2dr;
uniform isampler2DArray isamp2DA;
uniform sampler2DRectShadow u2drs;

#extension GL_ARB_separate_shader_objects : enable

layout(location = 6) in vec4 vl2;

void foo23()
{
    const ivec2[3] offsets = ivec2[3](ivec2(1,2), ivec2(3,4), ivec2(15,16));

    outp.x += textureProjGradOffset(u2drs, outp, vec2(0.0), vec2(0.0), offsets[1]);
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
    dvec4v  = sqrt(dvec4(doublev));

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
    dmat2x3v = matrixCompMult(dmat2x3v, dmat2x3v);  // For now, relying on no dead-code elimination
    dmat2x4v = matrixCompMult(dmat2x4v, dmat2x4v);
    dmat3x2v = matrixCompMult(dmat3x2v, dmat3x2v);
    dmat3x4v = matrixCompMult(dmat3x4v, dmat3x4v);
    dmat4x2v = matrixCompMult(dmat4x2v, dmat4x2v);
    dmat4x3v = matrixCompMult(dmat4x3v, dmat4x3v);

    dmat2v   *= transpose(dmat2v);
    dmat3v   *= transpose(dmat3v);
    dmat4v   *= transpose(dmat4v);
    dmat2x3v  = transpose(dmat3x2v);  // For now, relying on no dead-code elimination
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
    
    outp *= float(doublev + dvec2v.y + dvec3v.z + dvec4v.w + 
                  dmat2v[1][1] + dmat3v[2][2] + dmat4v[3][3] + dmat2x3v[1][1] + dmat3x2v[1][1] + dmat3x4v[2][2] + dmat4x3v[2][2] + dmat2x4v[1][1] + dmat4x2v[1][1] +
                  float(boolv) + float(bvec2v.x) + float(bvec3v.x) + float(bvec4v.x));
}

void main()
{
    vec4 v;
    v = texture(arrayedSampler[i], c2D);
    outp.x = gl_ClipDistance[1];
    outp.yzw = v.yzw;

    ivec2 offsets[4];
    const ivec2 constOffsets[4] = ivec2[4](ivec2(1,2), ivec2(3,4), ivec2(15,16), ivec2(-2,0));
    uoutp = textureGatherOffsets(samp2dr, c2D, constOffsets, 2);
    outp += textureGather(arrayedSampler[0], c2D);
    ioutp = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 3);
    ioutp += textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 1+2);
    ioutp += textureGatherOffset(isamp2DA, vec3(0.1), ivec2(i));

    outp += gl_FragCoord + vl2;
    uo = u % i;
    foo23();
    doubles();

    int id = gl_PrimitiveID;
}

