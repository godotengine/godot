#version 450

layout(binding = 0, r32f) uniform coherent image1D      i1D;
layout(binding = 1, r32f) uniform volatile image2D      i2D;
layout(binding = 2, r32f) uniform restrict image2DRect  i2DRect;
layout(binding = 3, r32f) uniform readonly image3D      i3D;
layout(binding = 3, r32f) uniform writeonly imageCube   iCube;

struct Data
{
    float f1;
    vec2  f2;
};

coherent buffer Buffer
{
    volatile float f1;
    restrict vec2  f2;
    readonly vec3  f3;
    writeonly vec4 f4;
    int i1;
    Data data;
};

void main()
{
    vec4 texel = imageLoad(i1D, 1);
    texel += imageLoad(i2D, ivec2(1));
    texel += imageLoad(i2DRect, ivec2(1));
    texel += imageLoad(i3D, ivec3(1));
    imageStore(iCube, ivec3(1), texel);

    texel[i1] = f1;
    texel.xy += f2;
    texel.xyz -= f3;
    texel.w += data.f1 + data.f2[1];
    f4 = texel;
}