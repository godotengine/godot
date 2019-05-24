#version 450 core

#extension GL_AMD_shader_image_load_store_lod: enable

layout(rgba32f,  binding = 0) uniform image1D         i1D;
layout(rgba32f,  binding = 1) uniform image2D         i2D;
layout(rgba32f,  binding = 2) uniform image3D         i3D;
layout(rgba32i,  binding = 3) uniform iimageCube      iiCube;
layout(rgba32i,  binding = 4) uniform iimage1DArray   ii1DArray;
layout(rgba32ui, binding = 5) uniform uimage2DArray   ui2DArray;
layout(rgba32ui, binding = 6) uniform uimageCubeArray uiCubeArray;

layout(location = 0) out vec4 fragColor;

void main()
{
    const int c1 = 1;
    const ivec2 c2 = ivec2(2, 3);
    const ivec3 c3 = ivec3(4, 5, 6);

    const int lod = 3;

    vec4 f4 = vec4(0.0);
    f4 += imageLoadLodAMD(i1D, c1, lod);
    f4 += imageLoadLodAMD(i2D, c2, lod);
    f4 += imageLoadLodAMD(i3D, c3, lod);

    imageStoreLodAMD(iiCube, c3, lod, ivec4(f4));
    imageStoreLodAMD(ii1DArray, c2, lod, ivec4(f4));

    uvec4 u4;
    sparseImageLoadLodAMD(ui2DArray, c3, lod, u4);
    sparseImageLoadLodAMD(uiCubeArray, c3, lod, u4);

    fragColor = f4 + vec4(u4);
}