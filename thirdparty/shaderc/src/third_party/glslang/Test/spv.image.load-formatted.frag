#version 450

#extension GL_EXT_shader_image_load_formatted : require

layout(binding = 0)    uniform image1D         i1D;
layout(binding = 1)    uniform image2D         i2D;
layout(binding = 2)    uniform image3D         i3D;
layout(binding = 3)    uniform imageCube       iCube;
layout(binding = 4)    uniform imageCubeArray  iCubeArray;
layout(binding = 5)    uniform image2DRect     i2DRect;
layout(binding = 6)    uniform image1DArray    i1DArray;
layout(binding = 7)       uniform image2DArray    i2DArray;
layout(binding = 8)    uniform imageBuffer     iBuffer;
layout(binding = 9)    uniform image2DMS       i2DMS;
layout(binding = 10)   uniform image2DMSArray  i2DMSArray;

flat in int     ic1D;
flat in ivec2   ic2D;
flat in ivec3   ic3D;
flat in ivec4   ic4D;

writeonly layout(binding = 1)   uniform image2D         wo2D;

flat in uint value;

out vec4 fragData;

void main()
{
    ivec3 iv = ivec3(0);
    iv.x    += imageSize(i1D);
    iv.xy   += imageSize(i2D);
    iv.xyz  += imageSize(i3D);
    iv.xy   += imageSize(iCube);
    iv.xyz  += imageSize(iCubeArray);
    iv.xy   += imageSize(i2DRect);
    iv.xy   += imageSize(i1DArray);
    iv.xyz  += imageSize(i2DArray);
    iv.x    += imageSize(iBuffer);
    iv.xy   += imageSize(i2DMS);
    iv.xyz  += imageSize(i2DMSArray);

    iv.x    += imageSamples(i2DMS);
    iv.x    += imageSamples(i2DMSArray);

    vec4 v = vec4(0.0);
    v += imageLoad(i1D, ic1D);
    imageStore(i1D, ic1D, v);
    v += imageLoad(i2D, ic2D);
    imageStore(i2D, ic2D, v);
    v += imageLoad(i3D, ic3D);
    imageStore(i3D, ic3D, v);
    v += imageLoad(iCube, ic3D);
    imageStore(iCube, ic3D, v);
    v += imageLoad(iCubeArray, ic3D);
    imageStore(iCubeArray, ic3D, v);
    v += imageLoad(i2DRect, ic2D);
    imageStore(i2DRect, ic2D, v);
    v += imageLoad(i1DArray, ic2D);
    imageStore(i1DArray, ic2D, v);
    v += imageLoad(i2DArray, ic3D);
    imageStore(i2DArray, ic3D, v);
    v += imageLoad(iBuffer, ic1D);
    imageStore(iBuffer, ic1D, v);
    v += imageLoad(i2DMS, ic2D, 1);
    imageStore(i2DMS, ic2D, 2, v);
    v += imageLoad(i2DMSArray, ic3D, 3);
    imageStore(i2DMSArray, ic3D, 4, v);

    imageStore(wo2D, ic2D, v);

    fragData = v;
}

