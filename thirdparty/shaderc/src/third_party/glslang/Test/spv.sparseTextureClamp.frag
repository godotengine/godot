#version 450
#extension GL_ARB_sparse_texture_clamp: enable

uniform sampler2D               s2D;
uniform sampler3D               s3D;
uniform sampler2DShadow         s2DShadow;
uniform samplerCubeShadow       sCubeShadow;
uniform sampler2DArrayShadow    s2DArrayShadow;
uniform sampler2DRectShadow     s2DRectShadow;
uniform samplerCubeArrayShadow  sCubeArrayShadow;

uniform isamplerCube            isCube;
uniform isampler2DArray         is2DArray;

uniform usamplerCubeArray       usCubeArray;
uniform usampler2DRect          us2DRect;

in vec2 c2;
in vec3 c3;
in vec4 c4;

in float lodClamp;

out vec4 outColor;

void main()
{
    int   resident = 0;
    vec4  texel  = vec4(0.0);
    ivec4 itexel = ivec4(0);
    uvec4 utexel = uvec4(0);

    resident |= sparseTextureClampARB(s2D, c2, lodClamp, texel);
    resident |= sparseTextureClampARB(s3D, c3, lodClamp, texel, 2.0);
    resident |= sparseTextureClampARB(isCube, c3, lodClamp, itexel);
    resident |= sparseTextureClampARB(s2DShadow, c3, lodClamp, texel.x);
    resident |= sparseTextureClampARB(sCubeArrayShadow, c4, 1.0, lodClamp, texel.x);

    texel   += textureClampARB(s2D, c2, lodClamp);
    texel   += textureClampARB(s3D, c3, lodClamp, 2.0);
    itexel  += textureClampARB(isCube, c3, lodClamp);
    texel.x += textureClampARB(s2DShadow, c3, lodClamp);
    texel.x += textureClampARB(sCubeArrayShadow, c4, 1.0, lodClamp);

    resident |= sparseTextureOffsetClampARB(s3D, c3, ivec3(2), lodClamp, texel, 2.0);
    resident |= sparseTextureOffsetClampARB(us2DRect, c2, ivec2(3), lodClamp, utexel);
    resident |= sparseTextureOffsetClampARB(s2DArrayShadow, c4, ivec2(5), lodClamp, texel.z);

    texel   += textureOffsetClampARB(s3D, c3, ivec3(2), lodClamp, 2.0);
    utexel  += textureOffsetClampARB(us2DRect, c2, ivec2(3), lodClamp);
    texel.z += textureOffsetClampARB(s2DArrayShadow, c4, ivec2(5), lodClamp);

    resident |= sparseTextureGradClampARB(s3D, c3, c3, c3, lodClamp, texel);
    resident |= sparseTextureGradClampARB(sCubeShadow, c4, c3, c3, lodClamp, texel.y);
    resident |= sparseTextureGradClampARB(usCubeArray, c4, c3, c3, lodClamp, utexel);

    texel   += textureGradClampARB(s3D, c3, c3, c3, lodClamp);
    texel.y += textureGradClampARB(sCubeShadow, c4, c3, c3, lodClamp);
    utexel  += textureGradClampARB(usCubeArray, c4, c3, c3, lodClamp);

    resident |= sparseTextureGradOffsetClampARB(s2D, c2, c2, c2, ivec2(5), lodClamp, texel);
    resident |= sparseTextureGradOffsetClampARB(s2DRectShadow, c3, c2, c2, ivec2(6), lodClamp, texel.w);
    resident |= sparseTextureGradOffsetClampARB(is2DArray, c3, c2, c2, ivec2(2), lodClamp, itexel);

    texel   += textureGradOffsetClampARB(s2D, c2, c2, c2, ivec2(5), lodClamp);
    texel.w += textureGradOffsetClampARB(s2DRectShadow, c3, c2, c2, ivec2(6), lodClamp);
    itexel  += textureGradOffsetClampARB(is2DArray, c3, c2, c2, ivec2(2), lodClamp);

    outColor = sparseTexelsResidentARB(resident) ? texel : vec4(itexel) + vec4(utexel);
}