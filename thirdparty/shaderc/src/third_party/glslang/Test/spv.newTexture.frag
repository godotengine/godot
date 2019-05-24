#version 430

uniform sampler2D s2D;
uniform sampler2DRect sr;
uniform sampler3D s3D;
uniform samplerCube sCube;
uniform samplerCubeShadow sCubeShadow;
uniform samplerCubeArrayShadow sCubeArrayShadow;
uniform sampler2DShadow s2DShadow;
uniform sampler2DArray s2DArray;
uniform sampler2DArrayShadow s2DArrayShadow;

uniform isampler2D is2D;
uniform isampler3D is3D;
uniform isamplerCube isCube;
uniform isampler2DArray is2DArray;
uniform isampler2DMS is2Dms;

uniform usampler2D us2D;
uniform usampler3D us3D;
uniform usamplerCube usCube;
uniform usampler2DArray us2DArray;

in float c1D;
in vec2  c2D;
in vec3  c3D;
in vec4  c4D;

flat in int   ic1D;
flat in ivec2 ic2D;
flat in ivec3 ic3D;
flat in ivec4 ic4D;

out vec4 FragData;

void main()
{
    vec4 v = texture(s2D, c2D);
    v.y += texture(sCubeArrayShadow, c4D, c1D);
    v += textureProj(s3D, c4D);
    v += textureLod(s2DArray, c3D, 1.2);
    v.y += textureOffset(s2DShadow, c3D, ivec2(3), c1D);
    v += texelFetch(s3D, ic3D, ic1D);
    v += texelFetchOffset(s2D, ic2D, 4, ivec2(3));
    v += texelFetchOffset(sr, ic2D, ivec2(4));
    v.y += textureLodOffset(s2DShadow, c3D, c1D, ivec2(3));
    v += textureProjLodOffset(s2D, c3D, c1D, ivec2(3));
    v += textureGrad(sCube, c3D, c3D, c3D);
    v.x += textureGradOffset(s2DArrayShadow, c4D, c2D, c2D, ivec2(3));
    v += textureProjGrad(s3D, c4D, c3D, c3D);
    v += textureProjGradOffset(s2D, c3D, c2D, c2D, ivec2(3));

    ivec4 iv = texture(is2D, c2D);
    v += vec4(iv);
    iv = textureProjOffset(is2D, c4D, ivec2(3));
    v += vec4(iv);
    iv = textureProjLod(is2D, c3D, c1D);
    v += vec4(iv);
    iv = textureProjGrad(is2D, c3D, c2D, c2D);
    v += vec4(iv);
    iv = texture(is3D, c3D, 4.2);
    v += vec4(iv);
    iv = textureLod(isCube, c3D, c1D);
    v += vec4(iv);
    iv = texelFetch(is2DArray, ic3D, ic1D);
    v += vec4(iv);

    ivec2 iv2 = textureSize(sCubeShadow, 2);
    // iv2 += textureSize(is2Dms);

    FragData = v + vec4(iv2, 0.0, 0.0);
}
