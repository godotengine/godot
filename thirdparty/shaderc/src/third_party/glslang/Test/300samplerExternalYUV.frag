#version 300 es

#extension GL_EXT_YUV_target : enable

uniform __samplerExternal2DY2YEXT sExt;
precision mediump __samplerExternal2DY2YEXT;
uniform __samplerExternal2DY2YEXT mediumExt;
uniform highp __samplerExternal2DY2YEXT highExt;

void main()
{
    texture2D(sExt, vec2(0.2));  // ERROR
    texture2D(mediumExt, vec2(0.2));  // ERROR
    texture2D(highExt, vec2(0.2));  // ERROR
    texture2DProj(sExt, vec3(0.3));  // ERROR
    texture2DProj(sExt, vec4(0.3));  // ERROR

    int lod = 0;
    highp float bias = 0.01;
    textureSize(sExt, lod);
    texture(sExt, vec2(0.2));
    texture(sExt, vec2(0.2), bias);
    textureProj(sExt, vec3(0.2));
    textureProj(sExt, vec3(0.2), bias);
    textureProj(sExt, vec4(0.2));
    textureProj(sExt, vec4(0.2), bias);
    texelFetch(sExt, ivec2(4), lod);

    texture3D(sExt, vec3(0.3));  // ERROR
    texture2DProjLod(sExt, vec3(0.3), 0.3);  // ERROR
    texture(sExt, vec3(0.3));  // ERROR
    textureProjLod(sExt, vec3(0.3), 0.3);  // ERROR
}

#extension GL_EXT_YUV_target : disable

uniform __samplerExternal2DY2YEXT badExt;  // ERROR

