#version 300 es

#extension GL_OES_EGL_image_external_essl3 : enable

uniform samplerExternalOES sExt;
precision mediump samplerExternalOES;
uniform samplerExternalOES mediumExt;
uniform highp samplerExternalOES highExt;

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

#extension GL_OES_EGL_image_external_essl3 : disable

#extension GL_OES_EGL_image_external : enable
uniform samplerExternalOES badExt;  // ERROR
#extension GL_OES_EGL_image_external : disable

uniform samplerExternalOES badExt;  // ERROR
