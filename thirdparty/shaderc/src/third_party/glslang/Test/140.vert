#version 140

uniform isamplerBuffer sbuf;

layout(std140) uniform blockName {
    int anonMem;
};

void main()
{
    int id = gl_InstanceID;
    id += anonMem;
    id += texelFetch(sbuf, 8).w;
    gl_ClipVertex;      // could be ERROR, but compiling under compatibility profile
    gl_Color;           // could be ERROR, but compiling under compatibility profile
    gl_LightSource[0];  // could be ERROR, but compiling under compatibility profile
    gl_DepthRange.far;
    gl_TexCoord;        // could be ERROR, but compiling under compatibility profile
    gl_FogFragCoord;    // could be ERROR, but compiling under compatibility profile
    gl_FrontColor;      // could be ERROR, but compiling under compatibility profile
}

out vec4 gl_Position;  // ERROR

layout(location = 9) in vec4 locBad;  // ERROR

#extension GL_ARB_explicit_attrib_location : enable

layout(location = 9) in vec4 loc;

#extension GL_ARB_separate_shader_objects : enable

out vec4 gl_Position;
in vec4 gl_Position;   // ERROR
out vec3 gl_Position;  // ERROR

out float gl_PointSize;
out vec4 gl_ClipVertex;
out float gl_FogFragCoord;

uniform sampler2DRect s2dr;
uniform sampler2DRectShadow s2drs;
in ivec2 itloc2;
in vec2 tloc2;
in vec3 tloc3;
in vec4 tloc4;

void foo()
{
    vec4 v = texelFetch(s2dr, itloc2);
    v += texelFetch(s2dr, itloc2, 0.2);     // ERROR, no lod
    v += texture(s2dr, tloc2);
    v += texture(s2dr, tloc2, 0.3);         // ERROR, no bias
    v += texture(s2drs, tloc3);
    v += textureProj(s2dr, tloc3);
    v += textureProj(s2dr, tloc4);
    v += textureProjGradOffset(s2dr, tloc4, ivec2(0.0), ivec2(0.0), ivec2(1,2));
    v += textureProjGradOffset(s2drs, tloc4, ivec2(0.0), ivec2(0.0), ivec2(1,2));
}

void devi()
{
    gl_DeviceIndex; // ERROR, no extension
    gl_ViewIndex;   // ERROR, no extension
}

#ifdef GL_EXT_device_group
#extension GL_EXT_device_group : enable
#endif

#ifdef GL_EXT_device_group
#extension GL_EXT_multiview : enable
#endif

void devie()
{
    gl_DeviceIndex;
    gl_ViewIndex;
}
