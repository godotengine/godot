#version 450 core

layout(binding = 1) uniform texture2D tex2D;
layout(binding = 1) uniform texture2DMS texMS;
layout(binding = 0) uniform textureBuffer buf;

void testBad()
{
    vec4 tex2DFetch = texelFetch(tex2D, ivec2(0, 0), 0);
    vec4 texMSFetch = texelFetch(texMS, ivec2(0, 0), 0);

    // Allowed by KHR_vulkan_glsl without the extension. All others should
    // error.
    vec4 bufFetch = texelFetch(buf, 0);

    vec4 tex2DFetchOffset = texelFetchOffset(tex2D, ivec2(0, 0), 0, ivec2(0, 0));

    ivec2 tex2DSize = textureSize(tex2D, 0);
    ivec2 texMSSize = textureSize(texMS);
    int bufSize = textureSize(buf);

    int tex2DLevels = textureQueryLevels(tex2D);

    int texMSSamples = textureSamples(texMS);
}

#extension GL_EXT_samplerless_texture_functions : enable

void main()
{
    // These should all succeed.

    vec4 tex2DFetch = texelFetch(tex2D, ivec2(0, 0), 0);
    vec4 texMSFetch = texelFetch(texMS, ivec2(0, 0), 0);
    vec4 bufFetch = texelFetch(buf, 0);

    vec4 tex2DFetchOffset = texelFetchOffset(tex2D, ivec2(0, 0), 0, ivec2(0, 0));

    ivec2 tex2DSize = textureSize(tex2D, 0);
    ivec2 texMSSize = textureSize(texMS);
    int bufSize = textureSize(buf);

    int tex2DLevels = textureQueryLevels(tex2D);

    int texMSSamples = textureSamples(texMS);
}
