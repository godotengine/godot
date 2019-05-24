#version 450 core

#extension GL_AMD_gpu_shader_half_float: enable
#extension GL_AMD_gpu_shader_int16: enable

layout(location = 0) in f16vec4 if16v4;
layout(location = 1) in i16vec4 ii16v4;
layout(location = 2) in u16vec4 iu16v4;

layout(location = 0, xfb_buffer = 0, xfb_stride = 6, xfb_offset = 0) out f16vec3 of16v3;
layout(location = 1, xfb_buffer = 1, xfb_stride = 6, xfb_offset = 0) out F16Out
{
    float16_t of16;
    f16vec2   of16v2;
};

layout(location = 5, xfb_buffer = 2, xfb_stride = 6, xfb_offset = 0) out i16vec3 oi16v3;
layout(location = 6, xfb_buffer = 3, xfb_stride = 6, xfb_offset = 0) out I16Out
{
    uint16_t ou16;
    u16vec2  ou16v2;
};

void main()
{
    of16v3 = if16v4.xyz;
    of16   = if16v4.x;
    of16v2 = if16v4.xy;

    oi16v3 = ii16v4.xyz;
    ou16   = iu16v4.x;
    ou16v2 = iu16v4.xy;
}