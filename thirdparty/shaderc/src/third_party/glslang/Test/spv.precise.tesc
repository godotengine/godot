#version 310 es
#extension GL_EXT_tessellation_shader : require
#extension GL_EXT_gpu_shader5 : require

layout(vertices = 3) out;

layout(location = 0) in highp vec2  in_tc_position[];
layout(location = 1) in highp float in_tc_tessParam[];

layout(location = 0) out highp vec2 in_te_position[];

precise gl_TessLevelOuter;

void main (void)
{
    in_te_position[gl_InvocationID] = in_tc_position[gl_InvocationID];

    gl_TessLevelInner[0] = 5.0;
    gl_TessLevelInner[1] = 5.0;

    gl_TessLevelOuter[0] = 1.0 + 59.0 * 0.5 * (in_tc_tessParam[1] + in_tc_tessParam[2]);
    gl_TessLevelOuter[1] = 1.0 + 59.0 * 0.5 * (in_tc_tessParam[2] + in_tc_tessParam[0]);
    gl_TessLevelOuter[2] = 1.0 + 59.0 * 0.5 * (in_tc_tessParam[0] + in_tc_tessParam[1]);
}
