#version 420 core

#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 4) out;

out gl_PerVertex {
    vec4 gl_Position;
} gl_out[3];                 // ERROR, wrong size

out int a[gl_out.length()];
out int outb[5];             // ERROR, wrong size
out int outc[];

void main()
{
    vec4 p = gl_in[1].gl_Position;
    float ps = gl_in[1].gl_PointSize;
    float cd = gl_in[1].gl_ClipDistance[2];

    int pvi = gl_PatchVerticesIn;
    int pid = gl_PrimitiveID;
    int iid = gl_InvocationID;

    gl_out[gl_InvocationID].gl_Position = p;
    gl_out[gl_InvocationID].gl_PointSize = ps;        // ERROR
}

out float outf;  // ERROR, no array

layout (location = 0) in dmat2x4 vs_tcs_first[];
layout (location = 12) in dmat2x4 vs_tcs_last[];

void foo()
{
 if ((dmat2x4(dvec4(-0.625, -0.5, -0.375lf, -0.25), dvec4(-0.375, -0.25, -0.125, 0)) != vs_tcs_first[0]) ||
        (dmat2x4(dvec4(0.375, 0.5, 0.625, 0.75), dvec4(0.625, 0.75, 0.875, -0.625)) != vs_tcs_last[0]))
    {
        ;
    }
}

layout(vertices = 0) out;  // ERROR, can't be 0
