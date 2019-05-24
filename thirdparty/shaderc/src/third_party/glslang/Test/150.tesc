#version 150

#extension GL_ARB_tessellation_shader : enable

layout(vertices = 4) out;
int outa[gl_out.length()];

patch out vec4 patchOut;

void main()
{
    barrier();

    int a = gl_MaxTessControlInputComponents +
            gl_MaxTessControlOutputComponents +
            gl_MaxTessControlTextureImageUnits +
            gl_MaxTessControlUniformComponents +
            gl_MaxTessControlTotalOutputComponents;

    vec4 p = gl_in[1].gl_Position;
    float ps = gl_in[1].gl_PointSize;
    float cd = gl_in[1].gl_ClipDistance[2];

    int pvi = gl_PatchVerticesIn;
    int pid = gl_PrimitiveID;
    int iid = gl_InvocationID;

    gl_out[gl_InvocationID].gl_Position = p;
    gl_out[gl_InvocationID].gl_PointSize = ps;
    gl_out[gl_InvocationID].gl_ClipDistance[1] = cd;

    gl_TessLevelOuter[3] = 3.2;
    gl_TessLevelInner[1] = 1.3;
}
