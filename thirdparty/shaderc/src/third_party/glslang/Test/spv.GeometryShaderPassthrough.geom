#version 450
#extension GL_NV_geometry_shader_passthrough : require

layout(triangles) in;

layout(passthrough) in gl_PerVertex {
    vec4 gl_Position;
};

layout(passthrough) in Inputs {
vec2 texcoord;
vec4 baseColor;
};

void main()
{
}