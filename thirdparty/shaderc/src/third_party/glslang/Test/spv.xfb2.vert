#version 450

layout (location = 0) in vec4 position;
layout (binding = 5) uniform ComponentsBlock
{
    vec4 c1;
    vec2 c2;
} components;

layout (xfb_buffer = 3, xfb_offset = 16) out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    gl_Position = position + components.c1 + vec4(components.c2, 0.0, 0.0);
}