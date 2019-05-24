#version 450 core

#extension GL_ARB_shader_stencil_export: enable

out int gl_FragStencilRefARB;

void main()
{
    gl_FragStencilRefARB = 100;
}
