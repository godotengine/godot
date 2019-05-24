#version 450

#extension GL_ARB_shader_draw_parameters : enable

out vec3 pos;

void main()
{
    int a = gl_BaseVertexARB;
    int b = gl_BaseInstanceARB;
    int c = gl_DrawIDARB;
    pos = vec3(a, b, c);
}
