#version 450

layout(constant_id = 11) const int a = 8;

void main()
{
    gl_Position = vec4(1.0) / a;
}
