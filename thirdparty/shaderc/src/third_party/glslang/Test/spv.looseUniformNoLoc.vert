#version 450 core

layout(location = 0)
in  vec4 foo;

layout(location = 0)
out vec4 bar;

uniform vec4 uv;

void main()
{
    bar = foo;
    gl_Position = foo;
}