#version 450 core

layout(location = 0)
in  vec4 foo;

layout(location = 0)
out vec4 bar;

uniform vec4 uv1;
uniform float uv2;
uniform vec3 uv3;

layout(binding = 0) uniform atomic_uint a_uint;

void main()
{
    bar = foo;
    gl_Position = foo;
}