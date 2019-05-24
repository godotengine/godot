#version 450

layout(constant_id = 3) const int a = 2;
layout(location = 2) uniform float f;
layout(location = 4, binding = 1) uniform sampler2D s1;
layout(binding = 2) uniform sampler2D s2;

void main()
{
}
