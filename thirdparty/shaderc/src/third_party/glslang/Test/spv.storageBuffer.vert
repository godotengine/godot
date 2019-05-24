#version 450

#pragma use_storage_buffer

uniform ub {
    vec4 a;
} ubi;

buffer bb {
    vec4 b;
} bbi;

void main()
{
    gl_Position = ubi.a + bbi.b;
}
