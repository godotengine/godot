#version 450

void main()
{
    mat4x3 m;

    vec2 v2 = vec2(m);
    vec3 v3 = vec3(m);
    vec4 v4 = vec4(m);

    ivec2 iv2 = ivec2(m);
    ivec3 iv3 = ivec3(m);
    ivec4 iv4 = ivec4(m);
}
