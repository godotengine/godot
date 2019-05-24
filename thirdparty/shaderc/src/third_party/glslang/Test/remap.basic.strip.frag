#version 450

in float  inf;
out vec4  outf4;

vec3 dead_fn() { return vec3(0); }

void main()
{
    outf4 = vec4(inf);
}
