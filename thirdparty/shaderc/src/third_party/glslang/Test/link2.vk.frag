#version 450

layout(binding=1) uniform sampler2D s2D;

int a1[];  // max size from link1
int a2[];  // max size from link2
int b[];
int c[7];
int i;

buffer bnameRuntime  { float r[]; };
buffer bnameImplicit { float m[4]; };

vec4 getColor()
{
    a1[2] = 1;
    a2[9] = 1;
    b[2] = 1;
    c[3] = 1;
    c[i] = 1;

    return texture(s2D, vec2(0.5));
}
