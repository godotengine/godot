#version 450

vec4 getColor();

layout(location=0) out vec4 color;

int a1[];  // max size from link1
int a2[];  // max size from link2
int b[5];
int c[];
int i;

buffer bnameRuntime  { float r[]; };
buffer bnameImplicit { float m[]; };

void main()
{
    color = getColor();

    a1[8] = 1;
    a2[1] = 1;
    b[i] = 1;
    c[3] = 1;
}
