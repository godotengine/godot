#version 450

#extension GL_EXT_buffer_reference : enable

layout(set = 1, binding = 2, buffer_reference, std430) buffer t4 {
    layout(offset = 0)  int j;
};

layout(std430) buffer t5 {
    t4 m;
} s5;

t4 f1(const t4 y) { return y; }
t4 f2(t4 y) { return y; }
t4 f3(const restrict t4 y) { return y; }
t4 f4(restrict t4 y) { return y; }

t4 g1;
restrict t4 g2;

void main()
{
    t4 a = s5.m;
    restrict t4 b = s5.m;

    f1(a);
    f2(a);
    f3(a);
    f4(a);
}
