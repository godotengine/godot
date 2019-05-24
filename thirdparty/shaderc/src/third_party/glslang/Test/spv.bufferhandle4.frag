#version 450

#extension GL_EXT_buffer_reference : enable

layout(buffer_reference) buffer t4;

layout(buffer_reference, std430) buffer t3 {
    int h;
    t4 i;
};

layout(set = 1, binding = 2, buffer_reference, std430) buffer t4 {
    layout(offset = 0)  int j;
    t3 k;
} x;

layout(std430) buffer t5 {
    t4 m;
} s5;

void main() {
    x.k.h = s5.m.k.i.k.i.k.h;

    bool b = true;
    s5.m = b ? s5.m : s5.m.k.i;
}
