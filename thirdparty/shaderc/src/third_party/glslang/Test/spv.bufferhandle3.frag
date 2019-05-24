#version 450

#extension GL_EXT_buffer_reference : enable

layout(buffer_reference, std430) buffer t3 {
    int h;
};

layout(set = 1, binding = 2, buffer_reference, std430) buffer t4 {
    layout(offset = 0)  int j;
    t3 k;
} x;

layout(std430) buffer t5 {
    t4 m;
} s5;

flat in t4 k;

t4 foo(t4 y) { return y; }

void main() {
    foo(s5.m).j = s5.m.k.h;
    x.j = k.k.h;
}
