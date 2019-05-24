#version 450

#extension GL_EXT_buffer_reference : enable

layout(buffer_reference, std140) buffer t3 {
    int h;
};

layout(set = 1, binding = 2, std140) uniform t4 {
    layout(offset = 0)  int j;
    t3 k;
} x;

void main() {
    x.k.h = x.j;
}
