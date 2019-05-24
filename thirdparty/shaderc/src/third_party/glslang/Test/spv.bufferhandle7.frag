#version 450

#extension GL_EXT_buffer_reference : enable

layout(buffer_reference, std430) buffer blockType {
    layout(offset = 0)  int a;
    layout(offset = 4)  int b;
    layout(offset = 8)  int c;
    layout(offset = 12) int d;
    layout(offset = 16) int e;
};

layout(std430, buffer_reference) buffer t2 {
    blockType f;
    blockType g;
} t;

layout(std430) buffer t3 {
    t2 f;
} u;

void main() {
    t.f = blockType(u.f);
}
