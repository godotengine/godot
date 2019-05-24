#version 450

#extension GL_EXT_buffer_reference : enable
#pragma use_vulkan_memory_model

layout(buffer_reference, std430) buffer blockType {
    layout(offset = 0)  int a;
    layout(offset = 4)  int b;
    layout(offset = 8)  int c;
    layout(offset = 12) int d;
    layout(offset = 16) int e;
    layout(offset = 32) int f[2];
    coherent layout(offset = 48) ivec4 g;
};

layout(std430) buffer t2 {
    blockType f;
    blockType g;
} t;

void main() {
    t.f.b = t.g.a;

    blockType j = t.f;
    j.d = j.c;
    j.d = j.f[1];
    j.d = j.g.y;
}
