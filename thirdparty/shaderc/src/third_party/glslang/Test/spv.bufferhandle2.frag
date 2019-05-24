#version 450

#extension GL_EXT_buffer_reference : enable

layout(buffer_reference, std430) buffer blockType {
    layout(offset = 0)  int a;
    layout(offset = 4)  int b;
    layout(offset = 8)  int c;
    layout(offset = 12) int d;
    layout(offset = 16) int e;
};

layout(std430) buffer t2 {
    blockType f;
    blockType g;
} t;

void main() {

    blockType b1[2] = blockType[2](t.f, t.g);
    b1[0].a = b1[1].b;
    blockType b2 = t.f;
    blockType b3 = t.g;
    b2.a = b3.b;
}
