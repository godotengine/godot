#version 450

#extension GL_ARB_gpu_shader_int64 : enable
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

flat in uint64_t h, i;

void main() {

    blockType b1[2] = blockType[2](blockType(h), blockType(i));
    b1[0].a = b1[1].b;
    blockType b2 = blockType(h);
    blockType b3 = blockType(i);
    b2.a = b3.b;
    uint64_t j = uint64_t(b2);
    b2 = blockType(j+256);
}
