#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_EXT_buffer_reference : enable

layout(buffer_reference, std430) buffer blockType {
    uint x[];
};

layout(std430) buffer t2 {
    blockType f;
} t;

layout(location = 0) flat in uint i;

void main() {

    atomicAdd(t.f.x[i], 1);

    coherent blockType b = t.f;
    b.x[0] = 2;

    volatile blockType b2 = t.f;
    b2.x[0] = 3;
}
