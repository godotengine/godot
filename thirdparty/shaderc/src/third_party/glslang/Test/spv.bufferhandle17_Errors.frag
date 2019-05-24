#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable

layout(buffer_reference) buffer T1 {
    int x;
};

const T1 a = T1(uint64_t(2));

void main()
{
    T1 b, c;
    const T1 d = b;

    b == c;
    b != c;
}
