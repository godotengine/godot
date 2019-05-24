#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable

layout(buffer_reference) buffer T1 {
    int x;
    bool y;
};
layout(buffer_reference) buffer T2 {
    int x;
};

const int s = int(uint64_t(T1(T2(uint64_t(3)))));
int x[s];
const uint64_t t = uint64_t(true ? T2(uint64_t(10)) : T2(uint64_t(11)));

void main()
{
    T1 a = T1(uint64_t(4)), b = T1(uint64_t(5));
    T1 c = true ? a : b;
    T1 d = (a,b);
    T1 e = true ? T1(uint64_t(6)) : T1(uint64_t(7));
    T1 f = a.y ? T1(uint64_t(8)) : T1(uint64_t(9));
}
