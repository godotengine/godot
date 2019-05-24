#version 460 core
#extension GL_AMD_gpu_shader_int16 : enable
#extension GL_ARB_gpu_shader_int64 : enable

void a(){
    int16_t u = -32768S / -1S; // SHRT_MIN
    int v = -2147483648 / -1; // INT_MIN
    int64_t w = -9223372036854775808L / -1L; // LLONG_MIN
    int16_t x = -32768S % -1S; // SHRT_MIN
    int y = -2147483648 % -1; // INT_MIN
    int64_t z = -9223372036854775808L % -1L; // LLONG_MIN
}