#version 450

layout(input_attachment_index = 2) uniform subpassInput subD1;     // ERROR, not this stage
layout(input_attachment_index = 2) uniform isubpassInput subD2;    // ERROR, not this stage
layout(input_attachment_index = 2) uniform usubpassInput subD3;    // ERROR, not this stage
layout(input_attachment_index = 2) uniform subpassInputMS subD4;   // ERROR, not this stage
layout(input_attachment_index = 2) uniform isubpassInputMS subD5;  // ERROR, not this stage
layout(input_attachment_index = 2) uniform usubpassInputMS subD6;  // ERROR, not this stage

out vec4 color;

layout(constant_id = 17) const ivec2 arraySizes = ivec2(12,13);    // ERROR, not a scalar
layout(constant_id = 17) uniform sampler2D s2D;                    // ERROR, not the right type or qualifier
layout(constant_id = 4000) const int c1 = 12;                      // ERROR, too big
layout(constant_id = 4) const float c2[2] = float[2](1.0, 2.0);    // ERROR, not a scalar
layout(constant_id = 4) in;

void main()
{
    color = subpassLoad(subD1); // ERROR, no such function in this stage
}

layout(binding = 0) uniform atomic_uint aui;   // ERROR, no atomics in Vulkan
layout(shared) uniform ub1n { int a; } ub1i;   // ERROR, no shared
layout(packed) uniform ub2n { int a; } ub2i;   // ERROR, no packed

layout(constant_id=222) const int arraySize = 4;

void foo()
{
    int a1[arraySize];
    int a2[arraySize] = a1;  // ERROR, can't use in initializer

    a1 = a2;      // ERROR, can't assign, even though the same type
    if (a1 == a2) // ERROR, can't compare either
        ++color;
}

layout(set = 1, push_constant) uniform badpc { int a; } badpcI;  // ERROR, no descriptor set with push_constant

#ifndef VULKAN
#error VULKAN should be defined
#endif

#if VULKAN != 100
#error VULKAN should be 100
#endif

float AofA0[2][arraySize];
float AofA1[arraySize][arraySize];
float AofA2[arraySize][2 + arraySize];
float AofA3[arraySize][2];

out ban1 {                              // ERROR, only outer dimension
    float f;
} bai1[2][arraySize];

out ban2 {
    float f;
} bai2[arraySize][2];

layout(binding = 3000) uniform sampler2D s3000;
layout(binding = 3001) uniform b3001 { int a; };
layout(location = 10) in vec4 in1;
layout(location = 10) in vec4 in2;  // ERROR, no location aliasing
