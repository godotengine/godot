#version 450

layout(constant_id = 17) const int arraySize = 12;            // ERROR
layout(input_attachment_index = 1) int foo;                   // ERROR
layout(push_constant) uniform ubn { int a; } ubi;             // ERROR

#ifdef VULKAN
#error VULKAN should not be defined
#endif
