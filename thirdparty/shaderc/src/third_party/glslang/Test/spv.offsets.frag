#version 450

layout(set = 0, binding = 0, std140) uniform n1 {
    layout(offset = 8)  int a;
    layout(offset = 4)  int b;
    layout(offset = 0)  int c;
    layout(offset = 12) int d;
} i1;

layout(set = 0, binding = 1, std430) buffer n2 {
    layout(offset = 32) vec3 e;
                        vec3 f;
    layout(offset = 16) vec3 g;
    layout(offset = 0)  vec3 h;
} i2;

void main() {}