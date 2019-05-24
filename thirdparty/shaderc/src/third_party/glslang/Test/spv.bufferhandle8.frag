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

layout(std430, buffer_reference) buffer T2 { int x; };
layout(std430, buffer_reference) buffer T1 { int x; };

struct Blah {
    T1 t1;
    T2 t2;
};

layout(set=0, binding=0) buffer T3 {
  Blah Bindings[];
} t3;

void main() {
    t3.Bindings[0] = t3.Bindings[1];
}
