#version 450 core

#extension GL_EXT_buffer_reference : enable
layout (push_constant, std430) uniform Block { int identity[32]; } pc;
layout(r32ui, set = 3, binding = 0) uniform uimage2D image0_0;
layout(buffer_reference) buffer T1;
layout(set = 3, binding = 1, buffer_reference) buffer T1 {
   layout(offset = 0) int a[2]; // stride = 4 for std430, 16 for std140
   layout(offset = 32) int b;
   layout(offset = 48) T1  c[2]; // stride = 8 for std430, 16 for std140
   layout(offset = 80) T1  d;
} x;
void main()
{
  int accum = 0, temp;
   accum |= x.a[0] - 0;
   accum |= x.a[pc.identity[1]] - 1;
   accum |= x.b - 2;
   accum |= x.c[0].a[0] - 3;
   accum |= x.c[0].a[pc.identity[1]] - 4;
   accum |= x.c[0].b - 5;
   accum |= x.c[pc.identity[1]].a[0] - 6;
   accum |= x.c[pc.identity[1]].a[pc.identity[1]] - 7;
   accum |= x.c[pc.identity[1]].b - 8;
   accum |= x.d.a[0] - 9;
   accum |= x.d.a[pc.identity[1]] - 10;
   accum |= x.d.b - 11;
  uvec4 color = (accum != 0) ? uvec4(0,0,0,0) : uvec4(1,0,0,1);
  imageStore(image0_0, ivec2(gl_FragCoord.x, gl_FragCoord.y), color);
}