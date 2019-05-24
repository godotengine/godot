#version 310 es
layout (binding=0) uniform Block {
  highp int a[];
} uni;
layout (location=0) out highp int o;
void main() {
  o = uni.a[2];
}
