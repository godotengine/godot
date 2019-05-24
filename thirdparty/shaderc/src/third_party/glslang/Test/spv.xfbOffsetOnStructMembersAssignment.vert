#version 450

layout(xfb_buffer=2) out;

struct S {
   float x1_out;
   float x2_out;
};

layout(location=0, xfb_offset = 16) out S s1;

layout(location=5, xfb_buffer=1, xfb_offset=8) out struct S2 {
   float y1_out;
   vec4 y2_out;
}s2;

void main() {
   s1.x1_out = 5.0;
   s1.x2_out = 6.0;
   s2.y1_out = 7.0;
   s2.y2_out = vec4(1.0, 0.0, 0.0, 1.0);
   gl_Position = vec4(0.0);
}
