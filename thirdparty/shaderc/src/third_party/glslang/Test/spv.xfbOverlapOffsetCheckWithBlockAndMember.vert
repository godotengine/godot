#version 450

/* block definition from GLSL spec 4.60, section 4.4.2, Output Layout Qualifiers */

layout(location=5, xfb_buffer = 3, xfb_offset = 12) out block2 {
   vec4 v; // v will be written to byte offsets 12 through 27 of buffer
   float u; // u will be written to offset 28
   layout(xfb_offset = 40) vec4 w;
   vec4 x; // x will be written to offset 56, the next available offset
};

void main() {
   v = vec4(1.0, 0.0, 1.0, 0.0);
   u = 5.0;
   w = vec4(1.0, 0.0, 0.0, 1.0);
   x = vec4(5.0, 0.0, 0.0, 0.0);

   gl_Position = vec4(0.0);
}
