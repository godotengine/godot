#version 450

layout(xfb_buffer=2) out;

layout(location=5, xfb_stride=20) out block {
   float y1_out;
   vec4 y2_out;
};

void main() {
   y1_out = 7.0;
   y2_out = vec4(1.0, 0.0, 0.0, 1.0);
   gl_Position = vec4(0.0);
}
