#version 450

layout(xfb_buffer = 3) out;
layout(xfb_stride = 48) out;
layout(xfb_offset = 12, location = 0) out float out1;

layout(xfb_buffer = 2) out;
layout(location=1) out outXfb {
    layout(xfb_buffer = 2, xfb_stride = 32, xfb_offset = 8) float out2;
};

layout(xfb_buffer = 1, location=3) out outXfb2 {
    layout(xfb_stride = 64, xfb_offset = 60) float out3;
};

layout(location = 4, xfb_buffer = 0, xfb_offset = 4) out float out4;

void main()
{
}