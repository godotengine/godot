#version 450

layout(xfb_buffer = 1, xfb_stride = 64) out;

layout (xfb_buffer = 1, xfb_offset = 16) out gl_PerVertex
{
    float gl_PointSize;
    vec4 gl_Position;
};

void main()
{
    gl_Position = vec4(1.0);
    gl_PointSize = 2.0;
}