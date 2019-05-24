#version 450

#extension GL_EXT_buffer_reference : enable

layout(buffer_reference) buffer bufType1 { int x; };
layout(buffer_reference) buffer bufType2 { int x; };
layout(buffer_reference) uniform bufType3 { int x; };

layout(buffer_reference) buffer;
layout(buffer_reference) uniform;
layout(buffer_reference) in;
layout(buffer_reference) out;
layout(buffer_reference) in badin { float x; } badin2;
layout(buffer_reference) out badout { float x; } badout2;

layout(buffer_reference) buffer bufType5;

layout(buffer_reference) buffer bufType6 { int x[]; };

buffer bufType4 {
    bufType1 b1;
    bufType2 b2;
    bufType3 b3;
    bufType6 b6;
} b4;

void f()
{
    bufType6 b;
    b.x.length();
    b4.b6.x.length();
}

void main() {
    bufType2 x1 = b4.b1;
    bufType2 x2 = bufType2(b4.b1);
    bufType2 x3 = bufType2(b4.b2);
    bufType2 x4 = bufType2(b4.b3);

    b4.b1 = b4.b2;
    b4.b1 = b4.b3;
    b4.b3 = b4.b2;
}

layout(buffer_reference) uniform bufType5 { int x; };
