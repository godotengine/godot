#version 140

in vec4 k;
out vec4 o;

in float gl_ClipDistance[5];

layout(row_major) uniform;

uniform sampler2D samp2Da[3];

layout(std140) uniform bn {
    layout(row_major) mat4 matra[4];
    layout(column_major) mat4 matca[4];
    layout(row_major) mat4 matr;
    layout(column_major) mat4 matc;
    layout(align=512, offset=1024) mat4 matrdef;
};

uniform sampler2DRect sampR;
uniform isamplerBuffer sampB;

float foo();

void main()
{
    o.y = gl_ClipDistance[2];
    o.z = gl_ClipDistance[int(k)];
    o.w = float(textureSize(sampR) + textureSize(sampB)) / 100.0;
    o.z = foo();
}

// Test extra-function initializers

float i1 = gl_FrontFacing ? -2.0 : 2.0;
float i2 = 102;

float foo()
{
    return i1 + i2;
}

// test arrayed block
layout(std140) uniform bi {
    vec3 v[2];
} bname[4];
