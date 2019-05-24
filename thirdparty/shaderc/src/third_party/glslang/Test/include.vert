#version 450

#extension GL_GOOGLE_include_directive : enable

#define float4 vec4

#include "bar.h"
#include "./inc1/bar.h"
#include "inc2\bar.h"

out vec4 color;

void main()
{
    color = i1 + i2 + i3 + i4 + i5 + i6;
}
