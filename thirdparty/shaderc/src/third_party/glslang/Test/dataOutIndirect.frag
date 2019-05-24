#version 130 

varying vec4 Color;

uniform int i;

void main()
{
    gl_FragData[i] = Color;
}
