#version 130 

varying vec4 Color;

void main()
{
    gl_FragData[1] = Color;
}
