#version 140 

in vec4 Color;

void main()
{
    gl_FragData[1] = Color;
}
