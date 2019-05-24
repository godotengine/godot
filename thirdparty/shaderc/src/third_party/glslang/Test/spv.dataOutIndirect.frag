#version 140 

in vec4 Color;

out vec4 fcolor[4];

uniform b { int i; } bName;

void main()
{
    fcolor[bName.i] = Color;
}
