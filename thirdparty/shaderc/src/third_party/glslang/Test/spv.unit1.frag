#version 460

float f;
float a1;

float foo();

out float cout;

void main()
{
    f = 10;
    float g = foo();
    f += g;
    f += gl_FragCoord.y;
}