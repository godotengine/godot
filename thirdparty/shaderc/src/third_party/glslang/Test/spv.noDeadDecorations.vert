#version 310 es
precision mediump float;

float func(float a)
{
    return -a;
    a = a * -1.0;
}

void main()
{
    gl_Position.x = func(0.0);
}
