#version 300 es
precision mediump float;
in vec4 pos;
layout (location = 2) in vec4 color;  // ERROR

layout(location = 1) out vec4 c;
layout(location = 3) out vec4 p;
layout(location = 4) out vec4 q[2];

void main()
{
    c = color;
    p = pos;
    q[1] = pos;
}

layout(location = 40) out float ca[4];  // ERROR, overlap, ERROR too big
layout(location = 41) out float cb[2];  // ERROR, overlap, ERROR too big
layout(location = 39) out float cc[6];  // ERROR, overlap, ERROR too big
