#version 140

in mat4 transform;

attribute vec4 position;
in vec2 uv_in;

out vec2 uv;

void main()
{
    uv = uv_in;
    gl_Position = transform * position;
}
