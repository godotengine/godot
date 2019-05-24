#version 400 core

// no layout(vertices = ...) out;
int outa[gl_out.length()];  // ERROR

patch out vec4 patchOut;

void main()
{

}
