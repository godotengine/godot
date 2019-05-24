#version 150

in vec4 BaseColor;

vec4 foo()
{
    return BaseColor;
}

void main()
{
    gl_FragColor = foo();
}
