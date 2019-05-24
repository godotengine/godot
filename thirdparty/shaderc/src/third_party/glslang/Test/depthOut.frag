#version 130

varying vec4 Color;
varying float Depth;

void main()
{
    gl_FragDepth = Depth;
    gl_FragColor = Color;
}
