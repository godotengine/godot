#version 110

uniform vec4 bigColor;
varying vec4 BaseColor;
uniform float d;

void main()
{
    vec4 color = BaseColor;

    do {
        color += bigColor;
    } while (color.x < d);

    gl_FragColor = color;
}
