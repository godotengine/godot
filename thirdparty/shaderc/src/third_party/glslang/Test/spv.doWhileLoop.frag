#version 140

in vec4 bigColor;
in vec4 BaseColor;
in float d;

void main()
{
    vec4 color = BaseColor;

    do {
        color += bigColor;
    } while (color.x < d);

    gl_FragColor = color;
}
