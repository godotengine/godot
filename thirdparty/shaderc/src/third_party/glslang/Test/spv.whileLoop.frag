#version 140

in vec4 bigColor;
in vec4 BaseColor;
in float d;

void main()
{
    vec4 color = BaseColor;

    while (color.x < d) {
        color += bigColor;
    }

    gl_FragColor = color;
}
