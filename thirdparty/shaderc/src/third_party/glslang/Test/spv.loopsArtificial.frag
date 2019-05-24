#version 140
in vec4 bigColor;
in vec4 bigColor1_1;
in vec4 bigColor1_2;
in vec4 bigColor1_3;
in vec4 bigColor2;
in vec4 bigColor3;
in vec4 bigColor4;
in vec4 bigColor5;
in vec4 bigColor6;
in vec4 bigColor7;
in vec4 bigColor8;

in vec4 BaseColor;

in float d;
in float d2;
in float d3;
in float d4;
in float d13;

flat in int Count;

void main()
{
    vec4 color = BaseColor;

    // Latchy2
    do {
        color += bigColor4;
        if (color.x < d4) {
            color.z += 2.0;
            if (color.z < d4) {
                color.x++;
                continue;
            }
        }
        if (color.y < d4)
            color.y += d4;
        else
            color.x += d4;
    } while (color.z < d4);

    // Immediate dominator
    while (color.w < d13) {
        if (color.z < d13)
            color++;
        else
            color--;
        // code from Latchy 2
        color += bigColor4;
        if (color.x < d4) {
            color.z += 2.0;
            if (color.z < d4) {
                color.x++;
                continue;
            }
        }
        if (color.y < d4)
            color.y += d4;
        else
            color.x += d4;
    }

    color++;
    gl_FragColor = color;
}
