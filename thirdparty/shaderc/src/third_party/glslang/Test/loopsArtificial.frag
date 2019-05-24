#version 130
uniform vec4 bigColor;
uniform vec4 bigColor1_1;
uniform vec4 bigColor1_2;
uniform vec4 bigColor1_3;
uniform vec4 bigColor2;
uniform vec4 bigColor3;
uniform vec4 bigColor4;
uniform vec4 bigColor5;
uniform vec4 bigColor6;
uniform vec4 bigColor7;
uniform vec4 bigColor8;

varying vec4 BaseColor;

uniform float d;
uniform float d2;
uniform float d3;
uniform float d4;
uniform float d5;
uniform float d6;
uniform float d7;
uniform float d8;
uniform float d9;
uniform float d10;
uniform float d11;
uniform float d12;
uniform float d13;
uniform float d14;
uniform float d15;
uniform float d16;
uniform float d17;
uniform float d18;
uniform float d19;
uniform float d20;
uniform float d21;
uniform float d22;
uniform float d23;
uniform float d24;
uniform float d25;
uniform float d26;
uniform float d27;
uniform float d28;
uniform float d29;
uniform float d30;
uniform float d31;
uniform float d32;
uniform float d33;
uniform float d34;

uniform int Count;

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
