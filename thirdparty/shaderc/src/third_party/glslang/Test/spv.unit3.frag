#version 460

float f;
float h3 = 3.0;

out float cout;
in float cin;

float bar()
{
    h3 *= f;
    float g3 = 2 * h3;
    cout = g3;
    return h3 + g3 + gl_FragCoord.y;
}
