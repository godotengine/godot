#version 410
// a different version number makes different id's for the same shared symbol

float a2;
float f;

float bar();

out float cout;
in float cin;

float foo()
{
    float h2 = 2 * f + cin;
    float g2 = bar();
    return h2 + g2 + gl_FragCoord.y;
}