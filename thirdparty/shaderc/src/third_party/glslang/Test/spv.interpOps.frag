#version 450

in float if1;
in vec2  if2;
in vec3  if3;
in vec4  if4;

flat in int samp;
flat in vec2 offset;

out vec4 fragColor;

void main()
{
    vec4 f4 = vec4(0.0);
    f4.x    += interpolateAtCentroid(if1);
    f4.xy   += interpolateAtCentroid(if2);
    f4.xyz  += interpolateAtCentroid(if3);
    f4      += interpolateAtCentroid(if4);

    f4.x    += interpolateAtSample(if1, samp);
    f4.xy   += interpolateAtSample(if2, samp);
    f4.xyz  += interpolateAtSample(if3, samp);
    f4      += interpolateAtSample(if4, samp);

    f4.x    += interpolateAtOffset(if1, offset);
    f4.xy   += interpolateAtOffset(if2, offset);
    f4.xyz  += interpolateAtOffset(if3, offset);
    f4      += interpolateAtOffset(if4, offset);

    fragColor = f4;
}
