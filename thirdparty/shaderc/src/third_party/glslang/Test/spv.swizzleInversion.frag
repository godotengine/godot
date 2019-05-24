#version 450

in vec4 in4;
in vec3 in3;

void main()
{
    vec3 v43 = interpolateAtCentroid(in4.wzx);
    vec2 v42 = interpolateAtSample(in4.zx, 1);
    vec4 v44 = interpolateAtOffset(in4.zyxw, vec2(2.0));
    float v41 = interpolateAtOffset(in4.y, vec2(2.0));

    vec3 v33 = interpolateAtCentroid(in3.yzx);
    vec2 v32 = interpolateAtSample(in3.zx, 1);
    float v31 = interpolateAtOffset(in4.y, vec2(2.0));
}
