#version 450

precision lowp sampler2D;
precision lowp int;
precision lowp float;

uniform lowp sampler2D s;
uniform highp sampler3D t;
layout(rgba32f) uniform lowp image2D i1;
layout(rgba32f) uniform highp image2D i2;

layout(location = 0) in lowp vec2 v2;
layout(location = 1) in lowp vec3 v3;
layout(location = 3) flat in lowp ivec2 iv2;

layout(location = 0) out lowp vec4 color;

void main()
{
    color = texture(s, v2);
    color = texture(t, v3);
    lowp vec4 vi1 = imageLoad(i1, iv2);
    lowp vec4 vi2 = imageLoad(i2, iv2);
}
