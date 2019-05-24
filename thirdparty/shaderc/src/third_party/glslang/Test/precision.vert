#version 300 es

in vec4 pos;

uniform sampler2D s2D;
uniform samplerCube sCube;
uniform isampler2DArray is2DAbad;      // ERROR, no default precision
uniform sampler2DArrayShadow s2dASbad; // ERROR, no default precision

precision highp sampler2D;
precision mediump sampler2DArrayShadow;

uniform sampler2DArrayShadow s2dAS;
uniform isampler2DArray is2DAbad2;     // ERROR, still no default precision

uniform sampler2D s2Dhigh;

void main()
{
    vec4 t = texture(s2D, vec2(0.1, 0.2));
    t += texture(s2Dhigh, vec2(0.1, 0.2));
    t += texture(s2dAS, vec4(0.5));

    gl_Position = pos;
}
