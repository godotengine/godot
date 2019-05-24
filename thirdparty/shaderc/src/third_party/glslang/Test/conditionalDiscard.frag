#version 110

uniform sampler2D tex;
varying vec2 coord;

void main (void)
{
    vec4 v = texture2D(tex, coord);

    if (v == vec4(0.1,0.2,0.3,0.4))
        discard;

    gl_FragColor = v;
}
