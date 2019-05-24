#version 400

uniform sampler2D tex;
in vec2 coord;

void main (void)
{
    vec4 v = texture(tex, coord);

    if (v == vec4(0.1,0.2,0.3,0.4))
        discard;

    gl_FragColor = v;
}
