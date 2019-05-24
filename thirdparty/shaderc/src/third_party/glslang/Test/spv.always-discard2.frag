#version 140
in vec2 tex_coord;

void main (void)
{
    vec4 white = vec4(1.0);
    vec4 black = vec4(0.2);
    vec4 color = white;

    // First, cut out our circle
    float x = tex_coord.x*2.0 - 1.0;
    float y = tex_coord.y*2.0 - 1.0;

    discard;


    gl_FragColor = color;

}
