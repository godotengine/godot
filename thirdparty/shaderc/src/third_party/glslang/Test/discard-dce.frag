#version 110
varying vec2 tex_coord;

void main (void)
{
    vec4 white = vec4(1.0);
    vec4 black = vec4(0.2);
    vec4 color = white;

    // First, cut out our circle
    float x = tex_coord.x*2.0 - 1.0;
    float y = tex_coord.y*2.0 - 1.0;

    float radius = sqrt(x*x + y*y);
    if (radius > 1.0) {
        if (radius > 1.1) {
            ++color;
        }

        gl_FragColor = color;

        if (radius > 1.2) {
            ++color;
        }

        discard;
    }

    // If we're near an edge, darken us a tiny bit
    if (radius >= 0.75)
        color -= abs(pow(radius, 16.0)/2.0);

    gl_FragColor = color;

}
