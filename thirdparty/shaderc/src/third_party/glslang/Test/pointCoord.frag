precision highp float;

uniform sampler2D sampler;

void main()
{
    vec4 color;

    if (length(gl_PointCoord) < 0.3)
        color = texture2D(sampler, gl_PointCoord);
    else
        color = vec4(0.0);

    gl_FragColor = color;
}
