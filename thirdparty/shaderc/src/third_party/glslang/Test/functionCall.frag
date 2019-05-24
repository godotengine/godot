#version 130

uniform vec4 bigColor;
varying vec4 BaseColor;
uniform float d;

float h = 0.0;

float foo(vec4 bar)
{
    return bar.x + bar.y;
}

void bar()
{
}

float unreachableReturn()
{
    if (d < 4.2)
        return 1.2;
    else
        return 4.5;
    // might be another return inserted here by builders, has to be correct type
}

float missingReturn()
{
    if (d < 4.5) {
        h = d;
        return 3.9;
    }
}

void main()
{
    vec4 color = vec4(foo(BaseColor));

    bar();
    float f = unreachableReturn();
    float g = missingReturn();
    
    gl_FragColor = color * f * h;
}
