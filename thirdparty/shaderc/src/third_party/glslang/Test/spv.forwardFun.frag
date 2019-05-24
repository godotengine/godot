#version 140

precision mediump float;

in vec4 bigColor;
in vec4 BaseColor;
in float d;

void bar();
float foo(vec4);
float unreachableReturn();

void main()
{
    vec4 color = vec4(foo(BaseColor));

    bar();
    float f = unreachableReturn();
    
    gl_FragColor = color * f;
}

void bar()
{
}

float unreachableReturn()
{
    bar();
    if (d < 4.2)
        return 1.2;
    else
        return 4.5;
}

float foo(vec4 bar)
{
    return bar.x + bar.y;
}
