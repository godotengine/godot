#version 100

float gu[];              // ERROR
float g4[4];
float g5[5];

uniform int a;

float[4] foo(float[5] a)  // ERROR  // ERROR
{
    return float[](a[0], a[1], a[2], a[3]);  // ERROR
}

void bar(float[5]) {}

void main()
{
    {
        float gu[2];  // okay, new scope

        gu[2] = 4.0;  // ERROR, overflow
    }

    g4 = foo(g5);  // ERROR
    g5 = g4;  // ERROR
    gu = g4;  // ERROR

    foo(gu);  // ERROR
    bar(g5);

    if (float[4](1.0, 2.0, 3.0, 4.0) == g4)  // ERROR
        gu[0] = 2.0;

    float u[5];
    u[5] = 5.0; // ERROR
    foo(u);     // okay

    gl_FragData[1000] = vec4(1.0); // ERROR
    gl_FragData[-1] = vec4(1.0);   // ERROR
    gl_FragData[3] = vec4(1.0);
}

struct SA {
    vec3 v3;
    vec2 v2[4];
};

struct SB {
    vec4 v4;
    SA sa;
};

SB bar9()
{
    SB s;
    return s;  // ERROR
}

void bar10(SB s)  // okay
{
}

void bar11()
{
    SB s1, s2;
    s1 = s2;   // ERROR
    bar10(s1);
    s2 = bar9(); // ERROR
    SB initSb = s1;  // ERROR
}
