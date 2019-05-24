#version 100

int ga, gb;
float f;

uniform sampler2D fsa[3];
uniform float fua[10];
attribute mat3 am3;
attribute vec2 av2;
varying vec4 va[4];

const mat2 m2 = mat2(1.0);
const vec3 v3 = vec3(2.0);

void foo(inout float a) {}

int bar()
{
    return 1;
}

void main()
{
    while (ga < gb) { }

    do { } while (false);

    for (           ;              ;         );           // ERROR
    for (           ;        ga==gb;         );           // ERROR
    for (           ;              ;      f++);           // ERROR
    for (     ga = 0;              ;         );           // ERROR
    for ( bool a = false;          ;         );           // ERROR
    for (float a = 0.0; a == sin(f);         );           // ERROR
    for (  int a = 0;       a  < 10;   a *= 2);           // ERROR
    for (  int a = 0;       a <= 20;      a++)  --a;      // ERROR
    for (  int a = 0;       a <= 20;      a++)  { if (ga==0) a = 4; } // ERROR
    for (float a = 0.0;   a <= 20.0; a += 2.0);
    for (float a = 0.0;   a != 20.0; a -= 2.0)  { if (ga==0) ga = 4; }
    for (float a = 0.0;   a == 20.0;      a--) for (float a = 0.0;   a == 20.0;      a--);  // two different 'a's, everything okay
    for (float a = 0.0;   a <= 20.0; a += 2.0);
    for (float a = 0.0;   a <= 20.0; a += 2.0);
    for (float a = 0.0;   a > 2.0 * 20.0; a += v3.y);
    for (float a = 0.0;   a >= 20.0; a += 2.0) foo(a);    // ERROR

    int ia[9];

    fsa[ga];  // ERROR
    fua[ga];
    am3[ga];  // ERROR
    av2[ga];  // ERROR
    va[2+ga]; // ERROR
    m2[ga];   // ERROR
    v3[ga/2]; // ERROR
    ia[ga];   // ERROR

    for (int a = 3; a >= 0; a--) {
        fsa[a];
        fua[a+2];
        am3[3*a];
        av2[3*a];
        va[a-1];
        m2[a/2];
        v3[a];
        ia[a];
        ia[bar()];  // ERROR
    }

    fsa[2];
    fua[3];
    am3[2];
    av2[1];
    va[1];
    m2[1];
    v3[1];
    ia[3];
}
