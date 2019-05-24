#version 310 es

// Check name mangling of functions with parameters that are multi-dimensional arrays.

#define NX 2
#define NY 3
#define NZ 4
void f(bool a, float b, uint[4] c, int[NY][NX] d) {
}

void main() {
  int[NY][NX] d;
  f(false, 12.1, uint[NZ](uint(0),uint(1),uint(1),uint(2)), d);
}

buffer b {
    float u[];  // ERROR
    vec4 v[];
} name[3];

uniform ub {
    float u;
    vec4 v[];   // ERROR
} uname[3];

buffer b2 {
    float u;
    vec4 v[][];  // ERROR
} name2[3];

buffer b3 {
    float u; 
    vec4 v[][7];
} name3[3];

// General arrays of arrays

float[4][5][6] many[1][2][3];

float gu[][7];     // ERROR, size required
float g4[4][7];
float g5[5][7];

float[4][7] foo(float a[5][7])
{
    float r[7];
    r = a[2];
    float[](a[0], a[1], r, a[3]);              // ERROR, too few dims
    float[4][7][4](a[0], a[1], r, a[3]);       // ERROR, too many dims
    return float[4][7](a[0], a[1], r, a[3]);
    return float[][](a[0], a[1], r, a[3]);
    return float[][7](a[0], a[1], a[2], a[3]);
}

void bar(float[5][7]) {}

void foo2()
{
    {
        float gu[3][4][2];

        gu[2][4][1] = 4.0;                     // ERROR, overflow
    }
    vec4 ca4[3][2] = vec4[][](vec4[2](vec4(0.0), vec4(1.0)),
                              vec4[2](vec4(0.0), vec4(1.0)),
                              vec4[2](vec4(0.0), vec4(1.0)));
    vec4 caim[][2] = vec4[][](vec4[2](vec4(4.0), vec4(2.0)),
                              vec4[2](vec4(4.0), vec4(2.0)),
                              vec4[2](vec4(4.0), vec4(2.0)));
    vec4 caim2[][] = vec4[][](vec4[2](vec4(4.0), vec4(2.0)),
                              vec4[2](vec4(4.0), vec4(2.0)),
                              vec4[2](vec4(4.0), vec4(2.0)));
    vec4 caim3[3][] = vec4[][](vec4[2](vec4(4.0), vec4(2.0)),
                               vec4[2](vec4(4.0), vec4(2.0)),
                               vec4[2](vec4(4.0), vec4(2.0)));

    g4 = foo(g5);
    g5 = g4;           // ERROR, wrong types
    gu = g4;           // ERROR, not yet sized

    foo(gu);           // ERROR, not yet sized
    bar(g5);

    if (foo(g5) == g4)
        ;
    if (foo(g5) == g5)  // ERROR, different types
        ;

    float u[5][7];
    u[5][2] = 5.0;      // ERROR
    foo(u);

    vec4 badAss[3];
    name[1].v[-1];     // ERROR
    name[1].v[1] = vec4(4.3);
    name[1].v = badAss;  // ERROR, bad assignemnt

    name3[0].v[1].length();  // 7
    name3[0].v.length();     // run time
}

struct badS {
    int sa[];     // ERROR
    int a[][];    // ERROR
    int b[][2];   // ERROR
    int c[2][];   // ERROR
    int d[][4];   // ERROR
};

in float inArray[2][3];    // ERROR
out float outArray[2][3];  // ERROR

uniform ubaa {
    int a;
} ubaaname[2][3];  // ERROR

vec3 func(in mat3[2] x[3])
{
	mat3 a0 = x[2][1];
    return a0[2];
}
