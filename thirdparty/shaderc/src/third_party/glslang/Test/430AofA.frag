#version 430

float[4][5][6] many[1][2][3];

float gu[][7];
float gimp[][];    // ERROR, implicit inner
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

void main()
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

    vec4 a4[3][2] = {vec4[](vec4(0.0), vec4(1.0)),
                     vec4[2](vec4(0.0), vec4(1.0)),
                     vec4[2](vec4(0.0), vec4(1.0)) };
    vec4 aim[][2] = {vec4[2](vec4(4.0), vec4(2.0)),
                     vec4[](vec4(4.0), vec4(2.0)),
                     vec4[2](vec4(4.0), vec4(2.0)) };
    vec4 aim2[][] = {vec4[2](vec4(4.0), vec4(2.0)),
                     vec4[2](vec4(4.0), vec4(2.0)),
                     vec4[](vec4(4.0), vec4(2.0)) };
    vec4 aim3[3][] = {vec4[2](vec4(4.0), vec4(2.0)),
                      vec4[2](vec4(4.0), vec4(2.0)),
                      vec4[2](vec4(4.0), vec4(2.0)) };

    vec4 bad2[3][] = {vec4[2](vec4(4.0), vec4(2.0)),              // ERROR
                      vec4[3](vec4(4.0), vec4(2.0), vec4(5.0)),
                      vec4[2](vec4(4.0), vec4(2.0)) };

    vec4 bad3[3][] = {vec4[3](vec4(4.0), vec4(2.0), vec4(5.0)),   // ERROR
                      vec4[2](vec4(4.0), vec4(2.0)),
                      vec4[2](vec4(4.0), vec4(2.0)) };

    vec4 bad4[4][] = {vec4[2](vec4(4.0), vec4(2.0)),              // ERROR
                      vec4[2](vec4(4.0), vec4(2.0)),
                      vec4[2](vec4(4.0), vec4(2.0)) };


    g4 = foo(g5);
    g5 = g4;           // ERROR, wrong types
    gu = g4;           // ERROR, not yet sized

    foo(gu);           // ERROR, not yet sized
    bar(g5);

    if (foo(g5) == g4)
        ;
    if (foo(g5) == g5)  // ERROR, different types
        ;

    float u[][7];
    u[2][2] = 3.0;
    float u[5][7];
    u[5][2] = 5.0;      // ERROR
    foo(u);
}

void foo3()
{
    float resize1[][5][7];
    resize1.length();           // ERROR
    resize1[1][4][5] = 2.0;
    resize1.length();           // ERROR
    float resize1[3][5][7];
    resize1.length();           // 3 in AST
    resize1[1].length();        // 5 in AST
    resize1[1][1].length();     // 7 in AST
    resize1[1][1][1].length();  // ERROR

    float resize2[][5][7];
    float resize2[3][4][7];     // ERROR, inner dim change

    float resize3[][5][7];
    float resize3[3][5][9];     // ERROR, inner dim changed

    float resize4[][5][7];
    int  resize4[3][5][7];      // ERROR, element type
}
