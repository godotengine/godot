#version 400

void main()
{
    int o00 = 00;
    int o000 = 000;
    int o0000 = 0000;
    int o5 = 05;
    int o05 = 005;
    int o006 = 0006;
    int o7 = 07;
    int o58 = 072;
    int omax = 037777777777;
    int o8 = 08;                      // ERROR
    int o08 = 008;                    // ERROR
    int o009 = 0009;                  // ERROR
    int obig = 07324327323472347234;  // ERROR
    int omax1 = 040000000000;         // ERROR

    uint uo5 = 05u;
    uint uo6 = 06u;
    uint uo7 = 07u;
    uint uo8 = 08u;                   // ERROR
    uint uo9 = 09u;                   // ERROR

    int h0 = 0x0;
    int h00 = 0x00;
    int h000 = 0x000;
    int h1 = 0x1;
    int h2 = 0x00000002;
    int h300 = 0x000300;
    int hABCDEF = 0xAbCdEF;
    int hFFFFFFFF = 0xFFFFFFFF;
    int h12345678 = 0xBC614E;
    int hToBeOrNotToBe = 0x2b | ~0x2B;

    uint uh0 = 0x0u;
    uint uhg = (0xcu);
    uint uh000 = 0x000u;
    uint uh1 = 0x1u;
    uint uh2 = 0x00000002u;
    uint uh300 = 0x000300u;
    uint uhABCDEF = 0xAbCdEFu;
    uint uhFFFFFFFF = 0xFFFFFFFFu;
    uint uh12345678 = 0xBC614Eu;
    uint uhToBeOrNotToBe = 0x2bu | ~0x2BU;

    //int he1 = 0xG;                     // ERROR
    int he2 = 0x;                      // ERROR
    int hbig = 0xFFFFFFFF1;            // ERROR

    float f1 = 1.0;
    float f2 = 2.;
    float f3 = 3e0;
    float f4 = 40e-1;
    float f5 = 05.;
    float f6 = 006.;
    float f7 = .7e1;
    float f8 = 08e0;
    float f9 = .9e+1;
    float f10 = 10.0;
    float f11 = .011e+3;
    float f12 = .0012e4;
    float f543 = 000000543.;
    float f6789 = 00006789.;
    float f88 = 0000088.;

    float g1 = 5.3876e4;
    float g2 = 4000000000e-11;
    float g3 = 1e+5;
    float g4 = 7.321E-3;
    float g5 = 3.2E+4;
    float g6 = 0.5e-5;
    float g7 = 0.45;
    float g8 = 6.e10;

    double gf1 = 1.0lf;
    double gf2 = 2.Lf;
    double gf3 = .3e1lF;
    double gf4 = .4e1LF;
    float gf5 = 5.f;
    float gf6 = 6.F;

    //float e1 = 1..;        // ERROR
    //float e2 = 2.l;        // ERROR
    //float e3 = ..3;        // ERROR
    //float e4 = 4ee1;       // ERROR
    float e5 = 5f;         // ERROR
}

layout (location = 2) out vec4 c2;
layout (location = 3u) out vec4 c3;
layout (location = 04) out vec4 c4;
layout (location = 005u) out vec4 c5;
layout (location = 0x6) out vec4 c6;
layout (location = 0x7u) out vec4 c7;

uint g1 = 4294967296u; // ERROR, too big
uint g2 = 4294967295u;
uint g3 = 4294967294u;
int g4 = 4294967296;   // ERROR, too big
int g5 = 4294967295;
int g6 = 4294967294;
float inf1 = -1.#INF;
float inf2 =  1.#INF;
float inf3 = +1.#INF;
