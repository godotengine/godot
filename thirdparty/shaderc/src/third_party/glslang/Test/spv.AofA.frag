#version 430

in float infloat;
out float outfloat;

uniform uAofA {
    float f[2][4];
} nameAofA[3][5];

float[4][5][6] many[1][2][3];

float g4[4][7];
in float g5[5][7];

flat in int i, j, k;

float[4][7] foo(float a[5][7])
{
    float r[7];
    r = a[2];

    return float[4][7](a[0], a[1], r, a[3]);
}

void main()
{
    outfloat = 0.0;

    g4 = foo(g5);

//    if (foo(g5) == g4)
//        ++outfloat;

    float u[][7];
    u[2][2] = 3.0;
    float u[5][7];

    foo(u);

    many[i][j][k][i][j][k] = infloat;
    outfloat += many[j][j][j][j][j][j];
    outfloat += nameAofA[1][2].f[0][3];
}
