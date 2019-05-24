#version 450

struct S {
    int[3] a[2], b[5];
};

S s;

int[5] c[4], d[8];
int[9] e[], f[];
int e[11][9];
int f[13][9];

int[14] g[], h[];

int [14][15][6] foo(int[6] p[14][15]) { return p; }

void main()
{
    g[3];
    h[2];
}

float[4][3][2] bar() { float[3][2] a[4]; return a; }

in inbname {
    float[7] f[8][9];
} inbinst[4][5][6];

float[3][2] barm[4]() { float[3][2] a[4]; return a; }  // ERROR
