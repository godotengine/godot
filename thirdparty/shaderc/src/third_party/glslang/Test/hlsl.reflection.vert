
cbuffer nameless {
    float3 anonMember1;
    float3x2 m23;
    int scalarAfterm23;
    float4 anonDeadMember2;
    float4 anonMember3;
    int scalarBeforeArray;
    float floatArray[5];
    int scalarAfterArray;
    float2x2 m22[9];
};

cbuffer c_nameless {
    float3 c_anonMember1;
    float3x2 c_m23;
    int c_scalarAfterm23;
    float4 c_anonDeadMember2;
    float4 c_anonMember3;
};

cbuffer namelessdead {
    int a;
};

struct N1 {
    float a;
};

struct N2 {
    float b;
    float c;
    float d;
};

struct N3 {
    N1 n1;
    N2 n2;
};

cbuffer nested {
    N3 foo;
}

struct TS {
    int a;
    int dead;
};

uniform TS s;

uniform float uf1;
uniform float uf2;
uniform float ufDead3;
uniform float ufDead4;

uniform float2x2 dm22[10];

struct deep1 {
    float2 va[3];
    bool b;
};

struct deep2 {
    int i;
    deep1 d1[4];
};

struct deep3 {
    float4 iv4;
    deep2 d2;
    int3 v3;
};

uniform deep3 deepA[2], deepB[2], deepC[3], deepD[2];

const bool control = true;

void deadFunction()
{
    float4 v = anonDeadMember2;
    float f = ufDead4;
}

void liveFunction2()
{
    float3 v = anonMember1;
    float f = uf1;
}

tbuffer abl {
    float foo1;
}

tbuffer abl2 {
    float foo2;
}

void flizv(in float attributeFloat, in float2 attributeFloat2, in float3 attributeFloat3, in float4 attributeFloat4, in float4x4 attributeMat4)
{
    liveFunction2();

    if (! control)
        deadFunction();

    float f;
    int i;
    if (control) {
        liveFunction2();
        f = anonMember3.z;
        f = s.a;
        f = m23[1].y + scalarAfterm23;
        f = c_m23[1].y + c_scalarAfterm23;
        f += scalarBeforeArray;
        f += floatArray[2];
        f += floatArray[4];
        f += scalarAfterArray;
        f += m22[i][1][0];
        f += dm22[3][0][1];
        f += m22[2][1].y;
        f += foo.n1.a + foo.n2.b + foo.n2.c + foo.n2.d;
        f += deepA[i].d2.d1[2].va[1].x;
        f += deepB[1].d2.d1[i].va[1].x;
        f += deepB[i].d2.d1[i].va[1].x;
        deep3 d = deepC[1];
        deep3 da[2] = deepD;
    } else
        f = ufDead3;

    f += foo1 + foo2;
    f += foo2;

    f += attributeFloat;
    f += attributeFloat2.x;
    f += attributeFloat3.x;
    f += attributeFloat4.x;
    f += attributeMat4[0][1];
}
