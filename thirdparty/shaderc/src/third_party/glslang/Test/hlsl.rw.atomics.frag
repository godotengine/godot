SamplerState       g_sSamp;

RWTexture1D <float> g_tTex1df1;
RWTexture1D <int>   g_tTex1di1;
RWTexture1D <uint>  g_tTex1du1;

RWTexture2D <float> g_tTex2df1;
RWTexture2D <int>   g_tTex2di1;
RWTexture2D <uint>  g_tTex2du1;

RWTexture3D <float> g_tTex3df1;
RWTexture3D <int>   g_tTex3di1;
RWTexture3D <uint>  g_tTex3du1;

RWTexture1DArray <float> g_tTex1df1a;
RWTexture1DArray <int>   g_tTex1di1a;
RWTexture1DArray <uint>  g_tTex1du1a;

RWTexture2DArray <float> g_tTex2df1a;
RWTexture2DArray <int>   g_tTex2di1a;
RWTexture2DArray <uint>  g_tTex2du1a;

RWBuffer <float> g_tBuffF;
RWBuffer <int>   g_tBuffI;
RWBuffer <uint>  g_tBuffU;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform uint  u1;
uniform uint2 u2;
uniform uint3 u3;
uniform uint  u1b;
uniform uint  u1c;

uniform int   i1;
uniform int2  i2;
uniform int3  i3;
uniform int   i1b;
uniform int   i1c;

PS_OUTPUT main()
{
    uint out_u1;
    int out_i1;

    // 1D int
    InterlockedAdd(g_tTex1di1[i1], i1b);
    InterlockedAdd(g_tTex1di1[i1], i1, out_i1);
    InterlockedAnd(g_tTex1di1[i1], i1b);
    InterlockedAnd(g_tTex1di1[i1], i1, out_i1);
    InterlockedCompareExchange(g_tTex1di1[i1], i1b, i1c, out_i1);
    InterlockedExchange(g_tTex1di1[i1], i1, out_i1);
    InterlockedMax(g_tTex1di1[i1], i1b);
    InterlockedMax(g_tTex1di1[i1], i1, out_i1);
    InterlockedMin(g_tTex1di1[i1], i1b);
    InterlockedMin(g_tTex1di1[i1], i1, out_i1);
    InterlockedOr(g_tTex1di1[i1], i1b);
    InterlockedOr(g_tTex1di1[i1], i1, out_i1);
    InterlockedXor(g_tTex1di1[i1], i1b);
    InterlockedXor(g_tTex1di1[i1], i1, out_i1);

    // 1D uint
    InterlockedAdd(g_tTex1du1[u1], u1);
    InterlockedAdd(g_tTex1du1[u1], u1, out_u1);
    InterlockedAnd(g_tTex1du1[u1], u1);
    InterlockedAnd(g_tTex1du1[u1], u1, out_u1);
    InterlockedCompareExchange(g_tTex1du1[u1], u1b, u1c, out_u1);
    InterlockedExchange(g_tTex1du1[u1], u1, out_u1);
    InterlockedMax(g_tTex1du1[u1], u1);
    InterlockedMax(g_tTex1du1[u1], u1, out_u1);
    InterlockedMin(g_tTex1du1[u1], u1);
    InterlockedMin(g_tTex1du1[u1], u1, out_u1);
    InterlockedOr(g_tTex1du1[u1], u1);
    InterlockedOr(g_tTex1du1[u1], u1, out_u1);
    InterlockedXor(g_tTex1du1[u1], u1);
    InterlockedXor(g_tTex1du1[u1], u1, out_u1);

    // 2D int
    InterlockedAdd(g_tTex2di1[i2], i1b);
    InterlockedAdd(g_tTex2di1[i2], i1, out_i1);
    InterlockedAnd(g_tTex2di1[i2], i1b);
    InterlockedAnd(g_tTex2di1[i2], i1, out_i1);
    InterlockedCompareExchange(g_tTex2di1[i2], i1b, i1c, out_i1);
    InterlockedExchange(g_tTex2di1[i2], i1, out_i1);
    InterlockedMax(g_tTex2di1[i2], i1b);
    InterlockedMax(g_tTex2di1[i2], i1, out_i1);
    InterlockedMin(g_tTex2di1[i2], i1b);
    InterlockedMin(g_tTex2di1[i2], i1, out_i1);
    InterlockedOr(g_tTex2di1[i2], i1b);
    InterlockedOr(g_tTex2di1[i2], i1, out_i1);
    InterlockedXor(g_tTex2di1[i2], i1b);
    InterlockedXor(g_tTex2di1[i2], i1, out_i1);

    // 2D uint
    InterlockedAdd(g_tTex2du1[u2], u1);
    InterlockedAdd(g_tTex2du1[u2], u1, out_u1);
    InterlockedAnd(g_tTex2du1[u2], u1);
    InterlockedAnd(g_tTex2du1[u2], u1, out_u1);
    InterlockedCompareExchange(g_tTex2du1[u2], u1b, u1c, out_u1);
    InterlockedExchange(g_tTex2du1[u2], u1, out_u1);
    InterlockedMax(g_tTex2du1[u2], u1);
    InterlockedMax(g_tTex2du1[u2], u1, out_u1);
    InterlockedMin(g_tTex2du1[u2], u1);
    InterlockedMin(g_tTex2du1[u2], u1, out_u1);
    InterlockedOr(g_tTex2du1[u2], u1);
    InterlockedOr(g_tTex2du1[u2], u1, out_u1);
    InterlockedXor(g_tTex2du1[u2], u1);
    InterlockedXor(g_tTex2du1[u2], u1, out_u1);

    // 3D int
    InterlockedAdd(g_tTex3di1[i3], i1b);
    InterlockedAdd(g_tTex3di1[i3], i1, out_i1);
    InterlockedAnd(g_tTex3di1[i3], i1b);
    InterlockedAnd(g_tTex3di1[i3], i1, out_i1);
    InterlockedCompareExchange(g_tTex3di1[i3], i1b, i1c, out_i1);
    InterlockedExchange(g_tTex3di1[i3], i1, out_i1);
    InterlockedMax(g_tTex3di1[i3], i1b);
    InterlockedMax(g_tTex3di1[i3], i1, out_i1);
    InterlockedMin(g_tTex3di1[i3], i1b);
    InterlockedMin(g_tTex3di1[i3], i1, out_i1);
    InterlockedOr(g_tTex3di1[i3], i1b);
    InterlockedOr(g_tTex3di1[i3], i1, out_i1);
    InterlockedXor(g_tTex3di1[i3], i1b);
    InterlockedXor(g_tTex3di1[i3], i1, out_i1);

    // 3D uint
    InterlockedAdd(g_tTex3du1[u3], u1);
    InterlockedAdd(g_tTex3du1[u3], u1, out_u1);
    InterlockedAnd(g_tTex3du1[u3], u1);
    InterlockedAnd(g_tTex3du1[u3], u1, out_u1);
    InterlockedCompareExchange(g_tTex3du1[u3], u1b, u1c, out_u1);
    InterlockedExchange(g_tTex3du1[u3], u1, out_u1);
    InterlockedMax(g_tTex3du1[u3], u1);
    InterlockedMax(g_tTex3du1[u3], u1, out_u1);
    InterlockedMin(g_tTex3du1[u3], u1);
    InterlockedMin(g_tTex3du1[u3], u1, out_u1);
    InterlockedOr(g_tTex3du1[u3], u1);
    InterlockedOr(g_tTex3du1[u3], u1, out_u1);
    InterlockedXor(g_tTex3du1[u3], u1);
    InterlockedXor(g_tTex3du1[u3], u1, out_u1);

    // 1D array int
    InterlockedAdd(g_tTex1di1a[i2], i1b);
    InterlockedAdd(g_tTex1di1a[i2], i1, out_i1);
    InterlockedAnd(g_tTex1di1a[i2], i1b);
    InterlockedAnd(g_tTex1di1a[i2], i1, out_i1);
    InterlockedCompareExchange(g_tTex1di1a[i2], i1b, i1c, out_i1);
    InterlockedExchange(g_tTex1di1a[i2], i1, out_i1);
    InterlockedMax(g_tTex1di1a[i2], i1b);
    InterlockedMax(g_tTex1di1a[i2], i1, out_i1);
    InterlockedMin(g_tTex1di1a[i2], i1b);
    InterlockedMin(g_tTex1di1a[i2], i1, out_i1);
    InterlockedOr(g_tTex1di1a[i2], i1b);
    InterlockedOr(g_tTex1di1a[i2], i1, out_i1);
    InterlockedXor(g_tTex1di1a[i2], i1b);
    InterlockedXor(g_tTex1di1a[i2], i1, out_i1);

    // 1D array uint
    InterlockedAdd(g_tTex1du1a[u2], u1);
    InterlockedAdd(g_tTex1du1a[u2], u1, out_u1);
    InterlockedAnd(g_tTex1du1a[u2], u1);
    InterlockedAnd(g_tTex1du1a[u2], u1, out_u1);
    InterlockedCompareExchange(g_tTex1du1a[u2], u1b, u1c, out_u1);
    InterlockedExchange(g_tTex1du1a[u2], u1, out_u1);
    InterlockedMax(g_tTex1du1a[u2], u1);
    InterlockedMax(g_tTex1du1a[u2], u1, out_u1);
    InterlockedMin(g_tTex1du1a[u2], u1);
    InterlockedMin(g_tTex1du1a[u2], u1, out_u1);
    InterlockedOr(g_tTex1du1a[u2], u1);
    InterlockedOr(g_tTex1du1a[u2], u1, out_u1);
    InterlockedXor(g_tTex1du1a[u2], u1);
    InterlockedXor(g_tTex1du1a[u2], u1, out_u1);

    // 2D array int
    InterlockedAdd(g_tTex1di1a[i2], i1b);
    InterlockedAdd(g_tTex1di1a[i2], i1, out_i1);
    InterlockedAnd(g_tTex1di1a[i2], i1b);
    InterlockedAnd(g_tTex1di1a[i2], i1, out_i1);
    InterlockedCompareExchange(g_tTex1di1a[i2], i1b, i1c, out_i1);
    InterlockedExchange(g_tTex1di1a[i2], i1, out_i1);
    InterlockedMax(g_tTex1di1a[i2], i1b);
    InterlockedMax(g_tTex1di1a[i2], i1, out_i1);
    InterlockedMin(g_tTex1di1a[i2], i1b);
    InterlockedMin(g_tTex1di1a[i2], i1, out_i1);
    InterlockedOr(g_tTex1di1a[i2], i1b);
    InterlockedOr(g_tTex1di1a[i2], i1, out_i1);
    InterlockedXor(g_tTex1di1a[i2], i1b);
    InterlockedXor(g_tTex1di1a[i2], i1, out_i1);

    // 2D array uint
    InterlockedAdd(g_tTex1du1a[u2], u1);
    InterlockedAdd(g_tTex1du1a[u2], u1, out_u1);
    InterlockedAnd(g_tTex1du1a[u2], u1);
    InterlockedAnd(g_tTex1du1a[u2], u1, out_u1);
    InterlockedCompareExchange(g_tTex1du1a[u2], u1b, u1c, out_u1);
    InterlockedExchange(g_tTex1du1a[u2], u1, out_u1);
    InterlockedMax(g_tTex1du1a[u2], u1);
    InterlockedMax(g_tTex1du1a[u2], u1, out_u1);
    InterlockedMin(g_tTex1du1a[u2], u1);
    InterlockedMin(g_tTex1du1a[u2], u1, out_u1);
    InterlockedOr(g_tTex1du1a[u2], u1);
    InterlockedOr(g_tTex1du1a[u2], u1, out_u1);
    InterlockedXor(g_tTex1du1a[u2], u1);
    InterlockedXor(g_tTex1du1a[u2], u1, out_u1);

    // buffer int
    InterlockedAdd(g_tBuffI[i1], i1b);
    InterlockedAdd(g_tBuffI[i1], i1, out_i1);
    InterlockedAnd(g_tBuffI[i1], i1b);
    InterlockedAnd(g_tBuffI[i1], i1, out_i1);
    InterlockedCompareExchange(g_tBuffI[i1], i1b, i1c, out_i1);
    InterlockedExchange(g_tBuffI[i1], i1, out_i1);
    InterlockedMax(g_tBuffI[i1], i1b);
    InterlockedMax(g_tBuffI[i1], i1, out_i1);
    InterlockedMin(g_tBuffI[i1], i1b);
    InterlockedMin(g_tBuffI[i1], i1, out_i1);
    InterlockedOr(g_tBuffI[i1], i1b);
    InterlockedOr(g_tBuffI[i1], i1, out_i1);
    InterlockedXor(g_tBuffI[i1], i1b);
    InterlockedXor(g_tBuffI[i1], i1, out_i1);

    // buffer uint
    InterlockedAdd(g_tBuffU[u1], u1);
    InterlockedAdd(g_tBuffU[u1], u1, out_u1);
    InterlockedAnd(g_tBuffU[u1], u1);
    InterlockedAnd(g_tBuffU[u1], u1, out_u1);
    InterlockedCompareExchange(g_tBuffU[u1], u1b, u1c, out_u1);
    InterlockedExchange(g_tBuffU[u1], u1, out_u1);
    InterlockedMax(g_tBuffU[u1], u1);
    InterlockedMax(g_tBuffU[u1], u1, out_u1);
    InterlockedMin(g_tBuffU[u1], u1);
    InterlockedMin(g_tBuffU[u1], u1, out_u1);
    InterlockedOr(g_tBuffU[u1], u1);
    InterlockedOr(g_tBuffU[u1], u1, out_u1);
    InterlockedXor(g_tBuffU[u1], u1);
    InterlockedXor(g_tBuffU[u1], u1, out_u1);

    PS_OUTPUT psout;
    psout.Color = 1.0;
    return psout;
}
