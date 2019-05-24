SamplerState       g_sSamp : register(s0);
uniform sampler2D          g_sSamp2d
{
    AddressU = MIRROR;
    AddressV = WRAP;
    MinLOD = 0;
    MaxLOD = 10;
    MaxAnisotropy = 2;
    MipLodBias = 0.2;
}, g_sSamp2D_b;

Texture1D          g_tTex1df4a : register(t1);

uniform Texture1D <float4> g_tTex1df4 : register(t0);
Texture1D <int4>   g_tTex1di4;
Texture1D <uint4>  g_tTex1du4;

Texture2D <float4> g_tTex2df4;
Texture2D <int4>   g_tTex2di4;
Texture2D <uint4>  g_tTex2du4;

Texture3D <float4> g_tTex3df4;
Texture3D <int4>   g_tTex3di4;
Texture3D <uint4>  g_tTex3du4;

TextureCube <float4> g_tTexcdf4;
TextureCube <int4>   g_tTexcdi4;
TextureCube <uint4>  g_tTexcdu4;

struct MemberTest
{
    int Sample;                          // in HLSL, method names are valid struct members.
    int CalculateLevelOfDetail;          // ...
    int CalculateLevelOfDetailUnclamped; // ...
    int Gather;                          // ...
    int GetDimensions;                   // ...
    int GetSamplePosition;               // ...
    int Load;                            // ...
    int SampleBias;                      // ...
    int SampleCmp;                       // ...
    int SampleCmpLevelZero;              // ...
    int SampleGrad;                      // ...
    int SampleLevel;                     // ...
};

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   MemberTest mtest;
   mtest.CalculateLevelOfDetail = 1;          // in HLSL, method names are valid struct members.
   mtest.CalculateLevelOfDetailUnclamped = 1; // ...
   mtest.Gather = 1;                          // ...
   mtest.GetDimensions = 1;                   // ...
   mtest.GetSamplePosition = 1;               // ...
   mtest.Load = 1;                            // ...
   mtest.Sample = 1;                          // ...
   mtest.SampleBias = 1;                      // ...
   mtest.SampleCmp = 1;                       // ...
   mtest.SampleCmpLevelZero = 1;              // ...
   mtest.SampleGrad = 1;                      // ...
   mtest.SampleLevel = 1;                     // ...

   float4 txval10 = g_tTex1df4 . Sample(g_sSamp, 0.1);
   int4   txval11 = g_tTex1di4 . Sample(g_sSamp, 0.2);
   uint4  txval12 = g_tTex1du4 . Sample(g_sSamp, 0.3);

   float4 txval20 = g_tTex2df4 . Sample(g_sSamp, float2(0.1, 0.2));
   int4   txval21 = g_tTex2di4 . Sample(g_sSamp, float2(0.3, 0.4));
   uint4  txval22 = g_tTex2du4 . Sample(g_sSamp, float2(0.5, 0.6));

   float4 txval30 = g_tTex3df4 . Sample(g_sSamp, float3(0.1, 0.2, 0.3));
   int4   txval31 = g_tTex3di4 . Sample(g_sSamp, float3(0.4, 0.5, 0.6));
   uint4  txval32 = g_tTex3du4 . Sample(g_sSamp, float3(0.7, 0.8, 0.9));

   float4 txval40 = g_tTexcdf4 . Sample(g_sSamp, float3(0.1, 0.2, 0.3));
   int4   txval41 = g_tTexcdi4 . Sample(g_sSamp, float3(0.4, 0.5, 0.6));
   uint4  txval42 = g_tTexcdu4 . Sample(g_sSamp, float3(0.7, 0.8, 0.9));

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}
