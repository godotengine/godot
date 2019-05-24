
sampler            g_sam     : register(t0);
sampler1D          g_sam1D   : register(t1);
sampler2D          g_sam2D   : register(t2);
sampler3D          g_sam3D	 : register(t3);
samplerCube        g_samCube : register(t4);

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   float4 ColorOut = float4(0,0,0,0);
   
   ColorOut += tex2D(   g_sam  ,   float2(0.4,0.3));
   ColorOut += tex1D(   g_sam1D,   0.5);
   ColorOut += tex2D(   g_sam2D,   float2(0.5,0.6));
   ColorOut += tex3D(   g_sam3D,   float3(0.5,0.6,0.4));
   ColorOut += texCUBE( g_samCube, float3(0.5,0.6,0.4));
   
   ColorOut += tex2Dlod(   g_sam  ,   float4(0.4,0.3,0.0,0.0));
   ColorOut += tex1Dlod(   g_sam1D,   float4(0.5,0.0,0.0,0.0));
   ColorOut += tex2Dlod(   g_sam2D,   float4(0.5,0.6,0.0,0.0));
   ColorOut += tex3Dlod(   g_sam3D,   float4(0.5,0.6,0.4,0.0));
   ColorOut += texCUBElod( g_samCube, float4(0.5,0.6,0.4,0.0));
  
   psout.Color = ColorOut / 10.0f;
   psout.Depth = 1.0;

   return psout;
}
