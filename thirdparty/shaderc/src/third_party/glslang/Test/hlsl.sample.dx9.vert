
sampler            g_sam   : register(t0);
sampler2D          g_sam2D : register(t1);

struct VS_OUTPUT
{
    float4 Pos : SV_Position;
};

VS_OUTPUT main()
{
   VS_OUTPUT vsout;

   float4 PosOut = float4(0,0,0,0);
   
   PosOut += tex2Dlod(   g_sam  ,  float4(0.3f, 0.4f, 0.0f, 1.0f));
   PosOut += tex2Dlod(   g_sam2D,  float4(0.5f, 0.6f, 0.0f, 1.0f));
   
   vsout.Pos = PosOut / 2.0f;

   return vsout;
}
