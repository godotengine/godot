struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

inline void MyFunc(in float x, out float y, inout float z, in out float w)
{
    y = x;
    z = y;
    x = -1; // no effect since x = in param
    w *= 1;
}

PS_OUTPUT main(noperspective in float4 inpos : SV_Position, out int sampleMask : SV_Coverage)
{
   PS_OUTPUT psout;

   float x = 7, y, z = 3;
   MyFunc(x, y, z, inpos.w);

   psout.Color = float4(x, y, z, 1);
   psout.Depth = inpos.w;

   return psout;
}
