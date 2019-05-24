layout(set=2,binding=0) Texture2D tex : register(t16);
SamplerState samp;

float4 main() : SV_Position 
{
    return tex.Sample(samp, float2(0.2, 0.3));
}