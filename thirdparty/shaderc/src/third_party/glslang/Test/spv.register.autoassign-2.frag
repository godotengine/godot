
SamplerState g_tSamp : register(s0);

Texture2D g_tScene[2] : register(t0);

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

void main(out PS_OUTPUT psout)
{
    psout.Color = g_tScene[0].Sample(g_tSamp, float2(0.3,0.4)) +
                  g_tScene[1].Sample(g_tSamp, float2(0.3,0.4));
}
